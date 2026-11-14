"""TETRA lower-MAC channel coding (ETSI EN 300 392-2, clause 8).

Coding chain for a logical channel (transmit direction)::

    type-1 (info) --CRC-16 + 4 tail--> type-2 --RCPC--> type-3
           --block interleave--> type-4 --scramble--> type-5 (on air)

Soft bits use the convention ``value > 0  <=>  bit 0`` throughout, so a soft
value can be read as a BPSK amplitude (+1 for 0, -1 for 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# --------------------------------------------------------------------- CRC-16
# 8.2.3.3: G(x) = x^16 + x^12 + x^5 + 1, register preset to all ones, the 16
# parity bits are the complement of the remainder. Running the same register
# over info + parity bits leaves this residue when there are no errors.
CRC16_POLY = 0x1021
CRC16_GOOD_RESIDUE = 0x1D0F


def crc16(bits) -> int:
    """Raw CRC register (preset 0xFFFF, no final XOR) after shifting in ``bits``."""
    crc = 0xFFFF
    for b in np.asarray(bits, dtype=np.uint8):
        fb = ((crc >> 15) ^ int(b)) & 1
        crc = (crc << 1) & 0xFFFF
        if fb:
            crc ^= CRC16_POLY
    return crc


def crc16_parity(bits) -> np.ndarray:
    """The 16 parity bits that are appended to ``bits`` (MSB first)."""
    c = crc16(bits) ^ 0xFFFF
    return np.array([(c >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)


@lru_cache(maxsize=None)
def _crc_matrix(n: int) -> tuple[int, np.ndarray]:
    """CRC is affine over GF(2): crc(b) = const ^ XOR_i b_i * M[i]. Cache M per length."""
    const = crc16(np.zeros(n, np.uint8))
    m = np.zeros((n, 16), np.int32)
    e = np.zeros(n, np.uint8)
    for i in range(n):
        e[i] = 1
        v = crc16(e) ^ const
        e[i] = 0
        m[i] = [(v >> (15 - j)) & 1 for j in range(16)]
    return const, m


_POW16 = (1 << np.arange(15, -1, -1)).astype(np.int64)


def crc16_check(bits: np.ndarray) -> np.ndarray:
    """Vectorised check of ``bits[..., :n]`` (info + 16 parity bits). Returns a bool array."""
    bits = np.atleast_2d(bits)
    const, m = _crc_matrix(bits.shape[-1])
    v = (bits.astype(np.int32) @ m) & 1
    reg = (v.astype(np.int64) @ _POW16) ^ const
    return reg == CRC16_GOOD_RESIDUE


# ------------------------------------------------------- RCPC convolutional
# 8.2.3.1: rate 1/4 mother code, constraint length 5.
#   G1 = 1 + D + D^4, G2 = 1 + D^2 + D^3 + D^4, G3 = 1 + D + D^2 + D^4, G4 = 1 + D + D^3 + D^4
# Encoder state s: bit0 = D^1 (previous input) ... bit3 = D^4.


def _branch_bits(s: int, u: int) -> tuple[int, int, int, int]:
    d1, d2, d3, d4 = s & 1, (s >> 1) & 1, (s >> 2) & 1, (s >> 3) & 1
    return (u ^ d1 ^ d4, u ^ d2 ^ d3 ^ d4, u ^ d1 ^ d2 ^ d4, u ^ d1 ^ d3 ^ d4)


def conv_encode(bits) -> np.ndarray:
    """Rate-1/4 mother code. Output length is 4 * len(bits); the encoder starts at state 0."""
    s = 0
    out = np.empty(4 * len(bits), dtype=np.uint8)
    for k, u in enumerate(np.asarray(bits, dtype=np.uint8)):
        u = int(u)
        out[4 * k : 4 * k + 4] = _branch_bits(s, u)
        s = ((s << 1) | u) & 15
    return out


# Puncturing (8.2.3.1.3): type-3 bit j (1-based) is mother-code bit
#   k = 8 * ((j-1) div t) + P[j - t*((j-1) div t)]
_PUNCTURERS = {
    "2/3": ((1, 2, 5), 3),
    "1/3": ((1, 2, 3, 5, 6, 7), 6),
}


@lru_cache(maxsize=None)
def puncture_index(n3: int, rate: str = "2/3") -> np.ndarray:
    """0-based mother-code indices of the ``n3`` type-3 bits."""
    p, t = _PUNCTURERS[rate]
    j = np.arange(1, n3 + 1)
    q = (j - 1) // t
    return 8 * q + np.asarray(p)[j - t * q - 1] - 1


def _trellis() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ns = np.arange(16)
    u = ns & 1
    p0 = ns >> 1
    p1 = p0 | 8
    sign = np.zeros((16, 2, 4), np.float32)
    for s in range(16):
        for b in (0, 1):
            sign[s, b] = [1 - 2 * x for x in _branch_bits(s, b)]
    return p0, p1, sign[p0, u], sign[p1, u]


_P0, _P1, _SIGN0, _SIGN1 = _trellis()


def viterbi(mother_soft: np.ndarray) -> np.ndarray:
    """Batch soft-decision Viterbi decoder for the rate-1/4 mother code.

    ``mother_soft``: shape (B, 4*N), positive = bit 0, 0 = erased (punctured).
    The trellis is assumed to start and end in state 0 (4 zero tail bits).
    Returns hard decoded bits, shape (B, N), including the tail.
    """
    mother_soft = np.atleast_2d(mother_soft).astype(np.float32, copy=False)
    b, m = mother_soft.shape
    n = m // 4
    soft = mother_soft.reshape(b, n, 4)
    metric = np.full((b, 16), -1e6, np.float32)
    metric[:, 0] = 0.0
    dec = np.empty((n, b, 16), dtype=bool)
    s0t, s1t = _SIGN0.T, _SIGN1.T
    for t in range(n):
        st = soft[:, t, :]
        c0 = metric[:, _P0] + st @ s0t
        c1 = metric[:, _P1] + st @ s1t
        d = c1 > c0
        metric = np.where(d, c1, c0)
        dec[t] = d
    state = np.zeros(b, dtype=np.int64)
    rows = np.arange(b)
    out = np.empty((b, n), dtype=np.uint8)
    for t in range(n - 1, -1, -1):
        out[:, t] = state & 1
        state = (state >> 1) | (dec[t, rows, state].astype(np.int64) << 3)
    return out


# ------------------------------------------------------- block interleaving
# 8.2.4.1: type-4 bit k = 1 + (a * i) mod K receives type-3 bit i (1-based).


@lru_cache(maxsize=None)
def _interleave_index(k: int, a: int) -> np.ndarray:
    return (a * np.arange(1, k + 1)) % k


def interleave(bits: np.ndarray, k: int, a: int) -> np.ndarray:
    out = np.empty_like(bits)
    out[..., _interleave_index(k, a)] = bits
    return out


def deinterleave(bits: np.ndarray, k: int, a: int) -> np.ndarray:
    return bits[..., _interleave_index(k, a)]


# ---------------------------------------------------------------- scrambling
# 8.2.5: 32-bit LFSR, c(x) = 1 + x + x^2 + x^4 + x^5 + x^7 + x^8 + x^10 + x^11 + x^12
# + x^16 + x^22 + x^23 + x^26 + x^32, preset with the 30-bit extended colour code
# e(1..30) = MCC(10) | MNC(14) | CC(6) followed by two ones. Every block (BSCH, AACH,
# each half/full slot block) restarts the sequence from the preset.
_LFSR_TAPS = (32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 4, 2, 1)
BSCH_SCRAMBLING_INIT = 0x00000003  # extended colour code = 0


def scrambling_init(mcc: int, mnc: int, cc: int) -> int:
    ecc = (cc & 0x3F) | ((mnc & 0x3FFF) << 6) | ((mcc & 0x3FF) << 20)
    return ((ecc << 2) | 3) & 0xFFFFFFFF


@lru_cache(maxsize=64)
def scrambling_sequence(init: int, n: int = 432) -> np.ndarray:
    r = init
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        bit = 0
        for t in _LFSR_TAPS:
            bit ^= (r >> (32 - t)) & 1
        r = (r >> 1) | (bit << 31)
        out[i] = bit
    out.setflags(write=False)
    return out


def scramble(bits: np.ndarray, init: int) -> np.ndarray:
    """XOR hard bits with the scrambling sequence (also descrambles)."""
    n = bits.shape[-1]
    return bits ^ scrambling_sequence(init, max(432, n))[:n]


def descramble_soft(soft: np.ndarray, init: int) -> np.ndarray:
    n = soft.shape[-1]
    seq = scrambling_sequence(init, max(432, n))[:n]
    return soft * (1.0 - 2.0 * seq).astype(np.float32)


# ------------------------------------------------------------- RM(30,14)
# 8.2.3.2: systematic Reed-Muller (30,14) code for the AACH (broadcast block).
# Codeword = 14 info bits followed by 16 parity bits; parity j = XOR_i u_i * G[i][j].
RM3014_PARITY = np.array(
    [
        [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    ],
    dtype=np.uint8,
)


def rm3014_encode(info14) -> np.ndarray:
    u = np.asarray(info14, dtype=np.uint8)
    return np.concatenate([u, (u.astype(np.int32) @ RM3014_PARITY) & 1]).astype(np.uint8)


@lru_cache(maxsize=1)
def _rm3014_codebook() -> tuple[np.ndarray, np.ndarray]:
    v = np.arange(1 << 14)
    info = ((v[:, None] >> np.arange(13, -1, -1)) & 1).astype(np.uint8)
    code = np.concatenate([info, (info.astype(np.int32) @ RM3014_PARITY) & 1], axis=1)
    return info, (1.0 - 2.0 * code).astype(np.float32)


def rm3014_decode(soft30: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood soft decoding (full codebook search).

    Returns (info bits (B,14), number of hard-bit disagreements with the chosen codeword (B,)).
    """
    soft30 = np.atleast_2d(soft30).astype(np.float32)
    info, cb = _rm3014_codebook()
    if soft30.shape[0] == 0:
        return np.zeros((0, 14), np.uint8), np.zeros(0, int)
    best = np.argmax(soft30 @ cb.T, axis=1)
    hard = soft30 < 0
    code = cb[best] < 0
    return info[best], np.count_nonzero(hard != code, axis=1)


# ------------------------------------------------------- logical channels


@dataclass(frozen=True)
class ChannelCoding:
    name: str
    k1: int  # type-1 (info) bits
    k3: int  # type-3/4/5 bits on air
    a: int  # interleaver parameter

    @property
    def k2(self) -> int:  # info + CRC-16 + 4 tail bits
        return self.k1 + 16 + 4


BSCH = ChannelCoding("BSCH", 60, 120, 11)
SCH_HD = ChannelCoding("SCH/HD", 124, 216, 101)  # also BNCH and STCH
SCH_F = ChannelCoding("SCH/F", 268, 432, 103)
SCH_HU = ChannelCoding("SCH/HU", 92, 168, 13)


def encode_block(info, ch: ChannelCoding, scramb_init: int) -> np.ndarray:
    info = np.asarray(info, dtype=np.uint8)
    assert info.size == ch.k1, (ch.name, info.size)
    type2 = np.concatenate([info, crc16_parity(info), np.zeros(4, np.uint8)])
    type3 = conv_encode(type2)[puncture_index(ch.k3)]
    return scramble(interleave(type3, ch.k3, ch.a), scramb_init)


def decode_blocks(soft5: np.ndarray, ch: ChannelCoding, scramb_init: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode a batch of type-5 soft blocks, shape (B, k3).

    Returns (type-1 bits (B, k1), CRC ok (B,)).
    """
    soft5 = np.atleast_2d(soft5)
    if soft5.shape[0] == 0:
        return np.zeros((0, ch.k1), np.uint8), np.zeros(0, bool)
    soft3 = deinterleave(descramble_soft(soft5, scramb_init), ch.k3, ch.a)
    mother = np.zeros((soft5.shape[0], 4 * ch.k2), np.float32)
    mother[:, puncture_index(ch.k3)] = soft3
    type2 = viterbi(mother)
    ok = crc16_check(type2[:, : ch.k1 + 16])
    return type2[:, : ch.k1], ok
