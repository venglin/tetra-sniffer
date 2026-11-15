"""Synthetic TETRA downlink generator (test signals, self-test, bundled example).

It builds a plausible main control carrier: the BSCH and BNCH in frame 18,
an MCCH on timeslot 1 carrying SYSINFO and some call-control / mobility
signalling, a traffic channel on timeslot 2, SCH/F signalling (including a
fragmented SDS) on timeslot 3 and an idle timeslot 4. Every block is coded
exactly as on air (CRC, RCPC, interleaving, scrambling, RM(30,14) for AACH).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from . import burst as B
from . import coding as C
from .dsp import CHANNEL_RATE, SPS, modulate


class BitWriter:
    def __init__(self):
        self.b: list[int] = []

    def u(self, v: int, n: int) -> "BitWriter":
        self.b.extend((int(v) >> (n - 1 - i)) & 1 for i in range(n))
        return self

    def bits(self, arr) -> "BitWriter":
        self.b.extend(int(x) for x in arr)
        return self

    def __len__(self) -> int:
        return len(self.b)

    def array(self) -> np.ndarray:
        return np.array(self.b, dtype=np.uint8)


@dataclass
class CellConfig:
    mcc: int = 901
    mnc: int = 9999
    cc: int = 17
    la: int = 1001
    band: int = 4
    main_carrier: int = 3612  # 400 MHz + 3612 * 25 kHz = 490.3 MHz (fictitious)
    offset: int = 0
    duplex: int = 0
    group: int = 1234567  # GSSI used in the synthetic calls
    ms: int = 7654321  # ISSI used in the synthetic signalling
    secondary: bool = False  # secondary carrier: no synchronisation bursts, no BNCH


# --------------------------------------------------------------- PDUs
def pdu_sync(cell: CellConfig, tn: int, fn: int, mn: int) -> np.ndarray:
    w = BitWriter()
    w.u(4, 4).u(cell.cc, 6).u(tn - 1, 2).u(fn, 5).u(mn, 6).u(0, 2).u(0, 3).u(0, 1).u(0, 1).u(0, 1)
    w.u(cell.mcc, 10).u(cell.mnc, 14).u(1, 2).u(1, 2).u(1, 1)
    return w.array()


def pdu_sysinfo(cell: CellConfig, hyperframe: int = 0) -> np.ndarray:
    w = BitWriter()
    w.u(2, 2).u(0, 2)
    w.u(cell.main_carrier, 12).u(cell.band, 4).u(cell.offset, 2).u(cell.duplex, 3).u(0, 1)
    w.u(0, 2).u(5, 3).u(3, 4).u(8, 4).u(0, 4).u(0, 1).u(hyperframe, 16).u(1, 2).u(0, 20)
    w.u(cell.la, 14).u(0xFFFF, 16).u(0b101001100110, 12)
    return w.array()


def pdu_aach(hdr: int, f1: int, f2: int) -> np.ndarray:
    return BitWriter().u(hdr, 2).u(f1, 6).u(f2, 6).array()


def _chan_alloc(carrier: int, ts_mask: int) -> BitWriter:
    return BitWriter().u(0, 2).u(ts_mask, 4).u(3, 2).u(0, 1).u(0, 1).u(carrier, 12).u(0, 1).u(1, 2)


def mac_resource(ssi: int | None, sdu=None, block_bits: int = 124, enc: int = 0,
                 chan_alloc: BitWriter | None = None, frag: bool = False) -> np.ndarray:
    """MAC-RESOURCE PDU in a block of ``block_bits``; ``ssi=None`` builds a Null PDU."""
    if ssi is None:
        w = BitWriter().u(0, 2).u(0, 1).u(0, 1).u(0, 2).u(0, 1).u(2, 6).u(0, 3)
        return np.concatenate([w.array(), np.zeros(block_bits - len(w), np.uint8)])
    sdu = np.zeros(0, np.uint8) if sdu is None else np.asarray(sdu, np.uint8)
    hdr_len = 43 + (len(chan_alloc) if chan_alloc else 0)
    if frag:
        sdu = sdu[: block_bits - hdr_len]
        li, fill = 0x3F, 0
    else:
        total = hdr_len + len(sdu)
        li = -(-total // 8)
        fill = int(total % 8 != 0)
    w = BitWriter().u(0, 2).u(fill, 1).u(0, 1).u(enc, 2).u(0, 1).u(li, 6).u(1, 3).u(ssi, 24)
    w.u(0, 1).u(0, 1).u(1 if chan_alloc else 0, 1)
    if chan_alloc:
        w.bits(chan_alloc.b)
    w.bits(sdu)
    if not frag and fill:
        w.u(1, 1)
        while len(w) % 8:
            w.u(0, 1)
    out = w.array()
    assert len(out) <= block_bits, (len(out), block_bits)
    return np.concatenate([out, np.zeros(block_bits - len(out), np.uint8)])


def mac_frag(sdu, block_bits: int) -> np.ndarray:
    w = BitWriter().u(1, 2).u(0, 1)
    room = block_bits - 4
    sdu = np.asarray(sdu, np.uint8)[:room]
    fill = int(len(sdu) < room)
    w.u(fill, 1).bits(sdu)
    if fill:
        w.u(1, 1)
    out = w.array()
    return np.concatenate([out, np.zeros(block_bits - len(out), np.uint8)])


def mac_end(sdu, block_bits: int, chan_alloc: BitWriter | None = None) -> np.ndarray:
    ca = chan_alloc.b if chan_alloc else []
    total = 13 + len(ca) + len(sdu)
    li = -(-total // 8)
    fill = int(total % 8 != 0)
    w = BitWriter().u(1, 2).u(1, 1).u(fill, 1).u(0, 1).u(li, 6).u(0, 1).u(1 if ca else 0, 1)
    w.bits(ca).bits(sdu)
    if fill:
        w.u(1, 1)
        while len(w) % 8:
            w.u(0, 1)
    out = w.array()
    return np.concatenate([out, np.zeros(block_bits - len(out), np.uint8)])


def llc(sdu: BitWriter, kind: str = "BL-DATA") -> np.ndarray:
    w = BitWriter()
    if kind == "BL-ADATA":
        w.u(0, 4).u(0, 1).u(1, 1)
    elif kind == "BL-DATA":
        w.u(1, 4).u(0, 1)
    else:
        w.u(2, 4)
    return w.bits(sdu.b).array()


def cmce_d_setup(call_id: int, prio: int = 3) -> BitWriter:
    return (BitWriter().u(2, 3).u(7, 5).u(call_id, 14).u(1, 4).u(0, 1).u(0, 1)
            .u(0, 3).u(0, 1).u(1, 2).u(0, 2).u(0, 2).u(1, 1).u(prio, 4).u(0, 1))


def cmce_d_tx_granted(call_id: int) -> BitWriter:
    return BitWriter().u(2, 3).u(11, 5).u(call_id, 14).u(0, 2).u(1, 1).u(0, 1).u(0, 1).u(0, 1)


def cmce_d_release(call_id: int, cause: int = 1) -> BitWriter:
    return BitWriter().u(2, 3).u(6, 5).u(call_id, 14).u(cause, 5).u(0, 1)


def cmce_d_status(ssi: int, status: int) -> BitWriter:
    return BitWriter().u(2, 3).u(8, 5).u(1, 2).u(ssi, 24).u(status, 16).u(0, 1)


def cmce_d_sds(ssi: int, payload_bits: int, rng: np.random.Generator) -> BitWriter:
    w = BitWriter().u(2, 3).u(15, 5).u(1, 2).u(ssi, 24).u(3, 2).u(payload_bits, 11).u(0x82, 8)
    return w.bits(rng.integers(0, 2, payload_bits - 8)).u(0, 1)


def cmce_d_setup_calling(call_id: int, calling: int, prio: int = 5) -> BitWriter:
    """D-SETUP with the optional calling party (type-2 elements)."""
    return (BitWriter().u(2, 3).u(7, 5).u(call_id, 14).u(1, 4).u(0, 1).u(0, 1)
            .u(0, 3).u(0, 1).u(1, 2).u(0, 2).u(0, 2).u(1, 1).u(prio, 4)
            .u(1, 1).u(0, 1).u(0, 1).u(1, 1).u(1, 2).u(calling, 24).u(0, 1).u(0, 1))


def cmce_d_tx_granted_talker(call_id: int, talker: int) -> BitWriter:
    """D-TX-GRANTED with the optional transmitting party."""
    return (BitWriter().u(2, 3).u(11, 5).u(call_id, 14).u(0, 2).u(1, 1).u(0, 1).u(0, 1)
            .u(1, 1).u(0, 1).u(1, 1).u(1, 2).u(talker, 24).u(0, 1))


def cmce_d_connect(call_id: int, prio: int = 7) -> BitWriter:
    return (BitWriter().u(2, 3).u(2, 5).u(call_id, 14).u(1, 4).u(0, 1).u(0, 1).u(0, 2).u(1, 1).u(0, 1)
            .u(1, 1).u(1, 1).u(prio, 4).u(0, 1).u(0, 1).u(0, 1).u(0, 1))


def network_time(year: int, day: int, seconds: int, offset_quarters: int) -> int:
    utc = (day * 86400 + seconds) // 2
    sign = 1 if offset_quarters < 0 else 0
    return (utc << 24) | (sign << 23) | (abs(offset_quarters) << 17) | ((year - 2000) << 11) | 0x7FF


def mle_d_nwrk_broadcast(neighbours: list[tuple[int, int, int | None]], time48: int | None) -> BitWriter:
    """D-NWRK-BROADCAST with optional network time and neighbour cells ``(ca, carrier, la or None)``."""
    w = BitWriter().u(5, 3).u(2, 3).u(0x1234, 16).u(2, 2).u(1, 1)
    if time48 is None:
        w.u(0, 1)
    else:
        w.u(1, 1).u(time48, 48)
    w.u(1, 1).u(len(neighbours), 3)
    for ca, carrier, la in neighbours:
        w.u(ca, 5).u(2, 2).u(1, 1).u(3, 2).u(carrier, 12)
        if la is None:
            w.u(0, 1)
        else:  # O-bit, then P-bits: extension, MCC, MNC absent, LA present, the rest absent
            w.u(1, 1).u(0, 1).u(0, 1).u(0, 1).u(1, 1).u(la, 14).u(0, 6)
    return w.u(0, 1)


def access_define(gssi: int | None = None) -> np.ndarray:
    w = BitWriter().u(2, 2).u(1, 2).u(0, 1).u(0, 2).u(8, 4).u(5, 4).u(3, 4).u(0, 1).u(0, 4).u(0, 3)
    if gssi is None:
        w.u(0, 2)
    else:
        w.u(2, 2).u(gssi, 24)
    w.u(1, 1)  # fill bit
    out = w.array()
    return np.concatenate([out, np.zeros(124 - len(out), np.uint8)])


def mm_d_lu_accept() -> BitWriter:
    return BitWriter().u(1, 3).u(5, 4).u(0, 3).u(0, 1)


# --------------------------------------------------------------- slots
class DownlinkGenerator:
    """Produces consecutive 510-bit bursts of a synthetic main control carrier."""

    def __init__(self, cell: CellConfig, seed: int = 0):
        self.cell = cell
        self.rng = np.random.default_rng(seed)
        self.scramb = C.scrambling_init(cell.mcc, cell.mnc, cell.cc)
        self.tn, self.fn, self.mn = 1, 1, 1
        self.call_id = 100
        self._frag_queue: list[np.ndarray] = []

    def _half(self, info124) -> np.ndarray:
        return C.encode_block(info124, C.SCH_HD, self.scramb)

    def _aach(self, hdr, f1, f2) -> np.ndarray:
        return C.scramble(C.rm3014_encode(pdu_aach(hdr, f1, f2)), self.scramb)

    def _mcch_message(self) -> np.ndarray:
        c = self.cell
        k = (self.mn * 18 + self.fn) % 6
        if k == 0:
            self.call_id = (self.call_id + 1) % 16384
            return mac_resource(c.group, llc(cmce_d_setup(self.call_id), "BL-ADATA"),
                                chan_alloc=_chan_alloc(c.main_carrier + 4, 0b0100))
        if k == 1:
            return mac_resource(c.group, llc(cmce_d_tx_granted_talker(self.call_id, c.ms)))
        if k == 2:
            return mac_resource(c.ms, llc(mm_d_lu_accept(), "BL-ADATA"))
        if k == 3:
            return mac_resource(c.ms, self.rng.integers(0, 2, 48), enc=1)
        if k == 4:
            return mac_resource(c.group, llc(cmce_d_release(self.call_id)))
        return mac_resource(None)

    def _sch_f(self) -> np.ndarray:
        c = self.cell
        if self._frag_queue:
            part = self._frag_queue.pop(0)
            return mac_end(part, 268) if not self._frag_queue else mac_frag(part, 268)
        k = (self.mn * 18 + self.fn) % 7
        if k == 0:
            return mac_resource(c.ms, llc(cmce_d_status(c.group, 0x8002), "BL-UDATA"), block_bits=268)
        if k == 1:
            return mac_resource(c.group, llc(cmce_d_setup_calling(self.call_id + 1000, c.ms), "BL-ADATA"),
                                block_bits=268)
        if k == 5:
            return mac_resource(c.ms, llc(cmce_d_connect(self.call_id)), block_bits=268)
        if k == 6:
            nt = network_time(2026, 317, 10 * 3600 + 23 * 60 + 12, 4)
            nb = [(1, c.main_carrier + 8, None), (2, c.main_carrier - 12, c.la + 1)]
            return mac_resource(0xFFFFFF, llc(mle_d_nwrk_broadcast(nb, nt), "BL-UDATA"), block_bits=268)
        if k == 2:  # long SDS, fragmented over three SCH/F blocks
            sdu = llc(cmce_d_sds(c.group, 520, self.rng), "BL-ADATA")
            first = 268 - 43
            rest = sdu[first:]
            self._frag_queue = [rest[: 268 - 4], rest[268 - 4 :]]
            return mac_resource(c.ms, sdu, block_bits=268, frag=True)
        return mac_resource(None, block_bits=268)

    def next_burst(self) -> np.ndarray:
        c, tn, fn, mn = self.cell, self.tn, self.fn, self.mn
        if fn == 18 and tn == 4 - ((mn + 1) % 4) and not c.secondary:
            bsch = C.encode_block(pdu_sync(c, tn, fn, mn), C.BSCH, C.BSCH_SCRAMBLING_INIT)
            bits = B.build_sdb(bsch, self._aach(0, 0, 0), self._half(pdu_sysinfo(c, mn)))
        elif fn == 18:
            bits = B.build_ndb(self._half(mac_resource(None) if c.secondary else pdu_sysinfo(c, mn)),
                               self._aach(0, 0, 0),
                               self._half(access_define(c.group if tn == 3 else None)), two_half_slots=True)
        elif tn == 1:
            second = pdu_sysinfo(c, mn) if fn % 4 == 0 and not c.secondary else mac_resource(None)
            bits = B.build_ndb(self._half(self._mcch_message()), self._aach(1, 2, 0), self._half(second),
                               two_half_slots=True)
        elif tn == 2:  # traffic (random payload), usage marker 9
            payload = self.rng.integers(0, 2, 432).astype(np.uint8)
            bits = B.build_ndb(payload[:216], self._aach(3, 9, 9), payload[216:], two_half_slots=False)
        elif tn == 3:
            full = C.encode_block(self._sch_f(), C.SCH_F, self.scramb)
            bits = B.build_ndb(full[:216], self._aach(1, 1, 0), full[216:], two_half_slots=False)
        else:
            full = C.encode_block(mac_resource(None, block_bits=268), C.SCH_F, self.scramb)
            bits = B.build_ndb(full[:216], self._aach(1, 0, 0), full[216:], two_half_slots=False)
        # advance TDMA time
        self.tn += 1
        if self.tn > 4:
            self.tn, self.fn = 1, self.fn + 1
            if self.fn > 18:
                self.fn, self.mn = 1, self.mn % 60 + 1
        return bits

    def bits(self, n_bursts: int) -> np.ndarray:
        return np.concatenate([self.next_burst() for _ in range(n_bursts)])


# --------------------------------------------------------------- signals
def baseband(cell: CellConfig, duration_s: float, seed: int = 0, start_fn: int = 1) -> np.ndarray:
    """Clean channel-rate (144 kHz) complex baseband of a synthetic carrier, unit power."""
    gen = DownlinkGenerator(cell, seed)
    gen.fn = start_fn
    n = int(np.ceil(duration_s * B.SYMBOL_RATE / B.SYMBOLS_PER_BURST)) + 1
    return modulate(gen.bits(n), SPS)


def impair(x: np.ndarray, fs: float, snr_db: float | None, cfo_hz: float = 0.0, ppm: float = 0.0,
           rng: np.random.Generator | None = None, add_noise: bool = True) -> np.ndarray:
    """Apply carrier offset, sample clock error and AWGN. SNR is Es/N0 of a unit-power signal."""
    rng = rng or np.random.default_rng(0)
    if ppm:
        t = np.arange(int(len(x) / (1 + ppm * 1e-6)) - 1) * (1 + ppm * 1e-6)
        x = (np.interp(t, np.arange(len(x)), x.real) + 1j * np.interp(t, np.arange(len(x)), x.imag))
    if cfo_hz:
        x = x * np.exp(2j * np.pi * cfo_hz * np.arange(len(x)) / fs)
    if snr_db is not None and add_noise:
        n0 = 1.0 / (B.SYMBOL_RATE * 10 ** (snr_db / 10))
        sigma = np.sqrt(n0 * fs / 2)
        x = x + sigma * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x.astype(np.complex64)


def wideband(carriers: list[tuple[float, CellConfig, float]], fs: float, duration_s: float, snr_db: float,
             ppm: float = 0.0, seed: int = 0) -> np.ndarray:
    """Several carriers ``(offset_hz, cell, relative_level_db)`` in one capture at ``fs``.

    ``snr_db`` is Es/N0 of a 0 dB carrier. ``fs`` must be an integer multiple of 144 kHz.
    Secondary carriers (``cell.secondary``) are slot-aligned with the first carrier, like the
    carriers of one base station; other carriers get independent timing.
    """
    up = int(round(fs / CHANNEL_RATE))
    assert abs(up * CHANNEL_RATE - fs) < 1, "fs must be a multiple of 144 kHz"
    rng = np.random.default_rng(seed)
    total = None
    for i, (off, cell, lvl) in enumerate(carriers):
        bb = baseband(cell, duration_s, seed=seed + i, start_fn=1 if cell.secondary else 1 + 5 * i)
        x = signal.resample_poly(bb, up, 1) if up > 1 else bb
        x = x * 10 ** (lvl / 20) * np.exp(2j * np.pi * off * np.arange(len(x)) / fs)
        total = x if total is None else total[: len(x)] + x[: len(total)]
    total = total[: int(duration_s * fs)]
    return impair(total, fs, snr_db, ppm=ppm, rng=rng)


def write_cs8(path: str | Path, x: np.ndarray, fs: float, center_hz: float, **meta) -> None:
    peak = float(np.max(np.abs(np.r_[x.real, x.imag])))
    scale = 100.0 / peak if peak > 0 else 1.0
    iq = np.empty(2 * len(x), np.int8)
    iq[0::2] = np.clip(np.round(x.real * scale), -127, 127)
    iq[1::2] = np.clip(np.round(x.imag * scale), -127, 127)
    Path(path).write_bytes(iq.tobytes())
    Path(str(path) + ".json").write_text(json.dumps(dict(center_hz=center_hz, sample_rate=fs, format="cs8",
                                                         **meta), indent=2) + "\n")
