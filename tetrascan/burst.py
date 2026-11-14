"""TETRA continuous downlink burst structure (EN 300 392-2, 9.4.4) and pi/4-DQPSK mapping.

A timeslot carries 255 symbols = 510 bits at 18 ksymbol/s (14.167 ms). Four
timeslots make a TDMA frame, 18 frames a multiframe, 60 multiframes a hyperframe.

All offsets below are 0-based bit indices inside the 510-bit burst.
"""

from __future__ import annotations

import numpy as np

SYMBOL_RATE = 18000.0
BITS_PER_BURST = 510
SYMBOLS_PER_BURST = 255

# 9.4.4.3.2 - 9.4.4.3.4 training sequences
TRAIN_N = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0], np.uint8)
TRAIN_P = np.array([0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0], np.uint8)
TRAIN_Q = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1], np.uint8)
TRAIN_X = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1],
                   np.uint8)  # extended training sequence (uplink control bursts)
TRAIN_Y = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                    1, 1, 0, 0, 1, 1, 1], np.uint8)
FREQ_CORRECTION = np.r_[np.ones(8), np.zeros(64), np.ones(8)].astype(np.uint8)

# Burst kinds as recognised by the receiver.
SDB = "SB"  # synchronisation continuous downlink burst (training y)
NDB_FULL = "NDB1"  # normal downlink burst, training n: one full-slot logical channel
NDB_HALF = "NDB2"  # normal downlink burst, training p: two half-slot logical channels

# Normal continuous downlink burst:
#   q11..q22(12) ha(2) block1(216) bb1(14) n|p(22) bb2(16) block2(216) hb(2) q1..q10(10)
NDB_BLK1 = slice(14, 230)
NDB_BBK1 = slice(230, 244)
NDB_TRAIN = slice(244, 266)
NDB_BBK2 = slice(266, 282)
NDB_BLK2 = slice(282, 498)
# Synchronisation continuous downlink burst:
#   q11..q22(12) hc(2) f(80) sb(120) y(38) bb(30) block2(216) hd(2) q1..q10(10)
SB_FREQ = slice(14, 94)
SB_BSCH = slice(94, 214)
SB_TRAIN = slice(214, 252)
SB_BBK = slice(252, 282)
SB_BLK2 = slice(282, 498)
Q_HEAD = slice(0, 12)  # q11..q22
Q_TAIL = slice(500, 510)  # q1..q10

# Symbol index (0-based, inside a burst) where each training sequence starts.
SYM_TRAIN_MID = NDB_TRAIN.start // 2  # 122, n or p
SYM_TRAIN_SYNC = SB_TRAIN.start // 2  # 107, y
SYM_Q_TAIL = Q_TAIL.start // 2  # 250; q spans 11 symbols across the burst boundary

# Table 5.1: phase change for each dibit (first bit, second bit).
DIBIT_PHASE = {(0, 0): np.pi / 4, (0, 1): 3 * np.pi / 4, (1, 1): -3 * np.pi / 4, (1, 0): -np.pi / 4}


def bits_to_dphi(bits) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1, 2)
    lut = np.array([[np.pi / 4, 3 * np.pi / 4], [-np.pi / 4, -3 * np.pi / 4]])
    return lut[b[:, 0], b[:, 1]]


def template(bits) -> np.ndarray:
    """Expected differential phasors exp(j*dphi) of a training sequence."""
    return np.exp(1j * bits_to_dphi(bits)).astype(np.complex64)


TEMPLATE_N = template(TRAIN_N)
TEMPLATE_P = template(TRAIN_P)
TEMPLATE_Q = template(TRAIN_Q)
TEMPLATE_Y = template(TRAIN_Y)


def build_ndb(blk1, bbk30, blk2, two_half_slots: bool) -> np.ndarray:
    b = np.zeros(BITS_PER_BURST, np.uint8)
    b[Q_HEAD] = TRAIN_Q[10:]
    b[NDB_BLK1] = blk1
    b[NDB_BBK1] = bbk30[:14]
    b[NDB_TRAIN] = TRAIN_P if two_half_slots else TRAIN_N
    b[NDB_BBK2] = bbk30[14:]
    b[NDB_BLK2] = blk2
    b[Q_TAIL] = TRAIN_Q[:10]
    return b


def build_sdb(bsch120, bbk30, blk2) -> np.ndarray:
    b = np.zeros(BITS_PER_BURST, np.uint8)
    b[Q_HEAD] = TRAIN_Q[10:]
    b[SB_FREQ] = FREQ_CORRECTION
    b[SB_BSCH] = bsch120
    b[SB_TRAIN] = TRAIN_Y
    b[SB_BBK] = bbk30
    b[SB_BLK2] = blk2
    b[Q_TAIL] = TRAIN_Q[:10]
    return b


def bbk_bits(burst: np.ndarray, kind: str) -> np.ndarray:
    """The 30 broadcast-block (AACH) bits of a burst (soft or hard)."""
    if kind == SDB:
        return burst[..., SB_BBK]
    return np.concatenate([burst[..., NDB_BBK1], burst[..., NDB_BBK2]], axis=-1)
