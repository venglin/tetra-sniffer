import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tetrascan import burst as B
from tetrascan import coding as C
from tetrascan import gsmtap, mac, scanner, sim
from tetrascan.dsp import CHANNEL_RATE, ChannelFilter, modulate
from tetrascan.lower_mac import LowerMac, TdmaTime
from tetrascan.receiver import MultiReceiver

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ coding
def test_crc_residue_and_check():
    rng = np.random.default_rng(0)
    for n in (60, 124, 268):
        info = rng.integers(0, 2, n).astype(np.uint8)
        full = np.r_[info, C.crc16_parity(info)]
        assert C.crc16(full) == C.CRC16_GOOD_RESIDUE
        assert C.crc16_check(full)[0]
        full[5] ^= 1
        assert not C.crc16_check(full)[0]


@pytest.mark.parametrize("ch", [C.BSCH, C.SCH_HD, C.SCH_F, C.SCH_HU])
def test_block_coding_roundtrip(ch):
    rng = np.random.default_rng(1)
    init = C.scrambling_init(901, 1234, 33)
    info = rng.integers(0, 2, (20, ch.k1)).astype(np.uint8)
    tx = np.array([C.encode_block(i, ch, init) for i in info])
    assert tx.shape == (20, ch.k3)
    # one hard error per block is always corrected
    soft = 1.0 - 2.0 * tx
    soft[np.arange(20), rng.integers(0, ch.k3, 20)] *= -1
    dec, ok = C.decode_blocks(soft, ch, init)
    assert ok.all()
    assert (dec == info).all()
    # soft decoding at ~4 % raw bit error rate
    soft = 1.0 - 2.0 * tx + rng.normal(0, 0.55, tx.shape)
    _, ok = C.decode_blocks(soft, ch, init)
    assert ok.mean() >= 0.95
    # wrong scrambling code -> CRC fails
    _, ok2 = C.decode_blocks(1.0 - 2.0 * tx, ch, C.scrambling_init(901, 1234, 34))
    assert not ok2.any()


def test_interleaver_is_permutation():
    for k, a in ((120, 11), (216, 101), (432, 103)):
        x = np.arange(k)
        y = C.interleave(x, k, a)
        assert sorted(y) == list(x)
        assert (C.deinterleave(y, k, a) == x).all()
        # 8.2.4.1: type-4 bit k = 1 + (a*i mod K) holds type-3 bit i (1-based)
        i = 7
        assert y[(a * i) % k] == i - 1


def test_scrambler_matches_spec_recursion():
    """Compare with a direct implementation of 8.2.5.2: p(k) = sum c_i p(k-i)."""
    mcc, mnc, cc = 901, 9999, 17
    e = [(mcc >> (9 - i)) & 1 for i in range(10)] + [(mnc >> (13 - i)) & 1 for i in range(14)] + \
        [(cc >> (5 - i)) & 1 for i in range(6)]
    p = {}
    for i in range(1, 31):
        p[-(i - 1)] = e[i - 1]
    p[-30] = p[-31] = 1
    taps = (1, 2, 4, 5, 7, 8, 10, 11, 12, 16, 22, 23, 26, 32)
    seq = []
    for k in range(1, 433):
        p[k] = sum(p[k - i] for i in taps) & 1
        seq.append(p[k])
    assert (C.scrambling_sequence(C.scrambling_init(mcc, mnc, cc), 432) == np.array(seq)).all()


def test_rm3014_corrects_three_errors():
    rng = np.random.default_rng(2)
    for _ in range(50):
        info = rng.integers(0, 2, 14)
        soft = 1.0 - 2.0 * C.rm3014_encode(info)
        soft[rng.choice(30, 3, replace=False)] *= -1
        dec, nerr = C.rm3014_decode(soft)
        assert (dec[0] == info).all() and nerr[0] == 3


def test_puncture_rate_2_3():
    idx = C.puncture_index(216)
    assert len(idx) == 216 and idx.max() < 4 * 144
    assert list(idx[:6]) == [0, 1, 4, 8, 9, 12]


# ------------------------------------------------------------------- burst
def test_burst_layout():
    rng = np.random.default_rng(3)
    blk = rng.integers(0, 2, 216).astype(np.uint8)
    ndb = B.build_ndb(blk, np.ones(30, np.uint8), blk, two_half_slots=True)
    sdb = B.build_sdb(rng.integers(0, 2, 120), np.zeros(30, np.uint8), blk)
    assert len(ndb) == len(sdb) == 510
    assert (ndb[B.NDB_TRAIN] == B.TRAIN_P).all()
    assert (sdb[B.SB_TRAIN] == B.TRAIN_Y).all()
    # q is continuous across the burst boundary: q1..q10 at the end, q11..q22 at the start
    assert (np.r_[ndb[B.Q_TAIL], sdb[B.Q_HEAD]] == B.TRAIN_Q).all()
    assert (B.bbk_bits(ndb, B.NDB_HALF) == 1).all()
    # training sequences start on symbol boundaries
    assert B.NDB_TRAIN.start % 2 == B.SB_TRAIN.start % 2 == B.Q_TAIL.start % 2 == 0


def test_dibit_mapping():
    assert np.allclose(B.bits_to_dphi([0, 0, 0, 1, 1, 1, 1, 0]), [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])


def test_tdma_time():
    t = TdmaTime(4, 18, 60).advance(1)
    assert (t.tn, t.fn, t.mn) == (1, 1, 1)
    t = TdmaTime(2, 5, 7).advance(-6)
    assert (t.tn, t.fn, t.mn) == (4, 3, 7)


# --------------------------------------------------------------------- MAC
def test_parse_sync_and_sysinfo():
    cell = sim.CellConfig(mcc=262, mnc=1001, cc=63, la=16383, main_carrier=1234, offset=3)
    s = mac.parse_sync(sim.pdu_sync(cell, 3, 18, 42))
    assert (s["mcc"], s["mnc"], s["cc"], s["tn"], s["fn"], s["mn"]) == (262, 1001, 63, 3, 18, 42)
    pdus = mac.parse_mac_block(sim.pdu_sysinfo(cell, 777), "BNCH")
    d = pdus[0]
    assert d["pdu"] == "SYSINFO" and d["la"] == 16383 and d["hyperframe"] == 777
    assert d["dl_hz"] == 400e6 + 1234 * 25e3 + 12500
    assert "air interface encryption" in d["services"]


def test_parse_mac_resource_with_chan_alloc_and_l3():
    blk = sim.mac_resource(1234567, sim.llc(sim.cmce_d_setup(321, prio=9), "BL-ADATA"),
                           chan_alloc=sim._chan_alloc(1000, 0b0010))
    d = mac.parse_mac_block(blk, "SCH/HD")[0]
    assert d["pdu"] == "MAC-RESOURCE" and d["ssi"] == 1234567
    assert d["chan_alloc"]["carrier"] == 1000 and d["chan_alloc"]["timeslots"] == 0b0010
    l3 = mac.parse_tm_sdu(d["sdu"])
    assert l3["llc"] == "BL-ADATA" and l3["mle"] == "CMCE" and l3["l3"] == "D-SETUP"
    assert l3["call_id"] == 321 and l3["priority"] == 9 and l3["comm_type"] == "group"


def test_null_pdu_and_encrypted():
    assert mac.parse_mac_block(sim.mac_resource(None), "SCH/HD")[0].get("null")
    d = mac.parse_mac_block(sim.mac_resource(42, np.ones(40, np.uint8), enc=2), "SCH/HD")[0]
    assert d["enc"] == 2
    assert mac.parse_tm_sdu(d["sdu"], encrypted=True) == {"encrypted": True}


# --------------------------------------------------------------------- DSP
@pytest.mark.parametrize("fs,offset", [(2.304e6, 312.5e3), (2.4e6, -100e3), (576e3, 0.0)])
def test_channel_filter_streaming_matches_oneshot(fs, offset):
    rng = np.random.default_rng(4)
    x = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)).astype(np.complex64)
    one = ChannelFilter(fs, offset).process(x)
    cf = ChannelFilter(fs, offset)
    parts, i = [], 0
    for n in (1, 777, 4096, 31, 65536, 200_000):
        parts.append(cf.process(x[i : i + n]))
        i += n
    streamed = np.concatenate(parts)
    assert abs(len(streamed) - len(x) * CHANNEL_RATE / fs) <= 2
    m = min(len(one), len(streamed))
    assert np.allclose(one[:m], streamed[:m], atol=1e-4)


def test_channel_filter_shifts_tone():
    fs, off = 2.304e6, 250e3
    n = np.arange(400_000)
    x = np.exp(2j * np.pi * (off + 1000) * n / fs).astype(np.complex64)
    y = ChannelFilter(fs, off).process(x)[2000:]
    f = np.angle(np.sum(y[1:] * np.conj(y[:-1]))) / (2 * np.pi) * CHANNEL_RATE
    assert abs(f - 1000) < 5


def test_modulator_symbol_phases():
    bits = np.random.default_rng(5).integers(0, 2, 200)
    x = modulate(bits)
    assert abs(np.mean(np.abs(x[200:-200]) ** 2) - 1) < 0.2


# -------------------------------------------------------------- end to end
def _decode(x, fs, offset):
    rx = MultiReceiver(fs, 0.0, [offset])
    events = []

    class P:
        def emit(self, ch, blk, pdus):
            events.append((blk, pdus))

    rx.printer = P()
    for i in range(0, len(x), 50_000):
        rx.feed(x[i : i + 50_000])
    return rx.channels[0], events


def test_end_to_end_with_impairments():
    fs = 576e3
    cell = sim.CellConfig()
    x = sim.wideband([(-75e3, cell, 0.0)], fs, 3.0, snr_db=14.0, ppm=-20.0, seed=3)
    x = (x * np.exp(2j * np.pi * 3200 * np.arange(len(x)) / fs)).astype(np.complex64)
    ch, events = _decode(x, fs, -75e3)
    assert ch.locked and abs(ch.demod.cfo - 3200) < 100
    assert ch.sync and (ch.sync["mcc"], ch.sync["mnc"], ch.sync["cc"]) == (cell.mcc, cell.mnc, cell.cc)
    assert ch.sysinfo and ch.sysinfo["la"] == cell.la
    st = ch.lmac.stats
    assert st["blk_bad"] <= 0.03 * st["blk_ok"]
    l3 = [p.get("tm", {}) for _, ps in events for p in ps if p.get("tm")]
    names = {t.get("l3") for t in l3}
    assert {"D-SETUP", "D-TX-GRANTED", "D-RELEASE", "D-LOCATION-UPDATE-ACCEPT", "D-STATUS", "D-SDS-DATA"} <= names
    assert any(t.get("encrypted") for t in l3)
    sds = [t for t in l3 if t.get("l3") == "D-SDS-DATA"]
    assert sds and all(t["sds_bits"] == 520 and t["sds_proto"] == "text messaging" for t in sds)


def test_no_lock_on_noise():
    fs = 576e3
    rng = np.random.default_rng(9)
    x = (rng.standard_normal(int(fs * 1.5)) + 1j * rng.standard_normal(int(fs * 1.5))).astype(np.complex64) * 0.1
    ch, events = _decode(x, fs, 50e3)
    assert ch.lmac.stats["bsch_ok"] == 0 and not events


def test_lower_mac_needs_bsch_first():
    lm = LowerMac()
    assert lm.process([]) == []
    assert lm.scramb is None


# ------------------------------------------------------------ scanner / IO
def test_find_candidates_and_windows():
    fs, nfft = 2.304e6, 1024
    cell = sim.CellConfig()
    x = sim.wideband([(300e3, cell, 0.0), (-412.5e3, cell, -10.0)], fs, 0.5, snr_db=30.0)
    from tetrascan.dsp import db, psd

    f, p = psd(x, fs, nfft)
    cands = scanner.find_candidates(f + 400e6, db(p), 10.0)
    freqs = sorted(c.freq_hz for c in cands if c.tetra_like)
    assert freqs == [400e6 - 412.5e3, 400e6 + 300e3]
    plan = scanner.plan_windows(freqs)
    assert len(plan) == 1
    lo, group = plan[0]
    assert all(40e3 <= abs(g - lo) <= 0.9e6 for g in group)


def test_gsmtap_header_and_pcap(tmp_path):
    pkt = gsmtap.gsmtap_packet("SCH/HD", np.ones(124, np.uint8), tn=3, frame_number=1000, arfcn=3612)
    v, hl, typ, ts, arfcn, _, _, fn, sub = struct.unpack(">BBBBHbbIB", pkt[:13])
    assert (v, hl, typ, ts, arfcn, fn, sub) == (2, 4, 5, 2, 3612, 1000, 4)
    assert len(pkt) == 16 + 16 and pkt[-1] == 0xF0
    w = gsmtap.PcapWriter(str(tmp_path / "x.pcap"))
    w.send(pkt, 1.5)
    w.close()
    data = (tmp_path / "x.pcap").read_bytes()
    assert data[:4] == b"\xd4\xc3\xb2\xa1" and len(data) == 24 + 16 + 28 + len(pkt)


def test_bundled_example_decodes():
    out = subprocess.run([sys.executable, "-m", "tetrascan", "file", str(ROOT / "examples" / "synthetic.cs8")],
                         capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "SYNC MCC 901 MNC 9999 CC 17" in out.stdout
    assert "SYNC MCC 901 MNC 9999 CC 42" in out.stdout
    assert "D-SDS-DATA" in out.stdout and "crc 100%" in out.stderr
