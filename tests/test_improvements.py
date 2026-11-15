"""Tests for coherent detection, shared colour codes and the extended PDU parsers."""

import queue

import numpy as np
import pytest

from tetrascan import mac, scanner, sim
from tetrascan.burst import bits_to_dphi
from tetrascan.coherent import coherent_soft
from tetrascan.receiver import MultiReceiver


# ------------------------------------------------------------ coherent detection
def _symbols(rng, b, n, esn0_db, drift=0.002):
    bits = rng.integers(0, 2, (b, 2 * (n - 1)))
    ph = np.concatenate([np.zeros((b, 1)), np.cumsum([bits_to_dphi(x) for x in bits], axis=1)], axis=1)
    ph += rng.uniform(0, 2 * np.pi, (b, 1)) + drift * np.arange(n)
    sigma = np.sqrt(1 / (2 * 10 ** (esn0_db / 10)))
    v = np.exp(1j * ph) + sigma * (rng.standard_normal((b, n)) + 1j * rng.standard_normal((b, n)))
    return bits, v


def test_coherent_soft_noiseless_mapping():
    bits, v = _symbols(np.random.default_rng(0), 4, 256, 60)
    soft = coherent_soft(v)
    assert soft.shape == (4, 510)
    assert ((soft < 0) == bits).all()


def test_coherent_beats_differential():
    bits, v = _symbols(np.random.default_rng(1), 200, 256, 8)
    d = v[:, 1:] * np.conj(v[:, :-1])
    diff = np.empty(bits.shape)
    diff[:, 0::2], diff[:, 1::2] = d.imag, d.real
    ber_diff = np.mean((diff < 0) != bits)
    ber_coh = np.mean((coherent_soft(v) < 0) != bits)
    assert ber_coh < 0.6 * ber_diff


def _run(x, fs, freqs, method="coherent", known=None):
    rx = MultiReceiver(fs, 0.0, freqs, method=method, known_cells=known)
    events = []

    class P:
        def emit(self, ch, blk, pdus):
            events.append((ch.freq_hz, blk, pdus))

    rx.printer = P()
    for i in range(0, len(x), 50_000):
        rx.feed(x[i : i + 50_000])
    return rx, events


def test_coherent_decodes_more_at_low_snr():
    fs = 576e3
    x = sim.wideband([(60e3, sim.CellConfig(), 0.0)], fs, 3.0, snr_db=7.0, ppm=10, seed=4)
    ok = {}
    for m in ("coherent", "differential"):
        rx, _ = _run(x, fs, [60e3], method=m)
        st = rx.channels[0].lmac.stats
        ok[m] = st["blk_ok"] + st["bsch_ok"]
    assert ok["coherent"] >= ok["differential"] + 25, ok


# ------------------------------------------------------------ shared colour codes
def test_secondary_carrier_shares_code_and_time():
    fs = 576e3
    main = sim.CellConfig()
    sec = sim.CellConfig(secondary=True, main_carrier=3616)
    x = sim.wideband([(100e3, main, 0.0), (-150e3, sec, 0.0)], fs, 3.0, snr_db=15, seed=2)
    rx, events = _run(x, fs, [100e3, -150e3])
    s_main, s_sec = (ch.summary() for ch in rx.channels)
    assert s_main["code_source"] == "bsch"
    assert s_sec["code_source"] == "shared" and s_sec["bsch_ok"] == 0 and s_sec["blk_ok"] > 50
    cell = [p for f, b, ps in events if f == -150e3 for p in ps if p["pdu"] == "CELL-CODE"]
    assert cell and cell[0]["time_inherited"] and cell[0]["cc"] == main.cc
    # inherited TDMA time agrees with the main carrier for the same slot
    t_main = {round(b.burst.start): str(b.time) for f, b, _ in events if f == 100e3 and b.time}
    t_sec = {round(b.burst.start): str(b.time) for f, b, _ in events if f == -150e3 and b.time}
    common = [k for k in t_sec if any(abs(k - m) <= 2 and t_main[m] == t_sec[k] for m in t_main)]
    assert len(common) > 20


def test_manual_cell_code():
    fs = 576e3
    sec = sim.CellConfig(secondary=True)
    x = sim.wideband([(80e3, sec, 0.0)], fs, 2.0, snr_db=15, seed=5)
    rx, _ = _run(x, fs, [80e3])
    assert rx.channels[0].summary().get("code_source") is None
    rx, _ = _run(x, fs, [80e3], known=[(sec.mcc, sec.mnc, sec.cc)])
    s = rx.channels[0].summary()
    assert s["code_source"] == "manual" and s["blk_ok"] > 30
    rx, _ = _run(x, fs, [80e3], known=[(sec.mcc, sec.mnc, sec.cc + 1)])  # wrong code is not adopted
    assert rx.channels[0].summary().get("code_source") is None


class FakeHackRF:
    """Plays synthetic captures through the scanner's verify pass."""

    def __init__(self, captures: dict[float, bytes]):
        self.captures = captures
        self.queue = queue.Queue()

    def retune(self, lo, settle_bytes=0):
        raw = self.captures[round(lo)]
        for i in range(0, len(raw), 262144):
            self.queue.put(raw[i : i + 262144])


def _cs8(x):
    x = x / np.max(np.abs(np.r_[x.real, x.imag])) * 100
    iq = np.empty(2 * len(x), np.int8)
    iq[0::2], iq[1::2] = np.round(x.real), np.round(x.imag)
    return iq.tobytes()


def test_scanner_second_pass_uses_codes_from_other_windows():
    fs = scanner.VERIFY_RATE
    main = sim.CellConfig(main_carrier=3612)
    sec = sim.CellConfig(main_carrier=3612, secondary=True)
    f_sec, f_main = 400e6 + 3540 * 25e3, 400e6 + 3612 * 25e3  # 1.8 MHz apart -> two windows
    plan = scanner.plan_windows([f_sec, f_main])
    assert len(plan) == 2
    caps = {}
    for lo, group in plan:
        cell = sec if group == [f_sec] else main
        caps[round(lo)] = _cs8(sim.wideband([(group[0] - lo, cell, 0.0)], fs, 1.5, snr_db=20, seed=3))
    cands = [scanner.Candidate(f, -40, 20, 22, True) for f in (f_sec, f_main)]
    scanner.verify(FakeHackRF(caps), cands, dwell=1.5)
    r_sec, r_main = cands[0].result, cands[1].result
    assert r_main["code_source"] == "bsch"
    assert r_sec["code_source"] == "manual" and r_sec["blk_ok"] > 0  # learned in the second pass
    assert scanner.status(cands[0]) == "TETRA"


# ------------------------------------------------------------ PDU parsers
def test_access_define():
    d = mac.parse_mac_block(sim.access_define(2345678), "SCH/HD")[0]
    assert d["pdu"] == "ACCESS-DEFINE"
    assert (d["access_code"], d["imm"], d["wt"], d["nu"], d["gssi"]) == ("A", 8, 5, 3, 2345678)


def test_d_nwrk_broadcast():
    nt = sim.network_time(2026, 317, 10 * 3600 + 23 * 60 + 12, -2)
    sdu = sim.llc(sim.mle_d_nwrk_broadcast([(1, 3600, None), (31, 4095, 16383)], nt), "BL-UDATA")
    l3 = mac.parse_tm_sdu(sdu)
    assert l3["mle"] == "MLE" and l3["l3"] == "D-NWRK-BROADCAST"
    assert l3["cell_reselect_params"] == 0x1234 and l3["cell_service_level"] == 2
    assert l3["network_time"] == {"utc": "2026-11-14 10:23:12", "local_offset_min": -30}
    n1, n2 = l3["neighbours"]
    assert (n1["ca"], n1["carrier"], n1["synchronised"]) == (1, 3600, 1) and "la" not in n1
    assert (n2["ca"], n2["carrier"], n2["la"]) == (31, 4095, 16383)


def test_cmce_optional_elements():
    l3 = mac.parse_tm_sdu(sim.llc(sim.cmce_d_setup_calling(42, 7654321, prio=11), "BL-ADATA"))
    assert (l3["l3"], l3["call_id"], l3["priority"], l3["calling_ssi"]) == ("D-SETUP", 42, 11, 7654321)
    assert l3["circuit_mode"] == "speech" and l3["comm_type"] == "group"
    l3 = mac.parse_tm_sdu(sim.llc(sim.cmce_d_tx_granted_talker(43, 1234567)))
    assert (l3["l3"], l3["call_id"], l3["tx_grant"], l3["transmitting_ssi"]) == \
        ("D-TX-GRANTED", 43, "granted", 1234567)
    l3 = mac.parse_tm_sdu(sim.llc(sim.cmce_d_connect(44, prio=6)))
    assert (l3["l3"], l3["call_id"], l3["priority"]) == ("D-CONNECT", 44, 6)


@pytest.mark.parametrize("method", ["coherent", "differential"])
def test_end_to_end_new_pdus(method):
    fs = 576e3
    x = sim.wideband([(50e3, sim.CellConfig(), 0.0)], fs, 3.0, snr_db=15, seed=8)
    _, events = _run(x, fs, [50e3], method=method)
    pdus = [p for _, _, ps in events for p in ps]
    names = {p["pdu"] for p in pdus} | {p.get("tm", {}).get("l3") for p in pdus if p.get("tm")}
    assert {"ACCESS-DEFINE", "D-NWRK-BROADCAST", "D-CONNECT", "D-TX-GRANTED", "D-SETUP"} <= names
    lchans = {b.lchan for _, b, ps in events for p in ps if p["pdu"] == "ACCESS-DEFINE"}
    assert lchans == {"SCH/HD"}  # only SYSINFO is labelled BNCH


# ------------------------------------------------------------ MAC-END channel allocation
def _upper_mac_run(enc, with_head=True):
    from tetrascan.lower_mac import Block, TdmaTime, UpperMac

    um = UpperMac()
    sdu = np.random.default_rng(0).integers(0, 2, 300).astype(np.uint8)
    blocks = []
    if with_head:
        blocks.append(sim.mac_resource(1234567, sdu, block_bits=268, enc=enc, frag=True))
    blocks.append(sim.mac_end(sdu[:40], 268, chan_alloc=sim._chan_alloc(3694, 0b0101)))
    out = []
    for fn, bits in enumerate(blocks, 1):
        out += um.process(Block("SCH/F", True, bits, None, TdmaTime(tn=1, fn=fn, mn=1)))
    return [d for d in out if d["pdu"] == "MAC-END"][0]


def test_mac_end_chan_alloc_clear():
    d = _upper_mac_run(enc=0)
    assert d["chan_alloc"]["carrier"] == 3694 and d["chan_alloc"]["timeslots"] == 0b0101
    assert "encrypted" not in d["tm"]


def test_mac_end_chan_alloc_encrypted_fragment():
    d = _upper_mac_run(enc=3)
    assert "chan_alloc" not in d and d["chan_alloc_encrypted"]
    assert d["tm"]["encrypted"]
    assert "chan-alloc(encrypted)" in mac.describe(d)


def test_mac_end_chan_alloc_without_first_fragment():
    d = _upper_mac_run(enc=0, with_head=False)
    assert "chan_alloc" not in d and d["chan_alloc_unknown"]


def test_pdus_after_mac_end_are_parsed():
    from tetrascan.lower_mac import Block, TdmaTime, UpperMac

    um = UpperMac()
    sdu = np.random.default_rng(1).integers(0, 2, 300).astype(np.uint8)
    first = sim.mac_resource(1234567, sdu, block_bits=268, frag=True)
    end = sim.mac_end(sdu[:40], 268)
    li = int("".join(map(str, end[5:11])), 2)
    status = sim.llc(sim.cmce_d_status(7654321, 0x8001))
    second = np.concatenate([end[: li * 8], sim.mac_resource(7654321, status, block_bits=268 - li * 8)])
    out = []
    for fn, bits in enumerate([first, second], 1):
        out += um.process(Block("SCH/F", True, bits, None, TdmaTime(tn=1, fn=fn, mn=1)))
    names = [d["pdu"] for d in out]
    assert names == ["MAC-RESOURCE", "MAC-END", "MAC-RESOURCE"]
    assert out[1]["reassembled_from"] == 1234567
    assert out[2]["ssi"] == 7654321 and out[2]["tm"]["l3"] == "D-STATUS"


# ------------------------------------------------------------ activity report
def test_activity_report_end_to_end():
    fs = 576e3
    x = sim.wideband([(50e3, sim.CellConfig(), 0.0)], fs, 3.0, snr_db=15, seed=8)
    rx, _ = _run(x, fs, [50e3])
    (ch, rep, text), = rx.reports()
    assert rep["slots"][2]["dl"] == {"traffic": 100.0} and rep["slots"][2]["ul"] == {"traffic": 100.0}
    assert rep["slots"][1]["dl"] == {"common control": 100.0}
    assert rep["sessions"]["ongoing"] == 1 and rep["sessions"]["usage_markers"] == 1
    sig = rep["signalling"]
    assert sig["pdus"] > 30 and sig["addresses"] == 3 and 0 < sig["encrypted"] < sig["pdus"]
    assert {"D-SETUP", "D-STATUS"} <= set(rep["clear_l3"])
    assert "TN2  DL traffic 100%" in text
    assert "1234567" not in text  # counts only, no addresses


def test_activity_sessions():
    from types import SimpleNamespace
    from tetrascan import burst as B
    from tetrascan.activity import Activity
    from tetrascan.dsp import CHANNEL_RATE
    from tetrascan.lower_mac import Block, TdmaTime

    act = Activity(gap_s=3.0)
    t0 = TdmaTime(1, 1, 1)

    def feed(t_s, um, n_frames):
        for k in range(n_frames):
            slots = 4 * (int(t_s / (255 / 18000)) // 4 + k) + 2  # always TN3
            t = t0.advance(slots)
            burst = SimpleNamespace(start=slots * 255 * 8 + 0.0, kind=B.NDB_FULL)
            traffic = um > 3
            a = dict(pdu="ACCESS-ASSIGN", hdr=3 if traffic else 1, field1=um, field2=um if traffic else 0,
                     traffic=traffic and t.fn != 18, dl_usage=2 if t.fn == 18 else um)
            act.add(Block("AACH", True, None, burst, t), [a])

    assert 255 * 8 / CHANNEL_RATE == pytest.approx(255 / 18000)
    feed(0.0, 10, 36)    # ~2 s call on UM 10 (frame 18 inside does not split it)
    feed(2.1, 0, 5)      # unallocated
    feed(2.5, 11, 18)    # ~1 s on UM 11
    feed(8.0, 11, 18)    # same UM after a 4.5 s gap: a new call
    rep = act.report(reset=True)
    assert rep["sessions"]["count"] == 2 and rep["sessions"]["ongoing"] == 1 and rep["sessions"]["usage_markers"] == 2
    assert rep["sessions"]["max_s"] == pytest.approx(2.04, abs=0.1)
    assert rep["slots"][3]["sessions"] == 2 and "unallocated" in rep["slots"][3]["dl"]
    rep = act.report()
    assert rep["sessions"]["count"] == 0 and rep["sessions"]["ongoing"] == 1  # the open call carries over


# ------------------------------------------------------------ monitoring log
@pytest.mark.parametrize("suffix", [".db", ".csv"])
def test_report_log_and_stats(tmp_path, suffix):
    from tetrascan import monitor

    fs = 576e3
    x = sim.wideband([(50e3, sim.CellConfig(), 0.0)], fs, 3.0, snr_db=15, seed=8)
    rx, _ = _run(x, fs, [50e3])
    path = tmp_path / ("log" + suffix)
    log = monitor.ReportLog(str(path))
    t0 = 1_800_000_000.0  # a fixed Unix time
    (ch, rep, _), = rx.reports(reset=True)
    rep["session_list"] = [dict(tn=2, um=12, start=1.0, duration_s=3.5)]
    row = monitor.report_row(rep, ch.freq_hz, t0, t0 + 3, ch.lmac.cell, 1001)
    log.write([row], monitor.session_rows(rep, ch.freq_hz, t0))
    log.write([dict(row, t_start=row["t_start"] + 3, t_end=row["t_end"] + 3)], [])
    log.close()
    rows = monitor.read_reports(str(path))
    assert len(rows) == 2 and rows[0]["cc"] == sim.CellConfig().cc and rows[0]["tn2_traffic"] == 100.0
    assert rows[0]["crc_pct"] == 100.0 and rows[0]["pdus"] > 30
    (s,) = monitor.summarise(rows, "hour")
    assert s["traffic_erlang"] == 1.0 and s["max_addresses"] == 3 and s["crc_pct"] == 100.0
    if suffix == ".csv":
        assert (tmp_path / "log.sessions.csv").read_text().splitlines()[1].endswith(",2,12,3.5")
    else:
        import sqlite3
        assert sqlite3.connect(path).execute("SELECT tn, usage_marker, duration_s FROM sessions").fetchall() \
            == [(2, 12, 3.5)]


def test_cli_file_log_quiet(tmp_path, capsys):
    from pathlib import Path
    from tetrascan import cli

    log = tmp_path / "act.db"
    ex = Path(__file__).parent.parent / "examples" / "synthetic.cs8"
    assert cli.main(["file", str(ex), "-q", "--report", "1", "--log", str(log)]) == 0
    assert capsys.readouterr().out == ""  # -q: no PDUs on stdout
    assert cli.main(["stats", str(log)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("hour") and len(out) == 3  # header + two carriers
