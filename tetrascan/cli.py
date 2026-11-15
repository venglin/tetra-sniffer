"""Command line interface: scan, live, record, file, selftest."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__, gsmtap, scanner, sim
from .dsp import iq_from_cs8, psd
from .mac import mhz
from .receiver import MultiReceiver, Printer

LIVE_RATE = scanner.VERIFY_RATE


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cells(args) -> list[tuple[int, int, int]]:
    out = []
    for v in getattr(args, "cell", None) or []:
        try:
            mcc, mnc, cc = (int(x) for x in v.split(":"))
        except ValueError:
            raise SystemExit(f"--cell expects MCC:MNC:CC, got {v!r}")
        out.append((mcc, mnc, cc))
    return out


def _mhz_list(values) -> list[float]:
    return [float(v) * 1e6 for v in values or []]


def _open_hackrf(args, freq: float, rate: float):
    from .hackrf import HackRF

    dev = HackRF(serial=args.serial)
    corr = 1 + args.ppm * 1e-6
    dev.configure(freq / corr, sample_rate=rate, lna_db=args.lna, vga_db=args.vga, amp=args.amp,
                  bb_filter_hz=min(rate * 0.75, 15e6) if rate < 10e6 else None)
    return dev


def _sinks(args) -> list:
    s = []
    if getattr(args, "gsmtap", None):
        s.append(gsmtap.GsmtapUdp(args.gsmtap))
    if getattr(args, "pcap", None):
        s.append(gsmtap.PcapWriter(args.pcap))
    return s


def _window(freqs: list[float]) -> float:
    plan = scanner.plan_windows(freqs)
    if len(plan) != 1:
        raise SystemExit(f"channels must fit into one {scanner.VERIFY_SPAN / 1e6:.1f} MHz window")
    return plan[0][0]


# ------------------------------------------------------------------ scan
def cmd_scan(args) -> int:
    f0, f1 = args.start * 1e6, args.stop * 1e6
    log = _log if args.verbose else None
    dev = _open_hackrf(args, f0, scanner.SURVEY_RATE)
    try:
        dev.start()
        t = time.monotonic()
        f, p, clip = scanner.survey(dev, f0, f1, dwell=args.survey_dwell, log=log, ppm=args.ppm)
        _log(f"survey {args.start:.3f}-{args.stop:.3f} MHz done in {time.monotonic() - t:.1f} s, "
             f"max clipping {100 * clip:.2f} %")
        if clip > 0.001:
            _log("warning: ADC clipping, consider lower --lna / --vga")
        cands = scanner.find_candidates(f, p, args.threshold)
        todo = [c for c in cands if c.tetra_like or args.all]
        _log(f"{len(cands)} carriers above {args.threshold} dB, {len(todo)} to verify")
        if todo and not args.no_verify:
            dev.stop()
            dev.configure(todo[0].freq_hz, sample_rate=scanner.VERIFY_RATE, lna_db=args.lna, vga_db=args.vga,
                          amp=args.amp, bb_filter_hz=1.75e6)
            dev.start()
            scanner.verify(dev, todo, dwell=args.dwell, log=log, ppm=args.ppm, method=args.demod)
    finally:
        dev.close()
    ppm = scanner.estimate_ppm(cands)
    shown = cands if args.all or args.no_verify else [c for c in cands if c.tetra_like]
    if args.json:
        for c in shown:
            print(json.dumps(dict(freq_hz=c.freq_hz, power_db=round(c.power_db, 1), snr_db=round(c.snr_db, 1),
                                  width_khz=c.width_khz, tetra_like=c.tetra_like, status=scanner.status(c),
                                  **{k: v for k, v in c.result.items() if k != "freq_hz"})))
        return 0
    print(f"{'freq MHz':>10} {'SNR':>5} {'width':>6}  {'status':<8} {'MCC':>4} {'MNC':>5} {'CC':>3} {'LA':>6} "
          f"{'MER':>5}  notes")
    for c in sorted(shown, key=lambda c: c.freq_hz):
        r = c.result
        status = scanner.status(c)
        freq = scanner.refine_frequency(c, ppm) if status.startswith("TETRA") else c.freq_hz
        notes = []
        if "main_carrier_hz" in r:
            if abs(r["main_carrier_hz"] - freq) < 1:
                notes.append("main carrier")
            else:
                notes.append(f"cell main carrier {mhz(r['main_carrier_hz'])}")
        if status == "TETRA?":
            notes.append("bursts locked, no colour code found")
        if r.get("code_source") == "shared":
            notes.append(f"secondary carrier, colour code from {r['code_from']} MHz")
        mer = f"{r['mer_db']:.1f}" if r.get("mer_db") is not None else ""
        print(f"{mhz(freq):>10} {c.snr_db:5.1f} {c.width_khz:5.1f}k  {status:<8} {r.get('mcc', ''):>4} "
              f"{r.get('mnc', ''):>5} {r.get('cc', ''):>3} {r.get('la', ''):>6} {mer:>5}  {'; '.join(notes)}")
    if ppm is not None:
        print(f"HackRF clock error: {ppm + args.ppm:+.2f} ppm (pass --ppm {ppm + args.ppm:.1f} to correct it)")
    return 0


# ------------------------------------------------------------------ live
def cmd_live(args) -> int:
    freqs = _mhz_list(args.freq)
    lo = _window(freqs)
    sinks = _sinks(args)
    dev = _open_hackrf(args, lo, LIVE_RATE)
    _log(f"tuned to {mhz(lo)} MHz at {LIVE_RATE / 1e6:.3f} Msps, channels: {', '.join(mhz(f) for f in freqs)}")
    printer = Printer(verbose=args.verbose, as_json=args.json, t0=time.time(), sinks=sinks)
    rx = MultiReceiver(LIVE_RATE, lo, freqs, printer, method=args.demod, known_cells=_cells(args))
    t_status = time.monotonic() + args.status if args.status else None
    t_end = time.monotonic() + args.duration if args.duration else None
    t_report = time.monotonic() + (args.report or 0)
    clipped = total = 0
    try:
        dev.start()
        printer.t0 = time.time()
        while True:
            try:
                raw = dev.queue.get(timeout=1.0)
            except queue.Empty:
                if not dev.is_streaming():
                    _log("HackRF stopped streaming")
                    return 1
                continue
            raw_i8 = np.frombuffer(raw, np.int8)
            clipped += int(np.count_nonzero((raw_i8 == 127) | (raw_i8 <= -127)))
            total += len(raw_i8)
            rx.feed(iq_from_cs8(raw))
            now = time.monotonic()
            if t_status and now >= t_status:
                t_status = now + args.status
                _log(f"[status] {rx.status_line()} | clip {100 * clipped / max(total, 1):.3f} % | "
                     f"usb drop {dev.dropped_bytes // 2} smp, backlog {dev.queue.qsize()}")
                clipped = total = 0
            if args.report and now >= t_report:
                t_report = now + args.report
                _report(rx, args, printer, reset=True)
            if t_end and now >= t_end:
                break
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()
        for s in sinks:
            s.close()
    _log(f"[final] {rx.status_line()}")
    _report(rx, args, printer)
    return 0


# ---------------------------------------------------------------- record
def cmd_record(args) -> int:
    freqs = _mhz_list(args.freq)
    lo = _window(freqs) if freqs else args.center * 1e6
    nbytes = int(args.duration * LIVE_RATE) * 2
    dev = _open_hackrf(args, lo, LIVE_RATE)
    got = 0
    start = time.time()
    try:
        with open(args.output, "wb") as f:
            dev.start()
            while got < nbytes:
                try:
                    raw = dev.queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                raw = raw[: nbytes - got]
                f.write(raw)
                got += len(raw)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()
    Path(args.output + ".json").write_text(json.dumps(dict(
        center_hz=lo, sample_rate=LIVE_RATE, format="cs8", start_time=start, channels_hz=freqs,
        lna_db=args.lna, vga_db=args.vga, amp=args.amp, ppm=args.ppm, dropped_samples=dev.dropped_bytes // 2),
        indent=2) + "\n")
    _log(f"wrote {got // 2} samples ({got / 2 / LIVE_RATE:.1f} s) to {args.output}")
    return 0


# ------------------------------------------------------------------ file
def _read_iq(path: str, fmt: str, offset_samples: int, count: int) -> np.ndarray:
    if fmt == "cs8":
        raw = np.fromfile(path, np.int8, count=2 * count if count >= 0 else -1, offset=2 * offset_samples)
        return raw.astype(np.float32).view(np.complex64) / 128.0
    if fmt == "cu8":
        raw = np.fromfile(path, np.uint8, count=2 * count if count >= 0 else -1, offset=2 * offset_samples)
        return (raw.astype(np.float32) - 127.5).view(np.complex64) / 128.0
    if fmt == "cf32":
        return np.fromfile(path, np.complex64, count=count, offset=8 * offset_samples)
    raise SystemExit(f"unknown format {fmt}")


def cmd_file(args) -> int:
    meta = {}
    side = Path(args.path + ".json")
    if side.exists():
        meta = json.loads(side.read_text())
    fmt = args.format or meta.get("format") or Path(args.path).suffix.lstrip(".") or "cs8"
    rate = args.rate * 1e6 if args.rate else meta.get("sample_rate")
    center = args.center * 1e6 if args.center else meta.get("center_hz", 0.0)
    if not rate:
        raise SystemExit("sample rate unknown: pass --rate (MHz) or provide a .json sidecar")
    freqs = _mhz_list(args.freq)
    if not freqs:
        probe = _read_iq(args.path, fmt, 0, int(rate * 0.5))
        f, p = psd(probe, rate, max(1024, int(2 ** np.ceil(np.log2(rate / 2500)))))
        cands = [c for c in scanner.find_candidates(f + center, 10 * np.log10(np.maximum(p, 1e-20)), args.threshold)
                 if c.tetra_like and abs(c.freq_hz - center) > 15e3 and abs(c.freq_hz - center) < rate / 2 - 20e3]
        freqs = [c.freq_hz for c in cands]
        if not freqs:
            raise SystemExit("no TETRA-like carriers found; pass channels with -f")
        _log(f"auto-detected channels: {', '.join(mhz(f) for f in freqs)}")
    sinks = _sinks(args)
    printer = Printer(verbose=args.verbose, as_json=args.json, t0=meta.get("start_time"), sinks=sinks)
    rx = MultiReceiver(rate, center, freqs, printer, method=args.demod, known_cells=_cells(args))
    chunk = 1 << 18
    pos = 0
    t_report = args.report or 0
    t = time.monotonic()
    while True:
        x = _read_iq(args.path, fmt, pos, chunk)
        if len(x) == 0:
            break
        rx.feed(x)
        pos += len(x)
        if args.report and pos / rate >= t_report:
            t_report += args.report
            _report(rx, args, printer, reset=True)
    for s in sinks:
        s.close()
    dt = time.monotonic() - t
    _log(f"[final] {pos / rate:.1f} s of IQ in {dt:.1f} s ({pos / rate / max(dt, 1e-9):.1f}x real time) | "
         f"{rx.status_line()}")
    _report(rx, args, printer)
    return 0


# -------------------------------------------------------------- selftest
def cmd_selftest(args) -> int:
    fs, dur = 576e3, args.duration
    cell = sim.CellConfig()
    rng = np.random.default_rng(args.seed)
    methods = ["coherent", "differential"] if args.demod == "both" else [args.demod]

    def run(snr, cfo, ppm, method):
        x = sim.wideband([(100e3, cell, 0.0)], fs, dur, snr_db=snr, ppm=ppm, seed=args.seed)
        x = (x * np.exp(2j * np.pi * cfo * np.arange(len(x)) / fs)).astype(np.complex64)
        rx = MultiReceiver(fs, 0.0, [100e3], method=method)
        t = time.monotonic()
        for i in range(0, len(x), 1 << 16):
            rx.feed(x[i : i + (1 << 16)])
        return rx.channels[0], time.monotonic() - t

    ref, _ = run(40, 0, 0, methods[0])
    n_ref = ref.lmac.stats["blk_ok"] + ref.lmac.stats["bsch_ok"]
    print(f"synthetic main carrier, {dur:.0f} s per point, random CFO (+-8 kHz) and clock error (+-30 ppm)")
    print(f"{'Es/N0 dB':>8} {'CFO Hz':>7} {'ppm':>6} {'MER dB':>7}" +
          "".join(f" {m + ' blocks':>22} {'speed':>6}" for m in methods))
    for snr in args.snr:
        cfo, ppm = rng.uniform(-8e3, 8e3), rng.uniform(-30, 30)
        line, mer = "", float("nan")
        for m in methods:
            ch, dt = run(snr, cfo, ppm, m)
            st = ch.lmac.stats
            ok = st["blk_ok"] + st["bsch_ok"]
            mer = ch.stats.mer_db if ch.stats.mer_db is not None else mer
            line += f" {100 * ok / n_ref:12.1f} % ({ok:>4})   {dur / dt:5.1f}x"
        print(f"{snr:8.1f} {cfo:7.0f} {ppm:6.1f} {mer:7.1f}{line}")
    return 0


# ------------------------------------------------------------------ main
def _add_radio(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("HackRF")
    g.add_argument("--lna", type=int, default=32, help="LNA (IF) gain 0-40 dB, step 8 (default 32)")
    g.add_argument("--vga", type=int, default=30, help="VGA (baseband) gain 0-62 dB, step 2 (default 30)")
    g.add_argument("--amp", action="store_true", help="enable the +14 dB RF amplifier")
    g.add_argument("--ppm", type=float, default=0.0, help="HackRF clock error correction in ppm")
    g.add_argument("--serial", help="open the HackRF with this serial number")


def _add_demod(p: argparse.ArgumentParser, cell: bool = True) -> None:
    p.add_argument("--demod", choices=["coherent", "differential"], default="coherent",
                   help="symbol detector: coherent (BCJR, default, ~2 dB better) or differential")
    if cell:
        p.add_argument("--cell", action="append", metavar="MCC:MNC:CC",
                       help="known colour code (as printed by scan); lets carriers without a BSCH "
                            "(secondary carriers) be decoded. Repeatable.")


def _report(rx: MultiReceiver, args, printer: Printer, reset: bool = False) -> None:
    """Print the activity report (text on stderr, plus one JSON line per channel with --json)."""
    if args.report is None:
        return
    for ch, rep, text in rx.reports(reset=reset):
        _log(text)
        if args.json:
            print(json.dumps(dict(t=round(time.time(), 3), freq=ch.freq_hz, report=rep)), file=printer.stream,
                  flush=True)


def _add_output(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("output")
    g.add_argument("--report", nargs="?", type=float, const=0.0, metavar="S",
                   help="print a cell activity report (slot occupancy, calls, signalling load) at the end, "
                        "and every S seconds if S is given")
    g.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v: also broadcast and null PDUs, -vv: also AACH")
    g.add_argument("--json", action="store_true", help="print one JSON object per PDU")
    g.add_argument("--gsmtap", nargs="?", const="127.0.0.1:4729", metavar="HOST[:PORT]",
                   help="send decoded blocks as GSMTAP (Wireshark) over UDP")
    g.add_argument("--pcap", metavar="FILE", help="write decoded blocks as GSMTAP into a PCAP file")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tetrascan", description="TETRA scanner and downlink decoder for HackRF One")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="find TETRA carriers in a band and identify the cells")
    p.add_argument("--start", type=float, default=380.0, help="start frequency, MHz (default 380)")
    p.add_argument("--stop", type=float, default=470.0, help="stop frequency, MHz (default 470)")
    p.add_argument("--threshold", type=float, default=10.0, help="detection threshold over noise, dB (default 10)")
    p.add_argument("--survey-dwell", type=float, default=0.04, help="seconds per 7.5 MHz survey step")
    p.add_argument("--dwell", type=float, default=1.5, help="seconds of IQ decoded per verify window (default 1.5)")
    p.add_argument("--no-verify", action="store_true", help="only run the power survey")
    p.add_argument("--all", action="store_true", help="verify and list all carriers, not only TETRA-like ones")
    p.add_argument("-v", "--verbose", action="store_true", help="log survey / verify progress")
    p.add_argument("--json", action="store_true", help="print one JSON object per carrier")
    _add_radio(p)
    _add_demod(p, cell=False)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("live", help="decode one or more carriers live")
    p.add_argument("-f", "--freq", action="append", required=True, metavar="MHZ",
                   help="carrier frequency in MHz (repeat for more carriers within 1.6 MHz)")
    p.add_argument("--status", type=float, default=10.0, metavar="S", help="status line every S seconds (0 = off)")
    p.add_argument("-d", "--duration", type=float, help="stop after this many seconds")
    _add_radio(p)
    _add_output(p)
    _add_demod(p)
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("record", help="record raw IQ (cs8, 2.304 Msps) with a .json sidecar")
    p.add_argument("-f", "--freq", action="append", metavar="MHZ", help="carrier(s) to cover")
    p.add_argument("--center", type=float, help="LO frequency in MHz (instead of -f)")
    p.add_argument("-d", "--duration", type=float, default=10.0, help="seconds (default 10)")
    p.add_argument("-o", "--output", required=True, help="output .cs8 file")
    _add_radio(p)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("file", help="decode a recording (cs8 / cu8 / cf32)")
    p.add_argument("path")
    p.add_argument("-f", "--freq", action="append", metavar="MHZ", help="carrier(s); default: auto-detect")
    p.add_argument("--center", type=float, help="centre frequency in MHz (default: from sidecar)")
    p.add_argument("--rate", type=float, help="sample rate in Msps (default: from sidecar)")
    p.add_argument("--format", choices=["cs8", "cu8", "cf32"], help="sample format (default: sidecar / extension)")
    p.add_argument("--threshold", type=float, default=10.0, help="auto-detect threshold, dB")
    _add_output(p)
    _add_demod(p)
    p.set_defaults(func=cmd_file)

    p = sub.add_parser("selftest", help="synthetic benchmark: decoded blocks vs Es/N0")
    p.add_argument("--snr", type=float, nargs="+", default=[4, 6, 8, 10, 12, 15, 20, 30])
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--demod", choices=["coherent", "differential", "both"], default="both",
                   help="symbol detector(s) to benchmark (default both)")
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
