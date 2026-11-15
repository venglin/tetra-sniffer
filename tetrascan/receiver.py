"""Multi-channel receiver: wideband IQ in, decoded TETRA signalling out."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import gsmtap
from .activity import Activity, format_report
from .demod import Demodulator
from .dsp import CHANNEL_RATE, ChannelFilter
from .lower_mac import Block, CellRegistry, LowerMac, UpperMac
from .mac import describe, mhz


@dataclass
class ChannelStats:
    bursts: int = 0
    good_bursts: int = 0
    mer_sum: float = 0.0
    level_sum: float = 0.0
    kinds: dict = field(default_factory=dict)

    def add(self, b) -> None:
        self.bursts += 1
        self.kinds[b.kind] = self.kinds.get(b.kind, 0) + 1
        if b.kind is not None:
            self.good_bursts += 1
            self.mer_sum += b.mer_db
            self.level_sum += b.level_db

    @property
    def mer_db(self) -> float | None:
        return self.mer_sum / self.good_bursts if self.good_bursts else None

    @property
    def level_db(self) -> float | None:
        return self.level_sum / self.good_bursts if self.good_bursts else None


@dataclass
class ChannelReceiver:
    freq_hz: float
    offset_hz: float  # channel frequency minus capture centre
    fs: float
    method: str = "coherent"
    registry: CellRegistry | None = None
    filt: ChannelFilter = field(init=False)
    demod: Demodulator = field(init=False)
    lmac: LowerMac = field(init=False)
    umac: UpperMac = field(init=False)
    stats: ChannelStats = field(default_factory=ChannelStats)
    activity: Activity = field(default_factory=Activity)
    sync: dict | None = None
    sysinfo: dict | None = None

    def __post_init__(self):
        self._q_last = (0, 0, 0.0, 0.0, 0, 0)
        self.filt = ChannelFilter(self.fs, self.offset_hz)
        self.demod = Demodulator(method=self.method)
        self.lmac = LowerMac(self.registry, mhz(self.freq_hz))
        self.umac = UpperMac()

    def feed(self, iq: np.ndarray) -> list[tuple[Block, list[dict]]]:
        bursts = self.demod.process(self.filt.process(iq))
        for b in bursts:
            self.stats.add(b)
        out = []
        for blk in self.lmac.process(bursts):
            pdus = self.umac.process(blk)
            for d in pdus:
                if d.get("pdu") == "SYNC":
                    self.sync = d
                elif d.get("pdu") == "SYSINFO":
                    self.sysinfo = d
            self.activity.add(blk, pdus)
            out.append((blk, pdus))
        return out

    def period_quality(self, reset: bool = True) -> dict:
        """Reception quality since the previous call: bursts, MER, level and CRC success rate."""
        st, lm = self.stats, self.lmac.stats
        now = (st.bursts, st.good_bursts, st.mer_sum, st.level_sum, lm["blk_ok"], lm["blk_bad"])
        b, g, mer, lvl, ok, bad = (x - y for x, y in zip(now, self._q_last))
        if reset:
            self._q_last = now
        return dict(locked=self.locked, bursts=b, mer_db=round(mer / g, 1) if g else None,
                    level_db=round(lvl / g, 1) if g else None,
                    crc_pct=round(100 * ok / (ok + bad), 1) if ok + bad else None)

    @property
    def locked(self) -> bool:
        return self.demod.state == "lock"

    def summary(self) -> dict:
        s = dict(freq_hz=self.freq_hz, locked=self.locked, bursts=self.stats.bursts,
                 good_bursts=self.stats.good_bursts, mer_db=self.stats.mer_db, level_db=self.stats.level_db,
                 cfo_hz=self.demod.cfo, bsch_ok=self.lmac.stats["bsch_ok"], blk_ok=self.lmac.stats["blk_ok"],
                 blk_bad=self.lmac.stats["blk_bad"], burst_kinds={str(k): v for k, v in self.stats.kinds.items()})
        if self.lmac.cell:
            c = self.lmac.cell
            s.update(mcc=c["mcc"], mnc=c["mnc"], cc=c["cc"], code_source=self.lmac.code_source,
                     code_from=c["source"])
        if self.sysinfo:
            s.update(la=self.sysinfo["la"], main_carrier_hz=self.sysinfo["dl_hz"], services=self.sysinfo["services"])
        return s


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, dict):
            v = _jsonable(v)
        out[k] = v
    return out


class Printer:
    """Formats decoded blocks. Verbosity 0 shows cell changes and dedicated signalling,
    1 adds every broadcast / null PDU, 2 adds AACH."""

    QUIET_PDUS = {"ACCESS-DEFINE", "SYSINFO-DA"}
    # SYSINFO fields that identify a configuration change (the hyperframe number, the CCK id
    # and the optional field change all the time and are ignored).
    SYSINFO_KEY = ("main_carrier", "band", "offset", "duplex", "reverse", "num_sec_cch", "ms_txpwr_max_dbm",
                   "rxlev_access_min_dbm", "access_parameter_dbm", "radio_dl_timeout", "la", "subscriber_class",
                   "bs_service_details")

    def __init__(self, verbose: int = 0, as_json: bool = False, t0: float | None = None, stream=None,
                 sinks: list | None = None, quiet: bool = False):
        self.quiet = quiet
        self.verbose = verbose
        self.as_json = as_json
        self.t0 = t0
        self.stream = stream or sys.stdout
        self.sinks = sinks or []
        self._last_cell: dict[float, tuple] = {}
        self._last_sysinfo: dict[float, tuple] = {}

    def timestamp(self, t_rel: float) -> str:
        if self.t0 is None:
            return f"{t_rel:10.3f}"
        t = self.t0 + t_rel
        return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"

    def emit(self, ch: ChannelReceiver, blk: Block, pdus: list[dict]) -> None:
        t_rel = blk.burst.start / CHANNEL_RATE
        if blk.ok and blk.bits is not None and blk.lchan in gsmtap.SUBTYPES and self.sinks:
            fn = blk.time.frame_number() if blk.time else 0
            pkt = gsmtap.gsmtap_packet(blk.lchan, blk.bits, tn=blk.time.tn if blk.time else 1, frame_number=fn,
                                       arfcn=int(round((ch.freq_hz % 100e6) / 25e3)),
                                       signal_dbm=round(blk.burst.level_db), snr_db=round(blk.burst.mer_db),
                                       sub_slot=max(0, blk.half - 1))
            for s in self.sinks:
                s.send(pkt, (self.t0 or 0) + t_rel)
        if self.quiet:
            return
        for d in pdus:
            if not self._wanted(ch, blk, d):
                continue
            if self.as_json:
                rec = dict(t=round((self.t0 or 0) + t_rel, 3), freq=ch.freq_hz, lchan=blk.lchan,
                           time=str(blk.time) if blk.time else None, **_jsonable(d))
                print(json.dumps(rec, default=str), file=self.stream, flush=True)
            else:
                tdma = str(blk.time) if blk.time else "TN? FN?? MN??"
                band = ch.sysinfo["band"] if ch.sysinfo else None
                print(f"{self.timestamp(t_rel)} {mhz(ch.freq_hz)} {tdma} {blk.lchan:<6s} {describe(d, band)}",
                      file=self.stream, flush=True)

    def _wanted(self, ch: ChannelReceiver, blk: Block, d: dict) -> bool:
        p = d.get("pdu")
        if p == "ACCESS-ASSIGN":
            return self.verbose >= 2
        if self.verbose >= 1:
            return True
        if p == "SYNC":
            key = (d["mcc"], d["mnc"], d["cc"])
            if self._last_cell.get(ch.freq_hz) == key:
                return False
            self._last_cell[ch.freq_hz] = key
            return True
        if p == "SYSINFO":
            key = tuple(d.get(k) for k in self.SYSINFO_KEY)
            if self._last_sysinfo.get(ch.freq_hz) == key:
                return False
            self._last_sysinfo[ch.freq_hz] = key
            return True
        if p in self.QUIET_PDUS or (p == "MAC-RESOURCE" and d.get("null")):
            return False
        return True


class MultiReceiver:
    def __init__(self, fs: float, center_hz: float, freqs: list[float], printer: Printer | None = None,
                 method: str = "coherent", known_cells: list[tuple[int, int, int]] | None = None):
        self.fs = fs
        self.center_hz = center_hz
        self.registry = CellRegistry()
        for mcc, mnc, cc in known_cells or []:
            self.registry.add(mcc, mnc, cc, "manual")
        self.channels = [ChannelReceiver(f, f - center_hz, fs, method, self.registry) for f in freqs]
        self.printer = printer
        self.samples = 0

    def feed(self, iq: np.ndarray) -> None:
        self.samples += len(iq)
        events = [(blk.burst.start, i, ch, blk, pdus)
                  for i, ch in enumerate(self.channels) for blk, pdus in ch.feed(iq)]
        if self.printer:
            for _, _, ch, blk, pdus in sorted(events, key=lambda e: (e[0], e[1])):
                self.printer.emit(ch, blk, pdus)

    def reports(self, reset: bool = False) -> list[tuple[ChannelReceiver, dict, str]]:
        """Activity report of every channel: (channel, report dict, formatted text)."""
        out = []
        for ch in self.channels:
            rep = ch.activity.report(reset=reset)
            rep["quality"] = ch.period_quality(reset=reset)
            text = format_report(mhz(ch.freq_hz), rep, ch.lmac.cell, ch.sysinfo["la"] if ch.sysinfo else None)
            out.append((ch, rep, text))
        return out

    def status_line(self) -> str:
        parts = []
        for ch in self.channels:
            s = ch.summary()
            mer = f"{s['mer_db']:.1f}" if s["mer_db"] is not None else "-"
            lvl = f"{s['level_db']:.0f}" if s["level_db"] is not None else "-"
            cfo = f"{s['cfo_hz']:+.0f}" if s["cfo_hz"] is not None else "-"
            tot = s["blk_ok"] + s["blk_bad"]
            crc = f"{100 * s['blk_ok'] / tot:.0f}%" if tot else "-"
            parts.append(f"{mhz(ch.freq_hz)} {'LOCK' if s['locked'] else 'srch'} MER {mer} dB lvl {lvl} dBFS "
                         f"cfo {cfo} Hz crc {crc}")
        return " | ".join(parts)
