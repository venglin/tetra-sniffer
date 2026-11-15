"""Cell activity statistics built from the AACH and the MAC layer, without decrypting anything.

The ACCESS-ASSIGN PDU on the AACH of every downlink burst says what that slot carries:
common or assigned control, nothing, or a traffic channel with a 6-bit usage marker. Its
uplink part says the same for the matching uplink slot. Counting it per timeslot gives the
occupancy of each slot; following the usage marker on a slot gives the traffic sessions
(each call, or each over when the network uses transmission trunking) and their duration.
MAC-RESOURCE headers add the signalling load: how many PDUs, how many distinct addresses,
how much of it is encrypted, random-access acknowledgements and channel allocations. Only
counts are reported, never addresses or contents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from . import burst as B
from .dsp import CHANNEL_RATE
from .lower_mac import Block

SLOT_S = 255 / 18000  # one timeslot
DL_CATEGORIES = ("traffic", "assigned control", "common control", "unallocated")
UL_CATEGORIES = ("traffic", "control", "unallocated")


@dataclass
class Session:
    tn: int
    um: int
    start: float
    last: float

    @property
    def duration(self) -> float:
        return self.last - self.start + 4 * SLOT_S


class Activity:
    """Accumulates activity statistics of one carrier over a reporting period."""

    def __init__(self, gap_s: float = 3.0):
        self.gap_s = gap_s  # a traffic session survives AACH gaps this long
        self.open: dict[int, Session] = {}
        self.reset()

    def reset(self) -> None:
        self.t_first: float | None = None
        self.t_last: float | None = None
        self.dl = {tn: Counter() for tn in range(1, 5)}
        self.ul = {tn: Counter() for tn in range(1, 5)}
        self.bursts = {tn: Counter() for tn in range(1, 5)}
        self.sessions: list[Session] = []
        self.mac = Counter()
        self.addresses: set[int] = set()
        self.l3 = Counter()

    # ------------------------------------------------------------------ input
    def add(self, blk: Block, pdus: list[dict]) -> None:
        if blk.burst is None:
            return
        t = blk.burst.start / CHANNEL_RATE
        if self.t_first is None:
            self.t_first = t
        self.t_last = t
        if blk.lchan == "AACH" and pdus and blk.time is not None:
            self._aach(blk, pdus[0], t)
            return
        for d in pdus:
            self._mac(d)

    def _aach(self, blk: Block, a: dict, t: float) -> None:
        tn = blk.time.tn
        self.bursts[tn]["SB" if blk.burst.kind == B.SDB else "NDB"] += 1
        if blk.time.fn == 18:
            return  # frame 18 is always control; it neither counts nor ends a call
        if a["traffic"]:
            self.dl[tn]["traffic"] += 1
            s = self.open.get(tn)
            if s and s.um == a["dl_usage"] and t - s.last <= self.gap_s:
                s.last = t
            else:
                self._close(tn)
                self.open[tn] = Session(tn, a["dl_usage"], t, t)
        else:
            self.dl[tn][{0: "unallocated", 1: "assigned control"}.get(a["dl_usage"], "common control")] += 1
            self._close(tn)
        if a["hdr"] == 3:
            self.ul[tn]["traffic" if a["field2"] else "unallocated"] += 1
        else:
            self.ul[tn]["control"] += 1

    def _close(self, tn: int) -> None:
        s = self.open.pop(tn, None)
        if s:
            self.sessions.append(s)

    def _mac(self, d: dict) -> None:
        p = d.get("pdu")
        if p == "MAC-RESOURCE":
            if d.get("null"):
                self.mac["null"] += 1
                return
            self.mac["resource"] += 1
            self.mac["encrypted" if d.get("enc") else "clear"] += 1
            if "ssi" in d:
                self.addresses.add(d["ssi"])
            if d.get("random_access_ack"):
                self.mac["random_access_ack"] += 1
            if d.get("frag_start"):
                self.mac["fragmented"] += 1
        elif p == "MAC-END":
            pass
        else:
            return
        if "chan_alloc" in d:
            self.mac["chan_alloc_clear"] += 1
        elif d.get("chan_alloc_encrypted") or d.get("chan_alloc_unknown"):
            self.mac["chan_alloc_encrypted"] += 1
        tm = d.get("tm")
        if tm and not tm.get("encrypted") and tm.get("l3"):
            self.l3[tm["l3"]] += 1

    # ----------------------------------------------------------------- output
    def report(self, reset: bool = False) -> dict:
        period = (self.t_last - self.t_first + SLOT_S) if self.t_first is not None else 0.0
        slots = {}
        for tn in range(1, 5):
            n_dl, n_ul, n_b = sum(self.dl[tn].values()), sum(self.ul[tn].values()), sum(self.bursts[tn].values())
            if not n_b:
                continue
            slots[tn] = dict(
                dl={c: round(100 * self.dl[tn][c] / n_dl, 1) for c in DL_CATEGORIES if self.dl[tn][c]} if n_dl else {},
                ul={c: round(100 * self.ul[tn][c] / n_ul, 1) for c in UL_CATEGORIES if self.ul[tn][c]} if n_ul else {},
                sync_bursts=round(100 * self.bursts[tn]["SB"] / n_b, 1),
                sessions=sum(1 for s in self.sessions if s.tn == tn),
            )
        durations = sorted(s.duration for s in self.sessions)
        sess = dict(count=len(durations), ongoing=len(self.open),
                     usage_markers=len({s.um for s in self.sessions} | {s.um for s in self.open.values()}))
        if durations:
            sess.update(total_s=round(sum(durations), 1), median_s=round(durations[len(durations) // 2], 1),
                         max_s=round(durations[-1], 1))
        m = self.mac
        sig = dict(pdus=m["resource"], null=m["null"], addresses=len(self.addresses), encrypted=m["encrypted"],
                   clear=m["clear"], fragmented=m["fragmented"], random_access_acks=m["random_access_ack"],
                   chan_allocs=m["chan_alloc_clear"] + m["chan_alloc_encrypted"],
                   chan_allocs_encrypted=m["chan_alloc_encrypted"])
        out = dict(period_s=round(period, 1), slots=slots, sessions=sess, signalling=sig, clear_l3=dict(self.l3))
        if reset:
            self.reset()
        return out


def _pct(d: dict) -> str:
    return ", ".join(f"{k} {v:.0f}%" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "-"


def format_report(freq_mhz: str, rep: dict, cell: dict | None = None, la: int | None = None) -> str:
    head = f"[report] {freq_mhz} MHz"
    if cell:
        head += f"  MCC {cell['mcc']} MNC {cell['mnc']} CC {cell['cc']}"
    if la is not None:
        head += f" LA {la}"
    lines = [head + f"  period {rep['period_s']:.1f} s"]
    for tn, s in rep["slots"].items():
        calls = f", {s['sessions']} sessions" if s["sessions"] else ""
        lines.append(f"  TN{tn}  DL {_pct(s['dl'])}{calls} | UL {_pct(s['ul'])} | sync bursts {s['sync_bursts']:.0f}%")
    c = rep["sessions"]
    line = f"  traffic sessions: {c['count']} ended, {c['ongoing']} ongoing, {c['usage_markers']} usage markers"
    if c["count"]:
        line += f"; duration total {c['total_s']:.1f} s, median {c['median_s']:.1f} s, max {c['max_s']:.1f} s"
    lines.append(line)
    g = rep["signalling"]
    enc = f"{100 * g['encrypted'] / g['pdus']:.0f}% encrypted" if g["pdus"] else "none"
    lines.append(f"  signalling: {g['pdus']} PDUs to {g['addresses']} addresses ({enc}), {g['null']} null, "
                 f"{g['fragmented']} fragmented, {g['random_access_acks']} random-access acks, "
                 f"{g['chan_allocs']} channel allocations ({g['chan_allocs_encrypted']} encrypted)")
    if rep["clear_l3"]:
        lines.append("  clear L3: " + ", ".join(f"{k} {v}" for k, v in sorted(rep["clear_l3"].items(),
                                                                               key=lambda kv: -kv[1])))
    return "\n".join(lines)
