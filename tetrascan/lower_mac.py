"""Lower MAC: burst -> logical channel blocks, plus the per-carrier cell state.

Routing of downlink bursts (9.4.4, 9.5):

* synchronisation burst (training y): BSCH + AACH + one half-slot block (BNCH or SCH/HD)
* normal burst, training p: AACH + two half-slot blocks (SCH/HD, BNCH or STCH)
* normal burst, training n: AACH + one full-slot block (SCH/F, or TCH on traffic slots)

The BSCH is scrambled with the all-zero colour code and carries MCC, MNC and the
colour code, which give the scrambling code of every other block on the carrier.
Until a BSCH is decoded nothing else can be descrambled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import burst as B
from . import coding as C
from .demod import Burst
from .mac import parse_aach, parse_mac_block, parse_sync, parse_tm_sdu


@dataclass
class TdmaTime:
    tn: int  # 1..4
    fn: int  # 1..18
    mn: int  # 1..60

    def advance(self, slots: int) -> "TdmaTime":
        x = ((self.mn - 1) * 18 + (self.fn - 1)) * 4 + (self.tn - 1) + slots
        x %= 60 * 18 * 4
        return TdmaTime(tn=x % 4 + 1, fn=(x // 4) % 18 + 1, mn=x // 72 + 1)

    def frame_number(self) -> int:
        return (self.mn - 1) * 18 + (self.fn - 1)

    def __str__(self) -> str:
        return f"TN{self.tn} FN{self.fn:02d} MN{self.mn:02d}"


@dataclass
class Block:
    """A decoded logical-channel block."""

    lchan: str  # BSCH, BNCH, SCH/HD, SCH/F, STCH, AACH, TCH
    ok: bool  # CRC (or RM code) check passed; TCH is never checked
    bits: np.ndarray | None  # type-1 bits
    burst: Burst
    time: TdmaTime | None
    half: int = 0  # 1 or 2 for half-slot blocks, 0 otherwise
    pdus: list[dict] = field(default_factory=list)


class CellRegistry:
    """Scrambling codes known within one capture, shared by all channels.

    Carriers of one cell use the same colour code, but only the main carrier is guaranteed
    to send synchronisation bursts. Codes come from decoded BSCHs (with the burst timing,
    so time-aligned carriers can inherit the TDMA time) or are given by the user.
    """

    SLOT_SAMPLES = 2040

    def __init__(self):
        self.codes: dict[int, dict] = {}

    def add(self, mcc: int, mnc: int, cc: int, source: str, start: float | None = None,
            time: TdmaTime | None = None) -> int:
        init = C.scrambling_init(mcc, mnc, cc)
        e = self.codes.setdefault(init, dict(mcc=mcc, mnc=mnc, cc=cc, source=source))
        if start is not None:
            e.update(source=source, start=start, time=time)
        return init

    def time_for(self, init: int, start: float) -> TdmaTime | None:
        """TDMA time of a burst starting at ``start`` if it is slot-aligned with the code's source."""
        e = self.codes.get(init)
        if not e or e.get("start") is None:
            return None
        n = (start - e["start"]) / self.SLOT_SAMPLES
        if abs(n - round(n)) > 0.02:
            return None
        return e["time"].advance(int(round(n)))


class LowerMac:
    AACH_FULL_SCORE = 0.7  # bursts scoring lower get a stricter AACH acceptance

    def __init__(self, registry: CellRegistry | None = None, name: str = ""):
        self.registry = registry
        self.name = name
        self.scramb: int | None = None
        self.cell: dict | None = None  # mcc / mnc / cc (+ source) of the scrambling code in use
        self.code_source: str | None = None  # "bsch", "shared" or "manual"
        self._ref: tuple[int, int, TdmaTime] | None = None  # (epoch, seq, time)
        self.stats = dict(bursts=0, bsch_ok=0, bsch_bad=0, blk_ok=0, blk_bad=0, tch=0, unknown=0)

    def _time(self, b: Burst) -> TdmaTime | None:
        if self._ref is None or self._ref[0] != b.epoch:
            # After a new lock, a slot-aligned BSCH seen earlier (on this or another
            # carrier) still gives the TDMA time; otherwise wait for the next BSCH.
            t = self.registry.time_for(self.scramb, b.start) if self.registry and self.scramb else None
            if t is None:
                return None
            self._ref = (b.epoch, b.seq, t)
        return self._ref[2].advance(b.seq - self._ref[1])

    def process(self, bursts: list[Burst]) -> list[Block]:
        out: list[Block] = []
        if not bursts:
            return out
        self.stats["bursts"] += len(bursts)
        sb = [b for b in bursts if b.kind == B.SDB]
        if sb:
            soft = np.stack([b.soft[B.SB_BSCH] for b in sb])
            bits, ok = C.decode_blocks(soft, C.BSCH, C.BSCH_SCRAMBLING_INIT)
            for b, bb, k in zip(sb, bits, ok):
                if k:
                    self.stats["bsch_ok"] += 1
                    s = parse_sync(bb)
                    self.scramb = C.scrambling_init(s["mcc"], s["mnc"], s["cc"])
                    self.cell = dict(mcc=s["mcc"], mnc=s["mnc"], cc=s["cc"], source=self.name)
                    self.code_source = "bsch"
                    t = TdmaTime(s["tn"], s["fn"], s["mn"])
                    self._ref = (b.epoch, b.seq, t)
                    if self.registry is not None:
                        self.registry.add(s["mcc"], s["mnc"], s["cc"], self.name, b.start, t)
                    out.append(Block("BSCH", True, bb, b, t, pdus=[s]))
                else:
                    self.stats["bsch_bad"] += 1
        self.stats["unknown"] += sum(1 for b in bursts if b.kind is None)
        if self.scramb is None:
            adopted = self._try_known_codes([b for b in bursts if b.kind is not None])
            if adopted is None:
                return out
            out.append(adopted)
        good = [b for b in bursts if b.kind is not None]
        if not good:
            return out

        # AACH of every burst.
        aach_soft = np.stack([B.bbk_bits(b.soft, b.kind) for b in good])
        info, nerr = C.rm3014_decode(C.descramble_soft(aach_soft, self.scramb))
        aach: dict[int, dict] = {}
        for b, i, e in zip(good, info, nerr):
            t = self._time(b)
            # RM(30,14) has minimum distance 8: up to 3 errors are correctable, but a random
            # word lies within distance 3 of some codeword 7 % of the time. Weak bursts
            # (possibly noise) must therefore decode almost exactly.
            if e <= (3 if b.score >= self.AACH_FULL_SCORE else 1):
                a = parse_aach(i, t.fn if t else None)
                aach[id(b)] = a
                out.append(Block("AACH", True, i, b, t, pdus=[a]))

        # Signalling blocks.
        half, half_meta, full, full_meta = [], [], [], []
        for b in good:
            t = self._time(b)
            a = aach.get(id(b))
            traffic = bool(a and a["traffic"])
            if b.kind == B.SDB:
                half.append(b.soft[B.SB_BLK2])
                half_meta.append((b, t, 2, False))
            elif b.kind == B.NDB_HALF:
                half.append(b.soft[B.NDB_BLK1])
                half_meta.append((b, t, 1, traffic))
                half.append(b.soft[B.NDB_BLK2])
                half_meta.append((b, t, 2, traffic))
            elif traffic:
                self.stats["tch"] += 1
                out.append(Block("TCH", False, None, b, t))
            else:
                full.append(np.concatenate([b.soft[B.NDB_BLK1], b.soft[B.NDB_BLK2]]))
                full_meta.append((b, t))
        if half:
            bits, ok = C.decode_blocks(np.stack(half), C.SCH_HD, self.scramb)
            for (b, t, h, traffic), bb, k in zip(half_meta, bits, ok):
                self._count(k)
                if not k:
                    continue
                lchan = "STCH" if traffic else "SCH/HD"
                if not traffic and bb[0] == 1 and not bb[1] and not bb[2] and not bb[3]:
                    lchan = "BNCH"  # MAC broadcast PDU of type SYSINFO
                out.append(Block(lchan, True, bb, b, t, half=h))
        if full:
            bits, ok = C.decode_blocks(np.stack(full), C.SCH_F, self.scramb)
            for (b, t), bb, k in zip(full_meta, bits, ok):
                self._count(k)
                if k:
                    out.append(Block("SCH/F", True, bb, b, t))
                else:
                    self.stats["tch"] += 1
                    out.append(Block("TCH", False, None, b, t))
        out.sort(key=lambda blk: (blk.burst.start, blk.half, blk.lchan != "BSCH"))
        return out

    def _try_known_codes(self, good: list[Burst]) -> Block | None:
        """Without a BSCH of our own, test the scrambling codes known from other carriers."""
        if self.registry is None or not self.registry.codes or not good:
            return None
        test = good[:8]
        half = [b.soft[B.SB_BLK2] if b.kind == B.SDB else b.soft[B.NDB_BLK1]
                for b in test if b.kind != B.NDB_FULL]
        full = [np.concatenate([b.soft[B.NDB_BLK1], b.soft[B.NDB_BLK2]]) for b in test if b.kind == B.NDB_FULL]
        aach = np.stack([B.bbk_bits(b.soft, b.kind) for b in test])
        for init, e in list(self.registry.codes.items()):
            ok = False
            if half:
                ok = bool(C.decode_blocks(np.stack(half), C.SCH_HD, init)[1].any())
            if not ok and full:
                ok = bool(C.decode_blocks(np.stack(full), C.SCH_F, init)[1].any())
            if not ok and len(test) >= 8:
                _, nerr = C.rm3014_decode(C.descramble_soft(aach, init))
                ok = bool(np.mean(nerr <= 3) >= 0.9)
            if not ok:
                continue
            self.scramb = init
            self.cell = dict(mcc=e["mcc"], mnc=e["mnc"], cc=e["cc"], source=e["source"])
            self.code_source = "manual" if e["source"] == "manual" else "shared"
            b0 = good[0]
            t = self.registry.time_for(init, b0.start)
            if t is not None:
                self._ref = (b0.epoch, b0.seq, t)
            info = dict(pdu="CELL-CODE", mcc=e["mcc"], mnc=e["mnc"], cc=e["cc"], source=e["source"],
                        time_inherited=t is not None)
            return Block("CELL", True, None, b0, t, pdus=[info])
        return None

    def _count(self, ok: bool) -> None:
        self.stats["blk_ok" if ok else "blk_bad"] += 1


class UpperMac:
    """Parses MAC PDUs in decoded blocks and reassembles fragmented TM-SDUs per timeslot."""

    def __init__(self):
        self._frag: dict[int, tuple[dict, list[np.ndarray]]] = {}

    def process(self, blk: Block) -> list[dict]:
        if blk.lchan in ("BSCH", "AACH", "CELL") or blk.bits is None:
            return blk.pdus
        key = blk.time.tn if blk.time else 0
        head = self._frag.get(key)
        pdus = parse_mac_block(blk.bits, blk.lchan, frag_enc=bool(head[0]["enc"]) if head else None)
        for d in pdus:
            p = d["pdu"]
            if p == "MAC-RESOURCE" and not d.get("null"):
                if d.get("frag_start"):
                    self._frag[key] = (d, [d.get("sdu", np.zeros(0, np.uint8))])
                    d["tm"] = None
                elif "sdu" in d:
                    d["tm"] = parse_tm_sdu(d["sdu"], encrypted=bool(d["enc"]))
            elif p == "MAC-FRAG" and key in self._frag:
                self._frag[key][1].append(d["sdu"])
            elif p == "MAC-END" and key in self._frag:
                head, parts = self._frag.pop(key)
                parts.append(d["sdu"])
                d["reassembled_from"] = head.get("ssi")
                d["tm"] = parse_tm_sdu(np.concatenate(parts), encrypted=bool(head["enc"]))
        blk.pdus = pdus
        return pdus
