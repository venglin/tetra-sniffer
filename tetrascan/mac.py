"""TETRA MAC / LLC / MLE PDU parsing for the downlink (EN 300 392-2 clauses 21, 22, 18, 16, 14).

Every parser returns a plain ``dict``; ``describe()`` turns it into a one-line summary.
Only signalling headers are decoded. Encrypted TM-SDUs are reported as such, and the
contents of short data (SDS) messages are deliberately not rendered.
"""

from __future__ import annotations

import numpy as np


class EndOfData(Exception):
    pass


class BitReader:
    def __init__(self, bits, pos: int = 0, end: int | None = None):
        self.bits = np.asarray(bits, dtype=np.uint8)
        self.pos = pos
        self.end = len(self.bits) if end is None else end

    def u(self, n: int) -> int:
        if self.pos + n > self.end:
            raise EndOfData(f"need {n} bits at {self.pos}, have {self.end - self.pos}")
        v = 0
        for b in self.bits[self.pos : self.pos + n]:
            v = (v << 1) | int(b)
        self.pos += n
        return v

    @property
    def left(self) -> int:
        return self.end - self.pos

    def rest(self) -> np.ndarray:
        r = self.bits[self.pos : self.end]
        self.pos = self.end
        return r


def bits_to_int(bits) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


# ------------------------------------------------------------- frequencies
CARRIER_OFFSET_HZ = (0, 6250, -6250, 12500)
# Duplex spacing in kHz, [duplex spacing field][frequency band]; None = reserved.
_N = None
DUPLEX_SPACING_KHZ = (
    (_N, 1600, 10000, 10000, 10000, 10000, 10000, _N, _N, _N, _N, _N, _N, _N, _N, _N),
    (_N, 4500, _N, 36000, 7000, _N, _N, _N, 45000, 45000, _N, _N, _N, _N, _N, _N),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (_N, _N, _N, 8000, 8000, _N, _N, _N, 18000, 18000, _N, _N, _N, _N, _N, _N),
    (_N, _N, _N, 18000, 5000, _N, 30000, 30000, _N, 39000, _N, _N, _N, _N, _N, _N),
    (_N, _N, _N, _N, 9500, _N, _N, _N, _N, _N, _N, _N, _N, _N, _N, _N),
    (_N,) * 16,
    (_N,) * 16,
)


def carrier_frequency(band: int, carrier: int, offset: int) -> float:
    """Downlink frequency in Hz from the SYSINFO / channel allocation fields."""
    return band * 100e6 + carrier * 25e3 + CARRIER_OFFSET_HZ[offset & 3]


def uplink_frequency(dl_hz: float, band: int, duplex: int, reverse: int) -> float | None:
    sp = DUPLEX_SPACING_KHZ[duplex & 7][band & 15]
    if sp is None:
        return None
    return dl_hz + sp * 1e3 if reverse else dl_hz - sp * 1e3


def mhz(f: float) -> str:
    return f"{f / 1e6:.4f}"


# ------------------------------------------------------- BSCH: MAC-SYNC
def parse_sync(bits) -> dict:
    """MAC-SYNC (21.4.4.2) + D-MLE-SYNC (18.4.2.1), 60 bits."""
    r = BitReader(bits)
    d = dict(pdu="SYNC")
    d["system_code"] = r.u(4)
    d["cc"] = r.u(6)
    d["tn"] = r.u(2) + 1
    d["fn"] = r.u(5)
    d["mn"] = r.u(6)
    d["sharing_mode"] = r.u(2)
    d["ts_reserved_frames"] = r.u(3)
    d["u_plane_dtx"] = r.u(1)
    d["frame18_ext"] = r.u(1)
    r.u(1)
    d["mcc"] = r.u(10)
    d["mnc"] = r.u(14)
    d["neighbour_cell_broadcast"] = r.u(2)
    d["cell_load"] = r.u(2)
    d["late_entry"] = r.u(1)
    return d


SHARING_MODES = ("continuous", "carrier sharing", "MCCH sharing", "traffic carrier sharing")


# ------------------------------------------------------- AACH: ACCESS-ASSIGN
DL_USAGE = {0: "unallocated", 1: "assigned control", 2: "common control", 3: "reserved"}


def parse_aach(bits14, fn: int | None) -> dict:
    """ACCESS-ASSIGN (21.4.7). ``traffic`` is True when the downlink slot carries a traffic channel."""
    v = bits_to_int(bits14)
    hdr, f1, f2 = v >> 12, (v >> 6) & 0x3F, v & 0x3F
    d = dict(pdu="ACCESS-ASSIGN", hdr=hdr, field1=f1, field2=f2, traffic=False, dl_usage=None)
    if fn == 18 or hdr == 0:
        d["dl"] = "common control"
        d["dl_usage"] = 2
    else:
        d["dl_usage"] = f1
        d["dl"] = DL_USAGE.get(f1, f"traffic UM={f1}")
        d["traffic"] = f1 > 3
        if hdr == 3:
            d["ul"] = "unallocated" if f2 == 0 else f"traffic UM={f2}"
    return d


# ------------------------------------------------------- BNCH: SYSINFO
MS_TXPWR = {1: 15, 2: 20, 3: 25, 4: 30, 5: 35, 6: 40, 7: 45}
BS_SERVICES = (
    (11, "registration"), (10, "deregistration"), (9, "priority cell"), (8, "no minimum mode"),
    (7, "migration"), (6, "system-wide services"), (5, "voice"), (4, "circuit data"),
    (2, "SNDCP/packet data"), (1, "air interface encryption"), (0, "advanced link"),
)


def _parse_sysinfo(r: BitReader) -> dict:
    d = dict(pdu="SYSINFO")
    d["main_carrier"] = r.u(12)
    d["band"] = r.u(4)
    d["offset"] = r.u(2)
    d["duplex"] = r.u(3)
    d["reverse"] = r.u(1)
    d["num_sec_cch"] = r.u(2)
    d["ms_txpwr_max_dbm"] = MS_TXPWR.get(r.u(3))
    d["rxlev_access_min_dbm"] = -125 + 5 * r.u(4)
    d["access_parameter_dbm"] = -53 + 2 * r.u(4)
    d["radio_dl_timeout"] = r.u(4)
    cck = r.u(1)
    v = r.u(16)
    d["cck_id" if cck else "hyperframe"] = v
    d["optional_field"] = r.u(2)
    d["option_value"] = r.u(20)
    # D-MLE-SYSINFO (18.4.2.2)
    d["la"] = r.u(14)
    d["subscriber_class"] = r.u(16)
    svc = r.u(12)
    d["bs_service_details"] = svc
    d["services"] = [name for bit, name in BS_SERVICES if svc >> bit & 1]
    d["dl_hz"] = carrier_frequency(d["band"], d["main_carrier"], d["offset"])
    d["ul_hz"] = uplink_frequency(d["dl_hz"], d["band"], d["duplex"], d["reverse"])
    return d


# ------------------------------------------------------ ACCESS-DEFINE
def _parse_access_define(r: BitReader) -> dict:
    """ACCESS-DEFINE (21.4.4.3): random access parameters for one access code."""
    d = dict(pdu="ACCESS-DEFINE")
    d["assigned"] = r.u(1)  # 0 = common control channel, 1 = assigned control channel
    d["access_code"] = "ABCD"[r.u(2)]
    d["imm"] = r.u(4)
    d["wt"] = r.u(4)
    d["nu"] = r.u(4)
    d["frame_len_factor"] = r.u(1)
    d["ts_pointer"] = r.u(4)
    d["min_priority"] = r.u(3)
    opt = r.u(2)
    if opt == 1:
        d["subscriber_class"] = r.u(16)
    elif opt == 2:
        d["gssi"] = r.u(24)
    return d


# ------------------------------------------------- channel allocation
def _parse_chan_alloc(r: BitReader) -> dict:
    """Channel allocation element (21.5.2)."""
    d = dict(alloc_type=r.u(2), timeslots=r.u(4), ul_dl=r.u(2), clch=r.u(1), cell_change=r.u(1))
    d["carrier"] = r.u(12)
    if r.u(1):
        d["band"], d["offset"], d["duplex"], d["reverse"] = r.u(4), r.u(2), r.u(3), r.u(1)
    mon = r.u(2)
    if mon == 0:
        r.u(2)
    if d["ul_dl"] == 0:  # augmented channel allocation (TEDS)
        r.u(2), r.u(3), r.u(3), r.u(3), r.u(3), r.u(3), r.u(4), r.u(5)
        if r.u(2) == 1:
            r.u(11)
        r.u(4)
        if r.u(1):
            r.u(16)
        if r.u(1):
            r.u(16)
        r.u(1)
    return d


ALLOC_TYPES = ("replace", "additional", "quit+go", "replace+CSS")
UL_DL = ("augmented", "DL", "UL", "UL+DL")


def _describe_chan_alloc(ca: dict, cell_band: int | None = None) -> str:
    ts = "".join(str(i + 1) for i in range(4) if ca["timeslots"] >> (3 - i) & 1) or "-"
    s = f"chan[{ALLOC_TYPES[ca['alloc_type']]} carrier {ca['carrier']}"
    band = ca.get("band", cell_band)
    if band:
        s += f" ({mhz(carrier_frequency(band, ca['carrier'], ca.get('offset', 0)))} MHz)"
    return s + f" ts {ts} {UL_DL[ca['ul_dl']]}]"


# ------------------------------------------------------- MAC PDUs
ADDRESS_TYPES = {1: "SSI", 2: "event", 3: "USSI", 4: "SMI", 5: "SSI+event", 6: "SSI+usage", 7: "SMI+event"}
LEN_2ND_HALF_STOLEN = 0x3E
LEN_START_FRAG = 0x3F


def _strip_fill(bits: np.ndarray) -> np.ndarray:
    """Remove fill bits (a 1 followed by zeros) from the end of a PDU."""
    ones = np.flatnonzero(bits)
    return bits[: ones[-1]] if len(ones) else bits[:0]


def _parse_mac_resource(r: BitReader, start: int, block_end: int) -> dict:
    d = dict(pdu="MAC-RESOURCE")
    fill = r.u(1)
    r.u(1)  # position of grant
    d["enc"] = r.u(2)
    d["random_access_ack"] = r.u(1)
    li = r.u(6)
    d["li"] = li
    at = r.u(3)
    if at == 0:
        d["null"] = True
        d["end"] = start + (li * 8 if 0 < li <= 0x3A else block_end - start)
        return d
    d["addr_type"] = ADDRESS_TYPES[at]
    if at in (1, 3, 4, 5, 6, 7):
        d["ssi"] = r.u(24)
    if at in (2, 5, 7):
        d["event_label"] = r.u(10)
    if at == 6:
        d["usage_marker"] = r.u(6)
    if r.u(1):
        d["power_control"] = r.u(4)
    if r.u(1):
        d["slot_granting"] = r.u(8)
    if r.u(1):
        if d["enc"]:
            d["chan_alloc_encrypted"] = True
            d["end"] = block_end
            return d
        d["chan_alloc"] = _parse_chan_alloc(r)
    if li == LEN_START_FRAG:
        d["frag_start"] = True
        end = block_end
    elif li == LEN_2ND_HALF_STOLEN:
        d["second_half_stolen"] = True
        end = block_end
    elif 0 < li <= 0x3A:
        end = min(start + li * 8, block_end)
    else:
        d["bad_length"] = True
        end = block_end
    sdu = r.bits[r.pos : end] if r.pos <= end else r.bits[:0]
    d["sdu"] = _strip_fill(sdu) if fill else sdu
    d["end"] = end
    return d


def parse_mac_block(bits, lchan: str) -> list[dict]:
    """Parse all MAC PDUs in a decoded downlink signalling block (SCH/HD, SCH/F, BNCH, STCH)."""
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    pos = 0
    out: list[dict] = []
    while n - pos >= 16 and bits[pos:].any():
        r = BitReader(bits, pos)
        try:
            t = r.u(2)
            if t == 0:
                d = _parse_mac_resource(r, pos, n)
                out.append(d)
                if d.get("null") or d["end"] <= pos:
                    break
                pos = d["end"]
                if d.get("frag_start") or d.get("second_half_stolen"):
                    break
                continue
            if t == 1:
                if r.u(1) == 0:
                    d = dict(pdu="MAC-FRAG")
                    fill = r.u(1)
                    sdu = r.rest()
                    d["sdu"] = _strip_fill(sdu) if fill else sdu
                else:
                    d = dict(pdu="MAC-END")
                    fill = r.u(1)
                    r.u(1)
                    li = r.u(6)
                    if r.u(1):
                        d["slot_granting"] = r.u(8)
                    if r.u(1):
                        d["chan_alloc"] = _parse_chan_alloc(r)
                    end = min(pos + li * 8, n) if 0 < li <= 0x3A else n
                    sdu = bits[r.pos : end] if r.pos <= end else bits[:0]
                    d["sdu"] = _strip_fill(sdu) if fill else sdu
                out.append(d)
                break
            if t == 2:
                bt = r.u(2)
                if bt == 0:
                    out.append(_parse_sysinfo(r))
                elif bt == 1:
                    out.append(_parse_access_define(r))
                elif bt == 2:
                    out.append(dict(pdu="SYSINFO-DA"))
                else:
                    out.append(dict(pdu="BROADCAST-RESERVED"))
                break
            out.append(dict(pdu="MAC-D-BLCK" if lchan != "STCH" else "MAC-U-SIGNAL"))
            break
        except EndOfData as e:
            out.append(dict(pdu="TRUNCATED", error=str(e)))
            break
    return out


# ------------------------------------------------------- LLC / MLE / L3
LLC_TYPES = (
    "BL-ADATA", "BL-DATA", "BL-UDATA", "BL-ACK", "BL-ADATA+FCS", "BL-DATA+FCS", "BL-UDATA+FCS",
    "BL-ACK+FCS", "AL-SETUP", "AL-DATA", "AL-UDATA", "AL-ACK", "AL-RECONNECT", "SUPPL", "L2SIG",
    "AL-DISC",
)
MLE_PD = {1: "MM", 2: "CMCE", 4: "SNDCP", 5: "MLE", 6: "MGMT", 7: "TEST"}
MM_D_TYPES = {
    0: "D-OTAR", 1: "D-AUTHENTICATION", 2: "D-CK-CHANGE-DEMAND", 3: "D-DISABLE", 4: "D-ENABLE",
    5: "D-LOCATION-UPDATE-ACCEPT", 6: "D-LOCATION-UPDATE-COMMAND", 7: "D-LOCATION-UPDATE-REJECT",
    9: "D-LOCATION-UPDATE-PROCEEDING", 10: "D-ATTACH/DETACH-GROUP-ID", 11: "D-ATTACH/DETACH-GROUP-ID-ACK",
    12: "D-MM-STATUS", 15: "MM-PDU-NOT-SUPPORTED",
}
CMCE_D_TYPES = {
    0: "D-ALERT", 1: "D-CALL-PROCEEDING", 2: "D-CONNECT", 3: "D-CONNECT-ACK", 4: "D-DISCONNECT",
    5: "D-INFO", 6: "D-RELEASE", 7: "D-SETUP", 8: "D-STATUS", 9: "D-TX-CEASED", 10: "D-TX-CONTINUE",
    11: "D-TX-GRANTED", 12: "D-TX-WAIT", 13: "D-TX-INTERRUPT", 14: "D-CALL-RESTORE", 15: "D-SDS-DATA",
    16: "D-FACILITY",
}
MLE_D_TYPES = {
    0: "D-NEW-CELL", 1: "D-PREPARE-FAIL", 2: "D-NWRK-BROADCAST", 3: "D-NWRK-BROADCAST-EXT",
    4: "D-RESTORE-ACK", 5: "D-RESTORE-FAIL", 6: "D-CHANNEL-RESPONSE",
}
COMM_TYPES = ("individual", "group", "group-ack", "broadcast")
CIRCUIT_MODES = ("speech", "data 7.2", "data 4.8 x1", "data 4.8 x4", "data 2.4 x1", "data 2.4 x4",
                 "data 2.4 x8", "reserved")
CA_RESELECTION = ("no", "no", "yes", "yes")
TX_GRANT = ("granted", "not granted", "queued", "granted to other")
SDS_PID = {0x02: "simple text", 0x03: "simple location", 0x82: "text messaging", 0x83: "LIP location",
           0x0C: "LIP (simple)", 0x8C: "status/ack"}


def _type2(r: BitReader, n: int) -> int | None:
    """A type-2 element: P-bit, then ``n`` bits if present."""
    return r.u(n) if r.u(1) else None


def _party(r: BitReader, d: dict, prefix: str, cpti: int) -> None:
    """Party address after a (calling / transmitting) party type identifier."""
    if cpti == 0:
        d[prefix + "_sna"] = r.u(8)
    elif cpti in (1, 2):
        d[prefix + "_ssi"] = r.u(24)
        if cpti == 2:
            d[prefix + "_ext"] = r.u(24)


def _parse_network_time(v: int) -> dict:
    """TETRA network time (18.5.24), 48 bits."""
    utc = v >> 24
    sign = (v >> 23) & 1
    offset = (v >> 17) & 0x3F
    year = 2000 + ((v >> 11) & 0x3F)
    secs = 2 * utc
    days, rem = divmod(secs, 86400)
    import datetime as _dt

    t = _dt.datetime(year, 1, 1) + _dt.timedelta(days=days, seconds=rem)
    off = (-1 if sign else 1) * offset * 15
    return dict(utc=t.strftime("%Y-%m-%d %H:%M:%S"), local_offset_min=off)


def _parse_neighbour(r: BitReader) -> dict:
    """Neighbour cell information (18.5.17)."""
    n = dict(ca=r.u(5), reselection_types=r.u(2), synchronised=r.u(1), service_level=r.u(2), carrier=r.u(12))
    if r.u(1):
        ext = _type2(r, 10)
        if ext is not None:
            n["band"], n["offset"] = ext >> 6, (ext >> 4) & 3
            n["duplex"], n["reverse"] = (ext >> 1) & 7, ext & 1
        for key, bits in (("mcc", 10), ("mnc", 14), ("la", 14), ("ms_txpwr_max", 3), ("rxlev_access_min", 4),
                          ("subscriber_class", 16), ("bs_service_details", 12), ("timeshare", 5),
                          ("tdma_frame_offset", 6)):
            v = _type2(r, bits)
            if v is not None:
                n[key] = v
    return n


def _parse_mle(r: BitReader, d: dict) -> None:
    t = r.u(3)
    d["l3"] = MLE_D_TYPES.get(t, f"MLE-{t}")
    if t == 2:  # D-NWRK-BROADCAST (18.4.1.4.1)
        d["cell_reselect_params"] = r.u(16)
        d["cell_service_level"] = r.u(2)
        if r.u(1):
            nt = _type2(r, 48)
            if nt is not None:
                d["network_time"] = _parse_network_time(nt)
            count = _type2(r, 3)
            if count:
                d["neighbours"] = [_parse_neighbour(r) for _ in range(count)]


def _calling_party(r: BitReader, d: dict) -> None:
    cpti = r.u(2)
    if cpti == 0:
        d["calling_sna"] = r.u(8)
    elif cpti in (1, 2):
        d["calling_ssi"] = r.u(24)
        if cpti == 2:
            d["calling_ext"] = r.u(24)


def _basic_service(v: int, d: dict) -> None:
    """Basic service information (14.8.2): circuit mode type, encryption flag, communication type, slots."""
    d["circuit_mode"] = CIRCUIT_MODES[v >> 5]
    d["e2e_encrypted"] = (v >> 4) & 1
    d["comm_type"] = COMM_TYPES[(v >> 2) & 3]


def _notification(r: BitReader, d: dict) -> None:
    v = _type2(r, 6)
    if v is not None:
        d["notification"] = v


def _parse_cmce(r: BitReader, d: dict) -> None:
    t = r.u(5)
    d["l3"] = CMCE_D_TYPES.get(t, f"CMCE-{t}")
    if t <= 14 and t != 8:
        d["call_id"] = r.u(14)
        if t == 7:  # D-SETUP (14.7.1.12)
            r.u(4)
            r.u(1)
            d["duplex"] = r.u(1)
            _basic_service(r.u(8), d)
            d["tx_grant"] = TX_GRANT[r.u(2)]
            r.u(1)
            d["priority"] = r.u(4)
            if r.u(1):
                _notification(r, d)
                v = _type2(r, 24)
                if v is not None:
                    d["temporary_address"] = v
                cpti = _type2(r, 2)
                if cpti is not None:
                    _party(r, d, "calling", cpti)
        elif t == 2:  # D-CONNECT (14.7.1.4)
            r.u(4)
            r.u(1)
            d["duplex"] = r.u(1)
            d["tx_grant"] = TX_GRANT[r.u(2)]
            r.u(1)
            d["call_ownership"] = r.u(1)
            if r.u(1):
                v = _type2(r, 4)
                if v is not None:
                    d["priority"] = v
                v = _type2(r, 8)
                if v is not None:
                    _basic_service(v, d)
                v = _type2(r, 24)
                if v is not None:
                    d["temporary_address"] = v
                _notification(r, d)
        elif t == 11:  # D-TX-GRANTED (14.7.1.15)
            d["tx_grant"] = TX_GRANT[r.u(2)]
            r.u(1)
            d["encryption_control"] = r.u(1)
            r.u(1)
            if r.u(1):
                _notification(r, d)
                tpti = _type2(r, 2)
                if tpti is not None:
                    _party(r, d, "transmitting", tpti)
        elif t == 6 or t == 4:  # D-RELEASE / D-DISCONNECT
            d["cause"] = r.u(5)
            if t == 6 and r.u(1):
                _notification(r, d)
    elif t == 8:  # D-STATUS
        _calling_party(r, d)
        d["status"] = r.u(16)
    elif t == 15:  # D-SDS-DATA: report type and length only, never the content.
        _calling_party(r, d)
        sdti = r.u(2)
        if sdti < 3:
            d["sds_bits"] = (16, 32, 64)[sdti]
        else:
            n = r.u(11)
            d["sds_bits"] = n
            if n >= 8:
                pid = r.u(8)
                d["sds_pid"] = pid
                d["sds_proto"] = SDS_PID.get(pid, f"PID 0x{pid:02x}")


def parse_tm_sdu(sdu, encrypted: bool = False) -> dict:
    """Parse the LLC PDU (and the MLE / MM / CMCE header inside it) carried in a TM-SDU."""
    d: dict = {}
    if encrypted:
        d["encrypted"] = True
        return d
    r = BitReader(sdu)
    try:
        t = r.u(4)
        d["llc"] = LLC_TYPES[t]
        if t in (0, 4):
            r.u(2)
        elif t in (1, 5, 3, 7):
            r.u(1)
        elif t > 7:
            return d
        if t >= 4:
            r.end -= 32  # FCS
        if r.left < 3:
            return d
        pd = r.u(3)
        d["mle"] = MLE_PD.get(pd, f"PD{pd}")
        if pd == 1:
            t = r.u(4)
            d["l3"] = MM_D_TYPES.get(t, f"MM-{t}")
        elif pd == 2:
            _parse_cmce(r, d)
        elif pd == 5:
            _parse_mle(r, d)
    except EndOfData:
        d["truncated"] = True
    return d


# ------------------------------------------------------- descriptions
def describe(d: dict, cell_band: int | None = None) -> str:
    p = d.get("pdu")
    if p == "SYNC":
        return (f"SYNC MCC {d['mcc']} MNC {d['mnc']} CC {d['cc']} TN {d['tn']} FN {d['fn']} MN {d['mn']} "
                f"{SHARING_MODES[d['sharing_mode']]} load {d['cell_load']}")
    if p == "SYSINFO":
        ul = f" UL {mhz(d['ul_hz'])}" if d.get("ul_hz") else ""
        hf = f"HF {d['hyperframe']}" if "hyperframe" in d else f"CCK {d['cck_id']}"
        return (f"SYSINFO main {mhz(d['dl_hz'])} MHz (carrier {d['main_carrier']} band {d['band']}){ul} "
                f"LA {d['la']} {hf} rxlev_min {d['rxlev_access_min_dbm']} dBm "
                f"services: {', '.join(d['services']) or '-'}")
    if p == "CELL-CODE":
        src = "given with --cell" if d["source"] == "manual" else f"shared from {d['source']} MHz"
        return (f"colour code MCC {d['mcc']} MNC {d['mnc']} CC {d['cc']} {src}"
                + (", TDMA time inherited" if d.get("time_inherited") else ", TDMA time unknown"))
    if p == "ACCESS-DEFINE":
        s = (f"ACCESS-DEFINE code {d['access_code']} {'assigned' if d['assigned'] else 'common'} "
             f"IMM {d['imm']} WT {d['wt']} Nu {d['nu']} min prio {d['min_priority']}")
        if "gssi" in d:
            s += f" GSSI {d['gssi']}"
        return s
    if p == "ACCESS-ASSIGN":
        return f"AACH DL {d['dl']}" + (f" UL {d['ul']}" if "ul" in d else "")
    if p == "MAC-RESOURCE":
        if d.get("null"):
            return "MAC-RESOURCE null"
        s = f"MAC-RESOURCE {d['addr_type']}"
        if "ssi" in d:
            s += " broadcast" if d["ssi"] == 0xFFFFFF else f" {d['ssi']}"
        if "event_label" in d:
            s += f" EL {d['event_label']}"
        if "usage_marker" in d:
            s += f" UM {d['usage_marker']}"
        if d["enc"]:
            s += f" encrypted(mode {d['enc']})"
        if "chan_alloc" in d:
            s += " " + _describe_chan_alloc(d["chan_alloc"], cell_band)
        if d.get("frag_start"):
            s += " [frag start]"
        return s + _describe_l3(d, cell_band)
    if p in ("MAC-FRAG", "MAC-END"):
        s = p
        if "chan_alloc" in d:
            s += " " + _describe_chan_alloc(d["chan_alloc"], cell_band)
        return s + _describe_l3(d, cell_band)
    return p or "?"


def _describe_l3(d: dict, cell_band: int | None = None) -> str:
    l3 = d.get("tm")
    if l3 and cell_band and "neighbours" in l3:
        for n in l3["neighbours"]:
            n.setdefault("band", cell_band)
    if not l3:
        return ""
    if l3.get("encrypted"):
        return " | encrypted SDU"
    s = " | " + l3.get("llc", "")
    if "mle" in l3:
        s += f" {l3['mle']}"
    if "l3" in l3:
        s += f" {l3['l3']}"
    for k, lbl in (("call_id", "call"), ("circuit_mode", ""), ("comm_type", ""), ("tx_grant", ""),
                   ("priority", "prio"), ("cause", "cause"), ("calling_ssi", "from"), ("transmitting_ssi", "talker"),
                   ("temporary_address", "temp"), ("notification", "notif"), ("status", "status"),
                   ("sds_bits", "bits"), ("sds_proto", "")):
        if k in l3:
            s += f" {lbl} {l3[k]}" if lbl else f" {l3[k]}"
    if l3.get("e2e_encrypted"):
        s += " e2e-encrypted"
    if "network_time" in l3:
        nt = l3["network_time"]
        s += f" time {nt['utc']} UTC (local {nt['local_offset_min']:+d} min)"
    if "neighbours" in l3:
        parts = []
        for n in l3["neighbours"]:
            band = n.get("band")
            f = f" {mhz(carrier_frequency(band, n['carrier'], n.get('offset', 0)))} MHz" if band else ""
            extra = "".join(f" {k.upper()} {n[k]}" for k in ("mcc", "mnc", "la") if k in n)
            parts.append(f"[CA {n['ca']} carrier {n['carrier']}{f}{extra}]")
        s += f" neighbours {len(parts)}: " + " ".join(parts)
    if l3.get("truncated"):
        s += " (truncated)"
    return s
