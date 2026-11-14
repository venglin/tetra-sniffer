"""GSMTAP output (UDP to Wireshark, or a PCAP file) for decoded TETRA blocks.

Wireshark dissects GSMTAP (UDP port 4729) with type TETRA_I1 down to the TETRA MAC
and higher layers. The payload is the type-1 bits of the logical channel, packed MSB
first and zero padded to whole bytes.
"""

from __future__ import annotations

import socket
import struct
import time

import numpy as np

GSMTAP_PORT = 4729
GSMTAP_VERSION = 2
GSMTAP_TYPE_TETRA_I1 = 0x05
SUBTYPES = {"BSCH": 0x01, "AACH": 0x02, "SCH/HU": 0x03, "SCH/HD": 0x04, "SCH/F": 0x05, "BNCH": 0x06,
            "STCH": 0x07, "TCH": 0x08}
LINKTYPE_RAW = 101  # raw IPv4


def gsmtap_packet(lchan: str, bits: np.ndarray, tn: int = 1, frame_number: int = 0, arfcn: int = 0,
                  signal_dbm: int = 0, snr_db: int = 0, sub_slot: int = 0) -> bytes:
    hdr = struct.pack(">BBBBHbbIBBBB", GSMTAP_VERSION, 4, GSMTAP_TYPE_TETRA_I1, (tn - 1) & 3, arfcn & 0xFFFF,
                      max(-128, min(127, int(signal_dbm))), max(-128, min(127, int(snr_db))),
                      frame_number & 0xFFFFFFFF, SUBTYPES[lchan], 0, sub_slot, 0)
    return hdr + np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def _ipv4_udp(payload: bytes, dport: int = GSMTAP_PORT, sport: int = 50000) -> bytes:
    udp = struct.pack(">HHHH", sport, dport, 8 + len(payload), 0) + payload
    total = 20 + len(udp)
    ip = bytearray(struct.pack(">BBHHHBBH4s4s", 0x45, 0, total, 0, 0, 64, 17, 0, bytes([127, 0, 0, 1]),
                               bytes([127, 0, 0, 1])))
    s = sum(struct.unpack(">10H", bytes(ip)))
    s = (s & 0xFFFF) + (s >> 16)
    s = (s & 0xFFFF) + (s >> 16)
    ip[10:12] = struct.pack(">H", ~s & 0xFFFF)
    return bytes(ip) + udp


class GsmtapUdp:
    def __init__(self, target: str):
        host, _, port = target.partition(":")
        self.addr = (host or "127.0.0.1", int(port) if port else GSMTAP_PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, pkt: bytes, ts: float | None = None) -> None:
        self.sock.sendto(pkt, self.addr)

    def close(self) -> None:
        self.sock.close()


class PcapWriter:
    def __init__(self, path: str):
        self.f = open(path, "wb")
        self.f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, LINKTYPE_RAW))

    def send(self, pkt: bytes, ts: float | None = None) -> None:
        ts = time.time() if ts is None else ts
        frame = _ipv4_udp(pkt)
        sec = int(ts)
        self.f.write(struct.pack("<IIII", sec, int((ts - sec) * 1e6), len(frame), len(frame)) + frame)

    def close(self) -> None:
        self.f.close()
