"""Long-term monitoring: activity reports appended to an SQLite database or CSV files,
and hourly / daily summaries of such a log (`tetrascan stats`).

One row per carrier and reporting period goes into the ``reports`` table (or the CSV
file), one row per finished traffic session into ``sessions`` (or ``<name>.sessions.csv``).
Only counts, durations and usage markers are stored, never addresses or contents.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

REPORT_COLUMNS = [
    ("t_start", "REAL"), ("t_end", "REAL"), ("freq_mhz", "REAL"),
    ("mcc", "INTEGER"), ("mnc", "INTEGER"), ("cc", "INTEGER"), ("la", "INTEGER"),
    ("period_s", "REAL"), ("locked", "INTEGER"), ("bursts", "INTEGER"),
    ("mer_db", "REAL"), ("level_db", "REAL"), ("crc_pct", "REAL"),
]
for _tn in range(1, 5):
    REPORT_COLUMNS += [(f"tn{_tn}_{k}", "REAL") for k in
                       ("traffic", "assigned", "common", "unallocated", "ul_traffic", "sync", "sessions")]
REPORT_COLUMNS += [
    ("sessions", "INTEGER"), ("sessions_ongoing", "INTEGER"), ("usage_markers", "INTEGER"),
    ("session_s_total", "REAL"), ("session_s_median", "REAL"), ("session_s_max", "REAL"),
    ("pdus", "INTEGER"), ("null_pdus", "INTEGER"), ("addresses", "INTEGER"), ("encrypted", "INTEGER"),
    ("clear", "INTEGER"), ("fragmented", "INTEGER"), ("random_access_acks", "INTEGER"),
    ("chan_allocs", "INTEGER"), ("chan_allocs_encrypted", "INTEGER"), ("clear_l3", "TEXT"),
]
SESSION_COLUMNS = [("t_start", "REAL"), ("freq_mhz", "REAL"), ("tn", "INTEGER"), ("usage_marker", "INTEGER"),
                   ("duration_s", "REAL")]
_DL_KEYS = {"traffic": "traffic", "assigned": "assigned control", "common": "common control",
            "unallocated": "unallocated"}


def report_row(rep: dict, freq_hz: float, t0: float, t_wall: float, cell: dict | None = None,
               la: int | None = None) -> dict:
    """Flatten an activity report into one log row. ``t0`` converts report times to Unix time;
    ``t_wall`` is used when the period saw no bursts at all."""
    q = rep.get("quality", {})
    t_end = t0 + rep["t_end"] if rep.get("t_end") is not None else t_wall
    t_start = t0 + rep["t_start"] if rep.get("t_start") is not None else t_wall
    row = dict(t_start=round(t_start, 3), t_end=round(t_end, 3), freq_mhz=round(freq_hz / 1e6, 5),
               mcc=cell and cell["mcc"], mnc=cell and cell["mnc"], cc=cell and cell["cc"], la=la,
               period_s=rep["period_s"], locked=int(bool(q.get("locked"))), bursts=q.get("bursts"),
               mer_db=q.get("mer_db"), level_db=q.get("level_db"), crc_pct=q.get("crc_pct"))
    for tn in range(1, 5):
        s = rep["slots"].get(tn) or rep["slots"].get(str(tn))
        for k, cat in _DL_KEYS.items():
            row[f"tn{tn}_{k}"] = s["dl"].get(cat, 0.0) if s else None
        row[f"tn{tn}_ul_traffic"] = s["ul"].get("traffic", 0.0) if s else None
        row[f"tn{tn}_sync"] = s["sync_bursts"] if s else None
        row[f"tn{tn}_sessions"] = s["sessions"] if s else None
    c, g = rep["sessions"], rep["signalling"]
    row.update(sessions=c["count"], sessions_ongoing=c["ongoing"], usage_markers=c["usage_markers"],
               session_s_total=c.get("total_s", 0.0), session_s_median=c.get("median_s"),
               session_s_max=c.get("max_s"), pdus=g["pdus"], null_pdus=g["null"], addresses=g["addresses"],
               encrypted=g["encrypted"], clear=g["clear"], fragmented=g["fragmented"],
               random_access_acks=g["random_access_acks"], chan_allocs=g["chan_allocs"],
               chan_allocs_encrypted=g["chan_allocs_encrypted"],
               clear_l3=json.dumps(rep["clear_l3"], sort_keys=True))
    return row


def session_rows(rep: dict, freq_hz: float, t0: float) -> list[dict]:
    return [dict(t_start=round(t0 + s["start"], 3), freq_mhz=round(freq_hz / 1e6, 5), tn=s["tn"],
                 usage_marker=s["um"], duration_s=s["duration_s"]) for s in rep.get("session_list", [])]


class ReportLog:
    """Appends activity reports to ``path``: SQLite for .db / .sqlite / .sqlite3, CSV otherwise."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.sqlite = self.path.suffix.lower() in (".db", ".sqlite", ".sqlite3")
        if self.sqlite:
            self.db = sqlite3.connect(self.path)
            for name, cols in (("reports", REPORT_COLUMNS), ("sessions", SESSION_COLUMNS)):
                self.db.execute(f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(f'{c} {t}' for c, t in cols)})")
            self.db.execute("CREATE INDEX IF NOT EXISTS reports_time ON reports (t_start)")
            self.db.commit()
        else:
            self.session_path = self.path.with_name(self.path.stem + ".sessions.csv")

    def write(self, rows: list[dict], sessions: list[dict]) -> None:
        if self.sqlite:
            for name, cols, data in (("reports", REPORT_COLUMNS, rows), ("sessions", SESSION_COLUMNS, sessions)):
                if data:
                    names = [c for c, _ in cols]
                    self.db.executemany(f"INSERT INTO {name} ({', '.join(names)}) VALUES "
                                        f"({', '.join('?' * len(names))})", [[r.get(c) for c in names] for r in data])
            self.db.commit()
            return
        for path, cols, data in ((self.path, REPORT_COLUMNS, rows), (self.session_path, SESSION_COLUMNS, sessions)):
            if not data:
                continue
            new = not path.exists() or path.stat().st_size == 0
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, [c for c, _ in cols])
                if new:
                    w.writeheader()
                w.writerows(data)

    def close(self) -> None:
        if self.sqlite:
            self.db.close()


# ------------------------------------------------------------------ stats
def read_reports(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        db = sqlite3.connect(p)
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM reports ORDER BY t_start")]
        db.close()
        return rows
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    types = dict(REPORT_COLUMNS)
    for r in rows:
        for k, v in r.items():
            if types.get(k) in ("REAL", "INTEGER"):
                r[k] = float(v) if v not in ("", None) else None
    return rows


def summarise(rows: list[dict], by: str = "hour") -> list[dict]:
    """Aggregate report rows per carrier and hour (or day) of local time."""
    fmt = "%Y-%m-%d %H:00" if by == "hour" else "%Y-%m-%d"
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(time.strftime(fmt, time.localtime(r["t_start"])), r["freq_mhz"])].append(r)
    out = []
    for (bucket, freq), rs in sorted(groups.items()):
        cover = sum(r["period_s"] or 0 for r in rs)
        if cover <= 0:
            continue

        def wavg(key):
            pairs = [(r[key], r["period_s"]) for r in rs if r.get(key) is not None and r["period_s"]]
            w = sum(p for _, p in pairs)
            return sum(v * p for v, p in pairs) / w if w else None

        # mean number of downlink slots carrying traffic = traffic intensity in Erlang
        erl = sum((wavg(f"tn{tn}_traffic") or 0) / 100 for tn in range(1, 5))
        busiest = max((wavg(f"tn{tn}_traffic") or 0) for tn in range(1, 5))
        sessions = sum(r["sessions"] or 0 for r in rs)
        pdus = sum(r["pdus"] or 0 for r in rs)
        enc = sum(r["encrypted"] or 0 for r in rs)
        out.append(dict(
            bucket=bucket, freq_mhz=freq, cc=next((int(r["cc"]) for r in rs if r.get("cc") is not None), None),
            coverage_min=round(cover / 60, 1), traffic_erlang=round(erl, 2), busiest_slot_pct=round(busiest, 1),
            sessions_per_hour=round(sessions / cover * 3600, 1),
            mean_session_s=round(sum(r["session_s_total"] or 0 for r in rs) / sessions, 1) if sessions else None,
            pdus_per_min=round(pdus / cover * 60, 1), max_addresses=int(max((r["addresses"] or 0) for r in rs)),
            encrypted_pct=round(100 * enc / pdus, 1) if pdus else None,
            random_access_per_min=round(sum(r["random_access_acks"] or 0 for r in rs) / cover * 60, 1),
            mer_db=round(wavg("mer_db"), 1) if wavg("mer_db") is not None else None,
            crc_pct=round(wavg("crc_pct"), 1) if wavg("crc_pct") is not None else None,
        ))
    return out


def format_summary(summary: list[dict], by: str = "hour") -> str:
    head = (f"{by:<16} {'freq MHz':>9} {'CC':>3} {'cover':>6} {'Erl':>5} {'busy%':>6} {'sess/h':>7} "
            f"{'avg s':>6} {'PDU/min':>8} {'addr':>5} {'enc%':>5} {'RA/min':>7} {'MER':>5} {'CRC%':>5}")
    lines = [head]

    def f(v, w, p=1):
        return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

    for s in summary:
        lines.append(f"{s['bucket']:<16} {s['freq_mhz']:>9.4f} {s['cc'] if s['cc'] is not None else '-':>3} "
                     f"{f(s['coverage_min'], 5, 1)}m {f(s['traffic_erlang'], 5, 2)} {f(s['busiest_slot_pct'], 6)} "
                     f"{f(s['sessions_per_hour'], 7)} {f(s['mean_session_s'], 6)} {f(s['pdus_per_min'], 8)} "
                     f"{s['max_addresses']:>5} {f(s['encrypted_pct'], 5, 0)} {f(s['random_access_per_min'], 7)} "
                     f"{f(s['mer_db'], 5)} {f(s['crc_pct'], 5, 0)}")
    return "\n".join(lines)
