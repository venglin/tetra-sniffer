"""Band scanner.

Pass 1 (survey): the HackRF steps across the band at 20 Msps. Only one side of each
capture ([+0.25, +7.75] MHz from the LO) is used, so the LO step is 7.5 MHz and the
DC spike never lands in the data. Averaged PSDs give the channel power on a 6.25 kHz
raster; peaks above the local noise floor with a -10 dB width of 15-30 kHz look like
TETRA (pi/4-DQPSK at 18 ksymbol/s occupies ~22 kHz).

Pass 2 (verify): candidates are grouped into windows of at most 1.6 MHz, the HackRF
is retuned to each window at 2.304 Msps and a capture is decoded by the normal
multi-channel receiver. A channel counts as TETRA when the burst structure locks;
a decoded BSCH gives MCC / MNC / colour code and a BNCH gives the SYSINFO.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .dsp import db, iq_from_cs8, psd
from .receiver import MultiReceiver

SURVEY_RATE = 20e6
SURVEY_STEP = 7.5e6
SURVEY_LOW, SURVEY_HIGH = 0.25e6, 7.75e6
VERIFY_RATE = 2.304e6
VERIFY_SPAN = 1.6e6
RASTER = 6250.0


@dataclass
class Candidate:
    freq_hz: float
    power_db: float  # channel power, dBFS
    snr_db: float  # channel power over noise floor in the same bandwidth
    width_khz: float  # -10 dB width
    tetra_like: bool
    result: dict = field(default_factory=dict)


def collect(dev, nbytes: int, timeout: float = 5.0) -> bytes:
    chunks, have, t_end = [], 0, time.monotonic() + timeout
    while have < nbytes:
        if time.monotonic() > t_end:
            raise TimeoutError("no samples from HackRF")
        try:
            c = dev.queue.get(timeout=0.5)
        except Exception:
            continue
        chunks.append(c)
        have += len(c)
    return b"".join(chunks)[:nbytes]


def survey(dev, f_start: float, f_stop: float, dwell: float = 0.04, nfft: int = 8192, log=None,
           ppm: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (frequencies Hz, PSD dBFS per bin, worst clipping fraction)."""
    corr = 1 + ppm * 1e-6
    los = np.arange(f_start - SURVEY_LOW, f_stop, SURVEY_STEP)
    nbytes = int(SURVEY_RATE * dwell) * 2
    settle = int(SURVEY_RATE * 0.01) * 2
    freqs, powers = [], []
    clip_max = 0.0
    dev.retune(los[0] / corr, settle_bytes=settle)
    for lo in los:
        dev.retune(lo / corr, settle_bytes=settle)
        raw = collect(dev, nbytes)
        a = np.frombuffer(raw, np.int8)
        clip_max = max(clip_max, float(np.mean(np.abs(a.astype(np.int16)) >= 127)))
        f, p = psd(iq_from_cs8(raw), SURVEY_RATE, nfft)
        m = (f >= SURVEY_LOW) & (f < SURVEY_LOW + SURVEY_STEP)
        freqs.append(lo + f[m])
        powers.append(db(p[m]))
        if log:
            log(f"survey {lo / 1e6:8.3f} MHz  peak {np.max(db(p[m])):6.1f} dB/bin")
    f = np.concatenate(freqs)
    p = np.concatenate(powers)
    keep = (f >= f_start) & (f <= f_stop)
    return f[keep], p[keep], clip_max


def find_candidates(f: np.ndarray, p_db: np.ndarray, threshold_db: float = 10.0) -> list[Candidate]:
    df = float(np.median(np.diff(f)))
    lin = 10 ** (p_db / 10)
    half = max(1, int(round(9e3 / df)))
    k = np.ones(2 * half + 1)
    chan = np.convolve(lin, k, mode="same")
    win = max(5, int(round(1e6 / df)) | 1)
    floor = 10 ** (ndimage.percentile_filter(p_db, 25, size=win, mode="nearest") / 10)
    snr = db(chan / (floor * len(k)))
    sep = max(1, int(round(12.5e3 / df)))
    is_peak = (snr == ndimage.maximum_filter(snr, size=2 * sep + 1, mode="nearest")) & (snr >= threshold_db)
    out: list[Candidate] = []
    for i in np.flatnonzero(is_peak):
        lo, hi = max(0, i - sep), min(len(f), i + sep + 1)
        w = np.maximum(lin[lo:hi] - floor[lo:hi], 0)
        fc = float(np.sum(f[lo:hi] * w) / np.sum(w)) if np.sum(w) > 0 else float(f[i])
        fc = round(fc / RASTER) * RASTER
        # -10 dB width around the strongest bin near the peak
        j = lo + int(np.argmax(p_db[lo:hi]))
        ref = p_db[j] - 10
        a = j
        while a > 0 and p_db[a - 1] >= ref and j - a < 4 * sep:
            a -= 1
        b = j
        while b < len(f) - 1 and p_db[b + 1] >= ref and b - j < 4 * sep:
            b += 1
        width = (b - a + 1) * df / 1e3
        if any(abs(c.freq_hz - fc) < 12.5e3 for c in out):
            continue
        out.append(Candidate(fc, float(db(chan[i])), float(snr[i]), width, 15 <= width <= 30))
    return out


def plan_windows(freqs: list[float], span: float = VERIFY_SPAN, usable: float = 0.9e6,
                 dc_guard: float = 40e3) -> list[tuple[float, list[float]]]:
    """Group channel frequencies into capture windows; returns [(lo, [freqs])]."""
    freqs = sorted(freqs)
    groups: list[list[float]] = []
    for f in freqs:
        if groups and f - groups[-1][0] <= span:
            groups[-1].append(f)
        else:
            groups.append([f])
    out = []
    for g in groups:
        mid = (g[0] + g[-1]) / 2
        for shift in sorted(np.arange(-400e3, 400e3 + 1, 12.5e3), key=abs):
            lo = mid + shift
            off = np.array(g) - lo
            if np.all(np.abs(off) >= dc_guard) and np.all(np.abs(off) <= usable):
                out.append((float(lo), g))
                break
        else:
            out.extend((f + 100e3, [f]) for f in g)
    return out


def verify(dev, candidates: list[Candidate], dwell: float = 1.5, log=None, ppm: float = 0.0,
           method: str = "coherent") -> None:
    corr = 1 + ppm * 1e-6
    by_freq = {c.freq_hz: c for c in candidates}
    nbytes = int(VERIFY_RATE * dwell) * 2
    captures = []
    for lo, group in plan_windows([c.freq_hz for c in candidates]):
        if log:
            log(f"verify {lo / 1e6:8.3f} MHz: {', '.join(f'{f / 1e6:.4f}' for f in group)}")
        dev.retune(lo / corr, settle_bytes=int(VERIFY_RATE * 0.02) * 2)
        raw = collect(dev, nbytes)
        captures.append((lo, group, raw))
        _decode_window(lo, group, raw, by_freq, method)
    # Second pass: carriers that locked without a BSCH of their own (secondary carriers) are
    # decoded again with the colour codes learned from the other windows.
    codes = sorted({(r["mcc"], r["mnc"], r["cc"]) for c in candidates
                    if (r := c.result).get("code_source") == "bsch"})
    if not codes:
        return
    for lo, group, raw in captures:
        retry = [f for f in group if not by_freq[f].result.get("code_source") and by_freq[f].result.get("locked")]
        if retry:
            if log:
                log(f"retry  {lo / 1e6:8.3f} MHz with {len(codes)} known colour code(s)")
            _decode_window(lo, group, raw, by_freq, method, codes, only=set(retry))


def _decode_window(lo, group, raw, by_freq, method, known=None, only=None) -> None:
    rx = MultiReceiver(VERIFY_RATE, lo, group, method=method, known_cells=known)
    iq = iq_from_cs8(raw)
    for i in range(0, len(iq), 262144):
        rx.feed(iq[i : i + 262144])
    for ch in rx.channels:
        if only is None or ch.freq_hz in only:
            by_freq[ch.freq_hz].result = ch.summary()


def is_tetra(c: Candidate) -> bool:
    """Confirmed: a BSCH passed its CRC, or blocks decoded with a colour code known from another carrier."""
    return bool(c.result.get("code_source"))


def is_probably_tetra(c: Candidate) -> bool:
    """Burst structure locked with clean pi/4-DQPSK symbols, but no BSCH decoded (yet)."""
    r = c.result
    if not r or not r.get("bursts"):
        return False
    return (r.get("locked", False) and r["good_bursts"] >= 20 and r["good_bursts"] >= 0.9 * r["bursts"]
            and (r.get("mer_db") or 0) >= 8.0)


def status(c: Candidate) -> str:
    if is_tetra(c):
        return "TETRA"
    if is_probably_tetra(c):
        return "TETRA?"
    if c.result:
        return "no lock"
    return "?" if c.tetra_like else "other"


def estimate_ppm(candidates: list[Candidate]) -> float | None:
    """Residual HackRF clock error in ppm.

    With an LO error e (relative), a carrier at F is seen at F*(1-e). Only main carriers are
    used, because their exact frequency is known from SYSINFO.
    """
    e = []
    for c in candidates:
        r = c.result
        if not is_tetra(c) or r.get("cfo_hz") is None or "main_carrier_hz" not in r:
            continue
        true = r["main_carrier_hz"]
        measured = c.freq_hz + r["cfo_hz"]
        if abs(measured - true) < 20e3:
            e.append((true - measured) / true * 1e6)
    return float(np.median(e)) if e else None


def refine_frequency(c: Candidate, ppm: float | None) -> float:
    """Best estimate of the nominal carrier frequency of a verified channel."""
    r = c.result
    measured = c.freq_hz + (r.get("cfo_hz") or 0.0)
    if "main_carrier_hz" in r and abs(r["main_carrier_hz"] - measured) < 20e3:
        return r["main_carrier_hz"]
    if ppm is not None:
        measured *= 1 + ppm * 1e-6
    return round(measured / RASTER) * RASTER
