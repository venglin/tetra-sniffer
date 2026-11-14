"""pi/4-DQPSK burst demodulator for the TETRA continuous downlink.

Input: one channel at 144 kHz (8 samples/symbol, see :class:`dsp.ChannelFilter`).
Output: :class:`Burst` objects with 510 soft bits each.

Processing steps

1. **Acquisition.** The carrier offset is found by matching the PSD against the
   raised-cosine spectrum of an 18 ksymbol/s signal (coarse, robust to adjacent
   channels), then refined with a 4th-power estimator on the differential symbols
   (range +-2.25 kHz).
2. **Burst search.** The downlink is continuous, so every slot boundary carries
   the 22-bit training sequence q (q1..q10 ends a burst, q11..q22 starts the next
   one). The normalised correlation of the differential signal with q is folded
   modulo the slot length (2040 samples) over several slots; a clear peak gives
   the slot timing.
3. **Tracking.** For every expected burst the timing is refined (+-4 samples,
   parabolic interpolation) using the q sequences at both ends plus the
   mid-burst sequence (n, p or y), which also tells the burst type. Symbols are
   interpolated at the refined instants and detected differentially, so no
   carrier phase recovery is needed. The residual carrier offset is measured per
   burst and fed back slowly.

Differential detection: ``d_k = r_k * conj(r_{k-1})``. Table 5.1 maps the dibit
(b1, b2) to +pi/4 (00), +3pi/4 (01), -3pi/4 (11), -pi/4 (10), so b1 is the sign
of Im(d) and b2 the sign of Re(d); both are used directly as soft bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import burst as B
from .coherent import coherent_soft
from .dsp import CHANNEL_RATE, SPS, psd, rrc_taps

SLOT = B.SYMBOLS_PER_BURST * SPS  # 2040 samples


@dataclass
class Burst:
    start: float  # channel-rate sample index of symbol 0 (since demodulator start)
    seq: int  # slot counter within the current lock epoch
    epoch: int  # incremented on every (re)acquisition
    kind: str | None  # burst.SDB / NDB_FULL / NDB_HALF, None if unrecognised
    soft: np.ndarray  # 510 soft bits, > 0 means bit 0
    score: float  # training sequence correlation, 0..1
    mer_db: float  # modulation error ratio of the differential symbols
    cfo_hz: float  # carrier offset estimate at this burst
    level_db: float  # mean power in the channel, dB relative to full scale
    symbols: np.ndarray | None = None  # 256 complex symbols (-1..254), residual offset removed


def _upsampled(t: np.ndarray) -> np.ndarray:
    u = np.zeros(SPS * (len(t) - 1) + 1, np.complex64)
    u[::SPS] = t
    return u


_TQ = _upsampled(B.TEMPLATE_Q)  # used by the burst search (full-length correlation)


def _diff(y: np.ndarray) -> np.ndarray:
    d = np.zeros(len(y), np.complex64)
    d[SPS:] = y[SPS:] * np.conj(y[:-SPS])
    return d


def _ncorr(d: np.ndarray, tmpl: np.ndarray, absd: np.ndarray | None = None) -> np.ndarray:
    """Normalised correlation; element n has the first template symbol at d[n]. Range 0..1."""
    if absd is None:
        absd = np.abs(d)
    num = np.abs(signal.correlate(d, tmpl, mode="valid", method="direct"))
    den = signal.correlate(absd, np.abs(tmpl).astype(np.float32), mode="valid", method="direct")
    return num / np.maximum(den, 1e-12)


def _ncorr_at(d: np.ndarray, absd: np.ndarray, tmpl: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Normalised correlation with a symbol template, only at the given start indices."""
    idx = starts[:, None] + SPS * np.arange(len(tmpl))
    num = np.abs(d[idx] @ np.conj(tmpl))
    return num / np.maximum(absd[idx].sum(axis=1), 1e-12)


def _raised_cosine_psd(f: np.ndarray, rs: float = B.SYMBOL_RATE, alpha: float = 0.35) -> np.ndarray:
    f = np.abs(f)
    f1, f2 = (1 - alpha) * rs / 2, (1 + alpha) * rs / 2
    out = np.zeros_like(f)
    out[f <= f1] = 1.0
    m = (f > f1) & (f < f2)
    out[m] = 0.5 * (1 + np.cos(np.pi / (alpha * rs) * (f[m] - f1)))
    return out


def estimate_cfo(x: np.ndarray, fs: float = CHANNEL_RATE, max_offset: float = 10e3) -> float:
    """Carrier offset of a TETRA signal in ``x`` (channel-rate samples)."""
    nfft = 1024
    f, p = psd(x, fs, nfft)
    p = p - np.percentile(p, 20)
    df = f[1] - f[0]
    k = np.arange(-int(14e3 / df), int(14e3 / df) + 1)
    shape = _raised_cosine_psd(k * df)
    corr = np.correlate(np.maximum(p, 0), shape, mode="same")
    allowed = np.abs(f) <= max_offset
    f0 = f[np.argmax(np.where(allowed, corr, -np.inf))]
    # 4th-power refinement at the best of 8 timing phases.
    n = np.arange(len(x))
    xm = x * np.exp(-2j * np.pi * f0 * n / fs).astype(np.complex64)
    h = rrc_taps().astype(np.float32)
    y = signal.lfilter(h, 1, xm)[len(h) :]
    d = y[SPS:] * np.conj(y[:-SPS])
    d4 = d ** 4 / np.maximum(np.abs(d) ** 3, 1e-20)
    z = [np.sum(d4[p::SPS]) for p in range(SPS)]
    zb = z[int(np.argmax(np.abs(z)))]
    return float(f0 + np.angle(-zb) / 4 * B.SYMBOL_RATE / (2 * np.pi))


class Demodulator:
    ACQ_SAMPLES = int(0.15 * CHANNEL_RATE)
    SEARCH_SLOTS = 8
    MARGIN_BEFORE = 6 * SPS + SPS + 8
    MARGIN_AFTER = (B.SYMBOLS_PER_BURST + 6) * SPS + 8
    CFO_LOOP_GAIN = 0.05
    TIMING_LOOP_GAIN = 0.3
    MAX_MISSES = 8

    def __init__(self, lock_threshold: float = 0.55, track_threshold: float = 0.45, cfo_hz: float | None = None,
                 method: str = "coherent"):
        if method not in ("coherent", "differential"):
            raise ValueError(f"unknown detection method {method!r}")
        self.method = method
        self.lock_threshold = lock_threshold
        self.track_threshold = track_threshold
        self.h = rrc_taps().astype(np.float32)
        self._zi = np.zeros(len(self.h) - 1, np.complex64)
        self._raw: list[np.ndarray] = []
        self._raw_len = 0
        self._y = np.zeros(0, np.complex64)
        self._y0 = 0
        self.cfo = cfo_hz
        self._mix_phase = 0.0
        self.state = "acq"
        self.epoch = 0
        self._seq = 0
        self._misses = 0
        self._next = 0.0
        self.locks = 0
        self.last_lock_metric = 0.0

    # ----------------------------------------------------------------- input
    def process(self, x: np.ndarray) -> list[Burst]:
        if self.cfo is None:
            self._raw.append(x)
            self._raw_len += len(x)
            if self._raw_len < self.ACQ_SAMPLES:
                return []
            x = np.concatenate(self._raw)
            self._raw, self._raw_len = [], 0
            self.cfo = estimate_cfo(x)
        if self.state == "acq":
            self.state = "search"
        w = -2 * np.pi * self.cfo / CHANNEL_RATE
        ph = self._mix_phase + w * np.arange(len(x), dtype=np.float64)
        self._mix_phase = float((self._mix_phase + w * len(x)) % (2 * np.pi))
        xm = x * np.exp(1j * ph).astype(np.complex64)
        y, self._zi = signal.lfilter(self.h, 1, xm, zi=self._zi)
        self._y = np.concatenate([self._y, y.astype(np.complex64)])
        out: list[Burst] = []
        if self.state == "search":
            self._search()
        if self.state == "lock":
            out = self._track()
        return out

    # ---------------------------------------------------------------- search
    def _search(self) -> None:
        need = (self.SEARCH_SLOTS + 1) * SLOT
        if len(self._y) < need:
            return
        y = self._y[-need:]
        base = self._y0 + len(self._y) - need
        c = _ncorr(_diff(y), _TQ)
        nfold = len(c) // SLOT
        folded = c[: nfold * SLOT].reshape(nfold, SLOT).mean(axis=0)
        phi = int(np.argmax(folded))
        self.last_lock_metric = float(folded[phi])
        if folded[phi] < self.lock_threshold:
            keep = need // 2
            self._y0 += len(self._y) - keep
            self._y = self._y[-keep:]
            return
        start = base + phi + (B.SYMBOLS_PER_BURST - B.SYM_Q_TAIL) * SPS
        while start - self.MARGIN_BEFORE < self._y0:
            start += SLOT
        self._next = float(start)
        self.state = "lock"
        self.epoch += 1
        self.locks += 1
        self._seq = 0
        self._misses = 0

    # -------------------------------------------------------------- tracking
    def _track(self) -> list[Burst]:
        out: list[Burst] = []
        while self.state == "lock":
            s = self._next
            c = int(round(s))
            if c + self.MARGIN_AFTER > self._y0 + len(self._y):
                break
            base = c - self.MARGIN_BEFORE
            seg = self._y[base - self._y0 : c + self.MARGIN_AFTER - self._y0]
            out.append(self._demod_burst(seg, base, s))
            self._next += SLOT
            self._seq += 1
            if self._misses >= self.MAX_MISSES:
                self.state = "search"
        if out and self.method == "coherent":
            soft = coherent_soft(np.stack([b.symbols for b in out]))
            for b, sb in zip(out, soft):
                b.soft = sb
        keep_from = int(self._next) - self.MARGIN_BEFORE - SPS if self.state == "lock" else \
            self._y0 + len(self._y) - (self.SEARCH_SLOTS + 1) * SLOT
        if keep_from > self._y0:
            self._y = self._y[keep_from - self._y0 :]
            self._y0 = keep_from
        return out

    def _demod_burst(self, seg: np.ndarray, base: int, s: float) -> Burst:
        d = _diff(seg)
        absd = np.abs(d)
        c = int(round(s))
        deltas = np.arange(-4, 5)
        pos = c - base + deltas  # index of symbol 0 in seg
        q1 = _ncorr_at(d, absd, B.TEMPLATE_Q, pos - (B.SYMBOLS_PER_BURST - B.SYM_Q_TAIL) * SPS)
        q2 = _ncorr_at(d, absd, B.TEMPLATE_Q, pos + B.SYM_Q_TAIL * SPS)
        my = _ncorr_at(d, absd, B.TEMPLATE_Y, pos + B.SYM_TRAIN_SYNC * SPS)
        mn = _ncorr_at(d, absd, B.TEMPLATE_N, pos + B.SYM_TRAIN_MID * SPS)
        mp = _ncorr_at(d, absd, B.TEMPLATE_P, pos + B.SYM_TRAIN_MID * SPS)
        mid = np.maximum(my, np.maximum(mn, mp))
        score = (q1 + q2 + mid) / 3
        i = int(np.argmax(score))
        frac = 0.0
        if 0 < i < len(score) - 1:
            a, b_, cc = score[i - 1], score[i], score[i + 1]
            den = a - 2 * b_ + cc
            if den < 0:
                frac = float(np.clip(0.5 * (a - cc) / den, -0.5, 0.5))
        best = float(score[i])
        if best >= self.track_threshold:
            err = (c + deltas[i] + frac) - s
            s = s + err if abs(err) > 1.5 else s + self.TIMING_LOOP_GAIN * err
            self._next = s
            self._misses = 0
        else:
            self._misses += 1
        mids = {B.SDB: my[i], B.NDB_FULL: mn[i], B.NDB_HALF: mp[i]}
        kind = max(mids, key=mids.get)
        if mids[kind] < self.track_threshold:
            kind = None

        # Interpolate symbols -1..254 at the (refined) timing, detect differentially.
        t = (s - base) + SPS * np.arange(-1, B.SYMBOLS_PER_BURST)
        i0 = np.floor(t).astype(int)
        f = (t - i0).astype(np.float32)
        v = seg[i0] * (1 - f) + seg[i0 + 1] * f
        dd = v[1:] * np.conj(v[:-1])
        mag = np.abs(dd)
        z = np.sum(dd ** 4 / np.maximum(mag ** 3, 1e-20))
        w = float(np.angle(-z) / 4)
        dn = dd * np.exp(-1j * w).astype(np.complex64) / max(float(np.mean(mag)), 1e-20)
        soft = np.empty(B.BITS_PER_BURST, np.float32)
        soft[0::2] = dn.imag
        soft[1::2] = dn.real
        ideal = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.round((np.angle(dn) - np.pi / 4) / (np.pi / 2))))
        mer = 10 * np.log10(1.0 / max(float(np.mean(np.abs(dn - ideal) ** 2)), 1e-6))
        if best >= self.track_threshold:
            self.cfo += self.CFO_LOOP_GAIN * w * B.SYMBOL_RATE / (2 * np.pi)
        level = 10 * np.log10(max(float(np.mean(np.abs(v) ** 2)), 1e-20))
        sym = v.astype(np.complex64)
        return Burst(start=s, seq=self._seq, epoch=self.epoch, kind=kind, soft=soft, score=best,
                     mer_db=float(mer), cfo_hz=float(self.cfo + w * B.SYMBOL_RATE / (2 * np.pi)),
                     level_db=float(level), symbols=sym)
