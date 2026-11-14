"""Generic DSP helpers: RRC pulse, pi/4-DQPSK modulator, streaming channel filter, PSD."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from .burst import SYMBOL_RATE, bits_to_dphi

SPS = 8  # receiver samples per symbol
CHANNEL_RATE = SYMBOL_RATE * SPS  # 144 kHz
ROLLOFF = 0.35


def rrc_taps(sps: int = SPS, alpha: float = ROLLOFF, span: int = 10) -> np.ndarray:
    """Root-raised-cosine FIR, ``span`` symbols long, unit energy."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.empty_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-12:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(4 * alpha * x) - 1) < 1e-9:
            h[i] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                                         + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            h[i] = (np.sin(np.pi * x * (1 - alpha)) + 4 * alpha * x * np.cos(np.pi * x * (1 + alpha))) / (
                np.pi * x * (1 - (4 * alpha * x) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def modulate(bits, sps: int = SPS, phase0: float = 0.0) -> np.ndarray:
    """pi/4-DQPSK with RRC pulse shaping; returns complex baseband at sps*18 kHz, ~unit power."""
    phases = phase0 + np.cumsum(bits_to_dphi(bits))
    up = np.zeros(len(phases) * sps, np.complex128)
    up[::sps] = np.exp(1j * phases)
    h = rrc_taps(sps)
    return (np.convolve(up, h)[len(h) // 2 : len(h) // 2 + len(up)] * np.sqrt(sps)).astype(np.complex64)


def iq_from_cs8(raw: bytes | np.ndarray) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.int8).astype(np.float32) * (1.0 / 128.0)
    return a.view(np.complex64)


class ChannelFilter:
    """Streaming frequency shift + rational resampler: one TETRA channel out of a wideband stream.

    Input at ``fs_in`` (complex), channel centred at ``offset_hz`` from the input centre; output
    at ``fs_out`` (default 144 kHz = 8 samples per symbol), channel at 0 Hz.
    """

    def __init__(self, fs_in: float, offset_hz: float, fs_out: float = CHANNEL_RATE):
        r = Fraction(int(round(fs_out)), int(round(fs_in))).limit_denominator(4096)
        self.up, self.down = r.numerator, r.denominator
        self.fs_in, self.fs_out = fs_in, fs_in * self.up / self.down
        self.offset_hz = offset_hz
        fs_hi = fs_in * self.up
        # Pass the channel (+-13 kHz) and stop at fs_out - 13 kHz, where the first alias of the
        # channel would land. Anything in between aliases outside the channel and is removed
        # later by the RRC matched filter.
        width = self.fs_out - 26e3
        ntaps, beta = signal.kaiserord(70.0, width / (fs_hi / 2))
        ntaps |= 1
        h = signal.firwin(ntaps, (self.fs_out / 2) / (fs_hi / 2), window=("kaiser", beta)) * self.up
        self.h = h.astype(np.float32)
        # Integer decimation: frequency-translating FIR. With w = -2*pi*offset/fs_in,
        #   y[m] = sum_k h[k] x[mD-k] e^{jw(mD-k)} = e^{jwmD} * sum_k (h[k] e^{-jwk}) x[mD-k]
        # so the shift costs one rotation per *output* sample instead of one per input sample.
        self._xlate = self.up == 1 and offset_hz != 0
        if self._xlate:
            w = -2 * np.pi * offset_hz / fs_in
            self.h = (h * np.exp(-1j * w * np.arange(len(h)))).astype(np.complex64)
        self._buf = np.zeros(0, np.complex64)
        self._b0 = 0  # global input index of _buf[0], always a multiple of ``down``
        self._m = 0  # next output index
        self._n = 0  # input samples seen (for the mixer phase)
        self._w = -2 * np.pi * offset_hz / fs_in

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(0, np.complex64)
        if self.offset_hz and not self._xlate:
            ph = (self._w * (self._n + np.arange(len(x), dtype=np.float64))) % (2 * np.pi)
            x = x * np.exp(1j * ph).astype(np.complex64)
        self._n += len(x)
        m_first = self._m
        up, down, L = self.up, self.down, len(self.h)
        buf = np.concatenate([self._buf, x]) if len(self._buf) else x
        b0 = self._b0
        n_end = b0 + len(buf)
        m_last = ((n_end - 1) * up) // down
        out = np.zeros(0, np.complex64)
        if m_last >= self._m:
            z = signal.upfirdn(self.h, buf, up, down)
            i0 = self._m - (b0 * up) // down
            out = z[i0 : i0 + (m_last - self._m + 1)].astype(np.complex64)
            self._m = m_last + 1
        n_min = max(b0, -(-(self._m * down - L + 1) // up))
        new_b0 = (n_min // down) * down
        self._buf = buf[new_b0 - b0 :].copy()
        self._b0 = new_b0
        if self._xlate and len(out):
            ph = (self._w * down * (m_first + np.arange(len(out), dtype=np.float64))) % (2 * np.pi)
            out = out * np.exp(1j * ph).astype(np.complex64)
        return out


def psd(x: np.ndarray, fs: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (Hann, no overlap), fftshifted. Returns (freq offsets Hz, power per bin)."""
    nseg = len(x) // nfft
    if nseg == 0:
        raise ValueError("not enough samples for PSD")
    w = np.hanning(nfft).astype(np.float32)
    segs = x[: nseg * nfft].reshape(nseg, nfft) * w
    p = np.mean(np.abs(np.fft.fft(segs, axis=1)) ** 2, axis=0) / np.sum(w ** 2)
    f = np.fft.fftfreq(nfft, 1 / fs)
    return np.fft.fftshift(f), np.fft.fftshift(p)


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-20))
