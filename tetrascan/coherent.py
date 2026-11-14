"""Coherent soft-output detection of pi/4-DQPSK bursts.

pi/4-DQPSK phase changes are pi/4 + d*pi/2 with d = 0 (dibit 00), 1 (01), 2 (11), 3 (10).
Removing the deterministic pi/4 rotation per symbol (u_k = r_k * exp(-j*k*pi/4)) leaves
a QPSK signal whose constellation index m_k evolves as m_k = m_{k-1} + d_k (mod 4).

1. Carrier phase: the 4th power of u_k removes the modulation; a sliding average over
   ``window`` symbols, unwrapped, gives the phase modulo pi/2 (the pi/2 ambiguity does not
   matter because the information is in the phase changes).
2. A max-log BCJR (forward-backward) over the 4 states m_k turns the phase-corrected
   symbols into exact max-log LLRs for both bits of every dibit. This is the optimum
   coherent detector for differentially encoded QPSK and gains roughly 2 dB over the
   classic differential detector r_k * conj(r_{k-1}).

Soft bits follow the package convention (positive = bit 0) and are scaled as LLRs.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_STATES = np.arange(4)
_PREV = (_STATES[:, None] - _STATES[None, :]) % 4  # [m, d] -> previous state
_NEXT = (_STATES[:, None] + _STATES[None, :]) % 4  # [m, d] -> next state


def phase_correct(v: np.ndarray, window: int = 33) -> np.ndarray:
    """Remove the pi/4 rotation and the carrier phase. ``v``: (B, N) symbols. Returns QPSK on the axes."""
    v = np.atleast_2d(v)
    k = np.arange(v.shape[1])
    u0 = v * np.exp(-1j * np.pi / 4 * k)
    mag = np.maximum(np.abs(u0), 1e-20)
    z = u0 ** 4 / mag ** 3
    zs = ndimage.uniform_filter1d(z.real, window, axis=1, mode="nearest") + \
        1j * ndimage.uniform_filter1d(z.imag, window, axis=1, mode="nearest")
    theta = np.unwrap(np.angle(zs), axis=1) / 4
    u = u0 * np.exp(-1j * theta)
    return u / np.maximum(np.mean(np.abs(u), axis=1, keepdims=True), 1e-20)


def coherent_soft(v: np.ndarray, window: int = 33) -> np.ndarray:
    """Soft bits (B, 2*(N-1)) from N symbols per burst (the first symbol is the phase reference)."""
    u = phase_correct(v, window)
    b, n = u.shape
    hard = np.exp(1j * np.pi / 2 * np.round(np.angle(u) / (np.pi / 2)))
    sigma2 = np.maximum(np.mean(np.abs(u - hard) ** 2, axis=1), 1e-3)
    scale = (2.0 / sigma2)[:, None, None]
    g = (scale * np.stack([u.real, u.imag, -u.real, -u.imag], axis=-1)).astype(np.float32)  # (B, N, 4)

    alpha = np.empty_like(g)
    alpha[:, 0] = g[:, 0]
    for k in range(1, n):
        a = g[:, k] + alpha[:, k - 1][:, _PREV].max(axis=-1)
        alpha[:, k] = a - a.max(axis=-1, keepdims=True)
    beta = np.zeros_like(g)
    gb = g + beta  # filled backwards below
    for k in range(n - 1, 0, -1):
        gb[:, k] = g[:, k] + beta[:, k]
        bk = gb[:, k][:, _NEXT].max(axis=-1)
        beta[:, k - 1] = bk - bk.max(axis=-1, keepdims=True)
    gb[:, 1:] = g[:, 1:] + beta[:, 1:]
    # transition metric for each d at symbol k = 1..N-1: max over the previous state
    t = alpha[:, :-1, :, None] + gb[:, 1:][:, :, _NEXT]  # (B, N-1, 4 prev, 4 d)
    m = t.max(axis=2)  # (B, N-1, 4)
    soft = np.empty((b, 2 * (n - 1)), np.float32)
    soft[:, 0::2] = np.maximum(m[..., 0], m[..., 1]) - np.maximum(m[..., 2], m[..., 3])
    soft[:, 1::2] = np.maximum(m[..., 0], m[..., 3]) - np.maximum(m[..., 1], m[..., 2])
    return soft
