"""Integrated autocorrelation time: Sokal's windowed estimator, cross-checked by binning.

Convention: tau = 1 + 2 sum_{t>=1} rho(t), so a series of N samples holds N / tau independent ones.
Each row of an input array is one replica's time series; replicas are averaged, never chained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tau:
    tau: float  # Sokal estimate 1 + 2 sum_{t<=window} rho(t); a floor when no window was found
    window: int  # first lag W with W >= c * tau(W)
    binning: float  # binning ratio at the largest block size: the cross-check
    n: int  # samples per series
    reliable: bool  # a window was found, tau > 0 and n >= 50 tau


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """rho(t) for t = 0..n-1, computed per row by FFT, then averaged over rows."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    x = x[x.std(axis=1) > 0]  # a frozen series carries no correlation information
    if len(x) == 0:
        raise ValueError("every series is constant")
    n = x.shape[1]
    size = 1 << (2 * n - 1).bit_length()  # zero-padding to >= 2n stops the FFT wrapping around
    f = np.fft.rfft(x - x.mean(axis=1, keepdims=True), n=size, axis=1)
    acov = np.fft.irfft(f * f.conj(), n=size, axis=1)[:, :n]
    return (acov / acov[:, :1]).mean(axis=0)


def sokal(rho: np.ndarray, c: float = 5.0) -> tuple[float, int, bool]:
    """Sum rho up to the first window W >= c * tau(W): late lags add noise, not signal.

    Windows are sought below n/2 only. Subtracting the mean forces the full sum to zero
    (tau(n-1) = 0), which would fake a window, and a tau of 0, in a series far too short to have
    one. With no window the series cannot say; the largest running sum is returned as a floor.
    """
    taus = 2 * np.cumsum(rho[: max(1, len(rho) // 2)]) - 1  # tau(W) = 1 + 2 sum_{t=1..W} rho(t)
    found = np.arange(len(taus)) >= c * taus
    if not found.any():
        return float(taus.max()), len(taus) - 1, False
    window = int(np.argmax(found))
    return float(taus[window]), window, True


def binning(x: np.ndarray, min_blocks: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """R(b) = b * Var(means of blocks of b) / Var(x) for b = 1, 2, 4, ...; it plateaus at tau."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    x = x[x.std(axis=1) > 0]
    n = x.shape[1]
    sizes, ratios = [], []
    b = 1
    while n // b >= min_blocks:
        means = x[:, : n // b * b].reshape(len(x), n // b, b).mean(axis=2)
        ratios.append(float(np.mean(b * means.var(axis=1, ddof=1) / x.var(axis=1, ddof=1))))
        sizes.append(b)
        b *= 2
    return np.array(sizes), np.array(ratios)


def estimate(x: np.ndarray, c: float = 5.0) -> Tau:
    """tau of the replica series in x (one row each), with the binning cross-check."""
    x = np.atleast_2d(x)
    tau, window, found = sokal(autocorrelation(x), c)
    _, ratios = binning(x)
    n = x.shape[1]
    plateau = float(ratios[-1]) if len(ratios) else float("nan")
    return Tau(tau, window, plateau, n, found and 0 < tau <= n / 50)
