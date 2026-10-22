"""Surrogate series: draws of what a score looks like when nothing is coming, each from an explicit
np.random.Generator. A surrogate imitates the residuals it is made from and goes through the same
indicator and trend steps as they do (scoring.surrogate_test).

phase_randomised  Theiler et al. 1992: every Fourier amplitude kept, every phase drawn uniformly,
                  so the periodogram, mean and variance survive exactly; no model to choose
ar1_fitted        Dakos et al. 2012, Bury et al. 2021: an AR(1) with the series' lag-1
                  least-squares coefficient, innovation spread and mean, started stationary
block_bootstrap   moving blocks, for confidence intervals only: blocks must be >> tau_c
Inputs are (..., time); outputs are (..., n, time), n surrogates per input row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHI_MAX = 0.9999  # a fitted coefficient is kept inside the stationary range


def phase_randomised(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    length = x.shape[-1]
    spectrum = np.fft.rfft(x, axis=-1)[..., None, :]
    phases = rng.uniform(0.0, 2 * np.pi, (*x.shape[:-1], n, spectrum.shape[-1]))
    phases[..., 0] = 0.0  # the mean
    if length % 2 == 0:
        phases[..., -1] = 0.0  # the Nyquist term of a real series is real
    return np.fft.irfft(spectrum * np.exp(1j * phases), length, axis=-1)


@dataclass(frozen=True)
class AR1:
    """x_t - mean = phi (x_(t-1) - mean) + sigma e_t, one parameter set per input row."""

    phi: np.ndarray
    sigma: np.ndarray
    mean: np.ndarray


def fit_ar1(x: np.ndarray) -> AR1:
    """Least squares of each centred x_(t+1) on x_t; phi is clipped to +-PHI_MAX."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1)
    y = x - mean[..., None]
    a, b = y[..., :-1], y[..., 1:]
    phi = np.clip(np.sum(a * b, axis=-1) / np.sum(a * a, axis=-1), -PHI_MAX, PHI_MAX)
    innovations = b - phi[..., None] * a
    return AR1(phi, innovations.std(axis=-1, ddof=1), mean)


def ar1(model: AR1, length: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n stationary AR(1) series per parameter set: x_0 ~ N(mean, sigma^2 / (1 - phi^2))."""
    phi, sigma, mean = (
        np.asarray(v, dtype=np.float64)[..., None] for v in (model.phi, model.sigma, model.mean)
    )
    shape = np.broadcast_shapes(phi.shape[:-1], sigma.shape[:-1], mean.shape[:-1])
    e = rng.standard_normal((length, *shape, n))  # time first, so each step is contiguous
    p = np.broadcast_to(phi, (*shape, n))
    x = np.empty_like(e)
    x[0] = e[0] / np.sqrt(1 - p**2)
    for t in range(1, length):
        x[t] = p * x[t - 1] + e[t]
    return mean[..., None] + sigma[..., None] * np.moveaxis(x, 0, -1)


def ar1_fitted(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ar1(fit_ar1(x), x.shape[-1], n, rng)


def block_bootstrap(x: np.ndarray, block: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Each row rebuilt n times from blocks of `block` consecutive samples starting at uniform
    random points, joined end to end and cut to the row's length."""
    x = np.asarray(x, dtype=np.float64)
    length = x.shape[-1]
    if not 1 <= block <= length:
        raise ValueError(f"block must lie in [1, {length}], got {block}")
    count = -(-length // block)
    starts = rng.integers(0, length - block + 1, (*x.shape[:-1], n, count))
    index = (starts[..., None] + np.arange(block)).reshape(*starts.shape[:-1], count * block)
    return np.take_along_axis(x[..., None, :], index[..., :length], axis=-1)


KINDS = {"phase": phase_randomised, "ar1": ar1_fitted}  # the surrogate tests a prereg may name
