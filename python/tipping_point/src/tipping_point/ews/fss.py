"""Finite-size scaling: scaled temperature, power-law fits at Tc with bootstrap intervals, and the
pass / fail / inconclusive verdicts a pre-registration is judged by.

scaled_temperature  eps_L = L (T - Tc) / Tc: equal eps_L is the same place in scaled units
power_law           least squares of log|I| on log L, optionally with a 1/L correction term
fit_exponent        kappa of I(L) ~ L^kappa with a CI from bootstrapping seeds, then replicas,
                    independently at each L
peak_position       where along a ramp each replica's indicator reaches its extremum in its
                    expected direction, in eps_L, inside a search band (#98)
inverse_nu          a peak that obeys FSS sits at a fixed eps_L, so T_peak - Tc ~ L^(-1/nu) with
                    1/nu = 1 - d log|eps_peak| / d log L; an indicator merely monotone in T
                    peaks at a fixed T instead, eps_peak ~ L and 1/nu = 0 (#19's risk)
peak_convergence    1/nu from the median peak eps_L per L, with its bootstrap CI
match               fail when the CI lies wholly outside expected +- tolerance; otherwise
                    inconclusive when the CI is wider than the tolerance band (too weak to
                    fail), else pass when the estimate is within tolerance
excludes            pass when the CI excludes a value; when it holds it, inconclusive if the CI is
                    wider than the tolerance band, else fail
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..ising.exact import TC
from .scoring import Grouped, hierarchical_bootstrap, oriented, percentile_interval

PASS, FAIL, INCONCLUSIVE = "pass", "fail", "inconclusive"


def scaled_temperature(T: npt.ArrayLike, L: npt.ArrayLike, Tc: float = TC) -> np.ndarray:
    return np.asarray(L) * (np.asarray(T, dtype=np.float64) - Tc) / Tc


def temperature(eps_L: npt.ArrayLike, L: npt.ArrayLike, Tc: float = TC) -> np.ndarray:
    return Tc * (1.0 + np.asarray(eps_L, dtype=np.float64) / np.asarray(L))


def power_law(L: Sequence[int], values: np.ndarray, correction: bool = False) -> np.ndarray:
    """kappa of log|I| = kappa log L + c (+ d / L) by least squares; values (sizes, ...) give one
    kappa per trailing column. NaN where a value is 0 or NaN, or with too few sizes."""
    sizes = np.asarray(L, dtype=np.float64)
    y = np.log(np.abs(np.asarray(values, dtype=np.float64)))
    columns = [np.log(sizes), np.ones_like(sizes)] + ([1.0 / sizes] if correction else [])
    design = np.stack(columns, axis=1)
    flat = y.reshape(len(sizes), -1)
    kappa = np.full(flat.shape[1], np.nan)
    good = np.isfinite(flat).all(axis=0)
    if len(sizes) >= design.shape[1] and good.any():
        kappa[good] = np.linalg.lstsq(design, flat[:, good], rcond=None)[0][0]
    return kappa.reshape(y.shape[1:])


@dataclass(frozen=True)
class Exponent:
    kappa: float
    ci: tuple[float, float]
    kappa_corrected: float  # with a 1/L term: the finite-size robustness check (NaN below 3 sizes)
    values: dict[int, float]  # I(L) from all the data
    draws: int = 0  # bootstrap resamples drawn for ci
    draws_used: float = float("nan")  # the share of them with a finite exponent, ci's basis


def fit_exponent(
    samples: Mapping[int, Grouped],
    statistic: Callable[[int, np.ndarray], float],
    n: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> Exponent:
    """samples[L] holds per-seed arrays with replicas along axis 0; statistic(L, array) is I(L)."""
    sizes = sorted(samples)
    values = {L: float(statistic(L, np.concatenate(samples[L]))) for L in sizes}
    point = np.array([values[L] for L in sizes])
    draws = np.stack(
        [
            hierarchical_bootstrap(lambda a, L=L: statistic(L, a), [samples[L]], n, rng)
            for L in sizes
        ]
    )
    kappas = power_law(sizes, draws)
    return Exponent(
        kappa=float(power_law(sizes, point)),
        ci=percentile_interval(kappas, level),
        kappa_corrected=float(power_law(sizes, point, correction=True)),
        values=values,
        draws=n,
        draws_used=float(np.mean(np.isfinite(kappas))),
    )


def peak_position(
    values: np.ndarray, eps: np.ndarray, band: tuple[float, float], direction: str
) -> np.ndarray:
    """Per row of values (rows, time): the eps_L of its extremum in the expected direction (the
    maximum for "greater", the minimum for "less") over the samples whose eps_L lies in
    band = (top, bottom). NaN where that extremum is the band's first or last sample, since the
    true one may lie beyond it, or where the row has no value inside: the peak was not located.
    A NaN value is never the extremum."""
    eps = np.asarray(eps, dtype=np.float64)
    inside = np.flatnonzero((eps <= band[0]) & (eps >= band[1]))
    v = oriented(np.atleast_2d(values)[:, inside], direction)
    v = np.where(np.isnan(v), -np.inf, v)
    k = np.argmax(v, axis=1)
    lost = (k == 0) | (k == len(inside) - 1) | np.isneginf(v).all(axis=1)
    return np.where(lost, np.nan, eps[inside][k])


def inverse_nu(L: Sequence[int], eps: np.ndarray) -> np.ndarray:
    """1/nu = 1 - slope of log|eps_L| on log L for a position eps (sizes, ...). NaN unless the
    position lies on one side of Tc at every size: one that changes side has no power law."""
    eps = np.asarray(eps, dtype=np.float64)
    one_side = (eps > 0).all(axis=0) | (eps < 0).all(axis=0)
    return 1.0 - power_law(L, np.where(one_side, eps, np.nan))


def peak_convergence(
    samples: Mapping[int, Grouped], n: int, rng: np.random.Generator, level: float = 0.95
) -> Exponent:
    """1/nu from the median peak eps_L at each L (peaks not located are NaN and left out), with a
    CI from bootstrapping seeds, then replicas, at each L. A draw whose medians lie on both sides
    of Tc, or that drew no located peak at some L, has no 1/nu: the CI rests on the other draws,
    and draws_used is their share."""
    sizes = sorted(samples)
    median = lambda eps: float(np.nanmedian(eps)) if np.isfinite(eps).any() else np.nan  # noqa: E731
    point = np.array([median(np.concatenate(samples[L])) for L in sizes])
    draws = inverse_nu(
        sizes, np.stack([hierarchical_bootstrap(median, [samples[L]], n, rng) for L in sizes])
    )
    return Exponent(
        kappa=float(inverse_nu(sizes, point)),
        ci=percentile_interval(draws, level),
        kappa_corrected=float("nan"),
        values={},
        draws=n,
        draws_used=float(np.mean(np.isfinite(draws))),
    )


def _width(ci: tuple[float, float]) -> float:
    return (ci[1] - ci[0]) / 2


def match(estimate: float, ci: tuple[float, float], expected: float, tolerance: float) -> str:
    if not np.isfinite([estimate, *ci]).all():
        return INCONCLUSIVE
    if ci[1] < expected - tolerance or ci[0] > expected + tolerance:
        return FAIL  # the data exclude the whole tolerance band, however wide the CI
    if _width(ci) > tolerance:
        return INCONCLUSIVE
    return PASS if abs(estimate - expected) <= tolerance else FAIL


def excludes(ci: tuple[float, float], value: float, tolerance: float) -> str:
    if not np.isfinite(ci).all():
        return INCONCLUSIVE
    if ci[0] > value or ci[1] < value:
        return PASS
    return INCONCLUSIVE if _width(ci) > tolerance else FAIL


def combine(*verdicts: str) -> str:
    """fail if any fails, else inconclusive if any is, else pass."""
    if FAIL in verdicts:
        return FAIL
    return INCONCLUSIVE if INCONCLUSIVE in verdicts else PASS
