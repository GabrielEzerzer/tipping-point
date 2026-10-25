"""Paired margins of a trained checkpoint over its untrained twin (#28): "learned" is claimed only
as a margin over the step-0 twin, judged across model seeds.

Per model seed s, a scale-free quantity Q measured on the trained checkpoint and on its twin
gives the paired margin

    delta_s = Q(trained_s) - Q(twin_s)

The CI is a percentile bootstrap of the mean delta over model seeds. Given data-bootstrap draws of
Q, it is nested: each resampled seed also contributes one of its draws, picked at random, so the
CI holds the data's uncertainty within a seed beside the spread between seeds. The draws must be
paired: draw b of the trained checkpoint and draw b of its twin come from one resample of the
data (the caller draws both from one RNG seed per quantity), so their difference is the model's
alone.

The verdict passes when the CI excludes 0 in the predicted direction and the sign test holds:
every seed's margin carries the predicted sign. Otherwise it follows fss.excludes: fail when the
CI lies wholly on the wrong side of 0, or holds 0 within the tolerance band (precisely no margin);
inconclusive when it holds 0 and is wider. A margin of the wrong sign on any seed fails the sign
test, and with it the verdict (fss.combine); an undefined (NaN) margin makes both inconclusive.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from . import fss
from .scoring import oriented, percentile_interval


def paired(
    trained: npt.ArrayLike,
    twin: npt.ArrayLike,
    direction: str,
    tolerance: float,
    rng: np.random.Generator,
    trained_draws: npt.ArrayLike | None = None,
    twin_draws: npt.ArrayLike | None = None,
    n: int = 2_000,
    level: float = 0.95,
) -> dict[str, Any]:
    """One row for Q: trained and twin (seeds,), seed s at index s in both, with optional draws
    (seeds, draws), aligned. direction is the predicted sign of the margin, "greater" (trained
    above twin) or "less"; tolerance, in Q's units, is the band fss.excludes judges a CI by."""
    q1, q0 = np.asarray(trained, dtype=np.float64), np.asarray(twin, dtype=np.float64)
    if q1.ndim != 1 or q1.shape != q0.shape or len(q1) < 2:
        raise ValueError(
            f"trained and twin need one value per model seed, at least two seeds: "
            f"got shapes {q1.shape} and {q0.shape}"
        )
    delta = q1 - q0
    signed = oriented(delta, direction)  # > 0 where the prediction holds
    seeds = len(delta)
    if (trained_draws is None) != (twin_draws is None):
        raise ValueError("data draws come in pairs, the trained checkpoint's and its twin's")
    picks = rng.integers(0, seeds, (n, seeds))
    draws = 0
    if trained_draws is None:
        means = delta[picks].mean(1)
    else:
        d1, d0 = np.asarray(trained_draws, np.float64), np.asarray(twin_draws, np.float64)
        if d1.shape != d0.shape or d1.ndim != 2 or d1.shape[0] != seeds or d1.shape[1] == 0:
            raise ValueError(
                f"draws must be aligned (seeds, draws) arrays for {seeds} seeds: "
                f"got shapes {d1.shape} and {d0.shape}"
            )
        draws = d1.shape[1]
        means = (d1 - d0)[picks, rng.integers(0, draws, (n, seeds))].mean(1)
    # an undefined seed leaves the mean undefined, rather than resampled away
    lo, hi = percentile_interval(means, level) if np.isfinite(delta).all() else (np.nan, np.nan)
    low, high = np.sort(oriented(np.array([lo, hi]), direction))  # turned: the prediction is up
    ci = fss.FAIL if high < 0 else fss.excludes((float(low), float(high)), 0.0, tolerance)
    if np.any(signed <= 0):
        signs = fss.FAIL
    else:
        signs = fss.PASS if np.isfinite(signed).all() else fss.INCONCLUSIVE
    return {
        "direction": direction,
        "n_seeds": seeds,
        "n_draws": draws,
        "margins": delta.tolist(),
        "margin": float(delta.mean()),
        "lo": lo,
        "hi": hi,
        "level": level,
        "n_bootstrap": n,
        "tolerance": tolerance,
        "sign_test": signs,
        "verdict": fss.combine(ci, signs),
    }
