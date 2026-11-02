"""Paired margins of a trained checkpoint over its untrained twin (#28): "learned" is claimed only
as a margin over the step-0 twin, judged across model seeds.

Per model seed s, a scale-free quantity Q measured on the trained checkpoint and on its twin
gives the paired margin

    delta_s = Q(trained_s) - Q(twin_s)

The CI is a percentile bootstrap of the mean delta over model seeds. Given data-bootstrap draws of
Q, it is nested: each replicate resamples the seeds and takes one data resample, draw column b,
for all of them, so the CI holds the data's uncertainty beside the spread between seeds. Column b
must be the same resample of the data for every checkpoint of the quantity, trained and twin
alike (the caller draws them all from one RNG seed per quantity): every checkpoint is scored on
the same frames, so the data's noise is common to the seeds and never averages out over them,
and within a column the trained-twin difference is the model's alone.

With five seeds the percentile interval covers the mean margin only about 83% of the time even
without data noise (the mean of five draws is t-distributed with four degrees of freedom). That
is inherent to five seeds, which is why the verdict also asks every seed's margin to carry the
predicted sign.

The verdict passes when the CI excludes 0 in the predicted direction and the sign test holds:
every seed's margin carries the predicted sign. Otherwise it follows fss.excludes: fail when the
CI lies wholly on the wrong side of 0, or holds 0 within the tolerance band (precisely no margin);
inconclusive when it holds 0 and is wider. A margin of the wrong sign on any seed fails the sign
test, and with it the verdict (fss.combine); an undefined (NaN) margin makes both inconclusive.

table() is a latent file's margins table, one row per quantity and role. Three rows are judged, each
over the primary checkpoints: the energy probe's R^2 and its selectivity, the mean over L of each
checkpoint's value, by the verdict above, whose sign test under scoring.margins.sizes = "every"
also asks each seed's margin at every L to carry the predicted sign; and the surprise FSS
verdicts, by the rule hypothesis learned-beyond-random states for them, at least one surprise
statistic passing for at least 4 of 5 trained checkpoints and for none of their twins (an
inconclusive twin is no pass; inconclusive when the trained checkpoints that are inconclusive could
still reach the 4), a row a file with no surprise statistic of role test does not have. Every
other row is descriptive and has no verdict: each L's value, AUC (parity, saturated by
construction), the chi alignment and all of an ablation's.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from . import fss
from .ising import PRIMARY, latents_name
from .prereg import LEARNED
from .scoring import oriented, percentile_interval

ALL = "all"  # a Quantity's scope when it is the mean over L, not one L's
JUDGED = ("energy_r2", "energy_selectivity")
SURPRISE_FSS = "surprise_fss"
SURPRISE_SHARE = 4 / 5  # of the trained checkpoints, for a surprise statistic to count


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
    per_size: npt.ArrayLike = (),
) -> dict[str, Any]:
    """One row for Q: trained and twin (seeds,), seed s at index s in both, with optional draws
    (seeds, draws), aligned: column b of both arrays must be one resample of the data, the same
    for every checkpoint of the quantity, trained and twin alike. direction is the predicted sign
    of the margin, "greater" (trained above twin) or "less"; tolerance, in Q's units, is the band
    fss.excludes judges a CI by. per_size, each L's margins seed by seed (scoring.margins.sizes =
    "every"), joins the sign test: every one of them must carry the predicted sign too."""
    q1, q0 = np.asarray(trained, dtype=np.float64), np.asarray(twin, dtype=np.float64)
    if q1.ndim != 1 or q1.shape != q0.shape or len(q1) < 2:
        raise ValueError(
            f"trained and twin need one value per model seed, at least two seeds: "
            f"got shapes {q1.shape} and {q0.shape}"
        )
    delta = q1 - q0
    signed = oriented(np.append(delta, per_size), direction)  # > 0 where the prediction holds
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
        # one data resample per replicate, shared by its seeds: they were all scored on one dataset
        means = (d1 - d0)[picks, rng.integers(0, draws, (n, 1))].mean(1)
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


@dataclass(frozen=True)
class Quantity:
    """One scale-free quantity per checkpoint, the way the margins table reads it."""

    name: str
    scope: str  # ALL, or one L's "L=64"
    values: dict[str, float]  # checkpoint -> Q
    draws: dict[str, np.ndarray] = field(default_factory=dict)  # checkpoint -> Q per data resample,
    # column b the same resample for every checkpoint; empty where Q has no data bootstrap
    reference: float = float("nan")  # the baseline's value, where the quantity has a classic one


def _surprise_verdict(trained: list[str], twin: list[str]) -> str:
    """The rule for one statistic, from its verdicts over the trained checkpoints and the twins."""
    need = math.ceil(SURPRISE_SHARE * len(trained))
    passes = trained.count(fss.PASS)
    if fss.PASS in twin:
        return fss.FAIL
    if passes >= need:
        return fss.PASS
    return fss.INCONCLUSIVE if passes + trained.count(fss.INCONCLUSIVE) >= need else fss.FAIL


def table(
    doc: dict[str, Any],
    quantities: Iterable[Quantity],
    fss_rows: Sequence[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """The margins table of a latent file: see the module docstring. A role with one checkpoint has
    no spread between seeds to bootstrap over, and no rows."""
    sc, steps = doc["scoring"], int(doc["latents"]["steps"])
    tolerance = float(sc["margins"]["tolerance"])
    every = sc["margins"]["sizes"] == "every"
    n, level = int(sc["bootstrap"]), float(sc["ci_level"])
    direction = next(h["direction"] for h in doc["hypotheses"] if h["name"] == LEARNED)
    statistics = [
        f"{s['observable']}_{s['indicator']}"
        for s in doc["fss"]["dynamic"]["statistics"]
        if s["observable"] == "surprise" and s["role"] == "test"
    ]
    verdicts = {(r["checkpoint"], r["quantity"]): r["verdict"] for r in fss_rows}
    roles: dict[str, list[dict[str, Any]]] = {}
    for entry in doc["latents"]["checkpoints"]:
        roles.setdefault(entry["role"], []).append(entry)
    measured = list(quantities)
    rows = []
    for role, entries in roles.items():
        if len(entries) < 2:
            continue
        trained = [latents_name(entry["training"], steps) for entry in entries]
        twin = [latents_name(entry["training"], 0) for entry in entries]
        head = {
            "role": role,
            "checkpoint": ";".join(trained),  # what the row reads, in the order of its margins
            "twin": ";".join(twin),
            "model_seeds": [entry["model_seed"] for entry in entries],
        }
        for q in measured:
            draws = ()
            if q.draws:
                draws = tuple(
                    np.stack([q.draws[name] for name in side]) for side in (trained, twin)
                )
            judged = role == PRIMARY and q.scope == ALL and q.name in JUDGED
            per_size = [
                [p.values[a] - p.values[b] for a, b in zip(trained, twin, strict=True)]
                for p in measured
                if every and judged and p.name == q.name and p.scope != ALL
            ]
            found = paired(
                [q.values[name] for name in trained],
                [q.values[name] for name in twin],
                direction,
                tolerance,
                rng,
                *draws,
                n=n,
                level=level,
                per_size=per_size,
            )
            rows.append(_row(q.name, q.scope, judged, head, found, reference=q.reference))
        settled = []
        for statistic in statistics:
            key = f"inverse_nu_{statistic}"
            says = [verdicts[name, key] for name in trained], [verdicts[name, key] for name in twin]
            passed = [[float(v == fss.PASS) for v in side] for side in says]
            found = paired(passed[0], passed[1], direction, tolerance, rng, n=n, level=level)
            found |= {"sign_test": "", "verdict": _surprise_verdict(*says)}
            counts = {
                "trained_passes": says[0].count(fss.PASS),
                "twin_passes": says[1].count(fss.PASS),
            }
            rows.append(_row(f"{SURPRISE_FSS}_{statistic}", ALL, False, head, found, **counts))
            settled.append((found, counts))
        if role == PRIMARY and settled:
            order = (fss.PASS, fss.INCONCLUSIVE, fss.FAIL)
            found, counts = min(settled, key=lambda pair: order.index(pair[0]["verdict"]))
            rows.append(_row(SURPRISE_FSS, ALL, True, head, found, **counts))
    return rows


def _row(
    quantity: str,
    scope: str,
    judged: bool,
    head: dict[str, Any],
    found: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    """A margins row, every row with every column: a descriptive one keeps its CI, not its
    verdict."""
    return {
        "quantity": quantity,
        "scope": scope,
        "judged": judged,
        **head,
        **found,
        "verdict": found["verdict"] if judged else "",
        "reference": float("nan"),
        "trained_passes": None,
        "twin_passes": None,
        **extra,
    }
