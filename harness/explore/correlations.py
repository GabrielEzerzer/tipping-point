"""What a checkpoint's readouts follow (exploratory, not pre-registered, --latents only): the
Pearson correlation of surprise and pc_1..pc_8 with |m| and with e per site, from the staircases'
saved frames. A PCA's sign is arbitrary, so read a pc's r by its size.

  per plateau  r over the frames of one temperature, pooled over seeds and replicas: what a column
               says about a frame's fluctuation at fixed T, which is what FSS and "is pc_1 |m|-like"
               are about. The FSS plateau (T = 2.269231) is the headline, every plateau beside it
  pooled       r over all plateaus at equal weight (one burst in 8 at the FSS plateau too, as
          tc-plus
               keeps elsewhere): most of it is the shared trend in T
  across T     r over the plateaus of the plateau means, and of L^2 Var_T against the classic
               L^2 Var_T(|m|), the chi alignment #27 proposed, computed on the same kept frames
  partial      at the FSS plateau, a column's r with |m| given e and with e given |m|: which of the
           two
               it follows, since |m| and e correlate strongly at fixed T themselves

Intervals: percentile, one joint hierarchical bootstrap (seeds, then replicas) over the per-replica
sums at every plateau. A plateau's mean is re-estimated in each draw, so the interval holds it too.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tipping_point.ews import ising

from explore import common

TARGETS = ("abs_m", "energy")
LEVELS = (0.025, 0.975)
BURST = 8  # frames per burst, the staircases'
EVERY = 8  # tc-plus keeps one burst in 8 away from the FSS plateau


def _sums(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per replica, over its frames: [n, Sx, Sy, Sxx, Syy, Sxy]."""
    n = np.full(len(x), x.shape[1], dtype=np.float64)
    return np.stack([n, x.sum(1), y.sum(1), (x * x).sum(1), (y * y).sum(1), (x * y).sum(1)], axis=1)


def _pearson(s: np.ndarray) -> np.ndarray:
    n, sx, sy, sxx, syy, sxy = np.moveaxis(s, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))


def _uniform(s: ising.Series, smallest: int) -> ising.Series:
    """A plateau's frames at one burst in EVERY, where its table holds more than the others'."""
    if len(s.steps) <= smallest:
        return s
    return s.take(np.flatnonzero((np.arange(len(s.steps)) // BURST) % EVERY == 0))


def resamples(seed_of: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, replicas) multiplicities of a hierarchical bootstrap: seeds, then replicas in each."""
    seeds = np.unique(seed_of)
    rows = [np.flatnonzero(seed_of == s) for s in seeds]
    weights = np.zeros((n, len(seed_of)))
    for i in range(n):
        for pick in rng.integers(0, len(seeds), len(seeds)):
            np.add.at(weights[i], rng.choice(rows[pick], len(rows[pick])), 1.0)
    return weights


def ci(draws: np.ndarray) -> np.ndarray:
    return np.nanquantile(draws, LEVELS, axis=0).T


def _entry(point: Any, draws: np.ndarray) -> dict[str, Any]:
    lo, hi = np.nanquantile(draws, LEVELS)
    return {"value": float(point), "ci": [float(lo), float(hi)]}


def analyse(args: Any, inputs: ising.Inputs) -> dict[str, Any]:
    """The tables above from an L's staircases, read through the adapter."""
    names = [*common.ALL_COLUMNS, "abs_m"]
    plateaus = [
        ising.plateaus(common.allowed(r), "matched", args.latents)[0] for r in inputs.staircases
    ]
    temps = sorted(set.intersection(*(set(p) for p in plateaus)), reverse=True)
    T_fss = next(t for t in temps if ising._close(t, 2.269231))
    smallest = min(len(p[T].steps) for p in plateaus for T in temps)
    pairs = [(c, t) for c in names for t in TARGETS if c != t]
    seed_of = np.concatenate(
        [
            np.full(len(p[temps[0]].replicas), r.seed)
            for p, r in zip(plateaus, inputs.staircases, strict=True)
        ]
    )
    # (replicas, plateaus, pairs, 6) of the whole plateaus, and of the uniform frames
    whole = np.empty((len(seed_of), len(temps), len(pairs), 6))
    even = np.empty_like(whole)
    for k, T in enumerate(temps):
        rows = 0
        for p in plateaus:
            s = p[T]
            thin = _uniform(s, smallest)
            count = len(s.replicas)
            for j, (c, t) in enumerate(pairs):
                whole[rows : rows + count, k, j] = _sums(s.observable(c), s.observable(t))
                even[rows : rows + count, k, j] = _sums(thin.observable(c), thin.observable(t))
            rows += count
    rng = np.random.default_rng(args.seed)
    weights = resamples(seed_of, args.bootstrap, rng)

    def weighted(sums: np.ndarray) -> np.ndarray:
        """(draws, ...) of the sums over replicas under each draw's multiplicities."""
        shape = sums.shape[1:]
        return (weights @ sums.reshape(len(seed_of), -1)).reshape(args.bootstrap, *shape)

    plateau_point = _pearson(whole.sum(0))  # (plateaus, pairs)
    mean_draws = weighted(whole)
    plateau_draws = _pearson(mean_draws)
    pooled_point = _pearson(even.sum((0, 1)))  # (pairs,)
    pooled_draws = _pearson(weighted(even.sum(1)))
    means = whole.sum(0)  # (plateaus, pairs, 6)

    def across(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pearson r of two curves over the plateaus, last axis of each."""
        da, db = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
        return (da * db).sum(-1) / np.sqrt((da * da).sum(-1) * (db * db).sum(-1))

    def level(s: np.ndarray, which: int) -> np.ndarray:
        return s[..., which] / s[..., 0]

    def variance(s: np.ndarray, which: int) -> np.ndarray:
        square = 3 if which == 1 else 4
        return args.L**2 * (s[..., square] - s[..., which] ** 2 / s[..., 0]) / (s[..., 0] - 1)

    fss_at = temps.index(T_fss)
    m_with_e = pairs.index(("abs_m", "energy"))
    out: dict[str, Any] = {
        "T": temps,
        "eps_L": [float(inputs.staircases[0].eps(T)) for T in temps],
        "T_fss": T_fss,
        "bootstrap": args.bootstrap,
        "ci_method": "percentile, joint hierarchical bootstrap: seeds, then replicas",
        "abs_m_with_energy_at_fss": _entry(
            plateau_point[fss_at, m_with_e], plateau_draws[:, fss_at, m_with_e]
        ),
        "columns": {},
    }
    for c in common.ALL_COLUMNS:
        out["columns"][c] = {}
        for t in TARGETS:
            j = pairs.index((c, t))
            other = pairs.index((c, "energy" if t == "abs_m" else "abs_m"))

            def partial(p: np.ndarray, j: int = j, other: int = other) -> np.ndarray:
                r_x, r_z, r_m = p[..., j], p[..., other], p[..., m_with_e]
                return (r_x - r_z * r_m) / np.sqrt((1 - r_z**2) * (1 - r_m**2))

            out["columns"][c][t] = {
                "at_fss_plateau": _entry(plateau_point[fss_at, j], plateau_draws[:, fss_at, j]),
                "partial_at_fss_plateau": _entry(
                    partial(plateau_point[fss_at]), partial(plateau_draws[:, fss_at])
                ),
                "per_plateau": plateau_point[:, j],
                "per_plateau_ci": ci(plateau_draws[:, :, j]),
                "pooled": _entry(pooled_point[j], pooled_draws[:, j]),
                "means_across_T": _entry(
                    across(level(means[:, j], 1), level(means[:, j], 2)),
                    across(level(mean_draws[:, :, j], 1), level(mean_draws[:, :, j], 2)),
                ),
                "l2_var_across_T": _entry(
                    across(variance(means[:, j], 1), variance(means[:, j], 2)),
                    across(variance(mean_draws[:, :, j], 1), variance(mean_draws[:, :, j], 2)),
                ),
            }
    return out


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args(argv)
    if not args.latents:
        raise SystemExit("correlations reads a latents table: name it with --latents")
    common.be_nice()
    clock = common.Clock()
    doc = common.make_doc(args)
    pilot, inputs = common.load(args, doc)
    out = analyse(args, inputs)
    for c in common.JUDGED_COLUMNS:
        for t in TARGETS:
            e = out["columns"][c][t]["at_fss_plateau"]
            clock(
                f"{c:<9} with {t:<7} at the FSS plateau: r {e['value']:+.3f} {np.round(e['ci'], 3)}"
            )
    common.save(
        common.out_dir(args) / "correlations.json",
        {"provenance": common.provenance(args, pilot), "runs": common.run_ids(inputs)} | out,
    )
