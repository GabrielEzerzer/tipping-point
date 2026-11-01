"""The fixed-T null runs alone at one L (exploratory, not pre-registered): what the indicators and
their trend scores look like when nothing is coming, per eps_L and pooled.

For the null blocks (eps_L in {2, 4, 6} x seeds 1-5, measured rows), per column of the mode:
  static   the mean and L^2 Var over each whole block, pooled over replicas and seeds, with a 95% CI
           from the hierarchical bootstrap (seeds, then replicas): the off-Tc scaling check, since
           at fixed eps_L, L^2 Var(|m|) should grow as L^(7/4) and <|m|> fall as L^(-1/8)
  levels   the rolling indicators the headline pipeline computes on each null slice, median and
          5-95%
           over every window (L^2 x a variance)
  scores   s, each indicator's Kendall tau over the slice's scored segment (the AUC negatives):
          median
           and 5-95% per eps_L at the headline cell, and pooled per sensitivity cell
The slice span (window + segment - 1) is the slow ramps'. Runs on a --latents table at the matched
cadence, on |m| at the dense cadence without one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tipping_point.ews import ising, scoring
from tipping_point.ews.trend import INDICATORS, RAW, Pipeline

from explore import common

QUANTILES = (0.05, 0.5, 0.95)


def spread(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    q05, q50, q95 = np.quantile(v, QUANTILES)
    return {"q05": float(q05), "median": float(q50), "q95": float(q95), "n": int(v.size)}


def static_at(groups: list[np.ndarray], L: int, n: int, rng: np.random.Generator) -> dict[str, Any]:
    """The mean and L^2 Var pooled over (replicas, time) rows of each seed's group."""
    sums = [
        np.stack([np.full(len(g), g.shape[1]), g.sum(axis=1), (g * g).sum(axis=1)], axis=1)
        for g in groups
    ]

    def moments(total: np.ndarray) -> tuple[float, float]:
        N, S1, S2 = total
        return S1 / N, L**2 * (S2 - S1 * S1 / N) / (N - 1)

    mean, var = moments(np.asarray(sum(x.sum(axis=0) for x in sums)))
    draws = np.empty((n, 2))
    sizes = [len(g) for g in groups]
    for i in range(n):
        seeds = rng.integers(0, len(sizes), len(sizes))
        draws[i] = moments(
            np.asarray(sum(sums[s][rng.integers(0, sizes[s], sizes[s])].sum(axis=0) for s in seeds))
        )
    return {
        "mean": {"value": mean, "ci": scoring.percentile_interval(draws[:, 0])},
        "l2_var": {"value": var, "ci": scoring.percentile_interval(draws[:, 1])},
    }


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args(argv)
    common.be_nice()
    clock = common.Clock()
    doc = common.make_doc(args)
    pilot, inputs = common.load(args, doc)
    tau_c, L = pilot.tau(args.L), args.L
    ind, sens = doc["indicators"], doc["sensitivity"]
    segment = float(doc["event"]["segment"])
    kind = common.cadence(args)
    prep = ising.prepare(inputs, kind, tau_c, float(ind["thin"]), (segment, 0.0))
    rng = np.random.default_rng(args.seed)
    eps_of = [round(float(run.eps(s.temperature[0])), 3) for run, s in prep.nulls]
    levels_e = sorted(set(eps_of))
    clock(f"L = {L}: {len(prep.nulls)} null blocks, {prep.sweeps:.4f} sweeps per sample")
    pairs = common.pairs(args, doc)
    columns = sorted({observable for observable, *_ in pairs})
    headline = (float(sens["headline"]["window"]), float(sens["headline"]["bandwidth"]))
    lag = int(ind["lag"])

    out: dict[str, Any] = {
        "provenance": common.provenance(args, pilot),
        "runs": common.run_ids(inputs),
        "sweeps_per_sample": prep.sweeps,
        "columns": {},
    }
    for column in columns:
        entry: dict[str, Any] = {"per_eps": {}, "grid": {}}
        for e in levels_e:
            blocks = [s for (_, s), x in zip(prep.nulls, eps_of, strict=True) if x == e]
            groups = [b.observable(column) for b in blocks]
            entry["per_eps"][f"{e:g}"] = {"static": static_at(groups, L, args.bootstrap, rng)}
        clock(f"{column}: static values per eps_L done")
        _, length = ising.positives(prep, column, 0.0, segment, prep.samples(headline[0], tau_c))
        names = [name for observable, name, _ in pairs if observable == column]
        for window_taus in map(float, sens["window"]):
            window = prep.samples(window_taus, tau_c)
            span = window + length - 1
            slices = [ising.null_slices(s.observable(column), span) for _, s in prep.nulls]
            row_eps = np.concatenate(
                [np.full(len(g), x) for g, x in zip(slices, eps_of, strict=True)]
            )
            x = np.concatenate(slices)
            for bandwidth_taus in map(float, sens["bandwidth"]):
                bandwidth = bandwidth_taus * tau_c / prep.sweeps
                cell = f"{window_taus:g}x{bandwidth_taus:g}"
                entry["grid"][cell] = {}
                for name in names:
                    if name in RAW and bandwidth_taus != headline[1]:
                        continue
                    pipe = Pipeline(name, window, length, bandwidth, lag)
                    residuals = pipe.residuals(x)
                    scores = pipe.trend(residuals)
                    entry["grid"][cell][name] = spread(scores)
                    if (window_taus, bandwidth_taus) == headline:
                        values = INDICATORS[name](residuals, window, lag)[:, window - 1 :]
                        if name == "variance":
                            values = values * L**2
                        for e in levels_e:
                            here = row_eps == e
                            at = entry["per_eps"][f"{e:g}"]
                            at.setdefault("levels", {})[name] = spread(values[here])
                            at.setdefault("scores", {})[name] = spread(scores[here])
                clock(f"{column}: cell {cell} done")
        out["columns"][column] = entry
    common.save(common.out_dir(args) / "nulls.json", out)
    for column, entry in out["columns"].items():
        for e, at in entry["per_eps"].items():
            st = at["static"]
            print(
                f"{column} eps_L {e}: mean {st['mean']['value']:.4f} "
                f"L2Var {st['l2_var']['value']:.3f} | "
                + " | ".join(
                    f"{k} level {v['median']:.3f} [{v['q05']:.3f}, {v['q95']:.3f}] s 5-95% "
                    f"{at['scores'][k]['q05']:+.2f}..{at['scores'][k]['q95']:+.2f}"
                    for k, v in at["levels"].items()
                )
            )
    clock("done")
