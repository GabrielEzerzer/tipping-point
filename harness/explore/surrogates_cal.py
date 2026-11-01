"""The surrogate test calibrated on the REAL fixed-T null runs (exploratory, not pre-registered),
the #28 pre-flight. At the headline cell (window 20, bandwidth 10 tau_c), each null slice and each
slow-ramp positive is scored against N_s = 99 phase-randomised and AR(1)-fitted surrogates of its
own residuals with tipping_point.ews.scoring.surrogate_test, one-sided in the pair's direction.
Reports the false-positive rate on the nulls at alpha (overall, per eps_L, per seed) with its
binomial band, and the detection rate on the positives per h. A level (RAW, no detrending) takes the
phase surrogate alone: fit to a trending raw series, AR(1) draws near-random walks with no trend.

N_s = 99, not the pre-registered 999: at 999 the scorer's surrogate tests project to about 24 h on
this machine (#70). With 99, p = (1 + k) / 100 and p <= 0.05 means k <= 4: still exact. The null
rows are the same for every h whenever the span is, so they are tested once per pair and kind at the
first h's span. Each pair and kind is written to surrogates-<observable>-<indicator>-<kind>.json as
soon as it is done, so a long calibration can be split across runs with --indicators / --kinds;
report.py merges the files.

The binomial band treats the slices as independent, which is weaker than it looks on these data: the
three null runs at a seed shared a stream (README: Which stream a run draws).
"""

from __future__ import annotations

import math

import numpy as np
from tipping_point.ews import ising, scoring, surrogates
from tipping_point.ews.trend import RAW, Pipeline

from explore import common

ALPHAS = (0.01, 0.05, 0.10)
KINDS = ("phase", "ar1")


def band(alpha: float, n: int, z: float) -> tuple[float, float]:
    half = z * math.sqrt(alpha * (1 - alpha) / n)
    return alpha - half, alpha + half


def rates(p: np.ndarray) -> dict[str, float]:
    p = p[np.isfinite(p)]
    return {f"{a:g}": float(np.mean(p <= a)) for a in ALPHAS} | {"n": len(p)}


def main(argv: list[str] | None = None) -> None:
    parser = common.parser(__doc__)
    parser.add_argument("--n-surrogates", type=int, default=99)
    parser.add_argument("--indicators", nargs="+", help="observable_indicator; default all pairs")
    parser.add_argument("--kinds", nargs="+", default=list(KINDS))
    args = parser.parse_args(argv)
    common.be_nice()
    clock = common.Clock()
    doc = common.make_doc(args, n_surrogates=args.n_surrogates)
    pilot, inputs = common.load(args, doc)
    tau_c = pilot.tau(args.L)
    ind, sens, sc = doc["indicators"], doc["sensitivity"], doc["scoring"]
    segment = float(doc["event"]["segment"])
    horizons = [float(h) for h in doc["event"]["horizons"]]
    prep = ising.prepare(
        inputs,
        common.cadence(args),
        tau_c,
        float(ind["thin"]),
        (max(horizons) + segment, min(horizons)),
    )
    window = prep.samples(float(sens["headline"]["window"]), tau_c)
    bandwidth = float(sens["headline"]["bandwidth"]) * tau_c / prep.sweeps
    alpha, n = float(sc["alpha"]), int(args.n_surrogates)
    pairs = common.oriented_pairs(args, doc)
    out = common.out_dir(args)
    meta = None
    for i, (observable, name, direction) in enumerate(pairs):
        label = f"{observable}_{name}"
        if args.indicators and label not in args.indicators:
            continue
        _, length0 = ising.positives(prep, observable, min(horizons), segment, window)
        span0 = window + length0 - 1
        neg_groups = [ising.null_slices(s.observable(observable), span0) for _, s in prep.nulls]
        neg = np.concatenate(neg_groups)
        eps_of_row = np.concatenate(
            [
                np.full(len(g), round(float(run.eps(s.temperature[0])), 3))
                for (run, s), g in zip(prep.nulls, neg_groups, strict=True)
            ]
        )
        seed_of_row = np.concatenate(
            [np.full(len(g), run.seed) for (run, _), g in zip(prep.nulls, neg_groups, strict=True)]
        )
        clock(
            f"{label}: {len(neg)} null rows of {span0} samples; window {window}, length {length0}"
        )
        for j, kind in enumerate(KINDS):
            if kind not in args.kinds or (kind == "ar1" and name in RAW):
                continue
            make = surrogates.KINDS[kind]
            rng = np.random.default_rng([args.seed, i, j])  # the same draws however split
            pipe0 = Pipeline(name, window, length0, bandwidth, int(ind["lag"]))
            _, p_neg = scoring.surrogate_test(neg, pipe0, make, n, rng, direction)
            row = {
                "observable": observable,
                "indicator": name,
                "direction": direction,
                "surrogate": kind,
                "n_surrogates": n,
                "alpha": alpha,
                "null": rates(p_neg),
                "null_by_eps": {
                    f"{e:g}": rates(p_neg[eps_of_row == e]) for e in np.unique(eps_of_row)
                },
                "null_by_seed": {
                    str(s): rates(p_neg[seed_of_row == s]) for s in np.unique(seed_of_row)
                },
                "band95": band(alpha, len(neg), 1.96),
                "band99": band(alpha, len(neg), 2.576),
                "null_p_hist": np.histogram(p_neg[np.isfinite(p_neg)], bins=10, range=(0, 1))[
                    0
                ].tolist(),
                "positives": {},
            }
            for h in horizons:
                try:
                    pos_rows, length = ising.positives(prep, observable, h, segment, window)
                except ising.IsingDataError:
                    continue
                pipe = Pipeline(name, window, length, bandwidth, int(ind["lag"]))
                _, p_pos = scoring.surrogate_test(
                    np.concatenate(pos_rows), pipe, make, n, rng, direction
                )
                row["positives"][f"{h:g}"] = rates(p_pos)
            fp = row["null"][f"{alpha:g}"]
            lo, hi = row["band95"]
            row["inside95"] = lo <= fp <= hi
            row["inside99"] = row["band99"][0] <= fp <= row["band99"][1]
            if meta is None:
                meta = {
                    "provenance": common.provenance(args, pilot),
                    "runs": common.run_ids(inputs),
                    "cell": {
                        "window": window,
                        "bandwidth": bandwidth,
                        "window_taus": sens["headline"]["window"],
                        "bandwidth_taus": sens["headline"]["bandwidth"],
                    },
                }
            common.save(out / f"surrogates-{observable}-{name}-{kind}.json", meta | {"rows": [row]})
            clock(
                f"{label:<20} {kind:<6} false positives {100 * fp:5.2f}% (95% band "
                f"{100 * lo:.2f}-{100 * hi:.2f}%), detected "
                + ", ".join(
                    f"h={h}: {100 * v[f'{alpha:g}']:.1f}%" for h, v in row["positives"].items()
                )
            )
    clock("done")
