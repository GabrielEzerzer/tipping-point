"""The alarm threshold, examined (exploratory, not pre-registered). The freeze's rule (theta = the
lowest threshold whose false-alarm rate, and every higher one's, stays within the target on the null
runs of half the seeds) sets no threshold when no theta alarms that often on these s(t) paths (#97,
#98), so this reads what a rule could do at all. s(t) is computed exactly as
tipping_point.ews.ising._alarms does (the same Pipeline, every 1 tau_c, oriented by the pair's
direction): over each whole null series, and on the ramps at the first s(t) and within 0.5 eps_L of
the band, the only ramp scores the analysis reads (identical to the whole-series ones). They are
cached per direction in st-*.npz. For each pair:
  ceiling    the highest false-alarm rate any theta gives on the set half, with and without the
             scorer's convention that a series starting above theta counts one alarm
  the rules  the frozen target 1 per 1,000 tau_c, 1 per 100 tau_c, and the largest s(t) on the set
             half, each with theta, false alarms per 1,000 tau_c on the set and test halves, the
             first-alarm eps_L inside [2, -2] (median, IQR, a bootstrap CI of the median), the lead
             in tau_c, the share alarmed above Tc, censored and missed, and the ramps' s(t) as the
             band opens
  entry      where the ramps' first s(t) falls (eps_L), and whether it already exceeds every
             null s(t)
One JSON per pair, alarms_explore-<observable>-<indicator>.json, as soon as it is done; report.py
merges them. A table, as v1 has it: never a hypothesis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tipping_point.ews import ising, scoring
from tipping_point.ews.trend import Pipeline, trailing_score

from explore import common


def scores_at(x: np.ndarray, pipe: Pipeline, times: np.ndarray) -> np.ndarray:
    """s at the given end samples of each row of x, sliced exactly as trailing_score slices."""
    span = pipe.span
    rows = x.reshape(-1, x.shape[-1])
    out = np.empty((len(rows), len(times)))
    per = max(1, (1 << 23) // (len(rows) * span))
    for k in range(0, len(times), per):
        t = np.asarray(times[k : k + per])
        slices = rows[:, t[:, None] - span + 1 + np.arange(span)]
        out[:, k : k + per] = pipe(slices.reshape(-1, span)).reshape(len(rows), len(t))
    return out


def ramp_times(
    run: ising.Run,
    s: ising.Series,
    observable: str,
    pipe: Pipeline,
    every: int,
    band: tuple[float, float],
    margin: float = 0.5,
) -> np.ndarray:
    """The ramp's s(t) times on trailing_score's grid that the analysis reads: the first one, and
    those within `margin` eps_L of the alarm band (so the score just before it opens is there). Long
    ramps (L >= 32) thereby skip most of their scores."""
    grid = np.arange(pipe.span - 1, s.observable(observable).shape[-1], every)
    eps = run.eps(s.temperature[grid])
    keep = (eps <= band[0] + margin) & (eps >= band[1] - margin)
    keep[0] = True
    return grid[keep]


def scores_for(
    prep: ising.Prepared,
    observable: str,
    pipe: Pipeline,
    every: int,
    direction: str,
    cache: Any,
    band: tuple[float, float],
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Oriented s(t): per null block (replicas, times) on every point of the grid; per ramp (times,
    (replicas, times)) at ramp_times only."""
    if cache.is_file():
        z = np.load(cache)
        nulls = [z[f"null_{k}"] for k in range(len(prep.nulls))]
        ramps = [(z[f"ramp_t_{k}"], z[f"ramp_{k}"]) for k in range(len(prep.ramps))]
        return nulls, ramps
    nulls = [
        scoring.oriented(trailing_score(s.observable(observable), pipe, every)[1], direction)
        for _, s in prep.nulls
    ]
    ramps = []
    for run, s in prep.ramps:
        t = ramp_times(run, s, observable, pipe, every, band)
        values = scoring.oriented(scores_at(s.observable(observable), pipe, t), direction)
        ramps.append((t, values))
    arrays = {f"null_{k}": v for k, v in enumerate(nulls)}
    arrays |= {f"ramp_t_{k}": t for k, (t, _) in enumerate(ramps)}
    arrays |= {f"ramp_{k}": v for k, (_, v) in enumerate(ramps)}
    np.savez_compressed(cache, **arrays)  # pyright: ignore[reportArgumentType]
    return nulls, ramps


def ceiling(train: np.ndarray, duration: float) -> dict[str, Any]:
    """Max over theta of the false-alarm rate per 1,000 tau_c: with the scorer's convention (a
    series starting above theta counts one alarm), and counting only rises between scores."""
    finite = train[np.isfinite(train)]
    grid = np.unique(np.quantile(finite, np.linspace(0, 1, 2001)))
    with_first = np.array([scoring.alarms(train, t).sum() for t in grid])
    first_above = np.array([(train[:, 0] > t).sum() for t in grid])
    rises = with_first - first_above
    k, j = int(np.argmax(with_first)), int(np.argmax(rises))
    return {
        "with_first": float(with_first[k] / duration * 1000),
        "with_first_theta": float(grid[k]),
        "rises_only": float(rises[j] / duration * 1000),
        "rises_only_theta": float(grid[j]),
        "rows": int(train.shape[0]),
        "duration_taus": float(duration),
    }


def evaluate(
    theta: float,
    train: np.ndarray,
    test: np.ndarray,
    durations: tuple[float, float],
    ramps_eps: list[np.ndarray],
    ramps_scores: list[np.ndarray],
    band: tuple[float, float],
    n: int,
    rng: np.random.Generator,
    per_eps: float,
) -> dict[str, Any]:
    groups, censored, entries = [], [], []
    for eps, scores in zip(ramps_eps, ramps_scores, strict=True):
        inside = np.flatnonzero((eps <= band[0]) & (eps >= band[1]))
        entry = scores[:, max(0, inside[0] - 1)]
        alarm, late = scoring.first_alarm(eps[inside], scores[:, inside], theta, entry)
        groups.append(alarm)
        censored.append(late)
        entries.append(entry)
    alarm, late, entry = (np.concatenate(g) for g in (groups, censored, entries))
    hit = alarm[np.isfinite(alarm)]
    q1, med, q3 = np.quantile(hit, [0.25, 0.5, 0.75]) if len(hit) else (np.nan,) * 3

    def median(a: np.ndarray) -> float:
        return float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan

    lo, hi = np.nan, np.nan
    if len(hit):
        lo, hi = scoring.percentile_interval(
            scoring.hierarchical_bootstrap(median, [groups], n, rng)
        )
    measured = ~late
    return {
        "attainable": True,
        "theta": float(theta),
        "far_set": scoring.false_alarm_rate(train, theta, durations[0]) * 1000,
        "far_test": scoring.false_alarm_rate(test, theta, durations[1]) * 1000,
        "alarm_eps_median": float(med),
        "alarm_eps_iqr": [float(q1), float(q3)],
        "alarm_eps_median_ci": [lo, hi],
        "lead_taus_median": float(med) * per_eps,
        "alarmed_above_tc": float(np.mean(alarm > 0)),
        "censored": float(np.mean(late)),
        "missed": float(np.mean(~np.isfinite(alarm[measured]))) if measured.any() else np.nan,
        "hits": len(hit),
        "n": len(alarm),
        "entry_score_median": float(np.median(entry)),
        "entry_score_iqr": np.quantile(entry, [0.25, 0.75]).tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--indicators", nargs="+", help="observable_indicator; default all pairs")
    p.add_argument("--every", type=float, default=None, help="tau_c between scores, default v1's")
    args = p.parse_args(argv)
    common.be_nice()
    clock = common.Clock()
    doc = common.make_doc(args)
    pilot, inputs = common.load(args, doc)
    tau_c = pilot.tau(args.L)
    ind, sens, a = doc["indicators"], doc["sensitivity"], doc["scoring"]["alarms"]
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
    every = max(1, round(float(args.every or a["every"]) * tau_c / prep.sweeps))
    taus_per_score = every * prep.sweeps / tau_c
    band = (float(a["band"][0]), float(a["band"][1]))
    ramp = inputs.ramps[0]
    per_eps = float(ramp.schedule["params"]["rate"]) * ramp.Tc / args.L / tau_c
    rng = np.random.default_rng(args.seed)
    seeds = sorted({run.seed for run, _ in prep.nulls})
    out = common.out_dir(args)
    for observable, name, direction in common.oriented_pairs(args, doc):
        label = f"{observable}_{name}"
        if args.indicators and label not in args.indicators:
            continue
        _, length = ising.positives(prep, observable, min(horizons), segment, window)
        pipe = Pipeline(name, window, length, bandwidth, int(ind["lag"]))
        cache = out / f"st-{observable}-{name}-{direction}.npz"
        nulls, ramps = scores_for(prep, observable, pipe, every, direction, cache, band)
        clock(f"{label}: s(t) ready")
        halves: list[list[np.ndarray]] = [[], []]
        for (run, _), sc in zip(prep.nulls, nulls, strict=True):
            halves[seeds.index(run.seed) % 2].append(sc)
        train, test = (ising._ragged(h) for h in halves)
        durations = (
            sum(b.size for b in halves[0]) * taus_per_score,
            sum(b.size for b in halves[1]) * taus_per_score,
        )
        ramps_eps = [
            run.eps(s.temperature[t]) for (run, s), (t, _) in zip(prep.ramps, ramps, strict=True)
        ]
        ramps_scores = [sc for _, sc in ramps]
        firsts = np.concatenate([sc[:, 0] for sc in ramps_scores])
        highest = float(max(np.nanmax(train), np.nanmax(test)))
        row: dict[str, Any] = {
            "observable": observable,
            "indicator": name,
            "direction": direction,
            "taus_per_score": taus_per_score,
            "taus_per_eps": per_eps,
            "durations_taus": durations,
            "ceiling": ceiling(train, durations[0]),
            "null_s_max_set": float(np.nanmax(train)),
            "null_s_max_all": highest,
            # the first s(t) a ramp has (its trailing segment needs span samples): where, and
            # whether it already sits above every null score
            "first_score": {
                "eps_L": float(np.median([e[0] for e in ramps_eps])),
                "taus_before_tc": float(np.median([e[0] for e in ramps_eps]) * per_eps),
                "min": float(np.min(firsts)),
                "median": float(np.median(firsts)),
                "above_all_nulls": float(np.mean(firsts > highest)),
            },
            "null_s_q99_set": float(np.nanquantile(train, 0.99)),
            "variants": {},
        }
        rules = {
            "frozen far 1e-3": scoring.threshold_at_far(train, float(a["far"]), durations[0]),
            "far 1e-2": scoring.threshold_at_far(train, 0.01, durations[0]),
        }
        thetas = {k: found.theta for k, found in rules.items() if found.attainable}
        thetas["zero false alarms on set half"] = float(np.nanmax(train))
        for label2, found in rules.items():
            if not found.attainable:
                row["variants"][label2] = {
                    "attainable": False,
                    "far_max_per_1000": found.far_max * 1000,
                }
        for label2, theta in thetas.items():
            row["variants"][label2] = evaluate(
                theta,
                train,
                test,
                durations,
                ramps_eps,
                ramps_scores,
                band,
                int(doc["scoring"]["bootstrap"]),
                rng,
                per_eps,
            )
        common.save(
            out / f"alarms_explore-{observable}-{name}.json",
            {
                "provenance": common.provenance(args, pilot),
                "runs": common.run_ids(inputs),
                "cell": {
                    "window": window,
                    "bandwidth": bandwidth,
                    "length": length,
                    "every": every,
                    "band": band,
                },
                "rows": [row],
            },
        )
        c = row["ceiling"]
        clock(
            f"{label}: ceiling {c['with_first']:.2f} per 1000 tau_c (rises only "
            f"{c['rises_only']:.2f}); "
            + "; ".join(
                f"{k}: theta {v['theta']:+.3f}, far {v['far_set']:.2f}/{v['far_test']:.2f}, "
                f"alarm eps {v['alarm_eps_median']:+.2f}, censored {v['censored']:.2f}, "
                f"missed {v['missed']:.2f}"
                if v.get("attainable")
                else f"{k}: unattainable, at most {v['far_max_per_1000']:.2f} per 1000"
                for k, v in row["variants"].items()
            )
        )
    clock("done")
