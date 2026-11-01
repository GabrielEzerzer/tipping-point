"""Ensemble indicators on the ramps (exploratory, descriptive, not pre-registered). At every sample
of the ramp (one sweep apart on the fast ramp; the slow ramp thinned to >= 500 rows per unit eps_L
at the dense cadence, every saved frame at the matched), across the 5 seeds x 16 replicas: the
column's mean, L^2 Var, its skewness and the two-time autocorrelation corr(x(t - d), x(t)) at the
design lag d = 0.1 tau_c, smoothed over +-0.25 eps_L. |m| without --latents; each of surprise and
pc_1 with one. The Kibble-Zurek lag, in eps_L and in tau_c after the bath crosses Tc:
  peak lag   eps_L at which the ramp's ensemble L^2 Var peaks while the bath cools, minus the
             equilibrium peak of the staircase curve (static.json: run static_tc.py first)
  order lag  where the ramp's ensemble mean first reaches the equilibrium mean at Tc (the staircase
             FSS plateau), which equilibrium reaches at eps_L = 0 by definition. A column that falls
             on cooling is read turned over, so "reaches" means the same
95% CIs from one hierarchical bootstrap (seeds, then replicas) per ramp speed. Also the Kendall tau
of the raw per-sample ensemble variance over eps_L in [6, 0] (does it rise before Tc), with the
spread over seeds instead of a bootstrap CI: resampling replicas with replacement makes each step's
variance noisier and so biases that tau low. The fast ramp (x = 2) is the Kibble-Zurek arm, reported
this way by #26 decision 4; the slow ramp (x = 0.02) runs through the same code as the
near-adiabatic reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tipping_point.ews import ising, scoring
from tipping_point.ews.trend import kendall_tau

from explore import common

POINTS = (6.0, 4.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0)
HALF = 0.25  # smoothing half-width, eps_L
SPEEDS = (("fast", 2.0), ("slow", 0.02))
NAMES = (
    "peak_eps_L",
    "peak_lag_eps_L",
    "peak_taus_after_tc",
    "order_reach_eps_L",
    "order_lag_taus_after_tc",
    "kendall_tau_var_6_to_0",
    "peak_l2_var",
    "mean_at_peak",
    "peak_censored_at_ramp_end",
)


def ramps_of(root: Any, L: int, x: float, seeds: list[int]) -> list[ising.Run]:
    complete, _ = ising.find_runs(root)
    runs = [
        r
        for r in complete
        if r.L == L
        and r.kind == "ramp"
        and r.seed in seeds
        and ising._close(float(r.schedule["params"].get("x", np.nan)), x)
    ]
    by_seed: dict[int, list[ising.Run]] = {}
    for r in runs:
        by_seed.setdefault(common.allowed(r).seed, []).append(r)
    doubled = {s: [r.run_id for r in v] for s, v in by_seed.items() if len(v) > 1}
    missing = sorted(set(seeds) - set(by_seed))
    if doubled or missing:
        raise SystemExit(f"x = {x}: doubled {doubled}, missing seeds {missing}")
    return [by_seed[s][0] for s in seeds]


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average over 2k + 1 samples along the last axis (shorter at the edges), over
    the finite values only: NaN where a window holds none."""
    if k < 1:
        return y
    pad = [(0, 0)] * (y.ndim - 1) + [(1, 0)]
    ok = np.isfinite(y)
    c = np.cumsum(np.pad(np.where(ok, y, 0.0), pad), axis=-1)
    m = np.cumsum(np.pad(ok.astype(np.float64), pad), axis=-1)
    n = y.shape[-1]
    lo = np.clip(np.arange(n) - k, 0, n)
    hi = np.clip(np.arange(n) + k + 1, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c[..., hi] - c[..., lo]) / (m[..., hi] - m[..., lo])


def ensemble(a: np.ndarray, L: int, d: int) -> dict[str, np.ndarray]:
    """Per-sample ensemble indicators of a (replicas, samples)."""
    n = a.shape[0]
    mean = a.mean(0)
    dev = a - mean
    m2, m3 = np.mean(dev**2, 0), np.mean(dev**3, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.sqrt(n * (n - 1.0)) / (n - 2) * m3 / m2**1.5
        x, y = a[:, :-d], a[:, d:]
        dx, dy = x - x.mean(0), y - y.mean(0)
        ac = np.sum(dx * dy, 0) / np.sqrt(np.sum(dx * dx, 0) * np.sum(dy * dy, 0))
    ac = np.concatenate([np.full(d, np.nan), ac])  # value at t pairs (t - d, t)
    return {"mean": mean, "l2_var": L**2 * a.var(0, ddof=1), "skewness": skew, "ac": ac}


def first_reach(values: np.ndarray, level: float, start: int) -> int:
    """First index >= start at which values >= level, or -1."""
    hit = np.flatnonzero(values[start:] >= level)
    return int(start + hit[0]) if len(hit) else -1


def summary(
    a: np.ndarray,
    eps: np.ndarray,
    steps: np.ndarray,
    t_c: float,
    ramp_end: int,
    L: int,
    k: int,
    eq_peak: float,
    mean_eq_tc: float,
    tau_c: float,
    seg: tuple[int, int],
    sign: float,
) -> np.ndarray:
    """The lag statistics of one (replicas, samples) stack, as a vector. `sign` turns a column that
    falls on cooling over, so the mean rises to its equilibrium value."""
    l2_var = L**2 * a.var(0, ddof=1)  # only the two indicators the lags need
    var_s, mean_s = smooth(l2_var, k), smooth(sign * a.mean(0), k)
    start = int(np.flatnonzero(eps <= 8.0)[0])  # past the hot hold
    j = start + int(np.nanargmax(var_s[start : ramp_end + 1]))  # while the bath still cools
    censored = float(j >= ramp_end - k)  # the maximum sits at the ramp's end: not a peak
    peak_eps = eps[j] if not censored else np.nan
    peak_taus = (steps[j] - t_c) / tau_c if not censored else np.nan
    r = first_reach(mean_s, sign * mean_eq_tc, start)
    reach_eps = eps[r] if 0 <= r <= ramp_end else np.nan
    reach_taus = (steps[r] - t_c) / tau_c if r >= 0 else np.nan
    i0, i1 = seg
    ktau = float(kendall_tau(l2_var[i0 : i1 + 1]))
    return np.array(
        [
            peak_eps,
            peak_eps - eq_peak,
            peak_taus,
            reach_eps,
            reach_taus,
            ktau,
            float(var_s[j]),
            float(mean_s[j]),
            censored,
        ]
    )


def one_speed(
    args: Any,
    column: str,
    runs: list[ising.Run],
    tau_c: float,
    eq: dict[str, Any],
    x: float,
    rng: Any,
) -> dict[str, Any]:
    L = args.L
    series = [ising.series(r, common.cadence(args), args.latents)[0] for r in runs]
    sch = runs[0].schedule
    sweeps_per_eps = float(sch["params"]["rate"]) * runs[0].Tc / L
    stride = 1 if args.latents else int(sch.get("stats_stride", 1))
    every = (
        1 if args.latents else max(1, int(sweeps_per_eps / stride // 500))
    )  # >= 500 rows per eps_L
    series = [s.take(np.arange(0, len(s.steps), every)) for s in series]
    steps, temps = series[0].steps, series[0].temperature
    for s in series[1:]:
        if not (np.array_equal(s.steps, steps) and np.allclose(s.temperature, temps)):
            raise SystemExit(f"x = {x}: ramps do not share one schedule")
    eps = runs[0].eps(temps)
    t_c, t1 = float(sch["t_c"]), int(sch["t1"])
    ramp_end = int(np.searchsorted(steps, t1, side="right") - 1)  # last row still cooling
    inside = (eps <= 6.0) & (eps >= -4.0)
    per_sample = float(steps[inside][-1] - steps[inside][0]) / max(1, int(inside.sum()) - 1)
    d = max(1, round(0.1 * tau_c / per_sample))
    k = max(1, round(HALF * sweeps_per_eps / per_sample))
    i0 = int(np.flatnonzero(eps <= 6.0)[0])
    i1 = int(np.flatnonzero(eps <= 0.0)[0])
    groups = [s.observable(column) for s in series]
    a = np.concatenate(groups)
    mean_eq_tc = float(np.interp(0.0, eq["eps_L"][::-1], eq["mean"][::-1]))
    sign = 1.0 if a.mean(0)[i1] >= a.mean(0)[i0] else -1.0
    rest = (eps, steps, t_c, ramp_end, L, k, eq["peak_eps_L"], mean_eq_tc, tau_c, (i0, i1), sign)
    point = summary(a, *rest)
    tau_at = NAMES.index("kendall_tau_var_6_to_0")
    per_seed = [float(summary(g, *rest)[tau_at]) for g in groups]
    draws = np.empty((args.bootstrap, len(NAMES)))
    for b in range(args.bootstrap):
        picks = rng.integers(0, len(groups), len(groups))
        stack = np.concatenate(
            [groups[s][rng.integers(0, len(groups[s]), len(groups[s]))] for s in picks]
        )
        draws[b] = summary(stack, *rest)
    sm = {key: smooth(v, k) for key, v in ensemble(a, L, d).items()}
    at: dict[str, Any] = {}
    for pt in POINTS:
        if pt < eps[ramp_end] or pt > eps[0]:
            continue
        j = int(np.flatnonzero(eps <= pt)[0])
        at[f"{pt:g}"] = {key: float(sm[key][j]) for key in sm} | {
            "taus_from_tc": float((steps[j] - t_c) / tau_c)
        }
    stats = {}
    for i, name in enumerate(NAMES):
        col = draws[:, i]
        stats[name] = {
            "value": float(point[i]),
            "ci": scoring.percentile_interval(col),
            "undefined_share": float(np.mean(~np.isfinite(col))),
        }
    return {
        "x": x,
        "runs": [r.run_id for r in runs],
        "rows": len(steps),
        "lag_samples": d,
        "smooth_half_samples": k,
        "sweeps_per_eps_L": sweeps_per_eps,
        "taus_per_eps_L": sweeps_per_eps / tau_c,
        "eps_start": float(eps[0]),
        "eps_end": float(eps[ramp_end]),
        "sign": sign,
        "stats": stats,
        "kendall_tau_per_seed": per_seed,
        "at_eps_L": at,
    }


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args(argv)
    common.be_nice()
    clock = common.Clock()
    doc = common.make_doc(args)
    pilot, inputs = common.load(args, doc)
    tau_c = pilot.tau(args.L)
    static = common.read(args, args.L, "static.json")
    rng = np.random.default_rng(args.seed)
    seeds = list(doc["data"]["seeds"])
    columns = list(common.JUDGED_COLUMNS) if args.latents else ["abs_m"]
    result: dict[str, Any] = {
        "provenance": common.provenance(args, pilot),
        "runs": common.run_ids(inputs),
        "columns": {},
    }
    for column in columns:
        eq = static["equilibrium"][column]
        speeds = {}
        for label, x in SPEEDS:
            runs = ramps_of(common.root_of(args), args.L, x, seeds)
            speeds[label] = one_speed(args, column, runs, tau_c, eq, x, rng)
            s = speeds[label]["stats"]
            clock(
                f"{column} x = {x}: peak eps_L {s['peak_eps_L']['value']:.2f} (lag "
                f"{s['peak_lag_eps_L']['value']:+.2f}), reaches its Tc mean at eps_L "
                f"{s['order_reach_eps_L']['value']:.2f}; Kendall tau of Var over [6, 0] "
                f"{s['kendall_tau_var_6_to_0']['value']:.3f} "
            )
        result["columns"][column] = {
            "speeds": speeds,
            "eq_peak_eps_L": eq["peak_eps_L"],
        }
    common.save(common.out_dir(args) / "fast_ramps.json", result)
    clock("done")
