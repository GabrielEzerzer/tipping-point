"""Where each indicator takes its extremum along each slow-ramp replica (exploratory, not
pre-registered), the scorer's peak path. For every slow-ramp replica, the headline pipeline's
rolling indicator (Gaussian detrend at 10 tau_c over the whole series unless the indicator is RAW or
the spec says raw, window 20 tau_c ending at each sample, AC at one sample) is followed along the
ramp, and fss.peak_position takes its extremum inside eps_L in [6, -4]: NaN on the band's edge,
which is not located. Positions are read half a window late (0.2 eps_L at x = 0.02). Also the
ensemble peak (the replica-mean curve's extremum), which shows where the systematic extremum sits
when each replica's is noisy, and near_edge, the share of replicas on the band's edge or within one
window of it: the frozen edge rule counts only the first and last sample, and a channel that keeps
rising past the band piles up just inside it, at the same eps_L at every L whatever its physics.
CIs: hierarchical bootstrap (seeds, then replicas), 2,000 resamples.

--combine is the dynamic test on positions, fss.peak_convergence and the verdict ising._dynamic
would give. Under FSS, T_peak - Tc ~ 1/L, so eps_L_peak is the same at every L and 1/nu = 1 - d
log|eps_peak| / d log L = 1; an indicator merely monotone in T peaks at a fixed T: eps_L_peak ~ L
and 1/nu = 0. Two sizes give a slope, not a test.

With --latents the statistics are ising-latent-v1's, with both signs of the slope's extremum, since
this pass is what chooses one, and the level's and the raw variances' for reference.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from tipping_point.ews import fss, ising, scoring
from tipping_point.ews.trend import Pipeline

from explore import common

BAND = (6.0, -4.0)
EXPECTED, TOLERANCE, MIN_DRAW_SHARE = 1.0, 0.5, 0.95  # fss.dynamic of both pre-registrations
CLASSIC = {  # label -> (observable, indicator, detrended, direction)
    "abs_m_variance": ("abs_m", "variance", True, "greater"),
    "abs_m_ac": ("abs_m", "ac", True, "greater"),
    "abs_m_skewness": ("abs_m", "skewness", True, "less"),
    "abs_m_raw_variance": ("abs_m", "variance", False, "greater"),
}
LATENT = {
    "pc_1_variance": ("pc_1", "variance", True, "greater"),
    "pc_1_ac": ("pc_1", "ac", True, "greater"),
    "pc_1_raw_variance": ("pc_1", "variance", False, "greater"),
    "surprise_variance": ("surprise", "variance", True, "greater"),
    "surprise_raw_variance": ("surprise", "variance", False, "greater"),
    "surprise_slope_min": ("surprise", "slope", False, "less"),
    "surprise_slope_max": ("surprise", "slope", False, "greater"),
    "surprise_level_min": ("surprise", "level", False, "less"),
    "surprise_level_max": ("surprise", "level", False, "greater"),
}


def nanmedian(a: np.ndarray) -> float:
    return float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan


def run_one(args: Any) -> None:
    common.be_nice()
    clock = common.Clock()
    doc = common.make_doc(args)
    pilot, inputs = common.load(args, doc)
    tau_c = pilot.tau(args.L)
    ind, sens = doc["indicators"], doc["sensitivity"]
    kind = common.cadence(args)
    segment = float(doc["event"]["segment"])
    horizons = [float(h) for h in doc["event"]["horizons"]]
    prep = ising.prepare(
        inputs, kind, tau_c, float(ind["thin"]), (max(horizons) + segment, min(horizons))
    )
    window = prep.samples(float(sens["headline"]["window"]), tau_c)
    bandwidth = float(sens["headline"]["bandwidth"]) * tau_c / prep.sweeps
    lag = int(ind["lag"])
    rng = np.random.default_rng(args.seed)
    ramp = inputs.ramps[0]
    taus_per_eps = float(ramp.schedule["params"]["rate"]) * ramp.Tc / args.L / tau_c
    out = {
        "provenance": common.provenance(args, pilot),
        "runs": common.run_ids(inputs),
        "band": BAND,
        "window": window,
        "sweeps_per_sample": prep.sweeps,
        "taus_per_eps": taus_per_eps,
        "indicators": {},
    }
    for label, (observable, name, detrended, direction) in (
        LATENT if args.latents else CLASSIC
    ).items():
        pipe = Pipeline(name, window, 2, bandwidth if detrended else None, lag)
        groups, curves = [], []
        for run, s in prep.ramps:
            ising.ramp_segment(s, run, *BAND)  # refuses a ramp that does not cross the whole band
            values = pipe.rolling(s.observable(observable))
            eps = run.eps(s.temperature)
            groups.append(fss.peak_position(values, eps, BAND, direction))
            curves.append((values, eps))
        peaks = np.concatenate(groups)
        draws = scoring.hierarchical_bootstrap(nanmedian, [groups], args.bootstrap, rng)
        # the ensemble curve: mean over all replicas at each sample (the ramps share one grid)
        eps0 = curves[0][1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # the first window has no value yet
            curve = np.nanmean(np.concatenate([v for v, _ in curves]), axis=0)[None, :]
        ensemble = fss.peak_position(curve, eps0, BAND, direction)[0]
        inside = (eps0 <= BAND[0]) & (eps0 >= BAND[1])
        at = curve[0][inside]
        extreme = np.nanargmax(at) if direction == "greater" else np.nanargmin(at)
        located = peaks[np.isfinite(peaks)]
        q1, med, q3 = np.quantile(located, [0.25, 0.5, 0.75]) if len(located) else (np.nan,) * 3
        reach = window * prep.sweeps / (float(ramp.schedule["params"]["rate"]) * ramp.Tc / args.L)
        beside = (located >= BAND[0] - reach) | (located <= BAND[1] + reach)
        near_edge = (np.isnan(peaks).sum() + beside.sum()) / len(peaks)
        out["indicators"][label] = {
            "observable": observable,
            "indicator": name,
            "detrended": detrended,
            "direction": direction,
            "median": float(med),
            "iqr": [float(q1), float(q3)],
            "median_ci": scoring.percentile_interval(draws),
            "ci_method": "percentile, hierarchical bootstrap: seeds, then replicas",
            "median_taus_before_tc": float(med) * taus_per_eps,
            "on_edge": float(np.mean(np.isnan(peaks))),
            "near_edge": float(near_edge),
            "window_eps_L": reach,
            "n": len(peaks),
            "ensemble_peak": float(ensemble),
            "ensemble_extreme_eps": float(eps0[inside][extreme]),
            "peaks": [g.tolist() for g in groups],
        }
        r = out["indicators"][label]
        clock(
            f"{label:<22} median {med:+.2f} [IQR {q1:+.2f}, {q3:+.2f}], CI {r['median_ci']}, "
            f"on the band's edge {r['on_edge']:.1%}, within a window of it {r['near_edge']:.1%}, "
            f"ensemble {r['ensemble_extreme_eps']:+.2f}"
            f"{' (on its edge)' if np.isnan(ensemble) else ''}"
        )
    common.save(common.out_dir(args) / "peaks.json", out)


def combine(args: Any, sizes: list[int]) -> None:
    """1/nu over sizes per statistic, as the scorer computes it (fss.peak_convergence), with the
    verdict ising._dynamic would give, and the plain shift and ratio between the first and last
    size, which do not need one sign at every size. CIs: independent hierarchical bootstraps per
    size, percentile."""
    rng = np.random.default_rng(args.seed)
    data = {L: common.read(args, L, "peaks.json") for L in sizes}
    result: dict[str, Any] = {
        "sizes": sizes,
        "note": "two sizes give a slope, not a test; exploratory, not pre-registered",
        "indicators": {},
    }
    n = args.bootstrap
    for label in LATENT if args.latents else CLASSIC:
        samples = {
            L: [np.asarray(g, dtype=np.float64) for g in data[L]["indicators"][label]["peaks"]]
            for L in sizes
        }
        medians = np.array([data[L]["indicators"][label]["median"] for L in sizes])
        fit = fss.peak_convergence(samples, n, rng)
        lost = max(float(np.mean(np.isnan(np.concatenate(g)))) for g in samples.values())
        if not np.isfinite(fit.kappa) or lost > 0.5 or fit.draws_used < MIN_DRAW_SHARE:
            verdict = fss.INCONCLUSIVE
        else:
            verdict = fss.combine(
                fss.match(fit.kappa, fit.ci, EXPECTED, TOLERANCE),
                fss.excludes(fit.ci, 0.0, TOLERANCE),
            )
        draws = np.stack(
            [scoring.hierarchical_bootstrap(nanmedian, [samples[L]], n, rng) for L in sizes]
        )
        result["indicators"][label] = {
            "medians": dict(zip(map(str, sizes), medians.tolist(), strict=True)),
            "edge_shares": {str(L): data[L]["indicators"][label]["on_edge"] for L in sizes},
            "near_edge_shares": {str(L): data[L]["indicators"][label]["near_edge"] for L in sizes},
            "ensemble": {
                str(L): data[L]["indicators"][label]["ensemble_extreme_eps"] for L in sizes
            },
            "shift": float(medians[-1] - medians[0]),
            "shift_ci": scoring.percentile_interval(draws[-1] - draws[0]),
            "inverse_nu": fit.kappa,
            "inverse_nu_ci": fit.ci,
            "draws_used": fit.draws_used,
            "worst_edge_share": lost,
            "verdict_if_judged": verdict,
        }
        r = result["indicators"][label]
        print(
            f"{label:<22} medians {np.round(medians, 3)} edge {lost:.0%} shift {r['shift']:+.3f} "
            f"{np.round(r['shift_ci'], 3)} 1/nu {fit.kappa:+.3f} {np.round(fit.ci, 3)} "
            f"draws {fit.draws_used:.2f} -> {verdict}"
        )
    common.save(common.table_dir(args) / "peaks-combined.json", result)


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--combine", type=int, nargs="+")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args(argv)
    if args.combine:
        combine(args, sorted(args.combine))
    else:
        run_one(args)
