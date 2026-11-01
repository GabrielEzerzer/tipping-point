"""Static values and equilibrium curves from the staircases (exploratory, not pre-registered), one
size only: the L row of the future scaling table, no exponent fit (scaling.py fits them).

At the FSS plateau (T = 2.269231 = Tc + 4.5e-5), with 95% CIs from the hierarchical bootstrap
(seeds, then replicas within a seed), for |m| at the dense cadence or, with --latents, for surprise
and pc_1 at the matched cadence: L^2 Var, the mean, the skewness, 1 - AC1 at one sweep (pairs inside
bursts), and for |m| alone tau_int (Sokal window, sweeps) and, as a pipeline-independent check, the
Binder ratio <m^4>/<m^2>^2, whose infinite-L value on the periodic square lattice is 1.1679229(47)
(Salas and Sokal 2000). Classic mode sets beside them the same statistics of the #50 pilot run held
at exactly Tc (its replicas bootstrapped).

Over every plateau, for each column: L^2 Var and the mean against eps_L, from one joint bootstrap
(the same seeds and replicas drawn at every temperature), summed from per-replica power sums, with
the interpolated peak of L^2 Var, and the shape of the mean, whether it is monotone or turns
(shape, below). A latents table holds the frames its selection kept, one burst in
8 away from the FSS plateau under tc-plus, so each of those points rests on 1/8 of the plateau's
frames. The bootstrap draws at Tc go to static-draws.npz for scaling.py; the first equilibrium curve
is what fast_ramps.py reads.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tipping_point.ews import ising, scoring
from tipping_point.ews.trend import kendall_tau
from tipping_point.ising import autocorr

from explore import common

BINDER_INF = 1.1679229
PILOT_AT_TC = {  # the #50 tau_c pilot run per L, held at exactly Tc
    16: "20261022T150237Z-L16-pilot-s1",
    32: "20261022T150529Z-L32-pilot-s1",
}


def binder(a: np.ndarray) -> float:
    a = a.astype(np.float64)
    m2 = np.mean(a**2)
    return float(np.mean(a**4) / m2**2)


def _moments(sums: np.ndarray, L: int, name: str, shift: float = 0.0) -> dict[str, float]:
    """Pooled statistics from summed per-replica power sums [N, S1, S2, S3, S4] of x - shift: the
    same numbers as ising._statistic's pooled ones (var ddof 1, G1 skewness) and, with no shift, the
    Binder ratio. A column far from 0 beside its spread loses a third moment's digits unshifted."""
    N, S1, S2, S3, S4 = sums
    mu = S1 / N
    m2 = S2 / N - mu * mu
    m3 = S3 / N - 3 * mu * S2 / N + 2 * mu**3
    return {
        f"l2_var_{name}": L**2 * m2 * N / (N - 1),
        f"mean_{name}": mu + shift,
        f"skewness_{name}": np.sqrt(N * (N - 1.0)) / (N - 2) * m3 / m2**1.5,
        "binder_ratio": (S4 / N) / (S2 / N) ** 2,
    }


def _picks(sizes: list[int], rng: np.random.Generator) -> list[tuple[int, np.ndarray]]:
    """One hierarchical resample: seeds with replacement, then replicas within each drawn seed."""
    seeds = rng.integers(0, len(sizes), len(sizes))
    return [(s, rng.integers(0, sizes[s], sizes[s])) for s in seeds]


def _quantity(name: str, q: str, L: int, stride: int) -> Any:
    if q == "binder_ratio":
        return lambda a: binder(a)
    statistic = ising._statistic(q.removesuffix(f"_{name}"))
    return lambda a: statistic(L, a, stride)


def stats_of(
    name: str,
    groups: list[np.ndarray],
    bursts: list[np.ndarray],
    L: int,
    stride: int,
    n: int,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Point value (ising._statistic on all the data), 95% CI from the hierarchical bootstrap
    and per-seed values of each quantity of a column. The bootstrap works on per-replica summaries:
    power sums for the moments, each replica's rho(t) for tau_int, which autocorr.estimate averages
    over replicas before its Sokal window. So a resample costs no FFT, and the point values check
    that the summaries reproduce the library's numbers."""
    classic = name == "abs_m"
    quantities = [f"l2_var_{name}", f"mean_{name}", f"skewness_{name}"]
    if classic:
        quantities += [f"tau_int_{name}", "binder_ratio"]
    shift = 0.0 if classic else float(np.mean(np.concatenate(groups)))
    sums = [
        np.stack(
            [
                np.full(len(g), g.shape[1]),
                *(np.sum((g.astype(np.float64) - shift) ** k, axis=1) for k in range(1, 5)),
            ],
            axis=1,
        )
        for g in groups
    ]
    rhos = [autocorr._rows(g)[:, : max(1, g.shape[1] // 2)] for g in groups] if classic else []
    sizes = [len(g) for g in groups]
    draws = {q: np.empty(n) for q in quantities}
    for i in range(n):
        picks = _picks(sizes, rng)
        total = sum(sums[s][r].sum(axis=0) for s, r in picks)
        for q, v in _moments(np.asarray(total), L, name, shift).items():
            if q in draws:
                draws[q][i] = v
        if classic:
            rho = np.concatenate([rhos[s][r] for s, r in picks]).mean(axis=0)
            draws[f"tau_int_{name}"][i] = autocorr._window(rho, 5.0)[0] * stride
    whole = np.concatenate(groups)
    functions = {q: _quantity(name, q, L, stride) for q in quantities}
    library = {q: functions[q](whole) for q in quantities}
    check = _moments(np.asarray(sum(x.sum(axis=0) for x in sums)), L, name, shift)
    for q, v in check.items():
        if q in library and not np.isclose(v, library[q], rtol=1e-9, atol=0):
            raise SystemExit(f"{q}: power sums give {v}, the library {library[q]}")
    out = {
        q: {
            "value": library[q],
            "ci": scoring.percentile_interval(draws[q]),
            "per_seed": [functions[q](g) for g in groups] if len(groups) > 1 else [],
        }
        for q in quantities
    }
    if bursts:
        q = "one_minus_ac1" if classic else f"one_minus_ac1_{name}"
        f1 = ising._statistic("one_minus_ac1")

        def stat1(a: np.ndarray) -> float:
            return f1(L, a, 1)

        draws[q] = scoring.hierarchical_bootstrap(stat1, [bursts], n, rng)
        out[q] = {
            "value": stat1(np.concatenate(bursts)),
            "ci": scoring.percentile_interval(draws[q]),
            "per_seed": [stat1(b) for b in bursts] if len(bursts) > 1 else [],
        }
    return out, draws


def peak(eps: np.ndarray, values: np.ndarray) -> float:
    """eps_L of the maximum, refined by the parabola through it and its two neighbours."""
    k = int(np.nanargmax(values))
    if k == 0 or k == len(values) - 1:
        return float(eps[k])
    x, y = eps[k - 1 : k + 2], values[k - 1 : k + 2]
    a, b, _ = np.polyfit(x, y, 2)
    return float(-b / (2 * a)) if a < 0 else float(eps[k])


def _turn(curve: np.ndarray) -> tuple[float, int, float, int]:
    """The most prominent interior valley and peak of a curve: (prominence, index) of each, 0 and -1
    where it has none. A valley's prominence is how far it lies below the lower of the highest
    points on its two sides, a peak's how far above the higher of the lowest: so a shallow dip
    counts even beside a steep fall elsewhere, and a monotone curve has neither."""
    valley, valley_at, top, top_at = 0.0, -1, 0.0, -1
    for k in range(1, len(curve) - 1):
        left, right = curve[:k], curve[k + 1 :]
        if curve[k] <= curve[k - 1] and curve[k] <= curve[k + 1]:
            depth = float(min(left.max(), right.max()) - curve[k])
            if depth > valley:
                valley, valley_at = depth, k
        if curve[k] >= curve[k - 1] and curve[k] >= curve[k + 1]:
            height = float(curve[k] - max(left.min(), right.min()))
            if height > top:
                top, top_at = height, k
    return valley, valley_at, top, top_at


def shape(eps: np.ndarray, point: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    """Is a curve monotone, or does it turn? Kendall's tau over the plateaus says how nearly
    monotone it is; each interior turn's prominence (_turn) and position come with the same draws'
    CIs. Noise alone gives every draw small turns, so a turn is real when its CI clears `noise`, the
    median half-width of the curve's own CI. eps runs hot to cold, the order of a cooling ramp."""
    found = [_turn(row) for row in draws]
    noise = float(np.median(np.diff(np.quantile(draws, [0.025, 0.975], axis=0), axis=0)) / 2)
    valley, valley_at, top, top_at = _turn(point)

    def entry(prominence: float, at: int, column: int) -> dict[str, Any]:
        depths = np.array([row[column] for row in found])
        spots = np.array([int(row[column + 1]) for row in found])
        where = np.where(spots >= 0, eps[spots], np.nan)
        lo, hi = scoring.percentile_interval(depths)
        return {
            "prominence": prominence,
            "prominence_ci": [lo, hi],
            "turns": bool(lo > noise),
            "eps_L": float(eps[at]) if at >= 0 else float("nan"),
            "eps_L_ci": scoring.percentile_interval(where) if (spots >= 0).any() else (np.nan,) * 2,
        }

    return {
        "kendall_tau_cooling": float(kendall_tau(point)),
        "kendall_tau_cooling_ci": scoring.percentile_interval(kendall_tau(draws)),
        "noise": noise,
        "valley": entry(valley, valley_at, 0),
        "peak": entry(top, top_at, 2),
    }


def curve(
    plateaus: list[dict[float, ising.Series]], name: str, L: int, Tc: float, n: int, seed: int
) -> dict[str, Any]:
    """Equilibrium L^2 Var and mean per plateau temperature, hot to cold, pooled over seeds and
    replicas; CIs, the peak position and the shape from one joint bootstrap (the same seeds and
    replicas drawn at every temperature). `seed` draws the same resamples for every column, so two
    columns' curves compare draw by draw."""
    temps = sorted(set.intersection(*(set(p) for p in plateaus)), reverse=True)
    rng = np.random.default_rng(seed)
    sums: dict[float, list[np.ndarray]] = {}  # T -> per seed (replicas, 3): N, sum x, sum x^2
    for T in temps:
        sums[T] = []
        for p in plateaus:
            a = p[T].observable(name).astype(np.float64)
            sums[T].append(
                np.stack([np.full(len(a), a.shape[1]), a.sum(axis=1), (a * a).sum(axis=1)], axis=1)
            )
    eps = np.array([L * (T - Tc) / Tc for T in temps])

    def evaluate(seeds: Any, replicas: Any) -> tuple[np.ndarray, np.ndarray]:
        var, mean = np.empty(len(temps)), np.empty(len(temps))
        for k, T in enumerate(temps):
            total = sum(sums[T][s][r].sum(axis=0) for s, r in zip(seeds, replicas, strict=True))
            N, S1, S2 = np.asarray(total)
            var[k], mean[k] = L**2 * (S2 - S1 * S1 / N) / (N - 1), S1 / N
        return var, mean

    count = len(plateaus)
    reps = [len(p[temps[0]].replicas) for p in plateaus]
    v0, m0 = evaluate(range(count), [np.arange(r) for r in reps])
    draws_v, draws_m = np.empty((n, len(temps))), np.empty((n, len(temps)))
    for i in range(n):
        seeds = rng.integers(0, count, count)
        draws_v[i], draws_m[i] = evaluate(seeds, [rng.integers(0, reps[s], reps[s]) for s in seeds])
    peaks = np.array([peak(eps, v) for v in draws_v])
    return {
        "column": name,
        "eps_L": eps,
        "T": temps,
        "frames": [int(sum(p[T].observable(name).size for p in plateaus)) for T in temps],
        "l2_var": v0,
        "l2_var_ci": np.nanquantile(draws_v, [0.025, 0.975], axis=0).T,
        "mean": m0,
        "mean_ci": np.nanquantile(draws_m, [0.025, 0.975], axis=0).T,
        "peak_eps_L": peak(eps, v0),
        "peak_eps_L_ci": scoring.percentile_interval(peaks),
        "mean_shape": shape(eps, m0, draws_m),
        "l2_var_shape": shape(eps, v0, draws_v),
        "bootstrap": n,
        "ci_method": "percentile, joint hierarchical bootstrap: seeds, then replicas",
    }


def _burst_rows(frames: ising.Series, name: str, per: int) -> np.ndarray:
    """(replicas, bursts, frames) of a column on a plateau's saved frames."""
    whole = len(frames.steps) // per * per
    return frames.observable(name)[:, :whole].reshape(len(frames.replicas), -1, per)


def pilot_abs_m(run: ising.Run) -> np.ndarray:
    """|m| per site of a one-block fixed run's measured rows, (replicas, time), by the adapter."""
    if int(run.params["blocks"]) != 1:
        raise SystemExit(f"{run.run_id}: expected one temperature block")
    return ising.measured(ising.series(run, "dense")[0]).abs_m


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--pilot-at-tc", default=None)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--curve-bootstrap", type=int, default=400)
    args = p.parse_args(argv)
    common.be_nice()
    clock = common.Clock()
    doc = common.make_doc(args)
    pilot, inputs = common.load(args, doc)
    rng = np.random.default_rng(args.seed)
    L, T_fss = args.L, float(doc["data"]["staircase"]["temperature"])
    table = args.latents
    columns = list(common.JUDGED_COLUMNS) if table else ["abs_m"]
    curves = [*common.ALL_COLUMNS, "abs_m", "energy"] if table else ["abs_m"]

    samples: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    bursts: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    strides, dropped, plateaus, plateau_info = set(), 0, [], []
    for run in inputs.staircases:
        kept, lost = ising.plateaus(run, common.cadence(args), table)
        dropped += lost
        plateaus.append(kept)
        here = [t for t in kept if ising._close(t, T_fss)]
        if not here:
            raise SystemExit(f"{run.run_id}: no plateau at T = {T_fss}")
        plateau = kept[here[0]]
        stride = 1 if table else int(run.schedule["stats_strides"][int(plateau.segment[0])])
        strides.add(stride)
        for c in columns:
            samples[c].append(plateau.observable(c))
        plateau_info.append(
            {
                "run_id": run.run_id,
                "T": here[0],
                "rows": int(plateau.abs_m.shape[1]),
                "stride": stride,
                "sweeps": int(plateau.steps[-1] - plateau.steps[0] + 1),
            }
        )
        frames = plateau if table else ising.plateaus(run, "matched")[0].get(here[0])
        burst = run.schedule["params"]["burst"]
        if frames is not None and int(burst["stride"]) == 1:
            for c in columns:
                bursts[c].append(_burst_rows(frames, c, int(burst["frames"])))
    if len(strides) != 1:
        raise SystemExit(f"staircases disagree on the stats stride at Tc: {strides}")
    stride = strides.pop()
    clock(f"{len(plateaus)} staircases, {dropped} plateaus dropped, stride {stride}")
    stair: dict[str, dict[str, Any]] = {}
    stair_draws: dict[str, np.ndarray] = {}
    for c in columns:
        values, draws = stats_of(c, samples[c], bursts[c], L, stride, args.bootstrap, rng)
        stair |= values
        stair_draws |= draws
    clock("statistics at Tc done")

    Tc = inputs.staircases[0].Tc
    equilibrium = {c: curve(plateaus, c, L, Tc, args.curve_bootstrap, args.seed) for c in curves}
    for c in curves:
        e = equilibrium[c]
        clock(f"curve {c}: L^2 Var peak eps_L {e['peak_eps_L']:.3f} {e['peak_eps_L_ci']}")

    pilot_id = args.pilot_at_tc or (None if table else PILOT_AT_TC.get(L))
    at_tc, pilot_draws = None, {}
    if pilot_id and (common.root_of(args) / "ising" / pilot_id).is_dir():
        prun = common.allowed(ising.open_run(common.root_of(args) / "ising" / pilot_id))
        abs_m = pilot_abs_m(prun)
        T = float(prun.schedule["params"]["temperatures"][0])
        pstride = int(prun.schedule["params"].get("stats_stride", 1))
        pstats, pilot_draws = stats_of("abs_m", [abs_m], [], L, pstride, 2000, rng)
        at_tc = {
            "run_id": pilot_id,
            "T": T,
            "eps_L": float(prun.eps(T)),
            "replicas": int(abs_m.shape[0]),
            "rows": int(abs_m.shape[1]),
            "stats": pstats,
        }
        clock("pilot-at-Tc statistics done")

    out = common.out_dir(args)
    arrays = {f"staircase_{q}": v for q, v in stair_draws.items()}
    arrays |= {f"pilot_{q}": v for q, v in pilot_draws.items()}
    np.savez_compressed(out / "static-draws.npz", **arrays)  # pyright: ignore[reportArgumentType]
    common.save(
        out / "static.json",
        {
            "provenance": common.provenance(args, pilot),
            "runs": common.run_ids(inputs),
            "T_fss": T_fss,
            "plateaus": plateau_info,
            "plateaus_dropped": dropped,
            "staircase": stair,
            "binder_ratio_infinite_L": BINDER_INF,
            "pilot_at_tc": at_tc,
            "equilibrium": equilibrium,
        },
    )
    for q, v in stair.items():
        per_seed = " ".join(f"{x:.3f}" for x in v["per_seed"])
        print(f"{q:<22} {v['value']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]  per seed {per_seed}")
    if at_tc:
        for q, v in at_tc["stats"].items():
            print(f"pilot {q:<22} {v['value']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
    clock("done")
