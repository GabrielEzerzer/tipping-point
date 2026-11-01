"""The probes' controls on seeds 1-5 (exploratory, not pre-registered, --latents only): R^2 of ridge
probes of |m|, e and L^2 (|m| - <|m|>_T)^2, each beside its iid and shuffle-within-T control tasks,
with selectivity and the floors these give, through jepa.probes' own functions.

NOT the pre-registered probe. ising-latent-v1 fits on seed 16 and tests on seeds 6-10, on a
checkpoint's 128-dimensional embedding. This reads what a latents table holds, the frame's first 8
principal components, so R^2 here is a lower bound on the embedding's, and the bound may differ
between a trained checkpoint and its twin: a margin here is not the margin the file judges. `jepa
probe` on a training's folder gives the real one (--fit the seed 16 staircase, --test the seeds 1-5
staircases) wherever its checkpoints are. With no latents for seed 16, the fit rotates: each seed's
staircase in turn fits, the other four test, and a value is the mean over the rotations.

Frames: the prereg's rule, 20,000 per run drawn without replacement with probe seed 0, from one
burst in 8 at every plateau (all a tc-plus table holds away from the FSS plateau; the FSS plateau is
thinned to the same, so plateaus weigh alike). l2var's plateau means use every saved frame, as
probes.frames does.

Floors:
  capacity   the iid control's R^2 on test: what a probe scores on noise
  T content  the shuffle-within-T control's R^2: all that temperature alone implies
  noise      the 95th percentile of |R^2(shuffle a) - R^2(shuffle b)| over --shuffles redraws of
             the control: a selectivity inside it cannot be told from 0
Selectivity = R^2(target) - R^2(shuffle_within_T), and of_rest = selectivity / (1 - R^2(shuffle)),
its share of what temperature leaves unexplained, which the raw number hides when T explains most.

Intervals: percentile, hierarchical bootstrap of the test frames (seeds, then replicas), the same
draws for every column and for trained and twin when run with one --seed, so the margin's draws pair
up. They hold the data's noise alone: this is one model seed, so nothing here measures the spread
between seeds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tipping_point.ews import ising
from tipping_point.jepa import probes

from explore import common
from explore.correlations import BURST, EVERY, ci, resamples

FEATURES = tuple(f"pc_{k}" for k in range(1, 9))
LIMIT = 20_000  # frames per run, the prereg's
PROBE_SEED = 0  # the prereg's
ALPHA = 1e-3  # the prereg's


def uniform_frames(
    run: ising.Run, table: str, limit: int, seed: int
) -> tuple[probes.Frames, np.ndarray]:
    """One burst in EVERY of a staircase's plateaus as probes.Frames, targets and all, and the
    frames' pcs: `limit` of them, drawn as probes.frames draws. Nothing reads its packed spins
    here."""
    run = common.allowed(run)
    with_latents, _ = ising.plateaus(run, "matched", table)
    every_frame, _ = ising.plateaus(run, "matched")
    area = float(run.L**2)
    parts = []
    for T, s in with_latents.items():
        whole = len(s.steps) == len(every_frame[T].steps)  # else: tc-plus has thinned it already
        keep = (
            (np.arange(len(s.steps)) // BURST) % EVERY == 0
            if whole
            else np.ones(len(s.steps), bool)
        )
        picked = s.take(np.flatnonzero(keep))
        mean = float(every_frame[T].abs_m.mean())
        replica = np.repeat(picked.replicas[:, None], len(picked.steps), axis=1)
        parts.append(
            (
                picked.abs_m.ravel(),
                picked.energy.ravel(),
                area * (picked.abs_m.ravel() - mean) ** 2,
                replica.ravel(),
                np.full(picked.abs_m.size, T),
                np.stack([picked.observable(c).ravel() for c in FEATURES], axis=1),
            )
        )
    abs_m, energy, l2var, replica, temperature, x = (
        np.concatenate(c) for c in zip(*parts, strict=True)
    )
    rows = np.arange(len(abs_m))
    if limit < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(len(rows), limit, replace=False))
    targets = {"abs_m": abs_m[rows], "energy": energy[rows], "l2var": l2var[rows]}
    frames = probes.Frames(run.L, torch.empty(0), targets, replica[rows], temperature[rows])
    return frames, x[rows]


def _replica_sums(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, count: int) -> np.ndarray:
    """Per test group: [n, Sy, Syy, SSE]."""
    return np.stack(
        [
            np.bincount(groups, minlength=count).astype(np.float64),
            np.bincount(groups, y, minlength=count),
            np.bincount(groups, y * y, minlength=count),
            np.bincount(groups, (y - pred) ** 2, minlength=count),
        ],
        axis=1,
    )


def _r2(sums: np.ndarray) -> np.ndarray:
    n, sy, syy, sse = np.moveaxis(sums, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):  # a draw that left a rotation no test seed
        return 1.0 - sse / (syy - sy * sy / n)


def _entry(point: Any, draws: np.ndarray) -> dict[str, Any]:
    lo, hi = ci(draws)
    return {"value": float(point), "ci": [float(lo), float(hi)]}


def analyse(args: Any, inputs: ising.Inputs) -> dict[str, Any]:
    runs = [common.allowed(r) for r in inputs.staircases]
    loaded = [uniform_frames(r, args.latents, args.limit, PROBE_SEED) for r in runs]
    frames, features = zip(*loaded, strict=True)
    # a group is (seed index, replica): the bootstrap's unit
    width = int(max(f.replica.max() for f in frames)) + 1
    seed_of = np.repeat(np.arange(len(runs)), width)
    groups = [i * width + f.replica.astype(np.int64) for i, f in enumerate(frames)]
    weights = resamples(seed_of, args.bootstrap, np.random.default_rng(args.seed))
    keys = [(t, c) for t in probes.TARGETS for c in probes.COLUMNS]
    # the rotations: each seed fits, the others test
    point = np.empty((len(runs), len(keys)))
    draws = np.empty((len(runs), args.bootstrap, len(keys)))
    noise: dict[str, list[float]] = {t: [] for t in probes.TARGETS}
    for k in range(len(runs)):
        others = [i for i in range(len(runs)) if i != k]
        fit, test = [frames[k]], [frames[i] for i in others]
        x_fit, x_test = features[k], np.concatenate([features[i] for i in others])
        group = np.concatenate([groups[i] for i in others])
        targets = probes.controls(fit, test, PROBE_SEED)
        y_fit = np.stack([targets[key][0] for key in keys], axis=1)
        pred = probes.ridge(x_fit, y_fit, ALPHA).predict(x_test)
        checks = probes.score(x_fit, x_test, targets, ALPHA)
        sums = np.stack(
            [
                _replica_sums(targets[key][1], pred[:, j], group, len(seed_of))
                for j, key in enumerate(keys)
            ],
            axis=1,
        )  # (groups, keys, 4)
        point[k] = _r2(sums.sum(0))
        by_key = {(r["target"], r["column"]): r["r2"] for r in checks}
        for j, key in enumerate(keys):
            if not np.isclose(point[k, j], by_key[key], rtol=1e-9, atol=1e-12):
                raise SystemExit(f"{key}: the sums give {point[k, j]}, probes.score {by_key[key]}")
        shaped = sums.reshape(len(seed_of), -1)
        draws[k] = _r2((weights @ shaped).reshape(args.bootstrap, len(keys), 4))
        # the selectivity's own noise: other draws of the shuffle control, this rotation's fit
        for s in range(1, args.shuffles + 1):
            again = probes.controls(fit, test, PROBE_SEED + s)
            redone = [(t, "shuffle_within_T") for t in probes.TARGETS]
            y = np.stack([again[key][0] for key in redone], axis=1)
            guess = probes.ridge(x_fit, y, ALPHA).predict(x_test)
            for j, (t, _) in enumerate(redone):
                noise[t].append(probes.r2(again[t, "shuffle_within_T"][1], guess[:, j]))
    mean_point, mean_draws = point.mean(0), np.nanmean(draws, axis=0)
    at = {key: j for j, key in enumerate(keys)}
    out: dict[str, Any] = {
        "features": list(FEATURES),
        "note": "8 principal components, not the 128-dimensional embedding the prereg probes",
        "frames_per_run": args.limit,
        "fit_test": "each seed fits, the other four test; a value is the mean over the rotations",
        "alpha": ALPHA,
        "probe_seed": PROBE_SEED,
        "bootstrap": args.bootstrap,
        "ci_method": "percentile, hierarchical bootstrap of the test frames: seeds, then replicas",
        "rng_seed": args.seed,
        "targets": {},
        "draws": {},
    }
    for t in probes.TARGETS:
        raw, iid, shuffle = (at[t, c] for c in probes.COLUMNS)
        sel_p = mean_point[raw] - mean_point[shuffle]
        sel_d = mean_draws[:, raw] - mean_draws[:, shuffle]
        spread = np.array(noise[t])
        pairs = np.abs(spread[None, :] - spread[:, None])[np.triu_indices(len(spread), 1)]
        out["targets"][t] = {
            "r2": _entry(mean_point[raw], mean_draws[:, raw]),
            "iid": _entry(mean_point[iid], mean_draws[:, iid]),
            "shuffle_within_T": _entry(mean_point[shuffle], mean_draws[:, shuffle]),
            "selectivity": _entry(sel_p, sel_d),
            "of_rest": _entry(
                sel_p / (1 - mean_point[shuffle]), sel_d / (1 - mean_draws[:, shuffle])
            ),
            "per_rotation": {c: point[:, at[t, c]].tolist() for c in probes.COLUMNS},
            "selectivity_noise_q95": float(np.quantile(pairs, 0.95))
            if len(pairs)
            else float("nan"),
            "shuffle_redraws": len(spread),
        }
        out["draws"][t] = {"r2": mean_draws[:, raw], "selectivity": sel_d}
    return out


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--limit", type=int, default=LIMIT)
    p.add_argument("--shuffles", type=int, default=8, help="redraws per rotation, for the noise")
    args = p.parse_args(argv)
    if not args.latents:
        raise SystemExit("probes_explore reads a latents table: name it with --latents")
    common.be_nice()
    clock = common.Clock()
    doc = common.make_doc(args)
    pilot, inputs = common.load(args, doc)
    out = analyse(args, inputs)
    draws = out.pop("draws")
    for t, v in out["targets"].items():
        clock(
            f"{t:<6} R^2 {v['r2']['value']:.4f} {np.round(v['r2']['ci'], 4)} "
            f"iid {v['iid']['value']:+.4f} shuffle {v['shuffle_within_T']['value']:.4f} "
            f"selectivity {v['selectivity']['value']:+.4f} {np.round(v['selectivity']['ci'], 4)} "
            f"noise {v['selectivity_noise_q95']:.4f}"
        )
    folder = common.out_dir(args)
    common.save(
        folder / "probes.json",
        {"provenance": common.provenance(args, pilot), "runs": common.run_ids(inputs)} | out,
    )
    np.savez_compressed(
        folder / "probes-draws.npz",
        **{f"{t}_{q}": v for t, d in draws.items() for q, v in d.items()},
    )
