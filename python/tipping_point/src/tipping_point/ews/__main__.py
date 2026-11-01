"""Command line: python -m tipping_point.ews <command> [options]

  calibrate            the surrogate test's false-positive rate on synthetic nulls; the long
                       version of the gate's check, run before each freeze (exit 1 off the band)
  score --prereg FILE  score a frozen pre-registration: results to <data dir>/ews/<run_id>/, one
                       line per evaluated cell to <data dir>/ews/trials.jsonl; a stale pilot
                       record is refused unless --allow-stale-pilot, and so is any scored run
                       not made by the freeze tag's commit from a clean tree, any ramp or null
                       run not made against the pilot record the prereg pins, or any staircase
                       whose dwell was not read from the tau.json it pins at that L. A prereg with
                       a [latents] section reads the runs of the version its [baseline] names, so
                       they must come from that tag's commit, and scores each latents table it
                       pins, made from the pinned checkpoint by its own tag's commit; it is scored
                       whole by harness/ising_experiment.py, so here it is refused without
                       --cells-only, which would spend its trial budget on a partial result. A
                       result of the same version with no complete.json, torn by a crash, refuses
                       the scoring until it is deleted or the file amended (prereg.charge)
  prereg freeze ID     tag the commit holding paper/prereg/ID-v<k>.toml as prereg/ID-v<k>

The data dir is --data-dir, else the engine's own ($TP_DATA_DIR, or a per-user folder outside
the repository: tipping_point.engine.data_dir).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import engine
from . import ising, prereg, scoring, surrogates
from .trend import Pipeline

NULLS = ("ar1", "fast_slow")
DATA_DIR_HELP = "default: $TP_DATA_DIR, else the engine's per-user data dir"
CELL = (
    "checkpoint",
    "indicator",
    "observable",
    "cadence",
    "L",
    "h",
    "window_taus",
    "bandwidth_taus",
    "headline",
)


def null_series(kind: str, length: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Calibration nulls, unit variance per mode: AR(1) with phi = 0.9, or a fast plus a slow
    mode (phi = 0.5 and 0.995), the slow one standing in for the dynamics near Tc."""
    phis = {"ar1": (0.9,), "fast_slow": (0.5, 0.995)}[kind]
    total = np.zeros((n, length))
    for phi in phis:
        model = surrogates.AR1(np.array(phi), np.array(math.sqrt(1 - phi**2)), np.array(0.0))
        total += surrogates.ar1(model, length, n, rng)
    return total


def calibration(
    nulls: int,
    draws: int,
    seed: int,
    kinds: Sequence[str] = NULLS,
    indicators: Sequence[str] = ("variance", "ac"),
    tests: Sequence[str] = ("phase", "ar1"),
    window: int = 50,
    length: int = 256,
    bandwidth: float = 50.0,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """For each null kind, indicator and surrogate test: the share of nulls with p <= alpha, and
    the 99% binomial band it must sit in."""
    half = 2.576 * math.sqrt(alpha * (1 - alpha) / nulls)
    rows = []
    for k, kind in enumerate(kinds):
        x = null_series(kind, window + length - 1, nulls, np.random.default_rng([seed, k]))
        for indicator in indicators:
            pipe = Pipeline(indicator, window, length, bandwidth)
            for test in tests:
                rng = np.random.default_rng([seed, k, len(rows)])
                _, p = scoring.surrogate_test(x, pipe, surrogates.KINDS[test], draws, rng)
                share = float(np.mean(p <= alpha))
                rows.append(
                    {
                        "null": kind,
                        "indicator": indicator,
                        "surrogate": test,
                        "false_positive": share,
                        "band": (alpha - half, alpha + half),
                        "inside": abs(share - alpha) <= half,
                    }
                )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.command(args)
    except (prereg.PreregError, ising.IsingDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tipping_point.ews", description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("calibrate", help="surrogate-test false positives on synthetic nulls")
    p.add_argument("--nulls", type=int, default=2000)
    p.add_argument("--surrogates", type=int, default=999)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(command=_calibrate)

    p = sub.add_parser("score", help="score a frozen pre-registration")
    p.add_argument("--prereg", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--seed", type=int, default=26, help="bootstrap and surrogate draws")
    p.add_argument(
        "--allow-stale-pilot",
        action="store_true",
        help="score with a pilot record that did not measure the running kernel (stamped stale)",
    )
    p.add_argument(
        "--cells-only",
        action="store_true",
        help="a latent file's cell tables alone, without the probes and margins of "
        "harness/ising_experiment.py, spending its trials all the same",
    )
    p.set_defaults(command=_score)

    p = sub.add_parser("prereg", help="pre-registration files")
    actions = p.add_subparsers(required=True)
    f = actions.add_parser("freeze", help="tag the latest version of a pre-registration")
    f.add_argument("id")
    f.add_argument("--root", type=Path, help="repository root (default: the current one)")
    f.set_defaults(command=_freeze)
    return parser


def _calibrate(args: argparse.Namespace) -> int:
    rows = calibration(args.nulls, args.surrogates, args.seed)
    for r in rows:
        lo, hi = r["band"]
        verdict = "inside" if r["inside"] else "OUTSIDE"
        print(
            f"{r['null']:<10} {r['indicator']:<9} {r['surrogate']:<6} "
            f"false positives {100 * r['false_positive']:5.2f}%  "
            f"99% band {100 * lo:.2f}-{100 * hi:.2f}%  {verdict}"
        )
    return 0 if all(r["inside"] for r in rows) else 1


def score(
    pre: prereg.Prereg,
    ref: prereg.Frozen,
    data_dir: Path,
    seed: int,
    allow_stale: bool = False,
    more: Callable[[ising.Evaluation], dict[str, list[dict[str, Any]]]] | None = None,
) -> tuple[Path, ising.Evaluation, int]:
    """One scoring run: refused past the trial budget, evaluated, then written and logged as one
    (prereg.write_result, whose last step is complete.json). `more` adds tables from the evaluation
    (harness/ising_experiment.py: a latent file's probes and margins). A ref without a commit, a
    scoring of planted data only that runner allows, asks no commit of the runs. Returns the
    result's folder, the evaluation and the trials logged before it."""
    planned = prereg.cells(pre.doc)
    logged = prereg.charge(data_dir, pre, planned)
    if pre.doc["data"]["domain"] != "ising":
        raise prereg.PreregError(f"no scorer for domain {pre.doc['data']['domain']!r} yet")
    rng = np.random.default_rng(seed)
    evaluation = ising.evaluate(
        pre.doc,
        data_dir,
        rng,
        allow_stale=allow_stale,
        commit=ref.commit or None,
        runs_commit=prereg.baseline_commit(pre) if ref.commit else None,
    )
    if len(evaluation.tables["auc"]) != planned:  # the budget is only as good as this count
        raise prereg.PreregError(
            f"{pre.name}: {len(evaluation.tables['auc'])} AUC cells evaluated, {planned} counted"
        )
    stamp = prereg.stamp(pre, ref)
    pilot = evaluation.pilot
    params = {
        "seed": seed,
        "pilot": {
            "run_id": pilot.path.parent.name,
            "sha256": pilot.sha256,
            "kernel_commit": pilot.kernel_commit,
            "kernel_sha256": pilot.kernel_sha256,
            "stale": pilot.stale is not None,
            "tau_c": {str(L): tau for L, tau in sorted(pilot.tau_c.items())},
            "tau_c_ci": {str(L): ci for L, ci in sorted(pilot.tau_c_ci.items())},
        },
        "dropped": evaluation.dropped,
    }
    inputs = [  # pilot: the stamp of the record a ramp or null run was made with (#55)
        {
            "domain": "ising",
            "run_id": r.run_id,
            "complete_sha256": r.complete_sha256,
            "pilot": r.params.get("pilot"),
            **({"latents": tables} if (tables := evaluation.latents.get(r.run_id)) else {}),
        }
        for r in evaluation.inputs
    ]
    tables = evaluation.tables | (more(evaluation) if more else {})
    trials = [
        {
            "prereg": stamp,
            "cell": {key: row[key] for key in CELL},
            "auc": row["auc"],
            "auc_ci": [row["auc_lo"], row["auc_hi"]],
            "n_pos": row["n_pos"],
            "n_neg": row["n_neg"],
        }
        for row in evaluation.tables["auc"]
    ]
    run_dir = prereg.write_result(data_dir, pre, ref, tables, inputs, params, trials)
    return run_dir, evaluation, logged


def _score(args: argparse.Namespace) -> int:
    pre = prereg.load(args.prereg)
    ref = prereg.frozen(pre)
    planned = prereg.cells(pre.doc)
    if "latents" in pre.doc and not args.cells_only:
        raise prereg.PreregError(
            f"{pre.name} has a [latents] section: harness/ising_experiment.py scores it whole, "
            f"with its probes and margins. This would spend its {planned} trials on the cells "
            "alone; --cells-only says that is meant"
        )
    run_dir, evaluation, logged = score(pre, ref, args.data_dir, args.seed, args.allow_stale_pilot)
    for row in evaluation.tables["auc"]:
        if row["headline"]:
            print(
                f"{row['indicator']:<9} {row['observable']:<7} {row['cadence']:<8} "
                f"L={row['L']:<4} h={row['h']:g}  AUC {row['auc']:.3f} "
                f"[{row['auc_lo']:.3f}, {row['auc_hi']:.3f}]"
            )
    for row in evaluation.tables["fss"]:
        print(
            f"fss {row['quantity']:<22} {row['cadence']:<8} kappa {row['estimate']:+.3f} "
            f"[{row['lo']:+.3f}, {row['hi']:+.3f}]  {row['verdict']}"
        )
    budget = pre.doc["trials"]["budget"]
    print(f"{run_dir}  ({logged + planned} of {budget} trials logged)")
    return 0


def _freeze(args: argparse.Namespace) -> int:
    root = args.root or prereg.repo_root(Path.cwd())
    print(prereg.freeze(root, args.id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
