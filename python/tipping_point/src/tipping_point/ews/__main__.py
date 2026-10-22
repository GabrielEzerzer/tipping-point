"""Command line: python -m tipping_point.ews <command> [options]

  calibrate            the surrogate test's false-positive rate on synthetic nulls; the long
                       version of the gate's check, run before each freeze (exit 1 off the band)
  score --prereg FILE  score a frozen pre-registration: results to <data dir>/ews/<run_id>/, one
                       line per evaluated cell to <data dir>/ews/trials.jsonl
  prereg freeze ID     tag the commit holding paper/prereg/ID-v<k>.toml as prereg/ID-v<k>

The data dir is --data-dir or $TP_DATA_DIR.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import ising, prereg, scoring, surrogates
from .trend import Pipeline

NULLS = ("ar1", "fast_slow")
CELL = ("indicator", "observable", "cadence", "L", "h", "window_taus", "bandwidth_taus", "headline")


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
    if getattr(args, "needs_data", False) and args.data_dir is None:
        parser.error("set TP_DATA_DIR or pass --data-dir")
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
    p.add_argument("--data-dir", type=Path, default=os.environ.get("TP_DATA_DIR"))
    p.add_argument("--seed", type=int, default=26, help="bootstrap and surrogate draws")
    p.set_defaults(command=_score, needs_data=True)

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


def _score(args: argparse.Namespace) -> int:
    pre = prereg.load(args.prereg)
    ref = prereg.frozen(pre)
    planned, logged = prereg.cells(pre.doc), prereg.trials(args.data_dir, pre)
    budget = pre.doc["trials"]["budget"]
    if logged + planned > budget:
        raise prereg.PreregError(
            f"{pre.name}: {logged} trials logged and {planned} more would exceed its budget of "
            f"{budget}; amend the pre-registration to score again"
        )
    if pre.doc["data"]["domain"] != "ising":
        raise prereg.PreregError(f"no scorer for domain {pre.doc['data']['domain']!r} yet")
    evaluation = ising.evaluate(pre.doc, args.data_dir, np.random.default_rng(args.seed))
    stamp = prereg.stamp(pre, ref)
    pilot = evaluation.pilot
    params = {
        "seed": args.seed,
        "pilot": {
            "run_id": pilot.path.parent.name,
            "sha256": pilot.sha256,
            "kernel_commit": pilot.kernel_commit,
            "tau_c": {str(L): tau for L, tau in sorted(pilot.tau_c.items())},
        },
        "dropped": evaluation.dropped,
    }
    inputs = [
        {"domain": "ising", "run_id": r.run_id, "complete_sha256": r.complete_sha256}
        for r in evaluation.inputs
    ]
    run_dir = prereg.write_result(args.data_dir, pre, ref, evaluation.tables, inputs, params)
    for row in evaluation.tables["auc"]:
        prereg.log_trial(
            args.data_dir,
            {
                "prereg": stamp,
                "result": run_dir.name,
                "cell": {key: row[key] for key in CELL},
                "auc": row["auc"],
                "auc_ci": [row["auc_lo"], row["auc_hi"]],
                "n_pos": row["n_pos"],
                "n_neg": row["n_neg"],
            },
        )
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
    print(f"{run_dir}  ({logged + planned} of {budget} trials logged)")
    return 0


def _freeze(args: argparse.Namespace) -> int:
    root = args.root or prereg.repo_root(Path.cwd())
    print(prereg.freeze(root, args.id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
