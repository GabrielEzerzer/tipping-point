"""One training's exploratory latent pass: every step of the pipeline in order, on its trained table
and its untrained twin, at each size. NOT pre-registered: nothing is logged as a trial, and only
seeds 1-5 are read.

The core steps, about 15 minutes on the laptop at L = 16 and 32, answer what the pass is for; the
surrogate tests and the alarm tables take hours on this machine and answer nothing a draft needs, so
they run only with --long, once surprise's directions are read from the core steps' outputs.

  uv run --no-sync python harness/explore latent_pass --training <run id> --step 10000
      --data-dir <the join> --out <folder> [--sizes 16 32] [--long]
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from tipping_point.ews.ising import latents_name

from explore import common, latent_report

CORE = (
    "static_tc",
    "auc_alarms",
    "peaks",
    "correlations",
    "probes_explore",
    "nulls_explore",
    "fast_ramps",
)
LONG = ("surrogates_cal", "alarms_explore")


def _run(name: str, *argv: Any) -> None:
    print(f"== {name} {' '.join(map(str, argv))}", flush=True)
    importlib.import_module(f"explore.{name}").main([str(a) for a in argv])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--training", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=common.OUT)
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 32])
    p.add_argument("--seeds", type=int, nargs="+", default=list(common.SEEDS))
    p.add_argument("--replicas", type=int, default=16)
    p.add_argument("--pilot-run", default=common.PILOT_RUN)
    p.add_argument("--pilot-sha256", default=common.PILOT_SHA256)
    p.add_argument("--long", action="store_true", help="the surrogate tests and alarm tables too")
    p.add_argument("--headline-only", action="store_true", help="AUC at the headline cell alone")
    p.add_argument("--extra", nargs="*", default=[], help=argparse.SUPPRESS)  # a smoke test's knobs
    args = p.parse_args(argv)
    common.check_seeds(args.seeds)
    trained = latents_name(args.training, args.step)
    twin = latents_name(args.training, 0)
    shared = [
        "--out",
        args.out,
        "--seeds",
        *args.seeds,
        "--replicas",
        args.replicas,
        "--pilot-run",
        args.pilot_run,
        "--pilot-sha256",
        args.pilot_sha256,
    ]
    if args.data_dir is not None:
        shared += ["--data-dir", args.data_dir]
    knobs = {
        "auc_alarms": [
            "--alarm-cells",
            "none",
            *(["--headline-only"] if args.headline_only else []),
        ]
    }
    for table in (trained, twin):
        for L in args.sizes:
            for step in CORE:
                _run(step, "--L", L, "--latents", table, *shared, *knobs.get(step, []), *args.extra)
        _run("scaling", "--sizes", *args.sizes, "--latents", table, *shared)
        _run("peaks", "--combine", *args.sizes, "--latents", table, *shared)
        for column in common.JUDGED_COLUMNS:
            _run(
                "fixed_eps", "--sizes", *args.sizes, "--column", column, "--latents", table, *shared
            )
    directions = latent_report.directions(args.out, trained, args.sizes)
    print(f"surprise's directions, read from the trained table: {directions}", flush=True)
    if args.long:
        oriented = [
            "--surprise-variance",
            directions["variance"],
            "--surprise-level",
            directions["level"],
        ]
        for table in (trained, twin):
            for L in args.sizes:
                for step in LONG:
                    _run(step, "--L", L, "--latents", table, *shared, *oriented)
    for table in (trained, twin):
        for L in args.sizes:
            _run("report", "--L", L, "--latents", table, *shared)
    _run(
        "latent_report",
        "--training",
        args.training,
        "--step",
        args.step,
        "--out",
        args.out,
        "--sizes",
        *args.sizes,
    )
