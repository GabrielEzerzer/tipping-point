"""Exploratory scoring of the seeds 1-5 datasets (#26, #28): shared setup. NOT pre-registered.

Every number this pipeline prints is exploratory. No prereg is created or frozen, nothing is tagged,
nothing is ever logged to ews/trials.jsonl, and production code is only imported, never edited. The
CLI's freeze gate (`ews score` refuses an unfrozen prereg) is bypassed by calling the ews library
directly on a copy of paper/prereg/ising-ews-v1.toml held in memory, with these overrides:
  data.seeds = 1-5, the only seeds this code reads: anything else is refused, here and at every run
    it opens (allowed), so the confirmatory seeds 6-10 are never touched
  data.L = [L]; data.replicas = 16; data.ramps.eps_L dropped, since these ramps cool from T = 3.5
    to 1.5 rather than over the v1 window
  surrogates N_s = 99 (not 999, #70)
  indicators, without --latents: |m| only, dense cadence only, with its three indicators
  indicators, with --latents <table>: the pairs ising-latent-v1 judges, at the matched cadence
data.tau_c stays the #54 record, which ising-ews-v1 pins: the numbers on #26 used #50's, and
--pilot-run 20261022T182733Z-tauc-L16-32-64 --pilot-sha256 07a0ce67... gives them back (2-5% of a
window at these sizes).

From the worktree root, a pinned commit, one job at a time (the laptop is shared):
  uv run --no-sync python harness/explore <script> --L 16 [--latents <training>-step<step>] ...
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from tipping_point.ews import ising, trend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # the worktree root: harness/explore -> root
PREREG = ROOT / "paper" / "prereg" / "ising-ews-v1.toml"
PILOT_RUN = "20261025T194846Z-tauc-L16-32-64-128"
PILOT_SHA256 = "7c5a4a296f2d8d1d55e223e4c775cd98267c43e6ae89f5e6ec594ac5c0581698"
OUT = ROOT / "harness" / "out" / "explore"  # git-ignored; --out moves it
THREADS = 4  # the laptop is shared with other lanes
SEEDS = (1, 2, 3, 4, 5)  # all this pipeline reads: 6-10 are what the pre-registrations score
CLASSIC_PAIRS = (("abs_m", "variance"), ("abs_m", "ac"), ("abs_m", "skewness"))
LATENT_PAIRS = (
    ("pc_1", "variance"),
    ("pc_1", "ac"),
    ("surprise", "variance"),
    ("surprise", "level"),
)
UNSET = "unset"  # surprise's direction before this pass has read it: an AUC does not need one
JUDGED_COLUMNS = ("surprise", "pc_1")
ALL_COLUMNS = ("surprise", *(f"pc_{k}" for k in range(1, 9)))


class Refused(SystemExit):
    """A seed this pipeline never reads, or a run it must not open."""


def refuse(seed: int, what: str) -> Refused:
    return Refused(
        f"{what} has seed {seed}: this pipeline is exploratory and reads seeds {SEEDS[0]}-"
        f"{SEEDS[-1]} alone. Seeds 6-10 are what ising-ews-v1 and ising-latent-v1 score."
    )


def check_seeds(seeds: list[int]) -> list[int]:
    for seed in seeds:
        if seed not in SEEDS:
            raise refuse(seed, "--seeds")
    return seeds


def allowed(run: ising.Run) -> ising.Run:
    """The run, if this pipeline may read it. Every run a script opens passes through here."""
    if run.seed not in SEEDS:
        raise refuse(run.seed, run.run_id)
    return run


def be_nice() -> None:
    """Below-normal priority and at most THREADS worker threads for the Kendall-tau pool."""
    trend.THREADS = THREADS
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p  # a 64-bit pseudo-handle
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000):  # BELOW_NORMAL
            print("warning: could not lower the process priority", file=sys.stderr)
    else:
        os.nice(10)


def data_dir() -> Path:
    env = os.environ.get("TP_DATA_DIR")
    if env:
        return Path(env)
    return Path(os.environ["LOCALAPPDATA"]) / "tipping-point"


def root_of(args: argparse.Namespace) -> Path:
    return args.data_dir or data_dir()


def parser(description: str | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--latents", help="a table under each run's latents/, <training>-step<step>")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=OUT, help="outputs go under here")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="from 1-5, no others")
    p.add_argument("--replicas", type=int, default=16)
    p.add_argument("--pilot-run", default=PILOT_RUN)
    p.add_argument("--pilot-sha256", default=PILOT_SHA256)
    p.add_argument("--horizons", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    p.add_argument("--seed", type=int, default=28, help="the RNG's")
    p.add_argument("--surprise-variance", choices=("greater", "less"), default=UNSET)
    p.add_argument("--surprise-level", choices=("greater", "less"), default=UNSET)
    return p


def make_doc(args: argparse.Namespace, n_surrogates: int = 99) -> dict[str, Any]:
    """ising-ews-v1 with the exploratory overrides listed in the module docstring."""
    doc = tomllib.loads(PREREG.read_text("utf-8"))
    data = doc["data"]
    data["L"] = [args.L]
    data["seeds"] = check_seeds(list(args.seeds))
    data["replicas"] = args.replicas
    data["ramps"].pop("eps_L", None)
    data["tau_c"]["pilot_run"] = args.pilot_run
    data["tau_c"]["pilot_sha256"] = args.pilot_sha256
    doc["event"]["horizons"] = [float(h) for h in args.horizons]
    doc["scoring"]["surrogates"]["n"] = n_surrogates
    ind = doc["indicators"]
    if args.latents:
        ind["observables"] = sorted({observable for observable, _ in LATENT_PAIRS})
        ind["cadences"] = ["matched"]
        ind["names"] = sorted({name for _, name in LATENT_PAIRS})
        ind["pairs"] = [list(pair) for pair in LATENT_PAIRS]
        ind["direction"] = {"variance": "greater", "ac": "greater", "level": args.surprise_level}
        ind["direction_by_observable"] = {
            "surprise": {"variance": args.surprise_variance, "level": args.surprise_level}
        }
    else:
        ind["observables"] = ["abs_m"]
        ind["cadences"] = ["dense"]
        doc["fss"]["dynamic"]["indicators"] = [name for _, name in CLASSIC_PAIRS]
    return doc


def cadence(args: argparse.Namespace) -> str:
    return "matched" if args.latents else "dense"


def pairs(args: argparse.Namespace, doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(observable, indicator, direction) for each pair this mode scores."""
    ind = doc["indicators"]
    chosen = LATENT_PAIRS if args.latents else CLASSIC_PAIRS
    return [(o, name, ising._direction(ind, o, name)) for o, name in chosen]


def oriented_pairs(args: argparse.Namespace, doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    """pairs(), for a script a direction matters to (a one-sided test, an alarm): refused while
    surprise's is unset."""
    chosen = pairs(args, doc)
    if any(direction == UNSET for *_, direction in chosen):
        raise SystemExit("pass --surprise-variance and --surprise-level: the pass reads them first")
    return chosen


def load(args: argparse.Namespace, doc: dict[str, Any]) -> tuple[ising.Pilot, ising.Inputs]:
    """The pilot record and the L's runs, picked as ising.select picks them, each with its latents
    table when one is named: exactly one complete run per slot, thermalised verdicts required,
    failed blocks dropped."""
    root = root_of(args)
    pilot = ising.read_pilot(root, args.pilot_run, args.pilot_sha256, allow_stale=True)
    names = [args.latents] if args.latents else []
    chosen = ising.select(*ising.find_runs(root), doc["data"], names)[args.L]
    for run in chosen.runs():
        allowed(run)
    return pilot, dataclasses.replace(chosen, latents=args.latents)


def run_ids(inputs: ising.Inputs) -> dict[str, list[str]]:
    return {
        "ramps": [r.run_id for r in inputs.ramps],
        "nulls": [f"{r.run_id}#{b}" for r, b in inputs.nulls],
        "staircases": [r.run_id for r in inputs.staircases],
        "dropped_null_blocks": list(inputs.dropped),
    }


def _git(*args: str) -> str:
    done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return done.stdout.strip()


def provenance(args: argparse.Namespace, pilot: ising.Pilot | None = None) -> dict[str, Any]:
    out = {
        "exploratory": True,
        "pre_registered": False,
        "code_commit": _git("rev-parse", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "prereg_base": "paper/prereg/ising-ews-v1.toml, overrides in harness/explore/common.py",
        "prereg_base_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "L": getattr(args, "L", None),
        "latents": getattr(args, "latents", None),
        "seeds": getattr(args, "seeds", list(SEEDS)),
        "rng_seed": args.seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if pilot is not None:
        out["pilot"] = {
            "run_id": pilot.path.parent.name,
            "sha256": pilot.sha256,
            "stale": pilot.stale,
            "tau_c": pilot.tau(args.L),
            "tau_c_ci": pilot.tau_c_ci.get(args.L),
        }
    return out


def table_dir(args: argparse.Namespace) -> Path:
    """<out>/<latents table, or classic>/: what spans sizes lands here."""
    d = Path(args.out) / (args.latents or "classic")
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(args: argparse.Namespace, L: int | None = None) -> Path:
    """<out>/<latents table, or classic>/L<L>/."""
    d = table_dir(args) / f"L{args.L if L is None else L}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read(args: argparse.Namespace, L: int, name: str) -> dict[str, Any]:
    return json.loads((out_dir(args, L) / name).read_text("utf-8"))


def _default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, default=_default, allow_nan=True), "utf-8")
    print(f"wrote {path}", flush=True)


class Clock:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def __call__(self, what: str) -> None:
        print(f"[{time.perf_counter() - self.t0:7.1f} s] {what}", flush=True)
