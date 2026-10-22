"""Command line: python -m tipping_point.ising <command> [options]

  staircase  quasi-static sweep down the 40-point grid, dwell set by a pilot's tau.json
  ramp       linear cooling 3.5 -> 1.5 at a Kibble-Zurek point x (or an explicit rate)
  pilot      measure tau_int of |m| and e at given temperatures; writes tau.json
  validate   Gates A-C and the independence check on a small L = 4 run; exit 1 on a failure
  resume     continue a crashed or stopped run from its checkpoint

Runs land in <data dir>/ising/<run_id>/; the data dir is --data-dir or $TP_DATA_DIR.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import autocorr, io, protocol, run


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != _resume and args.data_dir is None:
        parser.error("set TP_DATA_DIR or pass --data-dir")
    return args.command(args)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=1)
    common.add_argument(
        "--device", default=run.default_device(), help="cuda if available, else cpu"
    )
    common.add_argument("--data-dir", type=Path, default=os.environ.get("TP_DATA_DIR"))
    common.add_argument("--checkpoint-every", type=int, default=100_000, help="sweeps")
    sized = argparse.ArgumentParser(add_help=False, parents=[common])
    sized.add_argument("--L", type=int, required=True)
    sized.add_argument("--replicas", type=int, default=16, help="per temperature")
    saving = argparse.ArgumentParser(add_help=False)
    saving.add_argument("--burst-stride", type=int, default=1, help="sweeps between burst frames")

    parser = argparse.ArgumentParser(prog="python -m tipping_point.ising", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("staircase", parents=[sized, saving], help="quasi-static sweep")
    p.add_argument("--tau", required=True, help="one tau for every T, or a pilot's tau.json")
    p.add_argument("--saves", type=int, default=200, help="bursts per temperature")
    p.add_argument("--warmup", type=int, default=1000, help="extra sweeps at the first T")
    p.add_argument("--init", choices=["hot", "cold"], default="hot")
    p.set_defaults(command=_staircase)

    p = sub.add_parser("ramp", parents=[sized, saving], help="linear cooling ramp")
    speed = p.add_mutually_exclusive_group(required=True)
    speed.add_argument("--x", type=float, help="Kibble-Zurek point: 30 fast, 0.3 slow")
    speed.add_argument("--rate", type=int, help="sweeps per unit T")
    p.add_argument("--pre", type=int, default=1000, help="equilibration sweeps at 3.5")
    p.add_argument("--hold", type=int, default=1000, help="sweeps held at 1.5")
    p.set_defaults(command=_ramp)

    p = sub.add_parser("pilot", parents=[sized], help="measure tau_int")
    p.add_argument(
        "--T", type=float, action="append", help="repeatable; default: the 40-point grid"
    )
    p.add_argument("--therm", type=int, default=2000)
    p.add_argument("--sweeps", type=int, default=20_000, help="measured sweeps")
    p.set_defaults(command=_pilot)

    p = sub.add_parser("validate", parents=[common], help="Gates A-C on a small L = 4 run")
    p.add_argument("--replicas", type=int, default=4096, help="per temperature")
    p.add_argument("--therm", type=int, default=400)
    p.add_argument("--sweeps", type=int, default=400)
    p.set_defaults(command=_validate)

    p = sub.add_parser("resume", help="continue a run from its checkpoint")
    p.add_argument("run_dir", type=Path)
    p.set_defaults(command=_resume)
    return parser


def _spec(args: argparse.Namespace) -> run.Spec:
    init = getattr(args, "init", "hot")
    return run.Spec(args.L, args.replicas, args.seed, init, checkpoint_every=args.checkpoint_every)


def _taus(value: str, temperatures: Sequence[float]) -> list[float]:
    """--tau: one number for every temperature, or a pilot's tau.json that covers them all with
    reliable points. A pilot too short for its tau underestimates it, and the dwell would end
    before the plateau equilibrates."""
    try:
        return [float(value)] * len(temperatures)
    except ValueError:
        pass
    points = {round(p["temperature"], 9): p for p in io.read_json(Path(value))["points"]}
    missing = [T for T in temperatures if round(T, 9) not in points]
    if missing:
        raise SystemExit(f"{value} has no tau for T = {missing}")
    unreliable = [T for T in temperatures if not points[round(T, 9)].get("reliable", False)]
    if unreliable:
        raise SystemExit(f"{value}: tau is unreliable at T = {unreliable}; run the pilot longer")
    return [points[round(T, 9)]["tau"] for T in temperatures]


def _staircase(args: argparse.Namespace) -> int:
    burst = protocol.Burst(stride=args.burst_stride)
    taus = _taus(args.tau, protocol.GRID)
    schedule = protocol.Staircase(taus, warmup=args.warmup, saves=args.saves, burst=burst)
    tag = f"L{args.L}-qs-s{args.seed}"
    print(run.start(_spec(args), schedule, args.data_dir, tag, args.device))
    return 0


def _ramp(args: argparse.Namespace) -> int:
    rate = args.rate if args.x is None else protocol.rate_for(args.L, args.x)
    burst = protocol.Burst(stride=args.burst_stride)
    schedule = protocol.Ramp(rate, pre=args.pre, hold=args.hold, burst=burst, x=args.x)
    speed = f"r{rate}" if args.x is None else f"x{args.x:g}"
    tag = f"L{args.L}-ramp-{speed}-s{args.seed}"
    print(run.start(_spec(args), schedule, args.data_dir, tag, args.device))
    return 0


def _pilot(args: argparse.Namespace) -> int:
    schedule = protocol.Fixed(args.T or protocol.GRID, args.therm, args.sweeps)
    clock = time.perf_counter()
    run_dir = run.start(
        _spec(args), schedule, args.data_dir, f"L{args.L}-pilot-s{args.seed}", args.device
    )
    seconds = time.perf_counter() - clock
    points = _measure(run_dir, args.L)
    io.write_json(run_dir / "tau.json", {"L": args.L, "seconds": seconds, "points": points})
    for p in points:
        samples = f"{p['sweeps'] / p['tau']:.0f}" if p["tau"] > 0 else "n/a"
        print(
            f"T={p['temperature']:.4f}  tau(|m|)={p['tau']:.1f} (window {p['window']}, "
            f"binning {p['binning']:.1f})  tau(e)={p['tau_energy']:.1f}  "
            f"N/tau={samples}{'' if p['reliable'] else '  UNRELIABLE'}"
        )
    print(f"{run_dir}  ({seconds:.1f} s)")
    return 0


def _measure(run_dir: Path, L: int) -> list[dict[str, Any]]:
    """tau of |m| and of e per temperature, from the measured (phase 1) stats rows."""
    stats = io.read_table(run_dir, "stats")
    columns = {name: stats[name].to_numpy() for name in stats.column_names}
    points = []
    for T in np.unique(columns["temperature"]):
        mine = (columns["phase"] == 1) & (columns["temperature"] == T)
        steps = len(np.unique(columns["step"][mine]))
        # rows are in (step, replica) order: reshape, then one row per replica
        m = np.abs(columns["magnetisation"][mine]).reshape(steps, -1).T / L**2
        e = columns["energy"][mine].reshape(steps, -1).T / L**2
        tau_m, tau_e = autocorr.estimate(m), autocorr.estimate(e)
        points.append(
            {
                "temperature": float(T),
                "tau": tau_m.tau,
                "window": tau_m.window,
                "binning": tau_m.binning,
                "tau_energy": tau_e.tau,
                "sweeps": steps,
                "replicas": m.shape[0],
                "reliable": tau_m.reliable,
            }
        )
    return points


def _validate(args: argparse.Namespace) -> int:
    run_dir, checks = run.validate(
        args.data_dir, args.seed, args.replicas, args.therm, args.sweeps, args.device
    )
    for c in checks:
        verdict = "PASS" if c.passed else "FAIL"
        print(f"{c.gate:<12} T={c.temperature:.4f}  {verdict}  {c.detail}")
    print(run_dir)
    return 0 if all(c.passed for c in checks) else 1


def _resume(args: argparse.Namespace) -> int:
    print(run.resume(args.run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
