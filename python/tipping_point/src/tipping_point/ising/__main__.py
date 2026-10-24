"""Command line: python -m tipping_point.ising <command> [options]

  staircase  quasi-static sweep down the 40-point grid, dwell set by a pilot's tau.json
  ramp       linear cooling 3.5 -> 1.5 at a Kibble-Zurek point x of a pilot record's tau_c
             (or an explicit rate)
  null       fixed-T null run at epsilon_L = 2, 4 or 6: 700 tau_c of a size a pilot record measured
             (ramp and null refuse a stale record, one that did not measure the running kernel,
             unless --allow-stale-pilot; their manifests record the record as params.pilot)
  pilot      measure tau_int of |m| and e at given temperatures; writes tau.json
  tauc       tau_c(L) at Tc from pilot runs (one per L) and its fit A L^z; writes pilot.json
  validate   Gates A-C and the independence check on a small L = 4 run; exit 1 on a failure
  resume     continue a crashed or stopped run from its checkpoint

Runs land in <data dir>/ising/<run_id>/. The data dir is --data-dir, else the engine's own
($TP_DATA_DIR, or a per-user folder outside the repository: tipping_point.engine.data_dir). A
run id is the UTC start plus a tag in the engine's charset, a number's '.' written 'p' (x = 0.02
is x0p02), so the engine can open every run.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import engine
from . import autocorr, io, pilot, protocol, run

DATA_DIR_HELP = "default: $TP_DATA_DIR, else the engine's per-user data dir"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.command(args)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=1)
    common.add_argument(
        "--device", default=run.default_device(), help="cuda if available, else cpu"
    )
    common.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    common.add_argument("--checkpoint-every", type=int, default=100_000, help="sweeps")
    sized = argparse.ArgumentParser(add_help=False, parents=[common])
    sized.add_argument("--L", type=int, required=True)
    sized.add_argument("--replicas", type=int, default=16, help="per temperature")
    saving = argparse.ArgumentParser(add_help=False)
    saving.add_argument("--burst-stride", type=int, default=1, help="sweeps between burst frames")
    piloted = argparse.ArgumentParser(add_help=False)
    piloted.add_argument(
        "--allow-stale-pilot",
        action="store_true",
        help="use a pilot record that did not measure the running kernel (stamped stale)",
    )

    parser = argparse.ArgumentParser(prog="python -m tipping_point.ising", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("staircase", parents=[sized, saving], help="quasi-static sweep")
    p.add_argument("--tau", required=True, help="one tau for every T, or a pilot's tau.json")
    p.add_argument("--saves", type=int, default=200, help="bursts per temperature")
    p.add_argument("--warmup", type=int, default=1000, help="extra sweeps at the first T")
    p.add_argument("--init", choices=["hot", "cold"], default="hot")
    p.set_defaults(command=_staircase)

    p = sub.add_parser("ramp", parents=[sized, saving, piloted], help="linear cooling ramp")
    speed = p.add_mutually_exclusive_group(required=True)
    speed.add_argument("--x", type=float, help="Kibble-Zurek point: 0.02 slow, 2 fast")
    speed.add_argument("--rate", type=int, help="sweeps per unit T")
    p.add_argument("--pilot", type=Path, help="pilot.json: --x is read against its A L^z")
    p.add_argument("--pre", type=int, default=1000, help="equilibration sweeps at 3.5")
    p.add_argument("--hold", type=int, default=1000, help="sweeps held at 1.5")
    p.set_defaults(command=_ramp)

    p = sub.add_parser("null", parents=[sized, saving, piloted], help="fixed-T null run above Tc")
    p.add_argument(
        "--epsilon-L",
        type=float,
        required=True,
        choices=protocol.NULL_EPSILON_L,
        help="scaled temperature L (T - Tc) / Tc",
    )
    p.add_argument("--pilot", type=Path, required=True, help="pilot.json that measured --L")
    p.set_defaults(command=_null)

    p = sub.add_parser("pilot", parents=[sized], help="measure tau_int")
    p.add_argument(
        "--T", type=float, action="append", help="repeatable; default: the 40-point grid"
    )
    p.add_argument("--therm", type=int, default=2000)
    p.add_argument("--sweeps", type=int, default=20_000, help="measured sweeps")
    p.set_defaults(command=_pilot)

    p = sub.add_parser("tauc", help="tau_c(L) at Tc and its fit A L^z -> pilot.json")
    p.add_argument("runs", type=Path, nargs="+", help="complete pilot runs at Tc, one per L")
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--draws", type=int, default=pilot.DRAWS, help="bootstrap resamples")
    p.set_defaults(command=_tauc)

    p = sub.add_parser("validate", parents=[common], help="Gates A-C on a small L = 4 run")
    p.add_argument("--replicas", type=int, default=4096, help="per temperature")
    p.add_argument("--therm", type=int, default=400)
    p.add_argument("--sweeps", type=int, default=400)
    p.set_defaults(command=_validate)

    p = sub.add_parser("resume", help="continue a run from its checkpoint")
    p.add_argument("run_dir", type=Path)
    p.set_defaults(command=_resume)
    return parser


def _number(value: float) -> str:
    """A number as a run tag holds it: 0.02 -> 0p02, 1e+20 -> 1e20. The engine opens no run
    whose name has a '.' (or a '+')."""
    return f"{value:g}".replace(".", "p").replace("+", "")


def _spec(args: argparse.Namespace) -> run.Spec:
    init = getattr(args, "init", "hot")
    return run.Spec(args.L, args.replicas, args.seed, init, checkpoint_every=args.checkpoint_every)


def _taus(value: str, temperatures: Sequence[float]) -> list[float]:
    """--tau: one number for every temperature, or a pilot's tau.json that covers them all with
    reliable points. A pilot too short for its tau underestimates it, and the dwell would end
    before the plateau equilibrates."""
    try:
        taus = [float(value)] * len(temperatures)
    except ValueError:
        points = {round(p["temperature"], 9): p for p in io.read_json(Path(value))["points"]}
        missing = [T for T in temperatures if round(T, 9) not in points]
        if missing:
            raise SystemExit(f"{value} has no tau for T = {missing}") from None
        unreliable = [T for T in temperatures if not points[round(T, 9)].get("reliable", False)]
        if unreliable:
            raise SystemExit(
                f"{value}: unreliable tau at T = {unreliable}; run the pilot longer"
            ) from None
        taus = [points[round(T, 9)]["tau"] for T in temperatures]
    if not all(0 < tau < math.inf for tau in taus):  # NaN fails both comparisons
        raise SystemExit(f"tau must be a positive number, got {value}")
    return taus


def _staircase(args: argparse.Namespace) -> int:
    burst = protocol.Burst(stride=args.burst_stride)
    taus = _taus(args.tau, protocol.GRID)
    schedule = protocol.Staircase(taus, warmup=args.warmup, saves=args.saves, burst=burst)
    tag = f"L{args.L}-qs-s{args.seed}"
    print(run.start(_spec(args), schedule, args.data_dir, tag, args.device))
    return 0


def _record(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """The --pilot record, refused when damaged or stale (unless --allow-stale-pilot), and the
    stamp the run's manifest carries."""
    try:
        record = pilot.read(args.pilot, args.allow_stale_pilot)
    except ValueError as refused:
        raise SystemExit(str(refused)) from None
    why = pilot.stale(record)
    if why:
        print(f"warning: {args.pilot}: {why}; used anyway (--allow-stale-pilot)", file=sys.stderr)
    return record, pilot.stamp(args.pilot, record)


def _ramp(args: argparse.Namespace) -> int:
    rate, A, z, stamp = args.rate, None, None, None
    if args.x is not None:
        if args.pilot is None:
            raise SystemExit("--x needs --pilot: x is measured against the record's tau_c = A L^z")
        record, stamp = _record(args)
        A, z = record["fit"]["A"], record["fit"]["z"]
        rate = protocol.rate_for(args.L, args.x, A, z)
    burst = protocol.Burst(stride=args.burst_stride)
    schedule = protocol.Ramp(
        rate, args.L, pre=args.pre, hold=args.hold, burst=burst, x=args.x, A=A, z=z
    )
    speed = f"r{rate}" if args.x is None else f"x{_number(args.x)}"
    tag = f"L{args.L}-ramp-{speed}-s{args.seed}"
    print(run.start(_spec(args), schedule, args.data_dir, tag, args.device, stamp))
    return 0


def _null(args: argparse.Namespace) -> int:
    record, stamp = _record(args)
    if str(args.L) not in record["per_L"]:
        raise SystemExit(f"{args.pilot} measured no tau_c at L = {args.L}; pilot that size first")
    tau_c = pilot.tau_c(record, args.L)
    burst = protocol.Burst(stride=args.burst_stride)
    schedule = protocol.null_run(args.L, args.epsilon_L, tau_c, burst)
    tag = f"L{args.L}-null-e{_number(args.epsilon_L)}-s{args.seed}"
    print(run.start(_spec(args), schedule, args.data_dir, tag, args.device, stamp))
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
        m, e = pilot.series(columns, T, L)
        tau_m, tau_e = autocorr.estimate(m), autocorr.estimate(e)
        points.append(
            {
                "temperature": float(T),
                "tau": tau_m.tau,
                "window": tau_m.window,
                "binning": tau_m.binning,
                "tau_energy": tau_e.tau,
                "sweeps": m.shape[1],
                "replicas": m.shape[0],
                "reliable": tau_m.reliable,
            }
        )
    return points


def _tauc(args: argparse.Namespace) -> int:
    try:
        record = pilot.record(args.runs, args.draws)
    except ValueError as refused:
        raise SystemExit(str(refused)) from None
    path = pilot.write(record, args.data_dir)
    for L, p in record["per_L"].items():
        lo, hi = p["tau_c_ci"]
        print(
            f"L={L:<4} tau_c={p['tau_c']:.1f} [{lo:.1f}, {hi:.1f}]  "
            f"binning {p['tau_c_binning']:.1f}  tau(e)={p['tau_e']:.1f}  "
            f"N/tau={p['n_over_tau']:.0f}  "
            f"({p['replicas']} replicas x {p['sweeps']:,} sweeps, seed {p['seed']})"
        )
    fit = record["fit"]
    print(
        f"tau_c = A L^z:  A = {fit['A']:.4f} [{fit['A_ci'][0]:.4f}, {fit['A_ci'][1]:.4f}]  "
        f"z = {fit['z']:.3f} [{fit['z_ci'][0]:.3f}, {fit['z_ci'][1]:.3f}]"
    )
    print(path)
    return 0


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
