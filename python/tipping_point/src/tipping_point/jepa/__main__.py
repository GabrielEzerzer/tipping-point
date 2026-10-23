"""Command line: python -m tipping_point.jepa <command> [options]

  train    train one model seed on the staircase runs of dedicated simulator seeds (never the
           pre-registered 1-5, never ramps or null runs): --L 16 32 64 --train-seeds 101 102
           --val-seed 103 --seed 1 --pilot <pilot.json> [--held-out] [--prereg <toml>]
  resume   continue a training from its checkpoint, after a crash or a killed session
  extract  a checkpoint's latents table for a simulator run (surprise, pc_1..pc_8 per frame),
           written to <run>/latents/<training>-step<step>/
  probe    ridge probes of |m|, energy and L^2 Var(|m|) from a training's frozen embeddings,
           trained and untrained (step 0) side by side
  verify   check finished directories (trainings, latents tables, simulator runs) against their
           complete.json; exit 1 on any problem
  budget   GPU hours per ISO week from <data dir>/jepa/gpu_hours.jsonl against a weekly quota

Trainings land in <data dir>/jepa/<run_id>/; the data dir is --data-dir, else the engine's own
($TP_DATA_DIR, or a per-user folder outside the repository: tipping_point.engine.data_dir).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .. import engine
from ..ews.ising import IsingDataError
from . import extract, probes, train
from .data import HELD_OUT, PREREG_SEEDS, find_staircases, load

DATA_DIR_HELP = "default: $TP_DATA_DIR, else the engine's per-user data dir"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.command(args)
    except (IsingDataError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tipping_point.jepa", description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = parser.add_subparsers(required=True)
    data_dir = argparse.ArgumentParser(add_help=False)
    data_dir.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    device = argparse.ArgumentParser(add_help=False)
    device.add_argument("--device", default=None, help="default: cuda if available, else cpu")

    defaults = train.Config()
    p = sub.add_parser("train", parents=[data_dir, device], help="train one model seed")
    p.add_argument("--L", type=int, nargs="+", required=True, help="sizes trained jointly")
    p.add_argument("--train-seeds", type=int, nargs="+", required=True, help="simulator seeds")
    p.add_argument("--val-seed", type=int, required=True, help="the validation simulator seed")
    p.add_argument("--seed", type=int, required=True, help="the model seed")
    p.add_argument("--pilot", type=Path, required=True, help="the #50 pilot record, pilot.json")
    p.add_argument("--prereg", type=Path, help="the pre-registration this model is scored under")
    p.add_argument("--held-out", action="store_true", help=f"ablation: drop T in {HELD_OUT}")
    p.add_argument("--steps", type=int, default=defaults.steps)
    p.add_argument("--warmup", type=int, help="updates; default steps / 20")
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--batch-sites", type=int, default=defaults.batch_sites)
    p.add_argument("--var-weight", type=float, default=defaults.var_weight, help="0: EMA only")
    p.add_argument("--cov-weight", type=float, default=defaults.cov_weight, help="0: EMA only")
    p.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    p.add_argument("--monitor-every", type=int, default=defaults.monitor_every)
    p.set_defaults(command=_train)

    p = sub.add_parser("resume", parents=[device], help="continue a training")
    p.add_argument("run_dir", type=Path)
    p.set_defaults(command=_resume)

    p = sub.add_parser("extract", parents=[device], help="a checkpoint's latents table")
    p.add_argument("checkpoint", type=Path, help="final.pt or ckpt/step_0000000.pt")
    p.add_argument("dataset_run", type=Path, help="a complete simulator run folder")
    p.add_argument("--sites", type=int, default=65_536, help="sites per readout batch")
    p.set_defaults(command=_extract)

    p = sub.add_parser("probe", parents=[device], help="ridge probes, trained beside untrained")
    p.add_argument("run_dir", type=Path, help="a finished training")
    p.add_argument("--fit", type=Path, nargs="+", required=True, help="simulator run folders")
    p.add_argument("--test", type=Path, nargs="*", default=[], help="default: odd replicas")
    p.add_argument("--limit", type=int, default=20_000, help="frames per run")
    p.add_argument("--alpha", type=float, default=1e-3, help="ridge penalty per sample")
    p.add_argument("--json", type=Path, help="also write the rows here")
    p.set_defaults(command=_probe)

    p = sub.add_parser("verify", help="check finished directories against complete.json")
    p.add_argument("dirs", type=Path, nargs="+")
    p.set_defaults(command=_verify)

    p = sub.add_parser("budget", parents=[data_dir], help="GPU hours per week against a quota")
    p.add_argument("--quota", type=float, default=30.0, help="hours per week")
    p.set_defaults(command=_budget)
    return parser


def _train(args: argparse.Namespace) -> int:
    prereg = train.prereg_record(args.prereg) if args.prereg else None
    refused = PREREG_SEEDS | frozenset(prereg["seeds"] if prereg else ())
    seeds = [*args.train_seeds, args.val_seed]
    scored = sorted(set(seeds) & refused)
    if scored:
        raise ValueError(f"seeds {scored} are scored by a pre-registration: never trained on")
    runs = find_staircases(args.data_dir, args.L, seeds)
    train_ids = [runs[(L, s)].run_id for L in args.L for s in args.train_seeds]
    val_ids = [runs[(L, args.val_seed)].run_id for L in args.L]
    data = load(args.data_dir, train_ids, val_ids, HELD_OUT if args.held_out else None, refused)
    config = train.Config(
        steps=args.steps,
        batch_sites=args.batch_sites,
        lr=args.lr,
        warmup=args.warmup if args.warmup is not None else max(1, args.steps // 20),
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        monitor_every=args.monitor_every,
    )
    sizes = "-".join(str(L) for L in sorted(args.L))
    tag = f"L{sizes}-m{args.seed}" + ("-heldout" if args.held_out else "")
    pilot = train.pilot_record(args.pilot)
    print(train.start(config, data, args.data_dir, tag, args.device, pilot, prereg))
    return 0


def _resume(args: argparse.Namespace) -> int:
    print(train.resume(args.run_dir, device=args.device))
    return 0


def _extract(args: argparse.Namespace) -> int:
    print(extract.extract(args.checkpoint, args.dataset_run, args.device, args.sites))
    return 0


def _probe(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not (run_dir / "complete.json").is_file():
        raise ValueError(f"{run_dir}: not a finished training (no complete.json)")
    checkpoints = {"trained": run_dir / train.FINAL, "untrained": run_dir / train.TWIN}
    rows = probes.probe(checkpoints, args.fit, args.test, args.limit, args.alpha, args.device)
    r2 = {(r["checkpoint"], r["target"]): r["r2"] for r in rows}
    print(
        f"{'target':<8} {'trained':>9} {'untrained':>9}   R^2, n_fit {rows[0]['n_fit']:,}, "
        f"n_test {rows[0]['n_test']:,}"
    )
    for target in probes.TARGETS:
        print(f"{target:<8} {r2['trained', target]:>9.4f} {r2['untrained', target]:>9.4f}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


def _verify(args: argparse.Namespace) -> int:
    bad = 0
    for path in args.dirs:
        problems = train.verify(path)
        bad += bool(problems)
        print(f"{path}: {'ok' if not problems else '; '.join(problems)}")
    return 1 if bad else 0


def _budget(args: argparse.Namespace) -> int:
    weeks = train.budget(args.data_dir, args.quota)
    if not weeks:
        print("no GPU hours logged")
    for w in weeks:
        print(f"{w['week']}  {w['hours']:6.2f} h of {w['quota']:g} ({w['left']:+.2f} h left)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
