from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from .. import engine
from ..market import archive
from ..market.__main__ import BAD, DATA_DIR_HELP, archive_dir
from . import coins, fetch, inputs, labels, ladder, span

DESCRIPTION = f"""\
Command line: python -m tipping_point.panel <command> [options]

  coins   derive the panel from Binance Vision (network): every USD-M symbol matching
          [A-Z0-9]+USDT but {", ".join(sorted(coins.EXCLUDED))}, with a monthly 1-day kline file
          holding every day of {coins.YEAR} and a daily metrics file on {coins.METRIC_DAYS[0]} and
          {coins.METRIC_DAYS[1]}, ranked by {coins.YEAR} quote volume; the panel is
          {" and ".join(coins.ALWAYS)} and the top {coins.OTHERS} others. Prints the ranking and
          exits 1 when it differs from the committed list (panel/coins.txt)
  fetch   download and verify the panel's klines, metrics and funding for --start..--end (default
          {span.START}..{span.END}) into --dest (default <data dir>/archives/binance), --workers
          files at a time. Each file is checked as market fetch checks it; a re-run skips the files
          already verified. Exit 1 when a file is refused, failed or unreadable
  labels  the panel's labels from the fetched files (--archive, same default), as a run in
          <data dir>/labels/<run_id>/ with the tables thresholds, cascades and grid: bars from
          --start, daily trailing thresholds and cascades from the first day that has a threshold,
          the hourly grid from --dev-start (default {span.DEV_START}) to the end of --end
  inputs  the model inputs of every coin and hour of a labels run's grid (--labels, a run id or
          folder), as a run in <data dir>/panel-inputs/<run_id>/ with the table inputs, the
          volatility inputs scaled by the labels run's threshold of each hour's day
  ladder  fit and score the rungs of a plan file (--plan, default panel/ladder.toml) on an inputs
          run (--inputs) and its labels run, out of fold by calendar quarter (each left out, and
          forward only), with a joint block bootstrap for each comparison, as a run in
          <data dir>/panel-ladder/<run_id>/ with the tables oof, scores, folds, comparisons and
          rows, and report.md
  sequence
          train the sequence models of a plan file (--plan, default panel/sequence.toml) on an
          inputs run (--inputs): a JEPA, its untrained twin and a supervised model on the last
          336 hours of a ladder rung's inputs, inside each quarter fold, with logistic probes on
          the JEPA's and twin's readouts over a grid of readouts and penalties, as a run in
          <data dir>/panel-sequence/<run_id>/ with the tables diagnostics, curves, folds and rows,
          report.md and every fitted model under encoders/
  scores  out-of-fold scores of one configuration of a sequence run (--sequence; --model,
          --variant, and for the JEPA and twin --readout and --c) from its saved models, the mean
          over seeds of each row's logit, as a run in <data dir>/panel-scores/<run_id>/ with one
          table per scheme in the columns a ladder plan's [external] section reads

Every command refuses any day from {span.HELD_BACK} on, which is held back as the test span. The
data dir is --data-dir, else $TP_DATA_DIR, else the engine's per-user data dir.
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.command(args)
    except (archive.ArchiveError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tipping_point.panel", description=DESCRIPTION)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = parser.add_subparsers(required=True)

    def spanned(p: argparse.ArgumentParser) -> None:
        p.add_argument("--start", type=date.fromisoformat, default=span.START)
        p.add_argument("--end", type=date.fromisoformat, default=span.END)
        p.add_argument("--symbols", nargs="+", default=None, help="default: the committed panel")

    p = sub.add_parser("coins", help="derive the panel from Binance Vision")
    p.add_argument("--workers", type=int, default=fetch.WORKERS)
    p.set_defaults(command=_coins)

    p = sub.add_parser("fetch", help="download and verify the panel's archive files")
    spanned(p)
    p.add_argument("--dest", type=Path, default=archive_dir())
    p.add_argument("--workers", type=int, default=fetch.WORKERS)
    p.set_defaults(command=_fetch)

    p = sub.add_parser("labels", help="the panel's thresholds, cascades and hourly grid")
    spanned(p)
    p.add_argument("--dev-start", type=date.fromisoformat, default=span.DEV_START)
    p.add_argument("--archive", type=Path, default=archive_dir())
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", help="the run id's tag (default panel-<dev start>-<end>)")
    p.set_defaults(command=_labels)

    p = sub.add_parser("inputs", help="the model inputs of every coin and hour of a labels run")
    p.add_argument("--labels", required=True, help="a labels run id, or its folder")
    p.add_argument("--archive", type=Path, default=archive_dir())
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", default="inputs", help="the run id's tag (default inputs)")
    p.set_defaults(command=_inputs)

    p = sub.add_parser("ladder", help="score a plan's rungs out of fold on an inputs run")
    p.add_argument("--inputs", required=True, help="an inputs run id, or its folder")
    p.add_argument("--plan", type=Path, default=ladder.PLAN)
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", default="ladder", help="the run id's tag (default ladder)")
    p.set_defaults(command=_ladder)

    p = sub.add_parser("sequence", help="train and score a plan's sequence models on an inputs run")
    p.add_argument("--inputs", required=True, help="an inputs run id, or its folder")
    p.add_argument("--plan", type=Path, default=None, help="default panel/sequence.toml")
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", default="sequence", help="the run id's tag (default sequence)")
    p.set_defaults(command=_sequence)

    p = sub.add_parser("scores", help="score one configuration of a sequence run")
    p.add_argument("--sequence", required=True, help="a sequence run id, or its folder")
    p.add_argument("--model", required=True, choices=("jepa", "twin", "supervised"))
    p.add_argument("--variant", required=True)
    p.add_argument("--readout", default=None, help="the JEPA's or twin's readout")
    p.add_argument("--c", type=float, default=None, help="the JEPA's or twin's probe penalty")
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", default="scores", help="the run id's tag (default scores)")
    p.set_defaults(command=_scores)
    return parser


def _run_dir(data_dir: Path, domain: str, name: str) -> Path:
    path = Path(name)
    return path if path.is_dir() else data_dir / domain / name


def _coins(args: argparse.Namespace) -> int:
    volumes = coins.derive(archive.http_get, args.workers)
    panel = coins.rank(volumes)
    order = sorted(volumes, key=lambda s: (-volumes[s], s))
    for i, symbol in enumerate(order, 1):
        mark = "*" if symbol in panel else " "
        print(f"{i:>4} {mark} {symbol:<16} {volumes[symbol]:>22,.0f}")
    print(f"{len(volumes)} eligible symbols; the panel: {' '.join(panel)}")
    held = coins.committed()
    if panel != held:
        print(f"differs from {coins.COMMITTED.name}: {' '.join(held)}", file=sys.stderr)
        return 1
    return 0


def _fetch(args: argparse.Namespace) -> int:
    symbols = args.symbols or coins.committed()
    done = fetch.fetch(
        symbols,
        args.start,
        args.end,
        args.dest,
        archive.http_get,
        args.workers,
        log=lambda line: print(line, flush=True),
    )
    groups: dict[tuple[str, str], Counter[str]] = {}
    for path, entry in done.items():
        kind, symbol, _ = path.split("/")
        groups.setdefault((symbol, kind), Counter())[entry["status"]] += 1
    for (symbol, kind), status in sorted(groups.items()):
        print(
            f"{symbol:<13} {kind:<12} "
            + ", ".join(f"{count} {name}" for name, count in sorted(status.items()))
        )
    bad = {path: e for path, e in done.items() if e["status"] in BAD}
    for path, entry in bad.items():
        why = entry.get("error") or f"sha256 {entry['sha256']}, .CHECKSUM {entry['expected']}"
        print(f"{entry['status']}: {path}: {why}", file=sys.stderr)
    print(args.dest / "manifest.json")
    return 1 if bad else 0


def _labels(args: argparse.Namespace) -> int:
    symbols = args.symbols or coins.committed()
    span.refuse(args.start, args.dev_start, args.end)
    tag = args.tag or f"panel-{args.dev_start:%Y%m%d}-{args.end:%Y%m%d}"
    run_dir, checks = labels.run(
        args.archive,
        args.data_dir,
        symbols,
        args.start,
        args.dev_start,
        args.end,
        tag,
        log=lambda line: print(line, flush=True),
    )
    kinds = ", ".join(f"{n} {kind}" for kind, n in sorted(checks["by_kind"].items()))
    print(
        f"{checks['asset_instances']} cascades ({kinds}) in {checks['market_events']} market "
        f"events, {checks['market_events_5_or_more']} with 5 or more coins"
    )
    grid = checks["grid"]
    for h in labels.HORIZONS:
        print(
            f"{h:>2} h: positive {grid[f'positive_{h}']:.4f} (long {grid[f'long_{h}']:.4f}, "
            f"short {grid[f'short_{h}']:.4f}), censored {grid[f'censored_{h}']:.4f}"
        )
    print(f"skip {grid['skip']:.4f}")
    print(run_dir)
    return 0


def _inputs(args: argparse.Namespace) -> int:
    run_dir, checks = inputs.run(
        _run_dir(args.data_dir, "labels", args.labels),
        args.archive,
        args.data_dir,
        args.tag,
        log=lambda line: print(line, flush=True),
    )
    print(f"{checks['rows']} rows")
    print(run_dir)
    return 0


def _ladder(args: argparse.Namespace) -> int:
    run_dir, out = ladder.run(
        _run_dir(args.data_dir, "panel-inputs", args.inputs),
        args.data_dir,
        args.plan,
        args.tag,
        log=lambda line: print(line, flush=True),
    )
    print(f"{out['seconds'] / 60:.1f} minutes")
    print(run_dir)
    return 0


def _sequence(args: argparse.Namespace) -> int:
    from . import sequence

    run_dir, out = sequence.run(
        _run_dir(args.data_dir, "panel-inputs", args.inputs),
        args.data_dir,
        args.plan or sequence.PLAN,
        args.tag,
        log=lambda line: print(line, flush=True),
    )
    print(f"{out['seconds'] / 60:.1f} minutes")
    print(run_dir)
    return 0


def _scores(args: argparse.Namespace) -> int:
    from . import sequence

    run_dir, out = sequence.scores_run(
        _run_dir(args.data_dir, "panel-sequence", args.sequence),
        args.data_dir,
        (args.model, args.variant, args.readout, args.c),
        args.tag,
        log=lambda line: print(line, flush=True),
    )
    for scheme, table in out["tables"].items():
        print(f"{scheme}: {table.num_rows} rows")
    print(f"{out['seconds'] / 60:.1f} minutes")
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
