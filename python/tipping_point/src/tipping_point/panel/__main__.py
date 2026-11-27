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
          report.md and every fitted model under encoders/. panel/sequence-final.toml is the
          fit-only fold: every model fitted once on the span's hours more than the gap before its
          end, nothing scored
  scores  out-of-fold scores of one model and variant of a sequence run (--sequence, --model,
          --variant) from its saved models, the JEPA's and twin's readout and penalty chosen from
          the run's folds table by the selection rule, the mean over seeds of each row's logit, as
          a run in <data dir>/panel-scores/<run_id>/ with one table per scheme in the columns a
          ladder plan's [external] section reads; refuses a fit-only run
  merge   one sequence run from several (--runs) whose plans differ only in variants, seeds and
          folds and which together fit every cell of the merged plan once, as a run in
          <data dir>/panel-sequence/<run_id>/ that scores reads like any scored sequence run
          (--plan panel/sequence-final.toml joins the fit-only pieces, which scores refuses)
  classical
          the classical comparators of a sequence plan (--plan, default panel/sequence.toml) on an
          inputs run (--inputs), on the sequence models' folds and rows: the plan's rungs at every
          probe penalty, and a logistic on the last value, means and spreads of each variant's
          336-hour window at every readout and penalty. Picks one configuration per rung and
          variant name by the sequence models' selection rule, or takes one given with --fixed
          MODEL VARIANT READOUT C (READOUT - for a rung), and writes its out-of-fold scores, as a
          run in <data dir>/panel-classical/<run_id>/ with the tables folds, rows and
          <name>-<scheme> in the columns a ladder plan's [external] section reads, and report.md
  compare score the arms of a comparison plan (--plan, default panel/compare.toml) as pairs on
          the ladder's rows of an inputs run (--inputs), each arm's run given with --arm NAME=RUN
          (a scores run or a classical run, id or folder): each run must come from that inputs
          run, its labels grid and the plan's ladder plan, and each arm's rows must be exactly the
          ladder's. Every arm takes the ladder's bootstrap draws; one pair, side and horizon is
          primary. Every run in each arm's chain and the inputs run must record its manifest's
          sha256 in complete.json; --unbound accepts runs written before that record existed, for
          a development comparison marked as not a test. Writes a run in
          <data dir>/panel-compare/<run_id>/ with the tables scores, folds, comparisons and rows,
          and report.md
  test    the held-back test of the frozen plan panel/test.toml, which pins the development runs,
          the final sequence run, the approved commits, the arms, the pairs and the decision rule;
          with no such file it refuses. test data fetches the archive and makes the whole-span
          labels and inputs runs, from the development labels run's start to the plan's last day,
          into <data dir>/{span.HELD_BACK_DIR}/, and runs the prefix check; a re-run reuses them.
          test run checks every pinned run, commit, plan and model, settles the support and writes
          it, then scores the arms and decides, as a run in
          <data dir>/{span.HELD_BACK_DIR}/panel-test/<run_id>/; it refuses a plan that already
          has a run there

Every command but test refuses any day from {span.HELD_BACK} on, which is held back as the test
span. The data dir is --data-dir, else $TP_DATA_DIR, else the engine's per-user data dir.
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
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", default="scores", help="the run id's tag (default scores)")
    p.set_defaults(command=_scores)

    p = sub.add_parser("merge", help="one sequence run from runs that split a plan's cells")
    p.add_argument("--runs", nargs="+", required=True, help="sequence run ids, or their folders")
    p.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="the plan the runs split (default panel/sequence.toml); they must fit all its cells",
    )
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", default="merged", help="the run id's tag (default merged)")
    p.set_defaults(command=_merge)

    p = sub.add_parser("classical", help="fit, choose and score a sequence plan's classical side")
    p.add_argument("--inputs", required=True, help="an inputs run id, or its folder")
    p.add_argument("--plan", type=Path, default=None, help="default panel/sequence.toml")
    p.add_argument(
        "--fixed",
        nargs=4,
        metavar=("MODEL", "VARIANT", "READOUT", "C"),
        help="fit and score this configuration only, without choosing (READOUT - for a rung)",
    )
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", default="classical", help="the run id's tag (default classical)")
    p.set_defaults(command=_classical)

    p = sub.add_parser("compare", help="score exported score files as pairs on the ladder's rows")
    p.add_argument("--inputs", required=True, help="an inputs run id, or its folder")
    p.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=RUN",
        help="the run of one arm of the plan (id or folder); every arm exactly once",
    )
    p.add_argument("--plan", type=Path, default=None, help="default panel/compare.toml")
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", default="compare", help="the run id's tag (default compare)")
    p.add_argument(
        "--unbound",
        action="store_true",
        help="accept runs whose manifests are not bound: a development comparison, never a test",
    )
    p.set_defaults(command=_compare)

    p = sub.add_parser("test", help="the held-back test of the frozen plan panel/test.toml")
    steps = p.add_subparsers(required=True)
    q = steps.add_parser("data", help="the whole-span labels and inputs runs, and the prefix check")
    q.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    q.set_defaults(command=_test_data)
    q = steps.add_parser("run", help="check every record, settle the support, score and decide")
    q.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    q.set_defaults(command=_test_run)
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
        args.model,
        args.variant,
        args.tag,
        log=lambda line: print(line, flush=True),
    )
    for scheme, table in out["tables"].items():
        print(f"{scheme}: {table.num_rows} rows")
    print(f"{out['seconds'] / 60:.1f} minutes")
    print(run_dir)
    return 0


def _merge(args: argparse.Namespace) -> int:
    from . import sequence

    run_dir, out = sequence.merge_run(
        [_run_dir(args.data_dir, "panel-sequence", name) for name in args.runs],
        args.plan or sequence.PLAN,
        args.data_dir,
        args.tag,
        log=lambda line: print(line, flush=True),
    )
    print(f"{len(out['folds'])} fold rows")
    print(run_dir)
    return 0


def _classical(args: argparse.Namespace) -> int:
    from . import classical, sequence

    fixed = None
    if args.fixed is not None:
        model, variant, readout, c = args.fixed
        fixed = classical.Config(model, variant, None if readout == "-" else readout, float(c))
    run_dir, out = classical.run(
        _run_dir(args.data_dir, "panel-inputs", args.inputs),
        args.data_dir,
        args.plan or sequence.PLAN,
        args.tag,
        fixed,
        log=lambda line: print(line, flush=True),
    )
    for name, config in out["exported"].items():
        print(f"{name}: {config.model} {config.variant} {config.readout or '-'} c {config.c:g}")
    print(f"{out['seconds'] / 60:.1f} minutes")
    print(run_dir)
    return 0


def _compare(args: argparse.Namespace) -> int:
    from . import compare

    arms = []
    for given in args.arm:
        name, sep, run = given.partition("=")
        if not (sep and name and run):
            raise ValueError(f"--arm {given}: expected NAME=RUN")
        arms.append((name, run))
    run_dir, out = compare.run(
        arms,
        _run_dir(args.data_dir, "panel-inputs", args.inputs),
        args.data_dir,
        args.plan or compare.PLAN,
        args.tag,
        log=lambda line: print(line, flush=True),
        unbound=args.unbound,
    )
    print(f"{len(out['comparisons'])} comparison rows, schemes {', '.join(out['schemes'])}")
    print(run_dir)
    return 0


def _test_data(args: argparse.Namespace) -> int:
    from . import frozen

    out = frozen.data(args.data_dir, archive.http_get, log=lambda line: print(line, flush=True))
    rows = out["prefix"]["rows"]
    print(
        f"prefix check: {rows['thresholds']} threshold, {rows['grid']} grid and {rows['inputs']} "
        f"inputs rows equal before {span.HELD_BACK}"
    )
    for path in (out["labels"], out["inputs"]):
        print(f"{'made' if path in out['made'] else 'reused'} {path}")
    return 0


def _test_run(args: argparse.Namespace) -> int:
    from . import frozen

    run_dir, out = frozen.run(args.data_dir, log=lambda line: print(line, flush=True))
    decision = out["decision"]
    print(decision["outcome"] + "".join(f"; {cause}" for cause in decision["causes"]))
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
