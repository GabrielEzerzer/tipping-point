"""Command line: python -m tipping_point.market <command> [options]

  fetch   download and verify Binance's USD-M 1-minute klines, 5-minute metrics and monthly
          funding of the UTC days --start..--end (inclusive) for --symbols, into --dest (default
          harness/fixtures/archives/binance, gitignored; relative to the current folder, so run
          it from the repository root). A re-run skips the files already verified. Exit 1 when a
          file is refused (checksum) or failed (network): run it again for those
  labels  the regime labels of --start..--end from the fetched files (--archive, same default),
          as a run in <data dir>/labels/<run_id>/. Every labeller parameter is an option
          (--d-min, --enter, ...) whose default is the research pass's start value (#29).
          Refuses any day from 2024-01-01: the confirmatory labels wait for the frozen
          market-labels-v1 pre-registration (#29 S5)

The data dir is --data-dir, else the engine's own ($TP_DATA_DIR, or a per-user folder outside
the repository: tipping_point.engine.data_dir).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from .. import engine
from . import archive, bars, msvol

ARCHIVE = Path("harness") / "fixtures" / "archives" / "binance"
CONFIRMATORY = date(2024, 1, 1)  # no label from here on until market-labels-v1 is frozen
DATA_DIR_HELP = "default: $TP_DATA_DIR, else the engine's per-user data dir"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.command(args)
    except archive.ArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tipping_point.market", description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = parser.add_subparsers(required=True)

    def span(p: argparse.ArgumentParser) -> None:
        p.add_argument("--start", type=date.fromisoformat, required=True, help="first UTC day")
        p.add_argument("--end", type=date.fromisoformat, required=True, help="last UTC day")
        p.add_argument("--symbols", nargs="+", default=list(archive.SYMBOLS))

    p = sub.add_parser("fetch", help="download and verify a span's archive files")
    span(p)
    p.add_argument("--dest", type=Path, default=ARCHIVE)
    p.set_defaults(command=_fetch)

    p = sub.add_parser("labels", help="the regime labels of a span, as a labels run")
    span(p)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", help="the run id's tag (default binance-<start>-<end>)")
    for field in dataclasses.fields(msvol.Params):
        flag = "--" + field.name.replace("_", "-")
        p.add_argument(flag, type=type(field.default), default=field.default)
    p.set_defaults(command=_labels)
    return parser


def _fetch(args: argparse.Namespace) -> int:
    done = archive.fetch(args.symbols, args.start, args.end, args.dest, archive.http_get)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path, entry in done.items():
        kind, symbol, _ = path.split("/")
        groups.setdefault((kind, symbol), []).append(entry)
    for (kind, symbol), entries in groups.items():
        status = Counter(e["status"] for e in entries)
        checks = [e["checks"] for e in entries if "checks" in e]
        flagged = {
            key: sum(bool(c.get(key)) for c in checks)
            for key in ("gaps", "duplicates", "unsorted", "off_grid")
        }
        print(
            f"{kind:<12} {symbol:<8} {status['verified']} verified "
            f"({sum(e['fetched'] for e in entries)} fetched), {status['missing']} missing, "
            f"{status['refused']} refused, {status['failed']} failed; files with "
            + ", ".join(f"{key} {count}" for key, count in flagged.items())
        )
    bad = {path: e for path, e in done.items() if e["status"] in ("refused", "failed")}
    for path, entry in bad.items():
        why = entry.get("error") or f"sha256 {entry['sha256']}, .CHECKSUM {entry['expected']}"
        print(f"{entry['status']}: {path}: {why}", file=sys.stderr)
    print(args.dest / "manifest.json")
    return 1 if bad else 0


def _labels(args: argparse.Namespace) -> int:
    if args.end >= CONFIRMATORY:
        print(
            f"error: labels end on {CONFIRMATORY.year - 1}-12-31: days from {CONFIRMATORY} are "
            "the confirmatory span, closed until market-labels-v1 is frozen (#29 S5)",
            file=sys.stderr,
        )
        return 1
    params = msvol.Params(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(msvol.Params)}
    )
    span = archive.collect(args.symbols, args.start, args.end, args.archive)
    days = (args.end - args.start).days + 1
    series, checks = {}, {}
    for symbol in args.symbols:
        bar, checks[symbol] = bars.bars(
            archive.day_ms(args.start),
            days,
            *(span.columns[(symbol, kind)] for kind in archive.KINDS),
        )
        series[symbol] = (bar, msvol.label(bar["ts_ns"], bar["rv"], params))
    run_params = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "symbols": list(args.symbols),
        "bars": "5-minute UTC grid; rv = Garman-Klass variance summed over 1-minute klines",
        "labeller": dataclasses.asdict(params),
    }
    tag = args.tag or f"binance-{args.start:%Y%m%d}-{args.end:%Y%m%d}"
    found = msvol.tables(series)
    run_dir = msvol.write_run(
        args.data_dir, tag, found, run_params, span.sources, span.missing, checks
    )
    for symbol, (bar, out) in series.items():
        kinds = Counter(e["kind"] for e in out.episodes)
        flagged = sum(r["flagged"] for r in out.refits)
        empty = checks[symbol]["bars_without_rv"]
        print(
            f"{symbol:<8} {len(bar['ts_ns'])} bars ({empty} without rv), "
            f"{len(out.refits)} refits ({flagged} flagged), {kinds['event']} events, "
            f"{kinds['episode']} other episodes, {kinds['flicker']} flickers, "
            f"{len(out.nulls)} null stretches"
        )
    print(f"{run_dir}  ({len(span.sources)} files read, {len(span.missing)} not published)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
