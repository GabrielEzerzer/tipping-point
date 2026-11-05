"""Command line: python -m tipping_point.market <command> [options]

  fetch    download and verify Binance's USD-M 1-minute klines, 5-minute metrics and monthly
           funding of the UTC days --start..--end (inclusive) for --symbols, into --dest (default
           harness/fixtures/archives/binance, gitignored; relative to the current folder, so run
           it from the repository root). A re-run skips the files already verified. Exit 1 when
           a file is refused (checksum), failed (network: run it again) or unreadable (it matches
           its checksum but does not parse: kept, recorded, and refused by labels)
  labels   the regime labels of --start..--end from the fetched files (--archive, same default),
           as a run in <data dir>/labels/<run_id>/. Every labeller parameter is an option
           (--d-min, --enter, ...) whose default is the research pass's start value (#29).
           Refuses any day from 2024-01-01: the confirmatory labels wait for the frozen
           market-labels-v1 pre-registration (#29 S5)
  cascade  the pooled list of the cascade candidates of --start..--end from the fetched files, as
           a run in <data dir>/labels/<run_id>/ holding the one table `cascades` (cascade.py).
           Every rule parameter is an option (--a, --merge-hours, ...) at the research pass's
           start value; the first --trailing-days only fill the trailing medians. Refuses any day
           from 2024-01-01, as labels does
  reports  the label-count and stability reports of the labels run --run, into <run>/reports/
           (reports.py). It labels the run's span again from the fetched files and refuses unless
           that gives back the run's own tables; the Parkinson variant reads the same klines.
           --cascades names a cascade run of the same symbols inside the run's span, whose pooled
           list the label-count report counts, as it counts the committed CPI, NFP and FOMC calendar
           (reports.CALENDAR). The aggTrades cross-check is not wired yet, so the stability report
           says "not supplied" for it

The floor (reports.floor) needs the feature pipelines, which no module here may import:
harness/market_floor.py runs it. The data dir is --data-dir, else the engine's own ($TP_DATA_DIR,
or a per-user folder outside the repository: tipping_point.engine.data_dir).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from .. import engine
from . import archive, bars, cascade, msvol, reports

ARCHIVE = Path("harness") / "fixtures" / "archives" / "binance"
CONFIRMATORY = date(2024, 1, 1)  # no label from here on until market-labels-v1 is frozen
DATA_DIR_HELP = "default: $TP_DATA_DIR, else the engine's per-user data dir"
BAD = ("refused", "failed", "unreadable")  # the fetch statuses that make it exit 1


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

    p = sub.add_parser("cascade", help="the pooled cascade list of a span, as a run")
    span(p)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    p.add_argument("--tag", help="the run id's tag (default cascades-<start>-<end>)")
    for field in dataclasses.fields(cascade.Params):
        flag = "--" + field.name.replace("_", "-")
        p.add_argument(flag, type=type(field.default), default=field.default)
    p.set_defaults(command=_cascade)

    p = sub.add_parser("reports", help="the label-count and stability reports of a labels run")
    p.add_argument("--run", type=Path, required=True, help="the labels run's folder")
    p.add_argument("--cascades", type=Path, help="a cascade run's folder (default: not supplied)")
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--seed", type=int, default=29, help="the delay simulation's draws")
    p.set_defaults(command=_reports)
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
            f"{status['refused']} refused, {status['failed']} failed, "
            f"{status['unreadable']} unreadable; files with "
            + ", ".join(f"{key} {count}" for key, count in flagged.items())
        )
    bad = {path: e for path, e in done.items() if e["status"] in BAD}
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


def _cascade(args: argparse.Namespace) -> int:
    if args.end >= CONFIRMATORY:
        print(
            f"error: cascades end on {CONFIRMATORY.year - 1}-12-31: days from {CONFIRMATORY} are "
            "the confirmatory span, closed until market-labels-v1 is frozen (#29 S5)",
            file=sys.stderr,
        )
        return 1
    params = cascade.Params(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(cascade.Params)}
    )
    span = archive.collect(args.symbols, args.start, args.end, args.archive)
    days = (args.end - args.start).days + 1
    found: dict[str, list[dict[str, Any]]] = {}
    checks: dict[str, Any] = {}
    for symbol in args.symbols:
        bar, checks[symbol] = bars.bars(
            archive.day_ms(args.start),
            days,
            *(span.columns[(symbol, kind)] for kind in archive.KINDS),
        )
        found[symbol], windows = cascade.candidates(bar, params)
        checks[symbol] |= windows | {"candidates": len(found[symbol])}
    rows = cascade.pool(found, params)
    per_symbol = Counter(row["symbol"] for row in rows)
    checks["asset_instances"] = len(rows)
    checks["market_events"] = len({row["market_event"] for row in rows})
    checks["liquidations"] = cascade.liquidations(args.data_dir, rows, params)
    run_params = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "symbols": list(args.symbols),
        "cascade": dataclasses.asdict(params),
    }
    run_dir = msvol.write_run(
        args.data_dir,
        args.tag or f"cascades-{args.start:%Y%m%d}-{args.end:%Y%m%d}",
        {"cascades": pa.Table.from_pylist(rows, schema=cascade.SCHEMA)},
        run_params,
        span.sources,
        span.missing,
        checks,
    )
    for symbol in args.symbols:
        c = checks[symbol]
        print(
            f"{symbol:<8} {c['windows_evaluated']} of {c['windows']} windows evaluated "
            f"({c['windows_missing_oi']} missing open interest, {c['windows_missing_bars']} "
            f"missing bars, {c['windows_no_history']} without history); {c['candidates']} "
            f"candidates, {per_symbol[symbol]} asset instances"
        )
    print(f"{checks['market_events']} market events, {checks['asset_instances']} asset instances")
    print(f"liquidations: {checks['liquidations']['status']}")
    print(f"{run_dir}  ({len(span.sources)} files read, {len(span.missing)} not published)")
    return 0


def _run_params(run: Path) -> dict[str, Any]:
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))["params"]


def _read(run: Path, table: str) -> pa.Table:
    return engine.read(run.parent.parent, run.parent.name, run.name, table)


def _reports(args: argparse.Namespace) -> int:
    run_params = _run_params(args.run)
    if "labeller" not in run_params:
        print(f"error: {args.run} is not a labels run", file=sys.stderr)
        return 1
    found = cascade_span = None
    if args.cascades is not None:
        other = _run_params(args.cascades)
        cascade_span = (other.get("start", ""), other.get("end", ""))  # ISO dates sort as text
        inside = run_params["start"] <= cascade_span[0] <= cascade_span[1] <= run_params["end"]
        same = sorted(other.get("symbols") or []) == sorted(run_params["symbols"])
        if "cascade" not in other or not inside or not same:
            print(
                f"error: {args.cascades} is not a cascade run of {args.run.name}'s symbols "
                "inside its span",
                file=sys.stderr,
            )
            return 1
        rows = _read(args.cascades, "cascades").to_pylist()
        found = {
            symbol: np.array([r["ts_ns"] for r in rows if r["symbol"] == symbol], np.int64)
            for symbol in run_params["symbols"]
        }
    out = args.run / "reports"
    for folder in (out / "label_count", out / "stability"):
        if folder.exists():
            print(f"error: {folder} exists", file=sys.stderr)
            return 1
    start, end = date.fromisoformat(run_params["start"]), date.fromisoformat(run_params["end"])
    days = (end - start).days + 1
    params = msvol.Params(**run_params["labeller"])
    span = archive.collect(run_params["symbols"], start, end, args.archive)
    series, parkinson = {}, {}
    for symbol in run_params["symbols"]:
        klines, metrics, funding = (span.columns[(symbol, kind)] for kind in archive.KINDS)
        bar, _ = bars.bars(archive.day_ms(start), days, klines, metrics, funding)
        series[symbol] = (bar, msvol.label(bar["ts_ns"], bar["rv"], params))
        parkinson[symbol] = reports.parkinson_rv(archive.day_ms(start), days, klines)
    for name, table in msvol.tables(series).items():
        if not table.equals(_read(args.run, name)):
            print(
                f"error: the fetched files and this code no longer give {args.run.name}'s "
                f"{name} table, so a report on them would not describe that run",
                file=sys.stderr,
            )
            return 1
    releases = reports.release_calendar()
    counts = reports.label_count(
        series, releases=releases, cascades=found, cascade_span=cascade_span
    )
    print(reports.write(counts, out))
    print(reports.write(reports.stability(series, params, parkinson, seed=args.seed), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
