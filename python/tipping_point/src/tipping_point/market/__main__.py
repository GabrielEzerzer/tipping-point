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
           Any day from 2024-01-01 is the confirmatory span, which only --prereg opens:
           --prereg paper/prereg/market-labels-v1.toml takes the span, the symbols and every
           labeller parameter from that file, and refuses unless the file is the one frozen at
           its tag, the modules that make labels and cascades (HELD: market/'s __init__,
           archive, bars, cascade and msvol) are byte for byte the tag's commit's, this command
           line, which gained --prereg after the freeze, and ews/prereg.py, which runs the check
           (CLEAN), have no change beyond HEAD, and no --start, --end, --symbols or labeller
           option is given. The run's manifest records the file's stamp, HEAD, both lists and
           uv.lock's sha256
  cascade  the pooled list of the cascade candidates of --start..--end from the fetched files, as
           a run in <data dir>/labels/<run_id>/ holding the one table `cascades` (cascade.py).
           Every rule parameter is an option (--a, --merge-hours, ...) at the research pass's
           start value; the first --trailing-days only fill the trailing medians. Any day from
           2024-01-01 needs --prereg, as labels does; with it, it reads the days from
           cascade.trailing_days + 1 before the file's confirmatory day to its end, as
           cascade.recall reads, and lists only the windows that close from the confirmatory
           day's 00:00 UTC on, with the file's symbols and [cascade]
  reports  the label-count and stability reports of the labels run --run, into <run>/reports/
           (reports.py). It labels the run's span again from the fetched files and refuses unless
           that gives back the run's own tables; the Parkinson variant reads the same klines.
           --cascades names a cascade run of the same symbols inside the run's span, whose pooled
           list the label-count report counts, as it counts the committed CPI, NFP and FOMC calendar
           (reports.CALENDAR). The aggTrades cross-check is not wired yet, so the stability report
           says "not supplied" for it
  book     fetch Bybit's order-book archive days of one --range (training 2023-01-18..09-30,
           exploratory 2023-10-01..12-31, confirmatory 2024-01-01..2026-09-30, the last only
           with --confirmatory), optionally narrowed by --start/--end, for --symbols. Each day is
           checked, imported with the engine's `store import` (--store: the release binary,
           default target/release/store) and recorded in <out>/<range>/manifest.json (--out,
           default <data dir>/bybit-book); its zip is deleted, and so are its 100 ms samples
           unless --keep-samples. --reduce "<cmd> <args>" runs per day on the samples' Parquet
           (its path appended) before they go (book.py); its last line of output, when a JSON
           object, is kept as the day's `reduced` ("python -m tipping_point.book" writes the
           day's 1-second features under <out>/<range>/features/). A first word `python` runs
           as this interpreter; any other program is named by a full path with forward slashes
           (D:/x/y.exe, as the string is split by shell rules). The reducer is called once with
           --version first and the run refuses to start unless that works. A re-run skips the
           days imported already; a day that fails is recorded and the range goes on; exit 1
           when one failed or was missing

The floor (reports.floor) needs the feature pipelines, which no module here may import:
harness/market_floor.py runs it. The data dir is --data-dir, else the engine's own ($TP_DATA_DIR,
or a per-user folder outside the repository: tipping_point.engine.data_dir).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from .. import engine
from ..ews import prereg
from . import archive, bars, book, cascade, msvol, reports

ARCHIVE = Path("harness") / "fixtures" / "archives" / "binance"
CONFIRMATORY = date(2024, 1, 1)  # no label from here on but through the frozen market-labels-v1
SRC = Path(__file__).resolve().parents[1]  # tipping_point/, whose files --prereg checks
# Held to their blobs at the frozen tag's commit: every module of market/ that _labels and _cascade
# reach, and the package's __init__, which importing any of them runs (test_confirmatory holds the
# list to their imports). Other files of market/ may change and new ones may come: no run reads them
HELD = tuple(f"market/{name}.py" for name in ("__init__", "archive", "bars", "cascade", "msvol"))
# Held to HEAD, which the manifest records: this command line, which gained --prereg after the
# freeze and with it only passes the file's span, symbols and parameters to the held modules, and
# the module that runs the checks
CLEAN = ("market/__main__.py", "ews/prereg.py")
SPAN = ("start", "end", "symbols")  # the options --prereg takes from the file instead
DATA_DIR_HELP = "default: $TP_DATA_DIR, else the engine's per-user data dir"
BAD = ("refused", "failed", "unreadable")  # the fetch statuses that make it exit 1
STORE = Path("target") / "release" / ("store.exe" if sys.platform == "win32" else "store")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.command(args)
    except (archive.ArchiveError, book.BookError, prereg.PreregError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tipping_point.market", description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = parser.add_subparsers(required=True)

    def span(p: argparse.ArgumentParser, required: bool = True) -> None:
        # labels and cascade leave them None when not given, which --prereg requires
        p.add_argument("--start", type=date.fromisoformat, required=required, help="first UTC day")
        p.add_argument("--end", type=date.fromisoformat, required=required, help="last UTC day")
        p.add_argument("--symbols", nargs="+", default=list(archive.SYMBOLS) if required else None)

    def labelling(p: argparse.ArgumentParser, params: type, tag: str) -> None:
        span(p, required=False)
        p.add_argument("--archive", type=Path, default=ARCHIVE)
        p.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
        p.add_argument("--tag", help=f"the run id's tag (default {tag}-<start>-<end>)")
        p.add_argument(
            "--prereg",
            type=Path,
            help=f"the frozen market-labels file, which alone opens days from {CONFIRMATORY}",
        )
        for field in dataclasses.fields(params):
            flag = "--" + field.name.replace("_", "-")
            p.add_argument(flag, type=type(field.default), help=f"default {field.default}")

    p = sub.add_parser("fetch", help="download and verify a span's archive files")
    span(p)
    p.add_argument("--dest", type=Path, default=ARCHIVE)
    p.set_defaults(command=_fetch)

    p = sub.add_parser("labels", help="the regime labels of a span, as a labels run")
    labelling(p, msvol.Params, "binance")
    p.set_defaults(command=_labels)

    p = sub.add_parser("cascade", help="the pooled cascade list of a span, as a run")
    labelling(p, cascade.Params, "cascades")
    p.set_defaults(command=_cascade)

    p = sub.add_parser("reports", help="the label-count and stability reports of a labels run")
    p.add_argument("--run", type=Path, required=True, help="the labels run's folder")
    p.add_argument("--cascades", type=Path, help="a cascade run's folder (default: not supplied)")
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--seed", type=int, default=29, help="the delay simulation's draws")
    p.set_defaults(command=_reports)

    p = sub.add_parser("book", help="fetch and import a range's Bybit order-book days")
    p.add_argument("--range", required=True, choices=list(book.RANGES), dest="range_name")
    p.add_argument("--start", type=date.fromisoformat, help="first UTC day (default: the range's)")
    p.add_argument("--end", type=date.fromisoformat, help="last UTC day (default: the range's)")
    p.add_argument("--symbols", nargs="+", default=list(book.SYMBOLS))
    p.add_argument("--store", type=Path, default=STORE, help="the engine's store binary")
    p.add_argument("--out", type=Path, default=engine.data_dir() / "bybit-book")
    p.add_argument("--confirmatory", action="store_true", help="allow the confirmatory range")
    p.add_argument("--keep-samples", action="store_true", help="keep each day's 100 ms samples")
    p.add_argument(
        "--reduce",
        type=shlex.split,
        help="a command run on each day's samples, such as 'python -m tipping_point.book'; "
        "full paths with forward slashes",
    )
    p.set_defaults(command=_book)
    return parser


def _book(args: argparse.Namespace) -> int:
    if not args.store.is_file():
        print(f"error: no store binary at {args.store} (cargo build --release)", file=sys.stderr)
        return 1
    done = book.fetch(
        args.range_name,
        args.symbols,
        args.out,
        book.http_open,
        book.engine_importer(args.store.resolve()),
        start=args.start,
        end=args.end,
        confirmatory=args.confirmatory,
        keep_samples=args.keep_samples,
        reduce=args.reduce,
        log=lambda line: print(line, flush=True),
    )
    print(book.report(done))
    print(args.out / args.range_name / "manifest.json")
    return 0 if all(e["status"] == "imported" for e in done.values()) else 1


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


def _settings(args: argparse.Namespace, section: str, kind: type) -> dict[str, Any]:
    """The run's start, end, symbols and Params (`kind`, the file's table `section`), the first bar
    whose windows a cascade run lists (`since`), and what its manifest records of the prereg: from
    the command line before the confirmatory span, or all from the frozen file --prereg names.
    Raises PreregError for what it refuses."""
    names = [f.name for f in dataclasses.fields(kind)]
    if args.prereg is None:
        if args.start is None or args.end is None:
            raise prereg.PreregError("--start and --end are required without --prereg")
        if args.end >= CONFIRMATORY:
            raise prereg.PreregError(
                f"days from {CONFIRMATORY} are the confirmatory span, which only --prereg "
                "paper/prereg/market-labels-v1.toml opens, with the file's own span and "
                "parameters (#29)"
            )
        given = {n: getattr(args, n) for n in names if getattr(args, n) is not None}
        return {
            "start": args.start,
            "end": args.end,
            "symbols": list(args.symbols or archive.SYMBOLS),
            "params": dataclasses.replace(kind(), **given),
            "since": 0,
            "record": {},
        }
    given = [n for n in (*SPAN, *names) if getattr(args, n) is not None]
    if given:
        flags = ", ".join("--" + n.replace("_", "-") for n in given)
        raise prereg.PreregError(
            f"--prereg takes the span, the symbols and every parameter from the file: drop {flags}"
        )
    pre = prereg.load(args.prereg)
    if pre.doc["data"]["domain"] != prereg.MARKET_LABELS:
        raise prereg.PreregError(f"{args.prereg} is not a {prereg.MARKET_LABELS} file")
    ref = prereg.frozen(pre)
    code = prereg.code_check(pre, ref, SRC, HELD, CLEAN)
    if code["changed"]:
        raise prereg.PreregError(
            f"these files differ from {ref.tag}'s commit {ref.commit[:12]}: "
            + ", ".join(code["changed"])
            + "; the labels and cascades are made from the tagged commit's code"
        )
    if code["uncommitted"]:
        raise prereg.PreregError(
            "changes not committed in "
            + ", ".join(code["uncommitted"])
            + ": commit them, so that the manifest's code.head pins what ran"
        )
    params = kind(**pre.doc[section])
    span = pre.doc["data"]["span"]
    start, since = span["start"], 0
    if section == "cascade":
        # As cascade.recall reads: a day more than the trailing days early, so that the windows
        # closing from the confirmatory day's 00:00 on all have their history, and only those are
        # listed, so that no earlier candidate enters the list or takes one of its candidates in
        start = span["confirmatory"] - timedelta(days=params.trailing_days + 1)
        since = (span["confirmatory"] - start).days * bars.DAY_BARS
    return {
        "start": start,
        "end": span["end"],
        "symbols": list(pre.doc["data"]["symbols"]),
        "params": params,
        "since": since,
        "record": {"prereg": prereg.stamp(pre, ref), "code": code},
    }


def _labels(args: argparse.Namespace) -> int:
    got = _settings(args, "labeller", msvol.Params)
    args.start, args.end, args.symbols = got["start"], got["end"], got["symbols"]
    params = got["params"]
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
        **got["record"],
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
    got = _settings(args, "cascade", cascade.Params)
    args.start, args.end, args.symbols = got["start"], got["end"], got["symbols"]
    params = got["params"]
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
        found[symbol], windows = cascade.candidates(bar, params, got["since"])
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
        **got["record"],
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
