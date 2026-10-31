"""The liquidation-cascade proxy and its pooled list (#29 S4; research pass section 3).

The span's liquidations are not freely available, so a cascade is read from a symbol's 5-minute
bars (bars.py). A window is the 30 minutes, W = 6 bars, that close at the start t_k of a bar k. Over
that one window, with OI the open interest in contracts (its USDT value falls whenever the price
does) and P the price:

  oi_change      ln(OI(t_k) / OI(t_k - 30 min)), the oi of bars k and k-W
  price_change   ln(P(t_k) / P(t_k - 30 min)), the close of bars k-1 and k-W-1
  volume         V, the volume of bars k-W..k-1
  flow           F = (2 taker_buy - V) / V over those bars, in [-1, 1]
  volume_median  the median V of the windows closing in the `trailing_days` before this one opens,
                 so never the window itself

A long cascade has oi_change <= -a, price_change <= -b, F <= -c and V >= v x volume_median. A short
squeeze has the same fall in open interest with price_change >= b and F >= c. Open interest falling
while price and aggressive flow move together is what separates forced deleveraging from positions
being opened.

The alignment is fixed by the feed's timing, so it is no parameter: a bar's oi is the snapshot at
its start and its close the price at its end, which pairs oi[k] with close[k-1]. The evidence, in
every month of 2023 on both symbols: a change in open interest correlates most with the 1-minute
returns between its two stamps, not with those of the step after. Pairing oi[k] with close[k] would
set open interest's window 5 minutes behind the price's, and in a sharp V it misses the fall and
takes the rebound for a squeeze.

A window that fires is a candidate at bar k, reading that bar's oi and none of its klines. Its onset
is the first bar of the window whose one-bar return, signed so that the candidate's direction is
positive, exceeds the 99th percentile of all returns signed the same way over the bar's own trailing
days; if none does, the window's first bar.

A window touching a missing open-interest step (bars k-W..k) or a missing bar (bars k-W-1..k-1) is
not evaluated and is counted, and so is one whose trailing days do not all exist yet or hold no
complete window. An open interest of zero or below counts as missing: Binance's metrics hold such
rows. A bar that traded nothing is not missing: it adds no volume and no flow.

Candidates of a symbol within `merge_hours` after the earliest of their group merge into it,
whichever their side, which makes one asset instance with the earliest's onset, side and
measurements. Instances firing within `both_hours` of the first of their group are one market
event. Both steps go forward in firing order, so nothing written is rewritten by a later bar
except a group's candidate count and last firing (tests/market/test_cascade.py checks it bit for
bit).

The pooled list holds one row per asset instance, with its market event's number and the bounds of
the L2 window #23 and #30 download, `bounds_hours` either side of its onset. It is written as the
table `cascades` of a labels run (msvol.write_run), with both counts in complete.json.

documented_cascades.toml lists cascades documented in the span. `recall` runs the rule over them;
it reads confirmatory days, so it runs only with TP_CASCADE_RECALL set and gives only what the rule
says inside each event's own days. `liquidations` reads the table #90 will record from Bybit's
allLiquidation stream, and reports "not recorded" until it exists. It stays a proxy: the research
pass found no peer-reviewed validation of one, and the paper is to say so.

Run it with `python -m tipping_point.market.cascade --start <day> --end <day>`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from numpy.lib.stride_tricks import sliding_window_view

from .. import engine
from . import archive, bars, msvol
from .__main__ import ARCHIVE, CONFIRMATORY, DATA_DIR_HELP

HOUR_NS = 3_600 * 10**9
LONG = "long_cascade"
SHORT = "short_squeeze"
SWITCH = "TP_CASCADE_RECALL"
DOCUMENTED = Path(__file__).with_name("documented_cascades.toml")
SCHEMA = pa.schema(
    [
        ("ts_ns", pa.int64()),  # the onset; like every table's ts_ns, a bar's start
        ("symbol", pa.string()),
        ("market_event", pa.int32()),
        ("kind", pa.string()),
        ("fired_ts_ns", pa.int64()),  # t_k of the earliest candidate: its window's close
        ("last_fired_ts_ns", pa.int64()),
        ("candidates", pa.int32()),
        ("window_start_ts_ns", pa.int64()),
        ("window_end_ts_ns", pa.int64()),
        *[
            (name, pa.float64())
            for name in ("oi_change", "price_change", "flow", "volume", "volume_median")
        ],
    ]
)


@dataclass(frozen=True)
class Params:
    """Every number of the rule, at the research pass's start values (#29, section 3). The 2023
    pilot sets a, b, c and v to about 10-15 pooled events a year, then market-labels-v1 freezes
    them. How open interest lines up with price is not among them: the feed's timing fixes it (see
    the notes above), so no count of events can choose it."""

    a: float = 0.03  # the fall in open interest, as a log ratio
    b: float = 0.025  # the move in price, as a log ratio
    c: float = 0.15  # the flow imbalance
    v: float = 3.0  # the window's volume over the trailing median
    window: int = 6  # W, in bars: 30 minutes
    trailing_days: int = 28
    onset_percentile: float = 99.0
    merge_hours: int = 24
    both_hours: int = 1
    bounds_hours: int = 12


def candidates(
    columns: Mapping[str, np.ndarray], params: Params | None = None, since: int = 0
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """A symbol's candidates in firing order, from the columns of bars.bars, and the counts of its
    windows, all for windows closing at bar `since` or later. Earlier bars are only history: the
    rule says nothing about them."""
    p = params or Params()
    ts, close, volume = columns["ts_ns"], columns["close"], columns["volume"]
    if np.any(np.diff(ts) != msvol.BAR_NS):
        raise ValueError("bars must lie on a gapless 5-minute grid")
    w, trailing, n = p.window, p.trailing_days * bars.DAY_BARS, len(ts)
    oi = np.where(columns["oi"] > 0, columns["oi"], np.nan)
    taker = np.where(volume == 0, 0.0, columns["taker_buy_share"] * volume)

    def total(values: np.ndarray) -> np.ndarray:
        # Each sum from its own W bars, not a running total, so it does not depend on where the
        # series starts, and one missing bar touches only the windows that hold it.
        out = np.full(n, np.nan)
        out[w - 1 :] = sliding_window_view(values, w).sum(axis=1)
        return out

    def touching(missing: np.ndarray) -> Any:
        return sliding_window_view(missing, w + 1).any(axis=1)

    summed = total(volume)  # by last bar; every array from k on holds one value per window
    with np.errstate(divide="ignore", invalid="ignore"):
        flows = (2 * total(taker) - summed) / summed
    k = np.arange(w + 1, n)
    b = k - 1  # the window's last bar, which closes as bar k's oi is stamped
    oi_change, price_change = np.log(oi[k] / oi[k - w]), np.log(close[b] / close[b - w])
    flow = flows[b]
    observed = np.concatenate([[0], np.cumsum(~np.isnan(summed))])
    low = b - w - trailing + 1  # the first of the windows in the trailing days
    no_history = (low < w - 1) | (observed[b - w + 1] == observed[np.maximum(low, 0)])
    no_oi = touching(np.isnan(oi))[k - w]
    no_bar = touching(np.isnan(close) | np.isnan(volume) | np.isnan(taker))[b - w]
    asked = k >= since
    evaluated = asked & ~no_oi & ~no_bar & ~no_history
    windows = {
        "windows": int(np.sum(asked)),
        "windows_missing_oi": int(np.sum(asked & no_oi)),
        "windows_missing_bars": int(np.sum(asked & ~no_oi & no_bar)),
        "windows_no_history": int(np.sum(asked & ~no_oi & ~no_bar & no_history)),
        "windows_evaluated": int(np.sum(evaluated)),
    }

    falls = oi_change <= -p.a
    is_short = falls & (price_change >= p.b) & (flow >= p.c)
    is_long = falls & (price_change <= -p.b) & (flow <= -p.c)
    returns = np.full(n, np.nan)
    returns[1:] = np.log(close[1:] / close[:-1])
    found: list[dict[str, Any]] = []
    # Few pass the three cheap legs, so the medians and percentiles are taken here, not per bar.
    for i in np.flatnonzero(evaluated & (is_long | is_short)).tolist():
        bar = int(b[i])
        median = float(np.median(_seen(summed[bar - w - trailing + 1 : bar - w + 1])))
        if summed[bar] >= p.v * median:
            found.append(
                {
                    "ts_ns": int(ts[_onset(returns, bar, 1 if is_short[i] else -1, trailing, p)]),
                    "fired_ts_ns": int(ts[k[i]]),
                    "kind": SHORT if is_short[i] else LONG,
                    "oi_change": float(oi_change[i]),
                    "price_change": float(price_change[i]),
                    "flow": float(flow[i]),
                    "volume": float(summed[bar]),
                    "volume_median": median,
                }
            )
    return found, windows


def _seen(values: np.ndarray) -> np.ndarray:
    return values[~np.isnan(values)]


def _onset(returns: np.ndarray, bar: int, side: int, trailing: int, p: Params) -> int:
    """The onset of the window ending at `bar`, side being the candidate's direction: +1 up, -1
    down."""
    moves = side * returns
    first = bar - p.window + 1
    for i in range(first, bar + 1):
        past = _seen(moves[max(i - trailing, 0) : i])
        if past.size and moves[i] > np.percentile(past, p.onset_percentile):
            return i
    return first


def merge(found: Sequence[dict[str, Any]], params: Params | None = None) -> list[dict[str, Any]]:
    """A symbol's candidates as asset instances: each adds its firing to the group of the earliest
    within merge_hours, so an instance never reaches past merge_hours however long the firing goes
    on."""
    p = params or Params()
    out: list[dict[str, Any]] = []
    for candidate in found:
        fired = candidate["fired_ts_ns"]
        if out and fired - out[-1]["fired_ts_ns"] <= p.merge_hours * HOUR_NS:
            out[-1]["candidates"] += 1
            out[-1]["last_fired_ts_ns"] = fired
        else:
            out.append({**candidate, "candidates": 1, "last_fired_ts_ns": fired})
    return out


def pool(
    found: Mapping[str, Sequence[dict[str, Any]]], params: Params | None = None
) -> list[dict[str, Any]]:
    """The pooled list from each symbol's candidates, in (onset, symbol) order: the rows of table
    `cascades`."""
    p = params or Params()
    rows = [{**row, "symbol": symbol} for symbol, some in found.items() for row in merge(some, p)]
    event, first = -1, 0
    for row in sorted(rows, key=lambda r: (r["fired_ts_ns"], r["symbol"])):
        if event < 0 or row["fired_ts_ns"] - first > p.both_hours * HOUR_NS:
            event, first = event + 1, row["fired_ts_ns"]
        row["market_event"] = event
        row["window_start_ts_ns"] = row["ts_ns"] - p.bounds_hours * HOUR_NS
        row["window_end_ts_ns"] = row["ts_ns"] + p.bounds_hours * HOUR_NS
    return sorted(rows, key=lambda r: (r["ts_ns"], r["symbol"]))


def documented() -> list[dict[str, Any]]:
    """The committed list of documented cascades: name, first_day, last_day, sources, note."""
    return tomllib.loads(DOCUMENTED.read_text(encoding="utf-8"))["event"]


def recall(
    root: Path,
    params: Params | None = None,
    events: Sequence[dict[str, Any]] | None = None,
    symbols: Sequence[str] = archive.SYMBOLS,
    environ: Mapping[str, str] = os.environ,
) -> list[dict[str, Any]]:
    """Whether the rule fires inside each documented event's own days, from the archive at `root`:
    one row per event, `recalled` None when no symbol's days could be read. Each read starts a day
    more than the trailing days early and ends with the event, and only windows closing inside the
    event are asked for, so nothing else in the confirmatory span comes out or is even computed."""
    if not environ.get(SWITCH):
        raise PermissionError(f"recall reads confirmatory days: it runs only with {SWITCH} set")
    p = params or Params()
    out = []
    for event in documented() if events is None else events:
        first, last = event["first_day"], event["last_day"]
        start = first - timedelta(days=p.trailing_days + 1)
        found: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            try:
                span = archive.collect([symbol], start, last, root)
            except archive.ArchiveError as error:
                found[symbol] = {"status": "not read", "error": str(error)}
                continue
            bar, _ = bars.bars(
                archive.day_ms(start),
                (last - start).days + 1,
                *(span.columns[(symbol, kind)] for kind in archive.KINDS),
            )
            since = (first - start).days * bars.DAY_BARS
            rows, windows = candidates(bar, p, since)
            found[symbol] = {
                "status": "read",
                "candidates": len(rows),
                "first": rows[0] if rows else None,
                **windows,
            }
        read = [f for f in found.values() if f["status"] == "read"]
        out.append(
            {
                "name": event["name"],
                "first_day": first.isoformat(),
                "last_day": last.isoformat(),
                "recalled": any(f["candidates"] for f in read) if read else None,
                "symbols": found,
            }
        )
    return out


def liquidations(
    data_dir: Path, rows: Sequence[dict[str, Any]], params: Params | None = None
) -> dict[str, Any]:
    """The forward check on the pooled list: for each row inside the span recorded in the
    `liquidations` tables of the market runs in `data_dir`, the notional liquidated on its symbol
    during the 30 minutes of its earliest window. "not recorded" while no such table holds a row."""
    # shortcut: the columns are assumed (ts_ns, symbol, price and qty as plain numbers) and there is
    # no precision or recall, until #90 fixes the table and months of it give a notional threshold
    p = params or Params()
    files = sorted(str(f) for f in (Path(data_dir) / "market").glob("*/liquidations/*.parquet"))
    if not files:
        return {"status": "not recorded"}
    table = ds.dataset(files, format="parquet").to_table(
        columns=["ts_ns", "symbol", "price", "qty"]
    )
    if not table.num_rows:
        return {"status": "not recorded"}
    ts, symbol = table["ts_ns"].to_numpy(), table["symbol"].to_numpy()
    notional = table["price"].to_numpy() * table["qty"].to_numpy()
    first_ts, last_ts = int(ts.min()), int(ts.max())
    checked = []
    for row in rows:
        end = row["fired_ts_ns"]
        start = end - p.window * msvol.BAR_NS
        if first_ts <= start and end <= last_ts:
            inside = (symbol == row["symbol"]) & (start <= ts) & (ts < end)
            checked.append(
                {
                    "ts_ns": row["ts_ns"],
                    "symbol": row["symbol"],
                    "notional": float(notional[inside].sum()),
                }
            )
    return {
        "status": "recorded",
        "rows": table.num_rows,
        "first_ts_ns": first_ts,
        "last_ts_ns": last_ts,
        "cascades": checked,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tipping_point.market.cascade",
        description="The pooled list of a span's cascade candidates, as a labels run. The first "
        "28 days only fill the trailing medians. Refuses any day from 2024-01-01, as `labels` "
        "does: the confirmatory span waits for the frozen market-labels-v1.",
    )
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="first UTC day")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="last UTC day")
    parser.add_argument("--symbols", nargs="+", default=list(archive.SYMBOLS))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--data-dir", type=Path, default=engine.data_dir(), help=DATA_DIR_HELP)
    parser.add_argument("--tag", help="the run id's tag (default cascades-<start>-<end>)")
    for field in dataclasses.fields(Params):
        flag = "--" + field.name.replace("_", "-")
        parser.add_argument(flag, type=type(field.default), default=field.default)
    args = parser.parse_args(argv)
    if args.end >= CONFIRMATORY:
        print(
            f"error: cascades end on {CONFIRMATORY.year - 1}-12-31: days from {CONFIRMATORY} are "
            "the confirmatory span, closed until market-labels-v1 is frozen (#29 S5)",
            file=sys.stderr,
        )
        return 1
    params = Params(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Params)})
    try:
        span = archive.collect(args.symbols, args.start, args.end, args.archive)
    except archive.ArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    days = (args.end - args.start).days + 1
    found: dict[str, list[dict[str, Any]]] = {}
    checks: dict[str, Any] = {}
    for symbol in args.symbols:
        bar, checks[symbol] = bars.bars(
            archive.day_ms(args.start),
            days,
            *(span.columns[(symbol, kind)] for kind in archive.KINDS),
        )
        found[symbol], windows = candidates(bar, params)
        checks[symbol] |= windows | {"candidates": len(found[symbol])}
    rows = pool(found, params)
    per_symbol = Counter(row["symbol"] for row in rows)
    checks["asset_instances"] = len(rows)
    checks["market_events"] = len({row["market_event"] for row in rows})
    checks["liquidations"] = liquidations(args.data_dir, rows, params)
    run_params = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "symbols": list(args.symbols),
        "cascade": dataclasses.asdict(params),
    }
    run_dir = msvol.write_run(
        args.data_dir,
        args.tag or f"cascades-{args.start:%Y%m%d}-{args.end:%Y%m%d}",
        {"cascades": pa.Table.from_pylist(rows, schema=SCHEMA)},
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


if __name__ == "__main__":
    sys.exit(main())
