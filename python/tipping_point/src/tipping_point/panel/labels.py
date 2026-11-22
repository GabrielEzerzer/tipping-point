# Per-coin cascade thresholds, the pooled cascades and an hourly label grid, as one labels run.
# Time convention: at hour t a feature reads only bars that start before t and metrics rows
# stamped at or before t. The grid holds labels alone, each looking forward from t.

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from numpy.lib.stride_tricks import sliding_window_view

from ..ising import io
from ..market import archive, bars, cascade, msvol
from . import span

SD_OI_MULTIPLE = 3.7
SD_RET_MULTIPLE = 5.0
FLOW = 0.15
VOLUME = 3.0
HORIZONS = (1, 6, 24)
SKIP_HOURS = 24
HOUR_NS = cascade.HOUR_NS
MIN_WINDOWS = 2
THRESHOLDS = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("symbol", pa.string()),
        ("windows", pa.int64()),
        ("sd_ret", pa.float64()),
        ("sd_oi", pa.float64()),
        ("a", pa.float64()),
        ("b", pa.float64()),
    ]
)
GRID = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("symbol", pa.string()),
        *[(f"{name}_{h}", pa.bool_()) for h in HORIZONS for name in ("long", "short", "censored")],
        ("skip", pa.bool_()),
    ]
)


def constants() -> dict[str, Any]:
    return {
        "sd_oi_multiple": SD_OI_MULTIPLE,
        "sd_ret_multiple": SD_RET_MULTIPLE,
        "c": FLOW,
        "v": VOLUME,
        "horizons_hours": list(HORIZONS),
        "skip_hours": SKIP_HOURS,
    }


def windows(
    columns: Mapping[str, np.ndarray], params: cascade.Params | None = None
) -> dict[str, np.ndarray]:
    p = params or cascade.Params()
    ts, close, volume = columns["ts_ns"], columns["close"], columns["volume"]
    if np.any(np.diff(ts) != msvol.BAR_NS):
        raise ValueError("bars must lie on a gapless 5-minute grid")
    w, trailing, n = p.window, p.trailing_days * bars.DAY_BARS, len(ts)
    oi = np.where(columns["oi"] > 0, columns["oi"], np.nan)
    taker = np.where(volume == 0, 0.0, columns["taker_buy_share"] * volume)
    summed = np.full(n, np.nan)
    summed[w - 1 :] = sliding_window_view(volume, w).sum(axis=1)

    def touching(missing: np.ndarray) -> Any:
        return sliding_window_view(missing, w + 1).any(axis=1)

    k = np.arange(w + 1, n)
    b = k - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        oi_change, price_change = np.log(oi[k] / oi[k - w]), np.log(close[b] / close[b - w])
    observed = np.concatenate([[0], np.cumsum(~np.isnan(summed))])
    low = b - w - trailing + 1
    no_history = (low < w - 1) | (observed[b - w + 1] == observed[np.maximum(low, 0)])
    no_oi = touching(np.isnan(oi))[k - w]
    no_bar = touching(np.isnan(close) | np.isnan(volume) | np.isnan(taker))[b - w]
    return {
        "ts_ns": ts[k],
        "oi_change": oi_change,
        "price_change": price_change,
        "evaluated": ~no_oi & ~no_bar & ~no_history,
    }


def thresholds(win: Mapping[str, np.ndarray], first_ns: int, end_ns: int) -> dict[str, Any]:
    inside = win["evaluated"] & (win["ts_ns"] >= first_ns) & (win["ts_ns"] < end_ns)
    count = int(inside.sum())
    if count < MIN_WINDOWS:
        raise ValueError(f"{count} evaluated windows in the span: too few for a threshold")
    sd_ret = float(np.std(win["price_change"][inside], ddof=1))
    sd_oi = float(np.std(win["oi_change"][inside], ddof=1))
    return {
        "windows": count,
        "sd_ret": sd_ret,
        "sd_oi": sd_oi,
        "a": SD_OI_MULTIPLE * sd_oi,
        "b": SD_RET_MULTIPLE * sd_ret,
    }


def params(threshold: Mapping[str, float]) -> cascade.Params:
    return cascade.Params(a=threshold["a"], b=threshold["b"], c=FLOW, v=VOLUME)


@dataclass(frozen=True)
class Coin:
    windows_ts: np.ndarray
    evaluated: np.ndarray
    last_end_ns: int


def coin(
    columns: Mapping[str, np.ndarray], first_ns: int, end_ns: int
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int], Coin]:
    win = windows(columns)
    threshold = thresholds(win, first_ns, end_ns)
    since = int(np.searchsorted(columns["ts_ns"], first_ns))
    found, counts = cascade.candidates(columns, params(threshold), since)
    asked = win["ts_ns"] >= first_ns
    if counts["windows_evaluated"] != int((win["evaluated"] & asked).sum()):
        raise RuntimeError("the panel's windows and cascade.candidates' disagree")
    ts = columns["ts_ns"]
    complete = np.flatnonzero(~np.isnan(columns["close"]) & (ts < end_ns))
    last_end = int(ts[complete[-1]]) + msvol.BAR_NS if complete.size else int(first_ns)
    return threshold, found, counts, Coin(win["ts_ns"], win["evaluated"], last_end)


def grid(
    hours: np.ndarray, coin: Coin, end_ns: int, rows: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    unevaluated = np.concatenate([[0], np.cumsum(~coin.evaluated)])
    out: dict[str, np.ndarray] = {}
    for h in HORIZONS:
        stop = hours + h * HOUR_NS
        lo = np.searchsorted(coin.windows_ts, hours, "right")
        hi = np.searchsorted(coin.windows_ts, stop, "right")
        gaps = (unevaluated[hi] > unevaluated[lo]) | (hi - lo < h * HOUR_NS // msvol.BAR_NS)
        for name, kind in (("long", cascade.LONG), ("short", cascade.SHORT)):
            onsets = np.sort(np.array([r["ts_ns"] for r in rows if r["kind"] == kind], np.int64))
            found = np.searchsorted(onsets, stop, "left") - np.searchsorted(onsets, hours, "left")
            out[f"{name}_{h}"] = found > 0
        out[f"censored_{h}"] = (stop > end_ns) | (stop > coin.last_end_ns) | gaps
    skip = np.zeros(len(hours), bool)
    for r in rows:
        last = r["last_fired_ts_ns"] + SKIP_HOURS * HOUR_NS
        first = np.searchsorted(hours, r["ts_ns"], "right")
        skip[first : np.searchsorted(hours, last, "right")] = True
    out["skip"] = skip
    return out


def uv_lock() -> str:
    for folder in Path(__file__).resolve().parents:
        lock = folder / "uv.lock"
        if lock.is_file():
            return hashlib.sha256(lock.read_bytes()).hexdigest()
    raise FileNotFoundError("no uv.lock above the package: the run records its sha256")


def ns(day: date) -> int:
    return archive.day_ms(day) * 1_000_000


def run(
    root: Path,
    data_dir: Path,
    symbols: Sequence[str],
    start: date,
    dev_start: date,
    end: date,
    tag: str,
    log: Callable[[str], None] = lambda line: None,
) -> tuple[Path, dict[str, Any]]:
    span.refuse(start, dev_start, end)
    if not start < dev_start <= end:
        raise ValueError(f"the span must run {start} < {dev_start} <= {end}")
    lock = uv_lock()
    first_ns, end_ns = ns(dev_start), ns(end + timedelta(days=1))
    days = (end - start).days + 1
    found: dict[str, list[dict[str, Any]]] = {}
    coins: dict[str, Coin] = {}
    rows_thresholds: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    missing: list[str] = []
    for symbol in symbols:
        got = archive.collect([symbol], start, end, root)
        sources += got.sources
        missing += got.missing
        columns, bar_checks = bars.bars(
            archive.day_ms(start),
            days,
            *(got.columns[(symbol, kind)] for kind in archive.KINDS),
        )
        threshold, found[symbol], counts, coins[symbol] = coin(columns, first_ns, end_ns)
        rows_thresholds.append({"ts_ns": first_ns, "symbol": symbol, **threshold})
        checks[symbol] = {
            **bar_checks,
            **counts,
            "candidates": len(found[symbol]),
            "files_missing": len(got.missing),
        }
        log(
            f"{symbol:<13} sd_ret {threshold['sd_ret']:.5f} sd_oi {threshold['sd_oi']:.5f} "
            f"a {threshold['a']:.4f} b {threshold['b']:.4f}; {counts['windows_evaluated']} of "
            f"{counts['windows']} windows evaluated, {len(found[symbol])} candidates"
        )
    rows = cascade.pool(found)
    hours = np.arange(first_ns, end_ns, HOUR_NS, dtype=np.int64)
    parts = []
    for symbol in symbols:
        mine = [r for r in rows if r["symbol"] == symbol]
        labels = grid(hours, coins[symbol], end_ns, mine)
        parts.append(
            pa.table({"ts_ns": hours, "symbol": [symbol] * len(hours), **labels}, schema=GRID)
        )
    table_grid = pa.concat_tables(parts).sort_by([("ts_ns", "ascending"), ("symbol", "ascending")])
    coins_per_event = Counter(r["market_event"] for r in rows)
    checks |= {
        "asset_instances": len(rows),
        "by_kind": dict(Counter(r["kind"] for r in rows)),
        "market_events": len(coins_per_event),
        "market_events_5_or_more": sum(n >= 5 for n in coins_per_event.values()),
        "grid": shares(table_grid),
    }
    run_params = {
        "symbols": list(symbols),
        "start": start.isoformat(),
        "dev_start": dev_start.isoformat(),
        "end": end.isoformat(),
        "constants": constants(),
        "cascade": dataclasses.asdict(cascade.Params(c=FLOW, v=VOLUME)) | {"a": None, "b": None},
        "code": {"head": io.code_version()["git_sha"], "uv_lock_sha256": lock},
    }
    tables = {
        "thresholds": pa.Table.from_pylist(rows_thresholds, schema=THRESHOLDS),
        "cascades": pa.Table.from_pylist(rows, schema=cascade.SCHEMA),
        "grid": table_grid,
    }
    run_dir = msvol.write_run(data_dir, tag, tables, run_params, sources, missing, checks)
    return run_dir, checks


def shares(table: pa.Table) -> dict[str, float]:
    n = max(table.num_rows, 1)
    column = {
        name: table[name].to_numpy(zero_copy_only=False)
        for name in table.column_names
        if name not in ("ts_ns", "symbol")
    }
    for h in HORIZONS:
        column[f"positive_{h}"] = column[f"long_{h}"] | column[f"short_{h}"]
    return {name: float(np.sum(values)) / n for name, values in column.items()}
