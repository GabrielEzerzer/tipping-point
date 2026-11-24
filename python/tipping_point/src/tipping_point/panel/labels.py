# Per-coin trailing cascade thresholds, the pooled cascades and an hourly label grid, as one run.
# Time convention: at hour t a feature reads only bars that start before t and metrics rows stamped
# at or before t. Thresholds, detection, skip and censoring read nothing at or after their own time.

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
TRAILING_DAYS = 90
MIN_WINDOWS = 28 * bars.DAY_BARS * 0.8
MAX_MISSING = 0.05
HORIZONS = (1, 6, 24)
SKIP_HOURS = 24
PAST_HOURS = 24
HOUR_NS = cascade.HOUR_NS
DAY_NS = bars.DAY_BARS * msvol.BAR_NS
LEAD_NS = cascade.Params().window * msvol.BAR_NS
THRESHOLDS = pa.schema(
    [
        ("ts_ns", pa.int64()),
        ("symbol", pa.string()),
        ("windows", pa.int64()),
        ("missing", pa.int64()),
        ("sd_ret", pa.float64()),
        ("sd_oi", pa.float64()),
        ("a", pa.float64()),
        ("b", pa.float64()),
    ]
)
CASCADES = cascade.SCHEMA.append(pa.field("in_span", pa.bool_()))
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
        "trailing_days": TRAILING_DAYS,
        "min_windows": MIN_WINDOWS,
        "max_missing": MAX_MISSING,
        "horizons_hours": list(HORIZONS),
        "skip_hours": SKIP_HOURS,
        "past_hours": PAST_HOURS,
    }


def windows(
    columns: Mapping[str, np.ndarray], params: cascade.Params | None = None
) -> dict[str, np.ndarray]:
    p = params or cascade.Params()
    ts, close, volume = columns["ts_ns"], columns["close"], columns["volume"]
    if np.any(np.diff(ts) != msvol.BAR_NS):
        raise ValueError("bars must lie on a gapless 5-minute grid")
    if np.any(close <= 0):
        first = int(ts[np.flatnonzero(close <= 0)[0]])
        raise ValueError(f"the bar starting at {first} ns closes at zero or below: no log return")
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
    absent = np.concatenate([[0], np.cumsum(np.isnan(close) | np.isnan(volume))])
    history_gap = absent[b - w + 1] > absent[np.maximum(low - w + 1, 0)]
    return {
        "ts_ns": ts[k],
        "oi_change": oi_change,
        "price_change": price_change,
        "evaluated": ~no_oi & ~no_bar & ~no_history,
        "history_gap": history_gap,
    }


def daily(win: Mapping[str, np.ndarray], first_ns: int, days: int) -> list[dict[str, Any]]:
    ok, every = win["evaluated"], win["ts_ns"]
    ts, ret, oi = every[ok], win["price_change"][ok], win["oi_change"][ok]
    since = int(ts[0]) if len(ts) else None
    out = []
    for d in range(days):
        day = first_ns + d * DAY_NS
        lo, hi = (int(i) for i in np.searchsorted(ts, [day - TRAILING_DAYS * DAY_NS, day]))
        opened = day if since is None else max(day - TRAILING_DAYS * DAY_NS, since)
        closing = np.searchsorted(every, [opened, day])
        row: dict[str, Any] = {
            "ts_ns": day,
            "windows": hi - lo,
            "missing": max(int(closing[1] - closing[0]), 0) - (hi - lo),
        }
        sd_ret = sd_oi = 0.0
        if hi - lo >= MIN_WINDOWS and missing_share(row) <= MAX_MISSING:
            sd_ret = float(np.std(ret[lo:hi], ddof=1))
            sd_oi = float(np.std(oi[lo:hi], ddof=1))
        if np.isfinite([sd_ret, sd_oi]).all() and sd_ret > 0 and sd_oi > 0:
            row |= {
                "sd_ret": sd_ret,
                "sd_oi": sd_oi,
                "a": SD_OI_MULTIPLE * sd_oi,
                "b": SD_RET_MULTIPLE * sd_ret,
            }
        else:
            row |= {"sd_ret": None, "sd_oi": None, "a": None, "b": None}
        out.append(row)
    return out


def missing_share(row: Mapping[str, Any]) -> float:
    total = row["windows"] + row["missing"]
    return row["missing"] / total if total else 0.0


def params(threshold: Mapping[str, Any]) -> cascade.Params | None:
    if threshold["a"] is None:
        return None
    return cascade.Params(a=threshold["a"], b=threshold["b"], c=FLOW, v=VOLUME)


def by_day(
    columns: Mapping[str, np.ndarray], day_params: Sequence[cascade.Params | None]
) -> tuple[list[dict[str, Any]], dict[str, int], np.ndarray]:
    ts = columns["ts_ns"]
    if len(ts) and int(ts[0]) % DAY_NS:
        raise ValueError("the bars must start at a UTC midnight")
    found: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    evaluated = np.zeros(len(day_params), np.int64)
    for d, p in enumerate(day_params):
        first, stop = d * bars.DAY_BARS, min((d + 1) * bars.DAY_BARS, len(ts))
        if p is None or first >= stop:
            continue
        lo = max(first - p.trailing_days * bars.DAY_BARS - 4 * p.window, 0)
        part = {name: values[lo:stop] for name, values in columns.items()}
        got, n = cascade.candidates(part, p, first - lo)
        found += got
        counts.update(n)
        evaluated[d] = n["windows_evaluated"]
    return found, dict(counts), evaluated


@dataclass(frozen=True)
class Coin:
    windows_ts: np.ndarray
    clean: np.ndarray


def coin(
    columns: Mapping[str, np.ndarray], end_ns: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], Coin]:
    kept = columns["ts_ns"] < end_ns
    columns = {name: values[kept] for name, values in columns.items()}
    ts = columns["ts_ns"]
    if not len(ts):
        raise ValueError("no bars before the end of the span")
    win = windows(columns)
    days = -(-len(ts) // bars.DAY_BARS)
    thresholds = daily(win, int(ts[0]), days)
    day_params = [params(row) for row in thresholds]
    found, counts, per_day = by_day(columns, day_params)
    day = (win["ts_ns"] - ts[0]) // DAY_NS
    has = np.array([p is not None for p in day_params])[day]
    evaluated = win["evaluated"] & has
    if not np.array_equal(np.bincount(day[evaluated], minlength=days), per_day):
        raise RuntimeError("the panel's windows and cascade.candidates' disagree")
    return thresholds, found, counts, Coin(win["ts_ns"], evaluated & ~win["history_gap"])


def grid(hours: np.ndarray, coin: Coin, rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    unclean = np.concatenate([[0], np.cumsum(~coin.clean)])
    begins = {
        kind: np.sort(
            np.array([r["fired_ts_ns"] - LEAD_NS for r in rows if r["kind"] == kind], np.int64)
        )
        for kind in (cascade.LONG, cascade.SHORT)
    }
    out: dict[str, np.ndarray] = {}
    lo = np.searchsorted(coin.windows_ts, hours - PAST_HOURS * HOUR_NS, "left")
    for h in HORIZONS:
        stop = hours + h * HOUR_NS
        hi = np.searchsorted(coin.windows_ts, stop + LEAD_NS, "left")
        need = ((PAST_HOURS + h) * HOUR_NS + LEAD_NS) // msvol.BAR_NS
        for name, kind in (("long", cascade.LONG), ("short", cascade.SHORT)):
            found = np.searchsorted(begins[kind], stop, "left") - np.searchsorted(
                begins[kind], hours, "left"
            )
            out[f"{name}_{h}"] = found > 0
        out[f"censored_{h}"] = (unclean[hi] > unclean[lo]) | (hi - lo < need)
    skip = np.zeros(len(hours), bool)
    for r in rows:
        first = np.searchsorted(hours, r["fired_ts_ns"], "left")
        last = r["last_fired_ts_ns"] + SKIP_HOURS * HOUR_NS
        skip[first : np.searchsorted(hours, last, "right")] = True
    out["skip"] = skip
    return out


def label(
    found: Mapping[str, Sequence[dict[str, Any]]],
    coins: Mapping[str, Coin],
    first_ns: int,
    end_ns: int,
) -> tuple[list[dict[str, Any]], pa.Table]:
    rows = cascade.pool(found)
    for row in rows:
        row["in_span"] = row["fired_ts_ns"] - LEAD_NS >= first_ns
    hours = np.arange(first_ns, end_ns, HOUR_NS, dtype=np.int64)
    parts = []
    for symbol in found:
        mine = [r for r in rows if r["symbol"] == symbol]
        labels = grid(hours, coins[symbol], mine)
        parts.append(
            pa.table({"ts_ns": hours, "symbol": [symbol] * len(hours), **labels}, schema=GRID)
        )
    table = pa.concat_tables(parts).sort_by([("ts_ns", "ascending"), ("symbol", "ascending")])
    return rows, table


def row_checks(files: Mapping[str, Any], symbol: str, start: date, end: date) -> dict[str, Any]:
    out: dict[str, Counter[str]] = {kind: Counter() for kind in archive.KINDS}
    for source in archive.sources([symbol], start, end):
        checks = files.get(source.path, {}).get("checks")
        if not checks:
            continue
        total = out[source.kind]
        total["files"] += 1
        total["rows"] += checks["rows"]
        total["files_with_header"] += bool(checks["header"])
        total["unsorted"] += checks["unsorted"]
        if "off_grid" in checks:
            total["off_grid"] += checks["off_grid"]
            total["duplicate_slots"] += len(checks["duplicates"])
            total["gap_runs"] += len(checks["gaps"])
    return {kind: dict(total) for kind, total in out.items()}


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
    span.refuse_archive(root)
    if not start < dev_start <= end:
        raise ValueError(f"the span must run {start} < {dev_start} <= {end}")
    symbols = list(dict.fromkeys(symbols))
    lock = uv_lock()
    first_ns, end_ns = ns(dev_start), ns(end + timedelta(days=1))
    days = (end - start).days + 1
    manifest_path = root / "manifest.json"
    files = io.read_json(manifest_path)["files"] if manifest_path.exists() else {}
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
        thresholds, found[symbol], counts, coins[symbol] = coin(columns, end_ns)
        rows_thresholds += [{"symbol": symbol, **row} for row in thresholds]
        set_days = [row for row in thresholds if row["b"] is not None]
        decided = [row for row in thresholds if row["windows"] >= MIN_WINDOWS]
        worst = max(decided, key=missing_share, default=None)
        checks[symbol] = {
            **bar_checks,
            **counts,
            "candidates": len(found[symbol]),
            "files_missing": len(got.missing),
            "days_with_threshold": len(set_days),
            "first_threshold_ns": set_days[0]["ts_ns"] if set_days else None,
            "days_over_max_missing": sum(missing_share(r) > MAX_MISSING for r in decided),
            "worst_missing_share": missing_share(worst) if worst else None,
            "worst_missing_ns": worst["ts_ns"] if worst else None,
            "row_checks": row_checks(files, symbol, start, end),
        }
        b = [row["b"] for row in set_days]
        log(
            f"{symbol:<13} {len(set_days)} of {len(thresholds)} days with a threshold, "
            + (f"b {min(b):.4f}..{max(b):.4f}, " if b else "")
            + f"{counts.get('windows_evaluated', 0)} windows evaluated, "
            f"{len(found[symbol])} candidates"
        )
    rows, table_grid = label(found, coins, first_ns, end_ns)
    inside = [r for r in rows if r["in_span"]]
    coins_per_event = Counter(r["market_event"] for r in inside)
    checks |= {
        "asset_instances": len(inside),
        "instances_before_span": len(rows) - len(inside),
        "by_kind": dict(Counter(r["kind"] for r in inside)),
        "by_symbol": {s: sum(r["symbol"] == s for r in inside) for s in symbols},
        "market_events": len(coins_per_event),
        "market_events_5_or_more": sum(n >= 5 for n in coins_per_event.values()),
        "grid": shares(table_grid),
    }
    run_params = {
        "symbols": symbols,
        "start": start.isoformat(),
        "dev_start": dev_start.isoformat(),
        "end": end.isoformat(),
        "constants": constants(),
        "cascade": dataclasses.asdict(cascade.Params(c=FLOW, v=VOLUME)) | {"a": None, "b": None},
        "code": {"head": io.code_version()["git_sha"], "uv_lock_sha256": lock},
    }
    rows_thresholds.sort(key=lambda r: (r["ts_ns"], r["symbol"]))
    tables = {
        "thresholds": pa.Table.from_pylist(rows_thresholds, schema=THRESHOLDS),
        "cascades": pa.Table.from_pylist(rows, schema=CASCADES),
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
