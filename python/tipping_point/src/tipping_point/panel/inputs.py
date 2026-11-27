# Model inputs per coin and UTC hour t of a labels grid (docs/market.md, "Coin panel"). At t an
# input reads bars starting before t, open interest stamped at or before t, the ratio and funding of
# bar t - 5 min and the labels run's thresholds of t's day; tests/panel/test_inputs.py checks it.

from __future__ import annotations

import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

from ..ews.trend import kendall_tau
from ..ising import io
from ..market import archive, bars
from . import labels, span, store

BAR_NS = labels.HOUR_NS // 12
HOUR_BARS = 12
WINDOWS = {"1h": 12, "6h": 72, "24h": 288, "7d": 2016, "28d": 8064}
CARRY_BARS = 12
NEAR_BARS = 6
DAY_NS = 24 * labels.HOUR_NS
FUNDING_CAP_BARS = 16 * HOUR_BARS
LOOKBACK_BARS = WINDOWS["28d"] + 1 + max(CARRY_BARS, FUNDING_CAP_BARS)
LOOKBACK_NS = LOOKBACK_BARS * BAR_NS
MARKET_MIN = 15
TAU_BARS = (72, 288)
TAU_HOURS = (84, 168)
CHUNK = 1 << 22
GROUPS: dict[str, tuple[str, ...]] = {
    "vol": (
        "vol_1h",
        "vol_6h",
        "vol_24h",
        "vol_7d",
        "vol_28d",
        "near_24h",
        "vsurge_1h",
        "volume_24h",
    ),
    "price": (
        "ret_1h",
        "ret_6h",
        "ret_24h",
        "ret_7d",
        "ret_28d",
        "aret_24h",
        "aret_7d",
        "dd_7d",
        "dd_28d",
        "ru_28d",
    ),
    "flow": ("flow_1h", "flow_24h"),
    "classic": ("var_tau_24h", "ac1_tau_24h", "var_tau_7d", "ac1_tau_7d"),
    "lev": ("oi_24h", "oi_7d", "oi_vs_28d", "funding", "funding_rel", "ls", "ls_rel"),
}
OWN = tuple(name for names in GROUPS.values() for name in names)
MARKET = tuple(f"m_{name}" for name in OWN)
SCHEMA = pa.schema(
    [("ts_ns", pa.int64()), ("symbol", pa.string()), *[(n, pa.float64()) for n in OWN + MARKET]]
)
Reduce = Callable[[np.ndarray, np.ndarray], np.ndarray]


def enough(count: np.ndarray, window: int) -> np.ndarray:
    return 6 * count >= 5 * window


def _running(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(x)])


def mean(
    x: np.ndarray, stop: np.ndarray, window: int, observed: np.ndarray | None = None
) -> np.ndarray:
    finite = np.isfinite(x)
    sums, counts = _running(np.where(finite, x, 0.0)), _running(finite)
    seen = counts if observed is None else _running(observed & finite)
    out = np.full(len(stop), np.nan)
    ok = stop >= window
    lo, hi = stop[ok] - window, stop[ok]
    n, present = counts[hi] - counts[lo], seen[hi] - seen[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = np.where(enough(present, window), (sums[hi] - sums[lo]) / n, np.nan)
    return out


def windowed(
    x: np.ndarray,
    stop: np.ndarray,
    window: int,
    reduce: Reduce,
    observed: np.ndarray | None = None,
) -> np.ndarray:
    out = np.full(len(stop), np.nan)
    ok = np.flatnonzero(stop >= window)
    step, offsets = max(1, CHUNK // window), np.arange(window)
    for i in range(0, len(ok), step):
        rows = ok[i : i + step]
        where = (stop[rows] - window)[:, None] + offsets
        block = x[where]
        finite = np.isfinite(block)
        count = finite.sum(axis=1)
        present = count if observed is None else (observed[where] & finite).sum(axis=1)
        good = enough(present, window)
        if good.any():
            out[rows[good]] = reduce(block[good], count[good])
    return out


def largest(block: np.ndarray, count: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(block), block, -np.inf).max(axis=1)


def smallest(block: np.ndarray, count: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(block), block, np.inf).min(axis=1)


def median(block: np.ndarray, count: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.where(np.isfinite(block), block, np.nan), axis=1)
    rows = np.arange(len(ordered))
    return 0.5 * (ordered[rows, (count - 1) // 2] + ordered[rows, count // 2])


def _sorted_sum(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.where(keep, values, np.inf), axis=1)
    return np.where(np.isfinite(ordered), ordered, 0.0).sum(axis=1)


def _spread(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    high = np.where(keep, values, -np.inf).max(axis=1)
    return high > np.where(keep, values, np.inf).min(axis=1)


def _per_window(x: np.ndarray, window: int, stat: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < window:
        return out
    view = sliding_window_view(x, window)
    step = max(1, CHUNK // window)
    for lo in range(0, len(view), step):
        block = view[lo : lo + step]
        out[lo + window - 1 : lo + window - 1 + len(block)] = stat(block)
    return out


def rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    def stat(block: np.ndarray) -> np.ndarray:
        keep = np.isfinite(block)
        n = keep.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = _sorted_sum(block, keep) / n
            var = _sorted_sum((block - m[:, None]) ** 2, keep) / (n - 1)
        return np.where(enough(n, window) & _spread(block, keep), var, np.nan)

    return _per_window(x, window, stat)


def rolling_autocorrelation(x: np.ndarray, window: int) -> np.ndarray:
    def stat(block: np.ndarray) -> np.ndarray:
        a, b = block[:, :-1], block[:, 1:]
        keep = np.isfinite(a) & np.isfinite(b)
        n = keep.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            da = a - (_sorted_sum(a, keep) / n)[:, None]
            db = b - (_sorted_sum(b, keep) / n)[:, None]
            va, vb = _sorted_sum(da * da, keep), _sorted_sum(db * db, keep)
            r = _sorted_sum(da * db, keep) / np.sqrt(va * vb)
        ok = enough(n, window - 1) & _spread(a, keep) & _spread(b, keep) & (va > 0) & (vb > 0)
        return np.where(ok, r, np.nan)

    return _per_window(x, window, stat)


def trailing_tau(v: np.ndarray, stop: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(stop), np.nan)
    ok = np.flatnonzero(stop >= window)
    step, offsets = max(1, CHUNK // window), np.arange(window)
    for i in range(0, len(ok), step):
        rows = ok[i : i + step]
        block = v[(stop[rows] - window)[:, None] + offsets]
        finite = np.isfinite(block)
        count = finite.sum(axis=1)
        full = count == window
        out[rows[full]] = kendall_tau(block[full])
        for m in np.unique(count[~full & enough(count, window)]):
            some = count == m
            out[rows[some]] = kendall_tau(block[some][finite[some]].reshape(-1, m))
    return out


def carried(x: np.ndarray, limit: int = CARRY_BARS) -> np.ndarray:
    good = np.isfinite(x) & (x > 0)
    index = np.arange(len(x))
    last = np.maximum.accumulate(np.where(good, index, -1))
    keep = (last >= 0) & (index - last <= limit)
    return np.where(keep, x[np.maximum(last, 0)], np.nan)


def at(x: np.ndarray, index: np.ndarray) -> np.ndarray:
    inside = (index >= 0) & (index < len(x))
    return np.where(inside, x[np.clip(index, 0, len(x) - 1)], np.nan)


def returns(close: np.ndarray, lag: int = 1) -> np.ndarray:
    r = np.full(len(close), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[lag:] = np.log(close[lag:] / close[:-lag])
    return r


def coin(
    columns: Mapping[str, np.ndarray],
    ratio: np.ndarray,
    stop: np.ndarray,
    sd_ret: np.ndarray,
    b: np.ndarray,
) -> dict[str, np.ndarray]:
    if np.any(stop % HOUR_BARS) or np.any(stop < 1):
        raise ValueError("every hour must start a bar after the first, on the hour")
    close, volume = columns["close"], columns["volume"]
    r = returns(close)
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = sd_ret * sd_ret / NEAR_BARS
        for name, w in WINDOWS.items():
            out[f"vol_{name}"] = np.log(mean(r * r, stop, w) / scale)
        out["near_24h"] = windowed(np.abs(returns(close, NEAR_BARS)), stop, 288, largest) / b
        out["vsurge_1h"] = np.log(mean(volume, stop, 12) / windowed(volume, stop, 8064, median))
        out["volume_24h"] = np.log(mean(volume, stop, 288) / mean(volume, stop, 8064))

        last = close[stop - 1]
        for name, w in WINDOWS.items():
            out[f"ret_{name}"] = np.log(last / at(close, stop - 1 - w))
        out["aret_24h"], out["aret_7d"] = np.abs(out["ret_24h"]), np.abs(out["ret_7d"])
        out["dd_7d"] = np.log(last / windowed(close, stop, 2016, largest))
        out["dd_28d"] = np.log(last / windowed(close, stop, 8064, largest))
        out["ru_28d"] = np.log(last / windowed(close, stop, 8064, smallest))

        taker = np.where(volume == 0, 0.0, columns["taker_buy_share"] * volume)
        present = np.isfinite(close) & np.isfinite(volume) & np.isfinite(taker)
        for name in ("1h", "24h"):
            w = WINDOWS[name]
            bought = mean(np.where(present, taker, np.nan), stop, w)
            out[f"flow_{name}"] = 2 * bought / mean(np.where(present, volume, np.nan), stop, w) - 1

        roll, span_bars = TAU_BARS
        out["var_tau_24h"] = trailing_tau(rolling_variance(r, roll), stop, span_bars)
        out["ac1_tau_24h"] = trailing_tau(rolling_autocorrelation(r, roll), stop, span_bars)
        hourly = np.full(len(close) // HOUR_BARS + 1, np.nan)
        hourly[1:] = close[HOUR_BARS - 1 :: HOUR_BARS][: len(hourly) - 1]
        hr, hour = returns(hourly), stop // HOUR_BARS + 1
        roll, span_hours = TAU_HOURS
        out["var_tau_7d"] = trailing_tau(rolling_variance(hr, roll), hour, span_hours)
        out["ac1_tau_7d"] = trailing_tau(rolling_autocorrelation(hr, roll), hour, span_hours)

        raw = columns["oi"]
        oi = carried(raw)
        now = oi[stop]
        out["oi_24h"] = np.log(now / at(oi, stop - 288))
        out["oi_7d"] = np.log(now / at(oi, stop - 2016))
        seen = np.isfinite(raw) & (raw > 0)
        out["oi_vs_28d"] = np.log(now / windowed(oi, stop + 1, 8064, median, seen))
        funding = columns["funding_rate"]
        out["funding"] = funding[stop - 1]
        hourly = funding[HOUR_BARS - 1 :: HOUR_BARS]
        out["funding_rel"] = funding[stop - 1] - mean(
            hourly, stop // HOUR_BARS, WINDOWS["28d"] // HOUR_BARS
        )
        ls = np.log(carried(ratio))
        out["ls"] = ls[stop - 1]
        out["ls_rel"] = ls[stop - 1] - mean(ls, stop, 8064, np.isfinite(ratio) & (ratio > 0))
    return {name: np.where(np.isfinite(out[name]), out[name], np.nan) for name in OWN}


def market(own: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for name in OWN:
        values = np.stack([o[name] for o in own])
        count = np.isfinite(values).sum(axis=0)
        total = np.where(np.isfinite(values), values, 0.0).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"m_{name}"] = np.where(count >= MARKET_MIN, total / count, np.nan)
    return out


def daily_scales(
    thresholds: pa.Table, symbol: str, hours: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    named = thresholds["symbol"].to_numpy(zero_copy_only=False)
    mine = thresholds.filter(pa.array(named == symbol))
    row = {int(day): i for i, day in enumerate(mine["ts_ns"].to_numpy())}
    if len(row) != mine.num_rows:
        raise ValueError(f"{symbol}: the thresholds table gives a day twice")
    of_hour = hours - hours % DAY_NS
    wanted = np.unique(of_hour)
    absent = [day for day in wanted.tolist() if day not in row]
    if absent:
        raise ValueError(f"{symbol}: the thresholds table has no row for {len(absent)} grid days")
    index = np.array([row[day] for day in wanted.tolist()], np.int64)[
        np.searchsorted(wanted, of_hour)
    ]
    sd_ret, b = (mine[n].to_numpy(zero_copy_only=False).astype(np.float64) for n in ("sd_ret", "b"))
    return sd_ret[index], b[index]


def read_ratio(path: Path) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
            if len(names) != 1:
                raise archive.ArchiveError(f"{path}: holds {len(names)} files, not one CSV")
            data = bundle.read(names[0])
        header = data.startswith(b"create_time")
        table = pacsv.read_csv(
            pa.py_buffer(data),
            read_options=pacsv.ReadOptions(
                column_names=None if header else list(archive._COLUMNS["metrics"])
            ),
            convert_options=pacsv.ConvertOptions(
                column_types={
                    "create_time": pa.timestamp("ms"),
                    "count_long_short_ratio": pa.float64(),
                },
                include_columns=["create_time", "count_long_short_ratio"],
                timestamp_parsers=["%Y-%m-%d %H:%M:%S"],
            ),
        )
    except (zipfile.BadZipFile, zlib.error, EOFError, pa.ArrowInvalid, KeyError) as error:
        raise archive.ArchiveError(f"{path}: {error}") from error
    nulls = table["create_time"].null_count
    if nulls:
        raise archive.ArchiveError(f"{path}: {nulls} rows have no create_time")
    stamps = table["create_time"].to_numpy().astype("datetime64[ms]").astype(np.int64)
    ratio = table["count_long_short_ratio"].to_numpy(zero_copy_only=False).astype(np.float64)
    return {"create_time": stamps, "count_long_short_ratio": ratio}


def ratios(
    root: Path,
    files: Mapping[str, Any],
    symbol: str,
    start: date,
    end: date,
    *,
    limit: date = span.HELD_BACK,
) -> dict[str, np.ndarray]:
    span.refuse(start, end, limit=limit)
    parts = []
    for source in archive.sources([symbol], start, end):
        if source.kind != "metrics":
            continue
        path = root / source.path
        if not path.exists() and files.get(source.path, {}).get("status") == "missing":
            continue
        archive.verify(path)
        parts.append(read_ratio(path))
    if not parts:
        return {"create_time": np.empty(0, np.int64), "count_long_short_ratio": np.empty(0)}
    return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}


def on_grid(found: Mapping[str, np.ndarray], start_ms: int, n: int) -> np.ndarray:
    rows, step, _ = bars._grid(found["create_time"], start_ms, bars.BAR_MS, n)
    out = np.full(n, np.nan)
    out[step] = found["count_long_short_ratio"][rows]
    return out


def labels_run(path: Path, *, limit: date = span.HELD_BACK) -> dict[str, Any]:
    params = io.read_json(path / "manifest.json")["params"]
    days = [date.fromisoformat(params[k]) for k in ("start", "dev_start", "end")]
    span.refuse(*days, limit=limit)
    manifest = io.read_json(path / "manifest.json")
    complete = io.read_json(path / "complete.json")
    grid_file = next(f for f in complete["files"] if f.startswith("grid/"))
    thresholds_file = next(f for f in complete["files"] if f.startswith("thresholds/"))
    return {
        "run_id": manifest["run_id"],
        "manifest_sha256": store.sha256(path / "manifest.json"),
        "params": params,
        "start": days[0],
        "dev_start": days[1],
        "end": days[2],
        "grid_file": grid_file,
        "grid_sha256": complete["files"][grid_file],
        "thresholds_file": thresholds_file,
        "thresholds_sha256": complete["files"][thresholds_file],
        "clean": complete.get("clean") is True,
    }


def last_bar(columns: Mapping[str, np.ndarray]) -> int:
    complete = np.flatnonzero(np.isfinite(columns["close"]))
    return int(complete[-1]) if complete.size else -1


def unfunded(got: archive.Span, symbol: str, start: date, end: date, last_ns: int) -> list[str]:
    monthly = [s for s in archive.sources([symbol], start, end) if s.kind == "fundingRate"]
    return [s.url for s in monthly if s.url in got.missing and s.start_ms * 1_000_000 < last_ns]


def funding_capped(rate: np.ndarray, ts_ns: np.ndarray, calc_ms: np.ndarray) -> np.ndarray:
    settled = np.sort(calc_ms.astype(np.int64)) * 1_000_000
    if not settled.size:
        return np.full(rate.shape, np.nan)
    at = np.searchsorted(settled, ts_ns, side="right") - 1
    age = ts_ns - settled[np.maximum(at, 0)]
    return np.where((at >= 0) & (age <= FUNDING_CAP_BARS * BAR_NS), rate, np.nan)


def run(
    labels_dir: Path,
    root: Path,
    data_dir: Path,
    tag: str,
    log: Callable[[str], None] = lambda line: None,
    *,
    limit: date = span.HELD_BACK,
) -> tuple[Path, dict[str, Any]]:
    span.held_back_root(limit, labels_dir, root, data_dir, Path(data_dir) / "panel-inputs")
    files = span.manifest_files(root, limit=limit)
    source = labels_run(labels_dir, limit=limit)
    if not source["clean"]:
        raise archive.ArchiveError(f"{labels_dir}: complete.json does not mark the run clean")
    symbols = span.symbols_of(source["params"]["symbols"])
    grid_path = labels_dir / source["grid_file"]
    thresholds_path = labels_dir / source["thresholds_file"]
    for path in (grid_path, thresholds_path):
        if not path.resolve().is_relative_to(labels_dir.resolve()):
            raise span.HeldBackError(f"{path} is not inside the labels run {labels_dir}")
    span.held_back_root(limit, grid_path, thresholds_path)
    for path in (grid_path, thresholds_path):
        store.refuse_held_back(path, limit=limit)
    for path, key in ((grid_path, "grid_sha256"), (thresholds_path, "thresholds_sha256")):
        if store.sha256(path) != source[key]:
            raise archive.ArchiveError(f"{path}: its sha256 differs from complete.json's")
    grid = pq.read_table(grid_path, columns=["ts_ns", "symbol"])
    thresholds = pq.read_table(thresholds_path, columns=["ts_ns", "symbol", "sd_ret", "b"])
    start, end = source["start"], source["end"]
    hours = np.unique(grid["ts_ns"].to_numpy())
    if hours.size and hours[-1] >= labels.ns(limit):
        raise span.HeldBackError("the grid holds an hour in the held-back test span")
    start_ns = labels.ns(start)
    if np.any((hours - start_ns) % BAR_NS) or np.any(hours <= start_ns):
        raise ValueError("every grid hour must lie on the bar grid after the span starts")
    stop = (hours - start_ns) // BAR_NS
    days = (end - start).days + 1
    own: list[dict[str, np.ndarray]] = []
    scales = {symbol: daily_scales(thresholds, symbol, hours) for symbol in symbols}
    sources: list[dict[str, Any]] = []
    for symbol in symbols:
        got = archive.collect([symbol], start, end, root)
        columns, _ = bars.bars(
            archive.day_ms(start), days, *(got.columns[(symbol, k)] for k in archive.KINDS)
        )
        last = last_bar(columns)
        last_ns = int(columns["ts_ns"][last]) + BAR_NS if last >= 0 else 0
        absent = unfunded(got, symbol, start, end, last_ns)
        if absent:
            raise archive.ArchiveError(f"{symbol}: funding files missing in the span: {absent}")
        sources += got.sources
        settled = got.columns[(symbol, "fundingRate")]["calc_time"]
        columns["funding_rate"] = funding_capped(columns["funding_rate"], columns["ts_ns"], settled)
        if stop.size and stop[-1] >= len(columns["close"]):
            raise ValueError("the grid holds an hour after the span's last bar")
        found = ratios(root, files, symbol, start, end, limit=limit)
        ratio = on_grid(found, archive.day_ms(start), len(columns["ts_ns"]))
        sd_ret, b = scales[symbol]
        own.append(coin(columns, ratio, stop, sd_ret, b))
        missing = {k: round(float(np.mean(np.isnan(v))), 4) for k, v in own[-1].items()}
        worst = sorted(missing.items(), key=lambda kv: -kv[1])[:3]
        unscaled = int(np.isnan(sd_ret).sum())
        log(f"{symbol:<13} {unscaled} hours without a threshold; most missing {worst}")
    shared = market(own)
    order = sorted(range(len(symbols)), key=lambda i: symbols[i])
    names = [symbols[i] for i in order]
    count = len(hours)
    table = {
        "ts_ns": np.repeat(hours, len(names)),
        "symbol": np.tile(np.array(names, dtype=object), count),
        **{n: np.stack([own[i][n] for i in order], axis=1).ravel() for n in OWN},
        **{n: np.repeat(shared[n], len(names)) for n in MARKET},
    }
    out = pa.table(table, schema=SCHEMA)
    if not (out["ts_ns"].equals(grid["ts_ns"]) and out["symbol"].equals(grid["symbol"])):
        raise RuntimeError("the inputs' rows are not the grid's rows in the grid's order")
    checks = {
        "rows": out.num_rows,
        "missing_share": {n: float(np.mean(np.isnan(table[n]))) for n in OWN + MARKET},
    }
    run_params = {
        "labels_run": source["run_id"],
        "labels_manifest_sha256": source["manifest_sha256"],
        "labels_grid": {"file": source["grid_file"], "sha256": source["grid_sha256"]},
        "labels_thresholds": {
            "file": source["thresholds_file"],
            "sha256": source["thresholds_sha256"],
        },
        "symbols": symbols,
        "start": start.isoformat(),
        "dev_start": source["dev_start"].isoformat(),
        "end": end.isoformat(),
        "windows_bars": WINDOWS,
        "carry_bars": CARRY_BARS,
        "market_min_coins": MARKET_MIN,
        "groups": {k: list(v) for k, v in GROUPS.items()},
        "code": {"uv_lock_sha256": labels.uv_lock()},
        "sources": sources,
        **({"limit": limit.isoformat()} if limit != span.HELD_BACK else {}),
    }
    tables = {"inputs": out}
    return store.write(data_dir, "panel-inputs", tag, tables, run_params, checks), checks
