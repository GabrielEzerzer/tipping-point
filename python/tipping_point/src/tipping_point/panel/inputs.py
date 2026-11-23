# Model inputs per coin and UTC hour t of a labels grid (docs/market.md, "Coin panel"). Timing rule:
# an input at t reads only 5-minute bars that start before t and metrics or funding rows stamped at
# or before t. tests/panel/test_inputs.py changes every later value and checks each row stays put.

from __future__ import annotations

import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from ..ews.trend import kendall_tau
from ..ising import io
from ..market import archive, bars
from . import labels, span, store

BAR_NS = labels.HOUR_NS // 12
HOUR_BARS = 12
WINDOWS = {"1h": 12, "6h": 72, "24h": 288, "7d": 2016, "28d": 8064}
CARRY_BARS = 12
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
CONSTANTS = pa.schema(
    [("ts_ns", pa.int64()), ("symbol", pa.string()), ("mean_r2", pa.float64()), ("b", pa.float64())]
)
Reduce = Callable[[np.ndarray, np.ndarray], np.ndarray]


def enough(count: np.ndarray, window: int) -> np.ndarray:
    return 6 * count >= 5 * window


def mean(x: np.ndarray, stop: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(x)
    sums = np.concatenate([[0.0], np.cumsum(np.where(finite, x, 0.0))])
    counts = np.concatenate([[0], np.cumsum(finite)])
    out = np.full(len(stop), np.nan)
    ok = stop >= window
    lo, hi = stop[ok] - window, stop[ok]
    n = counts[hi] - counts[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = np.where(enough(n, window), (sums[hi] - sums[lo]) / n, np.nan)
    return out


def windowed(x: np.ndarray, stop: np.ndarray, window: int, reduce: Reduce) -> np.ndarray:
    out = np.full(len(stop), np.nan)
    ok = np.flatnonzero(stop >= window)
    step, offsets = max(1, CHUNK // window), np.arange(window)
    for i in range(0, len(ok), step):
        rows = ok[i : i + step]
        block = x[(stop[rows] - window)[:, None] + offsets]
        count = np.isfinite(block).sum(axis=1)
        good = enough(count, window)
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


def _trailing_sums(window: int, *columns: np.ndarray) -> list[np.ndarray]:
    out = []
    for column in columns:
        c = np.concatenate([[0.0], np.cumsum(column)])
        sums = np.full(len(column), np.nan)
        sums[window - 1 :] = c[window:] - c[:-window]
        out.append(sums)
    return out


def rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(x)
    y = np.where(valid, x - np.mean(x[valid]) if valid.any() else 0.0, 0.0)
    n, s1, s2 = _trailing_sums(window, valid.astype(np.float64), y, y * y)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1)
    return np.where(enough(n, window), np.maximum(var, 0.0), np.nan)


def rolling_autocorrelation(x: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(x)
    y = np.where(valid, x - np.mean(x[valid]) if valid.any() else 0.0, 0.0)
    pair = np.zeros(len(x), bool)
    pair[1:] = valid[1:] & valid[:-1]
    a, b = np.zeros(len(x)), np.zeros(len(x))
    a[1:], b[1:] = np.where(pair[1:], y[:-1], 0.0), np.where(pair[1:], y[1:], 0.0)
    pairs = window - 1
    n, sa, sb, saa, sbb, sab = _trailing_sums(
        pairs, pair.astype(np.float64), a, b, a * a, b * b, a * b
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        va, vb = saa - sa * sa / n, sbb - sb * sb / n
        r = (sab - sa * sb / n) / np.sqrt(va * vb)
    r[: window - 1] = np.nan
    return np.where(enough(n, pairs) & (va > 0) & (vb > 0), r, np.nan)


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
    mean_r2: float,
    b: float,
) -> dict[str, np.ndarray]:
    if np.any(stop % HOUR_BARS) or np.any(stop < 1):
        raise ValueError("every hour must start a bar after the first, on the hour")
    close, volume = columns["close"], columns["volume"]
    r = returns(close)
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, w in WINDOWS.items():
            out[f"vol_{name}"] = np.log(mean(r * r, stop, w) / mean_r2)
        out["near_24h"] = windowed(np.abs(returns(close, 6)), stop, 288, largest) / b
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

        oi = carried(columns["oi"])
        now = oi[stop]
        out["oi_24h"] = np.log(now / at(oi, stop - 288))
        out["oi_7d"] = np.log(now / at(oi, stop - 2016))
        out["oi_vs_28d"] = np.log(now / windowed(oi, stop + 1, 8064, median))
        funding = columns["funding_rate"]
        out["funding"] = funding[stop]
        out["funding_rel"] = funding[stop] - mean(funding, stop + 1, 8064)
        ls = np.log(carried(ratio))
        out["ls"] = ls[stop]
        out["ls_rel"] = ls[stop] - mean(ls, stop + 1, 8064)
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


def mean_r2(columns: Mapping[str, np.ndarray], first_ns: int, end_ns: int) -> float:
    r = returns(columns["close"])
    inside = (columns["ts_ns"] >= first_ns) & (columns["ts_ns"] < end_ns) & np.isfinite(r)
    return float(np.mean(r[inside] ** 2))


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
    except (zipfile.BadZipFile, pa.ArrowInvalid, KeyError) as error:
        raise archive.ArchiveError(f"{path}: {error}") from error
    stamps = table["create_time"].to_numpy().astype("datetime64[ms]").astype(np.int64)
    ratio = table["count_long_short_ratio"].to_numpy(zero_copy_only=False).astype(np.float64)
    return {"create_time": stamps, "count_long_short_ratio": ratio}


def ratios(root: Path, symbol: str, start: date, end: date) -> dict[str, np.ndarray]:
    span.refuse(start, end)
    manifest_path = root / "manifest.json"
    files = io.read_json(manifest_path)["files"] if manifest_path.exists() else {}
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


def labels_run(path: Path) -> dict[str, Any]:
    params = io.read_json(path / "manifest.json")["params"]
    days = [date.fromisoformat(params[k]) for k in ("start", "dev_start", "end")]
    span.refuse(*days)
    manifest = io.read_json(path / "manifest.json")
    complete = io.read_json(path / "complete.json")
    grid_file = next(f for f in complete["files"] if f.startswith("grid/"))
    return {
        "run_id": manifest["run_id"],
        "params": params,
        "start": days[0],
        "dev_start": days[1],
        "end": days[2],
        "grid_file": grid_file,
        "grid_sha256": complete["files"][grid_file],
        "thresholds_file": next(f for f in complete["files"] if f.startswith("thresholds/")),
    }


def run(
    labels_dir: Path,
    root: Path,
    data_dir: Path,
    tag: str,
    log: Callable[[str], None] = lambda line: None,
) -> tuple[Path, dict[str, Any]]:
    source = labels_run(labels_dir)
    grid_path = labels_dir / source["grid_file"]
    if store.sha256(grid_path) != source["grid_sha256"]:
        raise archive.ArchiveError(f"{grid_path}: its sha256 differs from complete.json's")
    grid = pq.read_table(grid_path, columns=["ts_ns", "symbol"])
    thresholds = pq.read_table(labels_dir / source["thresholds_file"]).to_pylist()
    b = {row["symbol"]: row["b"] for row in thresholds}
    symbols = list(source["params"]["symbols"])
    start, end = source["start"], source["end"]
    first_ns, end_ns = labels.ns(source["dev_start"]), labels.ns(end + timedelta(days=1))
    hours = np.unique(grid["ts_ns"].to_numpy())
    if hours.size and hours[-1] >= labels.ns(span.HELD_BACK):
        raise span.HeldBackError("the grid holds an hour in the held-back test span")
    start_ns = labels.ns(start)
    stop = (hours - start_ns) // BAR_NS
    days = (end - start).days + 1
    own: list[dict[str, np.ndarray]] = []
    constants: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for symbol in symbols:
        got = archive.collect([symbol], start, end, root)
        sources += got.sources
        columns, _ = bars.bars(
            archive.day_ms(start), days, *(got.columns[(symbol, k)] for k in archive.KINDS)
        )
        ratio = on_grid(
            ratios(root, symbol, start, end), archive.day_ms(start), len(columns["ts_ns"])
        )
        m = mean_r2(columns, first_ns, end_ns)
        own.append(coin(columns, ratio, stop, m, b[symbol]))
        constants.append({"ts_ns": first_ns, "symbol": symbol, "mean_r2": m, "b": b[symbol]})
        missing = {k: round(float(np.mean(np.isnan(v))), 4) for k, v in own[-1].items()}
        worst = sorted(missing.items(), key=lambda kv: -kv[1])[:3]
        log(f"{symbol:<13} mean r^2 {m:.3e} b {b[symbol]:.4f}; most missing {worst}")
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
        "labels_grid": {"file": source["grid_file"], "sha256": source["grid_sha256"]},
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
    }
    tables = {"inputs": out, "constants": pa.Table.from_pylist(constants, schema=CONSTANTS)}
    return store.write(data_dir, "panel-inputs", tag, tables, run_params, checks), checks
