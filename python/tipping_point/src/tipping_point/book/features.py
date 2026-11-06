"""One day's book samples reduced to 1-second features (#30 M2; research pass section 2.2).

Input: an engine run's `book_samples` Parquet, the book on the absolute 100 ms grid with 20 levels
a side (`bid_px_i`, `bid_qty_i`, `ask_px_i`, `ask_qty_i`, best first, integer ticks and lots at the
run's scales; engine/src/store/samples.rs), and the run's manifest.json beside it (symbol, scales,
the day). Output: one row per UTC second of the day, 86,400 rows, second s covering the samples at
s + 0.0, 0.1, ..., 0.9 s.

Causality. Every feature of second s is computed from the samples in [s, s + 1) and nothing else:
no carry from the second before, no statistic over the day. Returns and order flow therefore use
the 9 steps between the second's 10 samples; the step from s - 0.1 to s is in neither second.
Each minute frame of the JEPA then reads only its own minute's samples.

Gaps. A sample counts only when the book was synced, both sides have a best level and the book is
not crossed. `samples` is the count of those (0 to 10) and `gap` is set when it is below 10. A
second with some samples missing is masked by `gap` and its features use only the samples present
(steps only between two present neighbours); nothing is filled in. A second with none is NaN.

Features, per second (a "sample" is one 100 ms book; means are over the second's samples):

  ret        sum of the 100 ms log mid returns between the second's samples
  rv100      sum of their squares
  moves      count of those steps where the mid moved
  micro_dev  mean of (Stoikov micro-price - mid) in ticks; micro = (bid ask_qty + ask bid_qty) /
             (bid_qty + ask_qty) at the best level
  imb_k      mean of (bid qty - ask qty) / (bid qty + ask qty) over the top k = 1, 5, 20 levels
  ofi        Cont-Kukanov-Stoikov order-flow imbalance at the best level, summed over the steps,
             divided by the mean top-5 quantity of both sides; positive is buying pressure (the bid
             grows or rises, the ask shrinks or falls)
  depth_5,   mean of log(total notional, both sides, top 5 and top 20 levels), in quote currency
  depth_20
  depth_asym mean of log(bid notional / ask notional), top 20 levels
  slope_b,   mean distance from the mid, in bps, of the level where the side's cumulative notional
  slope_a    first reaches half of its top-20 notional: how far from the touch the side's
             liquidity sits. (The research pass named "a fixed notional"; a fixed amount needs a
             cross-day calibration per asset and is undefined whenever 20 levels hold less, so
             the half-depth distance stands in for it.)
  wide       fraction of samples with a spread above one tick

Cross-day steps, not done here (they are fitted later, on the training span only, #30 4.1):
`depth_5` and `depth_20` become relative to their trailing 7-day median, and every feature gets
the robust standardisation (median and IQR) with its log or signed-log transform. Every other
feature is defined within the day and is final here.

shortcut: a tick is one unit of the run's price scale (10^-price_decimals), which the engine sets
from the finest price in the file; that matched the tick on the days checked (ETHUSDT 2023-06-10:
2 decimals, a 0.01 tick; the synthetic BTCUSDT day: 1 decimal, a 0.1 tick). A day whose scale
differs would shift `micro_dev` and `wide`; the scales are kept in each file's metadata so the
cross-day step can check them and refuse such a day.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STEP_NS = 100_000_000  # the engine's book_samples grid
SECOND_NS = 1_000_000_000
PER = SECOND_NS // STEP_NS  # samples per second
DAY = 86_400
LEVELS = 20
BATCH = 1 << 17  # samples reduced at a time (about 85 MB of levels as float64)
FEATURES = (
    "ret",
    "rv100",
    "moves",
    "micro_dev",
    "imb_1",
    "imb_5",
    "imb_20",
    "ofi",
    "depth_5",
    "depth_20",
    "depth_asym",
    "slope_b",
    "slope_a",
    "wide",
)
FORMAT = 1
# Each sample's quantities that the features are made from (_per_sample).
_SAMPLED = (
    "mid",
    "bp0",
    "bq0",
    "ap0",
    "aq0",
    "micro_dev",
    "top5",
    "imb_1",
    "imb_5",
    "imb_20",
    "depth_5",
    "depth_20",
    "depth_asym",
    "slope_b",
    "slope_a",
    "wide",
)
# Floats gain nothing from dictionaries; the timestamps compress to almost nothing as deltas.
_WRITE = {
    "compression": "zstd",
    "use_dictionary": ["samples", "gap", "moves", "wide"],
    "column_encoding": {"second_ts_ns": "DELTA_BINARY_PACKED"},
}


class FeatureError(ValueError):
    """Samples that cannot be reduced: off the grid, duplicated, or from a run of another shape."""


def reduce(
    samples: pa.Table | Iterable[pa.RecordBatch],
    day_start_ns: int,
    price_decimals: int,
    qty_decimals: int,
) -> pa.Table:
    """The day's 1-second feature table (see the module notes) from its `book_samples` rows, as a
    table or as record batches in any order. Batches keep the memory to one batch's levels."""
    if day_start_ns % (DAY * SECOND_NS):
        raise FeatureError(f"{day_start_ns} is not the start of a UTC day")
    table = isinstance(samples, pa.Table)
    batches = cast(pa.Table, samples).to_batches(BATCH) if table else samples
    unit = 10.0 ** -(price_decimals + qty_decimals)  # quote currency per (tick x lot)
    flat = {k: np.full(DAY * PER, np.nan) for k in _SAMPLED}
    taken = np.zeros(DAY * PER, dtype=bool)
    for batch in batches:
        offset = batch.column("sample_ts_ns").to_numpy() - day_start_ns
        inside = (offset >= 0) & (offset < DAY * SECOND_NS)
        if np.any(offset[inside] % STEP_NS):
            raise FeatureError("a sample is not on the 100 ms grid")
        slot = offset[inside] // STEP_NS
        if taken[slot].any() or np.unique(slot).size != slot.size:
            raise FeatureError("two samples share a grid instant")
        taken[slot] = True

        def side(prefix: str, batch: pa.RecordBatch = batch, inside: np.ndarray = inside) -> Any:
            cols = [batch.column(f"{prefix}_{i}").cast(pa.float64()) for i in range(LEVELS)]
            return np.column_stack([c.to_numpy(zero_copy_only=False) for c in cols])[inside]

        bp, bq, ap, aq = side("bid_px"), side("bid_qty"), side("ask_px"), side("ask_qty")
        synced = batch.column("synced").to_numpy(zero_copy_only=False)[inside]
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = synced & (bp[:, 0] < ap[:, 0]) & (bq[:, 0] > 0) & (aq[:, 0] > 0)
            values = _per_sample(bp, bq, ap, aq, unit)
        for k in _SAMPLED:
            flat[k][slot[ok]] = values[k][ok]
    grid = {k: v.reshape(DAY, PER) for k, v in flat.items()}  # (second, 100 ms step)
    count = np.isfinite(grid["mid"]).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        step = np.diff(np.log(grid["mid"]), axis=1)  # NaN unless both neighbours are present
        flow = _ofi(grid["bp0"], grid["bq0"], grid["ap0"], grid["aq0"])
        stepped = np.isfinite(step)
        out = {
            "ret": _sum(step),
            "rv100": _sum(step * step),
            "moves": np.where(stepped.any(axis=1), (stepped & (step != 0)).sum(axis=1), np.nan),
            "micro_dev": _mean(grid["micro_dev"]),
            "imb_1": _mean(grid["imb_1"]),
            "imb_5": _mean(grid["imb_5"]),
            "imb_20": _mean(grid["imb_20"]),
            "ofi": _sum(flow) / _mean(grid["top5"]),
            "depth_5": _mean(grid["depth_5"]),
            "depth_20": _mean(grid["depth_20"]),
            "depth_asym": _mean(grid["depth_asym"]),
            "slope_b": _mean(grid["slope_b"]),
            "slope_a": _mean(grid["slope_a"]),
            "wide": _mean(grid["wide"]),
        }
    columns: dict[str, Any] = {
        "second_ts_ns": day_start_ns + np.arange(DAY, dtype=np.int64) * SECOND_NS,
        "samples": count.astype(np.uint8),
        "gap": count < PER,
    }
    columns |= {k: out[k].astype(np.float32) for k in FEATURES}
    return pa.table(columns)


def _per_sample(
    bp: np.ndarray, bq: np.ndarray, ap: np.ndarray, aq: np.ndarray, notional_unit: float
) -> dict[str, np.ndarray]:
    """Each sample's book quantities; levels past the book's depth (null) count as empty."""
    bq0, aq0, bp0, ap0 = bq[:, 0], aq[:, 0], bp[:, 0], ap[:, 0]
    mid = (bp0 + ap0) / 2
    bqz, aqz = np.nan_to_num(bq), np.nan_to_num(aq)
    bn = np.nan_to_num(bp * bq) * notional_unit  # notional per level, quote currency
    an = np.nan_to_num(ap * aq) * notional_unit
    out = {
        "mid": mid,
        "bp0": bp0,
        "bq0": bq0,
        "ap0": ap0,
        "aq0": aq0,
        "micro_dev": (bp0 * aq0 + ap0 * bq0) / (bq0 + aq0) - mid,  # a tick is one price unit
        "top5": bqz[:, :5].sum(axis=1) + aqz[:, :5].sum(axis=1),
        "depth_5": np.log(bn[:, :5].sum(axis=1) + an[:, :5].sum(axis=1)),
        "depth_20": np.log(bn.sum(axis=1) + an.sum(axis=1)),
        "depth_asym": np.log(bn.sum(axis=1) / an.sum(axis=1)),
        "slope_b": (mid - _half_depth_price(bp, bn)) / mid * 1e4,
        "slope_a": (_half_depth_price(ap, an) - mid) / mid * 1e4,
        "wide": (ap0 - bp0 > 1).astype(np.float64),
    }
    for k in (1, 5, 20):
        b, a = bqz[:, :k].sum(axis=1), aqz[:, :k].sum(axis=1)
        out[f"imb_{k}"] = (b - a) / (b + a)
    return out


def _half_depth_price(price: np.ndarray, notional: np.ndarray) -> np.ndarray:
    """The price of the first level at which the side's cumulative notional reaches half of its
    top-20 total."""
    cum = np.cumsum(notional, axis=1)
    level = np.argmax(cum >= cum[:, -1:] / 2, axis=1)
    return np.take_along_axis(price, level[:, None], axis=1)[:, 0]


def _ofi(bp: np.ndarray, bq: np.ndarray, ap: np.ndarray, aq: np.ndarray) -> np.ndarray:
    """Cont, Kukanov and Stoikov (2014) e_n for each step between neighbouring samples at the best
    level: e = 1{Pb' >= Pb} qb' - 1{Pb' <= Pb} qb - 1{Pa' <= Pa} qa' + 1{Pa' >= Pa} qa, primes
    for the later sample. NaN where either sample is missing."""
    b0, b1, qb0, qb1 = bp[:, :-1], bp[:, 1:], bq[:, :-1], bq[:, 1:]
    a0, a1, qa0, qa1 = ap[:, :-1], ap[:, 1:], aq[:, :-1], aq[:, 1:]
    e = (
        np.where(b1 >= b0, qb1, 0.0)
        - np.where(b1 <= b0, qb0, 0.0)
        - np.where(a1 <= a0, qa1, 0.0)
        + np.where(a1 >= a0, qa0, 0.0)
    )
    present = np.isfinite(b0) & np.isfinite(b1)
    return np.where(present, e, np.nan)


def _sum(x: np.ndarray) -> np.ndarray:
    """Row sums of the present values; NaN for a row with none."""
    present = np.isfinite(x)
    return np.where(present.any(axis=1), np.where(present, x, 0.0).sum(axis=1), np.nan)


def _mean(x: np.ndarray) -> np.ndarray:
    present = np.isfinite(x)
    n = present.sum(axis=1)
    return np.where(n > 0, np.where(present, x, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)


def range_dir(samples: Path) -> Path:
    """The range folder a `market book` run keeps its manifest in, from the samples' path:
    <range>/<staging|samples>/market/<run>/book_samples/<file>."""
    parents = samples.resolve().parents
    if parents[2].name != "market" or parents[3].name not in ("staging", "samples"):
        raise FeatureError(f"{samples} is not in a `market book` range; name the folder with --out")
    return parents[4]


def reduce_run(samples: Path, out: Path) -> dict[str, Any]:
    """Reduces the run whose `book_samples` Parquet is `samples` into
    `<out>/features/<SYMBOL>/<day>.parquet` and returns its record for the range manifest."""
    manifest = json.loads((samples.parents[1] / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["params"]
    if params["book_sample"] != {"levels": LEVELS, "every_ns": STEP_NS}:
        raise FeatureError(f"the run sampled the book as {params['book_sample']}")
    if len(params["symbols"]) != 1:
        raise FeatureError(f"the run holds {params['symbols']}, not one symbol")
    symbol = params["symbols"][0]
    scales = params["scales"][symbol]
    day_start = int(manifest["started_ns"])
    day = datetime.fromtimestamp(day_start // SECOND_NS, UTC).date()
    batches = pq.ParquetFile(samples).iter_batches(BATCH)
    table = reduce(batches, day_start, scales["price_decimals"], scales["qty_decimals"])
    meta = {
        "format": FORMAT,
        "symbol": symbol,
        "day": day.isoformat(),
        "run_id": manifest["run_id"],
        "scales": scales,
        "features": list(FEATURES),
    }
    table = table.replace_schema_metadata({"tp.book_features": json.dumps(meta, sort_keys=True)})
    path = out / "features" / symbol / f"{day.isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    pq.write_table(table, part, **_WRITE)
    part.replace(path)
    gap = table["gap"].to_numpy(zero_copy_only=False)
    return {
        "path": path.relative_to(out).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "rows": table.num_rows,
        "gap_seconds": int(gap.sum()),
        "empty_seconds": int((table["samples"].to_numpy() == 0).sum()),
    }
