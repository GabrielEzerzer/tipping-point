# The prefix check: a whole-span labels run and inputs run against the pinned development runs.
# Every threshold and inputs row before HELD_BACK must be equal bit for bit, and every grid row too,
# except the labels of each horizon's last hours of 2023, which only 2024 data can settle.

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..ising import io
from . import labels, span, store

HELD_NS = labels.ns(span.HELD_BACK)


class PrefixError(ValueError):
    pass


def window_ns(horizon: int) -> int:
    return HELD_NS - horizon * labels.HOUR_NS - labels.LEAD_NS


def settled_late(ts: np.ndarray, column: str) -> np.ndarray:
    for h in labels.HORIZONS:
        if column in (f"long_{h}", f"short_{h}", f"censored_{h}"):
            return ts > window_ns(h)
    return np.zeros(len(ts), bool)


def _records(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, complete = io.read_json(run / "manifest.json"), io.read_json(run / "complete.json")
    if complete.get("clean") is not True:
        raise PrefixError(f"{run}: complete.json does not mark the run clean")
    return manifest, complete


def _file(run: Path, complete: dict[str, Any], prefix: str) -> tuple[Path, str]:
    names = [name for name in complete["files"] if name.startswith(prefix)]
    if len(names) != 1:
        raise PrefixError(f"{run}: complete.json lists {len(names)} files for {prefix}, not one")
    path, digest = run / names[0], complete["files"][names[0]]
    if store.sha256(path) != digest:
        raise PrefixError(f"{path}: its sha256 differs from complete.json's")
    return path, digest


def _bits(values: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    nulls = values.is_null().to_numpy(zero_copy_only=False)
    data = values.to_numpy(zero_copy_only=False)
    if pa.types.is_floating(values.type):
        data = np.where(nulls, 0.0, data).astype(np.float64).view(np.uint64)
    return nulls, data


def _compare(
    what: str, whole: Path, dev: Path, columns: Sequence[str], allowed: bool = False
) -> int:
    left, right = pq.ParquetFile(whole), pq.ParquetFile(dev)
    if left.schema_arrow != right.schema_arrow:
        raise PrefixError(f"{what}: the whole-span table's schema is not the development table's")
    whole_ts = left.read(columns=["ts_ns"])["ts_ns"].to_numpy()
    dev_ts = right.read(columns=["ts_ns"])["ts_ns"].to_numpy()
    n = int(np.searchsorted(whole_ts, HELD_NS))
    if np.any(np.diff(whole_ts) < 0) or n != len(dev_ts) or (len(dev_ts) and dev_ts[-1] >= HELD_NS):
        raise PrefixError(
            f"{what}: {n} whole-span rows lie before {span.HELD_BACK}, the development table has "
            f"{len(dev_ts)}"
        )
    for column in ["ts_ns", "symbol", *columns]:
        a = left.read(columns=[column])[column].slice(0, n)
        b = right.read(columns=[column])[column]
        (a_nulls, a_data), (b_nulls, b_data) = _bits(a), _bits(b)
        differ = (a_nulls != b_nulls) | (a_data != b_data)
        if allowed:
            differ &= ~settled_late(dev_ts, column)
        if differ.any():
            i = int(np.flatnonzero(differ)[0])
            raise PrefixError(
                f"{what}: {column} differs at row {i} (ts_ns {int(dev_ts[i])}), and "
                f"{int(differ.sum())} rows in all, outside the window only 2024 data can settle"
            )
    return n


def check(labels_run: Path, inputs_run: Path, dev_labels: Path, dev_inputs: Path) -> dict[str, Any]:
    (whole_lm, whole_lc), (dev_lm, dev_lc) = _records(labels_run), _records(dev_labels)
    (whole_im, whole_ic), (dev_im, dev_ic) = _records(inputs_run), _records(dev_inputs)
    whole_p, dev_p = whole_lm["params"], dev_lm["params"]
    for key in ("start", "dev_start", "symbols"):
        if whole_p[key] != dev_p[key]:
            raise PrefixError(f"the labels runs differ in {key}: {whole_p[key]} and {dev_p[key]}")
    if date.fromisoformat(dev_p["end"]) != span.HELD_BACK - timedelta(days=1):
        raise PrefixError(f"{dev_labels}: the development grid must end on {span.END}")
    if date.fromisoformat(whole_p["end"]) < span.HELD_BACK:
        raise PrefixError(f"{labels_run}: the whole-span grid ends before {span.HELD_BACK}")
    whole_grid, whole_sha = _file(labels_run, whole_lc, "grid/")
    dev_grid, dev_sha = _file(dev_labels, dev_lc, "grid/")
    for run, manifest, digest, source in (
        (inputs_run, whole_im, whole_sha, labels_run),
        (dev_inputs, dev_im, dev_sha, dev_labels),
    ):
        if manifest["params"]["labels_grid"]["sha256"] != digest:
            raise PrefixError(f"{run}: it was not made from the grid of {source}")
    whole_thresholds, whole_t_sha = _file(labels_run, whole_lc, "thresholds/")
    dev_thresholds, dev_t_sha = _file(dev_labels, dev_lc, "thresholds/")
    for run, manifest, digest, source in (
        (inputs_run, whole_im, whole_t_sha, labels_run),
        (dev_inputs, dev_im, dev_t_sha, dev_labels),
    ):
        if manifest["params"]["labels_thresholds"]["sha256"] != digest:
            raise PrefixError(f"{run}: it was not made from the thresholds of {source}")
    recorded = whole_im["params"].get("labels_manifest_sha256")
    if recorded != store.sha256(labels_run / "manifest.json"):
        raise PrefixError(f"{inputs_run}: it does not record the manifest of {labels_run}")
    thresholds = _compare(
        "thresholds",
        whole_thresholds,
        dev_thresholds,
        [n for n in labels.THRESHOLDS.names if n not in ("ts_ns", "symbol")],
    )
    grid_names = [n for n in labels.GRID.names if n not in ("ts_ns", "symbol")]
    grid_rows = _compare("grid", whole_grid, dev_grid, grid_names, allowed=True)
    whole_inputs = _file(inputs_run, whole_ic, "inputs.parquet")[0]
    dev_table = _file(dev_inputs, dev_ic, "inputs.parquet")[0]
    names = [
        n for n in pq.ParquetFile(dev_table).schema_arrow.names if n not in ("ts_ns", "symbol")
    ]
    inputs_rows = _compare("inputs", whole_inputs, dev_table, names)
    return {
        "rows": {"thresholds": thresholds, "grid": grid_rows, "inputs": inputs_rows},
        "window_after_ns": {f"{h}h": window_ns(h) for h in labels.HORIZONS},
    }
