# A panel run in the shared layout: <data dir>/<domain>/<run_id>/ with manifest.json first, one
# Parquet file per table, any text files, then complete.json with every file's sha256 and checks.

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .. import __version__, engine
from ..ising import io
from . import labels, span


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(
    data_dir: Path,
    domain: str,
    tag: str,
    tables: dict[str, pa.Table],
    params: dict[str, Any],
    checks: dict[str, Any],
    texts: dict[str, str] | None = None,
) -> Path:
    run_id = engine.run_id(tag)
    run_dir = Path(data_dir) / domain / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "format": 1,
        "domain": domain,
        "run_id": run_id,
        "producer": {"name": "tipping_point.panel", "version": __version__, **io.code_version()},
        "started_ns": time.time_ns(),
        "tables": list(tables),
        "params": params,
        "licence": "CC BY-NC-SA 4.0",
        "attribution": "Binance Vision",
    }
    io.write_json(run_dir / "manifest.json", manifest)
    files: dict[str, str] = {}
    for name, table in tables.items():
        path = run_dir / f"{name}.parquet"
        io.write_file(path, lambda f, table=table: pq.write_table(table, f, compression="zstd"))
        files[path.name] = sha256(path)
    for name, text in (texts or {}).items():
        path = run_dir / name
        io.write_file(path, lambda f, text=text: f.write(text.encode("utf-8")))
        files[path.name] = sha256(path)
    record = {
        "ended_ns": time.time_ns(),
        "rows": {name: table.num_rows for name, table in tables.items()},
        "clean": True,
        "files": files,
        "checks": checks,
    }
    io.write_json(run_dir / "complete.json", record)
    return run_dir


UNIT_NS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


def refuse_held_back(path: Path, column: str = "ts_ns") -> None:
    file = pq.ParquetFile(path)
    schema = file.schema_arrow
    if column not in schema.names:
        raise span.HeldBackError(f"{path}: it has no {column} column")
    kind = schema.field(column).type
    if not (pa.types.is_integer(kind) or pa.types.is_timestamp(kind)):
        raise span.HeldBackError(f"{path}: its {column} column is {kind}, not integer or timestamp")
    meta = file.metadata
    index = meta.schema.names.index(column)
    held = labels.ns(span.HELD_BACK)
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        if group.num_rows == 0:
            continue
        stats = group.column(index).statistics
        if stats is None or not stats.has_min_max:
            raise span.HeldBackError(f"{path}: its metadata gives no bounds for {column}")
        if pa.types.is_timestamp(kind):
            top = stats.max_raw * UNIT_NS[kind.unit] if isinstance(stats.max_raw, int) else None
        else:
            top = stats.max if isinstance(stats.max, int) else None
        if top is None:
            raise span.HeldBackError(f"{path}: its metadata gives no bounds for {column}")
        if top >= held:
            raise span.HeldBackError(f"{path}: {column} reaches the held-back test span")


def read(run_dir: Path, name: str) -> pa.Table:
    return pq.read_table(run_dir / f"{name}.parquet")
