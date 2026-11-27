# A panel run in the shared layout: <data dir>/<domain>/<run_id>/ with manifest.json first, one
# Parquet file per table, any text and binary files, then complete.json with every sha256 and check.

from __future__ import annotations

import hashlib
import time
from datetime import date
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
    blobs: dict[str, bytes] | None = None,
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
    for name, data in (blobs or {}).items():
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        io.write_file(path, lambda f, data=data: f.write(data))
        files[name] = sha256(path)
    record = {
        "ended_ns": time.time_ns(),
        "rows": {name: table.num_rows for name, table in tables.items()},
        "clean": True,
        "manifest_sha256": sha256(run_dir / "manifest.json"),
        "files": files,
        "checks": checks,
    }
    io.write_json(run_dir / "complete.json", record)
    return run_dir


UNIT_NS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


def refuse_held_back(
    path: Path, column: str = "ts_ns", data: bytes | None = None, *, limit: date = span.HELD_BACK
) -> None:
    file = pq.ParquetFile(path if data is None else pa.BufferReader(data))
    schema = file.schema_arrow
    if column not in schema.names:
        raise span.HeldBackError(f"{path}: it has no {column} column")
    kind = schema.field(column).type
    if not (pa.types.is_integer(kind) or pa.types.is_timestamp(kind)):
        raise span.HeldBackError(f"{path}: its {column} column is {kind}, not integer or timestamp")
    meta = file.metadata
    index = [meta.schema.column(i).path for i in range(meta.num_columns)].index(column)
    stamped = pa.types.is_timestamp(kind)
    scale = UNIT_NS[kind.unit] if stamped else 1
    first, held = labels.ns(span.START), labels.ns(limit)
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        if group.num_rows == 0:
            continue
        stats = group.column(index).statistics
        if stats is None or not stats.has_min_max:
            raise span.HeldBackError(f"{path}: its metadata gives no bounds for {column}")
        low, top = (stats.min_raw, stats.max_raw) if stamped else (stats.min, stats.max)
        if not (isinstance(low, int) and isinstance(top, int)):
            raise span.HeldBackError(f"{path}: its metadata gives no bounds for {column}")
        if top * scale >= held:
            raise span.HeldBackError(f"{path}: {column} reaches the held-back test span")
        if low * scale < first:
            raise span.HeldBackError(
                f"{path}: {column} starts before {span.START}, so it is not nanoseconds of the "
                f"study's span"
            )


def read(run_dir: Path, name: str) -> pa.Table:
    return pq.read_table(run_dir / f"{name}.parquet")
