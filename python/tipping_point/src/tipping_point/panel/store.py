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


def refuse_held_back(path: Path) -> None:
    meta = pq.ParquetFile(path).metadata
    index = meta.schema.names.index("ts_ns")
    held = labels.ns(span.HELD_BACK)
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        if group.num_rows == 0:
            continue
        stats = group.column(index).statistics
        if stats is None or not stats.has_min_max or not isinstance(stats.max, int):
            raise span.HeldBackError(f"{path}: its metadata gives no bounds for ts_ns")
        if stats.max >= held:
            raise span.HeldBackError(f"{path}: ts_ns reaches the held-back test span")


def read(run_dir: Path, name: str) -> pa.Table:
    return pq.read_table(run_dir / f"{name}.parquet")
