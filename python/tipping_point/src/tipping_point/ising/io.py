"""On-disk format, following the project's shared Parquet convention (#22 section 4).

<root>/ising/<run_id>/
  manifest.json                     written once, at the start
  <table>/<first_step:020>.parquet  chunks; each goes to .tmp, is synced, then renamed to a new name
  checkpoint.pt                     latest resumable state, removed after a clean end
  complete.json                     written once, at a clean end; absent = running or crashed

Tables, rows in (step, replica) order, zstd:
  stats    step i64 | replica u16 | temperature f64 | segment i32 | phase i8 | magnetisation i32 |
           energy i32
  configs  step | replica | temperature | segment | burst i32 | frame i8 | magnetisation | energy |
           spins fixed_size_binary[ceil(L*L/8)]
  balance  step | replica | temperature | segment | colour i8 | delta_e i8 | attempted i64 |
           accepted i64   (validation runs only; counts since the previous chunk)

Spins are packed one bit per site: 1 = +1, row-major, most significant bit first.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TABLES = ("stats", "configs", "balance")
_KEY = [
    ("step", pa.int64()),
    ("replica", pa.uint16()),
    ("temperature", pa.float64()),
    ("segment", pa.int32()),
]
_TOTALS = [("magnetisation", pa.int32()), ("energy", pa.int32())]


def packed_width(L: int) -> int:
    return (L * L + 7) // 8


def schema(name: str, L: int) -> pa.Schema:
    columns = {
        "stats": [("phase", pa.int8()), *_TOTALS],
        "configs": [
            ("burst", pa.int32()),
            ("frame", pa.int8()),
            *_TOTALS,
            ("spins", pa.binary(packed_width(L))),
        ],
        "balance": [
            ("colour", pa.int8()),
            ("delta_e", pa.int8()),
            ("attempted", pa.int64()),
            ("accepted", pa.int64()),
        ],
    }[name]
    return pa.schema(_KEY + columns)


def pack(spins: np.ndarray) -> np.ndarray:
    """(n, L, L) int8 +-1 -> (n, ceil(L*L/8)) uint8, one bit per site."""
    return np.packbits(spins.reshape(len(spins), -1) > 0, axis=1, bitorder="big")


def unpack(packed: np.ndarray, L: int) -> np.ndarray:
    """Inverse of `pack`."""
    bits = np.unpackbits(packed, axis=1, count=L * L, bitorder="big")
    return (bits.astype(np.int8) * 2 - 1).reshape(-1, L, L)


def unpack_column(column: pa.Array | pa.ChunkedArray, L: int) -> np.ndarray:
    """A spins column -> (n, L, L) int8."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    width = packed_width(L)
    data = np.frombuffer(array.buffers()[1], dtype=np.uint8)
    rows = data[array.offset * width : (array.offset + len(array)) * width]
    return unpack(rows.reshape(len(array), width), L)


def check_totals(M: np.ndarray, E: np.ndarray, L: int) -> None:
    """Free integrity check: on an even periodic lattice M is even and E = -2 L^2 (mod 4)."""
    if np.any(M % 2) or np.any((E + 2 * L * L) % 4):
        raise ValueError("corrupt totals: M must be even and E = -2 L^2 (mod 4)")


def write_file(path: Path, write: Callable[[IO[bytes]], object], replace: bool = False) -> None:
    """Write path.tmp, sync it, then rename it, so a crash never leaves a partial file at `path`.

    Data files only ever appear: unless `replace` is set, an existing `path` is an error.
    """
    if path.exists() and not replace:
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, record: Any, replace: bool = False) -> None:
    write_file(path, lambda f: f.write(json.dumps(record, indent=2).encode()), replace)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_chunk(run_dir: Path, name: str, L: int, columns: dict[str, np.ndarray]) -> None:
    """One Parquet chunk of table `name`, named after its first step."""
    if "magnetisation" in columns:
        check_totals(columns["magnetisation"], columns["energy"], L)
    table_schema = schema(name, L)
    arrays = [_arrow(columns[field.name], field.type) for field in table_schema]
    table = pa.Table.from_arrays(arrays, schema=table_schema)
    path = run_dir / name / f"{int(columns['step'][0]):020d}.parquet"
    path.parent.mkdir(exist_ok=True)
    write_file(path, lambda f: pq.write_table(table, f, compression="zstd"))


def _arrow(values: np.ndarray, kind: pa.DataType) -> pa.Array:
    if isinstance(kind, pa.FixedSizeBinaryType):  # packed spins: one row of bytes per config
        data = pa.py_buffer(np.ascontiguousarray(values))
        return pa.FixedSizeBinaryArray.from_buffers(kind, len(values), [None, data])
    return pa.array(values, type=kind)


def chunks(run_dir: Path, name: str) -> list[Path]:
    """A table's chunk files in step order (names are zero-padded first steps)."""
    return sorted((run_dir / name).glob("*.parquet"))


def read_table(run_dir: Path, name: str) -> pa.Table:
    paths = chunks(run_dir, name)
    if not paths:
        raise FileNotFoundError(f"no {name} chunks in {run_dir}")
    return pa.concat_tables(pq.read_table(p) for p in paths)


def discard_after(run_dir: Path, step: int) -> None:
    """Remove what a crash left behind: temp files and chunks begun after the checkpoint `step`."""
    for path in run_dir.rglob("*.tmp"):
        path.unlink()
    for name in TABLES:
        for path in chunks(run_dir, name):
            if int(path.stem) > step:
                path.unlink()


def write_complete(run_dir: Path, last_step: int) -> None:
    """complete.json: rows per table and the sha256 of every chunk; it marks a clean end."""
    rows: dict[str, int] = {}
    files: dict[str, str] = {}
    for name in TABLES:
        for path in chunks(run_dir, name):
            rows[name] = rows.get(name, 0) + pq.read_metadata(path).num_rows
            files[path.relative_to(run_dir).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    record = {
        "ended_ns": time.time_ns(),
        "last_key": last_step,
        "rows": rows,
        "clean": True,
        "files": files,
    }
    write_json(run_dir / "complete.json", record)
