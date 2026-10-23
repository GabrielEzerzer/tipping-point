"""The engine seam: Parquet reads that need no engine, and the engine's local HTTP API.

Offline work (training on Kaggle, experiments, baselines) reads a run's Parquet directly with
`read`, following the shared layout `<root>/<domain>/<run_id>/<table>/<first_key:020>.parquet`;
`run_id` names a new run by the engine's rule, so the engine can open it.
The engine is only on the live path: `live` streams book samples as Arrow record batches, each
one or more whole 100 ms grid instants with one row per symbol (one instant per batch live; a
backfill or an unpaced replay sends several), `table` reads a range of any table through the
engine, and `publish` hands results (meters, Jev labels, P&L) back for the app.

Every stream keeps the stored schema, with `tp.schema = "<table>/<major>"` in its metadata. A
column the schema declares non-nullable never holds a null, so `torch.from_dlpack(column)`
shares its buffer; nullable book levels (past the book's depth, or while unsynced) must be
filled first, which copies.

An engine stream either ends with Arrow's end-of-stream marker or fails: the engine aborts the
HTTP response on any error, so a short read raises instead of passing for a short table.

Stdlib and pyarrow only.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import quote, urlencode

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc

DEFAULT_URL = "http://127.0.0.1:7731"
STREAM_TYPE = "application/vnd.apache.arrow.stream"
SCHEMA_KEY = b"tp.schema"
_TAG = re.compile(r"[A-Za-z0-9_-]+")  # store::run::run_id's charset (engine/src/store/run.rs)
_MAX_TAG = 128 - len("YYYYMMDDTHHMMSSZ-")  # the API opens names of <= 128 (api/table.rs, safe)


def run_id(tag: str) -> str:
    """A new run's id, the UTC start time plus a tag (20261021T140512Z-L16-ramp-x0p02-s1), by
    the engine's rule (store::run::run_id): the tag holds only ASCII letters, digits, '-' and
    '_', and the whole id at most 128 characters. The API lists a run of any other name but
    refuses to open it (a '.' among them). Readers take any name already on disk."""
    if not (_TAG.fullmatch(tag) and len(tag) <= _MAX_TAG):
        raise ValueError(f"run tag {tag!r} must be 1-{_MAX_TAG} ASCII letters, digits, '-' or '_'")
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{tag}"


def read(
    root: str | Path,
    domain: str,
    run: str,
    table: str,
    *,
    start: int | None = None,
    end: int | None = None,
    columns: list[str] | None = None,
) -> pa.Table:
    """A table of a run straight from its Parquet, no engine: the rows whose key (the manifest's
    `key`, `seq` for market runs and `step` for Ising runs) is in [start, end), in key order.
    Row-group statistics prune the files a range does not touch."""
    run_dir = Path(root) / domain / run
    key = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["key"]
    files = sorted(str(p) for p in (run_dir / table).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {table} files in {run_dir}")
    dataset = ds.dataset(files, format="parquet")
    condition = None
    if start is not None:
        condition = ds.field(key) >= start
    if end is not None:
        below = ds.field(key) < end
        condition = below if condition is None else condition & below
    return dataset.to_table(columns=columns, filter=condition)


class Stream:
    """An Arrow IPC stream from the engine. Iterate it for record batches; `schema` is known
    before the first one. Close it, or use it in a `with` block, to hang up."""

    def __init__(self, url: str, timeout: float | None = None) -> None:
        self._response = urllib.request.urlopen(url, timeout=timeout)
        kind = self._response.headers.get("Content-Type", "")
        if kind != STREAM_TYPE:
            self._response.close()
            raise ValueError(f"{url} answered {kind!r}, not an Arrow stream")
        self._reader = pa.ipc.open_stream(self._response)

    @property
    def schema(self) -> pa.Schema:
        return self._reader.schema

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        return iter(self._reader)

    def read_all(self) -> pa.Table:
        return self._reader.read_all()

    def close(self) -> None:
        self._response.close()

    def __enter__(self) -> Stream:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _url(base: str, path: str, query: Mapping[str, object]) -> str:
    present = {k: v for k, v in query.items() if v is not None}
    return f"{base.rstrip('/')}{path}" + (f"?{urlencode(present)}" if present else "")


def live(
    url: str = DEFAULT_URL,
    *,
    backfill_s: float | None = None,
    run: str | None = None,
    speed: float | None = None,
    timeout: float | None = None,
) -> Stream:
    """The live book samples. With `backfill_s`, the stream first replays that many seconds of
    samples the engine still holds in memory, then carries on live with no gap and no overlap, so
    a reconnect refills its window. With `run`, it replays that recorded run's samples instead,
    paced at `speed` times real time (0 for as fast as possible): a live consumer tested on a
    recording."""
    query = {"backfill_s": backfill_s, "run": run, "speed": speed}
    return Stream(_url(url, "/v1/live/book_samples", query), timeout)


def table(
    domain: str,
    run: str,
    name: str,
    url: str = DEFAULT_URL,
    *,
    start: int | None = None,
    end: int | None = None,
    timeout: float | None = None,
) -> Stream:
    """A table of a run read through the engine, rows whose key is in [start, end), in the
    order stored, with the stored schema."""
    path = f"/v1/table/{quote(domain)}/{quote(run)}/{quote(name)}"
    return Stream(_url(url, path, {"start": start, "end": end}), timeout)


def publish(topic: str, payload: Mapping[str, Any], url: str = DEFAULT_URL) -> None:
    """Sends one result to the app through the engine: `meters` and `pnl` keep the latest per
    `symbol`, `jev` and `ising` keep every message."""
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        _url(url, f"/v1/publish/{quote(topic)}", {}),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        response.read()


def schema_id(schema: pa.Schema) -> str:
    """The `<table>/<major>` an engine stream declares."""
    metadata = schema.metadata or {}
    if SCHEMA_KEY not in metadata:
        raise ValueError("not an engine stream: no tp.schema in the schema metadata")
    return metadata[SCHEMA_KEY].decode()


def null_free(batch: pa.RecordBatch) -> list[str]:
    """The columns that hold no null, by name: the ones a consumer can hand to
    `torch.from_dlpack` as they are when their type is numeric."""
    return [
        field.name
        for field, column in zip(batch.schema, batch.columns, strict=True)
        if column.null_count == 0
    ]
