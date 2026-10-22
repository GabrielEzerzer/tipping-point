"""The Python side of the API round trip, run by `api_python.rs` against an engine it serves.

`live` reads the live book samples with a backfill until the feed ends and saves them; `tables`
then reads every table of the market run and of the Ising run through the engine and compares
each with a direct `pyarrow.dataset` read of the same Parquet, replays the run as if live,
compares the saved live rows with the rotated Parquet, and publishes a meter back for the app.
`cut` reads streams an engine cuts partway and requires every one to raise.

Every batch is checked for what a model consumer relies on: a column the schema declares
non-nullable holds no null, and every null-free numeric column goes to torch with
`torch.from_dlpack` sharing the Arrow buffer. Exits nonzero on the first failure.
"""

from __future__ import annotations

import argparse
import http.client
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc
import torch
from tipping_point import engine
from tipping_point.ising import io as ising_io

MARKET_TABLES = ("frames", "depth_updates", "agg_trades", "snapshots", "book_samples")
ISING_TABLES = ("stats", "configs")
METER = {"symbol": "BTCUSDT", "ts_ms": 1700000002000, "variance": 0.125, "from": "python"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_columns(batch: pa.RecordBatch, where: str) -> int:
    """Non-nullable columns hold no null; null-free numeric columns go to torch zero-copy.
    Returns how many columns went to torch."""
    shared = 0
    for field, column in zip(batch.schema, batch.columns, strict=True):
        if not field.nullable:
            check(column.null_count == 0, f"{where}: {field.name} is non-nullable but holds nulls")
        numeric = pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
        if column.null_count or not numeric or len(column) == 0:
            continue
        tensor = torch.from_dlpack(column)
        width = column.type.bit_width // 8
        start = column.buffers()[1].address + column.offset * width
        check(tensor.data_ptr() == start, f"{where}: {field.name} was copied into torch")
        check(
            np.array_equal(tensor.numpy(), column.to_numpy(zero_copy_only=True)),
            f"{where}: {field.name} differs in torch",
        )
        shared += 1
    return shared


def check_batches(batches: Iterable[pa.RecordBatch], where: str) -> list[pa.RecordBatch]:
    out = list(batches)
    check(sum(check_columns(batch, where) for batch in out) > 0 or not out, f"{where}: no column")
    return out


def same(api: pa.Table, direct: pa.Table, where: str) -> None:
    check(api.schema.equals(direct.schema, check_metadata=False), f"{where}: schemas differ")
    check(api.num_rows == direct.num_rows, f"{where}: {api.num_rows} rows vs {direct.num_rows}")
    check(api.equals(direct, check_metadata=False), f"{where}: rows differ")


def read_stream(stream: engine.Stream, table: str, where: str) -> pa.Table:
    with stream:
        check(engine.schema_id(stream.schema) == f"{table}/1", f"{where}: schema id")
        return pa.Table.from_batches(check_batches(stream, where), schema=stream.schema)


def live(args: argparse.Namespace) -> None:
    stream = engine.live(args.url, backfill_s=args.backfill_s, timeout=120)
    metadata = stream.schema.metadata or {}
    check(metadata.get(b"tp.domain") == b"market", "live: domain")
    check(metadata.get(b"tp.run") == args.run.encode(), "live: run")
    table = read_stream(stream, "book_samples", "live")
    check(table.num_rows > 0, "live: no rows")
    with pa.OSFile(str(args.out), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)


def tables(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for name in MARKET_TABLES:
        api = read_stream(engine.table("market", args.run, name, args.url), name, name)
        same(api, engine.read(root, "market", args.run, name), f"market/{name}")
    ranged = read_stream(
        engine.table("market", args.run, "frames", args.url, start=10, end=20), "frames", "range"
    )
    same(ranged, engine.read(root, "market", args.run, "frames", start=10, end=20), "range")
    check(ranged.column("seq").to_pylist() == list(range(10, 20)), "range: seqs")

    for name in ISING_TABLES:
        api = read_stream(engine.table("ising", args.ising_run, name, args.url), name, name)
        same(api, engine.read(root, "ising", args.ising_run, name), f"ising/{name}")
    configs = engine.table("ising", args.ising_run, "configs", args.url).read_all()
    check(configs.schema.field("spins").type == pa.binary(2), "spins keep their stored type")
    spins = ising_io.unpack_column(configs.column("spins"), 4)
    totals = spins.reshape(len(spins), -1).sum(axis=1)
    check(np.array_equal(totals, configs.column("magnetisation").to_numpy()), "spins decode")

    stored = engine.read(root, "market", args.run, "book_samples")
    replayed = read_stream(engine.live(args.url, run=args.run, speed=0), "book_samples", "replay")
    same(replayed, stored, "replay as live")
    with pa.memory_map(str(args.live)) as source:
        saved = pa.ipc.open_file(source).read_all()
    same(saved, stored, "live equals the rotated Parquet")

    engine.publish("meters", METER, args.url)


def cut(args: argparse.Namespace) -> None:
    """Streams the engine cuts partway must raise, never read as complete tables."""
    openers = {
        "replay": lambda: engine.live(args.url, run=args.run, speed=0),
        "table": lambda: engine.table("market", args.run, "frames", args.url),
    }
    complete = []
    for name, opener in openers.items():
        try:
            with opener() as stream:
                table = stream.read_all()
        except (OSError, pa.ArrowException, http.client.HTTPException):
            continue
        complete.append(f"{name} read {table.num_rows} rows as complete")
    check(not complete, "; ".join(complete))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("live")
    first.add_argument("--url", required=True)
    first.add_argument("--run", required=True)
    first.add_argument("--backfill-s", type=float, required=True)
    first.add_argument("--out", type=Path, required=True)
    second = commands.add_parser("tables")
    second.add_argument("--url", required=True)
    second.add_argument("--root", required=True)
    second.add_argument("--run", required=True)
    second.add_argument("--ising-run", required=True)
    second.add_argument("--live", type=Path, required=True)
    third = commands.add_parser("cut")
    third.add_argument("--url", required=True)
    third.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    {"live": live, "tables": tables, "cut": cut}[args.command](args)
    print(f"{args.command}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
