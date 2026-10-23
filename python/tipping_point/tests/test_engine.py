"""The engine seam's client, without an engine: the golden Arrow stream the engine's contract
test pins, direct Parquet reads, and the HTTP plumbing against a stand-in server, including a
stream cut partway, which must raise rather than pass for a short table. The round trip against
a real engine runs from Rust (`harness/engine-harness/tests/api_python.rs`)."""

from __future__ import annotations

import http.client
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point import engine
from tipping_point.ews.__main__ import _parser as ews_cli
from tipping_point.ising.__main__ import _parser as ising_cli
from tipping_point.jepa.__main__ import _parser as jepa_cli

GOLDEN = Path(__file__).resolve().parents[3] / "engine" / "tests" / "fixtures" / "api"
# The engine's run-name rule as a literal: store::run::run_id's tag charset (ASCII letters,
# digits, '-' and '_'; engine/src/store/run.rs), which api::table::safe applies to the whole run
# id, at most 128 characters, before any read (engine/src/api/table.rs).
ENGINE_RUN_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}")


def golden_stream() -> bytes:
    return (GOLDEN / "book_samples.arrows").read_bytes()


def test_the_golden_live_stream_keeps_the_stored_contract() -> None:
    """The engine's live book samples as pinned by its contract test: schema id and provenance
    in the metadata, no null where the schema forbids one, numeric columns into torch shared."""
    reader = pa.ipc.open_stream(golden_stream())
    assert engine.schema_id(reader.schema) == "book_samples/1"
    metadata = reader.schema.metadata or {}
    assert metadata[b"tp.domain"] == b"market"
    assert json.loads(metadata[b"tp.scales"])["BTCUSDT"] == {"price_decimals": 1, "qty_decimals": 3}
    batches = list(reader)
    assert batches
    for batch in batches:
        free = set(engine.null_free(batch))
        for field in batch.schema:
            if not field.nullable:
                assert field.name in free, field.name
        seq = batch.column("seq")
        tensor = torch.from_dlpack(seq)
        assert tensor.data_ptr() == seq.buffers()[1].address + seq.offset * 8
        assert tensor.tolist() == seq.to_pylist()


def test_a_schema_without_the_engine_metadata_is_not_an_engine_stream() -> None:
    with pytest.raises(ValueError, match="not an engine stream"):
        engine.schema_id(pa.schema([("seq", pa.int64())]))


def test_read_takes_a_key_range_and_columns_straight_from_parquet(tmp_path: Path) -> None:
    run = tmp_path / "ising" / "20261022T000000Z-t"
    (run / "stats").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"key": "step", "tables": ["stats"]}))
    for first in (0, 3):
        steps = list(range(first, first + 3))
        table = pa.table({"step": pa.array(steps, pa.int64()), "m": [s * 2 for s in steps]})
        pq.write_table(table, run / "stats" / f"{first:020d}.parquet")
    (run / "stats" / "00000000000000000006.parquet.tmp").write_bytes(b"half a file")
    whole = engine.read(tmp_path, "ising", run.name, "stats")
    assert whole.column("step").to_pylist() == [0, 1, 2, 3, 4, 5]
    part = engine.read(tmp_path, "ising", run.name, "stats", start=2, end=5, columns=["m"])
    assert part.column_names == ["m"]
    assert part.column("m").to_pylist() == [4, 6, 8]
    with pytest.raises(FileNotFoundError):
        engine.read(tmp_path, "ising", run.name, "configs")


def test_run_ids_follow_the_engines_rule() -> None:
    """Catches #67's root: Python named runs its own way, so a '.' reached a run id that the
    engine lists and then refuses to open. run_id refuses every tag the engine would, and the ids
    it makes pass the engine's filter."""
    made = engine.run_id("L16-ramp-x0p02-s1")
    assert ENGINE_RUN_NAME.fullmatch(made) and re.fullmatch(r"\d{8}T\d{6}Z-L16-ramp-x0p02-s1", made)
    assert ENGINE_RUN_NAME.fullmatch(engine.run_id("t" * 111))  # 17 + 111 = 128
    for bad in ["", "a=b", "a/b", "a b", "a.b", "x0.02", "a\\b", "1e+20", "é", "t" * 112]:
        with pytest.raises(ValueError, match="run tag"):
            engine.run_id(bad)


def test_the_data_dir_is_the_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches #81: with TP_DATA_DIR unset the engine fell back to a per-user folder while the
    Python CLIs refused, so following both halves of the README gave two roots. data_dir walks
    the engine's chain (the cases its own test pins in engine/src/store/run.rs), never into the
    repository, and the ising, ews and jepa CLIs default to it, the variable set or unset."""
    cases = [
        ({"TP_DATA_DIR": "D:/data", "LOCALAPPDATA": "C:/l"}, Path("D:/data")),
        ({"TP_DATA_DIR": "", "LOCALAPPDATA": "C:/l"}, Path("C:/l") / "tipping-point"),
        ({"XDG_DATA_HOME": "/x", "HOME": "/home/g"}, Path("/x") / "tipping-point"),
        ({"HOME": "/home/g"}, Path("/home/g/.local/share") / "tipping-point"),
        ({}, Path(tempfile.gettempdir()) / "tipping-point"),
    ]
    for environ, expected in cases:
        assert engine.data_dir(environ) == expected, environ
    unset = {k: v for k, v in os.environ.items() if k != engine.DATA_DIR_ENV}
    assert not engine.data_dir(unset).resolve().is_relative_to(Path(__file__).resolve().parents[3])
    commands = [
        (ising_cli, ["pilot", "--L", "4"]),
        (ews_cli, ["score", "--prereg", "p"]),
        (jepa_cli, ["budget"]),
    ]
    monkeypatch.delenv(engine.DATA_DIR_ENV, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    for parser, argv in commands:
        assert parser().parse_args(argv).data_dir == tmp_path / "local" / "tipping-point", argv
    monkeypatch.setenv(engine.DATA_DIR_ENV, str(tmp_path / "set"))
    for parser, argv in commands:
        assert parser().parse_args(argv).data_dir == tmp_path / "set", argv


def test_read_takes_runs_named_before_and_after_the_rule(tmp_path: Path) -> None:
    """A run already on disk under a dotted id (made before run_id) still reads, as does one
    named by the rule: the rule binds new names, never readers."""
    for name in ("20261022T000000Z-L16-ramp-x0.02-s1", "20261023T000000Z-L16-ramp-x0p02-s1"):
        run = tmp_path / "ising" / name
        (run / "stats").mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({"key": "step", "tables": ["stats"]}))
        table = pa.table({"step": pa.array([1, 2], pa.int64())})
        pq.write_table(table, run / "stats" / f"{1:020d}.parquet")
        assert engine.read(tmp_path, "ising", name, "stats").column("step").to_pylist() == [1, 2]


class Stand(ThreadingHTTPServer):
    """A stand-in for the engine: answers every GET with `answer` and keeps every POST."""

    def __init__(self, answer: Callable[[BaseHTTPRequestHandler], None]) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.answer = answer
        self.posted: list[tuple[str, object]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        cast(Stand, self.server).answer(self)

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        posted = (self.path, json.loads(self.rfile.read(length)))
        cast(Stand, self.server).posted.append(posted)
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


Start = Callable[[Callable[[BaseHTTPRequestHandler], None]], Stand]


@pytest.fixture
def stand() -> Iterator[Start]:
    servers: list[Stand] = []

    def start(answer: Callable[[BaseHTTPRequestHandler], None]) -> Stand:
        server = Stand(answer)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def chunked(handler: BaseHTTPRequestHandler, body: bytes, kind: str, finish: bool) -> None:
    """Sends `body` chunked; without `finish`, hangs up before the terminating chunk."""
    handler.send_response(200)
    handler.send_header("Content-Type", kind)
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()
    for start in range(0, len(body), 4096):
        piece = body[start : start + 4096]
        handler.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
    if finish:
        handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()
    handler.close_connection = True


def test_a_whole_stream_reads_back_whole(stand: Start) -> None:
    server = stand(lambda h: chunked(h, golden_stream(), engine.STREAM_TYPE, finish=True))
    with engine.live(server.url) as stream:
        assert engine.schema_id(stream.schema) == "book_samples/1"
        table = stream.read_all()
    assert table.equals(pa.ipc.open_stream(golden_stream()).read_all())


def test_a_stream_cut_partway_raises_instead_of_reading_short(stand: Start) -> None:
    body = golden_stream()
    server = stand(lambda h: chunked(h, body[: len(body) // 2], engine.STREAM_TYPE, finish=False))
    with pytest.raises((OSError, pa.ArrowException, http.client.HTTPException)):
        engine.table("market", "r", "book_samples", server.url).read_all()


def test_an_answer_that_is_not_an_arrow_stream_is_refused(stand: Start) -> None:
    server = stand(lambda h: chunked(h, b'{"error":"no"}', "application/json", finish=True))
    with pytest.raises(ValueError, match="not an Arrow stream"):
        engine.live(server.url, run="r", speed=0)


def test_results_are_posted_as_json_to_their_topic(stand: Start) -> None:
    server = stand(lambda h: None)
    engine.publish("meters", {"symbol": "BTCUSDT", "v": 1.5}, server.url)
    assert server.posted == [("/v1/publish/meters", {"symbol": "BTCUSDT", "v": 1.5})]
