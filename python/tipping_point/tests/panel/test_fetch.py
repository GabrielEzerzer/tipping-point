# The parallel fetch on a fake archive: it writes what archive.fetch writes, refuses a bad
# checksum unwritten, records a 404, skips verified files, never requests a held-back day.

import threading
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from market_planted import Vision, checksum, vision
from tipping_point.ising import io
from tipping_point.market import archive
from tipping_point.panel import fetch, span
from tipping_point.panel.__main__ import main

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIRST = date(2023, 1, 30)
DAYS = 4
LAST = date(2023, 2, 2)


class Counted(Vision):
    def __init__(self, files: dict[str, bytes]) -> None:
        super().__init__(files)
        self.lock = threading.Lock()
        self.running = self.most = 0

    def get(self, url: str) -> bytes | None:
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
        try:
            threading.Event().wait(0.002)
            return super().get(url)
        finally:
            with self.lock:
                self.running -= 1


@pytest.fixture
def fake() -> Counted:
    return Counted(dict(vision(SYMBOLS, FIRST, DAYS, seed=7).files))


def test_it_writes_what_the_sequential_fetch_writes(fake: Counted, tmp_path: Path) -> None:
    refused = archive.Source("klines", "ETHUSDT", "2023-01-31")
    gone = archive.Source("metrics", "BTCUSDT", "2023-02-01")
    good = fake.files[refused.url]
    fake.files[refused.url + ".CHECKSUM"] = checksum(refused.name, b"x")
    del fake.files[gone.url], fake.files[gone.url + ".CHECKSUM"]

    done = fetch.fetch(SYMBOLS, FIRST, LAST, tmp_path / "a", fake.get, workers=8)
    sequential = archive.fetch(SYMBOLS, FIRST, LAST, tmp_path / "b", Vision(fake.files).get)
    assert done == sequential
    assert fake.most > 1, "the fetch never ran two requests at once"
    manifests = [io.read_json(tmp_path / d / "manifest.json") for d in ("a", "b")]
    assert manifests[0] == manifests[1]
    assert done[refused.path]["status"] == "refused"
    assert not (tmp_path / "a" / refused.path).exists()
    assert done[gone.path]["status"] == "missing"
    verified = [p for p, e in done.items() if e["status"] == "verified"]
    assert len(verified) == len(done) - 2

    fake.files[refused.url + ".CHECKSUM"] = checksum(refused.name, good)
    fake.requests.clear()
    again = fetch.fetch(SYMBOLS, FIRST, LAST, tmp_path / "a", fake.get, workers=8)
    assert sorted(fake.requests) == sorted(
        [refused.url + ".CHECKSUM", refused.url, gone.url + ".CHECKSUM"]
    )
    assert again[refused.path]["status"] == "verified" and again[refused.path]["fetched"]
    assert not again[verified[0]]["fetched"]

    got = archive.collect(SYMBOLS, FIRST, LAST, tmp_path / "a")
    assert got.missing == [gone.url]
    assert len(got.sources) == len(done) - 1
    for key, columns in got.columns.items():
        assert all(len(v) == len(next(iter(columns.values()))) for v in columns.values()), key
    assert np.all(np.diff(got.columns[("ETHUSDT", "klines")]["open_time"]) > 0)


def test_it_keeps_the_entries_of_earlier_spans(fake: Counted, tmp_path: Path) -> None:
    fetch.fetch(SYMBOLS[:1], FIRST, FIRST, tmp_path, fake.get)
    fetch.fetch(SYMBOLS[1:], LAST, LAST, tmp_path, fake.get)
    files = io.read_json(tmp_path / "manifest.json")["files"]
    assert "klines/BTCUSDT/BTCUSDT-1m-2023-01-30.zip" in files
    assert "klines/ETHUSDT/ETHUSDT-1m-2023-02-02.zip" in files


def test_a_held_back_day_is_refused_before_any_request(fake: Counted, tmp_path: Path) -> None:
    with pytest.raises(span.HeldBackError):
        fetch.fetch(SYMBOLS, date(2023, 12, 31), date(2024, 1, 1), tmp_path, fake.get)
    with pytest.raises(span.HeldBackError):
        fetch.guarded(fake.get)(archive.Source("metrics", "BTCUSDT", "2024-01-01").url)
    with pytest.raises(span.HeldBackError):
        fetch.guarded(fake.get)(archive.Source("fundingRate", "BTCUSDT", "2024-01").url)
    assert fake.requests == []
    assert not (tmp_path / "manifest.json").exists()


def test_each_request_is_refused_on_a_held_back_day_past_the_span_check(
    fake: Counted, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(span, "refuse", lambda *days: None)
    with pytest.raises(span.HeldBackError):
        fetch.fetch(SYMBOLS, date(2024, 1, 1), date(2024, 1, 1), tmp_path, fake.get, workers=4)
    assert fake.requests == []
    assert not (tmp_path / "manifest.json").exists()


def test_a_repeated_symbol_is_fetched_once(fake: Counted, tmp_path: Path) -> None:
    done = fetch.fetch(["ETHUSDT", "BTCUSDT", "ETHUSDT"], FIRST, LAST, tmp_path / "a", fake.get)
    assert len(fake.requests) == len(set(fake.requests))
    once = archive.fetch(["ETHUSDT", "BTCUSDT"], FIRST, LAST, tmp_path / "b", fake.get)
    assert done == once
    assert next(iter(done)).split("/")[1] == "ETHUSDT"


def test_the_command_fetches_and_refuses_the_held_back_span(
    fake: Counted, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archive, "http_get", fake.get)
    where = ["--symbols", *SYMBOLS, "--dest", str(tmp_path)]
    assert main(["fetch", "--start", "2023-12-31", "--end", "2024-01-01", *where]) == 1
    assert main(["fetch", "--start", "2024-01-01", "--end", "2024-01-02", *where]) == 1
    assert fake.requests == []
    assert main(["fetch", "--start", str(FIRST), "--end", str(LAST), *where]) == 0
    assert main(["fetch", "--start", str(LAST), "--end", "2023-02-03", *where]) == 0
