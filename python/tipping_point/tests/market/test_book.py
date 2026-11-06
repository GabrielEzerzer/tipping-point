"""The Bybit order-book fetcher (#30 M1): the confirmatory range is refused without --confirmatory,
each range keeps its own manifest, no zip or sample is kept unless asked, a day imported before is
skipped, the reducer runs on each day's samples, and a failed or missing day is recorded while the
range goes on. Synthetic small zips served by a fake archive and a fake importer; no network.
The engine's own import runs only when TP_STORE_BIN names a built store binary."""

import hashlib
import io
import json
import os
import shutil
import sys
import urllib.error
import zipfile
from datetime import date
from http.client import HTTPMessage
from pathlib import Path
from typing import Any

import pytest
from tipping_point.market import book
from tipping_point.market.__main__ import main

REPO = Path(__file__).resolve().parents[4]
START = date(2023, 3, 1)
END = date(2023, 3, 3)
PARQUET = b"PAR1" * 10


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the archive, even when the refusal it checks is broken."""

    def urlopen(request: Any, timeout: float) -> None:
        raise AssertionError(f"a test opened {request.full_url}")

    monkeypatch.setattr(book.urllib.request, "urlopen", urlopen)


def _zip(member: str) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(member, date_time=(2023, 3, 2, 0, 0, 0))  # same bytes every time
        archive.writestr(info, b'{"topic":"orderbook.500.BTCUSDT","type":"snapshot"}\r\n')
    return out.getvalue()


class Response(io.BytesIO):
    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        super().__init__(body)
        self.headers = headers


class Archive:
    """Serves a small zip at every day's URL, with its true size and MD5 unless told otherwise."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.absent: set[str] = set()
        self.headers: dict[str, dict[str, str]] = {}
        self.broken: set[str] = set()

    def __call__(self, url: str) -> Response | None:
        self.opened.append(url)
        if url in self.broken:
            raise urllib.error.URLError("connection reset")
        if url in self.absent:
            return None
        body = _zip(url.rsplit("/", 1)[1].removesuffix(".zip"))
        headers = {
            "Content-Length": str(len(body)),
            "ETag": f'"{hashlib.md5(body).hexdigest()}"',
        }
        return Response(body, headers | self.headers.get(url, {}))


class Importer:
    """Stands in for `store import`: writes a run's complete.json and Parquet, or refuses."""

    def __init__(self) -> None:
        self.imported: list[str] = []
        self.refuse: set[str] = set()

    def __call__(self, archive: Path, url: str, data_dir: Path) -> Path:
        with zipfile.ZipFile(archive) as opened:
            assert opened.testzip() is None
        self.imported.append(url)
        if url in self.refuse:
            raise book.BookError("store import exited 1: line 3: u ran back")
        run = data_dir / "market" / archive.name.removesuffix(".data.zip")
        (run / "book_samples").mkdir(parents=True, exist_ok=True)
        (run / book.TABLE).write_bytes(PARQUET)
        complete = {"rows": {"book_samples": 864_000}, "integrity": {"id_gaps": 0, "errors": 0}}
        (run / "complete.json").write_text(json.dumps(complete))
        return run


def _fetch(tmp_path: Path, archive: Archive, importer: Importer, **kw: Any) -> dict[str, Any]:
    kw = {"start": START, "end": END} | kw
    symbols = kw.pop("symbols", ["BTCUSDT"])
    return book.fetch("training", symbols, tmp_path, archive, importer, **kw)


def _manifest(tmp_path: Path, name: str = "training") -> dict[str, Any]:
    return json.loads((tmp_path / name / "manifest.json").read_text())


def _files(tmp_path: Path) -> list[str]:
    """Every file under the output root but the manifests."""
    found = (p for p in tmp_path.rglob("*") if p.is_file() and p.name != "manifest.json")
    return sorted(p.relative_to(tmp_path).as_posix() for p in found)


def test_the_confirmatory_range_is_refused_without_confirmatory(tmp_path: Path) -> None:
    archive, importer = Archive(), Importer()
    with pytest.raises(book.BookError, match="--confirmatory"):
        book.fetch("confirmatory", ["BTCUSDT"], tmp_path, archive, importer)
    for start, end in ((date(2023, 12, 31), date(2024, 1, 1)), (date(2023, 9, 30),) * 2):
        with pytest.raises(book.BookError, match="not inside the exploratory range"):
            book.fetch("exploratory", ["BTCUSDT"], tmp_path, archive, importer, start, end)
    with pytest.raises(book.BookError, match="not inside the training range"):
        book.fetch("training", ["BTCUSDT"], tmp_path, archive, importer, date(2023, 1, 17))
    assert archive.opened == []
    assert not (tmp_path / "confirmatory").exists()
    day = date(2024, 1, 1)
    done = book.fetch(
        "confirmatory", ["BTCUSDT"], tmp_path, archive, importer, day, day, confirmatory=True
    )
    assert [e["status"] for e in done.values()] == ["imported"]
    assert book.span("training", None, None, False)[0] == date(2023, 1, 18)
    assert len(book.span("exploratory", None, None, False)) == 92


def test_the_command_line_refuses_the_confirmatory_range(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "store.exe"
    store.write_bytes(b"")
    argv = ["book", "--range", "confirmatory", "--store", str(store), "--out", str(tmp_path)]
    assert main(argv) == 1
    assert "--confirmatory" in capsys.readouterr().err
    assert not (tmp_path / "confirmatory").exists()


def test_each_range_keeps_its_own_manifest(tmp_path: Path) -> None:
    archive, importer = Archive(), Importer()
    _fetch(tmp_path, archive, importer)
    day = date(2023, 10, 1)
    book.fetch("exploratory", ["BTCUSDT"], tmp_path, archive, importer, day, day)
    assert list(_manifest(tmp_path)["days"]) == [f"BTCUSDT/2023-03-0{d}" for d in (1, 2, 3)]
    assert list(_manifest(tmp_path, "exploratory")["days"]) == ["BTCUSDT/2023-10-01"]
    assert _manifest(tmp_path, "exploratory")["range"] == "exploratory"


def test_no_zip_or_sample_is_kept_but_the_day_is_recorded(tmp_path: Path) -> None:
    archive, importer = Archive(), Importer()
    _fetch(tmp_path, archive, importer)
    assert _files(tmp_path) == []
    entry = _manifest(tmp_path)["days"]["BTCUSDT/2023-03-01"]
    assert entry["url"] == book.url("BTCUSDT", START)
    assert entry["sha256"] == hashlib.sha256(_zip("2023-03-01_BTCUSDT_ob500.data")).hexdigest()
    assert entry["md5_checked"] and entry["samples"] == 864_000
    assert entry["parquet_bytes"] == len(PARQUET) and entry["parquet"] is None
    assert entry["parquet_sha256"] == hashlib.sha256(PARQUET).hexdigest()
    assert entry["integrity"]["id_gaps"] == 0
    assert entry["peak_bytes"] > entry["bytes"] + len(PARQUET)  # zip and run on disk together


def test_keep_samples_keeps_them_under_the_range(tmp_path: Path) -> None:
    archive, importer = Archive(), Importer()
    done = _fetch(tmp_path, archive, importer, keep_samples=True)
    parquet = done["BTCUSDT/2023-03-02"]["parquet"]
    assert parquet == f"samples/market/2023-03-02_BTCUSDT_ob500/{book.TABLE.as_posix()}"
    assert (tmp_path / "training" / parquet).read_bytes() == PARQUET
    assert not any(f.endswith(".zip") for f in _files(tmp_path))


def test_a_day_imported_before_is_skipped(tmp_path: Path) -> None:
    archive, importer = Archive(), Importer()
    first = _fetch(tmp_path, archive, importer)
    assert [e["status"] for e in first.values()] == ["imported"] * 3
    assert len(archive.opened) == len(importer.imported) == 3
    assert _fetch(tmp_path, archive, importer) == first
    assert len(archive.opened) == 3
    # Keeping samples fetches again the days whose samples went, then skips them while present.
    kept = _fetch(tmp_path, archive, importer, keep_samples=True)
    assert len(archive.opened) == 6
    _fetch(tmp_path, archive, importer, keep_samples=True)
    assert len(archive.opened) == 6
    (tmp_path / "training" / kept["BTCUSDT/2023-03-02"]["parquet"]).unlink()
    _fetch(tmp_path, archive, importer, keep_samples=True)
    assert archive.opened[6:] == [book.url("BTCUSDT", date(2023, 3, 2))]


def test_the_reducer_runs_on_each_days_samples_before_they_go(tmp_path: Path) -> None:
    archive, importer = Archive(), Importer()
    seen = tmp_path / "seen.txt"
    script = (
        "import sys, pathlib; p = pathlib.Path(sys.argv[2]); assert p.read_bytes()[:4] == b'PAR1';"
        "open(sys.argv[1], 'a').write(p.parent.parent.name + chr(10))"
    )
    reduce = [sys.executable, "-c", script, str(seen)]
    done = _fetch(tmp_path / "out", archive, importer, reduce=reduce)
    assert {e["status"] for e in done.values()} == {"imported"}
    assert seen.read_text().split() == [f"2023-03-0{d}_BTCUSDT_ob500" for d in (1, 2, 3)]
    assert done["BTCUSDT/2023-03-01"]["reduce"] == reduce
    assert _files(tmp_path / "out") == []
    # A day imported without the reducer is fetched again to run it; a failing reducer fails it.
    _fetch(tmp_path / "plain", archive, importer, end=START)
    failing = [sys.executable, "-c", "import sys; sys.exit('no features')"]
    again = _fetch(tmp_path / "plain", archive, importer, end=START, reduce=failing)
    assert again["BTCUSDT/2023-03-01"]["status"] == "failed"
    assert "reduce exited 1: no features" in again["BTCUSDT/2023-03-01"]["error"]
    assert _files(tmp_path / "plain") == []


def test_a_failed_day_is_recorded_and_the_range_goes_on(tmp_path: Path) -> None:
    archive, importer = Archive(), Importer()
    symbols = ["BTCUSDT", "ETHUSDT"]
    url = {(s, d): book.url(s, date(2023, 3, d)) for s in symbols for d in (1, 2, 3)}
    archive.headers[url["BTCUSDT", 1]] = {"Content-Length": "999999"}  # short download
    archive.headers[url["BTCUSDT", 2]] = {"ETag": '"' + "0" * 32 + '"'}  # MD5 mismatch
    archive.broken.add(url["BTCUSDT", 3])  # network
    archive.absent.add(url["ETHUSDT", 1])  # not published
    importer.refuse.add(url["ETHUSDT", 2])  # the engine refuses it
    done = _fetch(tmp_path, archive, importer, symbols=symbols)
    status = {k: (e["status"], e.get("error", "")) for k, e in done.items()}
    assert status["BTCUSDT/2023-03-01"][1].startswith("download: ")
    assert "bytes, but the server said 999999" in status["BTCUSDT/2023-03-01"][1]
    assert "but the ETag says" in status["BTCUSDT/2023-03-02"][1]
    assert "connection reset" in status["BTCUSDT/2023-03-03"][1]
    assert status["ETHUSDT/2023-03-01"][0] == "missing"
    assert "u ran back" in status["ETHUSDT/2023-03-02"][1]
    assert status["ETHUSDT/2023-03-03"] == ("imported", "")
    assert importer.imported == [url["ETHUSDT", 2], url["ETHUSDT", 3]]
    assert _files(tmp_path) == []  # no zip, no partial download and no samples are kept
    assert "failed: 4" in book.report(done)
    # Every day not imported is tried again; the imported one is not.
    archive.headers.clear()
    archive.broken.clear()
    archive.absent.clear()
    importer.refuse.clear()
    archive.opened.clear()
    again = _fetch(tmp_path, archive, importer, symbols=symbols)
    assert {e["status"] for e in again.values()} == {"imported"}
    assert url["ETHUSDT", 3] not in archive.opened
    assert len(archive.opened) == 5


def test_a_failed_run_of_the_command_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    store.write_bytes(b"")
    archive = Archive()
    archive.absent.add(book.url("BTCUSDT", START))
    monkeypatch.setattr(book, "http_open", archive)
    monkeypatch.setattr(book, "engine_importer", lambda _: Importer())
    argv = ["book", "--range", "training", "--start", "2023-03-01", "--end", "2023-03-02"]
    argv += ["--symbols", "BTCUSDT", "--store", str(store), "--out", str(tmp_path / "out")]
    assert main(argv) == 1
    archive.absent.clear()
    assert main(argv) == 0
    days = list(_manifest(tmp_path / "out")["days"])
    assert days == ["BTCUSDT/2023-03-01", "BTCUSDT/2023-03-02"]


def test_files_are_named_by_the_depth_bybit_published_that_day() -> None:
    assert book.url("ETHUSDT", date(2025, 8, 20)) == (
        "https://quote-saver.bycsi.com/orderbook/linear/ETHUSDT/2025-08-20_ETHUSDT_ob500.data.zip"
    )
    assert book.file_name("BTCUSDT", date(2025, 8, 21)) == "2025-08-21_BTCUSDT_ob200.data.zip"


def test_the_run_is_read_from_the_engines_output() -> None:
    out = (
        "wrote C:\\Users\\x\\tp\\market\\20230610T000000Z-bybit-linear-ethusdt\\book_samples\\"
        "00000000000000000000.parquet\n"
        "C:\\Users\\x\\tp\\market\\20230610T000000Z-bybit-linear-ethusdt: book_samples=864000\n"
        "records=863999 outside_day=0 id_gaps=0 trade_id_gaps=- snapshots=1 errors=0\n"
    )
    assert book.run_dir(out) == Path(
        "C:\\Users\\x\\tp\\market\\20230610T000000Z-bybit-linear-ethusdt"
    )
    with pytest.raises(book.BookError, match="named no run"):
        book.run_dir("error: nothing\n")


def test_requests_carry_a_plain_user_agent_and_404_means_unpublished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, str]] = []

    def urlopen(request: Any, timeout: float) -> str:
        seen.append(dict(request.header_items()))
        if request.full_url.endswith("absent.zip"):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", HTTPMessage(), None)
        return "response"

    monkeypatch.setattr(book.urllib.request, "urlopen", urlopen)
    assert book.http_open(f"{book.BASE_URL}/x.zip") == "response"
    assert book.http_open(f"{book.BASE_URL}/absent.zip") is None
    assert seen == [{"User-agent": book.USER_AGENT}] * 2


@pytest.mark.skipif(not os.environ.get("TP_STORE_BIN"), reason="TP_STORE_BIN names no binary")
def test_the_engine_imports_the_committed_synthetic_day(tmp_path: Path) -> None:
    name = "2026-10-20_BTCUSDT_ob200.data.zip"
    archive = tmp_path / name
    shutil.copy(REPO / "engine/tests/fixtures/archives" / name, archive)
    importer = book.engine_importer(Path(os.environ["TP_STORE_BIN"]))
    run = importer(archive, book.url("BTCUSDT", date(2026, 10, 20)), tmp_path)
    assert run == tmp_path / "market" / "20261020T000000Z-bybit-linear-btcusdt"
    complete = json.loads((run / "complete.json").read_text())
    assert complete["rows"]["book_samples"] > 0 and complete["integrity"]["id_gaps"] == 1
    assert (run / book.TABLE).is_file()
