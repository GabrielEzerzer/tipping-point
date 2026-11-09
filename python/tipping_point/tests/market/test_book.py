"""The Bybit order-book fetcher (#30 M1): the confirmatory range is refused without --confirmatory,
each range keeps its own manifest, no zip or sample is kept unless asked, a day imported before is
skipped, the reducer runs on each day's samples (`python` meaning this interpreter, checked once
before any download, its format deciding whether a day is done), a failed or missing day is
recorded while the range goes on, whatever failed, what a day put in staging goes when it ends (a
file that will not go is listed with its day, and again when the day is retried), and a staging
folder that is a link is refused (a link inside it goes as a link), and each day is seeded with the
day before's closing book, the day before a span fetched for it alone (#170). Synthetic small zips
served by a fake archive and a fake importer; no network (tests/conftest.py).
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


# A reducer's script answers the run's --version probe first: exit 0, and a format when given.
PROBE = "import sys, json\nif sys.argv[-1] == '--version': print(json.dumps({F})); sys.exit(0)\n"


def _reducer(body: str, version: str = "{}") -> list[str]:
    return [sys.executable, "-c", PROBE.replace("{F}", version) + body]


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
    """Stands in for `store import`: writes a run's complete.json, Parquet and closing book, or
    refuses."""

    def __init__(self) -> None:
        self.imported: list[str] = []
        self.seeds: list[Path | None] = []  # the closing book each import was seeded with
        self.unchained: set[str] = set()  # refuses its seed, as when its update ids do not run on
        self.refuse: set[str] = set()
        self.no_rows: set[str] = set()  # complete.json without its row counts
        self.half: set[str] = set()  # writes the samples, then fails without naming its run
        self.found: list[list[str]] = []  # the files in its data dir as each import starts

    def __call__(self, archive: Path, url: str, data_dir: Path, seed: Path | None) -> Path:
        with zipfile.ZipFile(archive) as opened:
            assert opened.testzip() is None
        self.imported.append(url)
        self.seeds.append(seed)
        assert seed is None or seed.is_file()
        self.found.append(_files(data_dir))
        if url in self.refuse:
            raise book.BookError("store import exited 1: line 3: u ran back")
        run = data_dir / "market" / archive.name.removesuffix(".data.zip")
        (run / "book_samples").mkdir(parents=True, exist_ok=True)
        (run / book.TABLE).write_bytes(PARQUET)
        if url in self.half:  # as when writing complete.json fails
            raise book.BookError("store import exited 1: complete.json: no space left")
        complete: dict[str, Any] = {
            "rows": {"book_samples": 864_000},
            "integrity": {"id_gaps": 0, "errors": 0},
        }
        if seed:
            why = {"why": "it does not chain: its last u is 7, the day's first record's u is 9"}
            complete["seed"] = {"used": False, **why} if url in self.unchained else {"used": True}
        if url in self.no_rows:
            del complete["rows"]
        (run / "complete.json").write_text(json.dumps(complete))
        (run / book.CLOSE).write_text(f"the closing book of {archive.name}\n")
        return run


def _fetch(tmp_path: Path, archive: Archive, importer: Importer, **kw: Any) -> dict[str, Any]:
    kw = {"start": START, "end": END} | kw
    symbols = kw.pop("symbols", ["BTCUSDT"])
    return book.fetch("training", symbols, tmp_path, archive, importer, **kw)


def _manifest(tmp_path: Path, name: str = "training") -> dict[str, Any]:
    return json.loads((tmp_path / name / "manifest.json").read_text())


def _files(tmp_path: Path, states: bool = False) -> list[str]:
    """Every file under the output root but the manifests and, unless asked, the closing books."""
    found = (p for p in tmp_path.rglob("*") if p.is_file() and p.name != "manifest.json")
    listed = sorted(p.relative_to(tmp_path).as_posix() for p in found)
    return [f for f in listed if states or "/state/" not in f"/{f}"]


def _link(link: Path, target: Path, kind: str) -> None:
    """Makes `link` a junction or a symlink to the folder `target`, or skips the test where that
    kind cannot be made (a junction off Windows, a symlink on Windows without the privilege)."""
    if kind == "symlink":
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"cannot make a symlink here: {error.strerror}")
    elif sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        pytest.skip("a junction is a Windows link")


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
    assert len(archive.opened) == len(importer.imported) == 4  # and the day before, for its book
    assert _fetch(tmp_path, archive, importer) == first
    assert len(archive.opened) == 4
    # Keeping samples fetches again the days whose samples went, then skips them while present.
    kept = _fetch(tmp_path, archive, importer, keep_samples=True)
    assert len(archive.opened) == 7
    _fetch(tmp_path, archive, importer, keep_samples=True)
    assert len(archive.opened) == 7
    (tmp_path / "training" / kept["BTCUSDT/2023-03-02"]["parquet"]).unlink()
    _fetch(tmp_path, archive, importer, keep_samples=True)
    assert archive.opened[7:] == [book.url("BTCUSDT", date(2023, 3, 2))]


def test_the_reducer_runs_on_each_days_samples_before_they_go(tmp_path: Path) -> None:
    archive, importer = Archive(), Importer()
    seen = tmp_path / "seen.txt"
    script = (
        "import pathlib; p = pathlib.Path(sys.argv[2]); assert p.read_bytes()[:4] == b'PAR1';"
        "open(sys.argv[1], 'a').write(p.parent.parent.name + chr(10))"
    )
    reduce = [*_reducer(script), str(seen)]
    done = _fetch(tmp_path / "out", archive, importer, reduce=reduce)
    assert {e["status"] for e in done.values()} == {"imported"}
    assert seen.read_text().split() == [f"2023-03-0{d}_BTCUSDT_ob500" for d in (1, 2, 3)]
    assert done["BTCUSDT/2023-03-01"]["reduce"] == reduce
    assert _files(tmp_path / "out") == []
    # A day imported without the reducer is fetched again to run it; a failing reducer fails it.
    _fetch(tmp_path / "plain", archive, importer, end=START)
    failing = _reducer("sys.exit('no features')")
    again = _fetch(tmp_path / "plain", archive, importer, end=START, reduce=failing)
    assert again["BTCUSDT/2023-03-01"]["status"] == "failed"
    assert "reduce exited 1: no features" in again["BTCUSDT/2023-03-01"]["error"]
    assert _files(tmp_path / "plain") == []


def test_python_is_this_interpreter_and_the_reducer_is_probed_before_any_download(
    tmp_path: Path,
) -> None:
    assert book._interpreter(["python", "-m", "x"]) == [sys.executable, "-m", "x"]
    assert book._interpreter(["python3", "-m", "x"]) == [sys.executable, "-m", "x"]
    assert book._interpreter(["D:/x/python.exe", "-m", "x"]) == ["D:/x/python.exe", "-m", "x"]
    archive, importer = Archive(), Importer()
    for reduce in (
        [str(tmp_path / "no-such-reducer.exe")],  # cannot run
        ["python", "-c", "import sys; sys.exit(3)"],  # exits non-zero on --version
    ):
        with pytest.raises(book.BookError, match="reducer"):
            _fetch(tmp_path / "out", archive, importer, reduce=reduce)
    assert archive.opened == [] and importer.imported == []
    assert not (tmp_path / "out" / "training" / "manifest.json").exists()


def test_a_missing_reducer_fails_the_day_and_the_range_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reducer that vanishes after the probe (or a probe that lies) fails each day, recorded."""
    monkeypatch.setattr(book, "_probe", lambda reduce: None)
    archive, importer = Archive(), Importer()
    done = _fetch(tmp_path, archive, importer, reduce=[str(tmp_path / "gone.exe")])
    assert [e["status"] for e in done.values()] == ["failed"] * 3
    assert all("FileNotFoundError" in e["error"] for e in done.values())
    assert len(_manifest(tmp_path)["days"]) == 3 and _files(tmp_path) == []


def test_any_failure_after_the_download_is_recorded_and_the_range_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, importer = Archive(), Importer()
    url = {d: book.url("BTCUSDT", date(2023, 3, d)) for d in (1, 2, 3)}
    archive.headers[url[1]] = {"Content-Length": "many"}  # ValueError
    importer.no_rows.add(url[2])  # KeyError
    real_unlink, held = Path.unlink, {"times": 0}

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith("2023-03-03") and self.suffix == ".zip" and held["times"] < 2:
            held["times"] += 1  # a scanner holds the zip twice, then lets go
            raise PermissionError(13, "in use", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(book.time, "sleep", lambda _: None)
    done = _fetch(tmp_path, archive, importer)
    status = {k[-2:]: (e["status"], e.get("error", "")) for k, e in done.items()}
    assert status["01"][0] == "failed" and "invalid literal" in status["01"][1]
    assert status["02"] == ("failed", "KeyError: 'rows'")
    assert status["03"] == ("imported", "") and held["times"] == 2
    held["times"] = -10  # now it never lets go: the day fails, recorded, after the retries
    (tmp_path / "training" / "manifest.json").unlink()
    done = _fetch(tmp_path, archive, importer, start=date(2023, 3, 3))
    assert done["BTCUSDT/2023-03-03"]["status"] == "failed"
    assert "PermissionError" in done["BTCUSDT/2023-03-03"]["error"]
    assert done["BTCUSDT/2023-03-03"]["leftover"] == ["staging/2023-03-03_BTCUSDT_ob500.data.zip"]


def test_samples_an_import_wrote_before_failing_go_with_the_day(tmp_path: Path) -> None:
    """`store import` can write a day's samples and then fail (on complete.json, say) without
    naming its run: they go when the day ends all the same, however many days fail in a row."""
    archive, importer = Archive(), Importer()
    days = [date(2023, 3, d) for d in (1, 2, 3, 4)]
    importer.half = {book.url("BTCUSDT", d) for d in days[:3]}
    staging = tmp_path / "training" / "staging"
    after: list[list[str]] = []  # what staging holds as each day is logged
    done = _fetch(
        tmp_path, archive, importer, end=days[-1], log=lambda _: after.append(_files(staging))
    )
    assert [e["status"] for e in done.values()] == ["failed"] * 3 + ["imported"]
    assert after == [[]] * 4
    # Each import found its own zip and nothing else: no earlier day's samples, nor those of the
    # day before the span, imported first for its closing book.
    before = date(2023, 2, 28)
    assert importer.found == [[book.file_name("BTCUSDT", d)] for d in [before, *days]]
    assert not any("leftover" in e for e in done.values())


def test_a_file_that_will_not_go_is_listed_with_its_day_and_the_range_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A day's samples that something never lets go stay, listed in that day's `leftover` once the
    tries are spent; the later days are imported and list nothing, and the next run starts."""
    held = f"staging/market/2023-03-02_BTCUSDT_ob500/{book.TABLE.as_posix()}"
    real_unlink = os.unlink

    def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path).as_posix().endswith(held):
            raise PermissionError(13, "in use", str(path))
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)  # shutil.rmtree's too, so it outlasts the run
    monkeypatch.setattr(book.time, "sleep", lambda _: None)
    archive, importer = Archive(), Importer()
    done = _fetch(tmp_path, archive, importer)
    assert {e["status"] for e in done.values()} == {"imported"}
    left = {key[-2:]: entry.get("leftover") for key, entry in done.items()}
    assert left == {"01": None, "02": [held], "03": None}
    assert _manifest(tmp_path)["days"]["BTCUSDT/2023-03-02"]["leftover"] == [held]
    assert "not deleted" in book.summary(done["BTCUSDT/2023-03-02"])
    assert _files(tmp_path) == [f"training/{held}"]
    assert _fetch(tmp_path, archive, importer) == done  # the next run starts with it still there


def test_a_retry_of_the_day_lists_its_file_that_still_will_not_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file a failed day could not delete is still that day's when the day is tried again, though
    the retry found it there before the day began; once it can go, the day lists it no more."""
    held = f"staging/market/2023-03-01_BTCUSDT_ob500/{book.TABLE.as_posix()}"
    real_unlink, holding = os.unlink, {"on": True}

    def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        if holding["on"] and Path(path).as_posix().endswith(held):
            raise PermissionError(13, "in use", str(path))
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(book.time, "sleep", lambda _: None)
    archive, importer = Archive(), Importer()
    importer.half.add(book.url("BTCUSDT", START))  # writes the samples, then fails: tried again
    for still_held in (True, True, False):  # the first run, a retry, a retry once it lets go
        holding["on"] = still_held
        _fetch(tmp_path, archive, importer, end=START)
        entry = _manifest(tmp_path)["days"]["BTCUSDT/2023-03-01"]
        assert entry["status"] == "failed"
        assert entry.get("leftover") == ([held] if still_held else None)
    assert len(importer.imported) == 4 and _files(tmp_path) == []  # 3 and the day before, once


@pytest.mark.parametrize("kind", ["junction", "symlink"])
def test_a_staging_folder_that_is_a_link_is_refused_before_anything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    """The run deletes everything in staging, so when staging is a link to a folder elsewhere the
    command exits 1 before it downloads anything, leaving the link and that folder as they are."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("not the run's")
    staging = tmp_path / "out" / "training" / "staging"
    staging.parent.mkdir(parents=True)
    _link(staging, outside, kind)
    store = tmp_path / "store"
    store.write_bytes(b"")
    archive = Archive()
    monkeypatch.setattr(book, "http_open", archive)
    monkeypatch.setattr(book, "engine_importer", lambda _: Importer())
    argv = ["book", "--range", "training", "--start", "2023-03-01", "--end", "2023-03-01"]
    argv += ["--symbols", "BTCUSDT", "--store", str(store), "--out", str(tmp_path / "out")]
    assert main(argv) == 1
    assert "is a link" in capsys.readouterr().err
    assert archive.opened == []
    assert _files(outside) == ["sentinel.txt"]
    assert staging.is_symlink() or staging.is_junction()  # left as it was
    assert not (staging.parent / "manifest.json").exists()


@pytest.mark.parametrize("kind", ["junction", "symlink"])
def test_a_link_is_deleted_as_a_link_never_followed(tmp_path: Path, kind: str) -> None:
    """A link inside staging goes as a link, leaving the folder it led to; a folder that is itself
    a link is not cleared at all."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("not the run's")
    staging = tmp_path / "staging"
    run = staging / "market" / "2023-03-01_BTCUSDT_ob500"
    run.mkdir(parents=True)
    (run / "complete.json").write_text("{}")
    _link(run / "book_samples", outside, kind)
    book._clear(staging)
    assert list(staging.iterdir()) == []
    assert _files(outside) == ["sentinel.txt"]
    linked = tmp_path / "linked"
    _link(linked, outside, kind)
    with pytest.raises(book.BookError, match="is a link"):
        book._clear(linked)
    assert _files(outside) == ["sentinel.txt"]
    assert linked.is_symlink() or linked.is_junction()


def test_a_day_reduced_with_another_format_is_not_done(tmp_path: Path) -> None:
    archive, importer = Archive(), Importer()
    body = "print(json.dumps({'path': 'f.parquet', 'format': FORMAT}))"
    one = _reducer(f"FORMAT = 1\n{body}", '{"format": 1}')
    _fetch(tmp_path, archive, importer, end=START, reduce=one)
    (tmp_path / "training" / "f.parquet").write_bytes(b"")
    entry = _manifest(tmp_path)["days"]["BTCUSDT/2023-03-01"]
    assert entry["reduced"] == {"path": "f.parquet", "format": 1}
    _fetch(tmp_path, archive, importer, end=START, reduce=one)
    assert len(archive.opened) == 2  # same format: done (the first was the day before's book)
    two = _reducer(f"FORMAT = 2\n{body}", '{"format": 2}')
    # The commands differ too, which alone redoes the day, so check the format rule directly.
    assert not book._done(entry, tmp_path / "training", False, one, {"format": 2})
    assert book._done(entry, tmp_path / "training", False, one, {"format": 1})
    _fetch(tmp_path, archive, importer, end=START, reduce=two)
    assert _manifest(tmp_path)["days"]["BTCUSDT/2023-03-01"]["reduced"]["format"] == 2


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
    before = [book.url(s, date(2023, 2, 28)) for s in symbols]  # for their closing books
    assert importer.imported == [before[0], before[1], url["ETHUSDT", 2], url["ETHUSDT", 3]]
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


def test_each_day_is_seeded_with_the_day_befores_closing_book(tmp_path: Path) -> None:
    """The day before a span is fetched and imported for its closing book alone (not reduced, not
    a day of the range), then each day seeds the next; a later span goes on from the last book."""
    archive, importer = Archive(), Importer()
    seen = tmp_path / "seen.txt"
    script = "import pathlib; p = pathlib.Path(sys.argv[2]).parents[1].name"
    reduce = [*_reducer(f"{script}; open(sys.argv[1], 'a').write(p + chr(10))"), str(seen)]
    done = _fetch(tmp_path / "out", archive, importer, reduce=reduce)
    where = tmp_path / "out" / "training"
    days = [date(2023, 2, 28), START, date(2023, 3, 2), END]
    assert importer.imported == archive.opened == [book.url("BTCUSDT", d) for d in days]
    assert (
        list(_manifest(tmp_path / "out")["days"])
        == list(done)
        == [f"BTCUSDT/{d}" for d in days[1:]]
    )
    assert seen.read_text().split() == [f"{d}_BTCUSDT_ob500" for d in days[1:]]
    state = [f"state/BTCUSDT/{d}.jsonl" for d in days]
    assert importer.seeds == [None] + [where / f for f in state[:-1]]
    for d, entry in zip(days[1:], done.values(), strict=True):
        before = state[days.index(d) - 1]
        assert entry["seed"] == {"state": before, "used": True}
        assert entry["state"] == state[days.index(d)]
    assert _files(tmp_path / "out", states=True) == [f"training/{f}" for f in state]
    closing = (where / state[0]).read_text()
    assert closing == "the closing book of 2023-02-28_BTCUSDT_ob500.data.zip\n"
    # The next span starts from the closing book the last one left: nothing else is fetched.
    later = date(2023, 3, 4)
    _fetch(tmp_path / "out", archive, importer, start=later, end=later, reduce=reduce)
    assert archive.opened[4:] == [book.url("BTCUSDT", later)]
    assert importer.seeds[4:] == [where / state[-1]]


def test_a_day_not_seeded_is_imported_unseeded_and_says_why(tmp_path: Path) -> None:
    """No closing book for the day before (not published), or one the engine refuses (its update
    ids do not run on to the day's): the day is imported all the same, its entry saying why."""
    archive, importer = Archive(), Importer()
    archive.absent.add(book.url("BTCUSDT", date(2023, 2, 28)))
    importer.unchained.add(book.url("BTCUSDT", END))
    done = _fetch(tmp_path, archive, importer)
    assert {e["status"] for e in done.values()} == {"imported"}
    first, last = done["BTCUSDT/2023-03-01"]["seed"], done["BTCUSDT/2023-03-03"]["seed"]
    assert first == {
        "state": None,
        "why": "no closing book for 2023-02-28 (missing: the archive has no such file (404))",
        "used": False,
    }
    assert last["state"] == "state/BTCUSDT/2023-03-02.jsonl" and not last["used"]
    assert last["why"].startswith("it does not chain")
    assert importer.seeds[0] is None and importer.seeds[2] is not None
    # A day before that is in the manifest (here failed) is not fetched again for its book.
    archive.opened.clear()
    archive.broken.add(book.url("BTCUSDT", date(2023, 3, 5)))
    _fetch(tmp_path, archive, importer, start=date(2023, 3, 5), end=date(2023, 3, 6))
    assert archive.opened == [book.url("BTCUSDT", date(2023, 3, d)) for d in (4, 5, 6)]
    six = _manifest(tmp_path)["days"]["BTCUSDT/2023-03-06"]["seed"]
    assert six == {"state": None, "why": "no closing book for 2023-03-05", "used": False}


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
    run = importer(archive, book.url("BTCUSDT", date(2026, 10, 20)), tmp_path, None)
    assert run == tmp_path / "market" / "20261020T000000Z-bybit-linear-btcusdt"
    complete = json.loads((run / "complete.json").read_text())
    assert complete["rows"]["book_samples"] > 0 and complete["integrity"]["id_gaps"] == 1
    assert "seed" not in complete
    assert (run / book.TABLE).is_file() and (run / book.CLOSE).is_file()
