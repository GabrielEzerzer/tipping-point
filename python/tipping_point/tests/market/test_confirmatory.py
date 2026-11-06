"""The confirmatory span through the frozen file only (#29): `labels` and `cascade` with --prereg
take the span, the symbols and every parameter from a frozen market-labels file and stamp the run's
manifest with it, and refuse a file changed since its tag, an option given beside it, a held module
that is not the tag's commit's, and an uncommitted change to the command line or prereg.py. A
scratch repository holds the file, frozen as the real one is, and a stand-in for each file the
check reads. The archive is the fake one of test_cli, spanning the boundary, with a cascade planted
on BTCUSDT either side of it, and no socket may connect, the archive fixture's fetch included."""

import hashlib
import inspect
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from market_planted import Vision, vision
from tipping_point import engine
from tipping_point.ews import prereg
from tipping_point.market import __main__ as market_cli
from tipping_point.market import archive, bars, cascade

REPO = Path(__file__).resolve().parents[4]
V1 = REPO / "paper" / "prereg" / "market-labels-v1.toml"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIRST, DAYS = date(2023, 12, 4), 35  # a Monday, to 2024-01-07: two weeks of warm-up, three labelled
# the frozen v1 file shrunk to the fake archive, every other line kept
SHRUNK = {
    "start": "2023-12-04",
    "end": "2024-01-07",
    "warmup_days": "14",
    "baseline_days": "14",
    "d_min": "6",
    "trailing_days": "7",
}
SRC = Path("python") / "tipping_point" / "src" / "tipping_point"
# The cascades planted on BTCUSDT (_plant), as (day, minute): from 21:00 on the last pilot day, back
# by 23:00, and from 00:00 on the first confirmatory day, under 24 hours later
LATE, EARLY = (date(2023, 12, 31), 21 * 60), (date(2024, 1, 1), 0)
KINDS = ("klines", "metrics")  # the files a plant rewrites
MIDNIGHT_NS = archive.day_ms(market_cli.CONFIRMATORY) * 1_000_000
MINUTE_NS = 60 * 10**9


@pytest.fixture(scope="module", autouse=True)
def no_network() -> Iterator[None]:
    """For the whole module, so the archive fixture runs under it too."""

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("a test tried the network")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", refuse)
        patch.setattr(archive, "http_get", refuse)
        yield


def _unzipped(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        (member,) = bundle.namelist()
        return bundle.read(member).decode()


def _plant(fake: Vision, day: date, start: int, back: int | None = None) -> None:
    """In place, a long cascade in BTCUSDT's files of `day` from minute `start`, a bar's start: over
    its 30 minutes the price and the open interest fall 10 % in even steps, on eight times the
    volume, a fifth of it bought by takers. From minute `back` both climb back as evenly, and from
    then on the day is as it was. Each price is moved as at the end of its minute and each open
    interest as at its stamp, as test_cascade's plant moves bars."""

    def shift(minute: int) -> float:  # the log move by that instant, in minutes from midnight
        move = -0.1 * min(max((minute - start) / 30, 0.0), 1.0)
        return move if back is None else move + 0.1 * min(max((minute - back) / 30, 0.0), 1.0)

    klines, metrics = (archive.Source(kind, "BTCUSDT", f"{day:%Y-%m-%d}") for kind in KINDS)
    head, *rows = _unzipped(fake.files[klines.url]).splitlines()
    out = [head]
    for m, row in enumerate(rows):
        f = row.split(",")
        f[1:5] = [f"{float(price) * math.exp(shift(m + 1)):.2f}" for price in f[1:5]]
        if start <= m < start + 30:
            volume = 8 * float(f[5])
            f[5], f[9] = f"{volume:.3f}", f"{0.2 * volume:.3f}"
        out.append(",".join(f))
    fake.add(klines, ("\n".join(out) + "\n").encode())
    head, *rows = _unzipped(fake.files[metrics.url]).splitlines()
    out = [head]
    for step, row in enumerate(rows):
        f = row.split(",")
        f[2] = f"{float(f[2]) * math.exp(shift(5 * step)):.8f}"
        out.append(",".join(f))
    fake.add(metrics, ("\n".join(out) + "\n").encode())


@pytest.fixture(scope="module")
def fetched(no_network: None, tmp_path_factory: pytest.TempPathFactory) -> Path:
    fake = vision(SYMBOLS, FIRST, DAYS, seed=29)
    _plant(fake, *LATE, back=LATE[1] + 90)
    _plant(fake, *EARLY)
    dest = tmp_path_factory.mktemp("archive")
    span = ["--start", FIRST.isoformat(), "--end", "2024-01-07", "--dest", str(dest)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(archive, "http_get", fake.get)
        assert market_cli.main(["fetch", *span]) == 0
    return dest


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch repository frozen at prereg/market-labels-v1: the shrunk v1 file, the real
    .gitattributes, a uv.lock, and a stand-in for each held and clean file and for one more module
    the check does not read; market_cli.SRC points at its tipping_point/ folder."""
    root = tmp_path / "repo"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Test"),
        ("config", "user.email", "test@example.com"),
        ("config", "core.autocrlf", "false"),
    ):
        _git(root, *args)
    text = V1.read_text(encoding="utf-8")
    for key, value in SHRUNK.items():
        text, n = re.subn(rf"(?m)^{key} = \S+", f"{key} = {value}", text)
        assert n == 1, key
    path = root / "paper" / "prereg" / "market-labels-v1.toml"
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode())
    (root / ".gitattributes").write_bytes((REPO / ".gitattributes").read_bytes())
    (root / "uv.lock").write_bytes(b"# the locked libraries\n")
    for name in (*market_cli.HELD, *market_cli.CLEAN, "market/reports.py"):
        (root / SRC / name).parent.mkdir(parents=True, exist_ok=True)
        (root / SRC / name).write_bytes(f"# {name}\n".encode())
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "market-labels-v1")
    assert prereg.freeze(root, "market-labels") == "prereg/market-labels-v1"
    monkeypatch.setattr(market_cli, "SRC", root / SRC)
    return path


def _run(command: str, path: Path, fetched: Path, data: Path, *extra: str) -> int:
    where = ["--archive", str(fetched), "--data-dir", str(data)]
    return market_cli.main([command, "--prereg", str(path), *where, *extra])


def _run_dir(data: Path) -> Path:
    (run,) = (data / "labels").iterdir()
    return run


def _manifest(data: Path) -> dict[str, Any]:
    return json.loads((_run_dir(data) / "manifest.json").read_text(encoding="utf-8"))


def _stamp(path: Path) -> dict[str, Any]:
    pre = prereg.load(path)
    return prereg.stamp(pre, prereg.frozen(pre))


def test_labels_take_everything_from_the_frozen_file_and_stamp_the_run(
    frozen: Path, fetched: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    assert _run("labels", frozen, fetched, data) == 0
    params = _manifest(data)["params"]
    doc = prereg.load(frozen).doc
    assert (params["start"], params["end"]) == ("2023-12-04", "2024-01-07")
    assert params["symbols"] == doc["data"]["symbols"] == SYMBOLS
    assert params["labeller"] == doc["labeller"]
    assert params["labeller"]["d_min"] == 6 and params["labeller"]["enter"] == 0.8
    assert params["prereg"] == _stamp(frozen)
    assert params["prereg"]["tag"] == "prereg/market-labels-v1"
    root = frozen.parents[2]
    assert params["code"] == {
        "head": _git(root, "rev-parse", "HEAD"),
        "held": [(SRC / name).as_posix() for name in market_cli.HELD],
        "clean": [(SRC / name).as_posix() for name in market_cli.CLEAN],
        "uv_lock_sha256": hashlib.sha256(b"# the locked libraries\n").hexdigest(),
        "changed": [],
        "uncommitted": [],
    }
    assert f"{SRC.as_posix()}/market/msvol.py" in params["code"]["held"]


def test_the_held_files_are_every_market_module_the_two_commands_reach() -> None:
    """From _labels and _cascade, every module of market/ reached through the names they use, the
    __main__ functions they call, the modules they import and the functions and classes taken
    from those, plus the package's __init__, which importing any of them runs: exactly HELD. A
    module the commands do not reach, such as reports, may change without refusing a run."""
    package, names = "tipping_point.market.", vars(market_cli)
    todo: list[Any] = [market_cli._labels, market_cli._cascade]
    called, reached = set(), set()
    while todo:
        value = todo.pop()
        if inspect.ismodule(value):
            name = value.__name__
        elif callable(value):
            name = getattr(value, "__module__", None)
        else:
            continue
        if name == market_cli.__name__ and inspect.isfunction(value):
            if value not in called:
                called.add(value)
                todo += [names[n] for n in value.__code__.co_names if n in names]
        elif isinstance(name, str) and name.startswith(package) and name not in reached:
            reached.add(name)
            todo += vars(sys.modules[name]).values()
    files = {f"market/{Path(sys.modules[name].__file__ or '').name}" for name in reached}
    assert sorted({*files, "market/__init__.py"}) == sorted(market_cli.HELD)
    assert f"{package}reports" not in reached and market_cli._settings in called


def test_the_real_v1_file_is_frozen_and_this_checkouts_held_modules_are_its_tags() -> None:
    """The real run's check on this checkout, reading no data: the committed file is the one at
    prereg/market-labels-v1 and every held module is the tag's commit's. Whether the clean files
    are committed is left to the run, since this may run on a checkout being edited."""
    pre = prereg.load(V1)
    try:
        ref = prereg.frozen(pre)
    except prereg.PreregError as error:  # a clone without the tag: the fixture tests still hold
        pytest.skip(str(error))
    code = prereg.code_check(pre, ref, market_cli.SRC, market_cli.HELD, market_cli.CLEAN)
    assert code["changed"] == [], code


def test_cascade_takes_its_block_and_the_confirmatory_span_from_the_frozen_file(
    frozen: Path, fetched: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    assert _run("cascade", frozen, fetched, data) == 0
    params = _manifest(data)["params"]
    # read from trailing_days + 1 (8) days early, which only fill the medians and percentiles of
    # the windows closing from 2024-01-01 00:00 on, the windows it lists
    assert (params["start"], params["end"]) == ("2023-12-24", "2024-01-07")
    assert params["cascade"] == prereg.load(frozen).doc["cascade"]
    assert params["prereg"] == _stamp(frozen)


def test_the_cascade_list_holds_the_confirmatory_days_first_hour_and_nothing_before_it(
    frozen: Path, fetched: Path, tmp_path: Path
) -> None:
    """Every window closing from 2024-01-01 00:00 on is evaluated, its trailing days whole: the
    cascade planted from 00:00 is listed from its first firing, inside the hour that a read from
    only trailing_days early left without history. The one planted on 2023-12-31 neither enters
    the list nor takes the other in, though they fire under 24 hours apart."""
    data = tmp_path / "data"
    assert _run("cascade", frozen, fetched, data) == 0
    run = _run_dir(data)
    rows = engine.read(data, "labels", run.name, "cascades").to_pylist()
    assert [(r["symbol"], r["kind"], r["ts_ns"]) for r in rows] == [
        ("BTCUSDT", cascade.LONG, MIDNIGHT_NS)
    ]
    assert MIDNIGHT_NS < rows[0]["fired_ts_ns"] < MIDNIGHT_NS + 55 * MINUTE_NS
    checks = json.loads((run / "complete.json").read_text(encoding="utf-8"))["checks"]
    for symbol in SYMBOLS:  # the 7 days 2024-01-01..07, each window once
        assert checks[symbol]["windows"] == 7 * bars.DAY_BARS
        assert checks[symbol]["windows_no_history"] == 0


@pytest.mark.parametrize(
    ("command", "extra", "named"),
    [
        ("labels", ["--d-min", "36"], "drop --d-min"),
        ("labels", ["--d-min", "6"], "drop --d-min"),  # the file's own value: still refused
        ("labels", ["--start", "2023-12-04"], "drop --start"),
        ("labels", ["--symbols", "BTCUSDT"], "drop --symbols"),
        ("cascade", ["--b", "0.025", "--end", "2024-01-07"], "drop --end, --b"),
    ],
)
def test_an_option_beside_the_file_is_refused(
    frozen: Path,
    fetched: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    extra: list[str],
    named: str,
) -> None:
    data = tmp_path / "data"
    assert _run(command, frozen, fetched, data, *extra) == 1
    assert named in capsys.readouterr().err and not data.exists()


def test_a_file_changed_since_its_tag_is_refused(
    frozen: Path, fetched: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frozen.write_bytes(frozen.read_bytes().replace(b"d_min = 6", b"d_min = 12"))
    data = tmp_path / "data"
    assert _run("labels", frozen, fetched, data) == 1
    assert "differs from the version frozen at prereg/market-labels-v1" in capsys.readouterr().err
    assert not data.exists()


def test_an_unfrozen_file_is_refused(
    frozen: Path, fetched: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(frozen.parents[2], "tag", "-d", "prereg/market-labels-v1")
    assert _run("cascade", frozen, fetched, tmp_path / "data") == 1
    assert "market-labels-v1 is not frozen" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("held", "change"),
    [
        *((name, "edited") for name in market_cli.HELD),
        ("market/msvol.py", "committed"),  # a later commit: the tag's commit counts, not HEAD
        ("market/cascade.py", "deleted"),
    ],
)
def test_a_held_module_that_is_not_the_tags_commits_is_refused(
    frozen: Path,
    fetched: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    held: str,
    change: str,
) -> None:
    root = frozen.parents[2]
    target = root / SRC / held
    if change == "deleted":
        target.unlink()
    else:
        target.write_bytes(b"# changed\n")
    if change == "committed":
        _git(root, "commit", "-q", "-am", "change the labeller")
    for command in ("labels", "cascade"):
        data = tmp_path / command
        assert _run(command, frozen, fetched, data) == 1
        err = capsys.readouterr().err
        assert "these files differ from prereg/market-labels-v1's commit" in err
        assert f"{(SRC / held).as_posix()}; the labels" in err and not data.exists()


def test_an_edit_the_index_cannot_see_is_refused(
    frozen: Path, fetched: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each held file is read and hashed: an edit that keeps the file's size and modification
    time, which git's cached stat then takes for no change, is refused all the same."""
    root = frozen.parents[2]
    target = root / SRC / "market" / "msvol.py"
    _git(root, "config", "core.trustctime", "false")  # so the stat matches on every platform
    hour_ago = target.stat().st_mtime_ns - 3_600 * 10**9
    os.utime(target, ns=(hour_ago, hour_ago))
    _git(root, "update-index", "--refresh")  # the index caches that stat, too old to be racy
    target.write_bytes(target.read_bytes().upper())  # the same size
    os.utime(target, ns=(hour_ago, hour_ago))
    assert _git(root, "status", "--porcelain") == ""  # git's own look misses it
    assert _run("labels", frozen, fetched, tmp_path / "data") == 1
    assert f"{SRC.as_posix()}/market/msvol.py; the labels" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["new module", "unheld module edited", "CRLF line ends"])
def test_files_a_run_does_not_read_may_change_and_line_ends_do_not_count(
    frozen: Path, fetched: Path, tmp_path: Path, change: str
) -> None:
    """A new file in market/, such as #30's book.py, and an edit to a module the commands do not
    reach leave the run as it was; so does a held file checked out with CRLF, which is still the
    LF blob the tag holds."""
    market = frozen.parents[2] / SRC / "market"
    if change == "new module":
        (market / "book.py").write_bytes(b"# the order book\n")
    elif change == "unheld module edited":
        (market / "reports.py").write_bytes(b"# the reports, changed\n")
    else:
        held = market / "msvol.py"
        held.write_bytes(held.read_bytes().replace(b"\n", b"\r\n"))
    data = tmp_path / "data"
    assert _run("cascade", frozen, fetched, data) == 0
    assert _manifest(data)["params"]["code"]["changed"] == []


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("clean", market_cli.CLEAN)
def test_an_uncommitted_change_to_a_clean_file_is_refused(
    frozen: Path,
    fetched: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    clean: str,
    staged: bool,
) -> None:
    """The command line and prereg.py may differ from the tag's commit, but only as committed, so
    that the manifest's code.head pins them."""
    root = frozen.parents[2]
    (root / SRC / clean).write_bytes(b"# changed, not committed\n")
    if staged:
        _git(root, "add", (SRC / clean).as_posix())
    data = tmp_path / "data"
    assert _run("labels", frozen, fetched, data) == 1
    err = capsys.readouterr().err
    assert f"changes not committed in {(SRC / clean).as_posix()}: commit them" in err
    assert not data.exists()


def test_a_committed_change_to_the_command_line_runs_and_head_pins_it(
    frozen: Path, fetched: Path, tmp_path: Path
) -> None:
    """__main__.py gained --prereg after the freeze: committed, it runs, and code.head names the
    commit that holds it."""
    root = frozen.parents[2]
    (root / SRC / "market" / "__main__.py").write_bytes(b"# the command line, with --prereg\n")
    _git(root, "commit", "-q", "-am", "the command line gains --prereg")
    data = tmp_path / "data"
    assert _run("cascade", frozen, fetched, data) == 0
    code = _manifest(data)["params"]["code"]
    head = _git(root, "rev-parse", "HEAD")
    assert code["head"] == head != prereg.frozen(prereg.load(frozen)).commit
    assert (code["changed"], code["uncommitted"]) == ([], [])


def test_code_outside_the_files_repository_is_refused(
    frozen: Path,
    fetched: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(market_cli, "SRC", tmp_path / "elsewhere")
    assert _run("labels", frozen, fetched, tmp_path / "data") == 1
    assert "where market-labels-v1 is frozen" in capsys.readouterr().err
