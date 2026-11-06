"""The confirmatory span through the frozen file only (#29): `labels` and `cascade` with --prereg
take the span, the symbols and every parameter from a frozen market-labels file and stamp the run's
manifest with it, and refuse a file changed since its tag, an option given beside it, and labelling
code that differs from the tag's commit. A scratch repository holds the file, frozen as the real
one is, and a stand-in market/ folder; the archive is the fake one of test_cli, spanning the
boundary, and no socket may connect."""

import json
import re
import socket
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from market_planted import vision
from tipping_point.ews import prereg
from tipping_point.market import __main__ as market_cli
from tipping_point.market import archive

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
CODE = Path("python") / "tipping_point" / "src" / "tipping_point" / "market"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("a test tried the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(archive, "http_get", refuse)


@pytest.fixture(scope="module")
def fetched(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fake = vision(SYMBOLS, FIRST, DAYS, seed=29)
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
    """A scratch repository with the shrunk v1 file and a market/ folder, frozen at
    prereg/market-labels-v1; market_cli.CODE points at that folder."""
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
    (root / CODE).mkdir(parents=True)
    (root / CODE / "msvol.py").write_bytes(b"# the labeller\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "market-labels-v1")
    assert prereg.freeze(root, "market-labels") == "prereg/market-labels-v1"
    monkeypatch.setattr(market_cli, "CODE", root / CODE)
    return path


def _run(command: str, path: Path, fetched: Path, data: Path, *extra: str) -> int:
    where = ["--archive", str(fetched), "--data-dir", str(data)]
    return market_cli.main([command, "--prereg", str(path), *where, *extra])


def _manifest(data: Path) -> dict[str, Any]:
    (run,) = (data / "labels").iterdir()
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))


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
        "path": CODE.as_posix(),
        "skipped": ["__main__.py"],
        "changed": [],
    }


def test_the_command_line_file_may_differ_from_the_tags_commit(
    frozen: Path, fetched: Path, tmp_path: Path
) -> None:
    """__main__.py gained --prereg after the freeze; it is the one file the check leaves out."""
    root = frozen.parents[2]
    (root / CODE / "__main__.py").write_bytes(b"# the command line, with --prereg\n")
    data = tmp_path / "data"
    assert _run("labels", frozen, fetched, data) == 0
    assert _manifest(data)["params"]["code"]["changed"] == []


def test_the_real_v1_file_is_frozen_and_this_checkouts_labelling_code_is_its_tags() -> None:
    """The real run's checks on this checkout, reading no data: the committed file is the one at
    prereg/market-labels-v1 and market/, but the command line, is the tag's commit's code."""
    pre = prereg.load(V1)
    try:
        ref = prereg.frozen(pre)
    except prereg.PreregError as error:  # a clone without the tag: the fixture tests still hold
        pytest.skip(str(error))
    code = prereg.code_check(pre, ref, market_cli.CODE, skip=market_cli.CLI)
    assert code["changed"] == [], code


def test_cascade_takes_its_block_and_the_confirmatory_span_from_the_frozen_file(
    frozen: Path, fetched: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    assert _run("cascade", frozen, fetched, data) == 0
    params = _manifest(data)["params"]
    # the first trailing_days (7) only fill the medians: the list covers 2024-01-01 on
    assert (params["start"], params["end"]) == ("2023-12-25", "2024-01-07")
    assert params["cascade"] == prereg.load(frozen).doc["cascade"]
    assert params["prereg"] == _stamp(frozen)


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


@pytest.mark.parametrize("change", ["edited", "untracked", "committed"])
def test_labelling_code_that_differs_from_the_tags_commit_is_refused(
    frozen: Path,
    fetched: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: str,
) -> None:
    root = frozen.parents[2]
    if change == "untracked":
        (root / CODE / "extra.py").write_bytes(b"x = 1\n")
    else:
        (root / CODE / "msvol.py").write_bytes(b"# the labeller, changed\n")
    if change == "committed":  # a later commit: the tag's commit is what counts, not HEAD
        _git(root, "commit", "-q", "-am", "change the labeller")
    for command in ("labels", "cascade"):
        data = tmp_path / command
        assert _run(command, frozen, fetched, data) == 1
        err = capsys.readouterr().err
        assert f"{CODE.as_posix()} differs from prereg/market-labels-v1's commit" in err
        assert not data.exists()


def test_code_outside_the_files_repository_is_refused(
    frozen: Path,
    fetched: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(market_cli, "CODE", tmp_path / "elsewhere")
    assert _run("labels", frozen, fetched, tmp_path / "data") == 1
    assert "where market-labels-v1 is frozen" in capsys.readouterr().err
