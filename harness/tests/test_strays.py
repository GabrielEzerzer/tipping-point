"""The strays sweep (#61): which processes it kills, and that a dry run and a reused PID are safe.

Fixture process lists only: no test here kills a real process.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import strays
from strays import Process

MIN = 60.0


def _fixture(root: Path) -> tuple[list[Path], list[Process], list[Process]]:
    """The main checkout's and a worktree's target dirs, then the strays among a process list,
    then everything else on it."""
    main, lane = root / "code" / "target", root / ".worktrees" / "r2a" / "target"
    hash16 = "0123456789abcdef"
    found = [
        Process(101, lane / "debug" / "deps" / "api_load-8652aba3279d7f40.exe", 41 * MIN),
        Process(102, main / "debug" / "serve.exe", 11 * MIN),
        Process(103, lane / "debug" / "crash-writer.exe", 30 * MIN),
        Process(104, main / "debug" / "deps" / f"crash_writer-{hash16}.exe", 30 * MIN),
        Process(105, lane / "debug" / "deps" / f"api_python-{hash16}.exe", 12 * MIN),
        Process(106, main / "debug" / "store.exe", 90 * MIN),
        Process(107, main / "debug" / "ingest.exe", 90 * MIN),
        Process(108, main / "debug" / "record.exe", 90 * MIN),
        Process(109, main / "debug" / "soak.exe", 90 * MIN),
    ]
    spared = [
        Process(201, main / "release" / "serve.exe", 300 * MIN),  # a recording session
        Process(202, main / "release" / "soak.exe", 1440 * MIN),  # the 24-hour soak
        Process(203, lane / "debug" / "deps" / "api_load-8652aba3279d7f40.exe", 9 * MIN),
        Process(204, root / "other-repo" / "target" / "debug" / "serve.exe", 60 * MIN),
        Process(205, main / "debug" / "deps" / f"api_parity-{hash16}.exe", 60 * MIN),
        Process(206, main / "debug" / "build" / "cargo.exe", 60 * MIN),
        Process(207, root / "code" / ".venv" / "Scripts" / "python.exe", 60 * MIN),
        Process(208, main / "debug" / "serve-extra.exe", 60 * MIN),
        Process(209, root / "code" / "target2" / "debug" / "serve.exe", 60 * MIN),
    ]
    return [main, lane], found, spared


def test_only_old_debug_engine_processes_of_this_repo_are_strays(tmp_path: Path) -> None:
    """Spared: release builds (the soak, a recording), a test still within ten minutes, another
    repo's or a look-alike dir's binaries, binaries not named, and every other program."""
    targets, found, spared = _fixture(tmp_path)
    assert strays.find(spared + found, targets) == found


@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths ignore case")
def test_a_path_typed_in_another_case_is_the_same_dir(tmp_path: Path) -> None:
    target = tmp_path / "code" / "target"
    shouted = Path(str(target).upper()) / "DEBUG" / "DEPS" / "API_LOAD-8652ABA3279D7F40.EXE"
    assert strays.find([Process(1, shouted, 60 * MIN)], [target]) == [Process(1, shouted, 60 * MIN)]


def _git(cwd: Path, *argv: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run(["git", *identity, *argv], cwd=cwd, check=True, capture_output=True)


def test_the_target_dirs_are_the_checkouts_and_every_worktrees(tmp_path: Path) -> None:
    repo, lane = tmp_path / "code", tmp_path / "lane"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "init")
    _git(repo, "worktree", "add", "-q", str(lane))
    found = [d.resolve() for d in strays.target_dirs(lane)]
    assert found == [(repo / "target").resolve(), (lane / "target").resolve()]
    outside = tmp_path / "not-a-checkout"
    outside.mkdir()
    assert strays.target_dirs(outside) == [outside / "target"]


def test_a_dry_run_kills_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets, found, spared = _fixture(tmp_path)

    def never(process: Process) -> str:
        raise AssertionError(f"a dry run killed {process.pid}")

    monkeypatch.setattr(strays, "list_processes", lambda: spared + found)
    monkeypatch.setattr(strays, "target_dirs", lambda root: targets)
    monkeypatch.setattr(strays, "kill", never)
    rows = strays.sweep(dry_run=True)
    assert rows is not None
    assert [row["pid"] for row in rows] == [p.pid for p in found]
    assert {row["action"] for row in rows} == {"would kill"}


def test_a_kill_takes_the_children_and_spares_a_reused_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def taskkill(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        said = "SUCCESS: The process with PID 101 (child process of PID 9) has been terminated.\n"
        return subprocess.CompletedProcess(argv, 0, stdout=said, stderr="")

    monkeypatch.setattr(strays.subprocess, "run", taskkill)
    exe = Path("target") / "debug" / "deps" / "api_load-8652aba3279d7f40.exe"
    said = strays.kill(Process(101, exe, 60 * MIN))
    assert said == "SUCCESS: The process with PID 101 (child process of PID 9) has been terminated."
    assert calls == [
        ["taskkill", "/F", "/T", "/PID", "101", "/FI", "IMAGENAME eq api_load-8652aba3279d7f40.exe"]
    ]


def test_the_command_line_dry_run_reaches_the_sweep(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[bool] = []

    def sweep(root: Path = strays.ROOT, dry_run: bool = False) -> list[dict[str, object]]:
        asked.append(dry_run)
        return []

    monkeypatch.setattr(strays, "sweep", sweep)
    assert strays.main(["--dry-run"]) == 0
    assert asked == [True]
    assert capsys.readouterr().out == "strays: none\n"


def test_where_processes_cannot_be_listed_the_sweep_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(strays, "list_processes", lambda: None)
    assert strays.sweep() is None
    assert strays.describe(None) == [f"cannot list processes on {sys.platform}"]


@pytest.mark.skipif(sys.platform != "win32", reason="processes are listed on Windows only")
def test_the_listing_names_each_process_by_executable_and_age() -> None:
    processes = strays.list_processes()
    assert processes is not None
    (me,) = [p for p in processes if p.pid == os.getpid()]
    assert me.exe.is_absolute() and me.exe.name.lower().startswith("python")
    assert 0 <= me.age < 86400
