"""The strays sweep (#61): which processes it kills, and that a dry run and a reused PID are safe.

Fixture process lists only: no test here kills a real process.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import strays
from strays import Process

MIN = 60.0
HASH = "0123456789abcdef"


def _fixture(root: Path) -> tuple[list[Path], list[Process], list[Process]]:
    """The main checkout's and a worktree's target dirs, then the strays among a process list,
    then everything else on it. PIDs from 9000 up are gone; 300 is a live shell, 301 a live cargo
    run, 302 a new process that took over a dead parent's PID, 303 one this account cannot read;
    600 is the bash a killed background Git Bash forked, still alive under its gone parent, 601
    the cargo it ran, 602 a test binary whose cargo is gone; 604 is a live terminal (mintty)
    running bash 603, which runs cargo 605."""
    main, lane = root / "code" / "target", root / ".worktrees" / "r2a" / "target"
    git_bin = root / "git" / "usr" / "bin"
    found = [
        Process(102, 9001, main / "debug" / "serve.exe", 11 * MIN),
        Process(103, 302, lane / "debug" / "crash-writer.exe", 30 * MIN),  # parent PID reused
        Process(104, 9002, main / "debug" / "deps" / f"crash_writer-{HASH}.exe", 30 * MIN),
        Process(105, 9003, lane / "debug" / "deps" / f"api_python-{HASH}.exe", 12 * MIN),
        Process(106, 9004, main / "debug" / "store.exe", 90 * MIN),
        Process(107, 9004, main / "debug" / "ingest.exe", 90 * MIN),
        Process(108, 9004, main / "debug" / "record.exe", 90 * MIN),
        Process(109, 9004, main / "debug" / "soak.exe", 90 * MIN),
        Process(110, 601, main / "debug" / "serve.exe", 35 * MIN),  # under a dead shell's cargo
        Process(111, 602, lane / "debug" / "crash-writer.exe", 30 * MIN),  # its test's cargo gone
    ]
    spared = [
        Process(300, 9000, root / "bin" / "pwsh.exe", 600 * MIN),
        Process(301, 300, root / "bin" / "cargo.exe", 50 * MIN),
        Process(302, 300, root / "bin" / "notepad.exe", 5 * MIN),
        Process(303, 4, Path(), math.inf),
        Process(600, 9009, git_bin / "bash.exe", 120 * MIN),
        Process(601, 600, root / "bin" / "cargo.exe", 40 * MIN),
        Process(602, 9010, lane / "debug" / "deps" / f"crash_recovery-{HASH}.exe", 40 * MIN),
        Process(603, 604, git_bin / "bash.exe", 100 * MIN),
        Process(604, 9011, root / "git" / "usr" / "bin" / "mintty.exe", 200 * MIN),
        Process(605, 603, root / "bin" / "cargo.exe", 50 * MIN),
        Process(214, 605, main / "debug" / "serve.exe", 30 * MIN),  # a live terminal's cargo run
        Process(201, 300, main / "debug" / "serve.exe", 90 * MIN),  # `serve --record` by hand
        Process(202, 301, main / "debug" / "serve.exe", 20 * MIN),  # under a live cargo run
        Process(203, 303, main / "debug" / "store.exe", 60 * MIN),  # a parent it cannot read
        Process(204, 9005, lane / "debug" / "deps" / f"api_load-{HASH}.exe", 41 * MIN),
        Process(205, 9006, main / "release" / "serve.exe", 300 * MIN),  # a recording session
        Process(206, 9006, main / "release" / "soak.exe", 1440 * MIN),  # the 24-hour soak
        Process(207, 9007, lane / "debug" / "deps" / f"api_python-{HASH}.exe", 9 * MIN),
        Process(208, 9008, root / "other-repo" / "target" / "debug" / "serve.exe", 60 * MIN),
        Process(209, 9008, main / "debug" / "deps" / f"api_parity-{HASH}.exe", 60 * MIN),
        Process(210, 9008, main / "debug" / "build" / "cargo.exe", 60 * MIN),
        Process(211, 9008, root / "code" / ".venv" / "Scripts" / "python.exe", 60 * MIN),
        Process(212, 9008, main / "debug" / "serve-extra.exe", 60 * MIN),
        Process(213, 9008, root / "code" / "target2" / "debug" / "serve.exe", 60 * MIN),
    ]
    return [main, lane], found, spared


def test_only_old_orphaned_debug_engine_processes_of_this_repo_are_strays(tmp_path: Path) -> None:
    """Swept too: a serve whose cargo and bash still run under a killed shell, and one under a
    live test binary whose cargo is gone. Spared: a process whose owner lives (a debug serve
    started from a shell, one under a live cargo run, one a live terminal's bash and cargo run,
    one whose parent this account cannot read), the api_load test, release builds (the soak, a
    recording), a test within ten minutes, another repo's or a look-alike dir's binaries,
    binaries not named, and every other program."""
    targets, found, spared = _fixture(tmp_path)
    assert strays.find(spared + found, targets) == found


def test_a_long_api_load_run_is_never_swept(tmp_path: Path) -> None:
    # TP_LOAD_SECONDS=1800 runs it for half an hour, its server in-process: nothing to sweep
    target = tmp_path / "target"
    for parent in (9, 1):  # orphaned, or under a live cargo
        processes = [
            Process(1, 0, tmp_path / "cargo.exe", 60 * MIN),
            Process(2, parent, target / "debug" / "deps" / f"api_load-{HASH}.exe", 30 * MIN),
        ]
        assert strays.find(processes, [target]) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths ignore case")
def test_a_path_typed_in_another_case_is_the_same_dir(tmp_path: Path) -> None:
    target = tmp_path / "code" / "target"
    shouted = Path(str(target).upper()) / "DEBUG" / "DEPS" / f"CRASH_WRITER-{HASH.upper()}.EXE"
    orphan = Process(1, 9, shouted, 60 * MIN)
    assert strays.find([orphan], [target]) == [orphan]


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
    exe = Path("target") / "debug" / "deps" / f"crash_writer-{HASH}.exe"
    said = strays.kill(Process(101, 9, exe, 60 * MIN))
    assert said == "SUCCESS: The process with PID 101 (child process of PID 9) has been terminated."
    assert calls == [
        ["taskkill", "/F", "/T", "/PID", "101", "/FI", f"IMAGENAME eq crash_writer-{HASH}.exe"]
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
def test_the_listing_names_each_process_by_parent_executable_and_age() -> None:
    processes = strays.list_processes()
    assert processes is not None
    by_pid = {p.pid: p for p in processes}
    me = by_pid[os.getpid()]
    assert me.exe.is_absolute() and me.exe.name.lower().startswith("python")
    assert 0 <= me.age < 86400
    assert by_pid[me.parent].age >= me.age  # the launcher that started this process lives
