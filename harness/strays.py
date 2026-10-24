"""Strays: engine and harness processes that a killed test run left behind (#61).

Run as `uv run python harness/strays.py [--dry-run]`; the gate runs it as it starts and ends.

A stray is a process of an engine or harness binary (serve, store, ingest, record, soak,
crash-writer) or of a test that spawns them (api_load, api_python) whose executable lives in a
debug target dir of this repository, the main checkout's or any worktree's, and that has run for
more than ten minutes. A test run takes minutes; one killed by a tool watchdog or a rate-limit
stop leaves its children running, each holding a core. Release builds are never touched: the
long jobs (a soak, a recording session) run from target/release.

Lists processes on Windows only (elsewhere the sweep says it cannot). Stdlib only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_AGE = 600.0  # seconds a process must have run to count as a stray
# each binary as built (crash-writer.exe) and as its own test binary (deps/crash_writer-<hash>.exe)
NAMES = re.compile(
    r"(serve|store|ingest|record|soak|crash[-_]writer|api_load|api_python)(-[0-9a-f]{16})?(\.exe)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Process:
    pid: int
    exe: Path
    age: float  # seconds since it started


def target_dirs(root: Path = ROOT) -> list[Path]:
    """The target dir of the main checkout and of every worktree, as git lists them."""
    listed = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    trees = [
        ln.removeprefix("worktree ") for ln in listed.splitlines() if ln.startswith("worktree ")
    ]
    return [Path(tree) / "target" for tree in trees] or [root / "target"]


def _in_debug(exe: Path, target: Path) -> bool:
    return exe.is_relative_to(target) and exe.relative_to(target).parts[0].lower() == "debug"


def find(
    processes: Iterable[Process], targets: Sequence[Path], min_age: float = MIN_AGE
) -> list[Process]:
    """The strays among `processes`: a named binary in a target dir's debug profile that has run
    longer than `min_age` seconds."""
    return [
        p
        for p in processes
        if p.age > min_age
        and NAMES.fullmatch(p.exe.name)
        and any(_in_debug(p.exe, target) for target in targets)
    ]


def list_processes() -> list[Process] | None:
    """Every process whose executable and start time this account may read; None where listing
    is not built (anywhere but Windows)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]

    def ticks(t: wintypes.FILETIME) -> int:  # 100 ns since 1601
        return (t.dwHighDateTime << 32) | t.dwLowDateTime

    pids, needed = (wintypes.DWORD * 1024)(), wintypes.DWORD()
    while True:
        if not k32.K32EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value < ctypes.sizeof(pids):  # a full array may have cut the list short
            break
        pids = (wintypes.DWORD * (2 * len(pids)))()
    now = wintypes.FILETIME()
    k32.GetSystemTimeAsFileTime(ctypes.byref(now))
    processes = []
    for pid in pids[: needed.value // ctypes.sizeof(wintypes.DWORD)]:
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            continue  # a system process, or gone since the listing
        try:
            path, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
            times = [wintypes.FILETIME() for _ in range(4)]  # created, exited, kernel, user
            if k32.QueryFullProcessImageNameW(
                handle, 0, path, ctypes.byref(size)
            ) and k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                age = (ticks(now) - ticks(times[0])) / 1e7
                processes.append(Process(pid, Path(path.value), age))
        finally:
            k32.CloseHandle(handle)
    return processes


def kill(process: Process) -> str:
    """Kill a stray and every process it started; taskkill's own words for what happened. The
    image-name filter spares a process that took the PID over since the listing."""
    done = subprocess.run(
        [
            "taskkill",
            "/F",
            "/T",
            "/PID",
            str(process.pid),
            "/FI",
            f"IMAGENAME eq {process.exe.name}",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    words = (done.stdout + done.stderr).split("\n")
    return "; ".join(w.strip() for w in words if w.strip()) or f"taskkill exit {done.returncode}"


def sweep(root: Path = ROOT, dry_run: bool = False) -> list[dict[str, object]] | None:
    """Find the strays and, unless `dry_run`, kill them: one row per stray saying what was done;
    None where processes cannot be listed."""
    processes = list_processes()
    if processes is None:
        return None
    return [
        {
            "pid": p.pid,
            "exe": str(p.exe),
            "minutes": round(p.age / 60, 1),
            "action": "would kill" if dry_run else kill(p),
        }
        for p in find(processes, target_dirs(root))
    ]


def describe(rows: list[dict[str, object]] | None) -> list[str]:
    """A line per stray, or one line saying there were none."""
    if rows is None:
        return [f"cannot list processes on {sys.platform}"]
    return [f"{r['exe']} (pid {r['pid']}, {r['minutes']} min): {r['action']}" for r in rows] or [
        "none"
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list the strays, kill nothing")
    args = parser.parse_args(argv)
    for line in describe(sweep(dry_run=args.dry_run)):
        print(f"strays: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
