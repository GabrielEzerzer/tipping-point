"""Wheel provenance end to end (#109), run by hand: a wheel's runs record the commit stamped into
it at build time, in real venvs, and tauc accepts them.

    uv run python harness/wheel_provenance.py                  # a wheel built from this checkout
    uv run python harness/wheel_provenance.py --wheel <whl>    # a given wheel, say before Kaggle

1. outside: the wheel in a venv in a temporary folder outside any repository. Pilots at Tc
   (L = 8 and 4) record the stamped commit, dirty flag and kernel, and tauc writes a record that
   names them.
2. inside: the wheel in a venv inside a scratch git repository with a commit of its own. A run
   records the stamped commit, never that repository's.
3. checkout: this checkout's own install (editable, `uv sync`). A run records git's HEAD and
   status, as before #109.

One line per check, then exit 1 if any failed. About three minutes on a laptop CPU; every
dependency comes from the uv cache that `uv sync` filled. A wheel built before #109, or where no
repository tracked the source, fails 1 and 2. python/tipping_point/tests/ising/test_provenance.py
is the gate's quick version.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "python" / "tipping_point"
STAMP = "tipping_point/_build_info.py"
TC = "2.269185314213022"  # repr(Tc): --T must parse back to exactly Tc
PILOTS = (("8", "6000"), ("4", "3000"))  # (L, sweeps): >= 300 tau_c each at 32 replicas
TINY = ["--L", "4", "--T", "3.0", "--replicas", "2", "--therm", "10", "--sweeps", "50"]
REPORT = (
    "import json; from tipping_point.ising import io; "
    "print(json.dumps([io.__file__, io.code_version()]))"
)
ENV = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
results: list[bool] = []


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wheel", type=Path, help="default: build one from this checkout")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tp-wheel-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        if _git(work, "rev-parse", "--show-toplevel") is not None:
            raise SystemExit(f"{work} is inside a git repository: point TMP outside one")
        wheel = args.wheel.resolve() if args.wheel else _build(work / "dist")
        stamp = _stamp(wheel)
        print(f"{wheel.name}: {stamp}", flush=True)
        requirements = work / "requirements.txt"
        _uv(
            *("export", "--frozen", "--extra", "cpu", "--no-dev", "--no-emit-workspace"),
            *("--no-hashes", "--no-header", "--no-annotate", "--quiet", "-o", str(requirements)),
        )
        _outside(work / "outside", wheel, requirements, stamp)
        _inside(work / "other-project", wheel, requirements, stamp)
        _checkout(work / "checkout")
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


def _outside(folder: Path, wheel: Path, requirements: Path, stamp: dict[str, Any]) -> None:
    python = _venv(folder / "venv", wheel, requirements)
    data = folder / "data"
    for L, sweeps in PILOTS:
        argv = ["pilot", "--L", L, "--T", TC, "--replicas", "32", "--therm", "300"]
        _ising(python, folder, *argv, "--sweeps", sweeps, "--data-dir", str(data))
    runs = sorted((data / "ising").glob("*-pilot-s1"))
    for run in runs:
        producer, kernel = _manifest(run)
        _check(
            f"outside: {run.name} records the stamp",
            stamp["commit"] is not None
            and (producer["git_sha"], producer["dirty"]) == (stamp["commit"], stamp["dirty"])
            and kernel == stamp["kernel_sha256"],
            f"git_sha {producer['git_sha']}, dirty {producer['dirty']}, kernel {str(kernel)[:12]}",
        )
    tauc = _ising(
        python, folder, "tauc", *map(str, runs), "--data-dir", str(data), "--draws", "200"
    )
    records = sorted((data / "ising").glob("*-tauc-*/pilot.json"))
    record = json.loads(records[0].read_text(encoding="utf-8")) if records else {}
    named = stamp["commit"] and stamp["commit"] + ("-dirty" if stamp["dirty"] else "")
    _check(
        "outside: tauc accepts the runs and names the stamp",
        tauc.returncode == 0
        and named is not None
        and record.get("kernel_commit") == named
        and record.get("kernel_sha256") == stamp["kernel_sha256"],
        f"exit {tauc.returncode}, kernel_commit {record.get('kernel_commit')}"
        + ("" if records else ": " + (tauc.stderr.strip().splitlines() or [""])[-1]),
    )


def _inside(repo: Path, wheel: Path, requirements: Path, stamp: dict[str, Any]) -> None:
    repo.mkdir()
    for argv in (
        ["init", "-q"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@example.com"],
    ):
        _git(repo, *argv)
    (repo / "README").write_text("another project\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "another project")
    head = _git(repo, "rev-parse", "HEAD")
    if head is None:
        raise SystemExit(f"{repo}: the scratch repository has no commit of its own")
    python = _venv(repo / ".venv", wheel, requirements)
    where, _ = _report(python, repo)
    _ising(python, repo, "pilot", *TINY, "--device", "cpu", "--data-dir", str(repo / "data"))
    (run,) = (repo / "data" / "ising").glob("*-pilot-s1")
    producer, _ = _manifest(run)
    _check(
        "inside: a run records the stamp, not the enclosing repository",
        Path(where).resolve().is_relative_to((repo / ".venv").resolve())
        and stamp["commit"] is not None
        and (producer["git_sha"], producer["dirty"]) == (stamp["commit"], stamp["dirty"])
        and producer["git_sha"] != head,
        f"git_sha {producer['git_sha']}, enclosing repository's HEAD {head}",
    )


def _checkout(folder: Path) -> None:
    head, status = _git(ROOT, "rev-parse", "HEAD"), _git(ROOT, "status", "--porcelain")
    python = Path(sys.executable)
    where, _ = _report(python, ROOT)
    _ising(python, ROOT, "pilot", *TINY, "--device", "cpu", "--data-dir", str(folder))
    (run,) = (folder / "ising").glob("*-pilot-s1")
    producer, _ = _manifest(run)
    _check(
        "checkout: a run records git's HEAD and status, as before",
        Path(where).resolve().is_relative_to(PROJECT / "src")
        and (producer["git_sha"], producer["dirty"]) == (head, status != ""),
        f"git_sha {producer['git_sha']}, dirty {producer['dirty']}, HEAD {head}",
    )


def _build(out: Path) -> Path:
    """The wheel as the README builds it: `uv build --package tipping-point --wheel`."""
    _uv("build", "--package", "tipping-point", "--wheel", "--out-dir", str(out), "--quiet")
    (wheel,) = out.glob("*.whl")
    return wheel


def _stamp(wheel: Path) -> dict[str, Any]:
    """The wheel's _build_info fields; all None when it has none (built before #109)."""
    fields: dict[str, Any] = {}
    with zipfile.ZipFile(wheel) as archive:
        if STAMP in archive.namelist():
            exec(archive.read(STAMP).decode(), fields)
    return {key: fields.get(key) for key in ("commit", "dirty", "kernel_sha256", "built_at")}


def _venv(path: Path, wheel: Path, requirements: Path) -> Path:
    """A venv holding the locked CPU dependencies and the wheel, installed from the uv cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _uv("venv", str(path), "--python", sys.executable, "--quiet", cwd=path.parent)
    python = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    indexes = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    (cpu,) = [i["url"] for i in indexes["tool"]["uv"]["index"] if i["name"] == "pytorch-cpu"]
    _uv(
        *("pip", "install", "--quiet", "--offline", "--python", str(python), "--no-deps"),
        *("--index-strategy", "unsafe-best-match", "--index", cpu),
        *("-r", str(requirements), str(wheel)),
        cwd=path.parent,
    )
    return python


def _report(python: Path, cwd: Path) -> tuple[str, dict[str, Any]]:
    """Where `python` imports io from, and what its code_version() says."""
    ran = subprocess.run(
        [str(python), "-c", REPORT], cwd=cwd, env=ENV, capture_output=True, text=True, check=True
    )
    where, version = json.loads(ran.stdout)
    return where, version


def _ising(python: Path, cwd: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    """`python -m tipping_point.ising <argv>`: a failed pilot stops the harness, tauc does not."""
    ran = subprocess.run(
        [str(python), "-m", "tipping_point.ising", *argv],
        cwd=cwd,
        env=ENV,
        capture_output=True,
        text=True,
    )
    if ran.returncode != 0 and argv[0] != "tauc":
        raise SystemExit(f"{' '.join(argv[:3])} failed:\n{ran.stderr}")
    return ran


def _manifest(run: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    return manifest["producer"], manifest["params"]["algorithm"]["kernel_sha256"]


def _check(name: str, passed: bool, detail: str) -> None:
    results.append(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)


def _uv(*argv: str, cwd: Path = ROOT) -> None:
    subprocess.run(["uv", *argv], cwd=cwd, check=True)


def _git(cwd: Path, *argv: str) -> str | None:
    ran = subprocess.run(["git", *argv], cwd=cwd, capture_output=True, text=True)
    return ran.stdout.strip() if ran.returncode == 0 else None


if __name__ == "__main__":
    sys.exit(main())
