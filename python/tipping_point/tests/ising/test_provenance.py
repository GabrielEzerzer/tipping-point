"""Provenance (#109): the commit every manifest names (producer.git_sha and dirty). Imported from
its own checkout, the package reads git; installed from a wheel, it reads the stamp the build hook
(python/tipping_point/hatch_build.py) wrote into the wheel; it never reads a repository that only
encloses an installed copy. The wheels are built from this checkout by uv, as a Kaggle wheel is,
and "installed" by unpacking them into a folder imported through PYTHONPATH, which then comes
before the checkout's editable install. harness/wheel_provenance.py runs the same in real venvs,
with pilots and tauc."""

import datetime
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
from tipping_point import _build_info
from tipping_point.ising import io, pilot

PROJECT = Path(__file__).resolve().parents[2]  # python/tipping_point
STAMP = "tipping_point/_build_info.py"
UV = shutil.which("uv")
needs_uv = pytest.mark.skipif(UV is None, reason="building a wheel needs uv on PATH")
# Prints where io came from and what code_version() says, then runs the CLI on any arguments.
PROBE = """
import json, sys
from tipping_point.ising import io
print(json.dumps([io.__file__, io.code_version()]))
if len(sys.argv) > 1:
    from tipping_point.ising.__main__ import main
    sys.exit(main(sys.argv[1:]))
"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _build(out: Path, *flags: str) -> None:
    assert UV is not None
    argv = [UV, "build", str(PROJECT), "--out-dir", str(out), *flags]
    built = subprocess.run(argv, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr


def _stamp(source: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    exec(source, fields)
    return {key: fields[key] for key in ("commit", "dirty", "kernel_sha256", "built_at")}


def _install(wheel: Path, site: Path) -> Path:
    """Unpack the wheel into `site`, as pip installs it into a venv's site-packages."""
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    return site


def _probe(site: Path, cwd: Path, *argv: str) -> dict[str, str | bool | None]:
    """code_version() in a fresh interpreter importing the package from `site`, which must be
    the copy it imported, never the checkout's; then the CLI on `argv`, if any."""
    env = dict(os.environ, PYTHONPATH=str(site))
    ran = subprocess.run(
        [sys.executable, "-c", PROBE, *argv], cwd=cwd, env=env, capture_output=True, text=True
    )
    assert ran.returncode == 0, ran.stderr
    where, version = json.loads(ran.stdout.splitlines()[0])
    assert Path(where).resolve().is_relative_to(site.resolve()), where
    return version


def _foreign_repository(path: Path) -> str:
    """A scratch git repository with one commit of its own; returns its HEAD."""
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.com")
    (path / "README").write_text("another project\n", encoding="utf-8")
    _git(path, "add", "README")
    _git(path, "commit", "-q", "-m", "another project")
    return _git(path, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A wheel built straight from this checkout: `uv build --wheel`."""
    out = tmp_path_factory.mktemp("wheel")
    _build(out, "--wheel")
    (path,) = out.glob("*.whl")
    return path


@needs_uv
def test_a_wheel_built_from_the_checkout_is_stamped_with_its_commit(wheel: Path) -> None:
    """The stamp names HEAD, the checkout's dirty flag as code_version reads it, and the kernel;
    it replaces the placeholder rather than sitting beside it in the zip."""
    with zipfile.ZipFile(wheel) as archive:
        assert archive.namelist().count(STAMP) == 1
        stamp = _stamp(archive.read(STAMP).decode())
    checkout = io.code_version()
    assert stamp["commit"] == _git(PROJECT, "rev-parse", "HEAD") == checkout["git_sha"]
    assert stamp["dirty"] == checkout["dirty"]
    assert stamp["kernel_sha256"] == pilot.kernel_sha256()
    datetime.datetime.strptime(stamp["built_at"], "%Y-%m-%dT%H:%M:%SZ")


@needs_uv
def test_the_sdist_carries_its_stamp_into_the_wheel_built_from_it(tmp_path: Path) -> None:
    """Plain `uv build` makes the sdist here, then the wheel from the sdist in a folder no
    repository tracks: that wheel ships the sdist's stamp, neither lost nor re-stamped."""
    _build(tmp_path)
    (sdist,) = tmp_path.glob("*.tar.gz")
    (built,) = tmp_path.glob("*.whl")
    with tarfile.open(sdist) as tar:
        (name,) = [n for n in tar.getnames() if n.endswith("/src/" + STAMP)]
        member = tar.extractfile(name)
        assert member is not None
        in_sdist = _stamp(member.read().decode())
    with zipfile.ZipFile(built) as archive:
        in_wheel = _stamp(archive.read(STAMP).decode())
    assert in_sdist["commit"] == _git(PROJECT, "rev-parse", "HEAD")
    assert in_wheel == in_sdist  # built_at included: carried, not stamped again


@needs_uv
def test_an_installed_wheel_outside_any_checkout_runs_under_its_stamp(
    wheel: Path, tmp_path: Path
) -> None:
    """Old behaviour: no checkout, so git_sha None, which tauc refuses. Now a run's manifest
    names the stamped commit and dirty flag, and its kernel is the one the stamp names."""
    site = _install(wheel, tmp_path / "venv" / "Lib" / "site-packages")
    stamp = _stamp((site / STAMP).read_text(encoding="utf-8"))
    assert stamp["commit"] is not None
    data = tmp_path / "data"
    argv = ["pilot", "--L", "4", "--T", "3.0", "--replicas", "2", "--therm", "10", "--sweeps"]
    argv += ["50", "--device", "cpu", "--data-dir", str(data)]
    version = _probe(site, tmp_path, *argv)
    assert version == {"git_sha": stamp["commit"], "dirty": stamp["dirty"]}
    (run_dir,) = (data / "ising").iterdir()
    manifest = io.read_json(run_dir / "manifest.json")
    assert {key: manifest["producer"][key] for key in version} == version
    assert manifest["params"]["algorithm"]["kernel_sha256"] == stamp["kernel_sha256"]


@needs_uv
@pytest.mark.parametrize("layout", ["venv", "unpacked sdist"])
@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "placeholder"])
def test_a_copy_inside_another_repository_never_reports_that_repository(
    wheel: Path, tmp_path: Path, layout: str, stamped: bool
) -> None:
    """Old behaviour: git found the enclosing repository and recorded its commit. A venv's
    copy, or a source tree no repository tracks (an sdist unpacked inside another project),
    records its own stamp, or nothing when its build was not stamped."""
    foreign = tmp_path / "other-project"
    foreign_head = _foreign_repository(foreign)
    if layout == "venv":
        site = _install(wheel, foreign / ".venv" / "Lib" / "site-packages")
    else:
        tree = foreign / "vendor" / "tipping_point-0.1.0"
        site = _install(wheel, tree / "src")
        shutil.copy(PROJECT / "pyproject.toml", tree / "pyproject.toml")
    if not stamped:
        shutil.copy(Path(_build_info.__file__), site / STAMP)
    package = site / "tipping_point"
    assert Path(_git(package, "rev-parse", "--show-toplevel")).resolve() == foreign.resolve()
    stamp = _stamp((site / STAMP).read_text(encoding="utf-8"))
    version = _probe(site, tmp_path)
    assert version == {"git_sha": stamp["commit"], "dirty": stamp["dirty"]}
    assert version["git_sha"] != foreign_head
    assert (version["git_sha"] is None) == (not stamped)


def test_imported_from_its_checkout_it_reads_git_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """The editable install (`uv sync`) runs from the checkout: HEAD and its status, as before
    #109, and git wins over any stamp the source tree might hold."""
    assert Path(io.__file__).resolve().is_relative_to(PROJECT / "src")
    monkeypatch.setattr(_build_info, "commit", "f" * 40)
    monkeypatch.setattr(_build_info, "dirty", False)
    assert io.code_version() == {
        "git_sha": _git(PROJECT, "rev-parse", "HEAD"),
        "dirty": _git(PROJECT, "status", "--porcelain") != "",
    }
