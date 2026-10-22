"""Gate D: on one device type and torch version, a run is a pure function of its manifest.

Same seed -> byte-identical Parquet chunks; a crash at any point plus `resume` -> the same bytes
as a straight run; a different seed -> different bytes.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tipping_point.ising import io, kernel, protocol, run

SPEC = run.Spec(L=8, replicas=4, seed=11, checkpoint_every=150)


def _staircase() -> protocol.Staircase:
    return protocol.Staircase(
        [2.0, 2.0, 2.0],
        temperatures=(3.5, 2.5, 1.5),
        warmup=50,
        min_therm=50,
        min_meas=200,
        saves=4,
    )


def _ramp() -> protocol.Ramp:
    return protocol.Ramp(rate=200, L=SPEC.L, pre=50, hold=50, save_dT=0.05)  # 500 sweeps


def _files(run_dir: Path) -> dict[str, str]:
    """Chunk path -> sha256, from complete.json (which exists only after a clean end)."""
    return io.read_json(run_dir / "complete.json")["files"]


def _crash_after(calls: int, real: Callable[..., Any]) -> Callable[..., Any]:
    count = 0

    def crashing(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        if count > calls:
            raise RuntimeError("injected crash")
        return real(*args, **kwargs)

    return crashing


def test_same_seed_gives_identical_bytes(tmp_path: Path) -> None:
    """Catches any hidden nondeterminism: global RNG use, unordered rows, timestamps in chunks."""
    first = run.start(SPEC, _staircase(), tmp_path / "a", "d", "cpu")
    second = run.start(SPEC, _staircase(), tmp_path / "b", "d", "cpu")
    files = _files(first)
    assert {name.split("/")[0] for name in files} == {"stats", "configs"}
    assert files == _files(second)


def test_different_seed_gives_different_bytes(tmp_path: Path) -> None:
    """Catches a seed that is recorded but never used."""
    first = _files(run.start(SPEC, _staircase(), tmp_path / "a", "d", "cpu"))
    other = run.Spec(L=8, replicas=4, seed=12, checkpoint_every=150)
    second = _files(run.start(other, _staircase(), tmp_path / "b", "d", "cpu"))
    assert first.keys() == second.keys()
    assert all(first[name] != second[name] for name in first)


def test_resume_after_a_crash_equals_the_straight_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a resume that loses the generator state, restarts the schedule clock, or keeps
    rows the crashed process wrote after its last checkpoint."""
    straight = _files(run.start(SPEC, _ramp(), tmp_path / "straight", "d", "cpu"))
    monkeypatch.setattr(kernel, "sweep", _crash_after(333, kernel.sweep))
    with pytest.raises(RuntimeError, match="injected crash"):
        run.start(SPEC, _ramp(), tmp_path / "crashed", "d", "cpu")
    monkeypatch.undo()
    (run_dir,) = (tmp_path / "crashed" / "ising").iterdir()
    assert (run_dir / "checkpoint.pt").exists() and not (run_dir / "complete.json").exists()
    assert _files(run.resume(run_dir)) == straight


def test_resume_after_a_crash_between_flush_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nastiest crash point: chunks up to step 300 are on disk but the checkpoint still says
    150. Resume must drop those chunks and rewrite them identically."""
    straight = _files(run.start(SPEC, _ramp(), tmp_path / "straight", "d", "cpu"))
    monkeypatch.setattr(run, "_checkpoint", _crash_after(2, run._checkpoint))  # steps 0, 150, boom
    with pytest.raises(RuntimeError, match="injected crash"):
        run.start(SPEC, _ramp(), tmp_path / "crashed", "d", "cpu")
    monkeypatch.undo()
    (run_dir,) = (tmp_path / "crashed" / "ising").iterdir()
    assert any(int(p.stem) > 150 for p in io.chunks(run_dir, "stats"))  # the orphaned chunk
    assert _files(run.resume(run_dir)) == straight


def test_resume_refuses_a_complete_run(tmp_path: Path) -> None:
    """Catches appending a second copy of the data to a finished run."""
    run_dir = run.start(SPEC, _ramp(), tmp_path, "d", "cpu")
    with pytest.raises(RuntimeError, match="already complete"):
        run.resume(run_dir)
