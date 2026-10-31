"""Gate D: on one device type and torch version, a run is a pure function of its manifest.

The same run twice -> byte-identical Parquet chunks; a crash at any point plus `resume` -> the same
bytes as a straight run; a different seed -> different bytes; another run at the same seed ->
another stream (run.stream_key), however alike their shapes.
"""

import itertools
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from tipping_point.ising import io, kernel, protocol, run
from tipping_point.ising.__main__ import main
from tipping_point.ising.exact import TC

SPEC = run.Spec(L=8, replicas=4, seed=11, checkpoint_every=150)
PILOT = Path(__file__).resolve().parents[1] / "fixtures" / "ising_pilot.json"


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


def _null() -> protocol.Fixed:
    return protocol.null_run(SPEC.L, 2.0, tau_c=1.0)  # 500 + 700 sweeps


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


@pytest.mark.parametrize("make", [_staircase, _ramp, _null])
def test_the_same_run_twice_gives_identical_bytes(
    tmp_path: Path, make: Callable[[], protocol.Schedule]
) -> None:
    """Catches any hidden nondeterminism: global RNG use, unordered rows, timestamps in chunks; and
    a stream that hangs on the tag, which only names the folder."""
    first = run.start(SPEC, make(), tmp_path / "a", "d", "cpu")
    second = run.start(SPEC, make(), tmp_path / "b", "another-name", "cpu")
    files = _files(first)
    assert {name.split("/")[0] for name in files} == {"stats", "configs"}
    assert files == _files(second)


def test_bookkeeping_does_not_move_the_stream(tmp_path: Path) -> None:
    """Catches a tally or a flush that draws, or that enters the run's identity: another
    checkpoint cadence and Gate C's tallies write other files, and the same physics."""
    first = run.start(SPEC, _ramp(), tmp_path / "a", "d", "cpu")
    other = replace(SPEC, checkpoint_every=70, tally=True)
    second = run.start(other, _ramp(), tmp_path / "b", "d", "cpu")
    assert io.read_table(first, "stats").equals(io.read_table(second, "stats"))


def _v1_commands() -> dict[str, list[str]]:
    """The runs v1 makes at one (L, seed, replicas), as the command line asks for them."""
    record = ["--pilot", str(PILOT)]
    window = ["--eps-start", "12", "--eps-end", "-8"]
    return {
        "staircase": ["staircase", "--tau", "50"],
        "slow ramp": ["ramp", "--x", "0.02", *record, *window],
        "fast ramp": ["ramp", "--x", "2", *record, *window],
        "null 2": ["null", "--epsilon-L", "2", *record],
        "null 4": ["null", "--epsilon-L", "4", *record],
        "null 6": ["null", "--epsilon-L", "6", *record],
    }


def test_every_run_v1_makes_at_one_size_and_seed_draws_its_own_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Freeze review, round 1: run.start seeded its generator with spec.seed alone, so at (L = 16,
    seed 6, 64 replicas) the staircase, both ramps and the three nulls started from one lattice
    and drew one stream, and the scorer read them as independent. Each now starts from a lattice
    and a generator state of its own, from a key its manifest holds and names the rule for, which
    the manifest alone can reproduce. The tags alone would pass this; replicas, windows and dwells
    are not in them (the next test). Nothing is simulated: the loop is cut off after the setup."""
    started: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    real = run._checkpoint

    def setup(run_dir: Path, step: int, s: torch.Tensor, gen: torch.Generator) -> None:
        started[run_dir.parents[1].name] = (s.clone(), gen.get_state())
        real(run_dir, step, s, gen)

    monkeypatch.setattr(run, "_checkpoint", setup)
    monkeypatch.setattr(run, "_loop", lambda *_: None)
    keys = {}
    size = ["--L", "16", "--seed", "6", "--replicas", "64", "--device", "cpu"]
    for name, command in _v1_commands().items():
        assert main([*command, *size, "--data-dir", str(tmp_path / name)]) == 0
        (run_dir,) = (tmp_path / name / "ising").iterdir()
        params = io.read_json(run_dir / "manifest.json")["params"]
        spec = run.Spec(params["L"], params["replicas"], params["seed"], params["init"])
        keys[name] = params["rng"]["seed"]
        assert keys[name] == run.stream_key(spec, protocol.from_json(params["schedule"])), name
        assert 0 <= keys[name] < 2**63 and params["rng"]["derivation"] == run.STREAM_DERIVATION
        gen = torch.Generator().manual_seed(keys[name])  # the key is what seeded the generator
        spins = torch.randint(0, 2, (64, 16, 16), generator=gen, dtype=torch.int8) * 2 - 1
        assert torch.equal(spins, started[name][0]), name
    assert len(set(keys.values())) == len(keys) == 6, keys
    for a, b in itertools.combinations(keys, 2):
        assert not torch.equal(started[a][0], started[b][0]), (a, b)
        assert not torch.equal(started[a][1], started[b][1]), (a, b)


def test_the_stream_follows_the_runs_identity_and_nothing_else() -> None:
    """Catches an identity narrower than the run. A key from the seed and the tag separates the
    runs above, yet gives one stream to a ramp over two windows, a staircase of two dwells, and 16
    and 64 replicas, whose first 16 would share their starts. Bookkeeping stays out of it."""
    key = run.stream_key(SPEC, _ramp())
    assert key == run.stream_key(replace(SPEC, checkpoint_every=7, tally=True), _ramp())
    assert key == run.stream_key(SPEC, protocol.from_json(_ramp().to_json()))  # read back from JSON
    stairs: dict[str, Any] = {
        "temperatures": (3.5, 2.5, 1.5),
        "warmup": 50,
        "min_therm": 50,
        "min_meas": 200,
    }
    others = [
        run.stream_key(replace(SPEC, seed=12), _ramp()),
        run.stream_key(replace(SPEC, L=16), _ramp()),
        run.stream_key(replace(SPEC, replicas=8), _ramp()),
        run.stream_key(replace(SPEC, init="cold"), _ramp()),
        run.stream_key(SPEC, protocol.Ramp(rate=200, L=SPEC.L, T0=3.0, pre=50, hold=50)),
        run.stream_key(SPEC, protocol.Ramp(rate=400, L=SPEC.L, pre=50, hold=50, save_dT=0.05)),
        run.stream_key(SPEC, protocol.Staircase([2.0] * 3, **stairs)),
        run.stream_key(SPEC, protocol.Staircase([3.0] * 3, **stairs)),
        run.stream_key(SPEC, protocol.Staircase([2.0] * 3, saves=8, **stairs)),
        run.stream_key(SPEC, protocol.null_run(SPEC.L, 2.0, tau_c=1.0)),
        run.stream_key(SPEC, protocol.null_run(SPEC.L, 4.0, tau_c=1.0)),
    ]
    assert len({key, *others}) == 1 + len(others), others


def test_runs_of_one_shape_at_one_seed_do_not_share_their_noise(tmp_path: Path) -> None:
    """The same, read off what the real loop writes, on cut-down v1 schedules of one shape. With
    one stream, M after the first sweep correlates across replicas at r = 0.94-1.00 for any two
    (one start, one sweep's numbers), and the two ramps, held at one T0, are bit-equal throughout
    the hold; apart, r is at most 0.19 and 2% of those values agree, as M's even integers do by
    chance. Long-run correlations cannot show this: runs at different T decorrelate within ~20
    sweeps on one stream too."""
    L, replicas, pre = 16, 64, 30
    T0, T1 = (TC * (1 + eps / L) for eps in (12.0, -8.0))
    schedules: dict[str, protocol.Schedule] = {
        "staircase": protocol.Staircase(
            [0.1, 0.1], (3.5, 2.5), warmup=0, min_therm=pre, min_meas=pre, saves=0
        ),
        "slow ramp": protocol.Ramp(40, L, T0=T0, T1=T1, pre=pre, hold=1),
        "fast ramp": protocol.Ramp(20, L, T0=T0, T1=T1, pre=pre, hold=1),
    }
    for eps in protocol.NULL_EPSILON_L:
        schedules[f"null {eps:g}"] = protocol.Fixed((TC * (1 + eps / L),), therm=pre, sweeps=pre)
    spec = run.Spec(L=L, replicas=replicas, seed=6)
    magnetisation = {}
    for name, schedule in schedules.items():
        stats = io.read_table(run.start(spec, schedule, tmp_path / name, "t", "cpu"), "stats")
        step, replica = stats["step"].to_numpy(), stats["replica"].to_numpy()
        rows = np.lexsort((replica, step))
        magnetisation[name] = stats["magnetisation"].to_numpy()[rows].reshape(-1, replicas)
    for a, b in itertools.combinations(schedules, 2):
        r = np.corrcoef(magnetisation[a][0], magnetisation[b][0])[0, 1]
        assert abs(r) < 0.5, (a, b, r)
    slow, fast = magnetisation["slow ramp"][:pre], magnetisation["fast ramp"][:pre]
    assert (slow == fast).mean() < 0.2


def test_different_seed_gives_different_bytes(tmp_path: Path) -> None:
    """Catches a seed that is recorded but never used."""
    first = _files(run.start(SPEC, _staircase(), tmp_path / "a", "d", "cpu"))
    other = run.Spec(L=8, replicas=4, seed=12, checkpoint_every=150)
    second = _files(run.start(other, _staircase(), tmp_path / "b", "d", "cpu"))
    assert first.keys() == second.keys()
    assert all(first[name] != second[name] for name in first)


@pytest.mark.parametrize("make", [_staircase, _ramp, _null])
def test_resume_after_a_crash_equals_the_straight_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make: Callable[[], protocol.Schedule]
) -> None:
    """Catches a resume that loses the generator state (restarting it from the key, say), restarts
    the schedule clock, or keeps rows the crashed process wrote after its last checkpoint."""
    straight = _files(run.start(SPEC, make(), tmp_path / "straight", "d", "cpu"))
    monkeypatch.setattr(kernel, "sweep", _crash_after(333, kernel.sweep))
    with pytest.raises(RuntimeError, match="injected crash"):
        run.start(SPEC, make(), tmp_path / "crashed", "d", "cpu")
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
