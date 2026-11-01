"""The stream each planted run draws its noise from (#110): a run's own, and the same every time.

Runs with no explicit stream used to be seeded by the length of their tag, so L16-ramp-s1 and
L16-null-s1, both 11 characters, drew the same numbers: the old dataset's ramps and nulls at one L
and seed shared noise, and a test reading separation between them read correlated noise.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pytest
from ews_planted import SEEDS, SIZES, write_run
from tipping_point.ising import protocol

SHORT = protocol.Fixed([2.5], therm=2, sweeps=8)


def _stream(
    root: Path, tag: str, L: int = 16, seed: int = 1, rng: np.random.Generator | None = None
) -> list[float]:
    """The first numbers of the stream write_run hands a run's values."""
    drawn: list[float] = []

    def values(
        steps: np.ndarray, temps: np.ndarray, stream: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        drawn.extend(stream.random(8).tolist())
        return np.full(temps.shape, 0.3), np.full(temps.shape, -1.4)

    write_run(root, tag, SHORT, L, seed, values, rng=rng)
    return drawn


def test_two_runs_whose_tags_have_the_same_length_draw_different_streams(tmp_path: Path) -> None:
    """The issue's own case: a ramp and a null at one L and seed, with tags of 11 characters."""
    ramp, null = "L16-ramp-s1", "L16-null-s1"
    assert len(ramp) == len(null)
    assert _stream(tmp_path, ramp) != _stream(tmp_path, null)


def test_no_two_runs_of_the_old_dataset_share_a_stream(tmp_path: Path) -> None:
    """The tags conftest's dataset is written under, each at its own L and seed."""
    tags = [
        (f"L{L}-{kind}-s{seed}", L, seed)
        for L, seed, kind in product(SIZES, SEEDS, ("ramp", "null", "qs"))
    ]
    streams = {tuple(_stream(tmp_path, tag, L, seed)) for tag, L, seed in tags}
    assert len(streams) == len(tags) == 12


def test_a_run_draws_the_same_stream_every_time(tmp_path: Path) -> None:
    """A planted answer has to be the same answer on every run of the tests."""
    assert _stream(tmp_path / "a", "L16-ramp-s1") == _stream(tmp_path / "b", "L16-ramp-s1")


def test_the_size_and_the_seed_move_the_stream_too(tmp_path: Path) -> None:
    """One tag at another L or seed, which only a test that writes its own runs would do."""
    base = _stream(tmp_path / "a", "run")
    assert base != _stream(tmp_path / "b", "run", L=32)
    assert base != _stream(tmp_path / "c", "run", seed=2)


@pytest.mark.parametrize("tag", ["L16-ramp-x0p02-s6", "L16-null-s6"], ids=["ramp", "null"])
def test_an_explicit_stream_replaces_the_default(tmp_path: Path, tag: str) -> None:
    """What the v1 dataset passes: the golden ising-ews-v1 is held to rests on these streams, not
    on the default one."""
    drawn = _stream(tmp_path, tag, seed=6, rng=np.random.default_rng([6, 16, 1]))
    assert drawn == np.random.default_rng([6, 16, 1]).random(8).tolist()
