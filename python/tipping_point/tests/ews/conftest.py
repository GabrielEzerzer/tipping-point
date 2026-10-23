"""The planted Ising dataset (harness/ews_planted.py) as a session fixture: written once, read by
the adapter and scorer tests, never modified by them."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from ews_planted import (
    SEEDS,
    SIZES,
    nulls,
    pilot,
    plateau_values,
    ramp,
    ramp_values,
    staircase,
    write_run,
)


@dataclass(frozen=True)
class Planted:
    root: Path
    runs: dict[str, Path]  # tag -> run folder
    stamps: dict[str, dict[str, object]]  # tag -> the pilot stamp in its manifest (ramps, nulls)


@pytest.fixture(scope="session")
def planted(tmp_path_factory: pytest.TempPathFactory) -> Planted:
    root = tmp_path_factory.mktemp("ews-data")
    runs, stamps = {}, {}
    for L in SIZES:
        sweeps = 500 if L == 16 else 900
        for seed in SEEDS:
            plan = {
                f"L{L}-ramp-s{seed}": (ramp(L), ramp_values(L)),
                f"L{L}-null-s{seed}": (nulls(L, sweeps), ramp_values(L)),
                f"L{L}-qs-s{seed}": (staircase(), plateau_values(L)),
            }
            for tag, (schedule, values) in plan.items():
                if "-qs-" not in tag:  # a staircase dwells on tau.json, not on a pilot record
                    stamps[tag] = {"path": f"pilot-{tag}.json", "sha256": "0" * 64, "stale": False}
                runs[tag] = write_run(root, tag, schedule, L, seed, values, pilot=stamps.get(tag))
    pilot(root)
    return Planted(root, runs, stamps)
