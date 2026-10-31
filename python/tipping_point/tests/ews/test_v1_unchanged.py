"""ising-ews-v1 stays as it is at aa29cbc, the freeze lane's merge: every later commit must still
score it the same, whatever it adds for the latent signals of #28.

The golden (harness/ews_v1_golden.py) holds every row and value the scorer gave on planted data at
that commit, before any latent code existed. A scorer that drops or adds a row, moves a value, or
draws from the RNG once more, once less or in another order, differs from it: every draw feeds the
CIs and surrogate rates of the rows after it."""

import hashlib
import json
from pathlib import Path
from typing import Any

import ews_v1_golden
import numpy as np
import pytest
from ews_planted import V1, v1_copy, v1_dataset, write_latents
from tipping_point.ews import ising

V1_TOML = ews_v1_golden.V1_TOML
KERNEL = ews_v1_golden.REPO / "python" / "tipping_point" / "src" / "tipping_point" / "ising"
FROZEN = {  # the bytes at aa29cbc, the freeze lane's merge: a change is a new version, not an edit
    V1_TOML: "f1765c2d8b35b970705cb47e9c197c701d88b15235d1936b8d54448526355ee4",
    KERNEL / "kernel.py": "197acfe347462373092e4c9fa04388c0602cabe2a8cd1652ac67c208e09ca6a7",
}


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(ews_v1_golden.GOLDEN.read_text("utf-8"))


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> V1:
    return v1_dataset(tmp_path_factory.mktemp("v1-unchanged"), "fss")


def _without_checkpoint(scored: dict[str, Any]) -> dict[str, Any]:
    """The scoring as the golden holds it: every row must say it read no latents table."""
    tables = {}
    for name, rows in scored["tables"].items():
        assert all(row["checkpoint"] == "" for row in rows), name
        tables[name] = [{k: v for k, v in row.items() if k != "checkpoint"} for row in rows]
    return scored | {"tables": tables}


def test_the_golden_was_written_from_the_freeze_lanes_merge(golden: dict[str, Any]) -> None:
    """Catches a golden rewritten from a later commit, which would prove nothing."""
    assert golden["made_by"] == "aa29cbc887bd7507d241f9c48a83fc1a930b35be"
    assert {name: len(rows) for name, rows in golden["tables"].items()} == {
        "auc": 144,  # 3 indicators x 2 observables x 2 cadences x 3 L x 2 h x 2 bandwidths
        "surrogates": 144,  # the headline cell's 72, under 2 tests
        "alarms": 18,
        "peaks": 18,
        "fss": 7,
    }


def test_a_v1_scoring_with_no_latents_is_the_golden_row_for_row(
    planted: V1, golden: dict[str, Any]
) -> None:
    """The same rows in the same order and the same values, apart from a `checkpoint` column every
    row holds as "": a classic row read no latents table."""
    scored = _without_checkpoint(ews_v1_golden.score(planted))
    assert ews_v1_golden.differences(golden["tables"], scored["tables"]) == []
    assert (scored["dropped"], scored["inputs"]) == (golden["dropped"], golden["inputs"])


def test_latents_beside_the_runs_change_nothing_v1_reads(planted: V1, tmp_path: Path) -> None:
    """The confirmatory runs will hold latents tables once ising-latent-v1 has read them, and v1 may
    be scored again after that. With a table beside every run, the runs v1 finds and every array it
    reads from them are the same (the scoring above is a function of those and the RNG)."""
    copy = v1_copy(planted, tmp_path / "data")
    for run_dir in copy.runs.values():
        write_latents(run_dir, "training-step0050000")
    before, after = ising.find_runs(planted.root), ising.find_runs(copy.root)
    assert [r.run_id for r in before[0]] == [r.run_id for r in after[0]] and not after[1]
    for run, other in zip(before[0], after[0], strict=True):
        assert run.complete_sha256 == other.complete_sha256
        for cadence in ("dense", "matched"):
            for block, same in zip(
                ising.series(run, cadence), ising.series(other, cadence), strict=True
            ):
                assert not same.latents
                assert np.array_equal(block.abs_m, same.abs_m)
                assert np.array_equal(block.energy, same.energy)
                assert np.array_equal(block.steps, same.steps)


def test_the_golden_check_can_fail(golden: dict[str, Any]) -> None:
    """The control for the two above: a row moved or missing, a value off by more than rounding, or
    an extra column are all reported, and rounding is not."""
    tables = golden["tables"]
    moved = {**tables, "auc": [tables["auc"][1], tables["auc"][0], *tables["auc"][2:]]}
    assert ews_v1_golden.differences(tables, moved)
    assert ews_v1_golden.differences(tables, {**tables, "fss": tables["fss"][:-1]})
    row = tables["fss"][0]
    off = lambda delta: {**tables, "fss": [{**row, "estimate": row["estimate"] * (1 + delta)}]}  # noqa: E731
    one_row = {**tables, "fss": [row]}
    assert ews_v1_golden.differences(one_row, off(1e-6))
    assert ews_v1_golden.differences(one_row, off(1e-13)) == []
    assert ews_v1_golden.differences(one_row, {**tables, "fss": [{**row, "extra": 1}]})


@pytest.mark.parametrize("path", FROZEN, ids=lambda p: p.name)
def test_the_frozen_files_keep_their_bytes(path: Path) -> None:
    """The prereg's sha256 is what `frozen` holds a scoring to, and the kernel's what the #54 record
    was measured on (a stale record is refused)."""
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN[path]
