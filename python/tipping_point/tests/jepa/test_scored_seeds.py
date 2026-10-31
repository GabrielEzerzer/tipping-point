"""Where the seed refusal meets the readouts (#26 decision 6, #28): a staircase at a seed
ising-ews-v1 scores stays out of training, and it is what `extract --select`, the probes' controls
and the paired margins read. The refusal and the readouts came from separate lanes, each tested on
its own terms; these checks hold the two together on runs the simulator wrote."""

import dataclasses
import shutil
import tomllib
from pathlib import Path

import jepa_ising
import numpy as np
import pyarrow as pa
import pytest
from tipping_point.ews import fss, margins
from tipping_point.ews.ising import open_run
from tipping_point.ising import io, protocol
from tipping_point.jepa import data, extract, probes, train

PREREG = Path(__file__).resolve().parents[4] / "paper" / "prereg" / "ising-ews-v1.toml"
V1 = tomllib.loads(PREREG.read_text("utf-8"))["data"]
ROLES = ("trained", "twin")


def test_the_staircase_the_selections_read_is_one_training_refuses(
    ising: jepa_ising.Ising,
) -> None:
    """Catches a fixture blind to the freeze: at a pilot seed (1-5), which the refusal held before
    v1, every refusal and selection test passes with v1's 6-10 gone from PREREG_SEEDS. So the
    scored run sits at a seed v1 scores and saves a burst count 8 divides, and the runs the tiny
    model trained on stay clear of the refusal."""
    scored = open_run(ising.runs["scored"])
    assert scored.kind == "staircase" and scored.seed in V1["seeds"]
    assert int(scored.schedule["params"]["saves"]) % extract.EVERY == 0
    with pytest.raises(data.DataError, match=f"has seed {scored.seed}, which a pre-registration"):
        data.check_training_run(scored, data.PREREG_SEEDS)
    for role in ("train", "val"):
        assert open_run(ising.runs[role]).seed not in data.PREREG_SEEDS


def test_a_staircase_as_v1_generates_it_holds_what_the_selections_need() -> None:
    """Catches the freeze's staircases and `--select` parting ways, which no harness run can show,
    its plateaus and bursts being its own. On `ising staircase`'s grid and default saves, the
    plateau extract keeps is the one v1 reads and sits on the grid once, and every plateau holds
    all its bursts (extract's shortcut assumes it), in a count 8 divides."""
    assert V1["staircase"]["temperature"] == extract.TC_PLATEAU
    grid = list(protocol.GRID)
    rows = pa.table({"temperature": grid, "burst": [1] * len(grid)})  # burst 1: not one of 8
    assert extract.kept("staircase", rows, extract.selection("tc-plus")).sum() == 1
    assert extract.kept("staircase", rows, extract.selection("tc-only")).sum() == 1
    schedule = protocol.Staircase([1.0] * len(grid))  # the shortest dwell a plateau can get
    saves = schedule.params["saves"]
    assert saves % extract.EVERY == 0
    assert len(schedule.frames) == len(grid) * saves * protocol.BURST.frames


@pytest.fixture(scope="module")
def trainings(ising: jepa_ising.Ising, tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """The harness's training and a second model seed on the same runs: a margin needs two."""
    dataset = data.load(ising.root, [ising.runs["train"].name], [ising.runs["val"].name])
    config = dataclasses.replace(jepa_ising.CONFIG, seed=jepa_ising.CONFIG.seed + 1)
    root = tmp_path_factory.mktemp("second-model-seed")
    pilot = train.pilot_record(jepa_ising.PILOT)
    return [ising.training, train.start(config, dataset, root, "second", "cpu", pilot=pilot)]


@pytest.mark.parametrize("rule", ["tc-plus", "tc-only"])
def test_margins_judge_latents_selected_from_a_staircase_training_refuses(
    tmp_path: Path, ising: jepa_ising.Ising, trainings: list[Path], rule: str
) -> None:
    """#28's path on simulator output, at a seed v1 scores: training refuses the staircase, each
    model seed's checkpoint and twin write a `--select` table of it that verify accepts, over one
    set of rows, and the paired margin of their surprise comes out finite, nested in shared data
    draws. Surprise stands in here for #28's scale-free quantities: the wiring is under test, not
    its value. Catches a readout that inherits training's refusal, a twin read over other rows
    than its checkpoint, and a margin taken between a checkpoint and itself."""
    source = ising.runs["scored"]
    with pytest.raises(data.DataError, match="a pre-registration scores"):
        data.load(ising.root, [source.name], [ising.runs["val"].name])
    run = shutil.copytree(
        source, tmp_path / "ising" / source.name, ignore=shutil.ignore_patterns("latents")
    )

    def table(checkpoint: Path) -> pa.Table:
        out = extract.extract(checkpoint, run, "cpu", select=rule)
        assert extract.verify(out) == []
        return extract.read(run, out.name)

    final = [table(training / train.FINAL) for training in trainings]
    twins = [table(training / train.TWIN) for training in trainings]
    keys = list(extract.KEY)
    assert all(t.select(keys).equals(final[0].select(keys)) for t in [*final, *twins])
    assert final[0].num_rows < io.read_table(run, "configs").num_rows  # the rule dropped rows
    trained, twin = (
        np.stack([t["surprise"].to_numpy() for t in tables]).astype(np.float64)
        for tables in (final, twins)
    )
    rows = trained.shape[1]
    draws = np.random.default_rng(0).integers(0, rows, (64, rows))  # one resample per column
    row = margins.paired(
        trained.mean(1),
        twin.mean(1),
        "less",
        1.0,
        np.random.default_rng(1),
        trained[:, draws].mean(2),
        twin[:, draws].mean(2),
        n=200,
    )
    assert row["n_seeds"] == 2 and row["n_draws"] == 64
    np.testing.assert_allclose(row["margins"], trained.mean(1) - twin.mean(1))
    assert all(margin != 0 for margin in row["margins"])
    assert row["lo"] < row["hi"] and row["sign_test"] in {fss.PASS, fss.FAIL}


def test_margins_judge_the_probes_on_a_staircase_training_refuses(
    ising: jepa_ising.Ising, trainings: list[Path]
) -> None:
    """Fitted on the validation seed and scored on a staircase at a seed v1 scores, four checkpoints
    come back in order with each target's controls, and the paired margin of their selectivity
    comes out finite. Catches a probe that inherits training's refusal, rows that lose which model
    seed they belong to (the pairing a margin rests on), and a control that stops being one."""
    checkpoints = [t / name for t in trainings for name in (train.FINAL, train.TWIN)]
    rows = probes.probe(checkpoints, [ising.runs["val"]], [ising.runs["scored"]], device="cpu")
    assert len(rows) == len(checkpoints) * len(probes.TARGETS) * len(probes.COLUMNS)
    for target in probes.TARGETS:
        raw = [r for r in rows if r["target"] == target and r["column"] == "raw"]
        assert [(r["run_id"], r["role"]) for r in raw] == [
            (training.name, role) for training in trainings for role in ROLES
        ]
        trained, twin = ([r["selectivity"] for r in raw if r["role"] == role] for role in ROLES)
        row = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(2), n=200)
        np.testing.assert_allclose(row["margins"], np.subtract(trained, twin))
        assert row["n_seeds"] == 2 and row["lo"] < row["hi"]
        assert row["sign_test"] in {fss.PASS, fss.FAIL}
        iid = [r["r2"] for r in rows if r["target"] == target and r["column"] == "iid"]
        assert all(-0.2 < r2 < 0.05 for r2 in iid), (target, iid)  # as test_probes' bound
