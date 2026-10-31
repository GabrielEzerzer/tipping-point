"""The command line, in process: train's argument wiring and refusals (without training),
extract, verify and budget."""

from pathlib import Path
from typing import Any

import jepa_ising
import pytest
from tipping_point.jepa import train
from tipping_point.jepa.__main__ import main
from tipping_point.jepa.data import Dataset


def _train_argv(ising: jepa_ising.Ising, *extra: str) -> list[str]:
    argv = ["train", "--data-dir", str(ising.root), "--L", "16", "--val-seed", "102"]
    return [*argv, "--seed", "7", "--pilot", str(jepa_ising.PILOT), *extra]


def test_train_selects_runs_by_seed_and_records_its_calls(
    ising: jepa_ising.Ising, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches arguments that never reach the config or the data: the run is captured, not
    trained."""
    calls: list[dict[str, Any]] = []

    def capture(config: train.Config, data: Dataset, root: Path, tag: str, *rest: Any) -> Path:
        calls.append({"config": config, "data": data, "tag": tag, "rest": rest})
        return root

    monkeypatch.setattr(train, "start", capture)
    prereg = tmp_path / "jepa-v1.toml"
    prereg.write_text('id = "ising-jepa"\nversion = 1\n[data]\nseeds = [1, 2, 17]\n')
    argv = _train_argv(ising, "--train-seeds", "101", "--held-out", "--steps", "100")
    assert main([*argv, "--prereg", str(prereg)]) == 0
    (call,) = calls
    assert call["config"].seed == 7 and call["config"].steps == 100
    assert call["config"].warmup == 5 and call["tag"] == "L16-m7-heldout"
    sources = call["data"].sources
    assert sources["held_out"] == [2.0, 2.5]
    assert sources["refused_seeds"] == [*range(1, 11), 17]  # the JEPA prereg's 17 joins 1-10
    assert [r["run_id"] for r in sources["train"]] == [ising.runs["train"].name]
    assert [r["run_id"] for r in sources["val"]] == [ising.runs["val"].name]
    _, pilot, recorded = call["rest"]
    assert pilot["kernel_commit"] and recorded["id"] == "ising-jepa" and recorded["version"] == 1


@pytest.mark.parametrize(
    "seeds",
    [
        *(["--train-seeds", str(seed)] for seed in range(1, 11)),  # the pilot's 1-5 and v1's 6-10
        ["--train-seeds", "101", "3"],
        ["--train-seeds", "101", "8"],
        ["--train-seeds", "101", "--val-seed", "6"],
    ],
)
def test_train_refuses_preregistered_seeds_before_reading_data(
    ising: jepa_ising.Ising,
    seeds: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catches any seed of 1-10 getting through as a train or a validation seed. The harness holds
    a real staircase at seed 6, so a refusal that lapsed would train on it: here that fails."""

    def never(*_: object, **__: object) -> None:
        raise AssertionError("a scored seed reached training")

    monkeypatch.setattr(train, "start", never)
    assert main(_train_argv(ising, *seeds)) == 1
    assert "scored by a pre-registration" in capsys.readouterr().err


@pytest.mark.parametrize("seed", ["11", "16"])
def test_train_leaves_the_seeds_past_the_scored_range_to_the_run_lookup(
    ising: jepa_ising.Ising, seed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a refusal wider than the freeze's: #28's training staircases keep their own seeds,
    11-16 (#26 decision 6). The harness has no such run, so the lookup answers instead."""
    assert main(_train_argv(ising, "--train-seeds", seed)) == 1
    err = capsys.readouterr().err
    assert "scored by a pre-registration" not in err
    assert f"no complete staircase run for (L, seed) [(16, {seed})]" in err


def test_extract_verify_and_budget_commands(
    ising: jepa_ising.Ising, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    twin = ising.training / train.TWIN
    assert main(["extract", str(twin), str(ising.runs["null"]), "--device", "cpu"]) == 0
    out = Path(capsys.readouterr().out.strip())
    assert out.parent.name == "latents" and out.parent.parent == ising.runs["null"]
    assert main(["verify", str(ising.training), str(out), str(ising.runs["null"])]) == 0
    assert capsys.readouterr().out.count(": ok") == 3
    assert main(["verify", str(tmp_path)]) == 1
    assert main(["budget", "--data-dir", str(tmp_path)]) == 0
    assert "no GPU hours logged" in capsys.readouterr().out
