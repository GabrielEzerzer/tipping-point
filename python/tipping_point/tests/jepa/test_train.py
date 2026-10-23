"""Training mechanics: determinism, resume, the untrained twin, the manifest, what training refuses
to read, the held-out band, the GPU-hour ledger and verification."""

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jepa_ising
import jepa_toy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.ews.ising import open_run
from tipping_point.ising import io
from tipping_point.jepa import data, extract, lattice, train
from tipping_point.jepa.lattice import Jepa, unpack

SHORT = jepa_toy.config(steps=12, checkpoint_every=4, monitor_every=4, pca_frames=128)


@pytest.fixture(scope="module")
def small() -> data.Dataset:
    return jepa_toy.dataset(512, 64)


def _losses(run_dir: Path) -> np.ndarray:
    table = pa.concat_tables(pq.read_table(p) for p in io.chunks(run_dir, "train"))
    return table["loss"].to_numpy()


def _chunks(run_dir: Path) -> dict[str, str]:
    """Table chunk -> sha256: the bytes a run's tables hold (they never name the run)."""
    files = io.read_json(run_dir / "complete.json")["files"]
    return {k: v for k, v in files.items() if k.split("/")[0] in train.TABLES}


def _same_weights(a: Path, b: Path) -> bool:
    first, second = train.load_model(a), train.load_model(b)
    left, right = first.model.state_dict(), second.model.state_dict()
    assert first.pca is not None and second.pca is not None
    pcs = torch.equal(first.pca.components, second.pca.components)
    return pcs and all(torch.equal(left[k], right[k]) for k in left)


def test_one_seed_twice_gives_bitwise_identical_runs(tmp_path: Path, small: data.Dataset) -> None:
    """Catches hidden nondeterminism: a draw from the global RNG, an order that varies between
    runs, a timestamp in a chunk."""
    first = train.start(SHORT, small, tmp_path / "a", "d", "cpu")
    second = train.start(SHORT, small, tmp_path / "b", "d", "cpu")
    assert len(_losses(first)) == SHORT.steps
    assert np.array_equal(_losses(first), _losses(second))  # bitwise
    assert _chunks(first) == _chunks(second)
    assert _same_weights(first / train.FINAL, second / train.FINAL)


def test_another_seed_gives_other_losses(tmp_path: Path, small: data.Dataset) -> None:
    """Catches a seed that is recorded but never used."""
    brief = dataclasses.replace(SHORT, steps=4)
    first = train.start(brief, small, tmp_path / "a", "d", "cpu")
    other = train.start(dataclasses.replace(brief, seed=28), small, tmp_path / "b", "d", "cpu")
    assert (_losses(first) != _losses(other)).all()


def _crash_after(calls: int, real: Callable[..., Any]) -> Callable[..., Any]:
    count = 0

    def crashing(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        if count > calls:
            raise RuntimeError("injected crash")
        return real(*args, **kwargs)

    return crashing


def test_resume_after_a_crash_continues_bit_for_bit(
    tmp_path: Path, small: data.Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a resume that loses an optimiser, scaler or generator state, restarts a schedule,
    or keeps rows the crashed process wrote after its last checkpoint."""
    straight = train.start(SHORT, small, tmp_path / "straight", "d", "cpu")
    monkeypatch.setattr(train._Trainer, "update", _crash_after(7, train._Trainer.update))
    with pytest.raises(RuntimeError, match="injected crash"):
        train.start(SHORT, small, tmp_path / "crashed", "d", "cpu")
    monkeypatch.undo()
    (run_dir,) = [p for p in (tmp_path / "crashed" / "jepa").iterdir() if p.is_dir()]
    assert (run_dir / "checkpoint.pt").exists() and not (run_dir / "complete.json").exists()
    train.resume(run_dir, small)
    assert not (run_dir / "checkpoint.pt").exists()
    assert _chunks(run_dir) == _chunks(straight)
    assert np.array_equal(_losses(run_dir), _losses(straight))
    assert _same_weights(run_dir / train.FINAL, straight / train.FINAL)
    with pytest.raises(RuntimeError, match="already complete"):
        train.resume(run_dir, small)


def test_training_refuses_to_start_on_an_evaluation_mask_without_context(
    tmp_path: Path, small: data.Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an empty evaluation mask found only at extraction, after the GPU hours are spent."""

    def blind(
        B: int, n: int, spec: lattice.MaskSpec, gen: torch.Generator, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full = torch.ones(B, n, n, dtype=torch.bool)
        return full, ~full

    monkeypatch.setattr(lattice, "masks", blind)
    with pytest.raises(ValueError, match="leave no context"):
        train.start(SHORT, small, tmp_path, "blind", "cpu")
    assert not (tmp_path / "jepa").exists()


def test_the_twin_is_the_untrained_model_saved_before_any_update(toy_run: Path) -> None:
    """Catches a twin saved after the first update (it would not be the random-weights null)."""
    twin, final = train.load_model(toy_run / train.TWIN), train.load_model(toy_run / train.FINAL)
    fresh = Jepa(twin.config.model)
    fresh.initialise(torch.Generator().manual_seed(train.seeds(twin.config.seed)["init"]))
    expected = fresh.state_dict()
    assert all(torch.equal(v, expected[k]) for k, v in twin.model.state_dict().items())
    assert twin.untrained and twin.pca is not None and final.pca is not None
    assert final.step == jepa_toy.CONFIG.steps and not final.untrained
    trained = final.model.state_dict()
    assert any(not torch.equal(v, trained[k]) for k, v in twin.model.state_dict().items())
    files = io.read_json(toy_run / "complete.json")["files"]
    assert {train.TWIN.as_posix(), train.FINAL.as_posix(), "manifest.json"} <= files.keys()
    assert train.verify(toy_run) == []


def test_the_manifest_names_its_data_code_seeds_and_pilot(ising: jepa_ising.Ising) -> None:
    manifest = io.read_json(ising.training / "manifest.json")
    trained_on = open_run(ising.runs["train"])
    record = manifest["data"]["train"][0]
    assert record["run_id"] == trained_on.run_id and record["seed"] == 101
    assert record["complete_sha256"] == trained_on.complete_sha256
    assert manifest["data"]["val"][0]["seed"] == 102
    assert manifest["data"]["refused_seeds"] == [1, 2, 3, 4, 5]
    assert set(manifest["params"]["seeds"]) == set(train.STREAMS)
    assert train.Config.from_json(manifest["params"]["config"]) == jepa_ising.CONFIG
    pilot = hashlib.sha256(jepa_ising.PILOT.read_bytes()).hexdigest()
    assert manifest["pilot"]["sha256"] == pilot and manifest["prereg"] is None
    assert "git_sha" in manifest["producer"] and "dirty" in manifest["producer"]
    assert manifest["env"]["use_deterministic_algorithms"] and manifest["env"]["amp"] is None
    assert manifest["params"]["receptive_field"] == 14


def test_trainings_and_latents_take_their_commit_from_the_one_helper(
    ising: jepa_ising.Ising, small: data.Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#88 for the JEPA's two manifests: a training's and a latents table's producer come from
    io.code_version, the copy Ising runs and EWS results read, not a third of their own."""
    planted = {"git_sha": "f" * 40, "dirty": True}
    monkeypatch.setattr(io, "code_version", lambda: dict(planted))
    training = io.read_json(train.start(SHORT, small, tmp_path, "d", "cpu") / "manifest.json")
    ckpt = train.load_model(ising.training / train.FINAL)
    latents = extract._manifest(
        tmp_path / "l", open_run(ising.runs["ramp"]), ckpt, torch.device("cpu")
    )
    for producer in (training["producer"], latents["producer"]):
        assert {key: producer[key] for key in planted} == planted, producer


@pytest.mark.parametrize(
    ("role", "refusal"),
    [
        ("scored", "a pre-registration scores"),
        ("ramp", "never ramps or null runs"),
        ("null", "never ramps or null runs"),
    ],
)
def test_training_refuses_scored_seeds_ramps_and_null_runs(
    ising: jepa_ising.Ising, role: str, refusal: str
) -> None:
    """Catches surprise fitted on the frames it will be scored on, or on a non-equilibrium world."""
    run_id, val = ising.runs[role].name, ising.runs["val"].name
    with pytest.raises(data.DataError, match=refusal):
        data.load(ising.root, [run_id], [val])
    with pytest.raises(data.DataError, match=refusal):
        data.load(ising.root, [ising.runs["train"].name], [run_id])


def test_held_out_and_unthermalised_plateaus_are_never_read(ising: jepa_ising.Ising) -> None:
    """Catches the held-out band leaking into training, a failed plateau read as equilibrium,
    and bursts out of order: every burst decodes to its configs rows' magnetisation, frame by
    frame, through the device unpacker."""
    run = open_run(ising.runs["train"])
    every = data.read_bursts(run)
    assert len(every) == len(jepa_ising.PLATEAUS) * 4 * jepa_ising.REPLICAS
    configs = io.read_table(run.path, "configs")
    order = np.lexsort(
        [configs[k].to_numpy() for k in ("frame", "burst", "replica")]  # by replica, burst, frame
    )
    stored = configs["magnetisation"].to_numpy()[order].reshape(len(every), -1)
    decoded = unpack(every.packed, run.L).flatten(2).sum(-1, dtype=torch.int64).numpy()
    assert np.array_equal(decoded, stored)
    keep, record = data.plateaus(run, None)  # its own verdicts: Tc failed (a 250-sweep dwell)
    assert record["dropped_unthermalised"] == [2.269231] and record["plateaus"] == 2
    assert set(data.read_bursts(run, keep).temperature.tolist()) == {3.5, 1.5}
    clean = dataclasses.replace(run, checks={})
    keep, record = data.plateaus(clean, data.HELD_OUT)
    assert record["dropped_held_out"] == [2.269231] and record["dropped_unthermalised"] == []
    assert set(data.read_bursts(clean, keep).temperature.tolist()) == {3.5, 1.5}
    failed = dataclasses.replace(
        run, checks={"thermalised": [{"temperature": 3.5, "passed": False}]}
    )
    keep, record = data.plateaus(failed, data.HELD_OUT)
    assert record["dropped_unthermalised"] == [3.5] and record["dropped_held_out"] == [2.269231]
    assert set(data.read_bursts(failed, keep).temperature.tolist()) == {1.5}


def test_staircases_are_selected_by_size_and_seed(ising: jepa_ising.Ising) -> None:
    found = data.find_staircases(ising.root, [16], [101, 102])
    assert {k: v.run_id for k, v in found.items()} == {
        (16, 101): ising.runs["train"].name,  # the seed-101 ramp is not a staircase
        (16, 102): ising.runs["val"].name,
    }
    with pytest.raises(data.DataError, match="no complete staircase"):
        data.find_staircases(ising.root, [16], [104])


def test_the_ledger_counts_gpu_hours_per_week(tmp_path: Path, toy_run: Path) -> None:
    lines = (toy_run.parents[1] / "jepa" / train.LEDGER).read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert records and all(r["run_id"] == toy_run.name and r["device"] == "cpu" for r in records)
    assert [r["step"] for r in records] == sorted(r["step"] for r in records)
    assert train.budget(toy_run.parents[1], 30.0) == []  # CPU hours count against nothing
    week = int(datetime(2026, 10, 22, 12, tzinfo=UTC).timestamp()) * 10**9  # ISO week 43
    for hours in (1.5, 2.25):
        record = {"run_id": "r", "to_ns": week, "hours": hours, "device": "cuda"}
        train.append_line(tmp_path / "jepa" / train.LEDGER, record)
    (row,) = train.budget(tmp_path, 30.0)
    assert row == {"week": "2026-W43", "hours": 3.75, "quota": 30.0, "left": 26.25}


def test_verify_names_a_changed_or_missing_file(tmp_path: Path, ising: jepa_ising.Ising) -> None:
    copy = shutil.copytree(ising.training, tmp_path / "copy")
    assert train.verify(copy) == []
    chunk = next((copy / "train").glob("*.parquet"))
    chunk.write_bytes(chunk.read_bytes() + b"\0")
    (copy / train.FINAL).unlink()
    assert sorted(train.verify(copy)) == [
        f"missing {train.FINAL}",
        f"sha256 differs: train/{chunk.name}",
    ]


def test_resume_reloads_the_runs_it_read_and_refuses_changed_ones(
    tmp_path: Path, ising: jepa_ising.Ising, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a resume that trains on whatever the data dir now holds: it reloads the runs the
    manifest names and stops if any complete.json differs from the one it recorded."""
    root = shutil.copytree(ising.root, tmp_path / "root")
    train_run, val_run = ising.runs["train"].name, ising.runs["val"].name
    dataset = data.load(root, [train_run], [val_run])
    monkeypatch.setattr(train._Trainer, "update", _crash_after(5, train._Trainer.update))
    with pytest.raises(RuntimeError, match="injected crash"):
        train.start(jepa_ising.CONFIG, dataset, root, "crash", "cpu")
    monkeypatch.undo()
    (run_dir,) = (root / "jepa").glob("*-crash")
    done = root / "ising" / train_run / "complete.json"
    original = done.read_bytes()
    done.write_bytes(original + b" ")
    with pytest.raises(data.DataError, match="changed since"):
        train.resume(run_dir)
    done.write_bytes(original)
    train.resume(run_dir)
    assert _chunks(run_dir) == _chunks(ising.training)  # the straight run of the same seed
