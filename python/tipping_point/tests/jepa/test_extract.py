"""The latents table (#28's contract) on runs the simulator itself wrote: one row per configs row,
keyed and ordered like configs, readable the way the EWS adapter reads configs, its manifest
naming the checkpoint by sha256, written once and resumable."""

import dataclasses
import hashlib
import shutil
from pathlib import Path

import jepa_ising
import numpy as np
import pyarrow as pa
import pytest
import torch
from tipping_point.ews import ising as ews_ising
from tipping_point.ising import io
from tipping_point.jepa import data, extract, train
from tipping_point.jepa.data import DataError
from tipping_point.jepa.lattice import canonical
from tipping_point.jepa.monitor import Floors
from tipping_point.jepa.readout import readouts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_latents_round_trip_on_a_simulator_run(ising: jepa_ising.Ising, ramp_latents: Path) -> None:
    """Catches rows dropped, reordered or misaligned with their frames: every row's readouts are
    recomputed from that configs row's own spins."""
    ramp = ising.runs["ramp"]
    assert extract.available(ramp) == [ramp_latents.name]
    latents, configs = extract.read(ramp, ramp_latents.name), io.read_table(ramp, "configs")
    assert latents.num_rows == configs.num_rows
    for key in extract.KEY:
        assert latents[key].equals(configs[key]), key
    readout = np.stack([latents[c].to_numpy() for c in latents.column_names[len(extract.KEY) :]])
    assert np.isfinite(readout).all() and (readout[0] > 0).all()  # surprise, then pc_1..pc_8
    ckpt = train.load_model(ising.training / train.FINAL)
    assert ckpt.pca is not None
    hidden, keep = train.evaluation_masks(ckpt.config, jepa_ising.L)
    rows = np.array([0, 1, configs.num_rows // 2, configs.num_rows - 1])
    spins = torch.from_numpy(io.unpack_column(configs["spins"].take(rows), jepa_ising.L))
    embedding, surprise = readouts(ckpt.model, canonical(spins, 2), hidden, keep)
    assert surprise is not None
    pcs = ckpt.pca.project(embedding).to(torch.float32).numpy()
    np.testing.assert_allclose(readout[0, rows], surprise.numpy(), rtol=1e-5)
    np.testing.assert_allclose(readout[1:, rows].T, pcs, rtol=1e-4, atol=1e-5)


def test_the_manifest_names_the_checkpoint_and_the_dataset_by_sha256(
    ising: jepa_ising.Ising, ramp_latents: Path
) -> None:
    manifest = io.read_json(ramp_latents / "manifest.json")
    checkpoint, dataset = manifest["params"]["checkpoint"], manifest["params"]["dataset"]
    assert checkpoint["sha256"] == _sha256(ising.training / train.FINAL)
    assert checkpoint["path"] == f"jepa/{ising.training.name}/final.pt"
    assert checkpoint["step"] == jepa_ising.CONFIG.steps and not checkpoint["untrained"]
    assert dataset["run_id"] == ising.runs["ramp"].name
    assert dataset["complete_sha256"] == _sha256(ising.runs["ramp"] / "complete.json")
    assert manifest["params"]["pca"]["components"] == 8
    assert train.verify(ramp_latents) == []


def test_latents_keep_the_configs_contract_the_scorer_reads(
    ising: jepa_ising.Ising, ramp_latents: Path
) -> None:
    """The reader expectations of tipping_point.ews.ising.series, met by the latents table: the
    key columns with configs' own names and types, rows in (step, replica) order holding whole
    steps of replicas x blocks, so a (batch, time) array comes out of one reshape. The run itself
    stays the run the adapter knows: same complete.json, still found once."""
    ramp = ews_ising.open_run(ising.runs["ramp"])
    latents = extract.read(ramp.path, ramp_latents.name)
    configs_schema = io.schema("configs", ramp.L)
    for key in extract.KEY:
        assert latents.schema.field(key).equals(configs_schema.field(key)), key
    readouts_ = [f"pc_{k}" for k in range(1, 9)]
    assert latents.column_names == [*extract.KEY, "surprise", *readouts_]
    assert all(latents.schema.field(c).type == pa.float32() for c in ["surprise", *readouts_])
    lattices = int(ramp.params["replicas"]) * int(ramp.params["blocks"])
    steps = latents.num_rows // lattices
    assert steps * lattices == latents.num_rows
    step = latents["step"].to_numpy().reshape(steps, lattices)
    replica = latents["replica"].to_numpy().reshape(steps, lattices)
    assert (step == step[:, :1]).all() and (np.diff(step[:, 0]) > 0).all()
    assert (replica == np.arange(lattices)).all()
    surprise = latents["surprise"].to_numpy().reshape(steps, lattices).T  # (batch, time)
    assert surprise.shape == (lattices, steps)
    assert (
        ramp.complete_sha256
        == io.read_json(ramp_latents / "manifest.json")["params"]["dataset"]["complete_sha256"]
    )
    complete, incomplete = ews_ising.find_runs(ising.root)
    assert sorted(r.run_id for r in complete) == sorted(p.name for p in ising.runs.values())
    assert incomplete == []


def test_the_untrained_twin_extracts_end_to_end(
    ising: jepa_ising.Ising, ramp_latents: Path
) -> None:
    twin = extract.extract(ising.training / train.TWIN, ising.runs["ramp"], "cpu")
    assert twin.name.endswith("-step0000000") and twin != ramp_latents
    assert io.read_json(twin / "manifest.json")["params"]["checkpoint"]["untrained"]
    trained = extract.read(ising.runs["ramp"], ramp_latents.name)["surprise"].to_numpy()
    untrained = extract.read(ising.runs["ramp"], twin.name)["surprise"].to_numpy()
    assert not np.array_equal(trained, untrained)
    assert set(extract.available(ising.runs["ramp"])) == {twin.name, ramp_latents.name}


def test_extract_refuses_a_training_that_stopped_early(
    tmp_path: Path, ising: jepa_ising.Ising
) -> None:
    """Catches latents from a collapsed or interrupted training passing for a finished one: its
    final.pt would give a full table naming only its step. Here a real training stops at step 1
    of 8 (floors no model can meet, patience 2), and its final.pt is refused with the verdict its
    complete.json records; its untrained twin is still a valid null."""
    dataset = data.load(ising.root, [ising.runs["train"].name], [ising.runs["val"].name])
    config = dataclasses.replace(
        jepa_ising.CONFIG, floors=Floors(spatial_std=100.0), monitor_every=1, patience=2
    )
    training = train.start(config, dataset, tmp_path, "stopped", "cpu")
    done = io.read_json(training / "complete.json")
    assert done["stopped_early"] and done["collapse"]["collapsed_at"] == 1
    with pytest.raises(
        DataError, match=r"step 1 of 8, .*records a collapse at step 1 \(L16 spatial"
    ):
        extract.extract(training / train.FINAL, ising.runs["ramp"], "cpu")
    assert not (ising.runs["ramp"] / extract.LATENTS / f"{training.name}-step0000001").exists()


def test_a_latents_table_is_written_once_and_resumes_after_an_interruption(
    tmp_path: Path, ising: jepa_ising.Ising
) -> None:
    """Catches a second extraction overwriting the first, and a resumed one that differs from a
    straight one (or keeps a half-written chunk)."""
    scored = shutil.copytree(ising.runs["scored"], tmp_path / "ising" / ising.runs["scored"].name)
    final = ising.training / train.FINAL
    out = extract.extract(final, scored, "cpu")
    straight = io.read_json(out / "complete.json")["files"]
    with pytest.raises(DataError, match="written once"):
        extract.extract(final, scored, "cpu")
    chunks = sorted(out.glob("*.parquet"))
    (out / "complete.json").unlink()
    chunks[-1].unlink()
    (out / "00000000000000000000.parquet.tmp").write_bytes(b"torn")
    extract.extract(final, scored, "cpu")
    assert io.read_json(out / "complete.json")["files"] == straight
    assert not list(out.glob("*.tmp"))
