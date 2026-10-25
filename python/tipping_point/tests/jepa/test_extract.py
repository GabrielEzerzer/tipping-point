"""The latents table (#28's contract) on runs the simulator itself wrote: one row per configs row,
keyed and ordered like configs, readable the way the EWS adapter reads configs, its manifest
naming the checkpoint by sha256, written once and resumable."""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import jepa_ising
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.ews import ising as ews_ising
from tipping_point.ising import io
from tipping_point.jepa import data, extract, train
from tipping_point.jepa.__main__ import main
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
    assert manifest["params"]["selection"]["rule"] == "all"
    assert train.verify(ramp_latents) == [] and extract.verify(ramp_latents) == []


def test_a_checkpoint_outside_the_data_dir_is_named_by_its_sha256(
    tmp_path: Path, ising: jepa_ising.Ising
) -> None:
    """#94: a checkpoint outside the data dir (a training downloaded from Kaggle, say) went into
    the latents manifest as its full local path, account name included. It is named by its
    sha256, run id and step alone; inside the data dir, by its path relative to it."""
    elsewhere = shutil.copytree(ising.training, tmp_path / "downloads" / ising.training.name)
    ckpt = train.load_model(elsewhere / train.FINAL)
    run, rule = ews_ising.open_run(ising.runs["ramp"]), extract.selection("all")
    manifest = extract._manifest(tmp_path / "out", run, ckpt, torch.device("cpu"), rule)
    assert manifest["params"]["checkpoint"] == {
        "path": None,
        "sha256": _sha256(ising.training / train.FINAL),
        "run_id": ising.training.name,
        "step": jepa_ising.CONFIG.steps,
        "untrained": False,
    }


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


def test_extract_refuses_a_training_that_collapsed_on_its_last_step(
    tmp_path: Path, ising: jepa_ising.Ising
) -> None:
    """#89: a collapse confirmed at the final evaluation ends the run on step == steps, so it
    did not stop early and its final.pt passed the step check like a healthy seed's, although
    its complete.json records the collapse. Floors no model meets fire at every evaluation (steps
    0, 4 and 8) and patience 3 confirms the collapse on the third, the last step. Only the
    checkpoint's own training can vouch for it: a lone final.pt, or one dropped into another
    training's folder, is refused too."""
    dataset = data.load(ising.root, [ising.runs["train"].name], [ising.runs["val"].name])
    config = dataclasses.replace(jepa_ising.CONFIG, floors=Floors(spatial_std=100.0), patience=3)
    training = train.start(config, dataset, tmp_path, "late", "cpu")
    done = io.read_json(training / "complete.json")
    assert not done["stopped_early"] and done["collapse"]["collapsed_at"] == config.steps == 8
    ramp = ising.runs["ramp"]
    with pytest.raises(DataError, match=r"step 8 of 8, but .*records a collapse at step 8 \(L16"):
        extract.extract(training / train.FINAL, ramp, "cpu")
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(ising.training / train.FINAL, alone)
    with pytest.raises(DataError, match=r"no complete\.json"):
        extract.extract(alone / train.FINAL, ramp, "cpu")
    swapped = shutil.copytree(ising.training, tmp_path / "swapped")
    shutil.copy(training / train.FINAL, swapped / train.FINAL)  # healthy record, collapsed model
    with pytest.raises(DataError, match="does not list it with its sha256"):
        extract.extract(swapped / train.FINAL, ramp, "cpu")
    assert not (ramp / extract.LATENTS / f"{training.name}-step0000008").exists()


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


# ---------------------------------------------------------------- subsampled tables


def _copy(run: Path, root: Path) -> Path:
    """A simulator run copied to root/ising/ without the latents other tests wrote beside it."""
    return shutil.copytree(run, root / "ising" / run.name, ignore=shutil.ignore_patterns("latents"))


def _bursts(table: pa.Table) -> dict[float, list[int]]:
    """The bursts a table holds, per temperature."""
    pairs = zip(table["temperature"].to_pylist(), table["burst"].to_pylist(), strict=True)
    found: dict[float, set[int]] = {}
    for T, burst in pairs:
        found.setdefault(T, set()).add(burst)
    return {T: sorted(b) for T, b in sorted(found.items())}


@pytest.mark.parametrize(
    ("rule", "bursts"),
    [
        ("tc-plus", {1.5: [16], 2.269231: list(range(8, 16)), 3.5: [0]}),
        ("tc-only", {2.269231: list(range(8, 16))}),
    ],
)
def test_a_subsampled_table_is_the_full_tables_selected_rows(
    tmp_path: Path,
    ising: jepa_ising.Ising,
    capsys: pytest.CaptureFixture[str],
    rule: str,
    bursts: dict[float, list[int]],
) -> None:
    """Catches a rule keeping the wrong rows, or keeping the right ones with other values: on the
    scored staircase (bursts 0-7 at T = 3.5, 8-15 at Tc, 16-23 at 1.5), the table `extract
    --select` writes is the full table's rows that the rule keeps, row for row, keys and
    readouts alike. Of the run's four configs chunks the first holds no Tc frame and the last
    only bursts 22 and 23; each still gets its chunk, empty where the rule keeps nothing."""
    final = ising.training / train.FINAL
    whole = _copy(ising.runs["scored"], tmp_path / "all")
    full = extract.read(whole, extract.extract(final, whole, "cpu").name)
    run = _copy(ising.runs["scored"], tmp_path / rule)
    assert main(["extract", str(final), str(run), "--device", "cpu", "--select", rule]) == 0
    out = Path(capsys.readouterr().out.strip())
    table = extract.read(run, out.name)
    configs = io.read_table(run, "configs")
    expected = full.filter(extract.kept("staircase", configs, extract.selection(rule)))
    assert table.num_rows == 32 * sum(map(len, bursts.values()))  # 8 frames x 4 replicas a burst
    assert _bursts(table) == bursts
    assert table.equals(expected)
    chunks = [pq.read_metadata(p).num_rows for p in sorted(out.glob("*.parquet"))]
    assert len(chunks) == len(io.chunks(run, "configs")) == 4 and chunks[-1] == 0
    params = io.read_json(out / "manifest.json")["params"]
    assert params["selection"] == {"rule": rule, "tc_plateau": 2.269231, "every": 8}
    assert io.read_json(out / "complete.json")["rows"] == {"latents": table.num_rows}
    assert main(["verify", str(out)]) == 0
    assert capsys.readouterr().out.strip() == f"{out}: ok"


def test_ramps_and_null_runs_are_extracted_whole_under_any_rule(
    tmp_path: Path, ising: jepa_ising.Ising
) -> None:
    """Catches the Tc rule reaching runs it does not govern: the ramp crosses T = 2.269231, so a
    rule applied there would keep a few of its frames, not all of them."""
    for role in ("ramp", "null"):
        run = _copy(ising.runs[role], tmp_path)
        out = extract.extract(ising.training / train.TWIN, run, "cpu", select="tc-only")
        keys = list(extract.KEY)
        latents = extract.read(run, out.name).select(keys)
        assert latents.equals(io.read_table(run, "configs").select(keys)), role
        assert extract.verify(out) == [], role


def test_verify_refuses_a_latents_table_whose_rows_are_not_its_rules(
    tmp_path: Path, ising: jepa_ising.Ising, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a table whose files all hash right but whose rows are wrong: complete.json is
    rewritten after every edit, so only the row check can see a row dropped, a row of a burst the
    rule drops, a chunk gone, a miscount, or a dataset run that changed underneath."""
    run = _copy(ising.runs["scored"], tmp_path)
    out = extract.extract(ising.training / train.TWIN, run, "cpu", select="tc-plus")
    chunk = sorted(out.glob("*.parquet"))[1]  # bursts 8-13 of the Tc plateau: 192 rows
    original = pq.read_table(chunk)

    def rewrite(table: pa.Table | None, path: Path = chunk) -> list[str]:
        path.unlink()
        if table is not None:
            pq.write_table(table, path, compression="zstd")
        (out / "complete.json").unlink()
        extract._complete(out)
        assert train.verify(out) == []  # every file hashes as complete.json says
        return extract.verify(out)

    assert rewrite(original.slice(1)) == [
        f"{chunk.name}: 191 rows, not exactly the 192 configs rows the tc-plus selection keeps"
    ]
    assert main(["verify", str(out)]) == 1
    assert f"{out}: {chunk.name}: 191 rows" in capsys.readouterr().out
    configs = pq.read_table(run / "configs" / chunk.name, columns=list(extract.KEY))
    dropped = configs.filter(configs["burst"].to_numpy() == 6).slice(0, 1)  # T = 3.5, not kept
    readouts_ = original.slice(0, 1).columns[len(extract.KEY) :]
    extra = pa.Table.from_arrays([*dropped.columns, *readouts_], schema=original.schema)
    assert rewrite(pa.concat_tables([original, extra])) == [
        f"{chunk.name}: 193 rows, not exactly the 192 configs rows the tc-plus selection keeps"
    ]
    assert main(["verify", str(out)]) == 1
    assert rewrite(original) == []
    empty = sorted(out.glob("*.parquet"))[-1]  # bursts 22 and 23: tc-plus keeps none of them
    assert rewrite(None, empty) == [f"missing chunk {empty.name}"]
    record = io.read_json(out / "complete.json")
    record["rows"]["latents"] += 1
    (out / "complete.json").write_text(json.dumps(record), encoding="utf-8")
    assert extract.verify(out)[-1] == "complete.json counts 321 rows, the chunks hold 320"
    (run / "complete.json").write_bytes((run / "complete.json").read_bytes() + b"\n")
    assert extract.verify(out) == [
        f"{run.name}: complete.json differs from the one this table was read from"
    ]
    assert extract.verify(ising.training) == extract.verify(run) == []  # not latents tables


def _schedule(run: Path, **params: object) -> None:
    """Rewrite a copied run's recorded schedule parameters, the input extract's checks read."""
    manifest = io.read_json(run / "manifest.json")
    manifest["params"]["schedule"]["params"].update(params)
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_extract_refuses_a_rule_it_cannot_apply(tmp_path: Path, ising: jepa_ising.Ising) -> None:
    """Catches a subsampled table that silently lacks the plateau it exists for, one in 8 bursts
    taken unevenly across plateaus, and a resumed extraction mixing two rules in one folder;
    nothing is written for any of them."""
    twin = ising.training / train.TWIN
    with pytest.raises(ValueError, match="selection must be one of"):
        extract.extract(twin, ising.runs["scored"], "cpu", select="tc")
    run = _copy(ising.runs["scored"], tmp_path / "no-tc")
    _schedule(run, temperatures=[3.5, 2.3, 1.5])
    for rule in ("tc-plus", "tc-only"):
        with pytest.raises(DataError, match=r"no plateau at T = 2\.269231 for tc-"):
            extract.extract(twin, run, "cpu", select=rule)
    assert not (run / extract.LATENTS).exists()
    run = _copy(ising.runs["scored"], tmp_path / "uneven")
    _schedule(run, saves=12)  # plateaus would keep bursts 0 and 8, then 16 alone, then 24 and 32
    with pytest.raises(DataError, match="saves 12 bursts a plateau: one in 8 by burst index"):
        extract.extract(twin, run, "cpu", select="tc-plus")
    assert not (run / extract.LATENTS).exists()
    assert extract.extract(twin, run, "cpu", select="tc-only").is_dir()  # no 1-in-8 to even out
    run = _copy(ising.runs["scored"], tmp_path / "resumed")
    out = extract.extract(twin, run, "cpu", select="tc-only")
    (out / "complete.json").unlink()  # as if interrupted
    with pytest.raises(DataError, match="begun from another checkpoint, dataset or selection"):
        extract.extract(twin, run, "cpu")
    assert not (out / "complete.json").exists()
