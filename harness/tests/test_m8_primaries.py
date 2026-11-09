"""The M8 primaries' runner (harness/m8_primaries.py): its seeds, its data refusals and its pins,
on 12-step market toy trainings and a fake range manifest: no real data."""

import dataclasses
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import m4_pilot
import m8_primaries
import market_toy
import pytest
from tipping_point.book.features import FORMAT
from tipping_point.ising import io
from tipping_point.jepa import days, train


def test_train_refuses_any_seed_but_a_primary_or_a_spare(tmp_path: Path) -> None:
    assert [m8_primaries.role(s) for s in (101, 105, 106, 107)] == [
        "primary",
        "primary",
        "spare",
        "spare",
    ]
    for seed in (100, 108, 1):  # the pilot's, the next one, a default
        with pytest.raises(ValueError, match=f"seed {seed} is neither"):
            m8_primaries.run("btc", seed, tmp_path / "data", tmp_path / "out")
    assert m8_primaries.main(["train", "--asset", "eth", "--seed", "100", "--data", "x",
                              "--out", str(tmp_path / "out")]) == 1  # fmt: skip
    assert not (tmp_path / "out").exists()


def _manifest(
    folder: Path, days_: list[str], digests: dict[str, str] | None = None, **over: Any
) -> Path:
    """A fake training range folder: only its manifest.json, as book_assemble writes it, naming
    BTCUSDT on `days_`, each with its file's sha256 from `digests` (none when it is not there)."""
    folder.mkdir(parents=True, exist_ok=True)
    digests = digests or {}
    manifest = {
        "format": 1,
        "range": "training",
        "assembled": {"commit": "abc1234", "feature_format": FORMAT, "from": []},
        "days": {
            f"BTCUSDT/{d}": {"status": "imported", "reduced": {"sha256": digests.get(d)}}
            for d in days_
        },
        **over,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


@pytest.fixture
def toy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """m8_primaries.run on market toy trainings of STEPS = 12 (BTCUSDT, the run's seed and steps,
    which the tests pass): the data folder holds only a manifest naming the toy's days with the
    sha256 its source hands over, and days.load hands back those days. Returns the data folder."""
    data = market_toy.dataset()
    short = dict(checkpoint_every=4, monitor_every=4, pca_frames=128)
    monkeypatch.setattr(
        m4_pilot,
        "config",
        lambda asset, size, steps, seed: market_toy.config(**short, steps=steps, seed=seed),
    )
    monkeypatch.setattr(m8_primaries, "STEPS", 12)
    monkeypatch.setattr(days, "ParquetSource", lambda folder: None)
    monkeypatch.setattr(days, "load", lambda *a: data)
    read = {d["day"]: d["sha256"] for d in (*data.sources["train"], *data.sources["val"])}
    return _manifest(tmp_path / "data", [*read, "2023-09-30"], read)


def _crash_after(calls: int, real: Callable[..., Any]) -> Callable[..., Any]:
    count = 0

    def crashing(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        if count > calls:
            raise RuntimeError("injected crash")
        return real(*args, **kwargs)

    return crashing


def test_pins_refuse_an_incomplete_run_and_pin_it_once_resumed(
    toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    done = m8_primaries.run("btc", 101, toy, out, 12)
    with monkeypatch.context() as m:
        m.setattr(train._Trainer, "update", _crash_after(6, train._Trainer.update))
        with pytest.raises(RuntimeError, match="injected crash"):
            m8_primaries.run("btc", 106, toy, out, 12)
    smoke = m8_primaries.run("btc", 102, toy, out, 8)  # finished, short of STEPS
    with pytest.raises(ValueError, match="incomplete") as refused:
        m8_primaries.pins(out)
    why = str(refused.value)
    assert "BTCUSDT-S-s106" in why and "8 of 12 steps" in why and "s101" not in why
    assert not (out / m8_primaries.PINS).exists()

    shutil.rmtree(smoke.parents[1])
    spare = m8_primaries.run("btc", 106, toy, out, 12)  # resumes the killed run
    path = m8_primaries.pins(out)
    record = io.read_json(path)
    assert (record["size"], record["steps"]) == ("S", 12)
    first, second = record["checkpoints"]
    assert (first["seed"], first["role"], second["seed"], second["role"]) == (
        101,
        "primary",
        106,
        "spare",
    )
    assert first["run_id"] == done.name and second["run_id"] == spare.name
    assert (first["data_commit"], first["feature_format"]) == ("abc1234", FORMAT)
    files = io.read_json(done / "complete.json")["files"]
    assert first["trained_sha256"] == files[train.FINAL.as_posix()]
    assert first["twin_sha256"] == files[train.TWIN.as_posix()]
    assert first["trained_sha256"] != first["twin_sha256"]
    assert first["pca_sha256"] != second["pca_sha256"]  # each run's own basis
    assert first["scaling_sha256"] == second["scaling_sha256"]  # the same training days


def test_pins_refuse_a_run_whose_data_manifest_reaches_2024(toy: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    m8_primaries.run("btc", 101, toy, out, 12)
    copy = out / "btc-s101" / m8_primaries.DATA_MANIFEST
    manifest = io.read_json(copy)
    manifest["days"]["ETHUSDT/2024-01-01"] = {"status": "imported"}
    copy.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="2024-01-01"):
        m8_primaries.pins(out)


def test_train_refuses_a_range_that_is_not_training_data_of_this_format(
    toy: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    late = _manifest(tmp_path / "late", ["2023-12-31", "2024-01-01"])
    confirmatory = _manifest(tmp_path / "conf", ["2023-03-01"], range="confirmatory")
    old = _manifest(
        tmp_path / "old",
        ["2023-03-01"],
        assembled={"commit": "abc1234", "feature_format": FORMAT - 1, "from": []},
    )
    for folder, why in ((late, "2024-01-01"), (confirmatory, "not 'training'"), (old, "format")):
        with pytest.raises(ValueError, match=why):
            m8_primaries.run("btc", 101, folder, out, 12)
    assert not out.exists()


def test_train_refuses_a_training_whose_data_manifest_is_gone(toy: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    m8_primaries.run("btc", 101, toy, out, 12)
    copy = out / "btc-s101" / m8_primaries.DATA_MANIFEST
    copy.unlink()  # only jepa/ came back from the kernel, say
    other = _manifest(
        tmp_path / "other",
        ["2023-03-01"],
        assembled={"commit": "zzz9999", "feature_format": FORMAT, "from": []},
    )
    for folder in (toy, other):  # a finished training reads no days to check a new copy by
        with pytest.raises(RuntimeError, match=f"no {m8_primaries.DATA_MANIFEST}"):
            m8_primaries.run("btc", 101, folder, out, 12)
    assert not copy.exists()


def test_pins_refuse_a_data_manifest_without_the_files_the_training_read(
    toy: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    m8_primaries.run("btc", 101, toy, out, 12)
    copy = out / "btc-s101" / m8_primaries.DATA_MANIFEST
    changed, missing = io.read_json(copy), io.read_json(copy)
    changed["days"]["BTCUSDT/2023-03-08"]["reduced"]["sha256"] = "0" * 64  # a validation day
    del missing["days"]["BTCUSDT/2023-03-01"]  # a training day
    for manifest, day in ((changed, "2023-03-08"), (missing, "2023-03-01")):
        copy.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=rf"not its data manifest's files: \['{day}'\]"):
            m8_primaries.pins(out)
    assert not (out / m8_primaries.PINS).exists()


def test_pins_refuse_a_data_manifest_or_a_scaling_of_another_feature_format(
    toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    m8_primaries.run("btc", 101, toy, out, 12)
    copy = out / "btc-s101" / m8_primaries.DATA_MANIFEST
    manifest = io.read_json(copy)
    manifest["assembled"]["feature_format"] = FORMAT - 1
    copy.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=rf"feature formats \({FORMAT - 1}, {FORMAT}\)"):
        m8_primaries.pins(out)

    old = market_toy.dataset()  # days scaled by a record fitted under the format before
    old.sources["scaling"]["feature_format"] = FORMAT - 1
    monkeypatch.setattr(days, "load", lambda *a: old)
    m8_primaries.run("btc", 101, toy, tmp_path / "old", 12)
    with pytest.raises(ValueError, match=rf"feature formats \({FORMAT}, {FORMAT - 1}\)"):
        m8_primaries.pins(tmp_path / "old")


def test_pins_refuse_a_seed_trained_twice_or_outside_its_folder(toy: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    done = m8_primaries.run("btc", 101, toy, out, 12)
    again = done.with_name(done.name + "-copy")
    shutil.copytree(done, again)
    with pytest.raises(ValueError, match="btc seed 101: 2 trainings in btc-s101; keep one"):
        m8_primaries.pins(out)
    shutil.rmtree(again)
    (out / "btc-s101").rename(out / "btc-s102")
    with pytest.raises(ValueError, match="btc seed 101 is not in its folder btc-s101"):
        m8_primaries.pins(out)
    assert not (out / m8_primaries.PINS).exists()


def test_pins_list_every_training_they_refuse(
    toy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    copied = m8_primaries.run("btc", 101, toy, out, 12)
    (out / "btc-s101" / m8_primaries.DATA_MANIFEST).unlink()
    short = m8_primaries.run("btc", 102, toy, out, 8)
    foreign = out / "other" / "jepa" / "run"
    foreign.mkdir(parents=True)
    (foreign / "manifest.json").write_text("{}")
    capsys.readouterr()
    assert m8_primaries.main(["pins", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert f"{copied}: FileNotFoundError" in err and m8_primaries.DATA_MANIFEST in err
    assert f"{short}: ValueError: incomplete, 8 of 12 steps" in err
    assert f"{foreign}: KeyError: 'params'" in err
    assert not (out / m8_primaries.PINS).exists()


def test_pins_refuse_a_full_length_training_of_another_config(
    toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    primary = m4_pilot.config
    with monkeypatch.context() as m:  # all 12 steps, one other setting changed
        m.setattr(m4_pilot, "config", lambda *a: dataclasses.replace(primary(*a), lr=1e-3))
        m8_primaries.run("btc", 101, toy, out, 12)
    with pytest.raises(ValueError, match="not a primary's config"):
        m8_primaries.pins(out)


def test_train_refuses_another_range_manifest_than_its_copy(toy: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    m8_primaries.run("btc", 101, toy, out, 12)
    copy = out / "btc-s101" / m8_primaries.DATA_MANIFEST
    before = copy.read_bytes()
    manifest = io.read_json(toy / "manifest.json")
    manifest["assembled"]["commit"] = "def5678"  # the range assembled again
    (toy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="another range manifest"):
        m8_primaries.run("btc", 101, toy, out, 12)
    assert copy.read_bytes() == before


def test_train_finds_a_finished_seed_and_does_not_train_it_again(
    toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    done = m8_primaries.run("btc", 101, toy, out, 12)
    record = (done / "complete.json").read_bytes()

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("trained or read days again")

    for module, name in ((train, "start"), (train, "resume"), (days, "load")):
        monkeypatch.setattr(module, name, never)
    assert m8_primaries.run("btc", 101, toy, out, 12) == done
    assert (done / "complete.json").read_bytes() == record


def test_pins_refuse_a_training_that_fails_verify(toy: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    done = m8_primaries.run("btc", 101, toy, out, 12)
    (done / train.TWIN).write_bytes(b"another twin")
    with pytest.raises(ValueError, match=f"sha256 differs: {train.TWIN.as_posix()}"):
        m8_primaries.pins(out)
