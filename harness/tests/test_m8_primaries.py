"""The M8 primaries' runner (harness/m8_primaries.py): its seeds, its data refusals and its pins,
on 12-step market toy trainings and a fake range manifest: no real data."""

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


def _manifest(folder: Path, days_: list[str], **over: Any) -> Path:
    """A fake training range folder: only its manifest.json, as book_assemble writes it."""
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "range": "training",
        "assembled": {"commit": "abc1234", "feature_format": FORMAT, "from": []},
        "days": {f"BTCUSDT/{d}": {"status": "imported"} for d in days_},
        **over,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


@pytest.fixture
def toy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """m8_primaries.run on market toy trainings of STEPS = 12 (BTCUSDT, the run's seed and steps,
    which the tests pass): the data folder holds only a manifest, days.load hands back the toy's
    days. Returns the data folder."""
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
    return _manifest(tmp_path / "data", ["2023-03-01", "2023-09-30"])


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
