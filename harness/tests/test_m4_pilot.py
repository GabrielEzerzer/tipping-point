"""The M4 pilot's rules as code (harness/m4_pilot.py) and the merge of the book's range folders
(harness/book_assemble.py), on planted numbers and fake range folders: no training, no real data."""

import copy
import dataclasses
import hashlib
import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import book_assemble
import m4_pilot
import market_toy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.ising import io
from tipping_point.jepa import days, train


def test_the_split_is_a_validation_block_after_the_training_days() -> None:
    train, val = m4_pilot.split()
    assert (train[0], train[-1], len(train)) == (date(2023, 1, 18), date(2023, 8, 31), 226)
    assert (val[0], val[-1], len(val)) == (date(2023, 9, 1), date(2023, 9, 30), 30)
    assert m4_pilot.split(3) == (train[-3:], val[:3])


def _record(final: np.ndarray, twin: float = 1.5) -> dict:
    return {"final": list(final), "twin": [twin] * len(final)}


def test_s_is_chosen_only_when_it_matches_m_on_both_assets() -> None:
    """Planted: M's daily errors near 0.80; S 0.5 % worse matches, 5 % worse does not, and an M
    that does not beat its twin hands the choice to the owner."""
    rng = np.random.default_rng(1)
    m = 0.80 + 0.01 * rng.standard_normal(30)
    close, far = m * 1.005, m * 1.05
    both = {
        "btc": {"S": _record(close), "M": _record(m)},
        "eth": {"S": _record(close), "M": _record(m)},
    }
    assert m4_pilot.choose_size(both)["size"] == "S"
    one = {**both, "eth": {"S": _record(far), "M": _record(m)}}
    assert m4_pilot.choose_size(one)["size"] == "M"
    lazy = {**both, "btc": {"S": _record(close, twin=0.5), "M": _record(m)}}
    assert m4_pilot.choose_size(lazy)["size"] == "M"  # S not usable
    broken = {**both, "btc": {"S": _record(close), "M": _record(m, twin=0.5)}}
    assert m4_pilot.choose_size(broken)["size"] == "owner"


def test_nan_days_are_left_out_of_the_paired_difference() -> None:
    m = np.full(10, 0.8)
    s = m.copy()
    s[3] = np.nan
    ok, upper = m4_pilot.matches(s.tolist(), m.tolist())
    assert ok and upper == 0.0


def test_steps_are_10k_when_the_10k_model_is_within_2_percent_on_both_assets() -> None:
    flat = {"half": [0.81, 0.81], "final": [0.80, 0.80]}  # 1.25 % worse at 10k
    falling = {"half": [0.85, 0.85], "final": [0.80, 0.80]}  # 6 %
    assert m4_pilot.choose_steps({"btc": flat, "eth": flat})["steps"] == 10_000
    assert m4_pilot.choose_steps({"btc": flat, "eth": falling})["steps"] == 20_000


def test_floors_are_a_fifth_of_the_healthy_minimum_after_warm_up_rounded_down() -> None:
    assert m4_pilot.floor_value(0.87) == pytest.approx(0.1)
    assert m4_pilot.floor_value(0.5) == pytest.approx(0.1)
    assert m4_pilot.floor_value(31.0) == pytest.approx(6.0)
    steps = np.array([0, 500, 1000, 1250])
    a = {
        "step": steps,
        "spatial_std": np.array([0.01, 0.9, 0.6, 0.7]),
        "pooled_std": np.array([0.3, 0.3, 0.3, 0.3]),
        "effective_rank": np.array([40.0] * 4),
    }
    b = {**a, "spatial_std": np.array([0.9, 0.9, 0.8, 0.45])}
    got = m4_pilot.floors([a, b])  # a's 0.01 at step 0 is warm-up and does not count
    assert got == pytest.approx({"spatial_std": 0.09, "pooled_std": 0.06, "effective_rank": 8.0})


# ---------------------------------------------------------------- book_assemble


def _range(root: Path, name: str, symbols: tuple[str, ...], days: list[date], **over) -> Path:
    """A fake `market book` range folder holding one tiny feature file per (symbol, day)."""
    folder = root / name
    entries = {}
    for s in symbols:
        for d in days:
            rel = f"features/{s}/{d.isoformat()}.parquet"
            body = f"{s} {d}".encode()
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_bytes(body)
            reduced = {
                "format": book_assemble.FORMAT,
                "git_sha": "fcd5b6f" + "0" * 33,
                "dirty": False,
                "path": rel,
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            entries[f"{s}/{d.isoformat()}"] = {"status": "imported", "reduced": {**reduced, **over}}
    (folder / "manifest.json").write_text(json.dumps({"format": 1, "range": name, "days": entries}))
    return folder


@pytest.fixture
def training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """A 3-day training range, one folder per symbol as the Kaggle kernels made them."""
    monkeypatch.setitem(book_assemble.RANGES, "training", (date(2023, 1, 18), date(2023, 1, 20)))
    days = [date(2023, 1, 18) + timedelta(days=k) for k in range(3)]
    return [_range(tmp_path / s, "training", (s,), days) for s in ("BTCUSDT", "ETHUSDT")]


def test_assembly_merges_symbols_and_is_idempotent(training: list[Path], tmp_path: Path) -> None:
    out = tmp_path / "out"
    plan = book_assemble.plan(training, out, "fcd5b6f")
    assert plan.problems == [] and len(plan.copies) == 6
    book_assemble.apply(plan, out)
    manifest = json.loads((out / "training" / "manifest.json").read_text())
    assert len(manifest["days"]) == 6 and len(manifest["assembled"]["from"]) == 2
    again = book_assemble.plan(training, out, "fcd5b6f")
    assert again.copies == [] and again.kept == 6
    assert book_assemble.apply(again, out)[-1].endswith("manifest unchanged, 6 days")


def test_assembly_refuses_missing_duplicate_and_foreign_days(
    training: list[Path], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    [problem] = book_assemble.plan(training[:1], out, "fcd5b6f").problems
    assert "3 day(s) missing" in problem
    assert any(
        "in both" in p
        for p in book_assemble.plan([*training, training[0]], out, "fcd5b6f").problems
    )
    assert book_assemble.plan(training, out, "abc1234").problems  # another commit
    for prefix in ("", "fcd5b6", "FCD5B6F"):  # every sha starts with ""; 7 hex digits at least
        assert any("hex digits" in p for p in book_assemble.plan(training, out, prefix).problems)
    escape = _range(tmp_path / "f3", "training", ("BTCUSDT",), [date(2023, 1, 18)], path="../x")
    assert any("not features/" in p for p in book_assemble.plan([escape], out, "fcd5b6f").problems)
    old = book_assemble.FORMAT - 1
    dirty = _range(tmp_path / "f2", "training", ("BTCUSDT",), [date(2023, 1, 18)], format=old)
    assert any(
        f"feature format {old}" in p for p in book_assemble.plan([dirty], out, "fcd5b6f").problems
    )
    (training[1] / "features" / "ETHUSDT" / "2023-01-19.parquet").write_bytes(b"changed")
    bad = book_assemble.plan(training, out, "fcd5b6f")
    assert any("not its sha256" in p for p in bad.problems)
    with pytest.raises(ValueError, match="refused"):
        book_assemble.apply(bad, out)
    assert not out.exists()


def test_the_probe_scores_a_latent_that_knows_the_next_minute_near_0_and_noise_near_1() -> None:
    """Planted: 4 days of minutes whose feature means follow an AR(1); a latent holding minute
    t + 1's means scores near 0, one holding minute t's (a probe aimed at minute t or t - 1) about
    0.1, pure noise near 1; days 3 and 4 are scored."""
    rng = np.random.default_rng(0)
    n = 4 * days.MINUTES
    means = np.zeros((n, 3), np.float32)
    for t in range(1, n):
        means[t] = 0.95 * means[t - 1] + 0.3 * rng.standard_normal(3)
    frames = torch.from_numpy(np.repeat(means[:, None, :], 60, 1))
    valid = torch.ones(n, dtype=torch.bool)
    valid[100:110] = False
    minutes = days.Minutes(frames, valid, torch.zeros(n, dtype=torch.int64))
    ahead = m4_pilot.probe_errors(np.roll(means, -1, 0).astype(float), minutes)
    now = m4_pilot.probe_errors(means.astype(float), minutes)
    noise = m4_pilot.probe_errors(rng.standard_normal((n, 3)), minutes)
    assert len(ahead) == 2 and max(ahead) < 0.02
    assert min(now) > 0.05 and min(noise) > 0.9


# ---------------------------------------------------------------- run, on the market toy


@pytest.fixture
def toy_pilot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> days.Dataset:
    """m4_pilot.run on a 12-step market toy run, the step-SHORT copy at step 4; the data folder is
    never read, days.load hands back the toy's days."""
    toy = market_toy.dataset()
    short = market_toy.config(steps=12, checkpoint_every=4, monitor_every=4, pca_frames=128)
    monkeypatch.setattr(m4_pilot, "config", lambda *a, **k: short)
    monkeypatch.setattr(m4_pilot, "SHORT", 4)
    monkeypatch.setattr(m4_pilot, "HALF", Path("ckpt") / "step_0000004.pt")
    monkeypatch.setattr(days, "ParquetSource", lambda folder: None)
    monkeypatch.setattr(days, "load", lambda *a: toy)
    return toy


def _crash_after(calls: int, real: Callable[..., Any]) -> Callable[..., Any]:
    count = 0

    def crashing(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        if count > calls:
            raise RuntimeError("injected crash")
        return real(*args, **kwargs)

    return crashing


def test_the_step_short_copy_survives_a_crash_and_resume(
    toy_pilot: days.Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as m:
        m.setattr(train._Trainer, "update", _crash_after(6, train._Trainer.update))
        with pytest.raises(RuntimeError, match="injected crash"):
            m4_pilot.run("btc", "M", tmp_path / "data", tmp_path / "out")
    (run_dir,) = m4_pilot._runs(tmp_path / "out" / "btc-M")
    assert (run_dir / m4_pilot.HALF).is_file() and not (run_dir / "complete.json").exists()
    assert m4_pilot.run("btc", "M", tmp_path / "data", tmp_path / "out") == run_dir
    assert train.verify(run_dir) == []
    assert m4_pilot.HALF.as_posix() in io.read_json(run_dir / "complete.json")["files"]
    steps = pa.concat_tables(pq.read_table(p) for p in io.chunks(run_dir, "train"))["step"]
    assert steps.to_pylist() == list(range(1, 13))
    record = io.read_json(tmp_path / "out" / "btc-M" / m4_pilot.VALIDATION)
    assert record["half"] is not None and record["steps"] == 12


def test_a_finished_run_refuses_validation_on_other_days(
    toy_pilot: days.Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run trained with --limit whose validation.json is missing, re-run without it, must not
    probe the smoke model and record it as a full run."""
    out = tmp_path / "out"
    m4_pilot.run("btc", "M", tmp_path / "data", out)
    (out / "btc-M" / m4_pilot.VALIDATION).unlink()
    sources = copy.deepcopy(toy_pilot.sources)
    sources["val"] = sources["val"][:1]
    monkeypatch.setattr(days, "load", lambda *a: dataclasses.replace(toy_pilot, sources=sources))
    with pytest.raises(RuntimeError, match="other days"):
        m4_pilot.run("btc", "M", tmp_path / "data", out)
    assert not (out / "btc-M" / m4_pilot.VALIDATION).exists()
