"""Linear probes: the ridge and its scores on known answers, the control tasks and selectivity on
planted embeddings, and the library function and command line reporting a trained checkpoint
beside its untrained twin."""

import hashlib
from pathlib import Path

import jepa_ising
import numpy as np
import pytest
import torch
from tipping_point.ews.ising import open_run
from tipping_point.jepa import probes, train
from tipping_point.jepa.__main__ import main
from tipping_point.jepa.data import DataError

PLANTED_T = (1.8, 2.1, 2.269231, 2.5, 3.0)
SPREAD = {"abs_m": (0.9, 0.05), "energy": (-1.6, 0.04), "l2var": (3.0, 0.2)}  # plateau, frame


def _planted(rng: np.random.Generator, per: int) -> tuple[probes.Frames, np.ndarray]:
    """One planted run: `per` frames at each of PLANTED_T, every target a plateau value that
    depends on T plus a frame-level fluctuation; returns the frames and those fluctuations."""
    temperature = np.repeat(PLANTED_T, per)
    plateau = np.repeat(np.arange(len(PLANTED_T)), per)
    noise = rng.standard_normal((len(temperature), len(probes.TARGETS)))
    targets = {}
    for k, (target, (level, sd)) in enumerate(SPREAD.items()):
        steps = level * (1 + 0.1 * np.sin(np.arange(len(PLANTED_T)) * 1.7 + k))
        targets[target] = steps[plateau] + sd * noise[:, k]
    packed = torch.zeros((len(temperature), 1), dtype=torch.uint8)  # embeddings come planted
    replica = np.arange(len(temperature)) % 4
    return probes.Frames(16, packed, targets, replica, temperature), noise


def _embedding(
    f: probes.Frames, noise: np.ndarray, rng: np.random.Generator, frame_level: bool
) -> np.ndarray:
    """Features that know the plateau (one-hot of T), four of pure noise, and with frame_level
    the targets' own fluctuations, blurred."""
    onehot = (f.temperature[:, None] == np.array(PLANTED_T)[None, :]).astype(np.float64)
    parts = [onehot, rng.standard_normal((len(onehot), 4))]
    if frame_level:
        parts.append(noise + 0.3 * rng.standard_normal(noise.shape))
    return np.concatenate(parts, 1)


def test_ridge_scores_recover_known_answers() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 6)) * np.array([1.0, 10.0, 0.1, 1.0, 1.0, 5.0])
    y = x @ np.array([0.5, -0.2, 3.0, 0.0, 1.0, 0.1]) + 2.0
    model = probes.ridge(x[:300], y[:300], alpha=1e-6)
    assert probes.r2(y[300:], model.predict(x[300:])) > 0.9999
    labels = rng.integers(0, 3, 300)
    features = np.eye(3)[labels] + rng.normal(scale=0.05, size=(300, 3))
    onehot = probes.ridge(features, np.eye(3)[labels]).predict(features)
    assert probes.accuracy(labels, onehot) == 1.0
    assert probes.r2(y, np.full_like(y, y.mean())) == 0.0


def test_the_iid_control_scores_nothing_and_shuffle_within_t_keeps_only_temperature() -> None:
    """Planted embeddings: features that know only the plateau score the same R^2 on the real
    target as on its shuffle within T, so selectivity is 0; features that also carry the
    frame-level fluctuation score well above the shuffle. A Gaussian target scores R^2 ~ 0 on
    test either way. Catches a shuffle across plateaus (selectivity would be the whole R^2 on
    T-only features), no shuffle at all (0 on both), selectivity taken against the wrong control,
    and an iid column that still carries the target."""
    rng = np.random.default_rng(7)
    (fit, fit_noise), (test, test_noise) = _planted(rng, 600), _planted(rng, 600)
    targets = probes.controls([fit], [test], seed=0)
    for frame_level in (False, True):
        x_fit = _embedding(fit, fit_noise, rng, frame_level)
        x_test = _embedding(test, test_noise, rng, frame_level)
        rows = probes.score(x_fit, x_test, targets)
        assert [(r["target"], r["column"]) for r in rows] == [
            (t, c) for t in probes.TARGETS for c in probes.COLUMNS
        ]
        by = {(r["target"], r["column"]): r for r in rows}
        for t in probes.TARGETS:
            raw, iid, shuffled = (by[t, c]["r2"] for c in probes.COLUMNS)
            assert by[t, "raw"]["selectivity"] == raw - shuffled
            assert by[t, "iid"]["selectivity"] is None
            assert by[t, "shuffle_within_T"]["selectivity"] is None
            assert abs(iid) < 0.01, (t, frame_level, iid)
            assert 0.2 < shuffled < 0.95, (t, frame_level, shuffled)  # T explains part of it
            if frame_level:
                assert raw - shuffled > 0.03 and raw > 0.9, (t, raw, shuffled)
            else:
                assert abs(raw - shuffled) < 0.01, (t, raw, shuffled)


def test_controls_shuffle_within_each_runs_plateaus_and_match_the_targets_moments() -> None:
    """Catches a shuffle that crosses runs (two fit runs share every temperature here, with
    different values) or plateaus, one that leaves the order alone, iid draws with the wrong
    moments, and controls that change with anything but the seed."""
    rng = np.random.default_rng(8)
    fit = [_planted(rng, 200)[0], _planted(rng, 200)[0]]
    test = [_planted(rng, 200)[0]]
    fit[1].targets["abs_m"][:] += 0.5  # the second run's values are its own
    targets = probes.controls(fit, test, seed=3)
    assert set(targets) == {(t, c) for t in probes.TARGETS for c in probes.COLUMNS}
    for t in probes.TARGETS:
        for sets, values in zip((fit, test), targets[t, "shuffle_within_T"], strict=True):
            start = 0
            for f in sets:
                part, real = values[start : start + len(f.temperature)], f.targets[t]
                start += len(part)
                for T in PLANTED_T:
                    rows = f.temperature == T
                    assert np.array_equal(np.sort(part[rows]), np.sort(real[rows])), (t, T)
                assert not np.array_equal(part, real)
        real = targets[t, "raw"][0]
        for drawn in targets[t, "iid"]:
            assert abs(drawn.mean() - real.mean()) < 0.1 * real.std()
            assert drawn.std() == pytest.approx(real.std(), rel=0.1)
    again = probes.controls(fit, test, seed=3)
    assert all(np.array_equal(again[k][i], targets[k][i]) for k in targets for i in (0, 1))
    other = probes.controls(fit, test, seed=4)
    assert not np.array_equal(other["abs_m", "iid"][0], targets["abs_m", "iid"][0])


def test_probe_reports_trained_beside_twin(ising: jepa_ising.Ising) -> None:
    """Fitted on the validation seed and scored on a pre-registered seed's staircase, as #28
    runs them; without test runs, on the fit runs' odd replicas. Every row names its checkpoint
    by run id, step and sha256, its role and L."""
    final, twin = ising.training / train.FINAL, ising.training / train.TWIN
    frames = len(jepa_ising.PLATEAUS) * jepa_ising.SAVES * jepa_ising.REPLICAS * 8
    rows = probes.probe([final, twin], [ising.runs["val"]], [ising.runs["scored"]], device="cpu")
    assert [(r["role"], r["target"], r["column"]) for r in rows] == [
        (role, t, c) for role in ("trained", "twin") for t in probes.TARGETS for c in probes.COLUMNS
    ]
    sha = {
        role: hashlib.sha256(p.read_bytes()).hexdigest()
        for role, p in [("trained", final), ("twin", twin)]
    }
    assert all(r["sha256"] == sha[r["role"]] for r in rows)
    assert {(r["role"], r["step"]) for r in rows} == {
        ("trained", jepa_ising.CONFIG.steps),
        ("twin", 0),
    }
    assert {r["run_id"] for r in rows} == {ising.training.name}
    assert all(r["L"] == jepa_ising.L and r["n_fit"] == r["n_test"] == frames for r in rows)
    assert all(np.isfinite(r["r2"]) for r in rows)
    by = {(r["role"], r["target"], r["column"]): r for r in rows}
    assert by["twin", "abs_m", "raw"]["r2"] > 0.5  # random local features read |m| already
    assert by["twin", "abs_m", "raw"]["selectivity"] > 0.1  # frame by frame, not just per T
    for role in ("trained", "twin"):
        for t in probes.TARGETS:
            raw, shuffled = by[role, t, "raw"]["r2"], by[role, t, "shuffle_within_T"]["r2"]
            assert by[role, t, "raw"]["selectivity"] == raw - shuffled
            assert -0.2 < by[role, t, "iid"]["r2"] < 0.05  # 32 features fit 768 frames' noise
    split = probes.probe([final, twin], [ising.runs["val"]], limit=200, device="cpu")
    assert all(r["n_fit"] + r["n_test"] == 200 for r in split)
    with pytest.raises(DataError, match="ramp"):
        probes.frames(open_run(ising.runs["ramp"]))


def test_every_checkpoint_is_scored_on_the_same_control_draws(ising: jepa_ising.Ising) -> None:
    """Catches controls redrawn per checkpoint, which would make the trained-minus-twin
    difference of a control column noise: one checkpoint passed twice scores identically."""
    final = ising.training / train.FINAL
    rows = probes.probe([final, final], [ising.runs["val"]], limit=200, device="cpu")
    half = len(rows) // 2
    assert [r["r2"] for r in rows[:half]] == [r["r2"] for r in rows[half:]]


def test_probe_command_prints_both_checkpoints(
    ising: jepa_ising.Ising, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "probes.json"
    argv = ["probe", str(ising.training), "--fit", str(ising.runs["val"])]
    argv += ["--test", str(ising.runs["scored"]), "--device", "cpu", "--json", str(out)]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "trained" in printed and "twin" in printed and "L = 16" in printed
    assert all(f"\n{t} " in printed for t in probes.TARGETS)
    for label in [*probes.COLUMNS, "selectivity"]:
        assert printed.count(f" {label} ") == len(probes.TARGETS), label
    assert len(out.read_text()) > 0


def test_replica_sums_hold_what_r2_reads_and_a_resample_counts_each_replica_as_drawn() -> None:
    """r2_of on all a probe's test frames is r2, and on replicas drawn twice it is r2 on the frames
    written out twice: the data bootstrap needs no frame at hand. Catches frames attributed to the
    wrong run or replica, and a total that forgets the resample's weights."""
    rng = np.random.default_rng(9)
    test = [_planted(rng, 40)[0], _planted(rng, 40)[0]]
    y = np.concatenate([f.targets["energy"] for f in test])
    pred = y + rng.normal(0.0, 0.1, len(y))
    sums = probes.replica_sums(test, y, pred)
    assert [s.shape for s in sums] == [(4, 4), (4, 4)]
    assert probes.r2_of(np.concatenate(sums)) == pytest.approx(probes.r2(y, pred))
    first, second = test[0], test[1]
    offset = len(first.replica)
    rows = np.concatenate(
        [
            *[np.flatnonzero(first.replica == 1)] * 2,
            np.flatnonzero(second.replica == 3) + offset,
        ]
    )
    drawn = np.concatenate([sums[0][[1, 1]], sums[1][[3]]])
    assert probes.r2_of(drawn) == pytest.approx(probes.r2(y[rows], pred[rows]))


def test_scored_rows_are_score_rows_from_the_predictions() -> None:
    """The runner asks for predictions and rows apart, not to solve twice: they must be score's."""
    rng = np.random.default_rng(10)
    (fit, fit_noise), (test, test_noise) = _planted(rng, 100), _planted(rng, 100)
    targets = probes.controls([fit], [test], seed=0)
    x_fit, x_test = _embedding(fit, fit_noise, rng, True), _embedding(test, test_noise, rng, True)
    assert probes.scored(probes.fitted(x_fit, x_test, targets), targets) == probes.score(
        x_fit, x_test, targets
    )


def test_the_runner_embeds_the_checkpoint_the_prereg_pins(
    ising: jepa_ising.Ising, tmp_path: Path
) -> None:
    """The seam's real side: a checkpoint's table name finds its file under <data dir>/jepa/,
    final.pt or the twin, and gives what readout.embed gives on those frames; a file that is not the
    sha256 the prereg pins is refused before anything is scored, and so is one that is not there."""
    from ising_experiment import checkpoint_embedder
    from tipping_point.ews.ising import Checkpoint, IsingDataError, latents_name
    from tipping_point.jepa import readout

    run_id, steps = ising.training.name, jepa_ising.CONFIG.steps
    pins = {
        step: hashlib.sha256((ising.training / path).read_bytes()).hexdigest()
        for step, path in ((steps, train.FINAL), (0, train.TWIN))
    }
    passes = [Checkpoint(latents_name(run_id, step), pins[step], "primary") for step in (steps, 0)]
    embed = checkpoint_embedder(ising.root, passes, "cpu")
    frames = probes.frames(open_run(ising.runs["val"]), limit=16, seed=0)
    for checkpoint, path in zip(passes, (train.FINAL, train.TWIN), strict=True):
        model = train.load_model(ising.training / path, "cpu").model
        direct = readout.embed(model, frames.packed, frames.L, torch.device("cpu"), 65_536).numpy()
        assert np.array_equal(embed(checkpoint, [frames, frames]), np.concatenate([direct, direct]))
    wrong = [Checkpoint(passes[0].name, "0" * 64, "primary"), passes[1]]
    with pytest.raises(IsingDataError, match="not the sha256 the prereg pins") as refused:
        checkpoint_embedder(ising.root, wrong, "cpu")
    assert "final.pt" in str(refused.value) and "step_0000000" not in str(refused.value)
    absent = [Checkpoint("no-such-training-step0000008", "0" * 64, "primary")]
    with pytest.raises(IsingDataError, match="not the sha256 the prereg pins"):
        checkpoint_embedder(ising.root, absent, "cpu")
