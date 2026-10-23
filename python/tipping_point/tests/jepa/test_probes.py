"""Linear probes: the ridge and its scores on known answers, and the library function and command
line reporting a trained checkpoint beside its untrained twin."""

from pathlib import Path

import jepa_ising
import numpy as np
import pytest
from tipping_point.ews.ising import open_run
from tipping_point.jepa import probes, train
from tipping_point.jepa.__main__ import main
from tipping_point.jepa.data import DataError


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


def test_probe_reports_trained_beside_untrained(ising: jepa_ising.Ising) -> None:
    """Fitted on the validation seed and scored on a pre-registered seed's staircase, as #28
    runs them; without test runs, on the fit runs' odd replicas."""
    checkpoints = {
        "trained": ising.training / train.FINAL,
        "untrained": ising.training / train.TWIN,
    }
    frames = len(jepa_ising.PLATEAUS) * 4 * jepa_ising.REPLICAS * 8
    rows = probes.probe(checkpoints, [ising.runs["val"]], [ising.runs["scored"]], device="cpu")
    assert [(r["checkpoint"], r["target"]) for r in rows] == [
        (c, t) for c in checkpoints for t in probes.TARGETS
    ]
    assert all(np.isfinite(r["r2"]) and r["n_fit"] == r["n_test"] == frames for r in rows)
    assert {r["step"] for r in rows} == {0, jepa_ising.CONFIG.steps}
    by = {(r["checkpoint"], r["target"]): r["r2"] for r in rows}
    assert by["untrained", "abs_m"] > 0.5  # random local features read |m| already
    split = probes.probe(checkpoints, [ising.runs["val"]], limit=200, device="cpu")
    assert all(r["n_fit"] + r["n_test"] == 200 for r in split)
    with pytest.raises(DataError, match="ramp"):
        probes.frames(open_run(ising.runs["ramp"]))


def test_probe_command_prints_both_checkpoints(
    ising: jepa_ising.Ising, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "probes.json"
    argv = ["probe", str(ising.training), "--fit", str(ising.runs["val"])]
    argv += ["--test", str(ising.runs["scored"]), "--device", "cpu", "--json", str(out)]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "trained" in printed and "untrained" in printed
    assert all(f"\n{t} " in printed for t in probes.TARGETS)
    assert len(out.read_text()) > 0
