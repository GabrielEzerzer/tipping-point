"""The planted-stripe toy (harness/jepa_toy.py) as the known answer: the JEPA learns it, probes
cannot tell the trained model from its untrained twin while prediction error can, and the
collapse detector catches a broken loss on its absolute floors while effective rank alone would
not."""

from pathlib import Path

import jepa_toy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.ising import io
from tipping_point.jepa import lattice, probes, train
from tipping_point.jepa.lattice import canonical, channel_norm
from tipping_point.jepa.readout import readouts
from tipping_point.jepa.train import evaluation_masks


def _table(run_dir: Path, name: str) -> dict[str, np.ndarray]:
    table = pa.concat_tables(pq.read_table(p) for p in io.chunks(run_dir, name))
    return {c: table[c].to_numpy() for c in table.column_names if c != "dim_std"}


def test_the_toy_is_learned_within_its_step_budget(toy_run: Path) -> None:
    """The research pass's known answer: from no skill (error above 1, the score of predicting
    nothing) to below 0.5 within the step budget, every floor held throughout."""
    monitor = _table(toy_run, "monitor")
    error = dict(zip(monitor["step"].tolist(), monitor["val_error"].tolist(), strict=True))
    assert error[0] > 1.0
    assert error[jepa_toy.CONFIG.steps] < 0.5
    assert not monitor["fired"].any()
    verdict = io.read_json(toy_run / "complete.json")["collapse"]
    assert not verdict["collapsed"] and verdict["first_fired"] is None


def test_probes_cannot_tell_trained_from_untrained_but_prediction_error_can(
    toy_run: Path,
) -> None:
    """The untrained null (#27): random local features already read the planted class and the
    noise level, so the trained model and its step-0 twin probe alike, and only prediction error
    separates them. #28 claims learning only as such a margin over the twin."""
    spins, label, noise = jepa_toy.bursts(768, seed=99)
    frames = canonical(torch.from_numpy(spins[:, 0]), 2)
    fit, test = slice(0, 512), slice(512, None)
    onehot = np.eye(jepa_toy.CLASSES)[label]
    scores = {}
    for name, path in (("trained", train.FINAL), ("untrained", train.TWIN)):
        ckpt = train.load_model(toy_run / path)
        hidden, keep = evaluation_masks(ckpt.config, jepa_toy.L)
        embedding, surprise = readouts(ckpt.model, frames, hidden, keep)
        assert surprise is not None
        x = embedding.double().numpy()
        classes = probes.ridge(x[fit], onehot[fit]).predict(x[test])
        level = probes.ridge(x[fit], noise[fit]).predict(x[test])
        scores[name] = {
            "class": probes.accuracy(label[test], classes),
            "noise": probes.r2(noise[test], level),
            "error": float(surprise.mean()),
        }
    trained, untrained = scores["trained"], scores["untrained"]
    assert min(trained["class"], untrained["class"]) >= 0.9
    assert abs(trained["class"] - untrained["class"]) <= 0.1
    assert min(trained["noise"], untrained["noise"]) >= 0.7
    assert abs(trained["noise"] - untrained["noise"]) <= 0.15
    assert untrained["error"] > 1.0
    assert trained["error"] < 0.5


def test_the_detector_catches_collapse_on_its_floors_while_effective_rank_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberately broken loss: targets from the context encoder itself, the gradient through
    both branches, VICReg off. The features shrink towards a constant while their last directions
    stay spread out (the research pass's mutant kept an effective rank of 11.6); the detector must
    fire on the absolute spread floors, never on effective rank."""

    def leaky(self: lattice.Jepa, spins: torch.Tensor) -> torch.Tensor:
        return channel_norm(self.context(spins))  # no stop-gradient: the trivial solution is open

    monkeypatch.setattr(lattice.Jepa, "targets", leaky)
    config = jepa_toy.config(
        var_weight=0.0,
        cov_weight=0.0,
        steps=80,
        monitor_every=10,
        checkpoint_every=40,
        pca_frames=128,
    )
    run_dir = train.start(config, jepa_toy.dataset(512, 64), tmp_path, "mutant", "cpu")
    done = io.read_json(run_dir / "complete.json")
    assert done["collapse"]["collapsed"] and done["stopped_early"]
    fired = done["collapse"]["first_fired"]["fired"]
    assert fired and not any("effective_rank" in f for f in fired)
    monitor, floors = _table(run_dir, "monitor"), config.floors
    assert (monitor["effective_rank"] > 4 * floors.effective_rank).all()
    assert monitor["spatial_std"][0] > 5 * floors.spatial_std  # healthy at the start
    assert monitor["pooled_std"][-1] < floors.pooled_std
    assert monitor["spatial_std"][-1] < floors.spatial_std
