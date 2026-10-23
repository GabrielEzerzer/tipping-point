"""The objective's terms and schedules, gradient flow, the EMA, and capacity (overfitting one
batch)."""

import math
from pathlib import Path

import jepa_toy
import torch
from tipping_point.jepa import objective, train
from tipping_point.jepa.lattice import (
    PATCH,
    Jepa,
    MaskSpec,
    ModelSpec,
    canonical,
    channel_norm,
    masks,
    pairs,
    unpack,
)


def test_a_prediction_of_zero_scores_one() -> None:
    """The normalisation surprise is read against: LayerNorm'd targets have unit mean square per
    cell (up to LayerNorm's eps), so error 1 means no skill and anything below it is prediction."""
    target = channel_norm(torch.randn(4, 16, 8, 8, generator=torch.Generator().manual_seed(0)))
    hidden = torch.rand(4, 8, 8, generator=torch.Generator().manual_seed(1)) < 0.3
    zero = torch.zeros_like(target)
    ones = objective.errors(zero, target, hidden)
    torch.testing.assert_close(ones, torch.ones(4), rtol=0, atol=1e-4)
    loss = objective.prediction_loss(zero, target, hidden)
    torch.testing.assert_close(loss, torch.tensor(1.0), rtol=0, atol=1e-4)


def test_vicreg_matches_a_direct_computation_over_the_kept_tokens() -> None:
    """Catches masked sums that count hidden cells, or a covariance over frames, not tokens."""
    gen = torch.Generator().manual_seed(2)
    features = torch.randn(6, 8, 5, 5, generator=gen) * torch.linspace(0.1, 2.0, 8).view(1, 8, 1, 1)
    keep = torch.rand(6, 5, 5, generator=gen) < 0.6
    variance, covariance = objective.vicreg(features, keep, gamma=1.0)
    tokens = features.permute(0, 2, 3, 1)[keep]
    cov = torch.cov(tokens.T)
    expected_var = torch.relu(1.0 - torch.sqrt(cov.diagonal() + 1e-4)).mean()
    off = cov - torch.diag(cov.diagonal())
    torch.testing.assert_close(variance, expected_var)
    torch.testing.assert_close(covariance, off.pow(2).sum() / 8)


def test_schedules_run_from_their_start_to_their_end() -> None:
    total, warmup = 1000, 50
    assert math.isclose(objective.learning_rate(warmup - 1, total, 1e-3, 1e-6, warmup), 1e-3)
    assert objective.learning_rate(0, total, 1e-3, 1e-6, warmup) == 1e-3 / warmup
    assert math.isclose(objective.learning_rate(total, total, 1e-3, 1e-6, warmup), 1e-6)
    assert math.isclose(objective.cosine(0, total, 0.04, 0.4), 0.04)
    assert math.isclose(objective.cosine(total, total, 0.04, 0.4), 0.4)
    assert objective.momentum(0, total, 0.996, 1.0) == 0.996
    assert objective.momentum(total, total, 0.996, 1.0) == 1.0


def test_gradients_reach_every_trainable_parameter_and_never_the_target() -> None:
    """Catches a detached branch (a block, the mask token or the delta embedding left out of
    training), a target that learns by gradient, and an EMA that never moves."""
    trainer = train._Trainer(
        Path("unused/jepa/run"),
        "run",
        jepa_toy.config(),
        jepa_toy.dataset(64, 16),
        torch.device("cpu"),
    )
    target_before = [p.clone() for p in trainer.model.target.parameters()]
    context_before = [p.clone() for p in trainer.model.context.parameters()]
    trainer.update()
    for name, p in [
        *trainer.model.context.named_parameters(),
        *trainer.model.predictor.named_parameters(),
    ]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in trainer.model.target.parameters())
    assert all(
        not torch.equal(a, p)
        for a, p in zip(context_before, trainer.model.context.parameters(), strict=True)
    )
    moved = [
        not torch.equal(a, p)
        for a, p in zip(target_before, trainer.model.target.parameters(), strict=True)
    ]
    assert all(moved)  # the EMA pulled every target parameter towards the context encoder


def test_one_batch_can_be_overfitted() -> None:
    """Catches a model that cannot fit what it sees (a broken predictor path or loss): with the
    target frozen (momentum 1) and one fixed batch, the error falls to near zero."""
    L = jepa_toy.L
    data = jepa_toy.dataset(8, 8)
    gen = torch.Generator().manual_seed(4)
    bursts = canonical(unpack(data.train[L].packed, L), 3)
    context, target, delta = pairs(bursts, 7, gen)
    hidden, keep = masks(len(bursts), L // PATCH, MaskSpec(), gen, "cpu")
    model = Jepa(ModelSpec(dim=32, pred_width=32))
    model.initialise(torch.Generator().manual_seed(5))
    optimiser = torch.optim.AdamW(model.trainable(), lr=3e-3, weight_decay=0.0)
    targets = model.targets(target)
    losses = []
    for _ in range(60):
        optimiser.zero_grad()
        _, pred = model.predict(context, keep, delta)
        loss = objective.prediction_loss(pred, targets, hidden)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert losses[0] > 1.0 and losses[-1] < 0.05
