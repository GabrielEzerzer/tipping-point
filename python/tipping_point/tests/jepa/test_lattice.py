"""The lattice pieces against what they promise: bit unpacking, canonicalisation, torus symmetries,
masks, locality, and the readouts' exact symmetries."""

import numpy as np
import pytest
import torch
from tipping_point.ising import io
from tipping_point.jepa import lattice
from tipping_point.jepa.lattice import (
    PATCH,
    Encoder,
    Jepa,
    MaskSpec,
    ModelSpec,
    augment,
    canonical,
    fixed_masks,
    masks,
    unpack,
)
from tipping_point.jepa.readout import readouts

SMALL = ModelSpec(dim=16, pred_width=16)


def _spins(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, shape, generator=gen) * 2 - 1).to(torch.int8)


def _model(spec: ModelSpec = SMALL, seed: int = 0) -> Jepa:
    model = Jepa(spec)
    model.initialise(torch.Generator().manual_seed(seed))
    return model.eval()


@pytest.mark.parametrize("L", [4, 6, 8, 16])
def test_device_unpacking_matches_the_simulator_reader(L: int) -> None:
    """Catches a bit order or row order that differs from ising.io (L = 6 pads its last byte)."""
    spins = _spins((40, L, L), L).numpy()
    packed = io.pack(spins)
    assert np.array_equal(unpack(torch.from_numpy(packed), L).numpy(), io.unpack(packed, L))
    assert unpack(torch.from_numpy(packed.reshape(5, 8, -1)), L).shape == (5, 8, L, L)


def test_canonical_flips_to_positive_and_is_exactly_even() -> None:
    """Catches a tie broken by something a global flip does not negate (the sign would leak
    back into every readout), and a burst whose frames are flipped one by one."""
    frames = _spins((64, 8, 8), 1)
    frames[:8] = torch.tensor([1, -1], dtype=torch.int8).repeat(32).view(8, 8)  # ties: sum 0
    frames[8:16] = -frames[:8]
    for spins, dims in ((frames, 2), (frames.view(8, 8, 8, 8), 3)):
        flipped = canonical(spins, dims)
        assert torch.equal(canonical(-spins, dims), flipped)
        totals = flipped.flatten(-dims).sum(-1, dtype=torch.int64)
        assert (totals >= 0).all()
        assert (flipped.flatten(-dims)[totals == 0][:, 0] == 1).all()  # a tie: first spin up
    bursts = frames.view(8, 8, 8, 8)
    signs = (canonical(bursts, 3) * bursts).flatten(1)  # +1 where kept, -1 where flipped
    assert (signs == signs[:, :1]).all()


def test_augmentation_is_one_torus_symmetry_per_burst() -> None:
    """Catches frames of a burst transformed differently (breaking its dynamics), a map that is
    not a symmetry of the torus (neighbours must stay neighbours), and a group smaller than
    D4 x translations (8 L^2 maps)."""
    L, B = 8, 8000
    sites = torch.arange(L * L).view(1, 1, L, L).expand(B, 3, L, L)  # output = its source site
    gen = torch.Generator().manual_seed(0)
    out = augment(sites, gen, d4=True, roll=True).reshape(B, 3, L * L)
    assert (out == out[:, :1]).all()
    source = out[:, 0]
    assert (source.sort(1).values == torch.arange(L * L)).all()
    i, j = source.view(B, L, L) // L, source.view(B, L, L) % L
    for shift in ((0, 1), (1, 0)):
        di = (i.roll(shift, (1, 2)) - i) % L
        dj = (j.roll(shift, (1, 2)) - j) % L
        assert (
            ((di == 0) & ((dj == 1) | (dj == L - 1))) | ((dj == 0) & ((di == 1) | (di == L - 1)))
        ).all()
    assert len({tuple(s.tolist()) for s in source}) == 8 * L * L
    d4 = augment(sites[:200], gen, d4=True, roll=False).reshape(200, 3, L * L)[:, 0]
    assert len({tuple(s.tolist()) for s in d4}) == 8
    assert augment(sites, gen, d4=False, roll=False) is sites


@pytest.mark.parametrize("L", [16, 32, 64])
def test_masks_hide_lattice_unit_blocks_and_buffer_the_context(L: int) -> None:
    """Catches coverage outside 20-25% (plus at most one block's overshoot), a context that sees a
    hidden cell or its neighbour, blocks that do not wrap, and blocks that grow with L (image
    fractions, under which a mask would mean something different at every size)."""
    spec, n = MaskSpec(), L // PATCH
    hidden, keep = masks(256, n, spec, torch.Generator().manual_seed(L), "cpu")
    share = hidden.flatten(1).float().mean(1)
    largest = (spec.block[1] // PATCH) ** 2 / n**2
    assert (share >= spec.coverage[0]).all() and (share <= spec.coverage[1] + largest).all()
    grown = hidden.clone()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= hidden.roll((dy, dx), (1, 2))
    assert torch.equal(keep, ~grown) and keep.flatten(1).any(1).all()
    assert (hidden[:, 0, :] & hidden[:, -1, :]).any()  # some block wraps around the torus
    starts = (hidden & ~hidden.roll(1, 2)).sum()  # runs of hidden cells along rows
    assert 2.0 <= hidden.sum() / starts <= 6.0  # 2-4 cells wide, merged now and then; any L
    with pytest.raises(ValueError, match="cells a side"):
        masks(1, 4, spec, torch.Generator(), "cpu")
    first, again = fixed_masks(n, spec, 7, 2), fixed_masks(n, spec, 7, 2)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))


def test_the_default_encoder_is_the_research_size_and_sees_exactly_14_sites() -> None:
    """Catches an encoder that sees further than 14 sites: then it is no longer one local
    function at every L, and its pooled variance stops being a translation average of local
    operators (the #27 locality argument)."""
    full = Jepa(ModelSpec())
    assert lattice.parameters(full.context) == 400_896  # 0.40M
    assert lattice.parameters(full.predictor) == 162_944  # 0.16M
    assert full.context.receptive_field == 14 < 16
    encoder = Encoder(dim=16)
    with torch.no_grad():
        for k, p in enumerate(encoder.parameters()):
            p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(k)))
    L = 32
    n = L // PATCH
    spins = _spins((1, L, L), 3)
    flipped = spins.clone()
    flipped[0, 0, 0] *= -1  # a spin of cell (0, 0)
    with torch.no_grad():
        moved = (encoder(spins) - encoder(flipped)).abs().amax(1)[0] > 0
    near = [(c - 3) % n for c in range(7)]  # cells -3..3 around cell 0, on the torus
    expected = torch.zeros(n, n, dtype=torch.bool)
    expected[np.ix_(near, near)] = True
    assert torch.equal(moved, expected)  # 7 cells = 14 sites a side, and nothing further
    assert encoder.receptive_field == 7 * PATCH


def test_one_set_of_weights_runs_at_every_size() -> None:
    model = _model()
    for L in (8, 16, 32):
        spins, n = _spins((3, L, L), L), L // PATCH
        assert model.context(spins).shape == (3, SMALL.dim, n, n)
        keep = torch.ones(3, n, n, dtype=torch.bool)
        _, pred = model.predict(spins, keep, torch.tensor([0, 3, 7]))
        assert pred.shape == (3, SMALL.dim, n, n)


def test_the_context_path_never_sees_hidden_spins() -> None:
    """Catches a mask applied once rather than after every layer, or a predictor reading hidden
    cells: changing the spins under hidden and buffer cells changes nothing downstream."""
    model, L = _model(), 16
    _, keep = masks(6, L // PATCH, MaskSpec(), torch.Generator().manual_seed(1), "cpu")
    spins = _spins((6, L, L), 2)
    sites = keep.repeat_interleave(PATCH, 1).repeat_interleave(PATCH, 2)
    other = torch.where(sites, spins, _spins((6, L, L), 9))
    assert not torch.equal(other, spins)
    delta = torch.arange(6)
    with torch.no_grad():
        a, pred_a = model.predict(spins, keep, delta)
        b, pred_b = model.predict(other, keep, delta)
    assert torch.equal(a, b) and torch.equal(pred_a, pred_b)
    assert (a.abs().amax(1)[~keep] == 0).all()


def test_readouts_are_exactly_even_and_invariant_to_rolls_by_the_patch() -> None:
    """Catches a readout that leaks the global sign (it must be even) or depends on where the
    lattice starts (a torus has no origin)."""
    model, L = _model(), 16
    hidden, keep = fixed_masks(L // PATCH, MaskSpec(), 5, 2)
    spins = _spins((32, L, L), 4)
    tied = spins.flatten(1).sum(1) == 0  # a tie goes by the first spin, which a roll moves
    spins[tied, 0, 0] *= -1
    a, surprise_a = readouts(model, canonical(spins, 2), hidden, keep)
    b, surprise_b = readouts(model, canonical(-spins, 2), hidden, keep)
    assert surprise_a is not None and surprise_b is not None
    assert torch.equal(a, b) and torch.equal(surprise_a, surprise_b)
    rolled, _ = readouts(model, canonical(spins.roll((2, 4), (1, 2)), 2))
    torch.testing.assert_close(rolled, a, rtol=0, atol=1e-5)
