"""Surrogates keep what they promise to keep, and draw everything else from the generator."""

import numpy as np
import pytest
from scipy.signal import lfilter
from tipping_point.ews import surrogates as sur


def _ar1(phi: float, rows: int, n: int, seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).standard_normal((rows, n))
    noise[:, 0] /= np.sqrt(1 - phi**2)
    out = lfilter([1.0], [1.0, -phi], noise, axis=1)
    assert isinstance(out, np.ndarray)
    return out


@pytest.mark.parametrize("length", [200, 201])
def test_phase_randomised_keeps_the_periodogram_mean_and_variance(length: int) -> None:
    """Catches a phase drawn for the mean or for the (real) Nyquist term, either of which would
    break the exact amplitude match, and a complex or mis-shaped output."""
    x = _ar1(0.9, rows=3, n=length, seed=1) + 4.0
    s = sur.phase_randomised(x, 5, np.random.default_rng(2))
    assert s.shape == (3, 5, length) and s.dtype == np.float64
    amplitudes = np.abs(np.fft.rfft(x, axis=-1))[:, None, :]
    assert np.allclose(np.abs(np.fft.rfft(s, axis=-1)), amplitudes, rtol=1e-9, atol=1e-9)
    assert np.allclose(s.mean(axis=-1), x.mean(axis=-1)[:, None])
    assert np.allclose(s.var(axis=-1), x.var(axis=-1)[:, None])
    assert not np.allclose(s[:, 0], s[:, 1])  # the phases really are drawn


def test_surrogates_are_reproducible_from_the_generator() -> None:
    x = _ar1(0.8, rows=2, n=100, seed=3)
    for draw in (
        lambda g: sur.phase_randomised(x, 4, g),
        lambda g: sur.ar1_fitted(x, 4, g),
        lambda g: sur.block_bootstrap(x, 10, 4, g),
    ):
        assert np.array_equal(draw(np.random.default_rng(7)), draw(np.random.default_rng(7)))
        assert not np.array_equal(draw(np.random.default_rng(7)), draw(np.random.default_rng(8)))


def test_ar1_fit_recovers_the_parameters_and_draws_stationary_series() -> None:
    """Catches a fit on uncentred data (phi pulled towards 1 by the mean) or a start that is not
    stationary (the first samples' variance too small)."""
    x = 3.0 + 0.5 * _ar1(0.9, rows=4, n=50_000, seed=4)
    model = sur.fit_ar1(x)
    assert np.allclose(model.phi, 0.9, atol=0.01)
    assert np.allclose(model.sigma, 0.5, rtol=0.01)
    assert np.allclose(model.mean, 3.0, atol=0.1)
    draws = sur.ar1(
        sur.AR1(np.array(0.9), np.array(0.5), np.array(3.0)), 300, 20_000, np.random.default_rng(5)
    )
    assert draws.shape == (20_000, 300)
    stationary = 0.25 / (1 - 0.81)
    assert draws[:, 0].var() == pytest.approx(stationary, rel=0.05)  # from the very first step
    assert draws[:, -1].var() == pytest.approx(stationary, rel=0.05)
    lag1 = np.corrcoef(draws[:, 100], draws[:, 101])[0, 1]
    assert lag1 == pytest.approx(0.9, abs=0.01)


def test_fitted_surrogates_follow_their_own_row() -> None:
    x = np.stack([_ar1(0.2, 1, 4000, 6)[0], 10.0 + _ar1(0.95, 1, 4000, 7)[0]])
    s = sur.ar1_fitted(x, 50, np.random.default_rng(8))
    assert s.shape == (2, 50, 4000)
    refit = sur.fit_ar1(s.reshape(-1, 4000)).phi.reshape(2, 50).mean(axis=1)
    assert refit == pytest.approx([0.2, 0.95], abs=0.03)
    assert s[1].mean() == pytest.approx(10.0, abs=0.5)


def test_block_bootstrap_rebuilds_rows_from_contiguous_blocks() -> None:
    x = np.arange(100.0)[None, :] + np.array([[0.0], [1000.0]])
    s = sur.block_bootstrap(x, 10, 3, np.random.default_rng(9))
    assert s.shape == (2, 3, 100)
    for row in range(2):
        for b in range(3):
            blocks = s[row, b].reshape(10, 10)
            assert np.all(np.diff(blocks, axis=1) == 1.0)  # each block is a run of the original
            assert np.all((blocks >= x[row, 0]) & (blocks <= x[row, -1]))
    with pytest.raises(ValueError):
        sur.block_bootstrap(x, 101, 1, np.random.default_rng(0))
