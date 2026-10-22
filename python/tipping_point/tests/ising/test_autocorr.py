"""The tau_int instrument against series whose answer is known in closed form.

AR(1), x_t = phi x_(t-1) + noise, has rho(t) = phi^t, so tau = 1 + 2 sum phi^t
= (1 + phi) / (1 - phi): 199 at phi = 0.99. White noise has tau = 1.
"""

import numpy as np
import pytest
from scipy.signal import lfilter
from tipping_point.ising import autocorr


def _ar1(phi: float, replicas: int, n: int, seed: int) -> np.ndarray:
    """Stationary AR(1) series, one per row (x_0 drawn from the stationary distribution)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((replicas, n))
    noise[:, 0] /= np.sqrt(1 - phi**2)
    series = lfilter([1.0], [1.0, -phi], noise, axis=1)
    assert isinstance(series, np.ndarray)
    return series


def test_ar1_tau_is_199() -> None:
    """Catches a wrong convention (tau = 1/2 + sum rho gives ~100), a wrapped FFT, or a window
    that stops too early (biased low) or never stops (noisy)."""
    x = _ar1(0.99, replicas=128, n=2**16, seed=1)
    tau = autocorr.estimate(x)
    assert tau.tau == pytest.approx(199, rel=0.10)
    assert tau.window >= 5 * tau.tau and tau.reliable


def test_binning_plateau_agrees_with_sokal() -> None:
    """The independent cross-check: block-mean variance must plateau near the same tau."""
    x = _ar1(0.99, replicas=128, n=2**16, seed=2)
    tau = autocorr.estimate(x)
    sizes, ratios = autocorr.binning(x)
    assert ratios[0] == pytest.approx(1.0)  # blocks of one sample: no gain
    assert np.all(np.diff(ratios) > -0.05 * ratios[1:])  # rises towards the plateau
    assert tau.binning == pytest.approx(tau.tau, rel=0.15)
    assert sizes[-1] >= 10 * tau.tau  # the plateau value comes from blocks much longer than tau


def test_white_noise_has_tau_one() -> None:
    """Catches an off-by-one in the window sum (rho(0) counted twice gives tau = 2)."""
    x = np.random.default_rng(3).standard_normal((32, 10_000))
    assert autocorr.estimate(x).tau == pytest.approx(1.0, abs=0.1)


def test_fft_autocorrelation_matches_a_direct_sum() -> None:
    """Catches the circular-correlation wrap that zero-padding exists to prevent."""
    x = _ar1(0.8, replicas=1, n=300, seed=4)[0]
    y = x - x.mean()
    direct = np.array([np.dot(y[: len(y) - t], y[t:]) for t in range(len(y))]) / np.dot(y, y)
    assert np.allclose(autocorr.autocorrelation(x), direct, atol=1e-12)


def test_the_forced_zero_tail_never_passes_as_a_window() -> None:
    """Subtracting the mean makes tau(n-1) = 0 exactly; a window taken there reported tau = 0 as
    reliable (n >= 50 * 0). Series far shorter than tau must come back unreliable, with tau > 0."""
    for n in (2, 3, 5, 20, 100, 600):
        tau = autocorr.estimate(_ar1(0.99, replicas=4, n=n, seed=n))
        assert not tau.reliable and tau.tau > 0, (n, tau)


def test_short_series_are_flagged_unreliable() -> None:
    """A series shorter than ~50 tau cannot be trusted, and the result must say so."""
    x = _ar1(0.99, replicas=4, n=2000, seed=5)
    assert not autocorr.estimate(x).reliable
