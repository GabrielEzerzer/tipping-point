"""Finite-size scaling fits and verdicts on data whose exponents are known exactly."""

import numpy as np
import pytest
from tipping_point.ews import fss
from tipping_point.ising.exact import TC

SIZES = [16, 32, 64, 128]


def test_scaled_temperature_round_trips_and_is_zero_at_tc() -> None:
    assert fss.scaled_temperature(TC, 64) == 0.0
    assert fss.scaled_temperature(TC * 1.1, 20) == pytest.approx(2.0)
    eps = np.array([-2.0, 0.5, 6.0])
    assert np.allclose(fss.scaled_temperature(fss.temperature(eps, 32), 32), eps)


def test_power_law_recovers_exponents_and_the_correction_removes_a_1_over_L_term() -> None:
    """Catches a fit on values instead of their logs, or the slope and intercept swapped."""
    L = np.array(SIZES, dtype=float)
    assert fss.power_law(SIZES, 3.0 * L**1.75) == pytest.approx(1.75)
    assert fss.power_law(SIZES, -1.5 * L**0.0) == pytest.approx(0.0)  # sign-free: |I|
    corrected = 2.0 * L ** (-0.125) * np.exp(4.0 / L)  # a finite-size correction
    assert fss.power_law(SIZES, corrected) != pytest.approx(-0.125, abs=0.01)
    assert fss.power_law(SIZES, corrected, correction=True) == pytest.approx(-0.125)
    assert np.isnan(fss.power_law(SIZES[:2], corrected[:2], correction=True))
    columns = np.stack([L**1.0, L**2.0], axis=1)  # one fit per column
    assert fss.power_law(SIZES, columns) == pytest.approx([1.0, 2.0])


def test_fit_exponent_bootstraps_a_ci_that_covers_the_truth() -> None:
    """Replicas scatter around 5 L^1.75; seeds differ too. The CI must cover 1.75 and be tight
    enough to judge a tolerance of 0.1."""
    rng = np.random.default_rng(1)
    samples = {
        L: [
            5 * L**1.75 * (1 + 0.02 * rng.standard_normal()) + rng.normal(0, 0.05 * L**1.75, 16)
            for _ in range(5)
        ]
        for L in SIZES
    }
    fit = fss.fit_exponent(samples, lambda L, a: float(np.mean(a)), 400, np.random.default_rng(2))
    assert fit.ci[0] < 1.75 < fit.ci[1]
    assert fit.kappa == pytest.approx(1.75, abs=0.02) and fit.ci[1] - fit.ci[0] < 0.1
    assert set(fit.values) == set(SIZES)
    assert fss.match(fit.kappa, fit.ci, 1.75, 0.1) == fss.PASS


def test_inverse_nu_separates_scaling_from_monotone_alarms() -> None:
    """Alarms at a fixed eps_L give 1/nu = 1; alarms at a fixed T (eps_L ~ L) give 0."""
    fixed_eps = np.full(len(SIZES), 1.3)
    fixed_T = fss.scaled_temperature(TC + 0.05, np.array(SIZES))
    assert fss.inverse_nu(SIZES, fixed_eps) == pytest.approx(1.0)
    assert fss.inverse_nu(SIZES, fixed_T) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(fss.inverse_nu(SIZES, np.array([1.0, 0.5, -0.2, 1.0])))  # below Tc


def test_alarm_convergence_on_scaling_alarms() -> None:
    rng = np.random.default_rng(3)
    samples = {L: [1.5 + 0.1 * rng.standard_normal(16) for _ in range(5)] for L in SIZES}
    samples[16][0][:3] = np.nan  # misses are left out of the median
    inv_nu, ci = fss.alarm_convergence(samples, 300, np.random.default_rng(4))
    assert inv_nu == pytest.approx(1.0, abs=0.05) and ci[0] < 1.0 < ci[1]
    assert fss.combine(fss.match(inv_nu, ci, 1.0, 0.1), fss.excludes(ci, 0.0, 0.1)) == fss.PASS


@pytest.mark.parametrize(
    ("estimate", "ci", "verdict"),
    [
        (1.70, (1.66, 1.75), fss.PASS),
        (1.60, (1.55, 1.65), fss.FAIL),  # precise, and off by more than the tolerance
        (1.75, (1.40, 2.10), fss.INCONCLUSIVE),  # right on, but too wide to have failed
        (2.30, (2.10, 2.50), fss.FAIL),  # wide, yet wholly outside 1.65-1.85
        (1.20, (1.00, 1.40), fss.FAIL),
        (np.nan, (1.0, 2.0), fss.INCONCLUSIVE),
    ],
)
def test_match_verdicts(estimate: float, ci: tuple[float, float], verdict: str) -> None:
    assert fss.match(estimate, ci, 1.75, 0.1) == verdict


def test_excludes_and_combine_verdicts() -> None:
    assert fss.excludes((0.2, 0.4), 0.0, 0.1) == fss.PASS
    assert fss.excludes((-0.05, 0.05), 0.0, 0.1) == fss.FAIL  # precisely no growth
    assert fss.excludes((-0.5, 0.9), 0.0, 0.1) == fss.INCONCLUSIVE
    assert fss.combine(fss.PASS, fss.INCONCLUSIVE) == fss.INCONCLUSIVE
    assert fss.combine(fss.INCONCLUSIVE, fss.FAIL, fss.PASS) == fss.FAIL
    assert fss.combine(fss.PASS, fss.PASS) == fss.PASS
