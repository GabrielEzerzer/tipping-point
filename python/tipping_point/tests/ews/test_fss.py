"""Finite-size scaling fits and verdicts on data whose exponents are known exactly."""

from collections.abc import Callable

import numpy as np
import pytest
from tipping_point.ews import fss, ising
from tipping_point.ising.exact import TC

SIZES = [16, 32, 64, 128]
PerSize = dict[int, list[np.ndarray]]  # peak eps_L per seed, NaN where not located


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


def test_peak_position_finds_the_extremum_inside_the_band() -> None:
    """#98's statistic on hand-built series over a ramp cooling from eps_L = 10 to -6, band
    [6, -4]. A bump at eps_L = 1.5 is found there; read as falling, a dip is found instead; a
    larger bump outside the band is ignored; NaN values are never the extremum. Catches a band
    read in the wrong order, a direction dropped (the maximum taken for a falling indicator), or
    the extremum placed by sample index rather than by eps_L."""
    eps = np.linspace(10.0, -6.0, 161)  # 0.1 apart, hot first
    bump = lambda at: np.exp(-((eps - at) ** 2))  # noqa: E731
    rows = np.stack([bump(1.5), -bump(-2.0), bump(1.5) + 5 * bump(8.0), bump(-1.0)])
    rows[3, :60] = np.nan  # undefined down to eps_L = 4.1, inside the band
    band = (6.0, -4.0)
    rising = fss.peak_position(rows, eps, band, "greater")
    assert rising[0] == pytest.approx(1.5) and rising[2] == pytest.approx(1.5)
    assert rising[3] == pytest.approx(-1.0)
    assert fss.peak_position(rows[1], eps, band, "less") == pytest.approx([-2.0])
    assert np.isnan(fss.peak_position(rows[0], eps, band, "less")).all()  # a bump's minimum: edge


def test_a_peak_on_the_band_edge_is_not_located() -> None:
    """A row whose extremum is the band's first or last sample peaks there or beyond it: its
    position is unknown, so it is NaN (counted against the inconclusive rule), not the edge's
    eps_L, which would put every such replica at one scaled temperature and fake FSS. A row with
    no value inside the band is not located either."""
    eps = np.linspace(10.0, -6.0, 161)
    falling = 10.0 - np.abs(eps - 9.0)  # largest at eps_L = 9, above the band: hot edge
    rising = -eps  # largest at the coldest sample: cold edge
    inside = np.exp(-((eps - 5.9) ** 2) / 0.001)  # just inside the hot edge: located
    blank = np.full_like(eps, np.nan)
    peaks = fss.peak_position(
        np.stack([falling, rising, inside, blank]), eps, (6.0, -4.0), "greater"
    )
    assert np.isnan(peaks[:2]).all() and np.isnan(peaks[3])
    assert peaks[2] == pytest.approx(5.9)


def test_rolling_is_the_pipelines_detrend_window_and_lag() -> None:
    """The peak statistic reads the headline pipeline's own indicator: rolling(x) equals the
    rolling indicator of the pipeline's residuals, NaN until the first window is full."""
    from tipping_point.ews import indicators
    from tipping_point.ews.trend import Pipeline

    x = np.random.default_rng(6).standard_normal((3, 400)).cumsum(axis=1)
    pipe = Pipeline("ac", window=50, length=2, bandwidth=30.0, lag=2)
    expected = indicators.rolling_autocorrelation(indicators.gaussian_detrend(x, 30.0), 50, 2)
    assert np.allclose(pipe.rolling(x), expected, equal_nan=True)
    assert np.isnan(pipe.rolling(x)[:, :49]).all() and np.isfinite(pipe.rolling(x)[:, 49:]).all()


def test_inverse_nu_separates_scaling_from_monotone_peaks() -> None:
    """Peaks at a fixed eps_L give 1/nu = 1, on either side of Tc; peaks at a fixed T
    (eps_L ~ L) give 0. A position that changes side of Tc between sizes has no power law."""
    fixed_eps = np.full(len(SIZES), 1.3)
    fixed_T = fss.scaled_temperature(TC + 0.05, np.array(SIZES))
    assert fss.inverse_nu(SIZES, fixed_eps) == pytest.approx(1.0)
    assert fss.inverse_nu(SIZES, -fixed_eps) == pytest.approx(1.0)  # below Tc at every L
    assert fss.inverse_nu(SIZES, fixed_T) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(fss.inverse_nu(SIZES, np.array([1.0, 0.5, -0.2, 1.0])))
    assert np.isnan(fss.inverse_nu(SIZES, np.array([1.0, 0.5, 0.0, 1.0])))
    columns = np.stack([fixed_eps, -fixed_eps, [1.0, -1.0, 1.0, 1.0]], axis=1)
    assert fss.inverse_nu(SIZES, columns)[:2] == pytest.approx([1.0, 1.0])
    assert np.isnan(fss.inverse_nu(SIZES, columns)[2])  # one verdict per column (a draw)


def test_peak_convergence_on_scaling_peaks() -> None:
    rng = np.random.default_rng(3)
    samples = {L: [1.5 + 0.1 * rng.standard_normal(16) for _ in range(5)] for L in SIZES}
    samples[16][0][:3] = np.nan  # peaks not located are left out of the median
    inv_nu, ci = fss.peak_convergence(samples, 300, np.random.default_rng(4))
    assert inv_nu == pytest.approx(1.0, abs=0.05) and ci[0] < 1.0 < ci[1]
    assert fss.combine(fss.match(inv_nu, ci, 1.0, 0.1), fss.excludes(ci, 0.0, 0.1)) == fss.PASS


def test_dynamic_verdicts_on_planted_peak_positions() -> None:
    """#64, #98: the wiring of ising._dynamic. Peaks planted at one eps_L at every L give
    1/nu = 1 and pass; at one T (eps_L ~ L, an indicator merely monotone in T) 1/nu = 0 and fail;
    with more than half of one L's replicas not located (NaN, on the search band's edge) the
    verdict is inconclusive whatever the rest say, while exactly half is still measured. Only
    fss.dynamic.indicators are judged: skewness's peaks get no verdict row."""
    rng = np.random.default_rng(5)

    def planted(eps: Callable[[int], float]) -> PerSize:
        return {L: [eps(L) + 0.01 * rng.standard_normal(16) for _ in range(5)] for L in SIZES}

    def lost(per_size: PerSize, L: int, count: int) -> PerSize:
        """The same peaks with `count` of L's 80 replicas not located."""
        flat = np.concatenate(per_size[L])
        flat[:count] = np.nan
        return per_size | {L: np.split(flat, 5)}

    at_eps = planted(lambda L: 1.3)
    at_T = planted(lambda L: float(fss.scaled_temperature(TC + 0.05, L)))
    doc = {
        "fss": {
            "dynamic": {
                "indicators": ["variance", "ac"],
                "expected_inverse_nu": 1.0,
                "tolerance": 0.1,
            }
        },
        "scoring": {"bootstrap": 200, "ci_level": 0.95},
    }
    cases = {("variance", "dense"): at_eps, ("variance", "matched"): at_T}
    cases |= {("ac", "dense"): lost(at_eps, 16, 40), ("ac", "matched"): lost(at_eps, 16, 41)}
    cases |= {("skewness", "dense"): at_eps}
    rows = {(r["quantity"], r["cadence"]): r for r in ising._dynamic(doc, cases, rng)}
    assert ("inverse_nu_skewness", "dense") not in rows and len(rows) == 4
    scaling, monotone = rows["inverse_nu_variance", "dense"], rows["inverse_nu_variance", "matched"]
    assert scaling["estimate"] == pytest.approx(1.0, abs=0.02) and scaling["verdict"] == fss.PASS
    assert monotone["estimate"] == pytest.approx(0.0, abs=0.02) and monotone["verdict"] == fss.FAIL
    assert rows["inverse_nu_ac", "dense"]["verdict"] == fss.PASS
    assert rows["inverse_nu_ac", "matched"]["verdict"] == fss.INCONCLUSIVE
    assert all(r["expected"] == 1.0 and r["tolerance"] == 0.1 for r in rows.values())
    assert {r["role"] for r in rows.values()} == {"dynamic"}


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
