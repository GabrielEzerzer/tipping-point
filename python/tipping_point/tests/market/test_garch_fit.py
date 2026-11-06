"""The floor's GARCH(1,1)-t fit (harness/market_garch.py, #29 S5) and the minute process it feeds.

A planted process recovered from a poor start; the bridge from minutes to 10-second sub-steps,
whose sums are the minute GARCH itself; the refusals at the edges of a finite variance; and the
command on a fetched span, which never reads past the pilot year.
"""

import math
from pathlib import Path

import market_garch
import numpy as np
import pyarrow.parquet as pq
import pytest
from tipping_point.market import nulls
from tipping_point.market.nulls import Garch

# alpha 0.08, beta 0.9: a half-life of 34 minutes, 1e-6 per minute (73% a year)
PLANTED = Garch(1e-6 * (1 - 0.98), 0.08, 0.90, 5.0)
N = 200_000


@pytest.fixture(scope="module")
def planted_returns() -> np.ndarray:
    return nulls.simulate(PLANTED, N, np.random.default_rng(29))


def test_the_fit_recovers_a_planted_process_from_a_poor_start(planted_returns: np.ndarray) -> None:
    """From persistence 0.9, an even split and nu = 10, against 0.98, 0.08 / 0.98 and 5."""
    assert market_garch.START == (0.9, 0.5, 10.0)
    found = market_garch.fit(planted_returns)
    g = found.garch
    assert abs(g.alpha - PLANTED.alpha) < 0.02
    assert abs(g.beta - PLANTED.beta) < 0.02
    assert abs(g.nu - PLANTED.nu) < 1
    assert found.returns == N
    assert found.sample_variance == pytest.approx(float(np.mean(planted_returns**2)))
    # the optimum is the likelihood's: the planted values themselves score no better
    planted = float(np.mean(market_garch.loglik(planted_returns, PLANTED)))
    assert found.loglik >= planted


def test_the_likelihood_is_the_student_t_of_the_recursion() -> None:
    """Three returns by hand: h_0 at the unconditional variance, then the recursion, and the
    density of a t with nu degrees of freedom scaled to unit variance."""
    g = Garch(2e-7, 0.1, 0.7, 6.0)
    r = np.array([1e-3, -2e-3, 5e-4])
    h0 = g.variance
    h1 = g.omega + g.alpha * r[0] ** 2 + g.beta * h0
    h2 = g.omega + g.alpha * r[1] ** 2 + g.beta * h1
    scale = math.sqrt((g.nu - 2) / g.nu)
    expected = []
    for x, h in zip(r, (h0, h1, h2), strict=True):
        t = x / math.sqrt(h) / scale  # a plain t variate
        density = (
            math.gamma((g.nu + 1) / 2)
            / (math.sqrt(g.nu * math.pi) * math.gamma(g.nu / 2))
            * (1 + t * t / g.nu) ** (-(g.nu + 1) / 2)
        )
        expected.append(math.log(density / (math.sqrt(h) * scale)))
    assert market_garch.loglik(r, g) == pytest.approx(expected, rel=1e-12)


def test_the_bridge_sums_to_the_minute_garch_itself() -> None:
    """The sub-steps are drawn after the minute returns, so one seed gives the same minutes with
    or without them, and each minute's six sub-steps sum to its return."""
    minutes = nulls.simulate(PLANTED, 5000, np.random.default_rng(4))
    steps = nulls.simulate(PLANTED, 5000, np.random.default_rng(4), nulls.STEPS_PER_MINUTE)
    assert steps.shape == (5000 * nulls.STEPS_PER_MINUTE,)
    assert np.allclose(steps.reshape(-1, nulls.STEPS_PER_MINUTE).sum(axis=1), minutes, rtol=1e-12)
    assert not np.allclose(steps.reshape(-1, nulls.STEPS_PER_MINUTE), minutes[:, None] / 6)


def test_the_sub_steps_are_uncorrelated_with_a_sixth_of_the_variance_each() -> None:
    """Sub-step j is z_j - mean(z) + r_t / 6 with z of variance h_t / 6: its variance is
    h_t / 6 (5/6 + 1/6) and two of a minute covary by -h_t / 36 + E[r_t^2] / 36 = 0."""
    n = 200_000
    steps = nulls.simulate(PLANTED, n, np.random.default_rng(5), nulls.STEPS_PER_MINUTE)
    by_minute = steps.reshape(n, nulls.STEPS_PER_MINUTE)
    for j in range(nulls.STEPS_PER_MINUTE):
        assert by_minute[:, j].var() == pytest.approx(PLANTED.variance / 6, rel=0.1)
    within = np.corrcoef(by_minute[:, :-1].ravel(), by_minute[:, 1:].ravel())[0, 1]
    assert abs(within) < 0.01


def test_minutes_summed_from_the_sub_steps_give_back_the_fitted_variance_and_half_life() -> None:
    """The floor's minutes are what the pilot fit describes: a minute fit of summed sub-steps
    finds the planted half-life within 15% and the planted variance within 10%. The 10-second
    GARCH matched to the minute fit (persistence s^(1/6), alpha's share and nu kept) missed both
    by far (nulls.py), which is why the floor bridges a minute process instead."""
    steps = nulls.simulate(PLANTED, N, np.random.default_rng(6), nulls.STEPS_PER_MINUTE)
    g = market_garch.fit(steps.reshape(N, nulls.STEPS_PER_MINUTE).sum(axis=1)).garch
    planted_half_life = market_garch.half_life(PLANTED.alpha + PLANTED.beta)
    assert planted_half_life == pytest.approx(34.3, abs=0.05)
    assert market_garch.half_life(g.alpha + g.beta) == pytest.approx(planted_half_life, rel=0.15)
    assert g.variance == pytest.approx(PLANTED.variance, rel=0.10)


@pytest.mark.parametrize(
    ("s", "nu", "n", "match"),
    [
        (1.0, 5.0, 10**6, "alpha \\+ beta = 1.0 "),
        (1.2, 5.0, 10**6, "alpha \\+ beta = 1.2 "),
        (0.9999, 5.0, 6000, "not shorter than the 6000 returns"),
        (0.9, 2.0, 10**6, "runs into 2"),
        (0.9, 2.0005, 10**6, "runs into 2"),
        (0.9, 1.5, 10**6, "runs into 2"),
    ],
)
def test_the_fit_refuses_the_edges_of_a_finite_variance(
    s: float, nu: float, n: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        market_garch.edges(s, nu, n)


def test_a_fit_that_is_not_on_an_edge_passes() -> None:
    market_garch.edges(0.9999, 2.01, 10**6)  # half-life 6,931 steps, nu clear of 2


def test_shocks_with_no_finite_variance_drive_the_fit_into_the_nu_edge() -> None:
    r = 1e-3 * np.random.default_rng(3).standard_t(1.5, 50_000)
    with pytest.raises(ValueError, match="runs into 2"):
        market_garch.fit(r)


def test_a_variance_break_drives_the_fit_into_the_persistence_edge() -> None:
    """A sample whose variance steps up sixteen-fold halfway looks to GARCH like a shock that
    never decays: the half-life it fits is longer than the sample, and the fit refuses it."""
    rng = np.random.default_rng(3)
    r = 1e-3 * np.concatenate([rng.normal(size=25_000), 4 * rng.normal(size=25_000)])
    with pytest.raises(ValueError, match="not shorter than the 50000 returns"):
        market_garch.fit(r)


def test_minute_returns_skip_a_gap_in_the_klines() -> None:
    open_time = np.array([0, 60_000, 120_000, 300_000, 360_000], np.int64)
    close = np.array([100.0, 101.0, 100.0, 110.0, 99.0])
    r = market_garch.minute_returns({"open_time": open_time, "close": close})
    assert r == pytest.approx([math.log(1.01), math.log(100 / 101), math.log(99 / 110)])


def test_the_pilot_span_is_2023_and_a_kline_from_2024_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The span asked for ends on 2023-12-31; should the archive hand back a later kline anyway,
    the returns are refused rather than fitted."""
    assert market_garch.PILOT[0].year == market_garch.PILOT[1].year == 2023
    asked = []
    first_2024 = market_garch.archive.day_ms(nulls.CONFIRMATORY)

    def collect(symbols: list[str], start: object, end: object, root: Path) -> object:
        asked.append((start, end))
        klines = {
            "open_time": np.array([first_2024 - 60_000, first_2024], np.int64),
            "close": np.array([100.0, 101.0]),
        }
        return market_garch.archive.Span({(symbols[0], "klines"): klines}, [], [])

    monkeypatch.setattr(market_garch.archive, "collect", collect)
    with pytest.raises(ValueError, match="market-labels-v1"):
        market_garch.pilot_returns("BTCUSDT", Path("nowhere"))
    assert asked == [market_garch.PILOT]


def test_the_report_holds_one_row_per_symbol_with_its_half_life(tmp_path: Path) -> None:
    found = market_garch.Fit(PLANTED, 6.4, 1000, 9e-7)
    folder = market_garch.reports.write(market_garch.report({"BTCUSDT": found}), tmp_path)
    (row,) = pq.read_table(folder / "fits.parquet").to_pylist()
    assert row["symbol"] == "BTCUSDT" and row["alpha"] == PLANTED.alpha
    assert row["persistence"] == pytest.approx(0.98)
    assert row["half_life_minutes"] == pytest.approx(math.log(0.5) / math.log(0.98))
    assert row["variance"] == pytest.approx(1e-6) and row["sample_variance"] == 9e-7
    assert row["annual_vol"] == pytest.approx(math.sqrt(1e-6 * 1440 * 365))
    summary = (folder / "summary.md").read_text(encoding="utf-8")
    assert "| BTCUSDT | " in summary and "2023-01-01 to 2023-12-31" in summary
