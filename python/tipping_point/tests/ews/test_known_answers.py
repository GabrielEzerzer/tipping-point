"""Known answers for the whole pipeline (detrend, rolling indicator, Kendall's tau, AUC).

Fold: dx = (mu - x^2) dt + sigma dW drifts towards a saddle-node at mu = 0. With mu ramped from 1
to 0.05 the stable state's recovery rate 2 sqrt(mu) falls, so variance and lag-1 autocorrelation
must rise; with mu held at 1 nothing changes. sigma = 0.05, dt = 0.1, 2,000 Euler steps, 200
runs of each, window half the series and bandwidth a fifth (ewstools' default), fixed seed.
Research pass (#26): AUC 0.94 (variance) and 0.96 (AC1); at dt = 0.01, ten times fewer recovery
times per window, power drops to 0.65.

Null: AR(1) with phi = 0.9 and nothing changing; two independent sets of 200 must be told apart
no better than chance, and the scores must centre on 0.
"""

import numpy as np
from tipping_point.ews import scoring, surrogates
from tipping_point.ews.trend import Pipeline

STEPS, WINDOW, BANDWIDTH = 2000, 1000, 400.0


def _fold(runs: int, forced: bool, rng: np.random.Generator) -> np.ndarray:
    mu = np.linspace(1.0, 0.05, STEPS) if forced else np.full(STEPS, 1.0)
    dt, sigma = 0.1, 0.05
    noise = sigma * np.sqrt(dt) * rng.standard_normal((STEPS, runs))
    x = np.empty((STEPS, runs))
    x[0] = 1.0  # the stable state sqrt(mu) at mu = 1
    for t in range(1, STEPS):
        x[t] = x[t - 1] + (mu[t - 1] - x[t - 1] ** 2) * dt + noise[t]
    return x.T


def _pipeline(indicator: str) -> Pipeline:
    return Pipeline(indicator, WINDOW, STEPS - WINDOW + 1, BANDWIDTH)


def test_fold_warnings_rise() -> None:
    """Catches an indicator or trend that has lost its signal: a sign flip, a window misaligned
    with time, or detrending that eats the rise. Measured here: 0.948 and 0.956."""
    rng = np.random.default_rng(26)
    forced, fixed = _fold(200, True, rng), _fold(200, False, rng)
    for indicator in ("variance", "ac"):
        p = _pipeline(indicator)
        assert scoring.auc(p(forced), p(fixed)) >= 0.9, indicator


def test_ar1_null_warnings_do_not_rise() -> None:
    """Catches a bias that would fake a warning: a trend in the window alignment, an edge effect
    of the detrending, or a Kendall's tau that leans positive. AUC 0.42-0.58 is about 2.8 of its
    standard errors (0.029 at 200 against 200) either side of 0.5."""
    rng = np.random.default_rng(27)
    model = surrogates.AR1(np.array(0.9), np.array(np.sqrt(1 - 0.81)), np.array(0.0))
    a, b = surrogates.ar1(model, STEPS, 200, rng), surrogates.ar1(model, STEPS, 200, rng)
    for indicator in ("variance", "ac"):
        p = _pipeline(indicator)
        sa, sb = p(a), p(b)
        assert 0.42 <= scoring.auc(sa, sb) <= 0.58, indicator
        both = np.concatenate([sa, sb])
        assert abs(both.mean()) <= 3 * both.std(ddof=1) / np.sqrt(len(both)), indicator
