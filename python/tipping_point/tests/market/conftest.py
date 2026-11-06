"""The planted 3-month series (harness/market_planted.py) and its full labeller run, shared by the
lookahead and recovery tests: built once per session and never modified by them. Every test here
runs with the shared network block (tests/conftest.py)."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest
from market_planted import rv_series
from tipping_point.market import msvol

PARAMS = msvol.Params(warmup_days=28, baseline_days=14)  # 3 months hold the warm-up and 9 refits
FIRST = date(2022, 7, 4)
DAYS = 90


@dataclass(frozen=True)
class Planted:
    ts: np.ndarray
    rv: np.ndarray  # NaN through `gap`
    states: np.ndarray  # the planted path, 1 = high
    gap: slice  # six hours of missing bars inside a calm stretch of the labelled span
    full: msvol.Labels
    params: msvol.Params


@pytest.fixture(autouse=True)
def offline(no_network: None) -> None:
    """No test here reaches the network, even when the code under test breaks."""


@pytest.fixture(scope="session")
def planted() -> Planted:
    ts, rv, states = rv_series(FIRST, DAYS, np.random.default_rng(29))
    calm = np.convolve(states, np.ones(300), "valid") == 0  # calm from i to i + 300
    at = int(np.flatnonzero(calm[45 * 288 :])[0]) + 45 * 288 + 114
    gap = slice(at, at + 72)
    rv = rv.copy()
    rv[gap] = np.nan
    return Planted(ts, rv, states, gap, msvol.label(ts, rv, PARAMS), PARAMS)
