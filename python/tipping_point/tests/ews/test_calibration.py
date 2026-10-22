"""The surrogate test's false-positive rate on nulls, and two deliberate breaks it must catch.

Each null is scored against 19 phase-randomised surrogates of its own residuals, an exact rank
test at alpha = 0.05: a null is flagged only when it beats all 19, which an exchangeable draw
does 1 time in 20. Over 400 nulls the flagged share must sit inside the 99% binomial band around
5%, 2.2-7.8% (20 +- 2.576 sqrt(19) of 400). Nulls: AR(1) with phi = 0.9, and a fast plus a slow
mode (phi = 0.5 and 0.995, unit variance each), the case near Tc. Geometry: window 50, segment
256, bandwidth 50 samples. `python -m tipping_point.ews calibrate` is the long version.

Breaks: surrogates identical to their row tie it every time, so nothing is flagged (0%);
surrogates scored with half the window have a narrower spread of scores than the row they
imitate, so the row beats them too often (> 8%).
"""

import numpy as np
import pytest
from tipping_point.ews import scoring, surrogates
from tipping_point.ews.trend import Pipeline

NULLS, DRAWS, ALPHA = 400, 19, 0.05
BAND = (0.022, 0.078)
WINDOW, LENGTH, BANDWIDTH = 50, 256, 50.0
SPAN = WINDOW + LENGTH - 1


def _unit_ar1(phi: float, rng: np.random.Generator) -> np.ndarray:
    model = surrogates.AR1(np.array(phi), np.array(np.sqrt(1 - phi**2)), np.array(0.0))
    return surrogates.ar1(model, SPAN, NULLS, rng)


def _nulls(kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "ar1":
        return _unit_ar1(0.9, rng)
    return _unit_ar1(0.5, rng) + _unit_ar1(0.995, rng)


@pytest.mark.parametrize("kind", ["ar1", "fast_slow"])
def test_phase_surrogate_false_positives_sit_in_the_band(kind: str) -> None:
    """Catches p-values that are not exact: (N + 1) in the denominator, ties counted for the
    observed score, or surrogates that lose the spectrum the score depends on."""
    rng = np.random.default_rng(31 if kind == "ar1" else 32)
    x = _nulls(kind, rng)
    for indicator in ("variance", "ac"):
        pipe = Pipeline(indicator, WINDOW, LENGTH, BANDWIDTH)
        _, p = scoring.surrogate_test(x, pipe, surrogates.phase_randomised, DRAWS, rng)
        flagged = float(np.mean(p <= ALPHA))
        assert BAND[0] <= flagged <= BAND[1], (kind, indicator, flagged)


def test_identity_surrogates_flag_nothing() -> None:
    """Revert check: a 'surrogate' equal to its row ties it, and ties count against the row, so
    no null is ever flagged. A calibration that passed this would be blind."""
    rng = np.random.default_rng(33)
    x = _nulls("ar1", rng)[:100]

    def copies(r: np.ndarray, n: int, g: np.random.Generator) -> np.ndarray:
        return np.repeat(r[:, None, :], n, axis=1)

    _, p = scoring.surrogate_test(
        x, Pipeline("variance", WINDOW, LENGTH, BANDWIDTH), copies, DRAWS, rng
    )
    assert np.all(p == 1.0)


def test_mismatched_surrogates_flag_too_many() -> None:
    """Revert check: surrogates scored through a different pipeline (half the window) spread
    their scores less than the row does, so the row beats all 19 too often and the band is
    left. The calibration above must be able to fail."""
    rng = np.random.default_rng(34)
    x = _nulls("ar1", rng)
    pipe = Pipeline("variance", WINDOW, LENGTH, BANDWIDTH)
    narrow = Pipeline("variance", WINDOW // 2, SPAN - WINDOW // 2 + 1, BANDWIDTH)
    residuals = pipe.residuals(x)
    observed = pipe.trend(residuals)
    drawn = narrow.trend(surrogates.phase_randomised(residuals, DRAWS, rng))
    assert np.mean(scoring.surrogate_pvalue(observed, drawn) <= ALPHA) > 0.08
