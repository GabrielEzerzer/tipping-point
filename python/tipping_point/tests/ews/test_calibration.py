"""The surrogate test's false-positive rate on nulls, and two deliberate breaks it must catch.

Each null is scored against 19 phase-randomised surrogates of its own residuals, an exact rank
test at alpha = 0.05: a null is flagged only when it beats all 19, which an exchangeable draw
does 1 time in 20. Over 400 nulls the flagged share must sit inside the 99% binomial band around
5%, 2.2-7.8% (20 +- 2.576 sqrt(19) of 400). Nulls: AR(1) with phi = 0.9, and a fast plus a slow
mode (phi = 0.5 and 0.995), the case near Tc. Geometry: window 50, segment 256, bandwidth 50
samples. This is `python -m tipping_point.ews calibrate` at 400 nulls instead of 2,000 and 19
surrogates instead of 999.

Breaks: surrogates identical to their row tie it every time, so nothing is flagged (0%);
surrogates scored with half the window spread their scores less than the row they imitate, so
the row beats all 19 too often (> 8%).
"""

import numpy as np
from tipping_point.ews import scoring, surrogates
from tipping_point.ews.__main__ import calibration, null_series
from tipping_point.ews.trend import Pipeline

WINDOW, LENGTH, BANDWIDTH = 50, 256, 50.0
SPAN = WINDOW + LENGTH - 1


def test_phase_surrogate_false_positives_sit_in_the_band() -> None:
    """Catches p-values that are not exact: (N + 1) in the denominator, ties counted for the
    observed score, or surrogates that lose the spectrum the score depends on."""
    rows = calibration(nulls=400, draws=19, seed=26, tests=("phase",))
    assert len(rows) == 4
    for row in rows:
        assert np.allclose(row["band"], (0.022, 0.078), atol=5e-4)
        assert row["inside"], row


def test_identity_surrogates_flag_nothing() -> None:
    """Revert check: a 'surrogate' equal to its row ties it, and ties count against the row, so
    no null is ever flagged. A calibration that passed this would be blind."""
    rng = np.random.default_rng(33)
    x = null_series("ar1", SPAN, 100, rng)

    def copies(r: np.ndarray, n: int, g: np.random.Generator) -> np.ndarray:
        return np.repeat(r[:, None, :], n, axis=1)

    pipe = Pipeline("variance", WINDOW, LENGTH, BANDWIDTH)
    _, p = scoring.surrogate_test(x, pipe, copies, 19, rng)
    assert np.all(p == 1.0)


def test_mismatched_surrogates_flag_too_many() -> None:
    """Revert check: surrogates scored through a different pipeline (half the window) spread
    their scores less than the row does, so the row beats all 19 too often and the band is
    left. The calibration above must be able to fail."""
    rng = np.random.default_rng(34)
    x = null_series("ar1", SPAN, 400, rng)
    pipe = Pipeline("variance", WINDOW, LENGTH, BANDWIDTH)
    narrow = Pipeline("variance", WINDOW // 2, SPAN - WINDOW // 2 + 1, BANDWIDTH)
    residuals = pipe.residuals(x)
    observed = pipe.trend(residuals)
    drawn = narrow.trend(surrogates.phase_randomised(residuals, 19, rng))
    assert np.mean(scoring.surrogate_pvalue(observed, drawn) <= 0.05) > 0.08
