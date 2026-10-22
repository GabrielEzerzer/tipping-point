"""The surrogate test's false-positive rate on nulls, and two deliberate breaks it must catch.

Each null is scored against 19 phase-randomised surrogates of its own residuals, an exact rank
test at alpha = 0.05: a null is flagged only when it beats all 19, which an exchangeable draw
does 1 time in 20. Over 400 nulls the flagged share must sit inside the 99% binomial band around
5%, 2.2-7.8% (20 +- 2.576 sqrt(19) of 400). Nulls: AR(1) with phi = 0.9, and a fast plus a slow
mode (phi = 0.5 and 0.995), the case near Tc. Geometry: window 50, segment 256, bandwidth 50
samples. This is `python -m tipping_point.ews calibrate` at 400 nulls instead of 2,000 and 19
surrogates instead of 999.

What each check pins: the band, that the p-value is exact (N + 1 in the denominator, ties
counted against the row). It cannot tell a surrogate that keeps the spectrum from one that does
not: the score's null spread is set by the window, so even white-noise surrogates sit inside the
band (5.5% and 5.2% measured). The identity break pins tie handling (0% flagged); the half-window
break shows the band leaves its range when surrogates are scored through a different pipeline.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from tipping_point.ews import scoring, surrogates
from tipping_point.ews.__main__ import calibration, main, null_series
from tipping_point.ews.trend import Pipeline

WINDOW, LENGTH, BANDWIDTH = 50, 256, 50.0
SPAN = WINDOW + LENGTH - 1


@dataclass(frozen=True)
class Mismatched(Pipeline):
    """Scores each row as itself and its surrogates through `drawn`: the break, made to run
    inside surrogate_test's own batching and p-values."""

    drawn: Pipeline | None = None

    def trend(self, residuals: np.ndarray) -> np.ndarray:
        assert self.drawn is not None
        own = super().trend(residuals[:, :1])
        return np.concatenate([own, self.drawn.trend(residuals[:, 1:])], axis=1)


def test_phase_surrogate_false_positives_sit_in_the_band() -> None:
    """Catches p-values that are not exact: (N + 1) dropped from the denominator, or ties
    counted for the observed score."""
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
    """Revert check: surrogates scored with half the window spread their scores less than the
    row does, so the row beats all 19 too often and the band is left (13.5% measured)."""
    rng = np.random.default_rng(34)
    x = null_series("ar1", SPAN, 400, rng)
    narrow = Pipeline("variance", WINDOW // 2, SPAN - WINDOW // 2 + 1, BANDWIDTH)
    pipe = Mismatched("variance", WINDOW, LENGTH, BANDWIDTH, drawn=narrow)
    _, p = scoring.surrogate_test(x, pipe, surrogates.phase_randomised, 19, rng)
    assert np.mean(p <= 0.05) > 0.08


def test_calibrate_command_reports_every_cell(capsys: pytest.CaptureFixture[str]) -> None:
    """The long command's plumbing at toy size: every null, indicator and surrogate test gets a
    line with its band, and the exit code says whether all sat inside."""
    code = main(["calibrate", "--nulls", "40", "--surrogates", "9", "--seed", "2"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 8 and all("99% band" in line for line in lines)
    assert code == (0 if all(line.endswith("inside") for line in lines) else 1)
