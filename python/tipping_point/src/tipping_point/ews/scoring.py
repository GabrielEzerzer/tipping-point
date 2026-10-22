"""Scoring: how well a score separates series heading for a transition from nulls, how often it
alarms, and how sure we are.

auc                     Mann-Whitney AUC, P(positive > negative) + 1/2 P(tie); NaN scores dropped
surrogate_pvalue        (1 + #{surrogate >= observed}) / (1 + N): exact for exchangeable draws
surrogate_test          each series' score against N surrogates of its own residuals
alarms                  an alarm is s(t) rising above theta: s(t) > theta after s(t - 1) <= theta
false_alarm_rate        alarms per unit of time on null series
threshold_at_far        the lowest theta whose false-alarm rate, and every higher theta's, stays
                        within a target
first_alarm, lead_times when each series first rises above theta inside a stretch (a series
                        already above it as the stretch opens is censored), and how early
hierarchical_bootstrap  resample seeds, then replicas within each chosen seed
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from .trend import Pipeline

Surrogate = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]
SURROGATE_BYTES = 1 << 27  # surrogates generated per batch in surrogate_test


def auc(positives: npt.ArrayLike, negatives: npt.ArrayLike) -> float:
    pos = np.asarray(positives, dtype=np.float64).ravel()
    neg = np.asarray(negatives, dtype=np.float64).ravel()
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    _, inverse, counts = np.unique(
        np.concatenate([pos, neg]), return_inverse=True, return_counts=True
    )
    midranks = np.cumsum(counts) - (counts - 1) / 2.0  # ranks from 1; ties share their mean
    u = midranks[inverse[: len(pos)]].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def surrogate_pvalue(observed: npt.ArrayLike, surrogates: npt.ArrayLike) -> np.ndarray:
    """observed (...), surrogates (..., N): a NaN surrogate counts as reaching the observed score,
    and a NaN observed score has no p-value."""
    observed = np.asarray(observed, dtype=np.float64)
    surrogates = np.asarray(surrogates, dtype=np.float64)
    reached = (surrogates >= observed[..., None]) | np.isnan(surrogates)
    p = (1.0 + reached.sum(axis=-1)) / (1.0 + surrogates.shape[-1])
    return np.where(np.isnan(observed), np.nan, p)


def surrogate_test(
    x: np.ndarray,
    pipeline: Pipeline,
    surrogate: Surrogate,
    n: int,
    rng: np.random.Generator,
    direction: str = "greater",
) -> tuple[np.ndarray, np.ndarray]:
    """Scores of the rows of x (rows, time) and their one-sided surrogate p-values: for a trend
    expected to rise ("greater") a surrogate counts when it scores at least as high, for one
    expected to fall ("less") when it scores at least as low. Surrogates are drawn from each
    row's residuals and scored in the same batch as the row itself, so a surrogate identical to
    its row scores identically."""
    sign = _sign(direction)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    residuals = pipeline.residuals(x)
    observed, p = np.empty(len(x)), np.empty(len(x))
    rows = max(1, SURROGATE_BYTES // (8 * (n + 1) * x.shape[-1]))
    for start in range(0, len(x), rows):
        own = residuals[start : start + rows]
        batch = np.concatenate([own[:, None, :], surrogate(own, n, rng)], axis=1)
        scores = pipeline.trend(batch)
        observed[start : start + rows] = scores[:, 0]
        p[start : start + rows] = surrogate_pvalue(sign * scores[:, 0], sign * scores[:, 1:])
    return observed, p


def _sign(direction: str) -> float:
    if direction not in ("greater", "less"):
        raise ValueError(f"direction must be 'greater' or 'less', got {direction!r}")
    return 1.0 if direction == "greater" else -1.0


def oriented(scores: np.ndarray, direction: str) -> np.ndarray:
    """Scores turned so that the expected change is upward: negated for a falling indicator."""
    return _sign(direction) * np.asarray(scores, dtype=np.float64)


def _steps(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(previous, current) score at every time of every row of s(t) (rows, times). Before the
    first score, and after a missing one, the previous score is -inf; a missing current score
    is NaN, which never exceeds a threshold."""
    current = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    previous = np.full_like(current, -np.inf)
    previous[:, 1:] = current[:, :-1]
    previous[np.isnan(previous)] = -np.inf
    return previous, current


def alarms(scores: np.ndarray, theta: float) -> np.ndarray:
    """Alarms per row of s(t) (rows, times): rises from <= theta to > theta."""
    previous, current = _steps(scores)
    return np.sum((previous <= theta) & (current > theta), axis=1)


def false_alarm_rate(scores: np.ndarray, theta: float, duration: float) -> float:
    """Alarms per unit time over null series covering `duration` units in total."""
    return float(alarms(scores, theta).sum() / duration)


def threshold_at_far(scores: np.ndarray, far: float, duration: float) -> float:
    """The lowest theta at which the null series alarm at most `far` times per unit time, at that
    theta and at every theta above it. A step from a to b > a alarms for theta in [a, b), so the
    alarm count is #{a <= theta} - #{b <= theta}, a step function read at its breakpoints."""
    previous, current = _steps(scores)
    rising = previous < current  # False wherever the current score is missing
    previous, current = np.sort(previous[rising]), np.sort(current[rising])
    candidates = np.unique(np.concatenate([previous, current]))
    if len(candidates) == 0:
        return float("-inf")
    count = np.searchsorted(previous, candidates, "right") - np.searchsorted(
        current, candidates, "right"
    )
    worst_above = np.maximum.accumulate(count[::-1])[::-1]
    return float(candidates[np.argmax(worst_above <= far * duration)])


def first_alarm(
    times: np.ndarray, scores: np.ndarray, theta: float, previous: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """The first rise of each row of s(t) above theta inside a stretch: s > theta after
    s <= theta, `previous` being each row's score just before the stretch opens. A row already
    above theta there is censored: its alarm came earlier, so here it is neither an alarm nor a
    miss. Returns (alarm times, NaN for none; censored rows)."""
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    entry = np.asarray(previous, dtype=np.float64).reshape(len(scores))
    censored = entry > theta
    before = np.concatenate([entry[:, None], scores[:, :-1]], axis=1)
    rise = (before <= theta) & (scores > theta)
    found = rise.any(axis=1) & ~censored
    first = np.argmax(rise, axis=1)
    return np.where(found, np.asarray(times, dtype=np.float64)[first], np.nan), censored


def lead_times(event: float | np.ndarray, alarm: np.ndarray) -> np.ndarray:
    """How long before the event each alarm came (negative: after it); NaN for a miss."""
    return np.asarray(event, dtype=np.float64) - np.asarray(alarm, dtype=np.float64)


Grouped = Sequence[np.ndarray]  # one array per seed, replicas along axis 0


def hierarchical_bootstrap(
    statistic: Callable[..., float],
    samples: Sequence[Grouped],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """statistic(*resampled) for n resamples. Each sample is a list of per-seed arrays; a resample
    draws its seeds with replacement, then within each drawn seed its replicas with replacement,
    and joins them along axis 0."""
    out = np.empty(n)
    for i in range(n):
        drawn = []
        for seeds in samples:
            picks = rng.integers(0, len(seeds), len(seeds))
            drawn.append(
                np.concatenate(
                    [seeds[s][rng.integers(0, len(seeds[s]), len(seeds[s]))] for s in picks]
                )
            )
        out[i] = statistic(*drawn)
    return out


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """The central `level` interval of bootstrap values (NaN draws are ignored)."""
    lo, hi = np.nanquantile(
        np.asarray(values, dtype=np.float64), [(1 - level) / 2, (1 + level) / 2]
    )
    return float(lo), float(hi)
