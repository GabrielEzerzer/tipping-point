"""What the labels and the classic indicators give with no critical dynamics to find (#29 S3;
research pass section 4, "Circularity, dynamic", section 2 item 12, call 7).

  Garch              a GARCH(1,1)-t process: r_t = sqrt(h_t) e_t, h_t = omega + alpha r_t-1^2 +
                     beta h_t-1, e_t Student-t with nu degrees of freedom scaled to unit variance
  simulate           its returns, started from the unconditional variance
  minute_klines      1-minute OHLC from 10-second returns, in archive.collect's columns
  delays             how late the hysteresis state rises after a planted switch, under a theta
  shift_onsets       the event-time shift null: events moved by a random offset of at least a day
  circularity_floor  simulate at 10 seconds, build 1-minute OHLC, run the exact bars and labeller,
                     score close-to-close 1-minute returns: the AUC of pre-onset segments against
                     null slices, and of shifted segments against the same slices

The floor takes its scorers and its AUC as inputs. A scorer is any ews.trend.Pipeline, whose
windows and horizons #30's pre-registration fixes; scoring.auc is the AUC. No module of this package
may import those (tests/market/test_circularity.py: the labels cannot see the features), and the
floor needs nothing of them but the call. The GARCH parameters are an input too: their fit on the
2023 pilot year belongs to #29 S5.

Lengths: a segment or slice is `span` 1-minute samples. Bar b covers minutes [5b, 5b + 5), so the
segment at horizon h bars before onset bar b ends just before minute 5(b - h): it holds nothing of
the onset bar or of the h bars before it. Null slices are the non-overlapping spans cut in time
order from each null stretch, as the Ising scorer cuts its null runs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, NamedTuple, Protocol

import numpy as np

from . import archive, bars, msvol

CONFIRMATORY = date(2024, 1, 1)  # as __main__'s: no label from here on until the freeze (#29 S5)
FIRST = date(2022, 7, 4)  # a Monday: simulated days stand on the warm-up's calendar
STEPS_PER_MINUTE = 6  # 10-second returns
MINUTES_PER_BAR = 5
DAY_STEPS = 1440 * STEPS_PER_MINUTE
DELAY_BARS = 200_000  # draws per delay simulation: some 2,000 episodes at p00 = 0.985
SHIFT_DRAWS = 100
SHIFT_MIN_DAYS = 1
_NO_METRICS = {"create_time": np.empty(0, np.int64), "sum_open_interest": np.empty(0)}
_NO_FUNDING = {"calc_time": np.empty(0, np.int64), "last_funding_rate": np.empty(0)}


@dataclass(frozen=True)
class Garch:
    omega: float
    alpha: float
    beta: float
    nu: float

    def __post_init__(self) -> None:
        if not (self.omega > 0 and self.alpha >= 0 and self.beta >= 0):
            raise ValueError(f"need omega > 0 and alpha, beta >= 0: {self}")
        if not self.alpha + self.beta < 1:
            raise ValueError(f"need alpha + beta < 1 for a finite variance: {self}")
        if not self.nu > 2:
            raise ValueError(f"need nu > 2 for a finite variance: {self}")

    @property
    def variance(self) -> float:
        """The unconditional variance, omega / (1 - alpha - beta)."""
        return self.omega / (1.0 - self.alpha - self.beta)


def simulate(garch: Garch, n: int, rng: np.random.Generator) -> np.ndarray:
    """n returns from h_0 at the unconditional variance, so E[r_t^2] is that variance at every t."""
    shocks = rng.standard_t(garch.nu, n) * math.sqrt((garch.nu - 2.0) / garch.nu)
    omega, alpha, beta = garch.omega, garch.alpha, garch.beta
    h = garch.variance
    out = [0.0] * n
    for t, e in enumerate(shocks.tolist()):
        r = math.sqrt(h) * e
        out[t] = r
        h = omega + alpha * r * r + beta * h
    return np.array(out)


def minute_klines(
    start_ms: int, returns: np.ndarray, price: float = 20_000.0
) -> dict[str, np.ndarray]:
    """1-minute klines from 10-second log returns, the first opening at `price`: high and low over
    the minute's open and its six 10-second closes, each close a trade, and placeholder volumes
    (the labeller reads none, and a share of 1/2 is no flow)."""
    if len(returns) % STEPS_PER_MINUTE:
        raise ValueError(f"{len(returns)} returns are not whole minutes of {STEPS_PER_MINUTE}")
    path = price * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    steps = path[1:].reshape(-1, STEPS_PER_MINUTE)
    opens = path[:-1:STEPS_PER_MINUTE]
    minutes = len(steps)
    return {
        "open_time": start_ms + bars.MINUTE_MS * np.arange(minutes, dtype=np.int64),
        "open": opens,
        "high": np.maximum(opens, steps.max(axis=1)),
        "low": np.minimum(opens, steps.min(axis=1)),
        "close": steps[:, -1],
        "count": np.full(minutes, STEPS_PER_MINUTE, np.int64),
        "volume": np.full(minutes, float(STEPS_PER_MINUTE)),
        "taker_buy_volume": np.full(minutes, STEPS_PER_MINUTE / 2.0),
    }


def close_returns(klines: Mapping[str, np.ndarray]) -> np.ndarray:
    """Close-to-close 1-minute log returns, the kind of series features read; the first from its
    open."""
    return np.diff(np.log(klines["close"]), prepend=np.log(klines["open"][0]))


# ---------------------------------------------------------------- the detection delay


def _chain(theta: msvol.Theta, n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(size=n).tolist()
    s = u[0] < (1.0 - theta.p00) / (2.0 - theta.p00 - theta.p11)  # starts stationary
    states = [s] * n
    for t in range(1, n):
        s = (u[t] < theta.p11) if s else (u[t] >= theta.p00)
        states[t] = s
    return np.array(states)


def delays(
    theta: msvol.Theta, n: int, rng: np.random.Generator, enter: float = 0.8, exit: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """n bars of the labeller's own model at theta, run through its filter (at theta, as it runs
    between refits) and hysteresis. Per planted high episode wholly inside them: its length in
    bars, and the bars from its first bar to its first with H = 1, -1 when H never rises inside it
    (H still high from the episode before gives 0)."""
    states = _chain(theta, n, rng)
    noise = rng.normal(size=n)
    z = np.where(states, theta.mu1 + theta.sigma1 * noise, theta.mu0 + theta.sigma0 * noise)
    e0, e1, _ = msvol._emissions(z, theta)
    h = msvol.hysteresis(msvol._forward(e0, e1, theta), enter, exit)
    starts, ends = msvol.runs(states)
    whole = (starts > 0) & (ends < n)
    starts, ends = starts[whole], ends[whole]
    next_high = np.minimum.accumulate(np.where(h == 1, np.arange(n), n)[::-1])[::-1]
    first = next_high[starts]
    return ends - starts, np.where(first < ends, first - starts, -1)


def delay_rows(
    theta: msvol.Theta, params: msvol.Params, n: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """The delay distribution under theta at the labeller's thresholds: over all planted episodes,
    and over those of at least d_min bars, the ones that can become events."""
    lengths, delay = delays(theta, n, rng, params.enter, params.exit)
    rows = []
    for population, keep in (("all", lengths > 0), ("at_least_d_min", lengths >= params.d_min)):
        seen, risen = delay[keep], delay[keep][delay[keep] >= 0]
        rows.append(
            {
                "population": population,
                "episodes": int(seen.size),
                "missed": int(seen.size - risen.size),
                "missed_share": float(np.mean(seen < 0)) if seen.size else math.nan,
                "median": float(np.median(risen)) if risen.size else math.nan,
                "p90": float(np.percentile(risen, 90, method="higher")) if risen.size else math.nan,
                "max": int(risen.max()) if risen.size else None,
            }
        )
    return rows


def theta_of(refit: Mapping[str, Any]) -> msvol.Theta:
    return msvol.Theta(**{name: refit[name] for name in msvol.Theta._fields})


def final_theta(labels: msvol.Labels) -> msvol.Theta | None:
    """The theta in force at the last bar, None when no fit ever went into force."""
    if not len(labels.refit) or labels.refit[-1] < 0:
        return None
    return theta_of(labels.refits[labels.refit[-1]])


# ---------------------------------------------------------------- segments and slices


def pre_onset_rows(minutes: np.ndarray, onsets: np.ndarray, horizon: int, span: int) -> np.ndarray:
    """(onsets, span): the span samples of the 1-minute series ending just before minute
    5(onset - horizon), without the onsets too early to have them."""
    ends = MINUTES_PER_BAR * (np.asarray(onsets, np.int64) - horizon)
    return _rows(minutes, ends[ends >= span], span)


def _rows(minutes: np.ndarray, ends: np.ndarray, span: int) -> np.ndarray:
    return minutes[ends[:, None] - span + np.arange(span)]


def null_rows(minutes: np.ndarray, stretches: Sequence[Mapping[str, Any]], span: int) -> np.ndarray:
    """(slices, span): non-overlapping spans cut in time order from each null stretch."""
    ends = [
        MINUTES_PER_BAR * s["start"] + span * (k + 1)
        for s in stretches
        for k in range(MINUTES_PER_BAR * s["bars"] // span)
    ]
    return _rows(minutes, np.array(ends, np.int64), span)


def shift_onsets(
    onsets: np.ndarray,
    low: int,
    high: int,
    min_offset: int,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(draws, onsets): each onset moved, in each draw on its own, by a uniform offset around the
    bars [low, high) that keeps it at least min_offset bars from where it was, either way."""
    onsets = np.asarray(onsets, np.int64)
    size = high - low
    if np.any((onsets < low) | (onsets >= high)):
        raise ValueError(f"onsets must lie in the shift range [{low}, {high})")
    if size < 2 * min_offset:
        raise ValueError(f"{size} bars cannot hold an offset of {min_offset} either way")
    offsets = rng.integers(min_offset, size - min_offset, (draws, len(onsets)), endpoint=True)
    return low + (onsets - low + offsets) % size


# ---------------------------------------------------------------- the floor


class Scorer(Protocol):
    """How the floor reads a pipeline: ews.trend.Pipeline is one."""

    @property
    def span(self) -> int: ...

    def __call__(self, x: np.ndarray) -> np.ndarray: ...


class Floor(NamedTuple):
    cells: list[dict[str, Any]]  # one per (scorer, horizon)
    paths: list[dict[str, Any]]  # one per simulated path


def circularity_floor(
    garch: Garch,
    scorers: Mapping[str, Scorer],
    auc: Callable[[np.ndarray, np.ndarray], float],
    horizons: Sequence[int],
    *,
    days: int,
    paths: int,
    seed: int,
    params: msvol.Params | None = None,
    draws: int = SHIFT_DRAWS,
    first: date = FIRST,
) -> Floor:
    """The AUC a clustering process with no critical dynamics gives each scorer at each horizon
    (in bars), pooled over `paths` simulated series of `days` days, and the same with each path's
    events shifted, per draw. A horizon below the worst path's p90 detection delay, or with no
    delay known, is marked contaminated (research pass section 2, item 12)."""
    params = params or msvol.Params()
    refuse_confirmatory(archive.day_ms(first + timedelta(days=days - 1)) * 1_000_000)
    start_ms = archive.day_ms(first)
    longest = max(s.span for s in scorers.values())
    room = -(-longest // MINUTES_PER_BAR) + max(horizons)  # bars a segment needs before its end
    min_offset = SHIFT_MIN_DAYS * msvol.DAY_BARS
    pos: dict[tuple[str, int], list[np.ndarray]] = {(s, h): [] for s in scorers for h in horizons}
    moved: dict[tuple[str, int], list[np.ndarray]] = {key: [] for key in pos}
    neg: dict[str, list[np.ndarray]] = {s: [] for s in scorers}
    summary = []
    for path in range(paths):
        rng = np.random.default_rng([seed, path])
        klines = minute_klines(start_ms, simulate(garch, days * DAY_STEPS, rng))
        bar, _ = bars.bars(start_ms, days, klines, _NO_METRICS, _NO_FUNDING)
        out = msvol.label(bar["ts_ns"], bar["rv"], params)
        minutes = close_returns(klines)
        onsets = np.array([e["onset"] for e in out.episodes if e["kind"] == "event"], np.int64)
        labelled = np.flatnonzero(out.h >= 0)
        low = max(int(labelled[0]) if labelled.size else len(out.h), room)
        inside = onsets[onsets >= low]
        shifted = np.empty((draws, 0), np.int64)
        if inside.size and len(out.h) - low >= 2 * min_offset:
            shifted = shift_onsets(inside, low, len(out.h), min_offset, draws, rng)
        for name, scorer in scorers.items():
            neg[name].append(_score(scorer, null_rows(minutes, out.nulls, scorer.span)))
            for h in horizons:
                rows = pre_onset_rows(minutes, onsets, h, scorer.span)
                pos[name, h].append(_score(scorer, rows))
                rows = pre_onset_rows(minutes, shifted.ravel(), h, scorer.span)
                moved[name, h].append(_score(scorer, rows).reshape(draws, -1))
        theta = final_theta(out)
        p90 = delay_rows(theta, params, DELAY_BARS, rng)[1]["p90"] if theta else math.nan
        summary.append(
            {
                "path": path,
                "bars": len(out.h),
                "labelled_bars": int(labelled.size),
                "refits": len(out.refits),
                "flagged_refits": sum(r["flagged"] for r in out.refits),
                "events": int(onsets.size),
                "other_episodes": sum(e["kind"] == "episode" for e in out.episodes),
                "flickers": sum(e["kind"] == "flicker" for e in out.episodes),
                "null_bars": sum(s["bars"] for s in out.nulls),
                "delay_p90": p90,
            }
        )
    known = [row["delay_p90"] for row in summary if not math.isnan(row["delay_p90"])]
    worst = max(known, default=math.nan)
    cells = []
    for (name, h), found in pos.items():
        p, n = np.concatenate(found), np.concatenate(neg[name])
        by_draw = np.concatenate(moved[name, h], axis=1)
        null_aucs = np.array([auc(row, n) for row in by_draw])
        null_aucs = null_aucs[~np.isnan(null_aucs)]
        low_auc, mid, high_auc = (
            np.percentile(null_aucs, [2.5, 50, 97.5]) if null_aucs.size else (math.nan,) * 3
        )
        cells.append(
            {
                "scorer": name,
                "horizon_bars": h,
                "contaminated": not h >= worst,
                "delay_p90": worst,
                "auc": auc(p, n),
                "n_pos": int(np.isfinite(p).sum()),
                "n_neg": int(np.isfinite(n).sum()),
                "shift_auc_median": float(mid),
                "shift_auc_lo": float(low_auc),
                "shift_auc_hi": float(high_auc),
                "shift_draws": int(null_aucs.size),
            }
        )
    return Floor(cells, summary)


def _score(scorer: Scorer, rows: np.ndarray) -> np.ndarray:
    return scorer(rows) if len(rows) else np.empty(0)


def refuse_confirmatory(last_ts_ns: int) -> None:
    """The labeller's refusal, for what reads labels: nothing from the confirmatory span."""
    if last_ts_ns >= archive.day_ms(CONFIRMATORY) * 1_000_000:
        raise ValueError(
            f"days from {CONFIRMATORY} are the confirmatory span, closed until market-labels-v1 "
            "is frozen (#29 S5)"
        )
