"""The market toy for the JEPA harness (#30 M3): synthetic 1-second feature days under the loader's
input contract, with a planted slow variable that switches at known minutes, so that learning,
collapse, determinism and resume can be checked in seconds on a CPU.

The slow variable s is +1 or -1 per minute and holds for 60-240 minutes between switches, drawn
per day from its date (`switches`). It shows in the structure of the seconds inside a minute:
  f00, f01   share a per-second factor, with correlation +0.5 when s = +1 and -0.5 when s = -1
  f02        MA(1) noise e_t + 0.8 s e_(t-1), so its lag-1 autocorrelation is +-0.49
  f03, f04   unit noise scaled by exp(0.35 s), a 2:1 ratio of standard deviations
  f05-f08    unit noise scaled per minute by exp(N(0, 0.7^2)): big, minute-level and unpredictable
             from one minute to the next, the variance an untrained encoder soaks up
  f09-f15    unit noise
Every feature has mean 0 whatever s is: s shows only in second moments (a correlation, an
autocorrelation, a variance), so the toy is learned only by an encoder that reads them. With the
tokens pooled by their mean alone, a trained latent read s at chance. A minute's s is predictable
from the minutes before it (it persists), the distractors are not, so a JEPA has a reason to keep s
and drop the rest. Gaps: each day has a gap second in minutes 100 and 700 and a 5-minute gap block
from minute 1,000 (`GAP_MINUTES` are the minutes this leaves invalid); their values are NaN, as a
real gap's may be.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import numpy as np
import pyarrow as pa
from tipping_point.jepa import days, minutes

SYMBOL = "BTCUSDT"
FEATURES = tuple(f"f{k:02d}" for k in range(16))
TRAIN_DAYS = tuple(date(2023, 3, 1) + timedelta(days=k) for k in range(6))
VAL_DAYS = (date(2023, 3, 8), date(2023, 3, 9))
PROBE_DAYS = (date(2023, 4, 3), date(2023, 4, 4))  # fit and test the probes; never trained on
GAP_MINUTES = (100, 700, 1000, 1001, 1002, 1003, 1004)

SPEC = minutes.MinuteSpec(
    features=len(FEATURES), dim=32, depth=2, pred_depth=1, heads=2, context=16
)
# a budget of steps, never of wall-clock time; a flat learning rate and a faster EMA than the
# production 0.996 -> 1.0, so the target encoder follows the context encoder within 400 steps
# (at 0.996 the target barely moves from its random start in a run this short, and the latent
# never picks up s)
CONFIG = minutes.Config(
    symbol=SYMBOL,
    steps=400,
    model=SPEC,
    chunks=16,
    lr=2e-3,
    lr_final=2e-3,
    warmup=20,
    ema=(0.95, 0.99),
    seed=30,
    checkpoint_every=200,
    monitor_every=50,
    monitor_chunks=32,
    pca_frames=512,
)


def config(**changes: Any) -> minutes.Config:
    return replace(CONFIG, **changes)


def _rng(day: date, stream: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, day.toordinal(), stream])


def switches(day: date, seed: int = 0) -> np.ndarray:
    """The minutes of `day` at which s switches (s changes sign at the start of each)."""
    gaps = _rng(day, 0, seed).integers(60, 241, 24)
    at = np.cumsum(gaps)
    return at[at < days.MINUTES]


def slow(day: date, seed: int = 0) -> np.ndarray:
    """s per minute of `day`, (MINUTES,) int8, +1 or -1; it starts at +1 on even ordinals."""
    flips = np.zeros(days.MINUTES, np.int64)
    flips[switches(day, seed)] = 1
    start = 1 if day.toordinal() % 2 == 0 else -1
    return (start * (1 - 2 * (np.cumsum(flips) % 2))).astype(np.int8)


def features(day: date, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(values (SECONDS, 16) float32 with NaN at gap seconds, gap (SECONDS,) bool)."""
    rng = _rng(day, 1, seed)
    s = np.repeat(slow(day, seed), days.MINUTE).astype(np.float64)
    n = days.SECONDS
    x = rng.standard_normal((n, len(FEATURES)))
    shared = rng.standard_normal(n)
    x[:, 0] = (shared + x[:, 0]) / np.sqrt(2)
    x[:, 1] = (s * shared + x[:, 1]) / np.sqrt(2)
    e = rng.standard_normal(n + 1)
    theta = 0.8 * s
    x[:, 2] = (e[1:] + theta * e[:-1]) / np.sqrt(1 + theta**2)
    x[:, 3:5] *= np.exp(0.35 * s)[:, None]
    scale = np.exp(0.7 * rng.standard_normal((days.MINUTES, 4)))
    x[:, 5:9] *= np.repeat(scale, days.MINUTE, axis=0)
    gap = np.zeros(n, bool)
    gap[[100 * days.MINUTE + 17, 700 * days.MINUTE + 17]] = True
    gap[1000 * days.MINUTE : 1005 * days.MINUTE] = True
    values = x.astype(np.float32)
    values[gap] = np.nan
    return values, gap


class Source:
    """A days.DaySource of toy days for any symbol; it counts the days it was asked for."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.reads: list[tuple[str, date]] = []

    def table(self, day: date) -> pa.Table:
        values, gap = features(day, self.seed)
        columns = {name: pa.array(values[:, k]) for k, name in enumerate(FEATURES)}
        return pa.table({**columns, days.GAP: pa.array(gap)})

    def read(self, symbol: str, day: date) -> tuple[pa.Table, str]:
        self.reads.append((symbol, day))
        table = self.table(day)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return table, hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def dataset(source: Source | None = None) -> days.Dataset:
    return days.load(
        source or Source(), CONFIG.span, SYMBOL, FEATURES, list(TRAIN_DAYS), list(VAL_DAYS)
    )
