"""1-second feature days in, minute frames in host memory out: what the market JEPA trains on (#30).

The input contract, which the feature package (#30 M2) writes and any source must meet: one table
per symbol-day (UTC), 86,400 rows in second order, a float32 column per feature and a bool `gap`
column, True for a second without valid book data (a feature gap or an unsynced book). A source
hands over the table with the sha256 of the bytes it came from, which the training manifest pins.

A minute is valid when none of its 60 seconds is a gap. Invalid minutes are never trained on:
their frames are zeroed, their latents are left out of every loss term, and the predictor sees a
mask token in their place. Gap seconds are zeroed too, so no value behind a gap reaches the model.

Leakage (#30 research pass 4.3, tests 1 and 2): `load` refuses any day outside the model's training
span [start, end), any day on or after 2024-01-01 whatever the span says (the confirmatory span is
never a training input), and any symbol other than the model's own. The manifest's data section
lists every input day with its source sha256, and `resume` refuses a dataset whose list differs.

shortcut: no real source exists until M2 writes the per-day Parquet; tests use the synthetic one in
tests/fixtures (market_toy.py). Add a Parquet source here when M2 lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import torch
from torch import Tensor

SECONDS = 86_400
MINUTE = 60  # seconds per frame
MINUTES = SECONDS // MINUTE
GAP = "gap"
CONFIRMATORY = date(2024, 1, 1)  # no market day from here on is ever a training input


class DataError(ValueError):
    """A day the market loader refuses or cannot read."""


class DaySource(Protocol):
    def read(self, symbol: str, day: date) -> tuple[pa.Table, str]:
        """One symbol-day under the input contract, with the sha256 of the bytes it came from."""
        ...


@dataclass(frozen=True)
class Span:
    """What a model may train on: one symbol, days in [start, end)."""

    symbol: str
    start: date
    end: date

    def check(self, symbol: str, day: date) -> None:
        if symbol != self.symbol:
            raise DataError(f"{symbol} is not this model's symbol ({self.symbol})")
        if day >= CONFIRMATORY:
            raise DataError(f"{day} is in the confirmatory span (from {CONFIRMATORY}): never read")
        if not self.start <= day < self.end:
            raise DataError(f"{day} is outside the training span [{self.start}, {self.end})")


def read_day(
    source: DaySource, symbol: str, day: date, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, str]:
    """(frames (MINUTES, MINUTE, F) float32, valid (MINUTES,) bool, sha256) for one day, checked
    against the input contract."""
    table, digest = source.read(symbol, day)
    if table.num_rows != SECONDS:
        raise DataError(f"{symbol} {day}: {table.num_rows} rows, not {SECONDS}")
    missing = [name for name in (*features, GAP) if name not in table.column_names]
    if missing:
        raise DataError(f"{symbol} {day}: no column(s) {missing}")
    if table[GAP].type != pa.bool_() or table[GAP].null_count:
        raise DataError(f"{symbol} {day}: `gap` must be bool without nulls")
    gap = table[GAP].to_numpy()
    columns = []
    for name in features:
        column = table[name]
        if column.type != pa.float32():
            raise DataError(f"{symbol} {day}: {name} is {column.type}, not float32")
        values = column.to_numpy(zero_copy_only=False).astype(np.float32, copy=True)
        if np.isnan(values[~gap]).any() or np.isinf(values[~gap]).any():
            raise DataError(f"{symbol} {day}: {name} is not finite outside gap seconds")
        values[gap] = 0.0  # nulls and anything else behind a gap never reach the model
        columns.append(values)
    seconds = np.stack(columns, axis=1)
    valid = ~gap.reshape(MINUTES, MINUTE).any(1)
    return seconds.reshape(MINUTES, MINUTE, len(features)), valid, digest


@dataclass(frozen=True)
class Minutes:
    """Minute frames of whole days, in date order, in host memory."""

    frames: Tensor  # (minutes, MINUTE, F) float32
    valid: Tensor  # (minutes,) bool
    run: Tensor  # (minutes,) int64: which run of consecutive days each minute belongs to

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def starts(self, length: int, whole: bool = False) -> Tensor:
        """The first minutes of every chunk of `length` consecutive minutes inside one run; with
        `whole`, only chunks whose every minute is valid."""
        if len(self) < length:
            return torch.zeros(0, dtype=torch.int64)
        first = torch.arange(len(self) - length + 1)
        ok = self.run[first] == self.run[first + length - 1]
        if whole:
            invalid = torch.cat([torch.zeros(1, dtype=torch.int64), (~self.valid).cumsum(0)])
            ok &= invalid[first + length] == invalid[first]
        return first[ok]

    def chunks(self, first: Tensor, length: int) -> tuple[Tensor, Tensor]:
        """(frames (B, length, MINUTE, F), valid (B, length)) of the chunks starting at `first`."""
        index = first.unsqueeze(1) + torch.arange(length)
        return self.frames[index], self.valid[index]

    def sample(self, count: int, length: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
        """`count` chunks drawn uniformly, with replacement, by `gen` (a CPU generator)."""
        starts = self.starts(length)
        if len(starts) == 0:
            raise DataError(f"no run of {length} consecutive minutes to sample from")
        return self.chunks(starts[torch.randint(len(starts), (count,), generator=gen)], length)


@dataclass
class Dataset:
    train: Minutes
    val: Minutes
    sources: dict[str, Any]  # the manifest's data section: every input day and its sha256


def _minutes(
    source: DaySource, symbol: str, days: Sequence[date], features: Sequence[str]
) -> tuple[Minutes, list[dict[str, Any]]]:
    frames, valid, runs, records = [], [], [], []
    run = -1
    for k, day in enumerate(days):
        if k == 0 or day != days[k - 1] + timedelta(days=1):
            run += 1
        f, v, digest = read_day(source, symbol, day, features)
        frames.append(torch.from_numpy(f))
        valid.append(torch.from_numpy(v))
        runs.append(torch.full((MINUTES,), run, dtype=torch.int64))
        records.append({"day": day.isoformat(), "sha256": digest, "valid_minutes": int(v.sum())})
    return Minutes(torch.cat(frames), torch.cat(valid), torch.cat(runs)), records


def load(
    source: DaySource,
    span: Span,
    symbol: str,
    features: Sequence[str],
    train_days: Sequence[date],
    val_days: Sequence[date],
) -> Dataset:
    """Training and validation minutes of `symbol` from `source`, for a model that trains on
    `span`. Every day is checked against the span before any is read. No day may serve both sides,
    and days must be unique and in date order on each."""
    for name, days in (("training", train_days), ("validation", val_days)):
        if not days:
            raise DataError(f"no {name} days")
        if list(days) != sorted(set(days)):
            raise DataError(f"{name} days must be unique and in date order")
    shared = sorted(set(train_days) & set(val_days))
    if shared:
        raise DataError(f"days {[d.isoformat() for d in shared]} serve training and validation")
    for day in (*train_days, *val_days):  # every refusal before any read
        span.check(symbol, day)
    train, train_records = _minutes(source, symbol, train_days, features)
    val, val_records = _minutes(source, symbol, val_days, features)
    sources = {
        "kind": "market",
        "symbol": symbol,
        "span": [span.start.isoformat(), span.end.isoformat()],
        "features": list(features),
        "train": train_records,
        "val": val_records,
        "confirmatory_start": CONFIRMATORY.isoformat(),
    }
    return Dataset(train, val, sources)
