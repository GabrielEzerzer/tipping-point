"""1-second feature days in, minute frames in host memory out: what the market JEPA trains on (#30).

The input contract, which the feature package (#30 M2) writes and any source must meet: one table
per symbol-day (UTC), 86,400 rows in second order, a float32 column per feature and a bool `gap`
column, True for a second without valid book data (a feature gap or an unsynced book). A source
hands over the table with the sha256 of the bytes it came from, which the training manifest pins.
`ParquetSource` reads the files `book.features.reduce_run` writes and hashes their bytes as it
records them. It refuses a file whose metadata names another symbol, day, feature format
(`book.features.FORMAT`) or feature list (`book.features.FEATURES`).

The cross-day step (#30 research pass 2.2, #165) turns the raw features into model inputs. Per
feature, in this order:
  1. `relative` (depth_5, depth_20): minus a trailing baseline, the median of the daily medians
     (over each day's valid seconds) of the days in [d - 7, d) that the load holds. A training day
     draws only on earlier training days, so the fit never sees a validation day; any other day
     draws on every earlier day it is loaded with. A day with no such earlier day (the first day
     of a range, or one after a break of 7 days or more) has no baseline: it serves as a baseline
     for later days and trains on nothing (every second is set to a gap). Days 2 to 7 of a range
     use the 1 to 6 earlier days they have.
     `log` (slope_b, slope_a): the natural log; both are positive distances in bps with a long
     right tail.
  2. A robust standardisation (x - centre) / scale fitted on the valid (non-gap) seconds of the
     training days only. centre is the median. scale is 1.4826 times the median absolute deviation
     from the centre of the values that differ from it, floored at 1e-6 |centre|. For a feature
     that is rarely at its median this is the usual MAD, which equals the standard deviation of a
     normal variable. The research pass's IQR is 0 for ret, rv100 and moves (about 84 % of seconds
     have no mid move, BTCUSDT 2023-06-10) and for wide (0 in about 98.5 %), and a q05-q95 spread
     is still 0 for a feature at its median 95 % of the time; leaving the median's own seconds out
     measures the size of the moves themselves, so the scale is finite and non-zero whenever any
     value differs from the centre. A feature with no such value is refused. The floor keeps
     float32 rounding around a large centre from being blown up.
  3. `asinh` (ret, rv100, ofi): asinh of the standardised value, a signed log that is linear
     within about one scale unit of the centre and logarithmic in the tails. Taken after the
     scale, it needs no constant in the feature's own units.
Every other feature (moves, micro_dev, imb_*, depth_asym, wide, and any feature without an entry,
as in the market toy) is only standardised. Gap seconds are zeroed last, so a gap reads the
centre. The fitted record (`Scaling`, in the manifest's data section) holds each feature's
transform, centre and scale with the symbol, feature format, training span and days it was fitted
on.

Validation days should form a block after the training days. `load` takes them anywhere, but one
before the training days draws its depth baseline from earlier validation days alone, so the
first of them is all gap, and one between training days sits among days the model trained on,
which makes it an easier test than a later day.

`frames` reads days for readouts and extraction through a frozen record and never refits. It
refuses a record fitted on another symbol or feature format. Its baselines follow the rule of
step 1 with the record's training days, so a training day comes out bitwise as it trained, and a
validation day as it was validated when it is read with the same earlier days. A training day
whose earlier training days are not all handed in is refused. Days passed as `lead` are read for
the baseline alone and emit no minutes, so a series can be read one day at a time; the exploratory
read from 2023-10-01 takes the last training days as its lead.

A minute is valid when none of its 60 seconds is a gap. Invalid minutes are never trained on:
their latents are left out of every loss term, the predictor sees a mask token in their place, and
readouts give them NaN. Training draws only chunks that hold two valid minutes (`Minutes.starts`):
however long a feed outage, no batch divides a loss term by zero.

Leakage (#30 research pass 4.3, tests 1 and 2): `load` refuses any day outside the model's training
span [start, end), any day on or after 2024-01-01 whatever the span says (the confirmatory span is
never a training input), and any symbol other than the model's own; `frames` and `ParquetSource`
refuse the confirmatory span too. The manifest's data section lists every input day with its
source sha256 and the fitted record, and `resume` refuses a dataset whose section differs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor

from ..book.features import FEATURES, FORMAT

SECONDS = 86_400
MINUTE = 60  # seconds per frame
MINUTES = SECONDS // MINUTE
GAP = "gap"
CONFIRMATORY = date(2024, 1, 1)  # no market day from here on is ever a training input
# The cross-day transform of each book feature (module notes); a feature not named here is only
# standardised.
TRANSFORMS = {
    "ret": "asinh",
    "rv100": "asinh",
    "ofi": "asinh",
    "depth_5": "relative",
    "depth_20": "relative",
    "slope_b": "log",
    "slope_a": "log",
}
BASELINE_DAYS = 7  # the trailing window of the relative depth
MAD = 1.4826  # the median absolute deviation of a normal variable, in standard deviations
FLOOR = 1e-6  # the scale is at least this times |centre|


class DataError(ValueError):
    """A day the market loader refuses or cannot read."""


class DaySource(Protocol):
    def read(self, symbol: str, day: date) -> tuple[pa.Table, str]:
        """One symbol-day under the input contract, with the sha256 of the bytes it came from."""
        ...


class ParquetSource:
    """The per-day feature files under `root`, as `book.features.reduce_run` writes them:
    `<root>/features/<SYMBOL>/<YYYY-MM-DD>.parquet`, hashed over the file's bytes."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def read(self, symbol: str, day: date) -> tuple[pa.Table, str]:
        if day >= CONFIRMATORY:
            raise DataError(f"{day} is in the confirmatory span (from {CONFIRMATORY}): never read")
        path = self.root / "features" / symbol / f"{day.isoformat()}.parquet"
        if not path.is_file():
            raise DataError(f"{symbol} {day}: no feature file at {path}")
        raw = path.read_bytes()
        table = pq.read_table(pa.BufferReader(raw))
        meta = json.loads((table.schema.metadata or {}).get(b"tp.book_features", b"{}"))
        if meta.get("symbol") != symbol or meta.get("day") != day.isoformat():
            raise DataError(f"{path} holds {meta.get('symbol')} {meta.get('day')}")
        if meta.get("format") != FORMAT:
            raise DataError(
                f"{path} is feature format {meta.get('format')}; this code reads {FORMAT}"
            )
        if meta.get("features") != list(FEATURES):
            raise DataError(f"{path} holds the features {meta.get('features')}, not {FEATURES}")
        return table, hashlib.sha256(raw).hexdigest()


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


@dataclass(frozen=True)
class Scaling:
    """The fitted cross-day record (module notes), as the manifest's data section keeps it."""

    symbol: str  # the symbol whose days it was fitted on
    features: list[str]
    transforms: list[str]
    centre: list[float]
    scale: list[float]
    span: list[str]  # the model's training span [start, end) it was fitted for
    days: list[str]  # the training days it was fitted on
    feature_format: int  # book.features.FORMAT of the code that fitted it
    baseline_days: int = BASELINE_DAYS

    def check(self, symbol: str) -> None:
        """Refuses a record fitted on another symbol, or made under other transforms or another
        feature format than this code reads."""
        if self.symbol != symbol:
            raise DataError(f"a record fitted on {self.symbol}, not {symbol}")
        expected = [TRANSFORMS.get(name, "none") for name in self.features]
        if self.transforms != expected or self.baseline_days != BASELINE_DAYS:
            raise DataError(f"a record of transforms {self.transforms}; this code: {expected}")
        if self.feature_format != FORMAT:
            raise DataError(
                f"a record fitted on feature format {self.feature_format}; this code reads {FORMAT}"
            )


def read_day(
    source: DaySource, symbol: str, day: date, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, str]:
    """(values (SECONDS, F) float32 as read, gap (SECONDS,) bool, sha256) for one day, checked
    against the input contract. Values behind a gap are left as they are: nothing here is a model
    input before the cross-day step."""
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
        columns.append(values)
    return np.stack(columns, axis=1), gap.copy(), digest


@dataclass
class _Day:
    day: date
    values: np.ndarray  # (SECONDS, F) float32, transformed in place
    gap: np.ndarray  # (SECONDS,) bool
    digest: str
    depth: np.ndarray  # each `relative` feature's median over the day's valid seconds, as read


def _relative(features: Sequence[str]) -> list[int]:
    return [k for k, name in enumerate(features) if TRANSFORMS.get(name) == "relative"]


def _read(
    source: DaySource, symbol: str, days: Sequence[date], features: Sequence[str]
) -> list[_Day]:
    relative = _relative(features)
    out = []
    for day in days:
        values, gap, digest = read_day(source, symbol, day, features)
        held = values[~gap][:, relative].astype(np.float64)
        depth = np.median(held, axis=0) if len(held) else np.full(len(relative), np.nan)
        out.append(_Day(day, values, gap, digest, depth))
    return out


def _prepare(
    read: Sequence[_Day], pool: Sequence[_Day], features: Sequence[str], fitted: set[date]
) -> None:
    """Step 1 in place: `relative` features minus their baseline from the `pool` days before each
    day, only the `fitted` (training) ones for a day that is one of them (a day without a baseline
    becomes all gap), `log` features logged."""
    relative = _relative(features)
    if relative:
        for d in read:
            window = [
                p.depth
                for p in pool
                if d.day - timedelta(days=BASELINE_DAYS) <= p.day < d.day
                and (p.day in fitted or d.day not in fitted)
                and not np.isnan(p.depth).any()
            ]
            if not window:
                d.gap[:] = True  # a baseline for later days only
                continue
            base = np.median(np.stack(window), axis=0)
            d.values[:, relative] = d.values[:, relative].astype(np.float64) - base
    for k, name in enumerate(features):
        if TRANSFORMS.get(name) != "log":
            continue
        for d in read:
            with np.errstate(invalid="ignore", divide="ignore"):
                column = np.log(d.values[:, k].astype(np.float64))
            if not np.isfinite(column[~d.gap]).all():
                raise DataError(f"{d.day}: {name} is not positive outside gap seconds")
            d.values[:, k] = column


def _fit(train: Sequence[_Day], features: Sequence[str]) -> tuple[list[float], list[float]]:
    """Step 2's centre and scale per feature, from the valid seconds of the training days."""
    centre, scale = [], []
    for k, name in enumerate(features):
        y = np.concatenate([d.values[~d.gap, k] for d in train]).astype(np.float64)
        if not len(y):
            raise DataError("no valid training second to fit the scaling on")
        c = float(np.median(y))
        off = np.abs(y - c)
        off = off[off > 0]
        if not len(off):
            raise DataError(f"{name} is {c} in every valid training second: nothing to scale")
        centre.append(c)
        scale.append(max(MAD * float(np.median(off)), FLOOR * abs(c)))
    return centre, scale


def _apply(read: Sequence[_Day], scaling: Scaling) -> None:
    """Steps 2 and 3 in place, then gap seconds to zero (the centre)."""
    for k, transform in enumerate(scaling.transforms):
        for d in read:
            z = (d.values[:, k].astype(np.float64) - scaling.centre[k]) / scaling.scale[k]
            d.values[:, k] = np.arcsinh(z) if transform == "asinh" else z
    for d in read:
        d.values[d.gap] = 0.0  # nulls and anything else behind a gap never reach the model


@dataclass(frozen=True)
class Minutes:
    """Minute frames of whole days, in date order, in host memory."""

    frames: Tensor  # (minutes, MINUTE, F) float32
    valid: Tensor  # (minutes,) bool
    run: Tensor  # (minutes,) int64: which run of consecutive days each minute belongs to

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def starts(self, length: int, whole: bool = False) -> Tensor:
        """The first minutes of every chunk of `length` consecutive minutes inside one run that
        holds at least two valid minutes; with `whole`, only chunks whose every minute is valid.
        Two, so that the later one is a prediction target with a valid minute before it and VICReg
        pools at least two latents in any batch: a chunk with fewer leaves a loss term 0 / 0."""
        if len(self) < length:
            return torch.zeros(0, dtype=torch.int64)
        first = torch.arange(len(self) - length + 1)
        count = torch.cat([torch.zeros(1, dtype=torch.int64), self.valid.cumsum(0)])
        held = count[first + length] - count[first]
        ok = (self.run[first] == self.run[first + length - 1]) & (held >= (length if whole else 2))
        return first[ok]

    def chunks(self, first: Tensor, length: int) -> tuple[Tensor, Tensor]:
        """(frames (B, length, MINUTE, F), valid (B, length)) of the chunks starting at `first`."""
        index = first.unsqueeze(1) + torch.arange(length)
        return self.frames[index], self.valid[index]


@dataclass
class Dataset:
    train: Minutes
    val: Minutes
    sources: dict[str, Any]  # the manifest's data section: every input day, its sha256, the scaling


def _minutes(read: Sequence[_Day]) -> tuple[Minutes, list[dict[str, Any]]]:
    frames, valid, runs, records = [], [], [], []
    run = -1
    for k, d in enumerate(read):
        if k == 0 or d.day != read[k - 1].day + timedelta(days=1):
            run += 1
        v = ~d.gap.reshape(MINUTES, MINUTE).any(1)
        frames.append(torch.from_numpy(d.values.reshape(MINUTES, MINUTE, -1)))
        valid.append(torch.from_numpy(v))
        runs.append(torch.full((MINUTES,), run, dtype=torch.int64))
        records.append(
            {"day": d.day.isoformat(), "sha256": d.digest, "valid_minutes": int(v.sum())}
        )
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
    `span`, through the cross-day step fitted on the training days (module notes). Every day is
    checked against the span before any is read. No day may serve both sides, and days must be
    unique and in date order on each."""
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
    train = _read(source, symbol, train_days, features)
    val = _read(source, symbol, val_days, features)
    # the fit sees training days only, baselines included
    _prepare([*train, *val], [*train, *val], features, set(train_days))
    centre, scale = _fit(train, features)
    scaling = Scaling(
        symbol=symbol,
        features=list(features),
        transforms=[TRANSFORMS.get(name, "none") for name in features],
        centre=centre,
        scale=scale,
        span=[span.start.isoformat(), span.end.isoformat()],
        days=[d.isoformat() for d in train_days],
        feature_format=FORMAT,
    )
    _apply(train, scaling)
    _apply(val, scaling)
    train_minutes, train_records = _minutes(train)
    val_minutes, val_records = _minutes(val)
    sources = {
        "kind": "market",
        "symbol": symbol,
        "span": [span.start.isoformat(), span.end.isoformat()],
        "features": list(features),
        "train": train_records,
        "val": val_records,
        "confirmatory_start": CONFIRMATORY.isoformat(),
        "scaling": asdict(scaling),
    }
    return Dataset(train_minutes, val_minutes, sources)


def frames(
    source: DaySource,
    symbol: str,
    days: Sequence[date],
    scaling: dict[str, Any],
    lead: Sequence[date] = (),
) -> tuple[Minutes, list[dict[str, Any]]]:
    """Minutes of `days` through a frozen record (a manifest's `data["scaling"]`), for readouts
    and extraction, with a record of every day read: nothing is refitted. Relative depth draws on
    the earlier days of `lead` and `days` by the rule of `load` (module notes). `lead` days come
    before `days` and are read for that alone: they emit no minutes and their records say `lead`.
    A training day whose earlier training days are not all handed in is refused, since its
    baseline would differ from the one it trained on."""
    record = Scaling(**scaling)
    record.check(symbol)
    if not days or [*lead, *days] != sorted({*lead, *days}):
        raise DataError("days must be given, unique and in date order, after every lead day")
    for day in (*lead, *days):
        if day >= CONFIRMATORY:
            raise DataError(f"{day} is in the confirmatory span (from {CONFIRMATORY}): never read")
    fitted = {date.fromisoformat(d) for d in record.days}
    if _relative(record.features):
        held = {*lead, *days}
        for day in sorted(fitted.intersection(days)):
            window = day - timedelta(days=BASELINE_DAYS)
            missing = sorted(t for t in fitted - held if window <= t < day)
            if missing:
                raise DataError(
                    f"training day {day} needs the training days {[t.isoformat() for t in missing]}"
                    " for its baseline: hand them in as `lead`"
                )
    pool = _read(source, symbol, [*lead, *days], record.features)
    before, read = pool[: len(lead)], pool[len(lead) :]
    _prepare(read, pool, record.features, fitted)
    _apply(read, record)
    minutes, records = _minutes(read)
    leads = [{"day": p.day.isoformat(), "sha256": p.digest, "lead": True} for p in before]
    return minutes, [*leads, *records]
