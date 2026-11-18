# The same-input tier of the market warning test (#30; research pass 6.4 item 2, choice 8): the
# Bybit book features as classic series, descriptive in market-ews-v1. The README's market warning
# test section holds the rules.

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..book.features import DAY, FORMAT
from ..market import archive, book
from . import prereg
from .market_segments import Minutes, SegmentError, check_scoring

MINUTES = DAY // 60
OPEN_RANGES = ("training", "exploratory")  # never "confirmatory"


@dataclass(frozen=True)
class Read:
    series: dict[str, Minutes]  # by feature
    format: int  # book.features.FORMAT of every file read
    missing_days: list[str]  # days with no feature file: every minute has no value


def aggregate(table: pa.Table, aggregation: Mapping[str, str]) -> dict[str, np.ndarray]:
    """(1440,) per feature of `aggregation` from one day's 86,400-row feature table: the sum or
    mean of each minute's 60 seconds, NaN for a minute holding a gapped second."""
    if table.num_rows != DAY:
        raise SegmentError(f"a feature day holds {DAY} seconds, not {table.num_rows}")
    gap = table["gap"].to_numpy(zero_copy_only=False).reshape(MINUTES, 60)
    valid = ~gap.any(axis=1)
    out = {}
    for feature, how in aggregation.items():
        if how not in prereg.AGGREGATIONS:
            raise ValueError(
                f"{feature}: aggregation must be one of {prereg.AGGREGATIONS}: {how!r}"
            )
        x = table[feature].to_numpy(zero_copy_only=False).astype(np.float64).reshape(MINUTES, 60)
        if not np.isfinite(x[~gap]).all():  # every non-gap second, as jepa.days refuses such a day
            raise SegmentError(f"{feature} is not finite outside gap seconds")
        value = x.sum(axis=1) if how == "sum" else x.mean(axis=1)
        out[feature] = np.where(valid, value, np.nan)
    return out


def _range(day: date, ranges: tuple[str, ...]) -> str | None:
    for name in ranges:
        start, end = book.RANGES[name]
        if start <= day <= end:
            return name
    return None


def read(
    root: Path,
    symbol: str,
    first: date,
    last: date,
    aggregation: Mapping[str, str],
    *,
    scoring: prereg.Prereg | None = None,
) -> Read:
    """`symbol`'s minutes from `first` to `last` (UTC days, both included) under `root`, a
    bybit-book data folder holding the range folders: one Minutes per feature of `aggregation`.
    Refuses a span reaching 2024-01-01 before opening anything."""
    ranges = OPEN_RANGES
    if scoring is not None:
        check_scoring(scoring)
        fixed = scoring.doc["same_input"]["aggregation"]
        if dict(aggregation) != fixed:
            raise SegmentError(f"{scoring.name} fixes aggregation: {fixed!r}, not {aggregation!r}")
        ranges = (*OPEN_RANGES, "confirmatory")
    elif last >= prereg.CONFIRMATORY:
        raise SegmentError(
            f"the book features of days from {prereg.CONFIRMATORY}, the confirmatory span, which "
            f"only a frozen {prereg.MARKET_EWS} file opens (scoring=), with "
            f"{prereg.MARKET_LATENT} frozen too"
        )
    days = (last - first).days + 1
    values = {f: np.full(days * MINUTES, np.nan) for f in aggregation}
    missing = []
    for k in range(days):
        day = first + timedelta(days=k)
        name = _range(day, ranges)
        path = Path(root) / name / "features" / symbol / f"{day}.parquet" if name else None
        if path is None or not path.is_file():
            missing.append(day.isoformat())
            continue
        table = pq.read_table(path, columns=["gap", *aggregation])
        meta = json.loads((table.schema.metadata or {}).get(b"tp.book_features", b"{}"))
        if (meta.get("symbol"), meta.get("day")) != (symbol, day.isoformat()):
            raise SegmentError(f"{path} holds {meta.get('symbol')} {meta.get('day')}")
        if meta.get("format") != FORMAT:
            raise SegmentError(
                f"{path} is feature format {meta.get('format')}; this code reads {FORMAT}"
            )
        for feature, minutes in aggregate(table, aggregation).items():
            values[feature][k * MINUTES : (k + 1) * MINUTES] = minutes
    start = archive.day_ms(first)
    return Read({f: Minutes(start, v) for f, v in values.items()}, FORMAT, missing)
