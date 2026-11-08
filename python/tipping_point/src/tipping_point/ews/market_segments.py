"""The market warning test's segments (#30 M5; research pass sections 6.3 and 6.6, owner's calls
9 and 11): what market-ews scores, cut from a labels run of the frozen market-labels file.

  close_returns  a symbol's 1-minute close-to-close log returns on the minute grid, NaN for a minute
                 with no kline or whose previous minute has none (prereg.MISSING_MINUTE)
  slot_scale     the root mean square of the returns of each of the labeller's 2 x 288 slots
                 (weekday or weekend, by 5-minute time of day) over a fit span of UTC days
  deseasonalise  each return divided by its slot's scale (prereg.DESEASON)
  check_labels   the labels run's manifest, once the labels file is frozen at its tag and the run
                 was labelled with its parameters (and its stamp, when it has one)
  build          the event segments per (symbol, horizon) and the control segments per symbol

Lengths: a segment is `span` minutes, Pipeline.span = window + length - 1. An onset is the start of
a 5-minute bar; at horizon h bars its segment ends with the minute that closes at onset - 5h
minutes, so it holds nothing of the h bars before the onset (market.nulls.pre_onset_rows, the
floor's rule). The length is the same at every horizon, so at long horizons a segment starts before
the labels' clean window of C bars and may read bars with H = 1: each segment's share of minutes
in such bars is reported (call 11). Controls are the non-overlapping spans cut in time order from
the start of each null stretch (market.nulls.null_rows), one set for every horizon, then
optionally a seeded subsample of them. A segment that holds a minute with no return is excluded
(the same segments for every indicator), and so is one that reaches outside the series given;
both are counted.

The confirmatory span (from 2024-01-01) opens only through a frozen market-ews file, as the market
command's --prereg opens it for labels: build refuses a series, onset or control segment reaching
it unless `scoring` is a market-ews file frozen at its tag that names this labels file, and a run
labelling those days must carry the frozen labels' stamp in its manifest. An environment switch,
as cascade.recall takes, would open those days to any caller; the frozen file ties them to the
pre-registered test.

This module reads labels and is fed the series; the labels package (market/) imports none of it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .. import engine
from ..market import archive, msvol
from . import prereg

MINUTE_MS = 60_000
BAR_MINUTES = 5
BAR_MS = BAR_MINUTES * MINUTE_MS
CONFIRMATORY_MS = archive.day_ms(prereg.CONFIRMATORY)


class SegmentError(ValueError):
    """A labels run or series the segments may not be cut from."""


@dataclass(frozen=True)
class Minutes:
    """One value per minute from `start_ms`; NaN where there is none."""

    start_ms: int
    values: np.ndarray

    @property
    def end_ms(self) -> int:
        return self.start_ms + len(self.values) * MINUTE_MS


@dataclass(frozen=True)
class Block:
    """Kept segments, one row each: the series, the first minute, and the onset (an event's) or the
    null stretch's start (a control's), and the share of minutes in bars with H = 1."""

    x: np.ndarray  # (segments, span)
    first_ms: np.ndarray
    anchor_ns: np.ndarray
    high_share: np.ndarray


@dataclass(frozen=True)
class Segments:
    span: int
    events: dict[tuple[str, int], Block]  # by (symbol, horizon in bars)
    controls: dict[str, Block]  # by symbol, one set for every horizon
    counts: list[dict[str, Any]]  # per (symbol, kind, horizon): _count


def close_returns(klines: Mapping[str, np.ndarray], start_ms: int, minutes: int) -> Minutes:
    """The 1-minute close-to-close log returns of `minutes` minutes from `start_ms`, from the
    columns of archive.collect: the return of minute m is ln(close_m / close_(m-1)), NaN when
    either kline is missing (the first minute has no previous one). A stamp given twice keeps its
    first row, and rows off the grid are dropped, as market.bars reads them."""
    offset = klines["open_time"] - start_ms
    index = offset // MINUTE_MS
    on = np.flatnonzero((offset % MINUTE_MS == 0) & (index >= 0) & (index < minutes))
    _, first = np.unique(index[on], return_index=True)
    close = np.full(minutes, np.nan)
    close[index[on[first]]] = klines["close"][on[first]]
    returns = np.full(minutes, np.nan)
    returns[1:] = np.log(close[1:] / close[:-1])
    return Minutes(start_ms, returns)


def _slots(series: Minutes) -> np.ndarray:
    ms = series.start_ms + MINUTE_MS * np.arange(len(series.values), dtype=np.int64)
    return msvol._slots(ms * 1_000_000)  # the labeller's own slots


def slot_scale(series: Minutes, fit: tuple[date, date]) -> np.ndarray:
    """(2 x 288,) the root mean square of the returns of each slot over the UTC days fit[0] to
    fit[1], both included; NaN for a slot with no return there."""
    ms = series.start_ms + MINUTE_MS * np.arange(len(series.values), dtype=np.int64)
    inside = (ms >= archive.day_ms(fit[0])) & (ms < archive.day_ms(fit[1] + timedelta(days=1)))
    use = inside & np.isfinite(series.values)
    slot = _slots(series)[use]
    total = np.bincount(slot, weights=series.values[use] ** 2, minlength=2 * msvol.DAY_BARS)
    count = np.bincount(slot, minlength=2 * msvol.DAY_BARS)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(total / count)


def deseasonalise(series: Minutes, scale: np.ndarray) -> Minutes:
    """Each return divided by its slot's scale (a slot with no scale leaves NaN)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return Minutes(series.start_ms, series.values / scale[_slots(series)])


def _read(run: Path, table: str) -> Any:
    return engine.read(run.parent.parent, run.parent.name, run.name, table)


def check_labels(
    run: Path, labels: prereg.Prereg, scoring: prereg.Prereg | None = None
) -> dict[str, Any]:
    """The labels run's manifest, if the segments may be cut from it: the labels file is frozen at
    its tag (prereg.frozen), the run was labelled with that file's [labeller] and symbols, and its
    stamp, if it has one, is that file's. A run labelling the confirmatory span needs the stamp and
    a `scoring` file: a market-ews file frozen at its tag whose data.labels names this labels file.
    Raises PreregError or SegmentError."""
    if labels.doc["data"]["domain"] != prereg.MARKET_LABELS:
        raise SegmentError(f"{labels.path} is not a {prereg.MARKET_LABELS} file")
    ref = prereg.frozen(labels)
    manifest = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
    params = manifest.get("params", {})
    if "labeller" not in params:
        raise SegmentError(f"{run} is not a labels run")
    if params["labeller"] != labels.doc["labeller"]:
        raise SegmentError(f"{run} was labelled with other parameters than {labels.name}'s")
    if not set(params["symbols"]) <= set(labels.doc["data"]["symbols"]):
        raise SegmentError(f"{run} labels symbols {labels.name} does not name")
    stamp = params.get("prereg")
    if stamp is not None and (stamp["sha256"], stamp["tag"]) != (labels.sha256, ref.tag):
        raise SegmentError(f"{run} was labelled under another file than {labels.name}")
    if scoring is not None:
        if scoring.doc["data"]["domain"] != prereg.MARKET_EWS:
            raise SegmentError(f"{scoring.path} is not a {prereg.MARKET_EWS} file")
        prereg.frozen(scoring)
        named = scoring.doc["data"]["labels"]
        if (named["tag"], named["sha256"]) != (ref.tag, labels.sha256):
            raise SegmentError(f"{scoring.name} names other labels than {labels.name}")
    if date.fromisoformat(params["end"]) >= prereg.CONFIRMATORY:
        if scoring is None:
            raise SegmentError(_closed(f"{run} labels"))
        if stamp is None:
            raise SegmentError(f"{run} labels confirmatory days without {labels.name}'s stamp")
    return manifest


def _closed(what: str) -> str:
    return (
        f"{what} days from {prereg.CONFIRMATORY}, the confirmatory span, which only a frozen "
        f"{prereg.MARKET_EWS} file opens (scoring=)"
    )


def _cut(
    series: Minutes, first_ms: np.ndarray, span: int, high: np.ndarray, high_t0_ms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """The segments of `span` minutes from each first minute: (kept rows, kept mask, their share of
    minutes in bars with H = 1, outside, invalid)."""
    first_ms = np.asarray(first_ms, np.int64)
    start = (first_ms - series.start_ms) // MINUTE_MS
    inside = (start >= 0) & (start + span <= len(series.values))
    rows = series.values[start[inside, None] + np.arange(span)]
    valid = np.isfinite(rows).all(axis=1)
    kept = np.zeros(len(first_ms), bool)
    kept[np.flatnonzero(inside)[valid]] = True
    minute_ms = first_ms[kept, None] + MINUTE_MS * np.arange(span)
    bar = (minute_ms - high_t0_ms) // BAR_MS
    known = (bar >= 0) & (bar < len(high))
    is_high = np.zeros(bar.shape, bool)
    is_high[known] = high[bar[known]]
    share = is_high.mean(axis=1) if len(minute_ms) else np.empty(0)
    return rows[valid], kept, share, int(np.sum(~inside)), int(np.sum(~valid))


def build(
    run: Path,
    labels: prereg.Prereg,
    series: Mapping[str, Minutes],
    horizons: Sequence[int],
    span: int,
    *,
    subsample: int = 0,
    seed: int = 0,
    scoring: prereg.Prereg | None = None,
) -> Segments:
    """The event segments of each symbol's regime-shift onsets at each horizon (in bars) and its
    control segments, from the labels run `run` of the frozen file `labels` (check_labels), cut
    from `series` (one Minutes per symbol: returns, raw or deseasonalised). With subsample n > 0,
    n of each symbol's kept controls are drawn without replacement, symbols in sorted order from
    one generator seeded with `seed`, and kept in time order. Days from 2024-01-01 need `scoring`
    (the module docstring says why)."""
    manifest = check_labels(run, labels, scoring)
    unknown = set(series) - set(manifest["params"]["symbols"])
    if unknown:
        raise SegmentError(f"{run} has no labels for {sorted(unknown)}")
    if scoring is None:
        late = [s for s, m in series.items() if m.end_ms > CONFIRMATORY_MS]
        if late:
            raise SegmentError(_closed(f"the series of {late} reach"))
    episodes = _read(run, "episodes").to_pydict()
    stretches = _read(run, "nulls").to_pydict()
    table = _read(run, "labels").select(["ts_ns", "symbol", "h"]).to_pydict()
    rng = np.random.default_rng(seed)
    events: dict[tuple[str, int], Block] = {}
    controls: dict[str, Block] = {}
    counts: list[dict[str, Any]] = []
    for symbol in sorted(series):
        minutes = series[symbol]
        onsets = np.array(
            [
                t
                for t, s, k in zip(
                    episodes["ts_ns"], episodes["symbol"], episodes["kind"], strict=True
                )
                if s == symbol and k == "event"
            ],
            np.int64,
        )
        null = [
            (t, b)
            for t, s, b in zip(
                stretches["ts_ns"], stretches["symbol"], stretches["bars"], strict=True
            )
            if s == symbol
        ]
        mine = [i for i, s in enumerate(table["symbol"]) if s == symbol]
        bar_ns = np.array([table["ts_ns"][i] for i in mine], np.int64)
        high = np.array([table["h"][i] == 1 for i in mine], bool)
        high_t0 = int(bar_ns[0]) // 1_000_000 if len(bar_ns) else 0
        if len(bar_ns) and not np.array_equal(
            bar_ns, bar_ns[0] + msvol.BAR_NS * np.arange(len(bar_ns))
        ):
            raise SegmentError(f"{run}: {symbol}'s labels are not on one 5-minute grid")
        onset_ms = onsets // 1_000_000
        if scoring is None and np.any(onset_ms >= CONFIRMATORY_MS):
            raise SegmentError(_closed(f"{symbol}'s onsets reach"))
        for h in horizons:
            first = onset_ms - (BAR_MINUTES * h + span) * MINUTE_MS
            rows, kept, share, outside, invalid = _cut(minutes, first, span, high, high_t0)
            events[symbol, h] = Block(rows, first[kept], onsets[kept], share)
            counts.append(
                _count(symbol, "event", h, len(first), len(rows), len(rows), invalid, outside)
            )
        first_ms, anchor = [], []
        for t, bars in null:
            for k in range(BAR_MINUTES * bars // span):
                first_ms.append(t // 1_000_000 + k * span * MINUTE_MS)
                anchor.append(t)
        first = np.array(first_ms, np.int64)
        if scoring is None and np.any(first + span * MINUTE_MS > CONFIRMATORY_MS):
            raise SegmentError(_closed(f"{symbol}'s control segments reach"))
        rows, kept, share, outside, invalid = _cut(minutes, first, span, high, high_t0)
        block = Block(rows, first[kept], np.array(anchor, np.int64)[kept], share)
        if subsample and subsample < len(rows):
            pick = np.sort(rng.choice(len(rows), subsample, replace=False))
            block = Block(rows[pick], block.first_ms[pick], block.anchor_ns[pick], share[pick])
        controls[symbol] = block
        counts.append(
            _count(symbol, "control", None, len(first), len(rows), len(block.x), invalid, outside)
        )
    return Segments(span, events, controls, counts)


def _count(
    symbol: str,
    kind: str,
    horizon: int | None,
    cut: int,
    kept: int,
    scored: int,
    invalid: int,
    outside: int,
) -> dict[str, Any]:
    """cut = kept + excluded_invalid + excluded_outside; scored: kept, after any subsample."""
    return {
        "symbol": symbol,
        "kind": kind,
        "horizon_bars": horizon,
        "cut": cut,
        "kept": kept,
        "scored": scored,
        "excluded_invalid": invalid,
        "excluded_outside": outside,
    }
