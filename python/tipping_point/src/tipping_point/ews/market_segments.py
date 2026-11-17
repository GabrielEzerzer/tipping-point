# The market warning test's segments (#30 M5; research pass 6.3 and 6.6, choices 9 and 11), cut
# from a labels run of the frozen market-labels file. The README's market warning test section
# holds the rules.

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
    onsets: dict[str, np.ndarray]  # every event onset (ns) of each symbol the run labels
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


def _method(method: str) -> None:
    if method not in prereg.DESEASON_METHODS:
        raise ValueError(f"method must be one of {list(prereg.DESEASON_METHODS)}: {method!r}")


def slot_profile(series: Minutes, fit: tuple[date, date], method: str) -> np.ndarray:
    """(2 x 288,) per slot, over the UTC days fit[0] to fit[1], both included: the returns' root
    mean square ("scale") or mean ("mean"); NaN for a slot with no return there. The fit reads
    pilot days only: one ending on or after 2024-01-01 is refused."""
    _method(method)
    if fit[1] >= prereg.CONFIRMATORY:
        raise SegmentError(_closed("a slot profile fitted on"))
    ms = series.start_ms + MINUTE_MS * np.arange(len(series.values), dtype=np.int64)
    inside = (ms >= archive.day_ms(fit[0])) & (ms < archive.day_ms(fit[1] + timedelta(days=1)))
    use = inside & np.isfinite(series.values)
    slot = _slots(series)[use]
    power = 2 if method == "scale" else 1
    total = np.bincount(slot, weights=series.values[use] ** power, minlength=2 * msvol.DAY_BARS)
    count = np.bincount(slot, minlength=2 * msvol.DAY_BARS)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = total / count
    return np.sqrt(mean) if method == "scale" else mean


def deseasonalise(series: Minutes, profile: np.ndarray, method: str) -> Minutes:
    """Each return divided by its slot's scale ("scale") or less its slot's mean ("mean"), the
    profile slot_profile fitted with the same method; a slot with no profile leaves NaN."""
    _method(method)
    by_minute = profile[_slots(series)]
    with np.errstate(divide="ignore", invalid="ignore"):
        values = series.values / by_minute if method == "scale" else series.values - by_minute
    return Minutes(series.start_ms, values)


def eth_only(eth_ns: np.ndarray, btc_ns: np.ndarray, bars: int) -> np.ndarray:
    """Per ETH onset, True when no BTC onset lies within `bars` bars of it either side
    (reporting.eth_only_bars)."""
    eth_ns, btc_ns = np.asarray(eth_ns, np.int64), np.sort(np.asarray(btc_ns, np.int64))
    window = bars * msvol.BAR_NS
    low = np.searchsorted(btc_ns, eth_ns - window, side="left")
    high = np.searchsorted(btc_ns, eth_ns + window, side="right")
    return high == low


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
    score_from: date | None = None,
    scoring: prereg.Prereg | None = None,
) -> Segments:
    manifest = check_labels(run, labels, scoring)
    if scoring is not None:
        if score_from is not None:
            raise SegmentError(
                f"{scoring.name} fixes the scored span: score_from is for exploratory runs"
            )
        doc = scoring.doc
        fixed = {
            "horizons": list(doc["segments"]["horizons"]),
            "span": doc["indicators"]["window"] + doc["indicators"]["length"] - 1,
            "subsample": doc["segments"]["subsample"],
            "seed": doc["segments"]["seed"],
        }
        given = {"horizons": list(horizons), "span": span, "subsample": subsample, "seed": seed}
        differ = [k for k in fixed if given[k] != fixed[k]]
        if differ:
            raise SegmentError(
                f"{scoring.name} fixes {', '.join(differ)}: "
                + ", ".join(f"{k} {fixed[k]!r}, not {given[k]!r}" for k in differ)
            )
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
    all_onsets = {
        symbol: np.array(
            [
                t
                for t, s, k in zip(
                    episodes["ts_ns"], episodes["symbol"], episodes["kind"], strict=True
                )
                if s == symbol and k == "event"
            ],
            np.int64,
        )
        for symbol in manifest["params"]["symbols"]
    }
    first_day_ms = archive.day_ms(score_from) if score_from else None
    rng = np.random.default_rng(seed)
    events: dict[tuple[str, int], Block] = {}
    controls: dict[str, Block] = {}
    counts: list[dict[str, Any]] = []
    for symbol in sorted(series):
        minutes = series[symbol]
        onsets = all_onsets[symbol]
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
        if first_day_ms is not None:
            onsets = onsets[onsets // 1_000_000 >= first_day_ms]
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
        anchors = np.array(anchor, np.int64)
        if first_day_ms is not None:
            later = first >= first_day_ms
            first, anchors = first[later], anchors[later]
        if scoring is None and np.any(first + span * MINUTE_MS > CONFIRMATORY_MS):
            raise SegmentError(_closed(f"{symbol}'s control segments reach"))
        rows, kept, share, outside, invalid = _cut(minutes, first, span, high, high_t0)
        block = Block(rows, first[kept], anchors[kept], share)
        if subsample and subsample < len(rows):
            pick = np.sort(rng.choice(len(rows), subsample, replace=False))
            block = Block(rows[pick], block.first_ms[pick], block.anchor_ns[pick], share[pick])
        controls[symbol] = block
        counts.append(
            _count(symbol, "control", None, len(first), len(rows), len(block.x), invalid, outside)
        )
    return Segments(span, events, controls, all_onsets, counts)


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
