"""The labels' reports (#29 S3; research pass section 4), each as Parquet tables plus a Markdown
summary for the issue. Nothing in them reads a clock, so a report repeats byte for byte.

  label_count  per asset and month: events, episodes, flickers, durations, share of time high;
               onsets by UTC hour and weekday; onset coincidence between assets; onsets near
               scheduled releases and cascade candidates; null hours and 4-hour slices
  stability    as-of labels against the oracle and under the declared variants and a plain
               threshold rule; the theta path with its implied durations, flagged refits and
               boundary jumps; the detection delay under the fitted theta; the aggTrades cross-check
  floor        the tables of nulls.circularity_floor
  write        <out dir>/<name>/<table>.parquet and summary.md

The out dir is the reports/ folder of the labels run they describe: `python -m tipping_point.market
reports` writes the first two there, and harness/market_floor.py the floor.

An input that comes from elsewhere is optional: the committed CPI, NFP and FOMC calendar (call 10;
releases.csv, read by release_calendar), the cascade candidates (S4) and the aggTrades RV. Without
it the report's `inputs` table and its summary say "not supplied", and its table is empty.

An onset is a bar's start: `events` counts event onsets, `onsets` every run of H = 1. A variant's
agreement is read over the bars both label, and only their onsets are matched. Shifts are the as-of
onset less the other's, in bars: positive when the as-of label comes later.
"""

from __future__ import annotations

import csv
import itertools
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..ising import io
from . import bars, msvol, nulls

HOUR_NS = 3_600 * 10**9
SLICE_BARS = 48  # #30's 4-hour scoring segment (research pass section 2, item 10)
COINCIDENCE_NS = HOUR_NS  # BTC-ETH onsets within +-1 hour
RELEASE_NS = HOUR_NS // 4  # onsets within +-15 minutes of a release
CASCADE_NS = HOUR_NS  # a cascade candidate near a regime onset: as for coincidence
ONSET_TOLERANCE = 3  # bars: onsets matched within +-3 bars
D_MINS = (6, 12, 36)  # the declared variants (research pass section 2, item 13)
THRESHOLDS = ((0.7, 0.3), (0.8, 0.2), (0.9, 0.1))
BASELINES = (14, 28, 56)
PLAIN_K = 0.5  # the plain rule: z above k sd(z), for D_min bars at least
NOT_SUPPLIED = "not supplied"
CALENDAR = Path(__file__).with_name("releases.csv")

Series = Mapping[str, tuple[dict[str, np.ndarray], msvol.Labels]]
Row = dict[str, Any]

_TEXT, _INT, _FLOAT = pa.string(), pa.int64(), pa.float64()


def _schema(**fields: pa.DataType) -> pa.Schema:
    return pa.schema(list(fields.items()))


_INPUTS = _schema(item=_TEXT, status=_TEXT)
LABEL_COUNT = {
    "monthly": _schema(
        symbol=_TEXT,
        month=_TEXT,
        labelled_bars=_INT,
        missing_bars=_INT,
        share_high=_FLOAT,
        events=_INT,
        episodes=_INT,
        flickers=_INT,
        pending=_INT,
        event_bars_median=_FLOAT,
        event_bars_max=_INT,
        episode_bars_median=_FLOAT,
        episode_bars_max=_INT,
        flicker_bars_median=_FLOAT,
    ),
    "onset_clock": _schema(symbol=_TEXT, unit=_TEXT, value=_INT, events=_INT, onsets=_INT),
    "coincidence": _schema(
        symbol_a=_TEXT,
        symbol_b=_TEXT,
        population=_TEXT,
        count_a=_INT,
        count_b=_INT,
        a_near_b=_INT,
        b_near_a=_INT,
    ),
    "releases": _schema(
        symbol=_TEXT, kind=_TEXT, releases_in_span=_INT, events_near=_INT, onsets_near=_INT
    ),
    "cascades": _schema(
        symbol=_TEXT,
        candidates=_INT,
        candidates_near_event=_INT,
        candidates_near_onset=_INT,
        events=_INT,
        events_near_candidate=_INT,
    ),
    "nulls": _schema(
        symbol=_TEXT,
        stretches=_INT,
        bars=_INT,
        hours=_FLOAT,
        missing_bars=_INT,
        slices_4h=_INT,
        slices_4h_gapless=_INT,
    ),
    "inputs": _INPUTS,
}
STABILITY = {
    "agreement": _schema(
        symbol=_TEXT,
        against=_TEXT,
        compared_bars=_INT,
        agreement=_FLOAT,
        kappa=_FLOAT,
        onsets=_INT,
        onsets_other=_INT,
        matched=_INT,
        shift_median=_FLOAT,
    ),
    "onset_shifts": _schema(ts_ns=_INT, symbol=_TEXT, oracle_ts_ns=_INT, shift_bars=_INT),
    "refits": _schema(
        ts_ns=_INT,
        symbol=_TEXT,
        flagged=pa.bool_(),
        reason=_TEXT,
        **{name: _FLOAT for name in msvol.Theta._fields},
        low_duration_bars=_FLOAT,
        high_duration_bars=_FLOAT,
        p_before=_FLOAT,
        p_refiltered=_FLOAT,
        jump=_FLOAT,
    ),
    "delay": _schema(
        symbol=_TEXT,
        population=_TEXT,
        episodes=_INT,
        missed=_INT,
        missed_share=_FLOAT,
        median=_FLOAT,
        p90=_FLOAT,
        max=_INT,
    ),
    "aggtrades": _schema(symbol=_TEXT, bars=_INT, log_corr=_FLOAT, median_log_ratio=_FLOAT),
    "inputs": _INPUTS,
}
FLOOR = {
    "garch": _schema(parameter=_TEXT, value=_FLOAT),
    "cells": _schema(
        scorer=_TEXT,
        horizon_bars=_INT,
        contaminated=pa.bool_(),
        delay_p90=_FLOAT,
        auc=_FLOAT,
        n_pos=_INT,
        n_neg=_INT,
        shift_auc_median=_FLOAT,
        shift_auc_lo=_FLOAT,
        shift_auc_hi=_FLOAT,
        shift_draws=_INT,
    ),
    "paths": _schema(
        path=_INT,
        bars=_INT,
        labelled_bars=_INT,
        refits=_INT,
        flagged_refits=_INT,
        events=_INT,
        unshifted_events=_INT,
        other_episodes=_INT,
        flickers=_INT,
        null_bars=_INT,
        delay_p90=_FLOAT,
    ),
    "path_cells": _schema(
        path=_INT, scorer=_TEXT, horizon_bars=_INT, auc=_FLOAT, n_pos=_INT, n_neg=_INT
    ),
}


@dataclass(frozen=True)
class Report:
    name: str
    tables: dict[str, pa.Table]
    summary: str


def write(report: Report, out_dir: Path) -> Path:
    """A report's Parquet tables and summary.md in a new folder <out dir>/<name>."""
    folder = Path(out_dir) / report.name
    folder.mkdir(parents=True)
    for name, table in report.tables.items():
        io.write_file(
            folder / f"{name}.parquet",
            lambda f, table=table: pq.write_table(table, f, compression="zstd"),
        )
    io.write_file(folder / "summary.md", lambda f: f.write(report.summary.encode()))
    return folder


def _table(schema: pa.Schema, rows: list[Row]) -> pa.Table:
    clean = [
        {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}
        for row in rows
    ]
    return pa.Table.from_pylist(clean, schema=schema)


def _utc(ts_ns: int) -> str:
    return str(np.datetime_as_string(np.datetime64(ts_ns, "ns"), unit="m")).replace("T", " ")


def _markdown(table: pa.Table) -> str:
    def cell(name: str, v: Any) -> str:
        if v is None:
            return "n/a"
        if name.endswith("ts_ns"):
            return _utc(v)
        return f"{v:.3g}" if isinstance(v, float) else str(v)

    if not table.num_rows:
        return "none"
    names = [name.replace("ts_ns", "utc") for name in table.column_names]
    head = ["| " + " | ".join(names) + " |", "|" + " --- |" * table.num_columns]
    rows = [
        "| " + " | ".join(cell(k, v) for k, v in row.items()) + " |" for row in table.to_pylist()
    ]
    return "\n".join(head + rows)


def _optional(table: pa.Table, supplied: bool) -> str:
    return _markdown(table) if supplied else NOT_SUPPLIED


def _status(supplied: bool) -> str:
    return "supplied" if supplied else NOT_SUPPLIED


def _median(values: list[int] | np.ndarray) -> float:
    return float(np.median(values)) if len(values) else math.nan


def _refuse_confirmatory(series: Series) -> None:
    for bar, _ in series.values():
        if len(bar["ts_ns"]):
            nulls.refuse_confirmatory(int(bar["ts_ns"][-1]))


def _onsets(found: list[Row], kind: str | None = None) -> np.ndarray:
    return np.array([e["onset"] for e in found if kind is None or e["kind"] == kind], np.int64)


def _near(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """For each of a, whether some b lies within +-window, ends included."""
    b = np.sort(b)
    return np.searchsorted(b, a - window, "left") < np.searchsorted(b, a + window, "right")


# ---------------------------------------------------------------- the label-count report


def release_calendar(path: Path = CALENDAR) -> dict[str, np.ndarray]:
    """The committed calendar's release times per kind (CPI, NFP, FOMC), sorted, in ts_ns: the BLS
    and Federal Reserve page each row names gives its date. 11 of the 12 FOMC times are the
    standard 14:00 Eastern slot, not read off each statement page."""
    found: dict[str, list[datetime]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            when = datetime.fromisoformat(row["time_utc"])
            if when.utcoffset() != timedelta(0):
                raise ValueError(f"{path.name}: {row['kind']} at {row['time_utc']} is not UTC")
            found.setdefault(row["kind"], []).append(when.replace(tzinfo=None))
    return {
        kind: np.sort(np.array(times, "datetime64[ns]").astype(np.int64))
        for kind, times in sorted(found.items())
    }


def label_count(
    series: Series,
    *,
    releases: Mapping[str, np.ndarray] | None = None,
    cascades: Mapping[str, np.ndarray] | None = None,
    cascade_span: tuple[str, str] | None = None,
) -> Report:
    """`releases` holds each kind's release times (CPI, NFP, FOMC), `cascades` each symbol's
    candidate onsets, both in ts_ns; a symbol absent from `cascades` has no row. `cascade_span`
    is the candidates' first and last UTC day (ISO); when it is narrower than the series', the
    cascades table counts only the events whose onset falls inside it and the summary says so."""
    _refuse_confirmatory(series)
    note, window = "", None
    if cascades is not None and cascade_span is not None and series:
        ts = next(iter(series.values()))[0]["ts_ns"]
        first, last = ts[[0, -1]].astype("datetime64[ns]").astype("datetime64[D]").astype(str)
        if cascade_span != (first, last):
            lo, hi = np.array(cascade_span, "datetime64[ns]").astype(np.int64)
            window = lo, hi + msvol.DAY_NS  # first day 00:00 UTC to last day 24:00 UTC
            note = (
                f"The cascade candidates cover {cascade_span[0]} to {cascade_span[1]} only, inside "
                f"the labels span of {first} to {last}; the table counts only the events whose "
                "onset falls inside the candidates' span."
            )
    rows: dict[str, list[Row]] = {name: [] for name in LABEL_COUNT}
    stamps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol, (bar, out) in sorted(series.items()):
        ts = bar["ts_ns"]
        events, onsets = ts[_onsets(out.episodes, "event")], ts[_onsets(out.episodes)]
        stamps[symbol] = events, onsets
        rows["monthly"] += _monthly(symbol, bar, out)
        rows["onset_clock"] += _clock(symbol, events, onsets)
        rows["nulls"].append(_null_row(symbol, bar, out))
        labelled = ts[out.h >= 0]
        for kind, times in sorted((releases or {}).items()):
            times = np.asarray(times, np.int64)
            inside = (times >= labelled[0]) & (times <= labelled[-1]) if labelled.size else []
            rows["releases"].append(
                {
                    "symbol": symbol,
                    "kind": kind,
                    "releases_in_span": int(np.sum(inside)),
                    "events_near": int(_near(events, times, RELEASE_NS).sum()),
                    "onsets_near": int(_near(onsets, times, RELEASE_NS).sum()),
                }
            )
        if cascades is not None and symbol in cascades:
            found = np.asarray(cascades[symbol], np.int64)
            counted = events
            if window is not None:
                counted = events[(events >= window[0]) & (events < window[1])]
            rows["cascades"].append(
                {
                    "symbol": symbol,
                    "candidates": len(found),
                    "candidates_near_event": int(_near(found, events, CASCADE_NS).sum()),
                    "candidates_near_onset": int(_near(found, onsets, CASCADE_NS).sum()),
                    "events": len(counted),
                    "events_near_candidate": int(_near(counted, found, CASCADE_NS).sum()),
                }
            )
    for a, b in itertools.combinations(sorted(series), 2):
        for k, population in enumerate(("events", "onsets")):
            times_a, times_b = stamps[a][k], stamps[b][k]
            rows["coincidence"].append(
                {
                    "symbol_a": a,
                    "symbol_b": b,
                    "population": population,
                    "count_a": len(times_a),
                    "count_b": len(times_b),
                    "a_near_b": int(_near(times_a, times_b, COINCIDENCE_NS).sum()),
                    "b_near_a": int(_near(times_b, times_a, COINCIDENCE_NS).sum()),
                }
            )
    rows["inputs"] = [
        {"item": "releases", "status": _status(releases is not None)},
        {"item": "cascades", "status": _status(cascades is not None)},
    ]
    tables = {name: _table(LABEL_COUNT[name], found) for name, found in rows.items()}
    summary = _label_count_summary(tables, stamps, releases is not None, cascades is not None, note)
    return Report("label_count", tables, summary)


def _monthly(symbol: str, bar: dict[str, np.ndarray], out: msvol.Labels) -> list[Row]:
    months = bar["ts_ns"].astype("datetime64[ns]").astype("datetime64[M]")
    labelled = out.h >= 0
    missing = ~(bar["rv"] > 0)
    rows = []
    for month in np.unique(months[labelled]):
        inside = labelled & (months == month)
        found: dict[str, list[int]] = {k: [] for k in ("event", "episode", "flicker", "pending")}
        for e in out.episodes:
            if months[e["onset"]] == month:
                found[e["kind"]].append(e["bars"])
        rows.append(
            {
                "symbol": symbol,
                "month": str(month),
                "labelled_bars": int(inside.sum()),
                "missing_bars": int((inside & missing).sum()),
                "share_high": float(np.mean(out.h[inside] == 1)),
                "events": len(found["event"]),
                "episodes": len(found["episode"]),
                "flickers": len(found["flicker"]),
                "pending": len(found["pending"]),
                "event_bars_median": _median(found["event"]),
                "event_bars_max": max(found["event"], default=None),
                "episode_bars_median": _median(found["episode"]),
                "episode_bars_max": max(found["episode"], default=None),
                "flicker_bars_median": _median(found["flicker"]),
            }
        )
    return rows


def _hours(ts_ns: np.ndarray) -> np.ndarray:
    return np.bincount((ts_ns % msvol.DAY_NS) // HOUR_NS, minlength=24)


def _weekdays(ts_ns: np.ndarray) -> np.ndarray:
    """From Monday: day 0, 1970-01-01, was a Thursday."""
    return np.bincount((ts_ns // msvol.DAY_NS + 3) % 7, minlength=7)


def _clock(symbol: str, events: np.ndarray, onsets: np.ndarray) -> list[Row]:
    return [
        {"symbol": symbol, "unit": unit, "value": v, "events": int(e), "onsets": int(o)}
        for unit, count in (("hour", _hours), ("weekday", _weekdays))
        for v, (e, o) in enumerate(zip(count(events), count(onsets), strict=True))
    ]


def _null_row(symbol: str, bar: dict[str, np.ndarray], out: msvol.Labels) -> Row:
    gaps = np.concatenate([[0], np.cumsum(~(bar["rv"] > 0))])
    slices = gapless = 0
    for s in out.nulls:
        count = s["bars"] // SLICE_BARS
        starts = s["start"] + SLICE_BARS * np.arange(count)
        slices += count
        gapless += int(np.sum(gaps[starts + SLICE_BARS] == gaps[starts]))
    total = sum(s["bars"] for s in out.nulls)
    return {
        "symbol": symbol,
        "stretches": len(out.nulls),
        "bars": total,
        "hours": total / 12.0,
        "missing_bars": sum(s["missing"] for s in out.nulls),
        "slices_4h": slices,
        "slices_4h_gapless": gapless,
    }


def _label_count_summary(
    tables: dict[str, pa.Table],
    stamps: dict[str, tuple[np.ndarray, np.ndarray]],
    releases: bool,
    cascades: bool,
    cascade_note: str,
) -> str:
    months = tables["monthly"].to_pylist()
    hero = []
    for row in tables["nulls"].to_pylist():
        symbol = row["symbol"]
        own = [m for m in months if m["symbol"] == symbol]
        labelled = sum(m["labelled_bars"] for m in own)
        high = sum(m["share_high"] * m["labelled_bars"] for m in own)
        share = high / labelled if labelled else math.nan
        counts = {k: sum(m[k] for m in own) for k in ("events", "episodes", "flickers", "pending")}
        hero.append(
            f"- {symbol}: {counts['events']} events, {counts['episodes']} other episodes, "
            f"{counts['flickers']} flickers, {counts['pending']} pending; high {share:.1%} of "
            f"{labelled} labelled bars; {row['hours']:.0f} null hours hold {row['slices_4h']} "
            f"4-hour slices ({row['slices_4h_gapless']} without a missing bar)."
        )
    clock = [
        f"- {symbol}, all onsets by UTC hour 00-23: {' '.join(map(str, _hours(onsets).tolist()))}; "
        f"by weekday Mon-Sun: {' '.join(map(str, _weekdays(onsets).tolist()))}."
        for symbol, (_, onsets) in stamps.items()
    ]
    return "\n\n".join(
        [
            "# Label-count report",
            "\n".join(hero or ["No labelled series."]),
            "## Per asset and month",
            _markdown(tables["monthly"]),
            "## Onsets by the clock (the deseasonalisation check)",
            "\n".join(clock or ["none"]),
            f"## Onset coincidence within +-{COINCIDENCE_NS // HOUR_NS} hour",
            _markdown(tables["coincidence"]),
            f"## Onsets within +-{RELEASE_NS // (60 * 10**9)} minutes of scheduled releases "
            "(descriptive, no hypothesis)",
            *(
                [
                    "Calendar: market/releases.csv; 11 of the 12 FOMC times are the standard 14:00 "
                    "Eastern slot, not read off each statement page."
                ]
                if releases
                else []
            ),
            _optional(tables["releases"], releases),
            f"## Cascade candidates and regime onsets within +-{CASCADE_NS // HOUR_NS} hour",
            *([cascade_note] if cascade_note else []),
            _optional(tables["cascades"], cascades),
            "## Null stretches",
            _markdown(tables["nulls"]),
            "",
        ]
    )


# ---------------------------------------------------------------- the stability report


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa of two boolean paths; NaN without bars, or when chance alone agrees fully."""
    if not len(a):
        return math.nan
    chance = float(np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b))
    return (float(np.mean(a == b)) - chance) / (1.0 - chance) if chance < 1.0 else math.nan


def match_onsets(a: np.ndarray, b: np.ndarray, tolerance: int) -> list[tuple[int, int]]:
    """One-to-one pairs of onsets within tolerance: in a's time order, each takes the nearest b
    still free."""
    free = sorted(int(v) for v in b)
    pairs = []
    for onset in sorted(int(v) for v in a):
        near = [v for v in free if abs(v - onset) <= tolerance]
        if near:
            best = min(near, key=lambda v: abs(v - onset))
            free.remove(best)
            pairs.append((onset, best))
    return pairs


def _agreement(
    against: str, h: np.ndarray, onsets: np.ndarray, other_h: np.ndarray, other_onsets: np.ndarray
) -> Row:
    both = (h >= 0) & (other_h >= 0)
    a, b = h[both] == 1, other_h[both] == 1
    onsets, other_onsets = onsets[both[onsets]], other_onsets[both[other_onsets]]
    pairs = match_onsets(onsets, other_onsets, ONSET_TOLERANCE)
    return {
        "against": against,
        "compared_bars": int(both.sum()),
        "agreement": float(np.mean(a == b)) if both.any() else math.nan,
        "kappa": kappa(a, b),
        "onsets": len(onsets),
        "onsets_other": len(other_onsets),
        "matched": len(pairs),
        "shift_median": _median([x - y for x, y in pairs]),
    }


def oracle_path(
    ts_ns: np.ndarray, y: np.ndarray, out: msvol.Labels, params: msvol.Params
) -> np.ndarray:
    """H under the oracle over the bars `out` labels (-1 elsewhere): the final theta's smoothed P
    over all bars, on the z the last refit gave them, through the same hysteresis."""
    theta = nulls.final_theta(out)
    if theta is None:
        return np.full(len(y), -1, np.int8)
    last = out.refits[-1]["bar"]
    _, z = msvol._features(y, msvol._slots(ts_ns), last, len(y), params.baseline_days)
    h = msvol.hysteresis(msvol._estep(z, ~np.isnan(z), theta).g1, params.enter, params.exit)
    h[out.h < 0] = -1
    return h


def relabelled_p(
    ts_ns: np.ndarray, y: np.ndarray, slot: np.ndarray, params: msvol.Params
) -> np.ndarray:
    """P of msvol.label with these slots. With one slot for every bar, z is ln rv less its trailing
    median: the no-deseasonalisation variant.
    shortcut: repeats msvol.label's refit loop because label takes no slots (test_reports holds the
    two bit-equal on the real slots), give label a slots argument when S5 next changes msvol."""
    n = len(y)
    p = np.full(n, np.nan)
    theta: msvol.Theta | None = None
    schedule = msvol.refit_bars(ts_ns, params)
    for j, r in enumerate(schedule):
        end = schedule[j + 1] if j + 1 < len(schedule) else n
        _, z = msvol._features(y, slot, r, end, params.baseline_days)
        train = z[:r]
        if theta is None:
            result = msvol.fit(train, msvol.start(train, params), params.tol, params.max_iter)
        else:
            result = msvol.fit(train, theta, params.tol, params.max_iter_warm)
        if msvol.guard(result, params) is None:
            theta = result.theta
        if theta is not None:
            e0, e1, _ = msvol._emissions(z, theta)
            p[r:end] = msvol._forward(e0, e1, theta)[r:end]
    return p


def plain_path(z: np.ndarray, h: np.ndarray, k: float, d: int) -> np.ndarray:
    """H of the plain rule over the bars h labels (-1 elsewhere): 1 through each run of at least d
    bars with z above k sd(z), the sd over those labelled bars. A diagnostic, read with the whole
    sample's sd: never a label."""
    labelled = h >= 0
    seen = z[labelled & ~np.isnan(z)]
    sd = float(np.std(seen)) if seen.size else math.nan
    out = np.where(labelled, 0, -1).astype(np.int8)
    for a, b in zip(*msvol.runs((z > k * sd) & labelled), strict=True):
        if b - a >= d:
            out[a:b] = 1
    return out


def parkinson_rv(start_ms: int, days: int, klines: Mapping[str, np.ndarray]) -> np.ndarray:
    """The bars' rv under the Parkinson estimator, the named fallback: ln(H/L)^2 / (4 ln 2) summed
    over the bar's minutes, with bars.py's rule for minutes without trades and for missing bars."""
    n = days * bars.DAY_BARS
    # shortcut: repeats bars.bars's grid because bars takes one estimator, give bars an estimator
    # argument when S5 next changes bars.py
    rows, minute, _ = bars._grid(klines["open_time"], start_ms, bars.MINUTE_MS, 5 * n)
    traded = klines["count"][rows] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ranges = np.log(klines["high"][rows] / klines["low"][rows]) ** 2 / (4 * math.log(2))
    bar = minute // 5
    complete = np.bincount(bar, minlength=n) == 5
    some_trades = np.bincount(bar, weights=traded, minlength=n) > 0
    total = np.bincount(bar, weights=np.where(traded, ranges, 0.0), minlength=n)
    return np.where(complete & some_trades, total, np.nan)


def _variant_rows(
    ts: np.ndarray,
    rv: np.ndarray,
    y: np.ndarray,
    parkinson: np.ndarray,
    out: msvol.Labels,
    params: msvol.Params,
) -> list[Row]:
    """Each declared variant against the labels as run, skipping the one they were run with. The
    event-rule variants reread the run's P; the rest relabel."""
    missing = np.isnan(y)
    base = _onsets(out.episodes, "event")

    def reread(name: str, h: np.ndarray, p: msvol.Params = params) -> Row:
        found = _onsets(msvol.episodes(h, missing, p), "event")
        return _agreement(name, out.h, base, h, found)

    def relabel(name: str, labels: msvol.Labels) -> Row:
        return _agreement(name, out.h, base, labels.h, _onsets(labels.episodes, "event"))

    rows = [
        reread(f"d_min={d}", out.h, replace(params, d_min=d)) for d in D_MINS if d != params.d_min
    ]
    rows += [
        reread(f"thresholds={enter}/{leave}", msvol.hysteresis(out.p, enter, leave))
        for enter, leave in THRESHOLDS
        if (enter, leave) != (params.enter, params.exit)
    ]
    rows += [
        relabel(f"baseline_days={d}", msvol.label(ts, rv, replace(params, baseline_days=d)))
        for d in BASELINES
        if d != params.baseline_days
    ]
    rows.append(relabel("parkinson", msvol.label(ts, parkinson, params)))
    flat = relabelled_p(ts, y, np.zeros(len(y), np.int64), params)
    rows.append(reread("no_deseasonalisation", msvol.hysteresis(flat, params.enter, params.exit)))
    plain = f"plain_z>{PLAIN_K:g}sd_for_{params.d_min}_bars"
    rows.append(reread(plain, plain_path(out.z, out.h, PLAIN_K, params.d_min)))
    return rows


def _refit_rows(
    symbol: str, ts: np.ndarray, y: np.ndarray, out: msvol.Labels, params: msvol.Params
) -> list[Row]:
    """The theta path: what each refit fitted, kept or flagged, and the jump its renewed z and
    theta in force make in P, read at the bar before the refit: P refiltered less P as run."""
    slot, rows = msvol._slots(ts), []
    for refit in out.refits:
        r = refit["bar"]
        before = float(out.p[r - 1]) if r else math.nan
        after = math.nan
        if out.refit[r] >= 0 and not math.isnan(before):
            theta = nulls.theta_of(out.refits[out.refit[r]])
            _, z = msvol._features(y, slot, r, r, params.baseline_days)
            e0, e1, _ = msvol._emissions(z, theta)
            after = float(msvol._forward(e0, e1, theta)[-1])
        rows.append(
            {
                "ts_ns": int(ts[r]),
                "symbol": symbol,
                "flagged": refit["flagged"],
                "reason": refit["reason"],
                **{name: refit[name] for name in msvol.Theta._fields},
                "low_duration_bars": _duration(refit["p00"]),
                "high_duration_bars": _duration(refit["p11"]),
                "p_before": before,
                "p_refiltered": after,
                "jump": after - before,
            }
        )
    return rows


def _duration(p: float) -> float:
    """A state's implied duration in bars, from its stay probability."""
    return math.inf if p == 1.0 else 1.0 / (1.0 - p)


def stability(
    series: Series,
    params: msvol.Params,
    parkinson: Mapping[str, np.ndarray],
    *,
    seed: int,
    aggtrades: Mapping[str, np.ndarray] | None = None,
) -> Report:
    """`parkinson` holds each symbol's parkinson_rv; `aggtrades` each symbol's rv from 10-second
    returns on the same bars. A symbol absent from `aggtrades` has no row. The delay simulation
    draws from `seed`."""
    _refuse_confirmatory(series)
    rows: dict[str, list[Row]] = {name: [] for name in STABILITY}
    for k, (symbol, (bar, out)) in enumerate(sorted(series.items())):
        ts, rv = bar["ts_ns"], bar["rv"]
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(rv > 0, np.log(rv), np.nan)
        oracle = oracle_path(ts, y, out, params)
        base = _onsets(out.episodes, "event")
        other = _onsets(msvol.episodes(oracle, np.isnan(y), params), "event")
        found = [_agreement("oracle", out.h, base, oracle, other)]
        found += _variant_rows(ts, rv, y, parkinson[symbol], out, params)
        rows["agreement"] += [{"symbol": symbol, **row} for row in found]
        rows["onset_shifts"] += [
            {
                "ts_ns": int(ts[a]),
                "symbol": symbol,
                "oracle_ts_ns": int(ts[b]),
                "shift_bars": a - b,
            }
            for a, b in match_onsets(base, other, ONSET_TOLERANCE)
        ]
        rows["refits"] += _refit_rows(symbol, ts, y, out, params)
        theta = nulls.final_theta(out)
        if theta is not None:
            rng = np.random.default_rng([seed, k])
            rows["delay"] += [
                {"symbol": symbol, **row}
                for row in nulls.delay_rows(theta, params, nulls.DELAY_BARS, rng)
            ]
        if aggtrades is not None and symbol in aggtrades:
            rows["aggtrades"].append(_cross_check(symbol, rv, aggtrades[symbol]))
    rows["inputs"] = [{"item": "aggtrades", "status": _status(aggtrades is not None)}]
    tables = {name: _table(STABILITY[name], found) for name, found in rows.items()}
    summary = _stability_summary(tables, params, seed, aggtrades is not None)
    return Report("stability", tables, summary)


def _cross_check(symbol: str, rv: np.ndarray, other: np.ndarray) -> Row:
    if len(other) != len(rv):
        raise ValueError(f"{symbol}: {len(other)} aggTrades bars against {len(rv)} bars")
    both = (rv > 0) & (other > 0)
    a, b = np.log(rv[both]), np.log(other[both])
    return {
        "symbol": symbol,
        "bars": int(both.sum()),
        "log_corr": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else math.nan,
        "median_log_ratio": _median(b - a),
    }


def _stability_summary(
    tables: dict[str, pa.Table], params: msvol.Params, seed: int, cross_checked: bool
) -> str:
    hero = []
    delays = tables["delay"].to_pylist()
    refits = tables["refits"].to_pylist()
    for row in tables["agreement"].to_pylist():
        if row["against"] != "oracle":
            continue
        symbol = row["symbol"]
        own = [r for r in refits if r["symbol"] == symbol]
        flagged = sum(r["flagged"] for r in own)
        p90 = [
            d["p90"]
            for d in delays
            if d["symbol"] == symbol and d["population"] == "at_least_d_min"
        ]
        horizon = (
            f"p90 detection delay {p90[0]:.0f} bars, so a smaller horizon is contaminated"
            if p90 and p90[0] is not None
            else "no fitted theta, so no detection delay"
        )
        agreement = "n/a" if row["agreement"] is None else f"{row['agreement']:.1%}"
        kappa_text = "n/a" if row["kappa"] is None else f"{row['kappa']:.2f}"
        hero.append(
            f"- {symbol}: as-of labels agree with the oracle on {agreement} of bars (kappa "
            f"{kappa_text}), {row['matched']} of {row['onsets']} event onsets within "
            f"+-{ONSET_TOLERANCE} bars; {flagged} of {len(own)} refits flagged; {horizon}."
        )
    by_symbol: dict[str, list[int]] = {}
    for row in tables["onset_shifts"].to_pylist():
        by_symbol.setdefault(row["symbol"], []).append(row["shift_bars"])
    shifts = [
        f"- {symbol}: " + ", ".join(f"{s:+d}: {n}" for s, n in sorted(Counter(found).items())) + "."
        for symbol, found in by_symbol.items()
    ]
    labeller = ", ".join(f"{k} = {v:g}" for k, v in asdict(params).items())
    return "\n\n".join(
        [
            "# Stability report",
            "\n".join(hero or ["No labelled series."]),
            f"Labeller parameters: {labeller}. The delay simulation drew from seed = {seed}.",
            "## Agreement: the oracle, the declared variants and the plain threshold rule",
            _markdown(tables["agreement"]),
            "## Onset shifts against the oracle (as-of less oracle, in bars: count)",
            "\n".join(shifts or ["none"]),
            "## The theta path, flagged refits and boundary jumps in P",
            _markdown(tables["refits"].drop_columns(["p_before", "p_refiltered"])),
            "## Detection delay under the fitted theta (bars)",
            _markdown(tables["delay"]),
            "## aggTrades 10-second RV against Garman-Klass",
            _optional(tables["aggtrades"], cross_checked),
            "",
        ]
    )


# ---------------------------------------------------------------- the floor's tables


def floor(found: nulls.Floor, garch: nulls.Garch, settings: Mapping[str, Any]) -> Report:
    """The floor's tables and summary. `garch` and `settings` (its seed, paths, days and scorers)
    are what the floor ran on, kept with its result."""
    values = asdict(garch)
    tables = {
        "garch": _table(FLOOR["garch"], [{"parameter": k, "value": v} for k, v in values.items()]),
        "cells": _table(FLOOR["cells"], found.cells),
        "paths": _table(FLOOR["paths"], found.paths),
        "path_cells": _table(FLOOR["path_cells"], found.path_cells),
    }
    parameters = ", ".join(f"{k} = {v:g}" for k, v in values.items())
    ran_with = ", ".join(f"{k} = {v}" for k, v in settings.items())
    events = sum(p["events"] for p in found.paths)
    unshifted = sum(p["unshifted_events"] for p in found.paths)
    stranded = (
        f" {unshifted} of the {events} events had no eligible bar a day away, so the shift null "
        "leaves them out."
        if unshifted
        else ""
    )
    cells, per_path = (
        sum(not (c["n_pos"] and c["n_neg"]) for c in rows)
        for rows in (found.cells, found.path_cells)
    )
    empty = (
        f" {cells} of the {len(found.cells)} cells and {per_path} of the "
        f"{len(found.path_cells)} AUCs per path have no pre-onset segment or no null slice, so "
        "their AUC is n/a."
        if cells or per_path
        else ""
    )
    summary = "\n\n".join(
        [
            "# Circularity floor",
            f"GARCH(1,1)-t per minute, {parameters}, bridged to 10 seconds, through the exact bars "
            f"and labeller, over "
            f"{len(found.paths)} simulated paths with {events} events. A clustering process with "
            "no critical dynamics gives this AUC by construction: a real AUC counts only above "
            f"it.{stranded}{empty}\nThe floor ran with {ran_with}.",
            "## AUC of pre-onset segments against null slices, and of shifted events",
            _markdown(tables["cells"]),
            "## AUC per path: the floor's own spread",
            _markdown(tables["path_cells"]),
            "## Paths",
            _markdown(tables["paths"]),
            "",
        ]
    )
    return Report("floor", tables, summary)
