# The market warning test's scorer (#30 M5): a frozen market-ews file applied as written to the
# confirmatory span, or the same code run on pilot days as an exploratory dry run that logs no
# trial. docs/market.md holds the rules.

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__, engine
from ..ising import io
from ..market import archive, msvol, nulls
from . import market_segments as ms
from . import prereg, same_input, scoring
from .trend import Pipeline

Row = dict[str, Any]
BTC, ETH = "BTCUSDT", "ETHUSDT"
# The decision rule this scorer carries out, word for word each hypothesis's prediction in
# market-ews-v1; a file whose prediction says anything else is refused (check_file)
PREDICTION = (
    "Pooled and per asset at h = 5 and 6: the AUC's 95 % CI above 0.5, above the GARCH floor's "
    "AUC at that horizon, and above the event-time shift's 97.5th percentile; h = 12 and 24 are "
    "reported, not decisive. A CI is above a value when its lower end is. The CI is the "
    "percentile bootstrap interval of the scoring.bootstrap draws, and every percentile here, the"
    " shift's 97.5th included, uses numpy's linear interpolation (scoring.percentile_interval, "
    "np.nanquantile). The pooled AUC is that of both symbols' events against both symbols' "
    "controls. The event-time shift at each horizon (ews.market_segments.shifted_rows, "
    "market.nulls.shift_onsets): its source onsets are those kept in that horizon's event block, "
    "after its exclusions, not every onset; in each of scoring.shift.draws draws on its own, each"
    " is moved to a bar drawn uniformly from the symbol's eligible bars (Segments.eligible: the "
    "bars where the event rule could have placed an onset, inside the scored span) at least "
    "scoring.shift.min_offset_bars from it, with numpy's default_rng([scoring.seed, 2, h, g]), g "
    "the symbol's group code (see [scoring]); an onset with no such bar is left out of every draw"
    " and counted. A moved onset never leaves the scored span, its bar being eligible; its "
    "segment reaches back before it as an event segment's does. A moved segment holding a minute "
    "with no return, or reaching outside the series, is dropped from that draw, never redrawn. A "
    "draw's AUC is that of its moved segments' scores against the cell's control scores "
    "(ews.scoring.auc, which leaves NaN scores out); a draw with no scorable moved segment has no"
    " AUC, and a cell in which any draw has no AUC is inconclusive "
    "(ews.market_segments.shift_threshold). The pooled cell's draw k joins both symbols' draw k, "
    "each symbol's onsets moved within its own eligible bars, and its threshold is the 97.5th "
    "percentile of the pooled AUC over the draws. The pooled cell's floor is the larger of the "
    "BTCUSDT and ETHUSDT floor AUCs at that horizon. The hypothesis holds when all six cells "
    "(pooled, BTCUSDT and ETHUSDT, at h = 5 and 6) pass; a cell marked inconclusive does not "
    "pass."
)
DECISIVE_HORIZONS = (5, 6)
DECISIVE_GROUPS = ("pooled", BTC, ETH)
CHANCE = 0.5
ALARM_SPAN = (date(2023, 1, 1), prereg.CONFIRMATORY)  # alarms.rule's "2023 control segments"
SURROGATE_ALPHA = 0.05  # [scoring.surrogates]: the share with p <= 0.05
SRC = Path(__file__).resolve().parents[1]  # tipping_point/, whose files a confirmatory run checks
# Held to their blobs at the frozen tag's commit: the segment, shift and same-input code the file
# names, and the modules that compute its numbers. Held to HEAD: this scorer, written after the
# freeze, its command line and the module that runs the checks
HELD = (
    "ews/__init__.py",
    "ews/market_segments.py",
    "ews/same_input.py",
    "ews/scoring.py",
    "ews/trend.py",
    "ews/indicators.py",
    "ews/surrogates.py",
    "market/__init__.py",
    "market/nulls.py",
    "market/msvol.py",
    "market/bars.py",
    "market/archive.py",
    "market/book.py",
    "book/features.py",
)
CLEAN = ("ews/market_score.py", "ews/__main__.py", "ews/prereg.py")
EXPLORATORY = (
    "EXPLORATORY dry run on pilot days: not a trial, nothing logged to ews/trials.jsonl, no "
    "verdict here decides a hypothesis"
)


@dataclass(frozen=True)
class Evaluation:
    tables: dict[str, list[Row]]
    segments: ms.Segments  # the scored series' (seasonality.method)
    series: dict[str, ms.Minutes]  # that series, per symbol
    verdicts: dict[str, str]  # per hypothesis name


def check_file(pre: prereg.Prereg) -> None:
    """Refuse a file this scorer would not apply exactly as written: not a market-ews file, one
    the strict validator refuses, or one whose decision rule is not PREDICTION."""
    doc = pre.doc
    if doc["data"]["domain"] != prereg.MARKET_EWS:
        raise prereg.PreregError(f"{pre.path} is not a {prereg.MARKET_EWS} file")
    found = prereg.problems(doc, strict=True)
    if found:
        raise prereg.PreregError(f"{pre.path}:\n  " + "\n  ".join(found))
    for h in doc["hypotheses"]:
        if h["prediction"] != PREDICTION:
            raise prereg.PreregError(
                f"hypotheses {h['name']}: its prediction is not the decision rule this scorer "
                "carries out (market_score.PREDICTION)"
            )
    if not set(DECISIVE_HORIZONS) <= set(doc["segments"]["horizons"]):
        raise prereg.PreregError(f"segments.horizons: the decisive {list(DECISIVE_HORIZONS)}")
    if list(doc["data"]["symbols"]) != [BTC, ETH]:
        raise prereg.PreregError(f"data.symbols: {[BTC, ETH]}, the prediction's cells")


def labels_file(pre: prereg.Prereg) -> prereg.Prereg:
    """The labels file data.labels names, frozen at that tag with that sha256."""
    named = pre.doc["data"]["labels"]
    name = named["tag"].removeprefix("prereg/")
    path = prereg.repo_root(pre.path) / prereg.PREREG_DIR / f"{name}.toml"
    labels = prereg.load(path)
    ref = prereg.frozen(labels)
    if (ref.tag, labels.sha256) != (named["tag"], named["sha256"]):
        raise prereg.PreregError(
            f"{pre.name} names {named['tag']} (sha256 {named['sha256']}); {path} is {ref.tag} "
            f"(sha256 {labels.sha256})"
        )
    return labels


def scored_from(manifest: Mapping[str, Any], confirmatory: bool) -> date:
    """The first scored day: the confirmatory day, or, on pilot days, the run's first labelled day
    (its start plus the labeller's warm-up)."""
    if confirmatory:
        return prereg.CONFIRMATORY
    params = manifest["params"]
    return date.fromisoformat(params["start"]) + timedelta(days=params["labeller"]["warmup_days"])


def check_span(manifest: Mapping[str, Any], labels: prereg.Prereg, confirmatory: bool) -> None:
    """A confirmatory run labels the labels file's whole span; a dry run ends before 2024-01-01.
    Read from the manifest alone, before any table or kline."""
    params = manifest["params"]
    end = date.fromisoformat(params["end"])
    if confirmatory:
        span = labels.doc["data"]["span"]
        if (date.fromisoformat(params["start"]), end) != (span["start"], span["end"]):
            raise prereg.PreregError(
                f"{manifest['run_id']} spans {params['start']}..{params['end']}, not "
                f"{labels.name}'s {span['start']}..{span['end']}"
            )
    elif end >= prereg.CONFIRMATORY:
        raise ms.SegmentError(
            f"{manifest['run_id']} reaches {end}: a dry run reads no day from "
            f"{prereg.CONFIRMATORY}, the confirmatory span"
        )


def pipelines(doc: Mapping[str, Any], names: Sequence[str] | None = None) -> dict[str, Pipeline]:
    ind = doc["indicators"]
    bandwidth = ind.get("bandwidth") if ind["detrend"] == "gaussian" else None
    return {
        name: Pipeline(name, ind["window"], ind["length"], bandwidth, ind["lag"])
        for name in (names or ind["names"])
    }


def report_groups(doc: Mapping[str, Any]) -> list[str]:
    """reporting.groups with per-asset as each symbol: pooled, BTCUSDT, ETHUSDT, eth-only."""
    out: list[str] = []
    for group in doc["reporting"]["groups"]:
        out += list(doc["data"]["symbols"]) if group == "per-asset" else [group]
    return out


def sides(
    group: str,
    symbols: Sequence[str],
    events: Mapping[str, np.ndarray],
    controls: Mapping[str, np.ndarray],
    alone: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """A group's events and controls, symbol by symbol in data.symbols order; eth-only takes the
    ETH events `alone` marks, against the ETH controls."""
    if group == "eth-only":
        return [events[ETH][alone]], [controls[ETH]]
    chosen = list(symbols) if group == "pooled" else [group]
    return [events[s] for s in chosen], [controls[s] for s in chosen]


def auc_cell(
    events: Sequence[np.ndarray],
    controls: Sequence[np.ndarray],
    doc: Mapping[str, Any],
    indicator: int,
    horizon: int,
    group: int,
) -> Row:
    """The AUC of the joined events against the joined controls and its percentile bootstrap CI,
    drawn by ms.bootstrap_aucs from the cell's own stream."""
    sc = doc["scoring"]
    p, n = np.concatenate(events), np.concatenate(controls)
    draws = ms.bootstrap_aucs(
        events,
        controls,
        seed=sc["seed"],
        indicator=indicator,
        horizon=horizon,
        group=group,
        draws=sc["bootstrap"],
    )
    lo, hi = scoring.percentile_interval(draws, sc["ci_level"])
    return {
        "auc": scoring.auc(p, n),
        "auc_lo": lo,
        "auc_hi": hi,
        "n_pos": int(np.isfinite(p).sum()),
        "n_neg": int(np.isfinite(n).sum()),
    }


def cell_verdict(lo: float, floor: float, shift: float) -> str:
    """pass when the CI's lower end is above chance, the floor and the shift's 97.5th percentile;
    inconclusive when any of the three is missing (an empty side, an empty floor cell, a shift draw
    with no AUC); fail otherwise."""
    if any(math.isnan(v) for v in (lo, floor, shift)):
        return "inconclusive"
    return "pass" if lo > CHANCE and lo > floor and lo > shift else "fail"


def hypothesis_verdict(cells: Sequence[str]) -> str:
    """pass when every decisive cell passes; fail when one fails; inconclusive otherwise."""
    if cells and all(v == "pass" for v in cells):
        return "pass"
    return "fail" if "fail" in cells else "inconclusive"


def deseasoned(
    raw: ms.Minutes, fit: tuple[date, date], methods: Sequence[str]
) -> dict[str, ms.Minutes]:
    """The series raw and deseasonalised by each method, each by its own slot profile over fit."""
    out = {"raw": raw}
    for method in methods:
        out[method] = ms.deseasonalise(raw, ms.slot_profile(raw, fit, method), method)
    return out


def masked(series: ms.Minutes, keep: np.ndarray) -> ms.Minutes:
    return ms.Minutes(series.start_ms, np.where(keep, series.values, np.nan))


def _scores(pipe: Pipeline, x: np.ndarray) -> np.ndarray:
    return pipe(x) if len(x) else np.empty(0)


@dataclass(frozen=True)
class Scored:
    """Each kept segment's score: events by (indicator, symbol, horizon), controls by (indicator,
    symbol), and per horizon which ETH events are eth-only."""

    events: dict[tuple[str, str, int], np.ndarray]
    controls: dict[tuple[str, str], np.ndarray]
    alone: dict[int, np.ndarray]


def score_segments(
    segs: ms.Segments, pipes: Mapping[str, Pipeline], doc: Mapping[str, Any]
) -> Scored:
    symbols, horizons = doc["data"]["symbols"], doc["segments"]["horizons"]
    bars = doc["reporting"]["eth_only_bars"]
    alone = {
        h: ms.eth_only(segs.events[ETH, h].anchor_ns, segs.onsets[BTC], bars) for h in horizons
    }
    events = {
        (name, s, h): _scores(pipe, segs.events[s, h].x)
        for name, pipe in pipes.items()
        for s in symbols
        for h in horizons
    }
    controls = {
        (name, s): _scores(pipe, segs.controls[s].x)
        for name, pipe in pipes.items()
        for s in symbols
    }
    return Scored(events, controls, alone)


def auc_table(scored: Scored, doc: Mapping[str, Any], head: Row | None = None) -> list[Row]:
    """Every (indicator, horizon, reporting group) cell's AUC and CI, keyed by `head`."""
    symbols = doc["data"]["symbols"]
    rows = []
    for i, name in enumerate(doc["indicators"]["names"]):
        neg = {s: scored.controls[name, s] for s in symbols}
        for h in doc["segments"]["horizons"]:
            pos = {s: scored.events[name, s, h] for s in symbols}
            for group in report_groups(doc):
                ev, ct = sides(group, symbols, pos, neg, scored.alone[h])
                code = ms.group_code(symbols, group)
                cell = {"indicator": name, "horizon_bars": h, "group": group}
                rows.append((head or {}) | cell | auc_cell(ev, ct, doc, i, h, code))
    return rows


def floors(doc: Mapping[str, Any], params: msvol.Params) -> dict[str, nulls.Floor]:
    """[scoring.floor]: per symbol, the GARCH(1,1)-t paths of its pilot fit through the labels'
    own labeller, scored by the file's pipelines at its horizons (harness/market_floor.py's
    floor)."""
    f = doc["scoring"]["floor"]
    return {
        symbol: nulls.circularity_floor(
            nulls.Garch(**f["garch"][symbol]),
            pipelines(doc),
            scoring.auc,
            doc["segments"]["horizons"],
            days=f["days"],
            paths=f["paths"],
            seed=f["seed"],
            params=params,
        )
        for symbol in doc["data"]["symbols"]
    }


def floor_auc(found: Mapping[str, nulls.Floor], group: str, name: str, horizon: int) -> float:
    """The floor a cell's CI is held against: its symbol's floor AUC, the larger of the two for
    pooled, ETH's for eth-only; NaN (an empty floor cell, marked) when one it reads has none."""
    chosen = list(found) if group == "pooled" else [ETH if group == "eth-only" else group]
    values = [
        next(
            c["auc"] for c in found[s].cells if (c["scorer"], c["horizon_bars"]) == (name, horizon)
        )
        for s in chosen
    ]
    return math.nan if any(math.isnan(v) for v in values) else max(values)


@dataclass(frozen=True)
class Shift:
    """Per (indicator, symbol, horizon): the moved segments' scores (draws, kept onsets) and which
    onsets of the event block had an eligible bar far enough away."""

    scores: dict[tuple[str, str, int], np.ndarray]
    kept: dict[tuple[str, int], np.ndarray]


def shift_scores(
    segs: ms.Segments,
    series: Mapping[str, ms.Minutes],
    pipes: Mapping[str, Pipeline],
    doc: Mapping[str, Any],
) -> Shift:
    symbols, sc = doc["data"]["symbols"], doc["scoring"]
    scores, kept = {}, {}
    for s in symbols:
        for h in doc["segments"]["horizons"]:
            rows, kept[s, h] = ms.shifted_rows(
                segs,
                series[s],
                symbols,
                s,
                h,
                draws=sc["shift"]["draws"],
                min_offset_bars=sc["shift"]["min_offset_bars"],
                seed=sc["seed"],
            )
            flat = rows.reshape(-1, segs.span)
            for name, pipe in pipes.items():
                scores[name, s, h] = _scores(pipe, flat).reshape(rows.shape[:2])
    return Shift(scores, kept)


def shift_positives(
    shift: Shift, scored: Scored, symbols: Sequence[str], name: str, h: int, group: str
) -> np.ndarray:
    """(draws, onsets) moved scores of a group: pooled joins both symbols' draw k, eth-only keeps
    the moved ETH onsets that are eth-only."""
    if group == "eth-only":
        return shift.scores[name, ETH, h][:, scored.alone[h][shift.kept[ETH, h]]]
    chosen = list(symbols) if group == "pooled" else [group]
    return np.concatenate([shift.scores[name, s, h] for s in chosen], axis=1)


def decisive(group: str, h: int) -> bool:
    return group in DECISIVE_GROUPS and h in DECISIVE_HORIZONS


def counted_cells(
    scored: Scored,
    shift: Shift,
    found: Mapping[str, nulls.Floor],
    doc: Mapping[str, Any],
) -> tuple[list[Row], list[Row]]:
    """The counted cells, each with its floor, shift threshold, comparisons and verdict, and every
    shift draw's AUC."""
    symbols = doc["data"]["symbols"]
    rows, draws = [], []
    for row in auc_table(scored, doc):
        name, h, group = row["indicator"], row["horizon_bars"], row["group"]
        pos = {s: scored.events[name, s, h] for s in symbols}
        neg = {s: scored.controls[name, s] for s in symbols}
        negatives = np.concatenate(sides(group, symbols, pos, neg, scored.alone[h])[1])
        positives = shift_positives(shift, scored, symbols, name, h, group)
        threshold = ms.shift_threshold(positives, negatives)
        for k, moved in enumerate(positives):
            draws.append(
                {"indicator": name, "horizon_bars": h, "group": group, "draw": k}
                | {"auc": scoring.auc(moved, negatives), "n_pos": int(np.isfinite(moved).sum())}
            )
        if group == "eth-only":
            unshifted = int((~shift.kept[ETH, h])[scored.alone[h]].sum())
        else:
            chosen = list(symbols) if group == "pooled" else [group]
            unshifted = sum(int((~shift.kept[s, h]).sum()) for s in chosen)
        floor = floor_auc(found, group, name, h)
        lo = row["auc_lo"]
        rows.append(
            row
            | {
                "decisive": decisive(group, h),
                "floor_auc": floor,
                "shift_q975": threshold,
                "shift_draws": len(positives),
                "shifted_onsets": positives.shape[1],
                "unshifted_onsets": unshifted,
                "above_chance": bool(lo > CHANCE),
                "above_floor": bool(lo > floor),
                "above_shift": bool(lo > threshold),
                "verdict": cell_verdict(lo, floor, threshold),
            }
        )
    return rows, draws


def hypotheses(cells: Sequence[Row], doc: Mapping[str, Any]) -> tuple[list[Row], dict[str, str]]:
    rows, verdicts = [], {}
    for h in doc["hypotheses"]:
        mine = [c for c in cells if c["indicator"] == h["indicator"] and c["decisive"]]
        verdict = hypothesis_verdict([c["verdict"] for c in mine])
        verdicts[h["name"]] = verdict
        rows.append(
            {
                "hypothesis": h["name"],
                "indicator": h["indicator"],
                "verdict": verdict,
                "decisive_cells": len(mine),
                "passed": sum(c["verdict"] == "pass" for c in mine),
                "failed": sum(c["verdict"] == "fail" for c in mine),
                "inconclusive": sum(c["verdict"] == "inconclusive" for c in mine),
            }
        )
    return rows, verdicts


def strata_table(scored: Scored, segs: ms.Segments, doc: Mapping[str, Any]) -> list[Row]:
    """seasonality.strata onset-hour: per UTC hour of the onset, that hour's events against the
    group's controls, descriptively (no CI)."""
    symbols = doc["data"]["symbols"]
    rows = []
    for name in doc["indicators"]["names"]:
        neg = {s: scored.controls[name, s] for s in symbols}
        for h in doc["segments"]["horizons"]:
            hours = {s: (segs.events[s, h].anchor_ns // 3_600_000_000_000) % 24 for s in symbols}
            for hour in range(24):
                pos = {s: scored.events[name, s, h][hours[s] == hour] for s in symbols}
                alone = scored.alone[h][hours[ETH] == hour]
                for group in report_groups(doc):
                    ev, ct = sides(group, symbols, pos, neg, alone)
                    p, n = np.concatenate(ev), np.concatenate(ct)
                    rows.append(
                        {"indicator": name, "horizon_bars": h, "group": group, "hour": hour}
                        | {
                            "auc": scoring.auc(p, n),
                            "n_pos": int(np.isfinite(p).sum()),
                            "n_neg": int(np.isfinite(n).sum()),
                        }
                    )
    return rows


def surrogate_table(
    segs: ms.Segments, scored: Scored, pipes: Mapping[str, Pipeline], doc: Mapping[str, Any]
) -> list[Row]:
    """[scoring.surrogates]: each kept segment's one-sided phase-surrogate p from its own stream
    (ms.surrogate_pvalues), and per indicator, horizon and group the share of event and of control
    segments with p <= 0.05."""
    symbols, sc = doc["data"]["symbols"], doc["scoring"]
    n = sc["surrogates"]["n"]
    rows = []
    for i, (name, pipe) in enumerate(pipes.items()):
        ids = {"seed": sc["seed"], "indicator": i}
        neg = {
            s: ms.surrogate_pvalues(
                segs.controls[s].x,
                pipe,
                n,
                horizon=0,
                symbol=ms.group_code(symbols, s),
                control=True,
                **ids,
            )[1]
            for s in symbols
        }
        for h in doc["segments"]["horizons"]:
            pos = {
                s: ms.surrogate_pvalues(
                    segs.events[s, h].x,
                    pipe,
                    n,
                    horizon=h,
                    symbol=ms.group_code(symbols, s),
                    control=False,
                    **ids,
                )[1]
                for s in symbols
            }
            for group in report_groups(doc):
                ev, ct = sides(group, symbols, pos, neg, scored.alone[h])
                p, q = np.concatenate(ev), np.concatenate(ct)
                rows.append(
                    {"indicator": name, "horizon_bars": h, "group": group}
                    | {
                        "surrogate": sc["surrogates"]["secondary"],
                        "n_surrogates": n,
                        "alpha": SURROGATE_ALPHA,
                        "detected": float(np.nanmean(p <= SURROGATE_ALPHA)) if len(p) else math.nan,
                        "false_positive": float(np.nanmean(q <= SURROGATE_ALPHA))
                        if len(q)
                        else math.nan,
                        "n_pos": len(p),
                        "n_neg": len(q),
                    }
                )
    return rows


def year_controls(
    run: Path, series: Mapping[str, ms.Minutes], span: int
) -> tuple[dict[str, np.ndarray], list[Row]]:
    """alarms.rule's 2023 control segments per symbol: cut from the labels run's null stretches by
    the builder's control rule, those lying wholly in 2023, a segment holding a minute with no
    value excluded and counted."""
    stretches = engine.read(run.parent.parent, run.parent.name, run.name, "nulls").to_pydict()
    first_ms, end_ms = (archive.day_ms(d) for d in ALARM_SPAN)
    out, counts = {}, []
    for symbol, minutes in series.items():
        starts = [
            t // 1_000_000 + k * span * ms.MINUTE_MS
            for t, s, b in zip(
                stretches["ts_ns"], stretches["symbol"], stretches["bars"], strict=True
            )
            if s == symbol
            for k in range(ms.BAR_MINUTES * b // span)
        ]
        first = np.array(starts, np.int64)
        first = first[(first >= first_ms) & (first + span * ms.MINUTE_MS <= end_ms)]
        rows, _, _, outside, invalid = ms._cut(minutes, first, span, np.zeros(0, bool), 0)
        out[symbol] = rows
        counts.append(
            {"symbol": symbol, "kind": "alarm-control", "horizon_bars": None, "cut": len(first)}
            | {"kept": len(rows), "excluded_invalid": invalid, "excluded_outside": outside}
        )
    return out, counts


def alarm_table(
    scored: Scored,
    year: Mapping[str, np.ndarray],
    segs: ms.Segments,
    pipes: Mapping[str, Pipeline],
    doc: Mapping[str, Any],
) -> list[Row]:
    """[alarms]: a segment's score is one alarm when it is above theta. Per indicator, horizon and
    symbol, theta is the lowest whose 2023 control segments alarm at most far_per_day times per
    day of their minutes (scoring.threshold_at_far); under it, the scored controls' false alarms
    per day, and the share of events alarmed h bars (5h minutes) before their onset."""
    far = doc["alarms"]["far_per_day"]
    rows = []
    for name, pipe in pipes.items():
        for s in doc["data"]["symbols"]:
            base = _scores(pipe, year[s])
            base_days = len(base) * segs.span / 1440
            scored_ctl = scored.controls[name, s]
            ctl_days = len(scored_ctl) * segs.span / 1440
            found = (
                scoring.threshold_at_far(base[:, None], far, base_days)
                if len(base)
                else scoring.Threshold(math.nan, False, math.nan)
            )
            theta = found.theta
            for h in doc["segments"]["horizons"]:
                events = scored.events[name, s, h]
                finite = events[np.isfinite(events)]
                hit = finite > theta
                rows.append(
                    {"indicator": name, "horizon_bars": h, "symbol": s, "theta": theta}
                    | {
                        "attainable": found.attainable,
                        "far_max": found.far_max,
                        "far_per_day": far,
                        "far_2023": scoring.false_alarm_rate(base[:, None], theta, base_days)
                        if found.attainable
                        else math.nan,
                        "far_scored": scoring.false_alarm_rate(scored_ctl[:, None], theta, ctl_days)
                        if found.attainable and ctl_days
                        else math.nan,
                        "lead_minutes": ms.BAR_MINUTES * h,
                        "alarmed": int(hit.sum()) if found.attainable else 0,
                        "detected": float(hit.mean())
                        if found.attainable and len(finite)
                        else math.nan,
                        "n_events": len(finite),
                        "n_controls_2023": len(base),
                        "control_days_2023": base_days,
                        "n_controls": len(scored_ctl),
                        "control_days": ctl_days,
                    }
                )
    return rows


def cascade_onsets(
    run: Path, labels: prereg.Prereg, symbols: Sequence[str], confirmatory: bool
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The stress test's onsets per symbol from a cascade run of the labels file's [cascade]: in a
    confirmatory scoring stamped with the frozen file and ending at its span's end, in a dry run
    ending before 2024-01-01, which the manifest says before the table is opened."""
    manifest = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["params"]
    if params.get("cascade") != labels.doc["cascade"]:
        raise ms.SegmentError(f"{run} is not a cascade run of {labels.name}'s [cascade]")
    if set(params["symbols"]) != set(symbols):
        raise ms.SegmentError(f"{run} lists {params['symbols']}, not {list(symbols)}")
    end = date.fromisoformat(params["end"])
    if confirmatory:
        stamp = params.get("prereg") or {}
        ref = prereg.frozen(labels)
        if (stamp.get("sha256"), stamp.get("tag")) != (labels.sha256, ref.tag):
            raise ms.SegmentError(f"{run} was not made under {labels.name} at {ref.tag}")
        if end != labels.doc["data"]["span"]["end"]:
            raise ms.SegmentError(f"{run} ends {end}, not at {labels.name}'s span end")
    elif end >= prereg.CONFIRMATORY:
        raise ms.SegmentError(
            f"{run} reaches {end}: a dry run reads no day from {prereg.CONFIRMATORY}"
        )
    table = engine.read(run.parent.parent, run.parent.name, run.name, "cascades")
    found = table.select(["ts_ns", "symbol"]).to_pydict()
    onsets = {
        s: np.array(
            [t for t, k in zip(found["ts_ns"], found["symbol"], strict=True) if k == s], np.int64
        )
        for s in symbols
    }
    return onsets, manifest


def cascade_scores(
    onsets: Mapping[str, np.ndarray],
    series: Mapping[str, ms.Minutes],
    pipes: Mapping[str, Pipeline],
    doc: Mapping[str, Any],
    span: int,
    first_ms: int,
) -> tuple[Scored, list[Row]]:
    """The cascades cut as event segments are, from their onsets on the first scored day on, a
    segment with a missing minute or outside the series excluded and counted; eth-only: the ETH
    cascades with no BTC cascade onset within eth_only_bars."""
    symbols, horizons = doc["data"]["symbols"], doc["segments"]["horizons"]
    events, counts, kept_ns = {}, [], {}
    for s in symbols:
        mine = onsets[s][onsets[s] // 1_000_000 >= first_ms]
        for h in horizons:
            first = mine // 1_000_000 - (ms.BAR_MINUTES * h + span) * ms.MINUTE_MS
            rows, kept, _, outside, invalid = ms._cut(series[s], first, span, np.zeros(0, bool), 0)
            kept_ns[s, h] = mine[kept]
            for name, pipe in pipes.items():
                events[name, s, h] = _scores(pipe, rows)
            counts.append(
                {"symbol": s, "kind": "cascade", "horizon_bars": h, "cut": len(first)}
                | {"kept": len(rows), "excluded_invalid": invalid, "excluded_outside": outside}
            )
    bars = doc["reporting"]["eth_only_bars"]
    alone = {h: ms.eth_only(kept_ns[ETH, h], onsets[BTC], bars) for h in horizons}
    return Scored(events, {}, alone), counts


def tier_series(
    price: Mapping[str, ms.Minutes],
    book: Mapping[str, Mapping[str, ms.Minutes]],
    fit: tuple[date, date],
    method: str,
) -> dict[str, dict[str, ms.Minutes]]:
    """[same_input]: per series ("price", then each feature) and symbol, the minutes deseasonalised
    by [seasonality]'s method on their own slot profile, then emptied wherever the return or any
    feature has no value, so that every series of the tier keeps the same segments."""
    out: dict[str, dict[str, ms.Minutes]] = {}
    for symbol, features in book.items():
        raw = {"price": price[symbol], **features}
        keep = np.logical_and.reduce([np.isfinite(m.values) for m in raw.values()])
        for name, minutes in raw.items():
            x = ms.deseasonalise(minutes, ms.slot_profile(minutes, fit, method), method)
            out.setdefault(name, {})[symbol] = masked(x, keep)
    return out


def _shares(segs: ms.Segments, doc: Mapping[str, Any]) -> list[Row]:
    return [
        {"symbol": s, "horizon_bars": h, "events": len(segs.events[s, h].high_share)}
        | {
            "mean_high_share": float(np.mean(segs.events[s, h].high_share))
            if len(segs.events[s, h].high_share)
            else math.nan
        }
        for s in doc["data"]["symbols"]
        for h in doc["segments"]["horizons"]
    ]


def evaluate(
    pre: prereg.Prereg,
    labels: prereg.Prereg,
    run: Path,
    prices: Mapping[str, ms.Minutes],
    cascades: Mapping[str, np.ndarray],
    book: Mapping[str, Mapping[str, ms.Minutes]],
    found: Mapping[str, nulls.Floor],
    *,
    confirmatory: bool,
) -> Evaluation:
    """Every table of one scoring from the raw returns, the cascade onsets, the book features'
    minutes and the floor, each number as the file says: the counted cells on the series
    deseasonalised by seasonality.method, with their verdicts, and every descriptive table. With
    confirmatory, the segments are cut through the frozen file (ms.build's scoring); without, on
    pilot days from the run's first labelled day, and ms.build refuses anything from 2024-01-01."""
    doc = pre.doc
    manifest = ms.check_labels(run, labels, pre if confirmatory else None)
    first_day = scored_from(manifest, confirmatory)
    seg, season = doc["segments"], doc["seasonality"]
    pipes = pipelines(doc)
    span = max(p.span for p in pipes.values())
    fit = (season["fit"]["start"], season["fit"]["end"])
    method = season["method"]
    cut: dict[str, Any] = {"subsample": seg["subsample"], "seed": seg["seed"]}
    cut |= {"scoring": pre} if confirmatory else {"score_from": first_day}

    def build(series: Mapping[str, ms.Minutes]) -> ms.Segments:
        return ms.build(run, labels, series, seg["horizons"], span, **cut)

    variants: dict[str, dict[str, ms.Minutes]] = {}
    for symbol, raw in prices.items():
        for variant, x in deseasoned(raw, fit, prereg.DESEASON_METHODS).items():
            variants.setdefault(variant, {})[symbol] = x
    counts: list[Row] = []
    by_variant: dict[str, Scored] = {}
    segments: dict[str, ms.Segments] = {}
    for variant, series in variants.items():
        segments[variant] = build(series)
        by_variant[variant] = score_segments(segments[variant], pipes, doc)
        counts += [{"series": "returns", "variant": variant} | c for c in segments[variant].counts]
    segs, series, scored = segments[method], variants[method], by_variant[method]
    shift = shift_scores(segs, series, pipes, doc)
    cells, draws = counted_cells(scored, shift, found, doc)
    hyp, verdicts = hypotheses(cells, doc)
    descriptive = [
        row
        for variant in ("raw", *prereg.DESEASON_METHODS)
        if variant != method
        for row in auc_table(by_variant[variant], doc, {"variant": variant})
    ]
    year, year_counts = year_controls(run, series, span)
    first_ms = archive.day_ms(first_day)
    cascade, cascade_counts = cascade_scores(cascades, series, pipes, doc, span, first_ms)
    cascade = Scored(cascade.events, scored.controls, cascade.alone)
    tier_rows, tier_counts = [], []
    for name, by_symbol in tier_series(prices, book, fit, method).items():
        tier = build(by_symbol)
        tier_counts += [{"series": name, "variant": method} | c for c in tier.counts]
        tier_rows += auc_table(score_segments(tier, pipes, doc), doc, {"series": name})
    floor_rows = [{"symbol": s} | c for s, f in found.items() for c in f.cells]
    tables = {
        "cells": cells,
        "hypotheses": hyp,
        "shift_draws": draws,
        "seasonality": descriptive,
        "strata": strata_table(scored, segs, doc),
        "cascades": auc_table(cascade, doc),
        "surrogates": surrogate_table(segs, scored, pipes, doc),
        "alarms": alarm_table(scored, year, segs, pipes, doc),
        "same_input": tier_rows,
        "floor": floor_rows,
        "segments": counts + year_counts + cascade_counts,
        "same_input_segments": tier_counts,
        "high_share": _shares(segs, doc),
    }
    return Evaluation(tables, segs, series, verdicts)


def trial_records(stamp: Mapping[str, Any], cells: Sequence[Row]) -> list[Row]:
    return [
        {
            "prereg": dict(stamp),
            "cell": {k: row[k] for k in ("indicator", "horizon_bars", "group")},
            "auc": row["auc"],
            "auc_ci": [row["auc_lo"], row["auc_hi"]],
            "n_pos": row["n_pos"],
            "n_neg": row["n_neg"],
        }
        for row in cells
    ]


def score(
    pre: prereg.Prereg,
    ref: prereg.Frozen,
    labels: prereg.Prereg,
    run: Path,
    prices: Mapping[str, ms.Minutes],
    cascades: Mapping[str, np.ndarray],
    book: Mapping[str, Mapping[str, ms.Minutes]],
    found: Mapping[str, nulls.Floor],
    data_dir: Path,
    inputs: list[Row],
    params: Row,
) -> tuple[Path, Evaluation]:
    """The confirmatory scoring: refused past the trial budget before anything is cut, its folder
    begun (so that a crash leaves a torn result that blocks another scoring), evaluated, then
    written with its counted cells logged as trials (prereg.finish_result)."""
    planned = prereg.market_ews_cells(pre.doc)
    prereg.charge(data_dir, pre, planned)
    run_dir = prereg.begin_result(data_dir, pre, ref, {})
    evaluation = evaluate(pre, labels, run, prices, cascades, book, found, confirmatory=True)
    cells = evaluation.tables["cells"]
    if len(cells) != planned:  # the budget is only as good as this count
        raise prereg.PreregError(f"{pre.name}: {len(cells)} AUC cells evaluated, {planned} counted")
    trials = trial_records(prereg.stamp(pre, ref), cells)
    (run_dir / "report.md").write_text(markdown(evaluation, run_dir.name, None), encoding="utf-8")
    prereg.finish_result(data_dir, pre, run_dir, evaluation.tables, inputs, params, trials)
    return run_dir, evaluation


def dry_run(
    pre: prereg.Prereg,
    labels: prereg.Prereg,
    run: Path,
    prices: Mapping[str, ms.Minutes],
    cascades: Mapping[str, np.ndarray],
    book: Mapping[str, Mapping[str, ms.Minutes]],
    found: Mapping[str, nulls.Floor],
    data_dir: Path,
    out: Path,
    inputs: list[Row],
    params: Row,
) -> tuple[Path, Evaluation]:
    """The same evaluation on pilot days, written to `out` (never inside <data dir>/ews, where
    results count) and marked exploratory, with no trial logged."""
    counted = (Path(data_dir) / "ews").resolve()
    if counted == out.resolve() or counted in out.resolve().parents:
        raise prereg.PreregError(f"{out} is inside {counted}, where scored results live")
    evaluation = evaluate(pre, labels, run, prices, cascades, book, found, confirmatory=False)
    out.mkdir(parents=True)
    stamp = {"id": pre.id, "version": pre.version, "sha256": pre.sha256, "tag": "", "commit": ""}
    manifest = {
        "format": 1,
        "domain": "ews",
        "exploratory": EXPLORATORY,
        "run_id": out.name,
        "producer": {"name": "tipping_point.ews", "version": __version__, **io.code_version()},
        "started_ns": time.time_ns(),
        "key": "cell",
        "tables": [],
        "params": {"prereg": stamp},
    }
    io.write_json(out / "manifest.json", manifest)
    (out / "report.md").write_text(markdown(evaluation, out.name, EXPLORATORY), encoding="utf-8")
    params = {"exploratory": EXPLORATORY, **params}
    prereg.finish_result(data_dir, pre, out, evaluation.tables, inputs, params)
    return out, evaluation


def _run_input(kind: str, run: Path) -> Row:
    complete = hashlib.sha256((Path(run) / "complete.json").read_bytes()).hexdigest()
    return {"domain": kind, "run_id": Path(run).name, "complete_sha256": complete}


def run(
    pre: prereg.Prereg,
    labels_run: Path,
    cascade_run: Path,
    book_root: Path,
    archive_root: Path,
    data_dir: Path,
    *,
    exploratory: bool = False,
    out: Path | None = None,
) -> tuple[Path, Evaluation, bool]:
    """One scoring of a market-ews file from its inputs on disk. Every check that reads no table,
    kline or book feature comes first: the file and its labels file, the runs' spans from their
    manifests, and for a confirmatory scoring both market files frozen, the code (HELD at the
    tag's commit, CLEAN at HEAD, the checkout clean) and the trial budget. Then the archive's
    klines and the book features from a day before the 2023 control segments to the run's end,
    the floor, and score or dry_run. Returns the result's folder, the evaluation and whether it
    was confirmatory."""
    confirmatory = not exploratory
    check_file(pre)
    labels = labels_file(pre)
    manifest = json.loads((Path(labels_run) / "manifest.json").read_text(encoding="utf-8"))
    check_span(manifest, labels, confirmatory)
    doc, symbols = pre.doc, list(pre.doc["data"]["symbols"])
    ref, code = None, None
    if confirmatory:
        ref = prereg.frozen(pre)
        ms.check_scoring(pre)
        code = prereg.code_check(pre, ref, SRC, HELD, CLEAN)
        if code["changed"] or code["uncommitted"]:
            raise prereg.PreregError(
                f"files not at {ref.tag}'s commit: {', '.join(code['changed']) or 'none'}; "
                f"not committed: {', '.join(code['uncommitted']) or 'none'}"
            )
        if io.code_version()["dirty"] is not False:
            raise prereg.PreregError(
                "the checkout has uncommitted changes or untracked files: the result's commit "
                "must name what ran"
            )
        prereg.charge(data_dir, pre, prereg.market_ews_cells(doc))
    ms.check_labels(labels_run, labels, pre if confirmatory else None)
    onsets, _ = cascade_onsets(cascade_run, labels, symbols, confirmatory)
    first = ALARM_SPAN[0] - timedelta(days=1)  # history for the first 2023 segments
    last = date.fromisoformat(manifest["params"]["end"])
    span = archive.collect(symbols, first, last, archive_root)
    start, minutes = archive.day_ms(first), ((last - first).days + 1) * 1440
    prices = {s: ms.close_returns(span.columns[(s, "klines")], start, minutes) for s in symbols}
    aggregation = doc["same_input"]["aggregation"]
    opened = pre if confirmatory else None
    reads = {
        s: same_input.read(book_root, s, first, last, aggregation, scoring=opened) for s in symbols
    }
    found = floors(doc, msvol.Params(**manifest["params"]["labeller"]))
    sources = hashlib.sha256(json.dumps(span.sources, sort_keys=True).encode()).hexdigest()
    inputs = [
        _run_input("labels", labels_run),
        _run_input("cascades", cascade_run),
        {"domain": "archive", "files": len(span.sources), "sources_sha256": sources},
        {
            "domain": "bybit-book",
            "read": {
                s: {"format": r.format, "missing_days": r.missing_days} for s, r in reads.items()
            },
        },
    ]
    params = {"read": [first.isoformat(), last.isoformat()], "code": code}
    book = {s: r.series for s, r in reads.items()}
    given = (labels, labels_run, prices, onsets, book, found, data_dir)
    if ref is not None:
        run_dir, evaluation = score(pre, ref, *given, inputs, params)
    else:
        out = out or Path(data_dir) / "ews-exploratory" / engine.run_id(f"{pre.name}-dry-run")
        run_dir, evaluation = dry_run(pre, *given, out, inputs, params)
    return run_dir, evaluation, confirmatory


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.3f}"


def markdown(evaluation: Evaluation, name: str, exploratory: str | None) -> str:
    t = evaluation.tables
    title = "Market warning test" + (" (EXPLORATORY DRY RUN)" if exploratory else "")
    lines = [f"# {title}", "", f"Result `{name}`.", ""]
    if exploratory:
        lines += [f"**{exploratory}.**", ""]
    lines += ["| hypothesis | verdict | passed | failed | inconclusive |", "|---|---|---|---|---|"]
    lines += [
        f"| {r['hypothesis']} | {r['verdict']} | {r['passed']} | {r['failed']} | "
        f"{r['inconclusive']} |"
        for r in t["hypotheses"]
    ]
    lines += [
        "",
        "| indicator | h | group | decisive | n events | n controls | AUC [95 % CI] | floor | "
        "shift q97.5 | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t["cells"]:
        lines.append(
            f"| {r['indicator']} | {r['horizon_bars']} | {r['group']} | "
            f"{'yes' if r['decisive'] else 'no'} | {r['n_pos']} | {r['n_neg']} | "
            f"{_fmt(r['auc'])} [{_fmt(r['auc_lo'])}, {_fmt(r['auc_hi'])}] | "
            f"{_fmt(r['floor_auc'])} | {_fmt(r['shift_q975'])} | {r['verdict']} |"
        )
    return "\n".join(lines) + "\n"
