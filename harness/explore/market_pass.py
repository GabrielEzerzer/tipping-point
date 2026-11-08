"""The market JEPA's exploratory 2023-Q4 pass (#30 M6, research pass 6.2-6.4 and 4.3 item 6): event
and control segments of market_latents' series, their trend scores, the AUC, and from them a
proposal for each latent pair's direction and for which latent columns to judge. NOT
pre-registered: exploratory, nothing here is a trial, and the owner makes every choice it proposes.

Tables: each model's trained checkpoint and its untrained twin (step 0), each read on its own
symbol's Q4 and the other's: four cells per table (BTC->BTC, BTC->ETH, ETH->ETH, ETH->BTC).

Segments (research pass 6.3, the floor's rule in market.nulls): an onset is the start of a 5-minute
bar; at horizon h bars the event segment is the Pipeline.span = 299 minutes of series ending just
before minute 5 (onset - h), the 240 indicator values of a 60-minute window. The full 240 values
are used at every horizon, so at h = 24 a segment starts before the 72-bar clean window (6.3 leaves
that to market-ews-v1). Controls are the non-overlapping 299-minute spans cut in time order from
each matched-calm null stretch (nulls.null_rows), clipped to the quarter. A segment touching a
minute that any table read on that symbol marks invalid is excluded for every indicator and table
alike, and counted; so is an event too early in the quarter for its segment. Every comparison is
paired on the same segments.

Statistics per (table, read symbol, column, indicator, h): the trend score s, Kendall's tau of the
rolling indicator over the segment (ews.trend.Pipeline(indicator, 60, 240), no detrending); median
s on events and on controls; AUC of events against controls (ews.scoring.auc) with a percentile
95% CI from DRAWS bootstrap draws resampling events and controls independently; the twin margin
(trained AUC minus twin AUC on the same draws). The four pairs are ising-latent-v1's: level and
variance of surprise, variance and ac of pc_1. The screen adds every column: level and variance of
surprise and surprise5, variance and ac of each pc.

Proposals, by rules fixed here before the numbers are read:
  direction  of a pair: the sign of the median event s on the trained tables ("greater" when > 0),
             per cell and h. Reported as one word when every same-asset cell and h agrees, and
             "disagree" otherwise; cross-asset cells are reported beside it. Never chosen for the
             owner when the signs disagree.
  judged     a column is proposed for judging when, in both same-asset cells, some indicator of it
             at some h has a trained AUC whose CI excludes 0.5 and lies further from 0.5 than the
             twin's AUC there. Columns are also ranked by the mean over same-asset cells and h of
             |AUC - 0.5|, trained minus twin. With 17 BTC and 7 ETH events the CIs are wide.
Memorisation (4.3 item 6): mean surprise by month on the model's own training-span days and on the
quarter, trained beside twin.

  .venv/Scripts/python.exe harness/explore market_pass --latents <market_latents' out>
      --labels <labels run> [--out <folder>] [--draws 1000]

Writes <out>/market-pass.json (every row), market-proposal.json and market-report.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tipping_point.ews import scoring
from tipping_point.ews.trend import Pipeline
from tipping_point.market import nulls

from explore import common, market_latents

HORIZONS = (5, 6, 12, 24)  # bars; 5 is the p90 detection-delay floor
WINDOW, LENGTH = 60, 240
SPAN = WINDOW + LENGTH - 1
DRAWS = 1_000
SEED = 30
BAR_NS = 5 * market_latents.MINUTE_NS
PAIRS = tuple(common.LATENT_PAIRS)
SCREEN = tuple(
    (c, i)
    for c in market_latents.COLUMNS
    for i in (("level", "variance") if c.startswith("surprise") else ("variance", "ac"))
)
TABLE = re.compile(r"^(?P<run>.+)-step(?P<step>\d+)$")


# ---------------------------------------------------------------- segments


def segments(
    ts: np.ndarray,
    valid: np.ndarray,
    onsets_ns: np.ndarray,
    stretches: Sequence[Mapping[str, int]],
    horizons: Sequence[int] = HORIZONS,
    span: int = SPAN,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """Minute indices of the event segments per horizon and of the control segments, each (rows,
    span), with what was left out. `stretches` hold ts_ns (start) and bars, as the labels' nulls
    table does; `ts` is the series' regular minute grid, starting on a bar."""
    M = len(ts)
    if M and (np.any(np.diff(ts) != market_latents.MINUTE_NS) or ts[0] % BAR_NS):
        raise ValueError("the series must be a regular minute grid starting on a bar")
    minute = np.arange(M)
    onsets = (np.asarray(onsets_ns, np.int64) - ts[0]) // BAR_NS
    inside = (onsets >= 0) & (onsets * 5 < M)
    onsets = onsets[inside]
    events: dict[int, np.ndarray] = {}
    counts: dict[str, Any] = {"events": len(onsets), "events_by_h": {}}
    for h in horizons:
        rows = nulls.pre_onset_rows(minute, onsets, h, span)
        ok = valid[rows].all(1) if len(rows) else np.zeros(0, bool)
        events[h] = rows[ok]
        counts["events_by_h"][h] = {
            "kept": int(ok.sum()),
            "too_early": int(len(onsets) - len(rows)),
            "invalid": int((~ok).sum()),
        }
    bars_in = M // 5
    clipped = []
    for s in stretches:
        first = (int(s["ts_ns"]) - int(ts[0])) // BAR_NS
        start, end = max(first, 0), min(first + int(s["bars"]), bars_in)
        if end > start:
            clipped.append({"start": start, "bars": end - start})
    rows = nulls.null_rows(minute, clipped, span)
    rows = rows.reshape(-1, span)
    ok = valid[rows].all(1) if len(rows) else np.zeros(0, bool)
    counts["controls"] = {"kept": int(ok.sum()), "invalid": int((~ok).sum())}
    counts["null_stretches"] = len(clipped)
    return events, rows[ok], counts


# ---------------------------------------------------------------- statistics


def scores(series: np.ndarray, rows: np.ndarray, indicator: str) -> np.ndarray:
    if not len(rows):
        return np.empty(0)
    return Pipeline(indicator, WINDOW, LENGTH)(series[rows])


def _draws(n_pos: int, n_neg: int, draws: int, rng: np.random.Generator) -> list[tuple]:
    return [(rng.integers(0, n_pos, n_pos), rng.integers(0, n_neg, n_neg)) for _ in range(draws)]


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not (np.isfinite(pos).any() and np.isfinite(neg).any()):
        return math.nan
    return scoring.auc(pos, neg)


def _median(x: np.ndarray) -> float:
    return float(np.nanmedian(x)) if np.isfinite(x).any() else math.nan


def cell(
    pos: Mapping[str, np.ndarray],
    neg: Mapping[str, np.ndarray],
    draws: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    """Per table: median s on events and controls, the AUC with its CI; with tables "trained" and
    "twin", the margin trained - twin on the same draws."""
    n_pos = len(next(iter(pos.values())))
    n_neg = len(next(iter(neg.values())))
    picks = _draws(n_pos, n_neg, draws, rng) if n_pos and n_neg else []
    out: dict[str, dict[str, Any]] = {}
    boot: dict[str, np.ndarray] = {}
    for table in pos:
        p, n = pos[table], neg[table]
        boot[table] = np.array([_auc(p[i], n[j]) for i, j in picks])
        lo, hi = scoring.percentile_interval(boot[table]) if picks else (math.nan, math.nan)
        out[table] = {
            "median_s_events": _median(p),
            "median_s_controls": _median(n),
            "auc": _auc(p, n),
            "auc_ci": [lo, hi],
            "n_events": int(np.isfinite(p).sum()),
            "n_controls": int(np.isfinite(n).sum()),
        }
    if {"trained", "twin"} <= set(pos):
        diff = boot["trained"] - boot["twin"]
        lo, hi = scoring.percentile_interval(diff) if picks else (math.nan, math.nan)
        value = out["trained"]["auc"] - out["twin"]["auc"]
        out["margin"] = {"auc": value, "auc_ci": [lo, hi]}
    return out


def score_symbol(
    tables: Mapping[str, Mapping[str, np.ndarray]],
    ts: np.ndarray,
    valid: np.ndarray,
    onsets_ns: np.ndarray,
    stretches: Sequence[Mapping[str, int]],
    pairs: Sequence[tuple[str, str]] = SCREEN,
    horizons: Sequence[int] = HORIZONS,
    draws: int = DRAWS,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows for one read symbol of tables {name: columns} that share `valid` (the AND of theirs):
    one per (pair, h), each holding every table's statistics."""
    events, controls, counts = segments(ts, valid, onsets_ns, stretches, horizons)
    rows = []
    for k, (column, indicator) in enumerate(pairs):
        neg = {t: scores(c[column], controls, indicator) for t, c in tables.items()}
        for h in horizons:
            pos = {t: scores(c[column], events[h], indicator) for t, c in tables.items()}
            rng = np.random.default_rng([seed, k, h])
            rows.append(
                {"column": column, "indicator": indicator, "h": h, **cell(pos, neg, draws, rng)}
            )
    return rows, counts


# ---------------------------------------------------------------- inputs


def _labels(run: Path, table: str) -> list[dict[str, Any]]:
    parts = sorted((Path(run) / table).glob("*.parquet"))
    if not parts:
        raise SystemExit(f"{run}: no {table} table")
    return pa.concat_tables(pq.read_table(p) for p in parts).to_pylist()


def labels_run(run: Path) -> dict[str, Any]:
    """The labels run's record, if this pass may read it: a complete labels run that ends before
    2024-01-01 and was made without a pre-registration (a --prereg run is the confirmatory one)."""
    run = Path(run)
    if not (run / "complete.json").is_file() or not (run / "manifest.json").is_file():
        raise SystemExit(f"{run}: not a complete labels run")
    manifest = json.loads((run / "manifest.json").read_text("utf-8"))
    params = manifest.get("params", {})
    if manifest.get("domain") != "labels" or "end" not in params:
        raise SystemExit(f"{run}: not a labels run")
    if params["end"] >= market_latents.days.CONFIRMATORY.isoformat():
        raise SystemExit(f"{run} runs to {params['end']}: this pass never reads from 2024-01-01")
    if "prereg" in params or "prereg" in manifest:
        raise SystemExit(f"{run} was made under a pre-registration: this pass never reads it")
    complete = json.loads((run / "complete.json").read_text("utf-8"))
    return {
        "run_id": manifest["run_id"],
        "params": params,
        "producer": manifest.get("producer"),
        "files": complete["files"],
    }


def events_and_nulls(run: Path, symbol: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Event onsets (ns) and null stretches of `symbol`; nothing from 2024 is read."""
    labels_run(run)
    end = np.datetime64(market_latents.days.CONFIRMATORY.isoformat(), "ns").astype(np.int64)
    episodes = [
        e["ts_ns"]
        for e in _labels(run, "episodes")
        if e["symbol"] == symbol and e["kind"] == "event" and e["ts_ns"] < end
    ]
    stretches = [s for s in _labels(run, "nulls") if s["symbol"] == symbol and s["ts_ns"] < end]
    return np.array(sorted(episodes), np.int64), stretches


def find_tables(latents: Path) -> dict[str, dict[str, Any]]:
    """{run id: {"trained": table, "twin": table, "symbol": model symbol}} from market_latents'
    folders: the twin is step 0, the trained table the run's other step."""
    runs: dict[str, dict[str, Any]] = {}
    for folder in sorted(p for p in Path(latents).iterdir() if p.is_dir()):
        m = TABLE.match(folder.name)
        if not m or not any(folder.glob("*.json")):
            continue
        manifest = json.loads(next(folder.glob("*.json")).read_text("utf-8"))
        run = runs.setdefault(m["run"], {"symbol": manifest["model_symbol"]})
        key = "twin" if int(m["step"]) == 0 else "trained"
        if key in run:
            raise SystemExit(f"{latents}: two {key} tables of {m['run']}")
        run[key] = folder.name
        run[f"{key}_files"] = {
            j.stem: {
                "checkpoint_sha256": record["checkpoint_sha256"],
                "parquet_sha256": record["parquet_sha256"],
                "scaling_sha256": record["scaling"]["sha256"],
            }
            for j in sorted(folder.glob("*.json"))
            for record in [json.loads(j.read_text("utf-8"))]
        }
    for name, run in runs.items():
        if not {"trained", "twin"} <= set(run):
            raise SystemExit(f"{latents}: {name} needs a trained table and its step-0 twin")
    return runs


def load_series(latents: Path, table: str, symbol: str, span: str) -> dict[str, np.ndarray] | None:
    """A table's series, refused when its bytes are not the ones its manifest names."""
    path = Path(latents) / table / f"{symbol}-{span}.parquet"
    if not path.is_file():
        return None
    raw = path.read_bytes()
    manifest = json.loads(path.with_suffix(".json").read_text("utf-8"))
    if hashlib.sha256(raw).hexdigest() != manifest["parquet_sha256"]:
        raise SystemExit(f"{path}: not the bytes its manifest names")
    t = pq.read_table(pa.BufferReader(raw))
    return {name: t[name].to_numpy() for name in t.column_names}


# ---------------------------------------------------------------- the pass


def memorisation(latents: Path, runs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean surprise by month on each model's own symbol, training span and quarter, trained
    beside twin, with the part of the split each month served."""
    rows = []
    for run_id, run in runs.items():
        symbol = run["symbol"]
        folder = Path(latents) / run["trained"]
        trained_on = json.loads((folder / f"{symbol}-exploratory.json").read_text("utf-8"))[
            "model_trained_on"
        ]
        by: dict[str, dict[str, float]] = {}
        for who in ("trained", "twin"):
            for span in ("training", "exploratory"):
                s = load_series(latents, run[who], symbol, span)
                if s is None:
                    continue
                month = s["ts_ns"].astype("datetime64[ns]").astype("datetime64[M]").astype(str)
                ok = s["valid"]
                for m in np.unique(month):
                    sel = ok & (month == m)
                    by.setdefault(m, {})[who] = (
                        float(s["surprise"][sel].mean()) if sel.any() else math.nan
                    )
                    by[m].setdefault("minutes", int(sel.sum()))
        for m in sorted(by):
            if m >= "2023-10":
                part = "exploratory"
            elif m <= trained_on["train"][1][:7]:
                part = "training"
            else:
                part = "validation"
            rows.append({"run": run_id, "symbol": symbol, "month": m, "part": part, **by[m]})
    return rows


def run_pass(latents: Path, labels: Path, draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    runs = find_tables(latents)
    symbols = sorted({r["symbol"] for r in runs.values()})
    result: dict[str, Any] = {"runs": runs, "cells": [], "counts": {}, "rows": []}
    for symbol in symbols:
        onsets, stretches = events_and_nulls(labels, symbol)
        tables: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for run_id, run in runs.items():
            for who in ("trained", "twin"):
                s = load_series(latents, run[who], symbol, "exploratory")
                if s is not None:
                    tables[run_id, who] = s
        if not tables:
            continue
        first = next(iter(tables.values()))
        for s in tables.values():
            if not np.array_equal(s["ts_ns"], first["ts_ns"]):
                raise SystemExit(f"{symbol}: the tables cover different minutes")
        valid = np.logical_and.reduce([s["valid"] for s in tables.values()])
        ts = first["ts_ns"]
        q = (onsets >= ts[0]) & (onsets < ts[-1] + market_latents.MINUTE_NS)
        result["counts"][symbol] = {"events_in_series": int(q.sum())}
        for run_id in runs:
            mine = {
                who: tables[run_id, who] for who in ("trained", "twin") if (run_id, who) in tables
            }
            if len(mine) < 2:
                continue
            rows, counts = score_symbol(mine, ts, valid, onsets, stretches, draws=draws, seed=seed)
            kind = "same" if runs[run_id]["symbol"] == symbol else "cross"
            name = f"{runs[run_id]['symbol'][:3]}->{symbol[:3]}"
            result["cells"].append({"cell": name, "run": run_id, "read": symbol, "kind": kind})
            result["counts"][symbol].update(counts)
            result["counts"][symbol]["invalid_minutes"] = int((~valid).sum())
            midnight = ts % (1440 * market_latents.MINUTE_NS) == 0
            result["counts"][symbol]["invalid_at_midnight"] = int((~valid & midnight).sum())
            result["counts"][symbol]["minutes"] = len(valid)
            for row in rows:
                result["rows"].append({"cell": name, "kind": kind, **row})
    result["directions"] = directions(result["rows"])
    result["judged"] = judged(result["rows"])
    result["memorisation"] = memorisation(latents, runs)
    return result


def _sign(x: float) -> str:
    if not np.isfinite(x) or x == 0:
        return "none"
    return "greater" if x > 0 else "less"


def _one(signs: Sequence[str]) -> str:
    found = set(signs)
    return found.pop() if len(found) == 1 else ("none" if not found else "disagree")


KINDS = ("same", "cross")


def directions(rows: Sequence[Mapping[str, Any]], pairs: Sequence = PAIRS) -> dict[str, Any]:
    """Per pair: the trained median event s's sign per cell and h, and whether they agree."""
    out: dict[str, Any] = {}
    for column, indicator in pairs:
        mine = [r for r in rows if (r["column"], r["indicator"]) == (column, indicator)]
        by_cell: dict[str, dict[int, str]] = {}
        for r in mine:
            by_cell.setdefault(r["cell"], {})[r["h"]] = _sign(r["trained"]["median_s_events"])
        signs = {k: [by_cell[r["cell"]][r["h"]] for r in mine if r["kind"] == k] for k in KINDS}
        out[f"{column} {indicator}"] = {
            "same_asset": _one([x for x in signs["same"] if x != "none"]),
            "cross_asset": _one([x for x in signs["cross"] if x != "none"]),
            "by_cell": by_cell,
            "no_sign": sorted(
                f"{r['cell']} h{r['h']}" for r in mine if by_cell[r["cell"]][r["h"]] == "none"
            ),
        }
    return out


def _passes(row: Mapping[str, Any]) -> bool:
    lo, hi = row["trained"]["auc_ci"]
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.5 <= hi:
        return False
    return abs(row["trained"]["auc"] - 0.5) > abs(row["twin"]["auc"] - 0.5)


def _mean(x: Sequence[float]) -> float:
    a = np.asarray(x, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else math.nan


def judged(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The judged-column rule (module notes) and the ranking beside it."""
    same = [r for r in rows if r["kind"] == "same"]
    cells = sorted({r["cell"] for r in same})
    ranking = []
    for column in market_latents.COLUMNS:
        mine = [r for r in same if r["column"] == column]
        passing = {
            c: sorted(
                {f"{r['indicator']} h{r['h']}" for r in mine if r["cell"] == c and _passes(r)}
            )
            for c in cells
        }
        dev = [abs(r["trained"]["auc"] - 0.5) for r in mine]
        twin = [abs(r["twin"]["auc"] - 0.5) for r in mine]
        trained_dev, twin_dev = _mean(dev), _mean(twin)
        ranking.append(
            {
                "column": column,
                "proposed": len(cells) == 2 and all(passing[c] for c in cells),
                "passing": passing,
                "trained_mean_dev": trained_dev,
                "twin_mean_dev": twin_dev,
                "margin": trained_dev - twin_dev,
            }
        )
    ranking.sort(key=lambda r: -r["margin"] if np.isfinite(r["margin"]) else math.inf)
    return {
        "rule": "proposed when, in both same-asset cells, some indicator of the column at some h "
        "has a trained AUC whose 95% CI excludes 0.5 and lies further from 0.5 than the twin's",
        "proposed": [r["column"] for r in ranking if r["proposed"]],
        "ranking": ranking,
    }


# ---------------------------------------------------------------- the report


def _f(x: float, d: int = 3, sign: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _auc_text(entry: Mapping[str, Any]) -> str:
    lo, hi = entry["auc_ci"]
    return f"{_f(entry['auc'])} [{_f(lo)}, {_f(hi)}]"


def render(result: Mapping[str, Any], draws: int) -> str:
    lines = [
        "# Exploratory market latent pass, 2023-Q4",
        "",
        "Exploratory: nothing here is a pre-registered trial or a result. Every choice it proposes "
        "is the owner's.",
        "",
        "## Tables",
        "",
    ]
    for run in result["runs"].values():
        lines.append(f"- {run['symbol']}: trained `{run['trained']}`, twin `{run['twin']}`")
    lines += ["", "## Segments", ""]
    lines += [
        f"Segments of {SPAN} minutes (Pipeline(indicator, {WINDOW}, {LENGTH})), ending 5h minutes "
        "before the onset bar's start; controls cut from the matched-calm null stretches. A "
        "segment touching an invalid minute of any table on that symbol is left out of all. The "
        "first 95 minutes of the quarter have no predictor context. The last column counts invalid "
        "minutes at 00:00 UTC: a day whose first second is a feature gap leaves no segment that "
        "spans its midnight.",
        "",
        "| read symbol | events in Q4 | h | events kept | too early | invalid | controls kept | "
        "controls invalid | invalid minutes | of them at 00:00 UTC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for symbol, c in result["counts"].items():
        if "events_by_h" not in c:
            continue
        for h, e in c["events_by_h"].items():
            lines.append(
                f"| {symbol} | {c['events_in_series']} | {h} | {e['kept']} | {e['too_early']} "
                f"| {e['invalid']} | {c['controls']['kept']} | {c['controls']['invalid']} "
                f"| {c['invalid_minutes']} of {c['minutes']} | {c['invalid_at_midnight']} |"
            )
    lines += ["", "## Proposed directions", ""]
    lines += [
        "Rule: the sign of the median event s on the trained table, per cell and h; one word only "
        "when every same-asset cell and h agrees.",
        "",
        "| pair | same-asset | cross-asset | by cell: h -> sign |",
        "|---|---|---|---|",
    ]
    for pair, d in result["directions"].items():
        detail = "; ".join(
            f"{c}: " + ", ".join(f"{h} {s}" for h, s in by.items())
            for c, by in d["by_cell"].items()
        )
        lines.append(f"| {pair} | **{d['same_asset']}** | {d['cross_asset']} | {detail} |")
    disagree = [p for p, d in result["directions"].items() if d["same_asset"] == "disagree"]
    if disagree:
        lines += ["", f"The signs disagree for {', '.join(disagree)}: the owner decides."]
    for pair, d in result["directions"].items():
        if d["no_sign"]:
            lines.append(f"- {pair}: no sign (no kept event) at {', '.join(d['no_sign'])}")
    lines += ["", "## The four pairs, trained beside twin", ""]
    lines += [
        f"CI: percentile, {draws} bootstrap draws resampling events and controls; the margin is "
        "trained minus twin on the same draws. s: Kendall's tau of the rolling indicator over the "
        "segment.",
        "",
        "| pair | cell | h | events | controls | median s ev, trained | median s ctl, trained "
        "| AUC trained | median s ev, twin | median s ctl, twin | AUC twin | margin |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for column, indicator in PAIRS:
        for r in result["rows"]:
            if (r["column"], r["indicator"]) != (column, indicator):
                continue
            a, b = r["trained"], r["twin"]
            lines.append(
                f"| {column} {indicator} | {r['cell']} | {r['h']} | {a['n_events']} "
                f"| {a['n_controls']} | {_f(a['median_s_events'], 2, True)} "
                f"| {_f(a['median_s_controls'], 2, True)} | {_auc_text(a)} "
                f"| {_f(b['median_s_events'], 2, True)} | {_f(b['median_s_controls'], 2, True)} "
                f"| {_auc_text(b)} | {_auc_text(r['margin'])} |"
            )
    j = result["judged"]
    lines += ["", "## Which columns to judge", ""]
    lines += [f"Rule: {j['rule']}.", ""]
    shown = ", ".join(j["proposed"]) if j["proposed"] else "none passes; the owner decides"
    lines += [f"Proposed: **{shown}**.", ""]
    lines += [
        "| column | mean \\|AUC - 0.5\\| trained | twin | margin | passing (same-asset cells) |",
        "|---|---|---|---|---|",
    ]
    for r in j["ranking"]:
        passing = "; ".join(f"{c}: {', '.join(v) or '-'}" for c, v in r["passing"].items())
        lines.append(
            f"| {r['column']} | {_f(r['trained_mean_dev'])} | {_f(r['twin_mean_dev'])} "
            f"| {_f(r['margin'], 3, True)} | {passing} |"
        )
    lines += ["", "## The screen: every column, same-asset cells", ""]
    lines += ["| column | indicator | cell | h | AUC trained | AUC twin | margin |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        if r["kind"] != "same":
            continue
        lines.append(
            f"| {r['column']} | {r['indicator']} | {r['cell']} | {r['h']} "
            f"| {_auc_text(r['trained'])} | {_auc_text(r['twin'])} | {_auc_text(r['margin'])} |"
        )
    lines += ["", "## Memorisation: mean surprise by month (research pass 4.3 item 6)", ""]
    lines += [
        "A drop on the training months against the quarter that the twin does not share points to "
        "memorisation. Validation days were held out of the gradient but read by the monitor.",
        "",
        "| model | month | part | minutes | trained | twin | trained / twin |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["memorisation"]:
        a, b = r.get("trained", math.nan), r.get("twin", math.nan)
        lines.append(
            f"| {r['symbol']} | {r['month']} | {r['part']} | {r['minutes']} | {_f(a, 4)} "
            f"| {_f(b, 4)} | {_f(a / b if b else math.nan, 3)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--latents", type=Path, required=True, help="market_latents' --out")
    p.add_argument("--labels", type=Path, required=True, help="the labels run")
    p.add_argument("--out", type=Path, help="default: <latents>/pass")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--seed", type=int, default=SEED, help="the bootstrap's")
    args = p.parse_args(argv)
    out = args.out or args.latents / "pass"
    out.mkdir(parents=True, exist_ok=True)
    common.be_nice()
    clock = common.Clock()
    result = run_pass(args.latents, args.labels, args.draws, args.seed)
    result["provenance"] = {
        "exploratory": True,
        "pre_registered": False,
        "code_commit": common._git("rev-parse", "HEAD"),
        "code_dirty": bool(common._git("status", "--porcelain", "--untracked-files=no")),
        "labels_run": labels_run(args.labels),
        "draws": args.draws,
        "seed": args.seed,
        "horizons": list(HORIZONS),
        "pipeline": {"window": WINDOW, "length": LENGTH, "detrend": None},
    }
    common.save(out / "market-pass.json", result)
    proposal = {
        "exploratory": True,
        "directions": {k: v["same_asset"] for k, v in result["directions"].items()},
        "directions_detail": result["directions"],
        "judged": {k: result["judged"][k] for k in ("rule", "proposed")},
    }
    common.save(out / "market-proposal.json", proposal)
    target = out / "market-report.md"
    target.write_text(render(result, args.draws), "utf-8")
    print(f"wrote {target}")
    clock("done")
