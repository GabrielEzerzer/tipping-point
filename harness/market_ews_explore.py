"""EXPLORATORY: the classic market warning indicators on the 2023 pilot year (#30 M5). It is not a
trial and cannot log one: it reads no pre-registration of the warning test, writes nothing to
ews/trials.jsonl and freezes nothing. It shows what the segments of market-ews give before the
file is frozen, so whatever it prints must be disclosed in that file's [disclosure].

Binance 1-minute close-to-close log returns (ews.market_segments.close_returns) from the archive,
raw and deseasonalised by the slot profile fitted on 2023-01-18..2023-09-30 (the JEPA's training
span, call 9) by each of the two methods market-ews offers, "scale" and "mean", are cut into
the event and control segments of the labels run --run, which must be a run of the frozen
market-labels file at its parameters (market_segments.check_labels), and scored by
Pipeline("variance", 60, 240) and Pipeline("ac", 60, 240) at horizons 5, 6, 12 and 24 bars.
Onsets and control segments from 2023-01-01 only; the day before is read as history for
January's segments. Per symbol, pooled, and for the ETH events with no BTC event onset within an
hour (eth-only, against the ETH controls): the AUC of events against controls, with a percentile
bootstrap CI (events and controls resampled with replacement within each symbol, a generator per
cell seeded from --seed and the cell's name). The run must end before 2024-01-01 and no kline
from then on is read; the builder refuses otherwise. Writes report.json and report.md to --out.

  .venv/Scripts/python.exe harness/market_ews_explore.py --run <labels run folder> \\
      [--archive <archive folder>] [--out harness/out/market-ews-explore]
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from tipping_point.ews import market_segments as ms
from tipping_point.ews import prereg, scoring
from tipping_point.ews.trend import Pipeline
from tipping_point.market import archive
from tipping_point.market.__main__ import ARCHIVE

READ = (date(2022, 12, 31), date(2023, 12, 31))  # a day early: a 24-bar segment starts 419 min out
SCORED = date(2023, 1, 1)  # READ's first day is history only: no onset or control from it
FIT = (date(2023, 1, 18), date(2023, 9, 30))
HORIZONS = (5, 6, 12, 24)
SCORERS = {"variance": Pipeline("variance", 60, 240), "ac": Pipeline("ac", 60, 240)}
LABELS = Path("paper") / "prereg" / "market-labels-v1.toml"
OUT = Path("harness") / "out" / "market-ews-explore"
BTC, ETH = "BTCUSDT", "ETHUSDT"


def series(symbols: Sequence[str], root: Path) -> dict[str, dict[str, ms.Minutes]]:
    """Per variant ("raw", then each deseasonalisation method), each symbol's returns over READ."""
    start, minutes = archive.day_ms(READ[0]), ((READ[1] - READ[0]).days + 1) * 1440
    span = archive.collect(symbols, *READ, root)
    out: dict[str, dict[str, ms.Minutes]] = {v: {} for v in ("raw", *prereg.DESEASON_METHODS)}
    for symbol in symbols:
        raw = ms.close_returns(span.columns[(symbol, "klines")], start, minutes)
        out["raw"][symbol] = raw
        for method in prereg.DESEASON_METHODS:
            profile = ms.slot_profile(raw, FIT, method)
            out[method][symbol] = ms.deseasonalise(raw, profile, method)
    return out


def cell(
    pos: dict[str, np.ndarray], neg: dict[str, np.ndarray], draws: int, rng: np.random.Generator
) -> dict[str, Any]:
    """AUC of the joined positives against the joined negatives, and its bootstrap CI, each side
    resampled within each symbol."""
    p, n = np.concatenate(list(pos.values())), np.concatenate(list(neg.values()))
    boot = np.empty(draws)
    for d in range(draws):
        bp = [v[rng.integers(0, len(v), len(v))] for v in pos.values()]
        bn = [v[rng.integers(0, len(v), len(v))] for v in neg.values()]
        boot[d] = scoring.auc(np.concatenate(bp), np.concatenate(bn))
    lo, hi = scoring.percentile_interval(boot)
    n_pos, n_neg = int(np.isfinite(p).sum()), int(np.isfinite(n).sum())  # what auc reads
    return {"n_pos": n_pos, "n_neg": n_neg, "auc": scoring.auc(p, n), "lo": lo, "hi": hi}


def explore(run: Path, labels: Path, root: Path, draws: int, seed: int) -> dict[str, Any]:
    pre = prereg.load(labels)
    manifest = ms.check_labels(run, pre)
    symbols = list(manifest["params"]["symbols"])
    found = series(symbols, root)
    span = max(s.span for s in SCORERS.values())
    rows, counts, shares = [], None, {}
    for variant, by_symbol in found.items():
        segs = ms.build(run, pre, by_symbol, HORIZONS, span, score_from=SCORED)
        counts = counts or segs.counts
        assert segs.counts == counts  # the same segments for every variant
        alone = {  # eth-only: which kept ETH events have no BTC onset within the window
            h: ms.eth_only(segs.events[ETH, h].anchor_ns, segs.onsets[BTC], prereg.ETH_ONLY_BARS)
            for h in HORIZONS
        }
        for name, scorer in SCORERS.items():
            neg = {s: scorer(segs.controls[s].x) for s in symbols}
            for h in HORIZONS:
                pos = {s: scorer(segs.events[s, h].x) for s in symbols}
                groups = {s: [s] for s in symbols} | {"pooled": symbols, "eth-only": [ETH]}
                for group, picked in groups.items():
                    key = {"variant": variant, "indicator": name, "horizon_bars": h, "group": group}
                    rng = np.random.default_rng([seed, zlib.crc32(json.dumps(key).encode())])
                    p = {s: pos[s] for s in picked}
                    if group == "eth-only":
                        p = {ETH: pos[ETH][alone[h]]}
                    rows.append(key | cell(p, {s: neg[s] for s in picked}, draws, rng))
        if variant == "raw":
            shares = {
                f"{s} h={h}": float(np.mean(segs.events[s, h].high_share))
                for s in symbols
                for h in HORIZONS
            }
    return {
        "exploratory": "not a trial: nothing is logged to trials.jsonl and nothing is frozen",
        "run": run.name,
        "labels": {"file": labels.as_posix(), "sha256": pre.sha256},
        "read": [d.isoformat() for d in READ],
        "fit": [d.isoformat() for d in FIT],
        "scorers": {k: repr(v) for k, v in SCORERS.items()},
        "bootstrap": {"draws": draws, "seed": seed},
        "counts": counts,
        "mean_high_share": shares,
        "cells": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Classic market warning indicators, 2023 pilot year (exploratory)",
        "",
        f"Not a trial: nothing is logged and nothing is frozen. Labels run `{report['run']}`, "
        f"returns {report['read'][0]}..{report['read'][1]}, slot profile fitted on "
        f"{report['fit'][0]}..{report['fit'][1]}, bootstrap {report['bootstrap']['draws']} draws "
        f"(seed {report['bootstrap']['seed']}).",
        "",
        "## Segments",
        "",
        "| symbol | kind | horizon | cut | kept | excluded (missing minute) | excluded (outside) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in report["counts"]:
        h = c["horizon_bars"] if c["horizon_bars"] is not None else "all"
        lines.append(
            f"| {c['symbol']} | {c['kind']} | {h} | {c['cut']} | {c['kept']} | "
            f"{c['excluded_invalid']} | {c['excluded_outside']} |"
        )
    lines += ["", "Mean share of event-segment minutes in bars with H = 1:", ""]
    lines += [f"- {k}: {v:.3f}" for k, v in report["mean_high_share"].items()]
    lines += [
        "",
        "## AUC, events against controls (95 % bootstrap CI)",
        "",
        "| indicator | horizon | group | n events | n controls | raw | scale | mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    by = {(r["variant"], r["indicator"], r["horizon_bars"], r["group"]): r for r in report["cells"]}
    for r in report["cells"]:
        if r["variant"] != "raw":
            continue
        at = (r["indicator"], r["horizon_bars"], r["group"])
        rest = [by[m, *at] for m in prereg.DESEASON_METHODS]

        def fmt(x: dict[str, Any]) -> str:
            return f"{x['auc']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"

        lines.append(
            f"| {r['indicator']} | {r['horizon_bars']} | {r['group']} | {r['n_pos']} | "
            f"{r['n_neg']} | " + " | ".join(fmt(x) for x in [r, *rest]) + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="the 2023 pilot labels run")
    parser.add_argument("--labels", type=Path, default=LABELS, help="the frozen labels file")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args(argv)
    report = explore(args.run, args.labels, args.archive, args.draws, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
