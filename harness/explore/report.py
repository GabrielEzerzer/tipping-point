"""One size's exploratory JSON outputs as markdown tables, <out>/<table>/L<L>/report.md. Reads only
those files; every number in them is exploratory, never logged as a trial.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from explore import common

NAMES = {
    "variance": "Var",
    "ac": "AC1",
    "skewness": "Skew",
    "level": "Level",
    "slope": "Slope",
}


def f(x: Any, d: int = 3, sign: bool = False) -> str:
    if isinstance(x, float) and math.isinf(x):
        return "-inf" if x < 0 else "+inf"
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(bounds: Any, d: int = 3, sign: bool = False) -> str:
    return f"[{f(bounds[0], d, sign)}, {f(bounds[1], d, sign)}]"


def pct(x: Any, d: int = 1) -> str:
    return "-" if x is None or not math.isfinite(x) else f"{100 * x:.{d}f}%"


def load(out: Path, name: str) -> dict[str, Any] | None:
    p = out / name
    return json.loads(p.read_text("utf-8")) if p.is_file() else None


def label(row: dict[str, Any]) -> str:
    name = row["observable"].replace("abs_m", chr(92) + "|m" + chr(92) + "|")
    return f"{name} {NAMES[row['indicator']]}"


def merged_surrogates(out: Path) -> dict[str, Any] | None:
    """surrogates-<observable>-<indicator>-<kind>.json, merged."""
    files = sorted(out.glob("surrogates-*.json"))
    if not files:
        return None
    rows, meta = [], None
    for p in files:
        d = json.loads(p.read_text("utf-8"))
        meta = meta or {k: v for k, v in d.items() if k != "rows"}
        rows += d["rows"]
    return (meta or {}) | {"rows": rows}


def merged_alarms(out: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(out.glob("alarms_explore-*.json")):
        rows += json.loads(p.read_text("utf-8"))["rows"]
    return rows


def auc_tables(a: dict[str, Any]) -> list[str]:
    head = [r for r in a["auc"] if r["headline"]]
    lines = [
        "| pair | h | AUC [95% CI] | median s, ramps / nulls | ramps' s, 5-95% | nulls' s, 5-95% "
        "| n ramps / nulls |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in head:
        pq, nq = r["pos_q05_q95_s"], r["neg_q05_q95_s"]
        lines.append(
            f"| {label(r)} | {r['h']:g} | {f(r['auc'])} {ci((r['auc_lo'], r['auc_hi']))} "
            f"| {f(r['pos_median_s'], 2, True)} / {f(r['neg_median_s'], 2, True)} "
            f"| {f(pq[0], 2, True)} to {f(pq[1], 2, True)} "
            f"| {f(nq[0], 2, True)} to {f(nq[1], 2, True)} | {r['n_pos']} / {r['n_neg']} |"
        )
    lines.append("")
    if a["skipped"]:
        lines += [f"Skipped, no room on the ramp: {len(a['skipped'])} cells.", ""]
    if a["alarms"]:
        lines += [
            "| pair | theta | attainable | highest rate any theta gives, per 1,000 tau_c "
            "| false alarms per 1,000 tau_c, set / test | alarm eps_L, median [IQR] | censored "
            "| missed |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in a["alarms"]:
            lines.append(
                f"| {label(r)} | {f(r['theta'], 3, True)} | {r['attainable']} "
                f"| {f(r['far_max'], 2)} | {f(r['far_set'], 1)} / {f(r['far_test'], 1)} "
                f"| {f(r['alarm_eps_median'], 2, True)} "
                f"{ci((r['alarm_eps_q1'], r['alarm_eps_q3']), 2, True)} "
                f"| {pct(r['censored'])} | {pct(r['miss_rate'])} |"
            )
        lines.append("")
    return lines


def alarm_explore_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| pair | theta rule | theta | false alarms per 1,000 tau_c, set / test "
        "| alarm eps_L, median [IQR] | 95% CI of the median | median lead | alarmed above Tc "
        "| censored | missed | s(t) as the band opens, median |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for rule, v in r["variants"].items():
            if not v.get("attainable"):
                lines.append(
                    f"| {label(r)} | {rule} | unattainable: at most "
                    f"{f(v['far_max_per_1000'], 2)} per 1,000 | | | | | | | | |"
                )
                continue
            lo, hi = v["alarm_eps_median_ci"]
            lines.append(
                f"| {label(r)} | {rule} | {f(v['theta'], 3, True)} "
                f"| {f(v['far_set'], 2)} / {f(v['far_test'], 2)} "
                f"| {f(v['alarm_eps_median'], 2, True)} {ci(v['alarm_eps_iqr'], 2, True)} "
                f"| {ci((lo, hi), 2, True)} | {f(v['lead_taus_median'], 0)} tau_c "
                f"| {pct(v['alarmed_above_tc'])} | {pct(v['censored'])} | {pct(v['missed'])} "
                f"| {f(v['entry_score_median'], 2, True)} |"
            )
    lines.append("")
    for r in rows:
        c, s = r["ceiling"], r["first_score"]
        lines.append(
            f"- {label(r)}: the most false alarms any theta gives on the set half is "
            f"{f(c['with_first'], 2)} per 1,000 tau_c ({f(c['rises_only'], 2)} counting only rises "
            f"between scores; {c['rows']} null paths, {f(c['duration_taus'], 0)} tau_c of s(t)); "
            f"largest null s(t) {f(r['null_s_max_set'], 3, True)}. The ramps' first s(t) comes at "
            f"eps_L = {f(s['eps_L'], 2, True)}: median {f(s['median'], 2, True)}, above every null "
            f"s(t) in {pct(s['above_all_nulls'], 0)} of replicas."
        )
    lines.append("")
    return lines


def grid_table(a: dict[str, Any]) -> list[str]:
    """AUC and its 95% CI per cell of the grid, pair by pair: a RAW indicator has one cell a
    window."""
    rows = a["auc"]
    hs = sorted({r["h"] for r in rows})
    lines = [
        "| pair | window x bandwidth | " + " | ".join(f"h = {h:g}" for h in hs) + " |",
        "|---|---|" + "---|" * len(hs),
    ]
    keys = sorted(
        {(r["observable"], r["indicator"], r["window_taus"], r["bandwidth_taus"]) for r in rows},
        key=lambda k: (k[0], k[1], k[2], 0.0 if k[3] != k[3] else k[3]),
    )
    for o, i, w, b in keys:
        cells = []
        for h in hs:
            hit = [
                r
                for r in rows
                if (r["observable"], r["indicator"], r["window_taus"], r["h"]) == (o, i, w, h)
                and (
                    r["bandwidth_taus"] == b
                    or (b != b and r["bandwidth_taus"] != r["bandwidth_taus"])
                )
            ]
            cells.append(
                f"{f(hit[0]['auc'])} {ci((hit[0]['auc_lo'], hit[0]['auc_hi']))}" if hit else "-"
            )
        star = " **(headline)**" if (w, b) == (20.0, 10.0) or (b != b and w == 20.0) else ""
        pair = {"observable": o, "indicator": i}
        cell = f"{w:g}" if b != b else f"{w:g} x {b:g}"
        lines.append(f"| {label(pair)} | {cell}{star} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def surrogate_table(s: dict[str, Any]) -> list[str]:
    lines = [
        "| pair | surrogate | false positives on nulls, alpha = 0.05 | 95% band (99%) "
        "| by eps_L = 2 / 4 / 6 | at alpha = 0.01 / 0.10 | detected on ramps, h = 0 / 1 / 2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in s["rows"]:
        n = r["null"]
        lo, hi = r["band95"]
        lo9, hi9 = r["band99"]
        by = " / ".join(
            pct(r["null_by_eps"][e]["0.05"]) for e in sorted(r["null_by_eps"], key=float)
        )
        det = " / ".join(
            pct(v["0.05"], 0)
            for _, v in sorted(r["positives"].items(), key=lambda kv: float(kv[0]))
        )
        inside = (
            "inside" if r["inside95"] else ("inside 99% only" if r["inside99"] else "**outside**")
        )
        lines.append(
            f"| {label(r)} | {r['surrogate']} | {pct(n['0.05'])} of {n['n']} ({inside}) "
            f"| {pct(lo)}-{pct(hi)} ({pct(lo9)}-{pct(hi9)}) | {by} "
            f"| {pct(n['0.01'])} / {pct(n['0.1'])} | {det} |"
        )
    lines.append("")
    return lines


def nulls_table(nl: dict[str, Any]) -> list[str]:
    """The null runs alone: static values per eps_L, the headline pipeline's indicator levels and
    trend scores (median [5-95%])."""
    lines = [
        "| column | eps_L | mean [95% CI] | L^2 Var [95% CI] | rolling levels | s, 5-95% |",
        "|---|---|---|---|---|---|",
    ]
    for column, entry in nl["columns"].items():
        for e, at in sorted(entry["per_eps"].items(), key=lambda kv: float(kv[0])):
            st = at["static"]
            levels = "; ".join(
                f"{NAMES[k]} {f(v['median'], 3)} [{f(v['q05'], 3)}, {f(v['q95'], 3)}]"
                for k, v in at["levels"].items()
            )
            scores = "; ".join(
                f"{NAMES[k]} {f(v['q05'], 2, True)} to {f(v['q95'], 2, True)}"
                for k, v in at["scores"].items()
            )
            name = column.replace("abs_m", "\\|m\\|")
            lines.append(
                f"| {name} | {e} | {f(st['mean']['value'], 4)} {ci(st['mean']['ci'], 4)} "
                f"| {f(st['l2_var']['value'], 3)} {ci(st['l2_var']['ci'], 3)} "
                f"| {levels} | {scores} |"
            )
    lines.append("")
    return lines


def static_table(st: dict[str, Any]) -> list[str]:
    s, p = st["staircase"], (st["pilot_at_tc"] or {}).get("stats", {})
    L = st["provenance"]["L"]
    pilot = st.get("pilot_at_tc") or {}
    head = f"#50 pilot run at Tc, {pilot.get('replicas', '-')} x {pilot.get('rows', '-')} [95% CI]"
    lines = [
        f"| at Tc, L = {L} | staircases [95% CI] | per seed | {head} |",
        "|---|---|---|---|",
    ]
    for q, v in s.items():
        d = 1 if q.startswith("tau_int") else 4
        per = ", ".join(f(x, d - 1 if d > 1 else 1) for x in v["per_seed"])
        pv = p.get(q)
        pcol = f"{f(pv['value'], d)} {ci(pv['ci'], d)}" if pv else "-"
        name = q.replace("abs_m", "\\|m\\|")
        lines.append(f"| {name} | {f(v['value'], d)} {ci(v['ci'], d)} | {per} | {pcol} |")
    lines.append("")
    return lines


def fast_tables(fr: dict[str, Any], st: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for column, entry in fr["columns"].items():
        eq = st["equilibrium"][column]
        eps_eq = np.array(eq["eps_L"])[::-1]
        var_eq = np.array(eq["l2_var"])[::-1]
        mean_eq = np.array(eq["mean"])[::-1]
        fast, slow = entry["speeds"]["fast"], entry["speeds"]["slow"]
        name = column.replace("abs_m", "\\|m\\|")
        lines += [
            f"| {name}: eps_L | fast x = 2: tau_c from t_c | mean | L^2 Var | skew | AC(0.1 tau_c) "
            "| slow x = 0.02: mean | L^2 Var | equilibrium: mean | L^2 Var |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for key in fast["at_eps_L"]:
            e = float(key)
            a, b = fast["at_eps_L"][key], slow["at_eps_L"].get(key, {})
            lines.append(
                f"| {e:+g} | {f(a['taus_from_tc'], 1, True)} | {f(a['mean'])} "
                f"| {f(a['l2_var'], 2)} "
                f"| {f(a['skewness'], 2, True)} | {f(a['ac'], 3)} "
                f"| {f(b.get('mean'))} | {f(b.get('l2_var'), 2)} "
                f"| {f(float(np.interp(e, eps_eq, mean_eq)))} "
                f"| {f(float(np.interp(e, eps_eq, var_eq)), 2)} |"
            )
        lines.append("")
        lines += [
            f"| {name}: ramp | ensemble Var peak eps_L [95% CI] | peak lag vs equilibrium, eps_L "
            "| peak, tau_c after t_c | mean reaches its Tc value at eps_L | ... tau_c after t_c "
            "| Kendall tau of ensemble Var, eps_L 6 to 0 (seed range) |",
            "|---|---|---|---|---|---|---|",
        ]
        for tag, sp in (("fast x = 2", fast), ("slow x = 0.02", slow)):
            s = sp["stats"]

            def c(key: str, d: int = 2, sign: bool = True, stats: dict[str, Any] = s) -> str:
                v = stats[key]
                gone = v["undefined_share"]
                extra = f" ({pct(gone, 0)} undefined)" if gone > 0.005 else ""
                return f"{f(v['value'], d, sign)} {ci(v['ci'], d, sign)}{extra}"

            lines.append(
                f"| {tag} | {c('peak_eps_L')} | {c('peak_lag_eps_L')} "
                f"| {c('peak_taus_after_tc', 1)} "
                f"| {c('order_reach_eps_L')} | {c('order_lag_taus_after_tc', 1)} "
                f"| {f(s['kendall_tau_var_6_to_0']['value'])} "
                f"({f(min(sp['kendall_tau_per_seed']))} to {f(max(sp['kendall_tau_per_seed']))}) |"
            )
        lines.append("")
    return lines


def peaks_table(pk: dict[str, Any]) -> list[str]:
    lines = [
        "| statistic | median eps_L [95% CI] | IQR | on the band's edge | within a window of it "
        "| ensemble extremum |",
        "|---|---|---|---|---|---|",
    ]
    for key, v in pk["indicators"].items():
        lines.append(
            f"| {key} | {f(v['median'], 2, True)} {ci(v['median_ci'], 2, True)} "
            f"| {ci(v['iqr'], 2, True)} | {pct(v['on_edge'], 0)} | {pct(v['near_edge'], 0)} "
            f"| {f(v['ensemble_extreme_eps'], 2, True)} |"
        )
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    p = common.parser(__doc__)
    args = p.parse_args(argv)
    out = common.out_dir(args)
    a, st, fr, nl, pk = (
        load(out, n)
        for n in ("auc_alarms.json", "static.json", "fast_ramps.json", "nulls.json", "peaks.json")
    )
    s = merged_surrogates(out)
    lines: list[str] = []
    if a:
        lines += ["### 1. Headline cell", "", *auc_tables(a)]
    ax = merged_alarms(out)
    if ax:
        lines += ["### 1b. Alarm thresholds examined", "", *alarm_explore_table(ax)]
    if a:
        lines += ["### Sensitivity grid: AUC (95% CI)", "", *grid_table(a)]
    if pk:
        lines += ["### Peak positions", "", *peaks_table(pk)]
    if s:
        lines += ["### 2. Surrogate calibration on the real nulls", "", *surrogate_table(s)]
    if st:
        lines += ["### 3. Static values at Tc", "", *static_table(st)]
    if fr and st:
        lines += ["### 4. Ramps: ensemble indicators", "", *fast_tables(fr, st)]
    if nl:
        lines += ["### Null runs alone", "", *nulls_table(nl)]
    text = "\n".join(lines)
    (out / "report.md").write_text(text, "utf-8")
    print(text)
