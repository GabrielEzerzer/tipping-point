"""A training's exploratory latent pass as one markdown file: its trained checkpoint and its
untrained twin side by side at each size, every value with its interval and the interval's method.
Reads the JSON the other scripts wrote, and nothing else; its rule for surprise's directions is
directions(), below.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tipping_point.ews.ising import latents_name

HEADLINE = 20.0  # tau_c, the headline window


def _load(out: Path, table: str, L: int | None, name: str) -> Any:
    folder = out / table if L is None else out / table / f"L{L}"
    path = folder / name
    if path.is_file():
        return json.loads(path.read_text("utf-8"))
    return None


def _v(entry: dict[str, Any], d: int = 3, sign: bool = False) -> str:
    fmt = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    lo, hi = entry["ci"]
    return f"{fmt.format(entry['value'])} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _headline(auc: dict[str, Any], pair: tuple[str, str], h: float) -> dict[str, Any]:
    (row,) = [
        r
        for r in auc["auc"]
        if r["headline"] and (r["observable"], r["indicator"]) == pair and r["h"] == h
    ]
    return row


def directions(out: Path, trained: str, sizes: list[int]) -> dict[str, str]:
    """Surprise's directions, by the rule fixed before the numbers were read: an indicator's
    direction is the sign of the median trend score s on the trained checkpoint's slow ramps at the
    headline cell, s being Kendall's tau of the rolling indicator as the bath cools; s > 0 is
    "greater". They must agree at every size and h, or this says so instead of choosing. The slope's
    extremum follows the level: a level that falls as the bath cools has its steepest change at the
    slope's minimum."""
    found: dict[str, set[str]] = {"variance": set(), "level": set()}
    for L in sizes:
        auc = _load(out, trained, L, "auc_alarms.json")
        for row in auc["auc"]:
            if row["headline"] and row["observable"] == "surprise":
                found[row["indicator"]].add("greater" if row["pos_median_s"] > 0 else "less")
    answer: dict[str, str] = {}
    for name, signs in found.items():
        answer[name] = signs.pop() if len(signs) == 1 else "disagree"
    answer["slope_extremum"] = {"greater": "max", "less": "min"}.get(answer["level"], "disagree")
    return answer


def _shape(c: dict[str, Any]) -> str:
    m = c["mean_shape"]
    parts = []
    for name in ("peak", "valley"):
        s = m[name]
        lo, hi = s["prominence_ci"]
        turns = lo > 0
        parts.append(
            f"{name}: prominence {s['prominence']:.4f} [{lo:.4f}, {hi:.4f}], at eps_L "
            f"{s['eps_L']:+.2f} [{s['eps_L_ci'][0]:+.2f}, {s['eps_L_ci'][1]:+.2f}], "
            f"{'interior' if turns else 'no interior turn'}"
        )
    tau = m["kendall_tau_cooling"]
    lo, hi = m["kendall_tau_cooling_ci"]
    return (
        f"Kendall's tau over the plateaus, cooling: {tau:+.3f} [{lo:+.3f}, {hi:+.3f}]; "
        + "; ".join(parts)
    )


def _curve_rows(
    out: Path, tables: dict[str, str], L: int, column: str, every: int = 1
) -> list[str]:
    head = (
        f"| eps_L, L = {L} | T | {column} mean, trained | twin | L^2 Var, trained | twin "
        "| r with \\|m\\|, trained | twin | r with e, trained | twin |"
    )
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    st = {who: _load(out, t, L, "static.json") for who, t in tables.items()}
    cor = {who: _load(out, t, L, "correlations.json") for who, t in tables.items()}
    tr, tw = st["trained"]["equilibrium"][column], st["twin"]["equilibrium"][column]
    rt = cor["trained"]["columns"][column]
    rz = cor["twin"]["columns"][column]
    for k in range(0, len(tr["T"]), every):
        lines.append(
            f"| {tr['eps_L'][k]:+.2f} | {tr['T'][k]:.3f} "
            f"| {tr['mean'][k]:.4f} [{tr['mean_ci'][k][0]:.4f}, {tr['mean_ci'][k][1]:.4f}] "
            f"| {tw['mean'][k]:.4f} [{tw['mean_ci'][k][0]:.4f}, {tw['mean_ci'][k][1]:.4f}] "
            f"| {tr['l2_var'][k]:.3f} [{tr['l2_var_ci'][k][0]:.3f}, {tr['l2_var_ci'][k][1]:.3f}] "
            f"| {tw['l2_var'][k]:.3f} [{tw['l2_var_ci'][k][0]:.3f}, {tw['l2_var_ci'][k][1]:.3f}] "
            f"| {rt['abs_m']['per_plateau'][k]:+.3f} | {rz['abs_m']['per_plateau'][k]:+.3f} "
            f"| {rt['energy']['per_plateau'][k]:+.3f} | {rz['energy']['per_plateau'][k]:+.3f} |"
        )
    lines.append("")
    return lines


def _margin(out: Path, trained: str, twin: str, L: int) -> dict[str, dict[str, Any]]:
    """Trained minus twin on the same test draws: a data interval for one model seed."""
    a, b = (np.load(out / t / f"L{L}" / "probes-draws.npz") for t in (trained, twin))
    p, q = _load(out, trained, L, "probes.json"), _load(out, twin, L, "probes.json")
    margins: dict[str, dict[str, Any]] = {}
    for t in p["targets"]:
        margins[t] = {}
        for key in ("r2", "selectivity"):
            d = a[f"{t}_{key}"] - b[f"{t}_{key}"]
            lo, hi = np.nanquantile(d, [0.025, 0.975])
            value = p["targets"][t][key]["value"] - q["targets"][t][key]["value"]
            margins[t][key] = {"value": float(value), "ci": [float(lo), float(hi)]}
    return margins


def render(out: Path, training: str, step: int, sizes: list[int]) -> str:
    trained, twin = latents_name(training, step), latents_name(training, 0)
    tables = {"trained": trained, "twin": twin}
    lines = [
        f"# Exploratory latent pass: {training}, step {step} beside its step-0 twin",
        "",
        "Exploratory, seeds 1-5 at L = "
        + " and ".join(map(str, sizes))
        + ". Never logged as trials.",
        "",
        "## Directions, by the rule in latent_report.directions",
        "",
    ]
    answer = directions(out, trained, sizes)
    lines += [f"- {key}: **{value}**" for key, value in answer.items()]
    lines += ["", "## Trend scores and AUC at the headline cell", ""]
    lines += [
        "CI: percentile, hierarchical bootstrap, seeds then replicas. s: Kendall's tau of the "
        "rolling indicator as the bath cools, median over the ramps' replicas.",
        "",
        "| pair | L | h | AUC, trained | median s | AUC, twin | median s |",
        "|---|---|---|---|---|---|---|",
    ]
    for pair in (
        ("surprise", "level"),
        ("surprise", "variance"),
        ("pc_1", "variance"),
        ("pc_1", "ac"),
    ):
        for L in sizes:
            auc = {who: _load(out, t, L, "auc_alarms.json") for who, t in tables.items()}
            for h in sorted({r["h"] for r in auc["trained"]["auc"]}):
                a, b = _headline(auc["trained"], pair, h), _headline(auc["twin"], pair, h)
                lines.append(
                    f"| {pair[0]} {pair[1]} | {L} | {h:g} "
                    f"| {a['auc']:.3f} [{a['auc_lo']:.3f}, {a['auc_hi']:.3f}] "
                    f"| {a['pos_median_s']:+.2f} "
                    f"| {b['auc']:.3f} [{b['auc_lo']:.3f}, {b['auc_hi']:.3f}] "
                    f"| {b['pos_median_s']:+.2f} |"
                )
    lines += ["", "## Surprise against T: is it monotone or peaked?", ""]
    lines += ["CI: percentile, one joint hierarchical bootstrap over every plateau.", ""]
    for L in sizes:
        for who, t in tables.items():
            st = _load(out, t, L, "static.json")
            c = st["equilibrium"]["surprise"]
            v = c["peak_eps_L_ci"]
            lines.append(f"- L = {L}, {who}: {_shape(c)}.")
            lines.append(
                f"  L^2 Var(surprise) is highest at eps_L {c['peak_eps_L']:+.2f} "
                f"[{v[0]:+.2f}, {v[1]:+.2f}] on the plateaus."
            )
    lines.append("")
    for L in sizes:
        lines += _curve_rows(out, tables, L, "surprise")
    lines += ["## pc_1 against T", ""]
    for L in sizes:
        lines += _curve_rows(out, tables, L, "pc_1")
    lines += ["## What pc_1 follows, at the FSS plateau and overall", ""]
    lines += [
        "CI: percentile, joint hierarchical bootstrap, seeds then replicas. A PCA's sign is "
        "arbitrary: read r by its size. Per-plateau r is at fixed T; pooled r holds the shared "
        "trend in T.",
        "",
        "| column | L | with | r at the FSS plateau, trained | twin | partial, trained | twin "
        "| pooled r, trained | twin | chi alignment, trained | twin |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for column in ("pc_1", "pc_2", "pc_3", "surprise"):
        for L in sizes:
            a = _load(out, trained, L, "correlations.json")["columns"][column]
            b = _load(out, twin, L, "correlations.json")["columns"][column]
            for t in ("abs_m", "energy"):
                name = "\\|m\\|" if t == "abs_m" else "e"
                lines.append(
                    f"| {column} | {L} | {name} | {_v(a[t]['at_fss_plateau'], 3, True)} "
                    f"| {_v(b[t]['at_fss_plateau'], 3, True)} "
                    f"| {_v(a[t]['partial_at_fss_plateau'], 3, True)} "
                    f"| {_v(b[t]['partial_at_fss_plateau'], 3, True)} "
                    f"| {_v(a[t]['pooled'], 3, True)} | {_v(b[t]['pooled'], 3, True)} "
                    f"| {_v(a[t]['l2_var_across_T'], 3, True)} "
                    f"| {_v(b[t]['l2_var_across_T'], 3, True)} |"
                )
    lines += ["", "## Exponents over the sizes (a slope, not a test)", ""]
    lines += [
        "CI: percentile, independent hierarchical bootstraps per size, paired. kappa of log|I| "
        "on log L.",
        "",
        "| quantity | kappa, trained | half-width | excludes 0 | kappa, twin | half-width "
        "| excludes 0 |",
        "|---|---|---|---|---|---|---|",
    ]
    a, b = _load(out, trained, None, "scaling.json"), _load(out, twin, None, "scaling.json")
    for x, y in zip(a["rows"], b["rows"], strict=True):
        lines.append(
            f"| {x['label'].replace('|', chr(92) + '|')} | {_v(x['kappa'], 3, True)} "
            f"| {x['kappa']['half_width']:.3f} | {x['excludes_zero']} | {_v(y['kappa'], 3, True)} "
            f"| {y['kappa']['half_width']:.3f} | {y['excludes_zero']} |"
        )
    lines += ["", "## Peak positions along the slow ramps, and 1/nu over the sizes", ""]
    lines += [
        "CI: percentile, hierarchical bootstrap. Near the edge: on the search band's edge or "
        "within one window of it. The verdict is what ising._dynamic's rules give, read at two "
        "sizes, 80 replicas.",
        "",
        "| statistic | who | median eps_L per L | near the edge per L | ensemble extremum per L "
        "| 1/nu [95% CI] | verdict if judged |",
        "|---|---|---|---|---|---|---|",
    ]
    a, b = (
        _load(out, trained, None, "peaks-combined.json"),
        _load(out, twin, None, "peaks-combined.json"),
    )
    for key in a["indicators"]:
        for who, data in (("trained", a), ("twin", b)):
            v = data["indicators"][key]
            med = " / ".join(f"{v['medians'][str(L)]:+.2f}" for L in sizes)
            near = " / ".join(f"{v['near_edge_shares'][str(L)]:.0%}" for L in sizes)
            ens = " / ".join(f"{v['ensemble'][str(L)]:+.2f}" for L in sizes)
            ci = v["inverse_nu_ci"]
            lines.append(
                f"| {key} | {who} | {med} | {near} | {ens} "
                f"| {v['inverse_nu']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] "
                f"| {v['verdict_if_judged']} |"
            )
    lines += [
        "",
        "## Probe controls and the floors, from 8 pcs (not the 128-dimensional embedding)",
        "",
    ]
    lines += [
        "CI: percentile, hierarchical bootstrap of the test frames, seeds then replicas; data "
        "noise only, one model seed. Each seed fits, the other four test; a value is the mean over "
        "the rotations. "
        "Selectivity noise: 95th percentile of the gap between redraws of the shuffle control.",
        "",
        "| target | L | who | R^2 | iid floor | shuffle-within-T floor | selectivity | of the rest "
        "| selectivity noise |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in ("abs_m", "energy", "l2var"):
        for L in sizes:
            for who, name in tables.items():
                v = _load(out, name, L, "probes.json")["targets"][t]
                lines.append(
                    f"| {t} | {L} | {who} | {_v(v['r2'], 4)} | {_v(v['iid'], 4, True)} "
                    f"| {_v(v['shuffle_within_T'], 4)} | {_v(v['selectivity'], 4, True)} "
                    f"| {_v(v['of_rest'], 3, True)} | {v['selectivity_noise_q95']:.4f} |"
                )
    lines += ["", "| margin, trained minus twin | L | R^2 | selectivity |", "|---|---|---|---|"]
    for L in sizes:
        margins = _margin(out, trained, twin, L)
        for t, m in margins.items():
            lines.append(
                f"| {t} | {L} | {_v(m['r2'], 4, True)} | {_v(m['selectivity'], 4, True)} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 32])
    args = p.parse_args(argv)
    text = render(args.out, args.training, args.step, args.sizes)
    target = args.out / f"latent-report-{args.training}.md"
    target.write_text(text, "utf-8")
    print(text)
    print(f"wrote {target}")
