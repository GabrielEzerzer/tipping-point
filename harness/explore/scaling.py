"""The static finite-size-scaling table at Tc (exploratory, not pre-registered) from the per-size
outputs of static_tc.py: I(Tc, L) per size and the slope kappa of log|I| on log L.

kappa is fss.power_law over the sizes given; its 95% CI pairs the sizes' bootstrap draws (2,000
hierarchical resamples per size, drawn independently at each L, as fss.fit_exponent does). With two
sizes kappa is the slope between two points: a slope, not a test of the exponent. Classic mode sets
it beside theory and beside the #26 research pass's L = 16-64 pilot kappa, and takes the same slope
from the #50 pilot runs held at exactly Tc, which are independent of the staircases. With --latents
it is each quantity ising-latent-v1 names, with what that file expects of it. Also the half-width of
each CI, what a reference's tolerance in that file is set against. Writes scaling.json and
scaling.md under the table's folder; scaling_plot.py draws the PNG.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
from tipping_point.ews import fss, scoring

from explore import common

CLASSIC = {  # label, theory kappa, theory note, research-pass kappa (L = 16-64), prereg tol
    "l2_var_abs_m": ("L^2 Var(|m|)", 1.75, "gamma/nu = 7/4", "+1.71 [1.66, 1.75]", 0.1),
    "mean_abs_m": ("<|m|>", -0.125, "-beta/nu = -1/8", "-0.116 [-0.122, -0.110]", 0.02),
    "skewness_abs_m": ("skewness of |m|", 0.0, "0 (flat)", "flat: -1.52, -1.51, -1.57", 0.1),
    "tau_int_abs_m": ("tau_int(|m|)", 2.17, "z about 2.17", "+2.07 [1.98, 2.16]", None),
    "binder_ratio": ("<m^4>/<m^2>^2", 0.0, "0 (universal 1.1679)", "not reported", None),
    "one_minus_ac1": (
        "1 - AC1(|m|), lag 1 sweep",
        -1.75,
        "about -gamma/nu = -7/4 (1/chi, not -z)",
        "-1.65 [-1.68, -1.60]",
        None,
    ),
}
LATENT = {  # ising-latent-v1's fss.static, with what that file expects
    "l2_var_pc_1": ("L^2 Var(pc_1)", 1.75, "pipeline check: within 0.15 of 7/4", "", 0.15),
    "mean_pc_1": ("<pc_1>", float("nan"), "descriptive; centred on the training mean", "", None),
    "l2_var_surprise": ("L^2 Var(surprise)", float("nan"), "free; CI must exclude 0", "", None),
    "mean_surprise": ("<surprise>", float("nan"), "descriptive", "", None),
    "one_minus_ac1_pc_1": (
        "1 - AC1(pc_1)",
        float("nan"),
        "reference; v1's |m| about -1.6",
        "",
        None,
    ),
    "skewness_surprise": ("skewness of surprise", float("nan"), "not judged", "", None),
    "skewness_pc_1": ("skewness of pc_1", float("nan"), "not judged", "", None),
    "one_minus_ac1_surprise": ("1 - AC1(surprise)", float("nan"), "not judged", "", None),
}
DIGITS = {"tau_int_abs_m": 1, "l2_var_abs_m": 2, "l2_var_pc_1": 2}


def main(argv: list[str] | None = None) -> None:
    p = common.parser(__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 32])
    args = p.parse_args(argv)
    sizes = sorted(args.sizes)
    static = {L: common.read(args, L, "static.json") for L in sizes}
    draws = {L: np.load(common.out_dir(args, L) / "static-draws.npz") for L in sizes}
    rows = []
    for q, (label, theory, note, research, tol) in (LATENT if args.latents else CLASSIC).items():
        row: dict[str, Any] = {
            "quantity": q,
            "label": label,
            "theory": theory,
            "theory_note": note,
            "research_pass": research,
            "tolerance": tol,
            "per_L": {},
            "pilot_per_L": {},
        }
        for source, key, out in (
            ("staircase", "staircase", "per_L"),
            ("pilot", "pilot_at_tc", "pilot_per_L"),
        ):
            have = [L for L in sizes if f"{source}_{q}" in draws[L].files]
            if len(have) != len(sizes):
                continue
            for L in sizes:
                entry = static[L][key]["stats"][q] if source == "pilot" else static[L][key][q]
                row[out][str(L)] = {"value": entry["value"], "ci": entry["ci"]}
            point = np.array([row[out][str(L)]["value"] for L in sizes])
            n = min(len(draws[L][f"{source}_{q}"]) for L in sizes)
            stack = np.stack([draws[L][f"{source}_{q}"][:n] for L in sizes])
            kappa = float(fss.power_law(sizes, point))
            ci = scoring.percentile_interval(fss.power_law(sizes, stack))
            row["kappa" if source == "staircase" else "pilot_kappa"] = {"value": kappa, "ci": ci}
        k = row["kappa"]
        k["half_width"] = (k["ci"][1] - k["ci"][0]) / 2
        row["excludes_zero"] = bool(k["ci"][0] > 0 or k["ci"][1] < 0)
        if np.isfinite(theory):
            row["ci_covers_theory"] = bool(k["ci"][0] <= theory <= k["ci"][1])
        if tol is not None:
            row["within_tolerance"] = bool(abs(k["value"] - theory) <= tol)
            row["ci_half_width_vs_tolerance"] = k["half_width"] / tol
        rows.append(row)
    result = {
        "sizes": sizes,
        "note": "two sizes give a slope, not a test; exploratory, not pre-registered",
        "runs": {str(L): static[L]["plateaus"] for L in sizes},
        "pilot_runs": {str(L): (static[L]["pilot_at_tc"] or {}).get("run_id") for L in sizes},
        "provenance": common.provenance(args) | {"L": sizes},
        "rows": rows,
    }
    out_dir = common.table_dir(args)
    common.save(out_dir / "scaling.json", result)

    def f(v: float, d: int = 3) -> str:
        return f"{v:+.{d}f}"

    def fmt(entry: dict[str, Any], d: int) -> str:
        lo, hi = entry["ci"]
        return f"{f(entry['value'], d)} [{f(lo, d)}, {f(hi, d)}]"

    lines = [
        "| at Tc | "
        + " | ".join(f"L = {L}" for L in sizes)
        + " | kappa, staircases [95% CI] | half-width | kappa, #50 pilot runs at Tc | expected |",
        "|---|" + "---|" * len(sizes) + "---|---|---|---|",
    ]
    for row in rows:
        d = DIGITS.get(row["quantity"], 4)
        k, pk = row["kappa"], row.get("pilot_kappa")
        cells = [fmt(row["per_L"][str(L)], d) for L in sizes]
        label = row["label"].replace("|", "\\|")
        lines.append(
            f"| {label} | "
            + " | ".join(cells)
            + f" | {fmt(k, 3)} | {k['half_width']:.3f} | {fmt(pk, 3) if pk else '-'} "
            + f"| {row['theory_note'].replace('|', chr(92) + '|')} |"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "scaling.md").write_text(text, "utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    print(text)
    print(
        json.dumps({"sizes": sizes, "kappas": {r["quantity"]: r["kappa"] for r in rows}}, indent=1)
    )
