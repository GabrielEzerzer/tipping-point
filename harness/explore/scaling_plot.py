# pyright: reportMissingImports=false
"""Draw the static scaling plot from scaling.json (scaling.py): one log-log panel per quantity at
Tc, the staircases' and the #50 pilot runs' values with 95% CIs, and the theory and research-pass
slopes drawn through the staircases' geometric mean. matplotlib is not a project dependency, so this
runs in a throwaway environment and reads nothing but the JSON:
  uv run --isolated --no-project --with matplotlib==3.10.3 python \
      harness/explore/scaling_plot.py <out>/<table>/scaling.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter, NullLocator

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
STAIR, PILOT, RESEARCH = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated)
RESEARCH_KAPPA = {
    "l2_var_abs_m": 1.71,
    "mean_abs_m": -0.116,
    "tau_int_abs_m": 2.07,
    "one_minus_ac1": -1.65,
    "skewness_abs_m": 0.0,
}


def main() -> None:
    src = Path(sys.argv[1])
    data = json.loads(src.read_text("utf-8"))
    sizes = data["sizes"]
    rows = data["rows"]
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
        }
    )
    fig, axes = plt.subplots(
        math.ceil(len(rows) / 3), 3, figsize=(13, 4.1 * math.ceil(len(rows) / 3))
    )
    for ax in axes.ravel()[len(rows) :]:
        ax.set_visible(False)
    for ax, row in zip(axes.ravel(), rows, strict=False):
        q = row["quantity"]
        xs = [float(L) for L in sizes]
        vals = [abs(row["per_L"][str(L)]["value"]) for L in sizes]
        cis = [sorted(abs(v) for v in row["per_L"][str(L)]["ci"]) for L in sizes]
        lo = [v - c[0] for v, c in zip(vals, cis, strict=True)]
        hi = [c[1] - v for v, c in zip(vals, cis, strict=True)]
        # reference slopes through the staircases' geometric mean: a slope comparison only
        x0 = math.sqrt(xs[0] * xs[-1])
        y0 = math.sqrt(vals[0] * vals[-1])
        span = [xs[0] / 1.25, xs[-1] * 1.25]
        if row["theory"] == row["theory"]:  # NaN where nothing is expected
            ax.plot(
                span,
                [y0 * (x / x0) ** row["theory"] for x in span],
                color=INK2,
                lw=1.5,
                ls=(0, (5, 3)),
                zorder=1,
            )
        if q in RESEARCH_KAPPA:  # classic mode
            ax.plot(
                span,
                [y0 * (x / x0) ** RESEARCH_KAPPA[q] for x in span],
                color=RESEARCH,
                lw=2,
                ls=(0, (1, 2.2)),
                dash_capstyle="round",
                zorder=1,
            )
        ax.plot(xs, vals, color=STAIR, lw=2, solid_capstyle="round", zorder=2)
        ax.errorbar(
            xs, vals, yerr=[lo, hi], fmt="none", ecolor=STAIR, elinewidth=1.5, capsize=0, zorder=3
        )
        ax.plot(xs, vals, "o", ms=8, mfc=STAIR, mec=SURFACE, mew=2, zorder=4)
        pc: list[list[float]] = []
        if row["pilot_per_L"]:
            px = [x * 1.06 for x in xs]
            pv = [abs(row["pilot_per_L"][str(L)]["value"]) for L in sizes]
            pc = [sorted(abs(v) for v in row["pilot_per_L"][str(L)]["ci"]) for L in sizes]
            ax.errorbar(
                px,
                pv,
                yerr=[
                    [v - c[0] for v, c in zip(pv, pc, strict=True)],
                    [c[1] - v for v, c in zip(pv, pc, strict=True)],
                ],
                fmt="none",
                ecolor=PILOT,
                elinewidth=1.5,
                capsize=0,
                zorder=3,
            )
            ax.plot(px, pv, "o", ms=8, mfc=PILOT, mec=SURFACE, mew=2, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(span)
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{int(x)}" for x in xs]))
        ax.xaxis.set_minor_locator(NullLocator())
        # y: the data's own range with a margin, a few round ticks inside it (a log axis over a
        # narrow range would otherwise show no labelled tick at all)
        lows = [c[0] for c in [*cis, *pc]]
        highs = [c[1] for c in [*cis, *pc]]
        ymin, ymax = min(lows), max(highs)
        pad = (ymax / ymin) ** 0.12
        ax.set_ylim(ymin / pad, ymax * pad)
        lo_y, hi_y = ymin / pad, ymax * pad
        ticks = [
            m * 10.0**e
            for e in range(math.floor(math.log10(lo_y)), math.ceil(math.log10(hi_y)) + 1)
            for m in (1, 2, 5)
            if lo_y <= m * 10.0**e <= hi_y
        ]  # log ticks where they fit
        if len(ticks) < 3:  # a narrow range: round linear ticks instead
            ticks = [
                t
                for t in MaxNLocator(nbins=4).tick_values(lo_y, hi_y)
                if lo_y <= t <= hi_y and t > 0
            ]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=GRID, lw=1, ls="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("L")
        # name above the panel, the numbers on their own line beneath it, outside the plot area
        k = row["kappa"]
        text = (
            f"kappa = {k['value']:+.3f} [{k['ci'][0]:+.3f}, {k['ci'][1]:+.3f}]"
            f", {row['theory_note']}"
        )
        if row.get("pilot_kappa"):
            text += f", pilot runs {row['pilot_kappa']['value']:+.3f}"
        ax.set_title(
            row["label"], loc="left", color=INK, fontsize=11, fontweight="semibold", pad=22
        )
        ax.text(
            0, 1.02, text, transform=ax.transAxes, va="bottom", ha="left", color=INK2, fontsize=8.5
        )
    handles = [
        Line2D(
            [],
            [],
            color=STAIR,
            lw=2,
            marker="o",
            ms=8,
            mfc=STAIR,
            mec=SURFACE,
            mew=2,
            label="staircases, Tc plateau (5 seeds x 16 replicas), 95% CI",
        ),
        Line2D(
            [],
            [],
            color=PILOT,
            lw=0,
            marker="o",
            ms=8,
            mfc=PILOT,
            mec=SURFACE,
            mew=2,
            label="#50 pilot runs held at Tc (64 replicas), 95% CI",
        ),
        Line2D([], [], color=INK2, lw=1.5, ls=(0, (5, 3)), label="theory slope"),
        Line2D(
            [],
            [],
            color=RESEARCH,
            lw=2,
            ls=(0, (1, 2.2)),
            label="#26 research-pass kappa, L = 16-64",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.935),
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=INK2,
        handlelength=2.6,
    )
    fig.suptitle(
        "Static finite-size scaling at Tc, L = 16 and 32: exploratory, not pre-registered",
        x=0.012,
        y=0.995,
        ha="left",
        color=INK,
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.012,
        0.955,
        "Two sizes give a slope, not a test. Reference slopes are drawn through "
        "the staircases' geometric mean, so only the slopes compare; |skewness| is plotted on "
        "the log scale.",
        ha="left",
        va="top",
        color=INK2,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    png = src.with_name(f"scaling-L{'-'.join(str(L) for L in sizes)}.png")
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    print(png)


if __name__ == "__main__":
    main()
