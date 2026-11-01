# pyright: reportMissingImports=false
"""The exploratory pass's picture: surprise and pc_1 against eps_L for a trained checkpoint and its
twin, at each size, from the JSON the pass wrote. matplotlib is not a project dependency, so this
runs in a throwaway environment and reads nothing but that JSON:
  uv run --isolated --no-project --with matplotlib==3.10.3 python \
      harness/explore/latent_plot.py <out> <training> <step> [16 32]
Writes <out>/latent-curves-<training>.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID = "#e1e0d9"
COLOURS = {16: "#2a78d6", 32: "#eb6834", 64: "#1baf7a"}
PANELS = (
    ("surprise", "mean", "mean surprise"),
    ("surprise", "l2_var", "L^2 Var(surprise)"),
    ("pc_1", "l2_var", "L^2 Var(pc_1)"),
    ("abs_m", "l2_var", "L^2 Var(|m|), the same frames"),
)


def main() -> None:
    out, training, step = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    sizes = [int(a) for a in sys.argv[4:]] or [16, 32]
    tables = {
        "trained": f"{training}-step{step:07d}",
        "twin": f"{training}-step0000000",
    }
    plt.rcParams.update({"font.size": 9, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(2, len(PANELS), figsize=(14, 6.4), sharex=True)
    for row, (who, table) in enumerate(tables.items()):
        for col, (column, key, title) in enumerate(PANELS):
            ax = axes[row][col]
            for L in sizes:
                static = json.loads((out / table / f"L{L}" / "static.json").read_text("utf-8"))
                curve = static["equilibrium"][column]
                eps, value = curve["eps_L"], curve[key]
                lo, hi = zip(*curve[f"{key}_ci"], strict=True)
                ax.fill_between(eps, lo, hi, color=COLOURS[L], alpha=0.25, lw=0)
                ax.plot(eps, value, color=COLOURS[L], lw=1.5, label=f"L = {L}")
            ax.axvline(0.0, color=INK2, lw=0.8, ls=":")
            ax.set_xlim(8.7, -5.5)  # hot on the left: the way a ramp cools
            if key == "l2_var":
                ax.set_yscale("log")
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_title(f"{title}, {who}", loc="left", color=INK, fontsize=9.5)
            if row == 1:
                ax.set_xlabel("eps_L = L (T - Tc) / Tc, cooling to the right")
            if col == 0:
                ax.legend(frameon=False)
    fig.suptitle(
        f"{training} at step {step} beside its step-0 twin, seeds 1-5: exploratory, 95% CIs from a "
        "joint hierarchical bootstrap",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = out / f"latent-curves-{training}.png"
    fig.savefig(png, dpi=140, facecolor=SURFACE)
    print(png)


if __name__ == "__main__":
    main()
