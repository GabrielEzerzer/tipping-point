"""The figures of a latent experiment's result (#28, slice 5), drawn from its csv/ tables alone:

  scaling.png   L^2 Var at Tc against L: |m|, pc_1 trained and twin, surprise; kappa with its CI
  peaks.png     eps_L of each judged statistic's extremum per L, trained and twin, with 1/nu
  curves.png    mean surprise and L^2 Var(pc_1) against T per L, the classic chi(T) overlaid
  collapse.png  L^(-7/4) L^2 Var against eps_L across L: the classic |m|, pc_1 trained and twin
  probes.png    the probe table, trained beside twin, and the judged margins

The primary checkpoints alone: an ablation's rows are in the tables. A PCA basis is per checkpoint,
so a latent magnitude does not compare across checkpoints: pc_1 and surprise variances are divided
by a per-checkpoint number, the same across L, before anything is drawn together. A figure holds
nothing but what the tables hold, so the same tables draw the same bytes.

matplotlib is not a project dependency, so this runs in a throwaway environment and imports nothing
of the project (harness/ising_experiment.py runs it, and prints the command when it could not):

  uv run --isolated --no-project --python 3.13 --with matplotlib==3.10.3 python \
      harness/ising_figures.py <result folder>
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib  # pyright: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
TRAINED, TWIN, CLASSIC = "#2a78d6", "#eb6834", "#0b0b0b"
TC_TOLERANCE = 1e-3  # eps_L of the plateau nearest Tc, at any L


def read(folder: Path, name: str) -> list[dict[str, str]]:
    with open(folder / "csv" / f"{name}.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def number(text: str) -> float:
    return float(text) if text else math.nan


def save(fig: plt.Figure, folder: Path, name: str) -> None:
    fig.savefig(folder / "figures" / name, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


class Result:
    """The tables, and which checkpoints are the primary ones' trained and twin arms."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.fss = read(folder, "fss")
        self.curves = read(folder, "curves")
        self.peaks = read(folder, "peaks")
        self.probes = read(folder, "probes")
        self.margins = read(folder, "margins")
        self.baseline = read(folder, "baseline")
        self.arm = {r["checkpoint"]: r["arm"] for r in self.probes if r["role"] == "primary"}
        self.sizes = sorted({int(r["L"]) for r in self.curves})
        stamp = json.loads((folder / "manifest.json").read_text("utf-8"))["params"]["prereg"]
        frozen = stamp["tag"] or "not frozen: planted data"
        self.label = f"{stamp['id']}-v{stamp['version']} ({frozen})"

    def names(self, arm: str) -> list[str]:
        return sorted(name for name, a in self.arm.items() if a == arm)

    def curve(self, name: str, column: str, L: int, field: str) -> list[tuple[float, float, float]]:
        """(temperature, eps_L, value) along a checkpoint's plateaus at L."""
        return sorted(
            (float(r["temperature"]), float(r["eps_L"]), float(r[field]))
            for r in self.curves
            if (r["checkpoint"], r["observable"], int(r["L"])) == (name, column, L)
        )

    def at_tc(self, name: str, column: str, L: int) -> float:
        points = self.curve(name, column, L, "l2_var")
        return min(points, key=lambda p: abs(p[1]))[2]

    def kappa(self, names: list[str], quantity: str) -> str:
        """kappa over checkpoints: the mean of those that have one, how many, and their CIs'
        span."""
        rows = [r for r in self.fss if r["checkpoint"] in names and r["quantity"] == quantity]
        values = [number(r["estimate"]) for r in rows]
        values = [v for v in values if math.isfinite(v)]
        ends = [v for r in rows for v in (number(r["lo"]), number(r["hi"])) if math.isfinite(v)]
        if not values or not ends:
            return f"none of {len(rows)}"
        mean = sum(values) / len(values)
        count = "" if len(values) == len(rows) else f", {len(values)} of {len(rows)}"
        return f"{mean:+.2f} [{min(ends):+.2f}, {max(ends):+.2f}]{count}"


def scaling(result: Result) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sizes = result.sizes

    def line(names: list[str], column: str, color: str, style_: str, label: str) -> None:
        for k, name in enumerate(names):
            first = result.at_tc(name, column, sizes[0])
            values = [result.at_tc(name, column, L) / first for L in sizes]
            ax.plot(
                sizes, values, style_, color=color, lw=1.5, ms=5, label=label if k == 0 else None
            )

    classic = [r for r in result.baseline if r["quantity"] == "l2_var_abs_m"]
    kappa = classic[0] if classic else None
    label = "|m|, κ = no row"
    if kappa:
        label = (
            f"|m|, κ = {float(kappa['estimate']):+.2f} "
            f"[{float(kappa['lo']):+.2f}, {float(kappa['hi']):+.2f}]"
        )
    trained, twin = result.names("trained"), result.names("twin")
    line(
        trained, "pc_1", TRAINED, "o-", f"pc_1 trained, κ = {result.kappa(trained, 'l2_var_pc_1')}"
    )
    line(twin, "pc_1", TWIN, "o-", f"pc_1 twin, κ = {result.kappa(twin, 'l2_var_pc_1')}")
    for names, color, arm in ((trained, TRAINED, "trained"), (twin, TWIN, "twin")):
        kappa_text = result.kappa(names, "l2_var_surprise")
        line(names, "surprise", color, "s:", f"surprise {arm}, κ = {kappa_text}")
    line([""], "abs_m", CLASSIC, "o--", label)  # last: pc_1 reads |m| and would hide it
    reference = [(L / sizes[0]) ** 1.75 for L in sizes]
    ax.plot(sizes, reference, "--", color=MUTED, lw=1, label="slope 7/4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(L) for L in sizes])
    ax.minorticks_off()
    ax.set_xlabel("L")
    ax.set_ylabel(f"L²·Var at Tc, relative to L = {sizes[0]}")
    ax.set_title(f"Static scaling at Tc: {result.label}", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    save(fig, result.folder, "scaling.png")


def peaks(result: Result) -> None:
    keys = sorted(r["quantity"].removeprefix("inverse_nu_") for r in result.fss
                  if r["quantity"].startswith("inverse_nu_") and r["checkpoint"] in result.arm
                  and r["role"].startswith("dynamic-"))  # fmt: skip
    keys = sorted(set(keys))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2), squeeze=False)
    for ax, key in zip(axes[0], keys, strict=True):
        for arm, color, nudge in (("trained", TRAINED, 0.97), ("twin", TWIN, 1.03)):
            names = result.names(arm)
            for name in names:
                rows = sorted(
                    (int(r["L"]), number(r["peak_eps_median"]), number(r["peak_eps_q1"]),
                     number(r["peak_eps_q3"]))
                    for r in result.peaks
                    if r["checkpoint"] == name and f"{r['observable']}_{r['indicator']}" == key
                )  # fmt: skip
                sizes = [row[0] * nudge for row in rows]
                medians = [row[1] for row in rows]
                low = [abs(row[1] - row[2]) for row in rows]
                high = [abs(row[3] - row[1]) for row in rows]
                ax.errorbar(sizes, medians, yerr=[low, high], fmt="o-", color=color, ms=4, lw=1,
                            alpha=0.8)  # fmt: skip
            nu = result.kappa(names, f"inverse_nu_{key}")
            ax.plot([], [], "o-", color=color, label=f"{arm}, 1/\u03bd = {nu}")
        ax.axhline(0.0, color=MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(result.sizes, [str(L) for L in result.sizes])
        ax.minorticks_off()
        ax.set_xlabel("L")
        ax.set_ylabel("εL at the extremum, as read")
        ax.set_title(key, loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8)
        style(ax)
    fig.suptitle(f"Peak positions: {result.label}", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, result.folder, "peaks.png")


def curves(result: Result) -> None:
    sizes = result.sizes
    fig, axes = plt.subplots(2, len(sizes), figsize=(4.2 * len(sizes), 6.4), squeeze=False)
    for j, L in enumerate(sizes):
        top, bottom = axes[0][j], axes[1][j]
        for arm, color in (("trained", TRAINED), ("twin", TWIN)):
            for k, name in enumerate(result.names(arm)):
                label = arm if k == 0 else None
                points = result.curve(name, "surprise", L, "mean")
                top.plot(
                    [p[0] for p in points], [p[2] for p in points], color=color, lw=1, label=label
                )
                points = result.curve(name, "pc_1", L, "l2_var")
                peak = max(p[2] for p in points)
                bottom.plot([p[0] for p in points], [p[2] / peak for p in points], color=color,
                            lw=1, label=label)  # fmt: skip
        points = result.curve("", "abs_m", L, "l2_var")
        peak = max(p[2] for p in points)
        relative = [p[2] / peak for p in points]
        bottom.plot(
            [p[0] for p in points], relative, "--", color=CLASSIC, lw=1.5, label="classic χ(T)"
        )
        top.set_title(f"L = {L}", loc="left", fontsize=10, color=INK)
        top.set_ylabel("mean surprise")
        bottom.set_ylabel("L²·Var, relative to its maximum")
        bottom.set_xlabel("T")
        for ax in (top, bottom):
            style(ax)
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Against temperature: {result.label}", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, result.folder, "curves.png")


def collapse(result: Result) -> None:
    panels = (("classic |m|", [""], "abs_m"), ("pc_1 trained", result.names("trained"), "pc_1"),
              ("pc_1 twin", result.names("twin"), "pc_1"))  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    shades = [TRAINED, TWIN, CLASSIC, MUTED]
    for ax, (title, names, column) in zip(axes, panels, strict=True):
        pooled: dict[int, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        for name in names:
            rows = {L: result.curve(name, column, L, "l2_var") for L in result.sizes}
            scale = max(p[2] * L**-1.75 for L, points in rows.items() for p in points)
            for L, points in rows.items():
                for _, eps, value in points:
                    pooled[L][eps].append(value * L**-1.75 / scale)
        for k, L in enumerate(result.sizes):
            eps = sorted(pooled[L])
            mean = [sum(pooled[L][e]) / len(pooled[L][e]) for e in eps]
            ax.plot(eps, mean, "o-", color=shades[k % len(shades)], ms=3, lw=1.2, label=f"L = {L}")
        ax.set_xlim(-12, 12)
        ax.set_xlabel("εL")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8)
        style(ax)
    axes[0].set_ylabel("L^(\u22127/4)·L²·Var, relative to its maximum")
    fig.suptitle(f"Collapse: {result.label}", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, result.folder, "collapse.png")


def probe_table(result: Result) -> None:
    """R^2 of the raw target and its selectivity, the mean over primary checkpoints, per arm."""
    values: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in result.probes:
        if r["role"] == "primary":
            key = r["target"], int(r["L"]), r["arm"]
            values[key][r["column"]].append(number(r["r2"]))
            if r["column"] == "raw":
                values[key]["selectivity"].append(number(r["selectivity"]))

    def cell(key: tuple[str, int, str], column: str) -> str:
        numbers = values[key][column]
        return f"{sum(numbers) / len(numbers):.3f}" if numbers else "-"

    header = ["target", "L", "R² trained", "R² twin", "selectivity trained", "selectivity twin",
              "iid trained", "shuffle trained"]  # fmt: skip
    rows = []
    for target in sorted({key[0] for key in values}):
        for L in result.sizes:
            rows.append(
                [
                    target,
                    str(L),
                    cell((target, L, "trained"), "raw"),
                    cell((target, L, "twin"), "raw"),
                    cell((target, L, "trained"), "selectivity"),
                    cell((target, L, "twin"), "selectivity"),
                    cell((target, L, "trained"), "iid"),
                    cell((target, L, "trained"), "shuffle_within_T"),
                ]
            )
    judged = [r for r in result.margins if r["judged"] == "True"]
    lines = [
        f"{r['quantity']}: margin {number(r['margin']):+.3f} "
        f"[{number(r['lo']):+.3f}, {number(r['hi']):+.3f}], {r['verdict']}"
        for r in judged
    ]
    table_height, note_height = 0.24 * (len(rows) + 1), 0.18 * len(lines) + 0.15
    height = table_height + note_height + 0.4
    fig = plt.figure(figsize=(11, height))
    ax = fig.add_axes((0.02, note_height / height, 0.96, table_height / height))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header, bbox=(0, 0, 1, 1), cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, _), box in table.get_celld().items():
        box.set_edgecolor(AXIS)
        box.set_facecolor(SURFACE if row else GRID)
    ax.set_title(f"Probes, primary checkpoints: {result.label}", loc="left", fontsize=10, color=INK)
    fig.text(0.02, 0.05 / height, "\n".join(lines), fontsize=8, color=INK2, va="bottom")
    save(fig, result.folder, "probes.png")


def main(folder: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",  # the one matplotlib ships: the same bytes on any machine
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
        }
    )
    (folder / "figures").mkdir(exist_ok=True)
    result = Result(folder)
    for draw in (scaling, peaks, curves, collapse, probe_table):
        draw(result)
        print(draw.__name__, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
