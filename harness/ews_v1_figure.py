"""The figure of ising-ews-v1's result in the README, paper/figures/ising-ews-v1.png, drawn from the
result folder's fss, peaks and auc tables alone:

  left   each judged exponent minus its exact 2D Ising value, with its 95% CI, and the
         pre-registered tolerance around the exact value as a grey band. 1 - AC1 has no band: its
         fss row names no expected value, and its verdict asks only that the CI exclude 0
         (fss.excludes). It is drawn against -gamma/nu = -7/4, the value its hypothesis says it
         tracks.
  right  the median eps_L at which each replica's rolling Var(|m|) peaks, per L, on the dense and
         the matched cadence, with interquartile ranges and the 1/nu fitted on each. A flat line
         in L is finite-size scaling, 1/nu = 1.

A line under the panels gives the AUC of the classic indicators of |m| at the headline cell, dense
cadence, over every L and h: the numbers hypotheses 2 and 3 judge. A missing table, column or row
stops the script with its name, rather than leaving a panel without it; so does a headline cell
missing at an L and h the auc table scores elsewhere, rather than a line that says every L and h.

matplotlib is not a project dependency and pyarrow is, so this runs in the project's environment
with matplotlib layered on (3.10.3, the version harness/ising_figures.py pins, drew the committed
figure):

  uv run --no-sync --with matplotlib==3.10.3 python harness/ews_v1_figure.py \
      <result folder> paper/figures/ising-ews-v1.png
"""

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib  # pyright: ignore[reportMissingImports]
import pyarrow as pa
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

Row = dict[str, Any]

NU, MINUS = "\N{GREEK SMALL LETTER NU}", "\N{MINUS SIGN}"
TABLE_FILE = f"{0:020d}.parquet"  # an ews result writes each table as one chunk
COLUMNS = {
    "fss": ("quantity", "cadence", "estimate", "lo", "hi", "expected", "tolerance", "verdict",
            "checkpoint"),
    "peaks": ("indicator", "observable", "cadence", "L", "peak_eps_median", "peak_eps_q1",
              "peak_eps_q3", "checkpoint"),
    "auc": ("indicator", "observable", "cadence", "L", "h", "headline", "window_taus",
            "bandwidth_taus", "auc", "checkpoint"),
}  # fmt: skip
# The five judged exponents, top to bottom: fss quantity, cadence, label, the exact value, as text.
EXPONENTS = (
    ("l2_var_abs_m", "dense", "L² Var|m|", 7 / 4, "7/4"),
    ("mean_abs_m", "dense", "mean |m|", -1 / 8, f"{MINUS}1/8"),
    ("skewness_abs_m", "dense", "skewness of |m|", 0.0, "0"),
    ("one_minus_ac1", "matched", f"1 {MINUS} AC1 of |m|", -7 / 4, f"{MINUS}7/4"),
    ("inverse_nu_variance", "dense", f"1/{NU} from the peaks", 1.0, "1"),
)
CADENCES = (("dense", "C0", 0.97), ("matched", "C1", 1.03))  # colour, and a nudge along L
AUC_INDICATORS = (("variance", "Var"), ("ac", "lag-1 AC"), ("skewness", "skewness"))


def read(folder: Path, name: str) -> pa.Table:
    path = folder / name / TABLE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {name} table in {folder} (looked for {path})")
    return pq.read_table(path)


def rows(table: pa.Table, name: str) -> list[Row]:
    missing = [column for column in COLUMNS[name] if column not in table.column_names]
    if missing:
        raise ValueError(f"the {name} table has no column {', '.join(missing)}")
    return table.to_pylist()


def where(table: list[Row], **match: Any) -> list[Row]:
    return [r for r in table if all(r[key] == value for key, value in match.items())]


def one(table: list[Row], name: str, **match: Any) -> Row:
    found = where(table, **match)
    if len(found) != 1:
        raise ValueError(f"the {name} table has {len(found)} rows with {match}, not one")
    return found[0]


def finite(value: Any) -> bool:
    return value is not None and math.isfinite(value)


def exponents(ax: plt.Axes, fss: list[Row]) -> None:
    """Each judged exponent minus its exact value, with its CI, its tolerance band and verdict."""
    ys = list(range(len(EXPONENTS) - 1, -1, -1))
    reach, labels, banded = 0.0, [], False
    for y, (quantity, cadence, label, exact, exact_text) in zip(ys, EXPONENTS, strict=True):
        row = one(fss, "fss", quantity=quantity, cadence=cadence, checkpoint="")
        verdict = row["verdict"] or "no verdict"
        if finite(row["expected"]):  # judged against an expected value: its tolerance is the band
            if not math.isclose(row["expected"], exact, abs_tol=1e-9):
                raise ValueError(f"fss {quantity} is judged against {row['expected']}, not {exact}")
            if not finite(row["tolerance"]):
                raise ValueError(f"fss {quantity} names an expected value but no tolerance")
            tolerance = row["tolerance"]
            ax.barh(y, 2 * tolerance, left=-tolerance, height=0.5, color="0.85",
                    label=None if banded else "pre-registered tolerance")  # fmt: skip
            reach, banded = max(reach, tolerance), True
        else:  # the reference row: the CI must exclude 0, a question of sign, so no band
            verdict = f"{verdict} (judged on its sign)"
        estimate, lo, hi = (row[key] - exact for key in ("estimate", "lo", "hi"))
        ax.plot([lo, hi], [y, y], "|-", color="k", ms=8, label="95% CI" if y == ys[0] else None)
        ax.plot(estimate, y, "o", color="k", ms=4)
        ax.text(0.99, y, verdict, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8)  # fmt: skip
        reach = max(reach, abs(lo), abs(hi))
        labels.append(f"{label} vs {exact_text}")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlim(-1.3 * reach, 1.3 * reach)
    ax.set_ylim(-0.6, len(EXPONENTS) + 0.2)  # a strip above the top row for the legend
    ax.set_yticks(ys, labels)
    ax.set_xlabel("fitted exponent minus the exact value")
    ax.set_title("Fitted exponents against the exact 2D Ising values")
    ax.legend(loc="upper center", ncols=2, fontsize=8)


def peak_positions(ax: plt.Axes, peaks: list[Row], fss: list[Row]) -> None:
    """The median eps_L of the rolling Var(|m|) peak per L and cadence, with its fitted 1/nu."""
    sizes: set[int] = set()
    ends = [0.0]  # every quartile drawn, and Tc
    for cadence, colour, nudge in CADENCES:
        series = sorted(
            where(peaks, indicator="variance", observable="abs_m", cadence=cadence, checkpoint=""),
            key=lambda r: r["L"],
        )
        found = [r["L"] for r in series]
        if not series or len(set(found)) != len(found):
            raise ValueError(f"the peaks table needs one Var|m| row per L on {cadence}: {found}")
        fit = one(fss, "fss", quantity="inverse_nu_variance", cadence=cadence, checkpoint="")
        medians = [r["peak_eps_median"] for r in series]
        below = [m - r["peak_eps_q1"] for m, r in zip(medians, series, strict=True)]
        above = [r["peak_eps_q3"] - m for m, r in zip(medians, series, strict=True)]
        label = (
            f"{cadence}: 1/{NU} = {fit['estimate']:.3f} [{fit['lo']:.3f}, {fit['hi']:.3f}], "
            f"{fit['verdict'] or 'no verdict'}"
        )
        ax.errorbar([L * nudge for L in found], medians, yerr=[below, above], fmt="o-",
                    color=colour, ms=4, capsize=3, label=label)  # fmt: skip
        sizes |= set(found)
        ends += [r[q] for r in series for q in ("peak_eps_q1", "peak_eps_q3") if finite(r[q])]
    span = (max(ends) - min(ends)) or 1.0
    ax.set_ylim(min(ends) - 0.08 * span, max(ends) + 0.4 * span)  # headroom for the legend
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--", label="Tc")
    ax.set_xscale("log")
    ax.set_xticks(sorted(sizes), [str(L) for L in sorted(sizes)])
    ax.minorticks_off()
    ax.set_xlabel(f"L; a flat line means finite-size scaling (1/{NU} = 1)")
    ax.set_ylabel("εL of the rolling Var|m| peak (median, IQR)")
    ax.set_title("Where the variance peaks on the slow ramp")
    ax.legend(loc="upper left", fontsize=8)


def headline_auc(auc: list[Row]) -> str:
    """Each classic indicator's AUC range at the headline cell, |m| on the dense cadence."""
    cell = where(auc, headline=True, observable="abs_m", cadence="dense", checkpoint="")
    pairs = {(r["window_taus"], r["bandwidth_taus"]) for r in cell}
    if len(pairs) != 1:
        raise ValueError(f"the auc table's headline rows of |m|, dense, hold cells {sorted(pairs)}")
    ((window, bandwidth),) = pairs
    # every cell is scored at every L and h, so the table's own rows name the grid the footer claims
    scored = where(auc, observable="abs_m", cadence="dense", checkpoint="")
    grid = sorted({(r["L"], r["h"]) for r in scored})
    parts = []
    for indicator, label in AUC_INDICATORS:
        held = sorted((r["L"], r["h"]) for r in cell if r["indicator"] == indicator)
        if held != grid:
            raise ValueError(
                f"the auc table's headline {indicator} rows of |m|, dense, hold (L, h) {held}, "
                f"not one each of {grid}"
            )
        values = [r["auc"] for r in cell if r["indicator"] == indicator]
        low, high = f"{min(values):.3f}", f"{max(values):.3f}"
        parts.append(f"{label} {low}" if low == high else f"{label} {low} to {high}")
    return (
        f"AUC against the fixed-T nulls at the headline cell (window {window:g} τc, bandwidth "
        f"{bandwidth:g} τc), dense cadence, |m|, every L and h: {', '.join(parts)}."
    )


def draw(fss: pa.Table, peaks: pa.Table, auc: pa.Table, out: Path) -> None:
    """The two panels and the AUC line, written to `out` as a PNG of 10 by 4 inches at 150 dpi."""
    fss_rows, peak_rows, auc_rows = rows(fss, "fss"), rows(peaks, "peaks"), rows(auc, "auc")
    footer = headline_auc(auc_rows)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    try:
        exponents(left, fss_rows)
        peak_positions(right, peak_rows, fss_rows)
        fig.text(0.01, 0.015, footer, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, metadata={"Software": None})  # no version stamp in the bytes
    finally:
        plt.close(fig)


def main(folder: Path, out: Path) -> None:
    draw(read(folder, "fss"), read(folder, "peaks"), read(folder, "auc"), out)
    print(out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: ews_v1_figure.py <result folder> <out png>")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
