"""Does the fixed-eps_L scaling seen in the null runs repeat in independent data (exploratory)? At
fixed eps_L, FSS has L^2 Var(|m|) ~ L^(7/4) and <|m|> ~ L^(-1/8) up to corrections; the null runs
gave smaller slopes off Tc (nulls_explore.py). The same slopes, of the column given, log2 of the
second size's value over the first's at eps_L = 2, 4, 6, are read from three sources:
  nulls       the fixed-T null runs (700 tau_c each, 5 seeds x 16 replicas)
  staircases  the equilibrium curve over the staircase plateaus (static.json), interpolated linearly
              in eps_L (plateau grids 0.36 and 0.72 eps_L apart), on log values for |m| and L^2 Var
  slow ramps  the ensemble values at each eps_L on the slow ramps (fast_ramps.json), which track
              equilibrium at x = 0.02
and set beside the slopes at Tc (scaling.json).
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import numpy as np

from explore import common

POINTS = ("2", "4", "6")


def _read(args: Any, L: int, name: str) -> dict[str, Any] | None:
    path = common.out_dir(args, L) / name
    return json.loads(path.read_text("utf-8")) if path.is_file() else None


def curve_at(c: dict[str, Any], e: float, key: str) -> float:
    x = np.array(c["eps_L"])[::-1]
    y = np.array(c[key])[::-1]
    return float(np.interp(e, x, y))


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    p = common.parser(__doc__)
    p.add_argument("--sizes", type=int, nargs=2, default=[16, 32])
    p.add_argument("--column", default=None, help="abs_m, or surprise or pc_1 with --latents")
    args = p.parse_args(argv)
    column = args.column or ("pc_1" if args.latents else "abs_m")
    a, b = args.sizes
    rows: dict[str, Any] = {}
    for source in ("nulls", "staircases", "slow ramps"):
        values: dict[int, dict[str, tuple[float, float]]] = {}
        for L in (a, b):
            if source == "nulls":
                d = _read(args, L, "nulls.json")
                if d is None:
                    break
                per = d["columns"][column]["per_eps"]
                values[L] = {
                    e: (per[e]["static"]["l2_var"]["value"], per[e]["static"]["mean"]["value"])
                    for e in POINTS
                }
            elif source == "staircases":
                d = _read(args, L, "static.json")
                if d is None:
                    break
                c = d["equilibrium"][column]
                values[L] = {
                    e: (curve_at(c, float(e), "l2_var"), curve_at(c, float(e), "mean"))
                    for e in POINTS
                }
            else:
                d = _read(args, L, "fast_ramps.json")
                if d is None:
                    break
                at = d["columns"][column]["speeds"]["slow"]["at_eps_L"]
                values[L] = {e: (at[e]["l2_var"], at[e]["mean"]) for e in POINTS}
        if len(values) < 2:
            continue

        def slope(first: float, second: float) -> float:
            return math.log(abs(second / first)) / math.log(b / a)

        rows[source] = {
            e: {
                "l2_var": (
                    values[a][e][0],
                    values[b][e][0],
                    slope(values[a][e][0], values[b][e][0]),
                ),
                "mean": (values[a][e][1], values[b][e][1], slope(values[a][e][1], values[b][e][1])),
            }
            for e in POINTS
        }
    scaling = json.loads((common.table_dir(args) / "scaling.json").read_text("utf-8"))
    tc = {r["quantity"]: r["kappa"]["value"] for r in scaling["rows"]}
    out = {
        "column": column,
        "sizes": [a, b],
        "rows": rows,
        "at_tc": {"l2_var": tc[f"l2_var_{column}"], "mean": tc[f"mean_{column}"]},
    }
    common.save(common.table_dir(args) / f"fixed-eps-{column}.json", out)
    print("| source | " + " | ".join(f"eps_L = {e}: L^2 Var, mean" for e in POINTS) + " |")
    print("|---|" + "---|" * len(POINTS))
    for source, per in rows.items():
        cells = [f"{per[e]['l2_var'][2]:+.2f}, {per[e]['mean'][2]:+.3f}" for e in POINTS]
        print(f"| {source} | " + " | ".join(cells) + " |")
    print(f"| at Tc (staircases) | {out['at_tc']['l2_var']:+.2f}, {out['at_tc']['mean']:+.3f} |")
