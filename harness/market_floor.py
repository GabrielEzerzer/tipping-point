"""The circularity floor of a labels run (#29 S5, #111): GARCH(1,1)-t paths run through the run's
own labeller (its manifest's parameters) and scored by the classic variance and AC1 pipelines with
scoring.auc, written as the report `floor` into the run's reports/ folder (market/reports.py). It
lives outside market/ because no module there may import the feature code
(python/tipping_point/tests/market/test_circularity.py: the labels cannot see the features).

The GARCH parameters have no default: their fit on the 2023 pilot year belongs to S5. --days counts
the simulated days from nulls.FIRST (2022-07-04); they must end before 2024-01-01, and a path no
longer than the labeller's warm-up has no labels.

  uv run python harness/market_floor.py --run <labels run folder> \\
      --omega <omega> --alpha <alpha> --beta <beta> --nu <nu> --days <days>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tipping_point.ews import scoring
from tipping_point.ews.trend import Pipeline
from tipping_point.market import msvol, nulls, reports

# shortcut: the floor tests' 1-hour windows over a 4-hour segment and horizons up to 2 hours, read
# them from #30's pre-registration once it is frozen
SCORERS = {"variance": Pipeline("variance", 60, 240), "ac": Pipeline("ac", 60, 240)}
HORIZONS = (0, 3, 6, 12, 24)
PATHS = 30  # about 30 paths bring the floor's standard error near 0.017 (#111)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="the labels run's folder")
    for name in ("omega", "alpha", "beta", "nu"):
        parser.add_argument(f"--{name}", type=float, required=True)
    parser.add_argument("--days", type=int, required=True, help="simulated days per path")
    parser.add_argument("--paths", type=int, default=PATHS)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args(argv)
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    params = msvol.Params(**manifest["params"]["labeller"])
    garch = nulls.Garch(args.omega, args.alpha, args.beta, args.nu)
    found = nulls.circularity_floor(
        garch,
        SCORERS,
        scoring.auc,
        HORIZONS,
        days=args.days,
        paths=args.paths,
        seed=args.seed,
        params=params,
    )
    print(reports.write(reports.floor(found, garch), args.run / "reports"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
