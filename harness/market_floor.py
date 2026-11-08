"""The circularity floor of a labels run (#29 S5, #111): GARCH(1,1)-t paths run through the run's
own labeller (its manifest's parameters) and scored by the classic variance and AC1 pipelines with
scoring.auc, written as a report into the run's reports/ folder (market/reports.py). It lives
outside market/ because no module there may import the feature code
(python/tipping_point/tests/market/test_circularity.py: the labels cannot see the features).

The GARCH is a minute process, which nulls.simulate bridges to 10 seconds. --garch SYMBOL takes
that symbol's fit on the 2023 pilot year, frozen in PILOT_GARCH, and writes the report floor-SYMBOL;
--omega, --alpha, --beta and --nu together give any other minute GARCH, written as the report floor.
--days counts the simulated days from nulls.FIRST (2022-07-04); they must end before 2024-01-01,
and a path no longer than the labeller's warm-up has no labels.

  uv run python harness/market_floor.py --run <labels run folder> --garch BTCUSDT --days 546
  uv run python harness/market_floor.py --run <labels run folder> \\
      --omega <omega> --alpha <alpha> --beta <beta> --nu <nu> --days <days>
"""

from __future__ import annotations

import argparse
import dataclasses
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
HORIZONS = (0, 3, 5, 6, 12, 24)  # 5: market-labels-v1's horizon.smallest, market-ews's first
PATHS = 30  # about 30 paths bring the floor's standard error near 0.017 (#111)
# Per minute, from harness/market_garch.py on 2023-01-01..2023-12-31 (the report garch of labels
# run 20261106T182828Z-binance-20220701-20231231-dmin36), recorded in market-labels-v1's
# disclosure as the pilot fit; #30's pre-registration chooses the floor (#29 S5)
PILOT_GARCH = {
    "BTCUSDT": nulls.Garch(
        8.651445052105347e-11, 0.07110265726317705, 0.9288274749829264, 5.085328354482213
    ),
    "ETHUSDT": nulls.Garch(
        6.419879719145039e-10, 0.0801166904491908, 0.9197468505253973, 4.97264822567892
    ),
}
GIVEN = ("omega", "alpha", "beta", "nu")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="the labels run's folder")
    parser.add_argument("--garch", choices=sorted(PILOT_GARCH), help="a frozen pilot-year fit")
    for name in GIVEN:
        parser.add_argument(f"--{name}", type=float, help="per minute, with the other three")
    parser.add_argument("--days", type=int, required=True, help="simulated days per path")
    parser.add_argument("--paths", type=int, default=PATHS)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args(argv)
    given = [getattr(args, name) for name in GIVEN]
    if not (given == [None] * len(GIVEN) if args.garch else None not in given):
        parser.error("give --garch, or all four of --omega, --alpha, --beta and --nu")
    name = f"floor-{args.garch}" if args.garch else "floor"
    folder = args.run / "reports" / name
    if folder.exists():
        sys.exit(f"error: {folder} exists")
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if "labeller" not in manifest["params"]:
        sys.exit(f"error: {args.run} is not a labels run")
    params = msvol.Params(**manifest["params"]["labeller"])
    garch = PILOT_GARCH[args.garch] if args.garch else nulls.Garch(*given)
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
    settings = {"seed": args.seed, "paths": args.paths, "days": args.days, **SCORERS}
    report = dataclasses.replace(reports.floor(found, garch, settings), name=name)
    print(reports.write(report, args.run / "reports"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
