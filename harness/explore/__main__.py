"""The exploratory pipeline's one door. NOT pre-registered: nothing here is logged as a trial, and
seeds 6-10 are refused (common.py).

  uv run --no-sync python harness/explore <script> [options]
  uv run --no-sync python harness/explore <script> --help

Scripts, in the order a pass runs them (latent_pass runs them all for one training):
  join_pilot      a data dir holding a Kaggle output's latents tables beside the local runs
  static_tc       values at Tc, and the equilibrium curves against T, from the staircases
  scaling         exponents over sizes from static_tc's outputs
  auc_alarms      AUC of each pair, every cell of the sensitivity grid
  peaks           where each indicator peaks along each slow ramp, and 1/nu over sizes
  correlations    pc_k against |m| and e, frame by frame and across plateaus (--latents)
  probes_explore  ridge probes from the pcs, their control tasks, the floors (--latents)
  nulls_explore   the fixed-T null runs alone
  surrogates_cal  the surrogate tests on the real nulls
  alarms_explore  the alarm threshold examined
  fast_ramps      ensemble indicators on the ramps, the Kibble-Zurek lag
  fixed_eps       slopes over sizes at fixed eps_L, from three sources
  report          one size's outputs as markdown
  latent_report   a training's trained and twin side by side, as markdown (--latents)
  latent_pass     every step above for one training, in order, trained and twin
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

SCRIPTS = (
    "join_pilot",
    "static_tc",
    "scaling",
    "auc_alarms",
    "peaks",
    "correlations",
    "probes_explore",
    "nulls_explore",
    "surrogates_cal",
    "alarms_explore",
    "fast_ramps",
    "fixed_eps",
    "report",
    "latent_report",
    "latent_pass",
)


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in SCRIPTS:
        raise SystemExit(__doc__)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # harness/, as pytest has it
    importlib.import_module(f"explore.{argv[0]}").main(argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
