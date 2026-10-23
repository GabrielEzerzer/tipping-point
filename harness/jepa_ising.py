"""Tiny simulator runs for the JEPA harness, written by the simulator's own run path (no planted
values: real spins, real Parquet, real manifests), and a small model trained on two of them.

At L = 16 with 4 replicas, each staircase holds three plateaus (3.5, Tc + 4.5e-5 inside the
held-out band, and 1.5) of four bursts each; seeds 101 and 102 train and validate, seed 1 is a
pre-registered seed (scored, never trained on), and a ramp and a fixed (null) run complete the set
of kinds training must refuse. Everything runs on the CPU in seconds.
"""

from dataclasses import dataclass
from pathlib import Path

from tipping_point.ising import protocol, run
from tipping_point.jepa import data, train
from tipping_point.jepa.lattice import ModelSpec

FIXTURES = Path(__file__).resolve().parents[1] / "python" / "tipping_point" / "tests" / "fixtures"
PILOT = FIXTURES / "ising_pilot.json"
L, REPLICAS = 16, 4
PLATEAUS = (3.5, 2.269231, 1.5)
CONFIG = train.Config(
    steps=8,
    model=ModelSpec(dim=32, pred_width=32),
    batch_sites=16 * L * L,
    warmup=2,
    seed=3,
    checkpoint_every=4,
    monitor_every=4,
    monitor_frames=16,
    pca_frames=128,
)


def staircase() -> protocol.Staircase:
    return protocol.Staircase([2.0] * 3, PLATEAUS, warmup=50, min_therm=50, min_meas=200, saves=4)


@dataclass(frozen=True)
class Ising:
    root: Path
    runs: dict[str, Path]  # role -> simulator run folder
    training: Path  # a finished training on the "train" run, validated on "val"


def build(root: Path) -> Ising:
    plans: dict[str, tuple[int, protocol.Schedule]] = {
        "train": (101, staircase()),
        "val": (102, staircase()),
        "scored": (1, staircase()),
        "ramp": (101, protocol.Ramp(rate=100, L=L, pre=50, hold=50, save_dT=0.05)),
        "null": (103, protocol.Fixed((2.5,), therm=50, sweeps=100, saves=2)),
    }
    runs = {
        role: run.start(run.Spec(L=L, replicas=REPLICAS, seed=seed), schedule, root, role, "cpu")
        for role, (seed, schedule) in plans.items()
    }
    dataset = data.load(root, [runs["train"].name], [runs["val"].name])
    pilot = train.pilot_record(PILOT)
    training = train.start(CONFIG, dataset, root, "tiny", "cpu", pilot=pilot)
    return Ising(root, runs, training)
