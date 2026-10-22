"""The simulation loop: one run = one schedule, one batch of replicas, one generator, one folder.

The loop never waits on the device: records stay there until a flush. Every flush is followed by a
checkpoint (spins, generator state, step), so a resumed run continues bit for bit where a straight
run would be (Gate D). The unit of reproducibility is the run on one device type and torch version.
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from torch import Tensor

from .. import __version__
from . import gates, io, kernel, protocol
from .exact import TC

CONFIG_BUFFER_BYTES = 2**28  # flush before ~256 MB of configurations pile up on the device
MAX_LATTICES = 2**16  # replica ids are stored as uint16
VALIDATION_TEMPERATURES = (1.5, TC, 3.5)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Spec:
    """Everything that defines a run besides its schedule."""

    L: int
    replicas: int  # per temperature block
    seed: int
    init: str = "hot"  # hot: random spins; cold: all +1
    tally: bool = False  # count moves per dE for Gate C (validation runs)
    checkpoint_every: int = 100_000


def start(
    spec: Spec, schedule: protocol.Schedule, root: Path, tag: str, device: str | None = None
) -> Path:
    """Create <root>/ising/<run_id>/ and its manifest, then run the schedule to the end."""
    if spec.replicas * schedule.blocks > MAX_LATTICES:
        raise ValueError(
            f"{spec.replicas * schedule.blocks:,} lattices in one run, but replica ids are "
            f"uint16: at most {MAX_LATTICES:,} (replicas per temperature x temperatures)"
        )
    dev = torch.device(device or default_device())
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{tag}"
    run_dir = Path(root) / "ising" / run_id
    run_dir.mkdir(parents=True)
    io.write_json(run_dir / "manifest.json", _manifest(spec, schedule, run_id, dev))
    gen = torch.Generator(device=dev).manual_seed(spec.seed)
    shape = (spec.replicas * schedule.blocks, spec.L, spec.L)
    if spec.init == "hot":
        s = torch.randint(0, 2, shape, generator=gen, device=dev, dtype=torch.int8) * 2 - 1
    elif spec.init == "cold":
        s = torch.ones(shape, dtype=torch.int8, device=dev)
    else:
        raise ValueError(f"init must be 'hot' or 'cold', got {spec.init!r}")
    _checkpoint(run_dir, 0, s, gen)
    _loop(run_dir, spec, schedule, s, gen, 0)
    return run_dir


def resume(run_dir: Path) -> Path:
    """Continue a crashed or stopped run from its checkpoint, on the device type it started on."""
    manifest = io.read_json(run_dir / "manifest.json")
    if (run_dir / "complete.json").exists():
        raise RuntimeError(f"{run_dir.name} is already complete")
    if manifest["env"]["torch"] != torch.__version__:
        raise RuntimeError(f"{run_dir.name} needs torch {manifest['env']['torch']} to continue")
    params = manifest["params"]
    spec = Spec(**{f.name: params[f.name] for f in fields(Spec)})
    schedule = protocol.from_json(params["schedule"])
    dev = torch.device(manifest["env"]["device"])
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    gen = torch.Generator(device=dev)
    gen.set_state(ckpt["rng"])
    io.discard_after(run_dir, ckpt["step"])  # the straight run's chunk boundaries come back too
    _loop(run_dir, spec, schedule, ckpt["spins"].to(dev), gen, ckpt["step"])
    return run_dir


class _Buffer:
    """Records kept on the device between flushes, so the loop never waits for a host copy."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.rows: list[tuple[Tensor, ...]] = []

    def __len__(self) -> int:
        return len(self.steps)

    def add(self, step: int, *values: Tensor) -> None:
        self.steps.append(step)
        self.rows.append(values)

    def drain(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """The steps, and one (n, B, ...) host array per recorded value; empties the buffer."""
        columns = [torch.stack(c).cpu().numpy() for c in zip(*self.rows, strict=True)]
        steps = np.array(self.steps, dtype=np.int64)
        self.steps, self.rows = [], []
        return steps, columns


def _loop(
    run_dir: Path,
    spec: Spec,
    schedule: protocol.Schedule,
    s: Tensor,
    gen: torch.Generator,
    done: int,
) -> None:
    torch.use_deterministic_algorithms(True)  # tripwire: a nondeterministic op raises
    dev = s.device
    black = kernel.checkerboard(spec.L, dev)
    tallies = torch.zeros((len(s), 2, 2, 5), dtype=torch.int64, device=dev) if spec.tally else None
    stats, configs = _Buffer(), _Buffer()
    max_configs = max(1, CONFIG_BUFFER_BYTES // s.numel())
    temps: tuple[float, ...] = ()
    table = torch.empty(0)
    clock = time.perf_counter()
    for step in range(done + 1, schedule.total + 1):
        if (t := schedule.temperatures(step)) != temps:
            temps = t
            per_replica = [
                torch.full((spec.replicas,), T, dtype=torch.float64, device=dev) for T in t
            ]
            table = kernel.accept_table(torch.cat(per_replica))
        tally = tallies if tallies is not None and schedule.phase(step) == 1 else None
        s = kernel.sweep(s, black, table, gen, tally)
        frame = schedule.frames.get(step)
        record = schedule.records_stats(step)
        if record or frame is not None:
            M, E = kernel.totals(s)
            if record:
                stats.add(step, M, E)
            if frame is not None:
                configs.add(step, M, E, s)  # sweeps return new tensors, so s is never overwritten
        if (
            step == schedule.total
            or step % spec.checkpoint_every == 0
            or len(configs) >= max_configs
        ):
            _flush(run_dir, spec, schedule, step, stats, configs, tallies)
            if step < schedule.total:
                _checkpoint(run_dir, step, s, gen)
            rate = (step - done) / (time.perf_counter() - clock)
            print(
                f"{run_dir.name}: step {step:,}/{schedule.total:,} ({rate:,.0f} sweeps/s)",
                flush=True,
            )
    io.write_complete(run_dir, schedule.total, _final_checks(run_dir, schedule))
    (run_dir / "checkpoint.pt").unlink()


def _final_checks(run_dir: Path, schedule: protocol.Schedule) -> dict[str, Any]:
    """Verdicts recorded in complete.json: every staircase plateau must look thermalised."""
    if schedule.kind != "staircase":
        return {}
    checks = gates.thermalised(run_dir)
    drifting = [c.temperature for c in checks if not c.passed]
    verdict = f"still drifting at T = {drifting}" if drifting else "all thermalised"
    print(f"{run_dir.name}: {len(checks)} plateaus, {verdict}", flush=True)
    return {"thermalised": [asdict(c) for c in checks]}


def validate(
    root: Path,
    seed: int = 1,
    replicas: int = 4096,
    therm: int = 400,
    sweeps: int = 400,
    device: str | None = None,
) -> tuple[Path, list[gates.Check]]:
    """A small L = 4 run at 1.5, Tc and 3.5, judged by every gate; verdicts go to gates.json."""
    schedule = protocol.Fixed(VALIDATION_TEMPERATURES, therm, sweeps, stats_stride=8, saves=5)
    spec = Spec(L=4, replicas=replicas, seed=seed, tally=True)
    run_dir = start(spec, schedule, root, f"L4-val-s{seed}", device)
    checks = (
        gates.gate_a(run_dir)
        + gates.gate_b(run_dir)
        + gates.gate_c(run_dir)
        + gates.independence(run_dir)
    )
    io.write_json(run_dir / "gates.json", [asdict(c) for c in checks])
    return run_dir, checks


def _keys(spec: Spec, schedule: protocol.Schedule, steps: np.ndarray) -> dict[str, np.ndarray]:
    """step, replica, temperature and segment columns for every (step, replica) row."""
    B = spec.replicas * schedule.blocks
    temps = np.array([schedule.temperatures(int(t)) for t in steps])  # (steps, blocks)
    return {
        "step": np.repeat(steps, B),
        "replica": np.tile(np.arange(B), len(steps)),
        "temperature": np.repeat(temps, spec.replicas, axis=1).ravel(),
        "segment": np.repeat([schedule.segment(int(t)) for t in steps], B),
    }


def _flush(
    run_dir: Path,
    spec: Spec,
    schedule: protocol.Schedule,
    step: int,
    stats: _Buffer,
    configs: _Buffer,
    tallies: Tensor | None,
) -> None:
    B = spec.replicas * schedule.blocks
    if len(stats):
        steps, (M, E) = stats.drain()
        columns = _keys(spec, schedule, steps)
        columns["phase"] = np.repeat([schedule.phase(int(t)) for t in steps], B)
        columns["magnetisation"], columns["energy"] = M.ravel(), E.ravel()
        io.write_chunk(run_dir, "stats", spec.L, columns)
    if len(configs):
        steps, (M, E, S) = configs.drain()
        columns = _keys(spec, schedule, steps)
        burst, frame = np.array([schedule.frames[int(t)] for t in steps]).T
        columns["burst"], columns["frame"] = np.repeat(burst, B), np.repeat(frame, B)
        columns["magnetisation"], columns["energy"] = M.ravel(), E.ravel()
        columns["spins"] = io.pack(S.reshape(-1, spec.L, spec.L))
        io.write_chunk(run_dir, "configs", spec.L, columns)
    if tallies is not None and bool(tallies.any()):
        counts = tallies.cpu().numpy()  # (replica, colour, [attempted, accepted], dE index)
        columns = {k: np.repeat(v, 10) for k, v in _keys(spec, schedule, np.array([step])).items()}
        columns["colour"] = np.tile(np.repeat([0, 1], 5), B)
        columns["delta_e"] = np.tile(np.arange(-8, 9, 4), 2 * B)
        columns["attempted"] = counts[:, :, 0].ravel()
        columns["accepted"] = counts[:, :, 1].ravel()
        io.write_chunk(run_dir, "balance", spec.L, columns)
        tallies.zero_()  # only after the write: on the CPU, counts shares this memory


def _checkpoint(run_dir: Path, step: int, s: Tensor, gen: torch.Generator) -> None:
    state = {"step": step, "spins": s.cpu(), "rng": gen.get_state()}
    io.write_file(run_dir / "checkpoint.pt", lambda f: torch.save(state, f), replace=True)


def _manifest(
    spec: Spec, schedule: protocol.Schedule, run_id: str, dev: torch.device
) -> dict[str, Any]:
    used = {"stats": True, "configs": bool(schedule.frames), "balance": spec.tally}
    return {
        "format": 1,
        "domain": "ising",
        "run_id": run_id,
        "producer": {"name": "tipping_point.ising", "version": __version__, **_git()},
        "started_ns": time.time_ns(),
        "key": "step",
        "tables": [name for name in io.TABLES if used[name]],
        "params": {
            **asdict(spec),
            "blocks": schedule.blocks,
            "J": 1.0,
            "k_B": 1.0,
            "boundary": "periodic",
            "Tc": TC,
            "schedule": schedule.to_json(),
            "algorithm": {
                "name": "checkerboard-metropolis",
                "order": "black (i + j even) half-sweep, then white",
                "accept": "u < min(1, exp(-dE/T)); table built in float64, used as float32",
            },
            "rng": {
                "generator": f"torch.Generator({dev.type})",
                "unit": "run",
                "draws": "one (B, L, L) uniform tensor per half-sweep",
            },
            "encoding": {
                "spins": "1 bit per site, 1 = +1, row-major, most significant bit first",
                "magnetisation": "sum of spins",
                "energy": "-sum over the 2 L^2 bonds of s s'",
            },
            "tau": {"convention": "1 + 2 sum rho(t)", "window_c": 5.0},
        },
        "env": {
            "device": dev.type,
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
    }


def _git() -> dict[str, Any]:
    """The commit that produced the run, and whether the tree had uncommitted edits."""
    here = Path(__file__).parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"git_sha": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain") != ""}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty": None}
