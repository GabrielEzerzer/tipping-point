"""The training loop, generic over the bursts it is handed: seeding, precision per device,
monitors, checkpoints and resume, the run manifest and the GPU-hour ledger.

<data dir>/jepa/<run_id>/
  manifest.json                     written once at the start (#22's shape)
  train/<first_step:020>.parquet    one row per (step, L): loss terms, learning rate, weight
                                    decay, EMA momentum, gradient norm, loss scale
  monitor/<first_step:020>.parquet  one row per (evaluation, L), from step 0: validation error,
                                    feature spreads, effective rank, per-dimension std
  ckpt/step_0000000.pt              the untrained twin with its PCA basis, saved before any update
  checkpoint.pt                     the latest resumable state, replaced atomically; removed at a
                                    clean end
  final.pt                          the trained model with its PCA basis
  complete.json                     every other file's sha256 and the collapse verdict; written
                                    once at a clean end, absent = running or crashed
<data dir>/jepa/gpu_hours.jsonl     a line per checkpoint of every session: the hours since the
                                    session's previous line (`budget` sums them)

Reproducibility: generators from one master seed, one per stream (initialisation; data order,
on the CPU; pairs, masks and augmentation, on the device; and fixed draws for the monitor batch,
the PCA frames and the evaluation masks). Their states are ByteTensors, so checkpoints load with
weights_only=True. use_deterministic_algorithms(True) is a tripwire, cuDNN benchmarking is off
and CUBLAS_WORKSPACE_CONFIG is set before CUDA starts (tipping_point.jepa). The unit of
reproducibility is one run on one device type and torch version: two CPU runs of a seed give
bitwise-identical losses, and a resumed run continues bit for bit where a straight one would be.

Precision: fp16 autocast with a gradient scaler on a T4 (Turing has no bf16), float32 elsewhere;
never bf16 and never torch.compile (it fails on Windows without MSVC, and inductor refuses the
P100). Readouts always run in float32.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .. import __version__, engine
from ..ising import io
from . import monitor, objective
from .data import Dataset, reload
from .lattice import (
    PATCH,
    Jepa,
    MaskSpec,
    ModelSpec,
    augment,
    canonical,
    fixed_masks,
    masks,
    pairs,
    parameters,
)
from .lattice import unpack as unpack_spins
from .monitor import Detector, Floors
from .readout import Pca, fit_pca

STREAMS = ("init", "data", "augment", "monitor", "pca", "eval_masks")
TABLES = ("train", "monitor")
TWIN = Path("ckpt") / f"step_{0:07d}.pt"
FINAL = Path("final.pt")
LEDGER = "gpu_hours.jsonl"
TRAIN_SCHEMA = pa.schema(
    [("step", pa.int64()), ("L", pa.int16())]
    + [
        (name, pa.float64())
        for name in (
            "loss",
            "prediction",
            "variance",
            "covariance",
            "lr",
            "weight_decay",
            "momentum",
            "grad_norm",
            "scale",
        )
    ]
)
MONITOR_SCHEMA = pa.schema(
    [("step", pa.int64()), ("L", pa.int16())]
    + [
        (name, pa.float64())
        for name in (
            "val_error",
            "token_std",
            "dim_std_min",
            "spatial_std",
            "pooled_std",
            "effective_rank",
        )
    ]
    + [("dim_std", pa.list_(pa.float64())), ("fired", pa.bool_())]
)


@dataclass(frozen=True)
class Config:
    """Everything that defines a training besides its data; the defaults are the research
    pass's. Each step runs one micro-batch per size of about `batch_sites` sites (256, 64 and 16
    bursts at L = 16, 32 and 64) and one optimiser update."""

    steps: int = 50_000
    model: ModelSpec = field(default_factory=ModelSpec)
    mask: MaskSpec = field(default_factory=MaskSpec)
    batch_sites: int = 65_536
    lr: float = 1e-3
    lr_final: float = 1e-6
    warmup: int = 2_500
    weight_decay: tuple[float, float] = (0.04, 0.4)
    ema: tuple[float, float] = (0.996, 1.0)
    var_weight: float = 0.025  # C-JEPA's effective weights to start; raise them until the
    cov_weight: float = 0.001  # monitors are quiet on the toy and the validation seed
    var_gamma: float = 1.0
    d4: bool = True
    roll: bool = True
    seed: int = 1
    checkpoint_every: int = 1_000
    monitor_every: int = 250
    monitor_frames: int = 64
    pca_frames: int = 4_096  # per size
    pca_components: int = 8
    eval_masks: int = 2
    floors: Floors = field(default_factory=Floors)
    patience: int = 3
    stop_on_collapse: bool = True

    def batch(self, L: int) -> int:
        return max(1, self.batch_sites // (L * L))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> Config:
        r = dict(record)
        mask = r["mask"]
        r["model"] = ModelSpec(**r["model"])
        r["mask"] = MaskSpec(tuple(mask["block"]), tuple(mask["coverage"]), mask["buffer"])
        r["floors"] = Floors(**r["floors"])
        r["weight_decay"], r["ema"] = tuple(r["weight_decay"]), tuple(r["ema"])
        return cls(**r)


def seeds(master: int) -> dict[str, int]:
    """One seed per random stream, all derived from the master seed."""
    return {
        name: int(np.random.SeedSequence([master, k]).generate_state(1, np.uint64)[0])
        for k, name in enumerate(STREAMS)
    }


def evaluation_masks(config: Config, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The (hidden, keep) masks surprise is measured through at size L: drawn once from the
    training's seed, so a run's checkpoints, trained or not, see the same ones."""
    return fixed_masks(L // PATCH, config.mask, seeds(config.seed)["eval_masks"], config.eval_masks)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name(device: torch.device) -> str | None:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else None


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """fp16 autocast on a T4, float32 elsewhere. Never bf16: neither the T4 (sm_75) nor the P100
    (sm_60) has it natively, and is_bf16_supported() counts emulation by default."""
    name = gpu_name(device)
    return torch.float16 if name is not None and "T4" in name else None


def deterministic(device: torch.device) -> dict[str, Any]:
    """Determinism as a tripwire: a nondeterministic op raises rather than drifting."""
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device.type == "cuda" and workspace not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA starts")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "use_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": workspace,
    }


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------- frozen models


@dataclass
class Checkpoint:
    """A frozen model, as readouts load it."""

    path: Path
    sha256: str
    run_id: str
    step: int
    config: Config
    model: Jepa
    pca: Pca | None

    @property
    def untrained(self) -> bool:
        return self.step == 0


def save_model(path: Path, run_id: str, step: int, config: Config, model: Jepa, pca: Pca) -> None:
    record = {
        "format": 1,
        "run_id": run_id,
        "step": step,
        "config": config.to_json(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "pca": pca.record(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    io.write_file(path, lambda f: torch.save(record, f), replace=True)


def load_model(path: Path, device: str | torch.device = "cpu") -> Checkpoint:
    """A saved model (final.pt, the step-0 twin) in eval mode on `device`, with its file's
    sha256 (read once, so the hash is of the bytes loaded)."""
    raw = Path(path).read_bytes()
    record = torch.load(BytesIO(raw), weights_only=True, map_location="cpu")
    config = Config.from_json(record["config"])
    model = Jepa(config.model)
    model.load_state_dict(record["model"])
    model.to(device).eval()
    pca = Pca.from_record(record["pca"]) if record.get("pca") is not None else None
    return Checkpoint(
        Path(path), sha256(raw), record["run_id"], int(record["step"]), config, model, pca
    )


# ---------------------------------------------------------------- records


def pilot_record(path: Path) -> dict[str, Any]:
    """The #50 pilot record a training names: its run id (the folder it sits in, never a local
    path: manifests travel, #94), sha256, kernel commit and the sizes it measured."""
    raw = Path(path).read_bytes()
    try:
        record = json.loads(raw)
        commit, measured = record["kernel_commit"], sorted(int(L) for L in record["per_L"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: not a pilot record ({exc!r})") from exc
    run_id = Path(path).resolve().parent.name  # resolved: a bare "pilot.json" has a folder too
    return {"run_id": run_id, "sha256": sha256(raw), "kernel_commit": commit, "L": measured}


def prereg_record(path: Path) -> dict[str, Any]:
    """A pre-registration a training names: id, version, sha256 and the seeds it scores; id and
    version name its file and sha256 pins it, so no local path is kept (#94)."""
    raw = Path(path).read_bytes()
    try:
        doc = tomllib.loads(raw.decode("utf-8"))
        prereg_id, version = str(doc["id"]), int(doc["version"])
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError, ValueError) as exc:
        raise ValueError(f"{path}: not a pre-registration ({exc!r})") from exc
    scored = [int(s) for s in doc.get("data", {}).get("seeds", [])]
    return {"id": prereg_id, "version": version, "sha256": sha256(raw), "seeds": scored}


def append_line(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line, synced before returning; the file only ever grows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def budget(root: Path, quota: float) -> list[dict[str, Any]]:
    """GPU hours from the ledger per ISO week (UTC), against a weekly quota in hours."""
    path = Path(root) / "jepa" / LEDGER
    weeks: dict[str, float] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("device") != "cuda":
                continue
            year, week, _ = datetime.fromtimestamp(record["to_ns"] / 1e9, UTC).isocalendar()
            key = f"{year}-W{week:02d}"
            weeks[key] = weeks.get(key, 0.0) + float(record["hours"])
    return [
        {"week": k, "hours": h, "quota": quota, "left": quota - h} for k, h in sorted(weeks.items())
    ]


# ---------------------------------------------------------------- the loop


class _Trainer:
    def __init__(
        self, run_dir: Path, run_id: str, config: Config, data: Dataset, device: torch.device
    ) -> None:
        self.run_dir, self.run_id, self.config, self.data = run_dir, run_id, config, data
        self.device = device
        self.amp = amp_dtype(device)
        self.seeds = seeds(config.seed)
        self.sizes = sorted(data.train)
        model = Jepa(config.model)
        model.initialise(torch.Generator().manual_seed(self.seeds["init"]))
        self.model = model.to(device)
        trainable = model.trainable()
        groups = [
            {"params": [p for p in trainable if p.ndim >= 2]},
            {"params": [p for p in trainable if p.ndim < 2], "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            groups, lr=config.lr, weight_decay=config.weight_decay[0]
        )
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp is not None)
        self.data_gen = torch.Generator().manual_seed(self.seeds["data"])
        self.aug_gen = torch.Generator(device=device).manual_seed(self.seeds["augment"])
        self.detector = Detector(config.floors, config.patience)
        self.step = 0
        for L in self.sizes:  # drawn now: a mask with no context refuses the run before it starts
            evaluation_masks(config, L)
        self.monitor_batches = self._monitor_batches()
        self.train_rows: list[dict[str, Any]] = []
        self.monitor_rows: list[dict[str, Any]] = []
        self.clock = time.time_ns()

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.amp is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp)

    def _monitor_batches(self) -> dict[int, dict[str, torch.Tensor]]:
        """A fixed validation batch per size: bursts, pairs and masks drawn once."""
        gen = torch.Generator().manual_seed(self.seeds["monitor"])
        batches = {}
        for L in self.sizes:
            count = self.config.monitor_frames
            bursts = canonical(unpack_spins(self.data.val[L].sample(count, gen), L), 3)
            context, target, delta = pairs(bursts, self.config.model.max_delta, gen)
            hidden, keep = masks(count, L // PATCH, self.config.mask, gen, "cpu")
            batch = {"context": context, "target": target, "delta": delta}
            batches[L] = {
                k: v.to(self.device) for k, v in (batch | {"hidden": hidden, "keep": keep}).items()
            }
        return batches

    def pca(self) -> Pca:
        gen = torch.Generator().manual_seed(self.seeds["pca"])
        cfg = self.config
        return fit_pca(
            self.model,
            self.data.train,
            cfg.pca_frames,
            cfg.pca_components,
            gen,
            self.device,
            cfg.batch_sites,
        )

    @torch.no_grad()
    def evaluate(self) -> bool:
        """Validation error and feature spreads per size, judged by the detector; returns whether
        the run has collapsed."""
        stats = {}
        with torch.autocast(self.device.type, enabled=False):
            for L, b in self.monitor_batches.items():
                targets = self.model.targets(b["target"])
                _, pred = self.model.predict(b["context"], b["keep"], b["delta"])
                error = float(objective.prediction_loss(pred, targets, b["hidden"]))
                stats[L] = {"val_error": error, **monitor.spread(self.model.context(b["context"]))}
        self.detector.check(self.step, stats)
        for L, s in stats.items():
            row = {"step": self.step, "L": L, **s, "fired": bool(self.config.floors.broken(s))}
            self.monitor_rows.append(row)
        return self.detector.collapsed

    def update(self) -> None:
        """One optimiser update over a micro-batch of every size."""
        cfg, u = self.config, self.step
        lr = objective.learning_rate(u, cfg.steps, cfg.lr, cfg.lr_final, cfg.warmup)
        wd = objective.cosine(u, cfg.steps, *cfg.weight_decay)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.param_groups[0]["weight_decay"] = wd
        self.optimizer.zero_grad(set_to_none=True)
        terms = []
        for L in self.sizes:
            packed = self.data.train[L].sample(cfg.batch(L), self.data_gen).to(self.device)
            bursts = canonical(unpack_spins(packed, L), 3)
            bursts = augment(bursts, self.aug_gen, cfg.d4, cfg.roll)
            context, target, delta = pairs(bursts, cfg.model.max_delta, self.aug_gen)
            hidden, keep = masks(len(bursts), L // PATCH, cfg.mask, self.aug_gen, self.device)
            with self._autocast():
                targets = self.model.targets(target)
                features, pred = self.model.predict(context, keep, delta)
            prediction = objective.prediction_loss(pred, targets, hidden)
            variance, covariance = objective.vicreg(features, keep, cfg.var_gamma)
            loss = prediction + cfg.var_weight * variance + cfg.cov_weight * covariance
            self.scaler.scale(loss / len(self.sizes)).backward()
            terms.append((L, loss, prediction, variance, covariance))
        self.scaler.unscale_(self.optimizer)
        grads = [p.grad for p in self.model.trainable() if p.grad is not None]
        norm = float(torch.nn.utils.get_total_norm(grads))
        scale = float(self.scaler.get_scale()) if self.amp is not None else 1.0
        self.scaler.step(self.optimizer)
        self.scaler.update()
        m = objective.momentum(u, cfg.steps, *cfg.ema)
        objective.ema(self.model.target, self.model.context, m)
        self.step += 1
        for L, loss, prediction, variance, covariance in terms:
            self.train_rows.append(
                {
                    "step": self.step,
                    "L": L,
                    "loss": loss.item(),
                    "prediction": prediction.item(),
                    "variance": variance.item(),
                    "covariance": covariance.item(),
                    "lr": lr,
                    "weight_decay": wd,
                    "momentum": m,
                    "grad_norm": norm,
                    "scale": scale,
                }
            )

    def run(self) -> None:
        cfg = self.config
        collapsed = self.evaluate() if self.step == 0 else self.detector.collapsed
        while self.step < cfg.steps and not (collapsed and cfg.stop_on_collapse):
            self.update()
            if self.step % cfg.monitor_every == 0 or self.step == cfg.steps:
                collapsed = self.evaluate()
            stopping = self.step == cfg.steps or (collapsed and cfg.stop_on_collapse)
            if stopping or self.step % cfg.checkpoint_every == 0:
                loss = self.train_rows[-1]["loss"] if self.train_rows else float("nan")
                self.flush()
                if not stopping:
                    self.checkpoint()
                self.log_hours()
                print(
                    f"{self.run_id}: step {self.step:,}/{cfg.steps:,} loss {loss:.4f}", flush=True
                )
        self.finish()

    def flush(self) -> None:
        for name, rows, schema in (
            ("train", self.train_rows, TRAIN_SCHEMA),
            ("monitor", self.monitor_rows, MONITOR_SCHEMA),
        ):
            if rows:
                table = pa.Table.from_pylist(rows, schema=schema)
                path = self.run_dir / name / f"{int(rows[0]['step']):020d}.parquet"
                path.parent.mkdir(exist_ok=True)
                io.write_file(path, lambda f, t=table: pq.write_table(t, f, compression="zstd"))
        self.train_rows, self.monitor_rows = [], []

    def checkpoint(self) -> None:
        state = {
            "format": 1,
            "run_id": self.run_id,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "generators": {"data": self.data_gen.get_state(), "augment": self.aug_gen.get_state()},
            "detector": self.detector.state(),
        }
        io.write_file(self.run_dir / "checkpoint.pt", lambda f: torch.save(state, f), replace=True)

    def restore(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.data_gen.set_state(state["generators"]["data"])
        self.aug_gen.set_state(state["generators"]["augment"])
        self.detector.restore(state["detector"])
        self.step = int(state["step"])

    def log_hours(self) -> None:
        now = time.time_ns()
        record = {
            "run_id": self.run_id,
            "from_ns": self.clock,
            "to_ns": now,
            "hours": (now - self.clock) / 3.6e12,
            "step": self.step,
            "device": self.device.type,
            "gpu": gpu_name(self.device),
        }
        append_line(self.run_dir.parents[1] / "jepa" / LEDGER, record)
        self.clock = now

    def finish(self) -> None:
        self.flush()
        save_model(
            self.run_dir / FINAL, self.run_id, self.step, self.config, self.model, self.pca()
        )
        skip = {"complete.json", "checkpoint.pt"}
        files = {
            p.relative_to(self.run_dir).as_posix(): sha256(p.read_bytes())
            for p in sorted(self.run_dir.rglob("*"))
            if p.is_file() and p.name not in skip and p.suffix != ".tmp"
        }
        rows = {
            name: sum(pq.read_metadata(p).num_rows for p in io.chunks(self.run_dir, name))
            for name in TABLES
        }
        record = {
            "ended_ns": time.time_ns(),
            "last_key": self.step,
            "rows": rows,
            "clean": True,
            "stopped_early": self.step < self.config.steps,
            "collapse": self.detector.verdict(),
            "files": files,
        }
        io.write_json(self.run_dir / "complete.json", record)
        (self.run_dir / "checkpoint.pt").unlink(missing_ok=True)
        self.log_hours()


def _manifest(
    trainer: _Trainer,
    flags: dict[str, Any],
    pilot: dict[str, Any] | None,
    prereg: dict[str, Any] | None,
) -> dict[str, Any]:
    dev, model, cfg = trainer.device, trainer.model, trainer.config
    return {
        "format": 1,
        "domain": "jepa",
        "run_id": trainer.run_id,
        "producer": {"name": "tipping_point.jepa", "version": __version__, **io.code_version()},
        "started_ns": time.time_ns(),
        "key": "step",
        "tables": list(TABLES),
        "params": {
            "config": cfg.to_json(),
            "sizes": trainer.sizes,
            "batches": {str(L): cfg.batch(L) for L in trainer.sizes},
            "seeds": trainer.seeds,
            "parameters": {
                "encoder": parameters(model.context),
                "predictor": parameters(model.predictor),
            },
            "receptive_field": model.context.receptive_field,
        },
        "data": trainer.data.sources,
        "pilot": pilot,
        "prereg": prereg,
        "env": {
            "device": dev.type,
            "gpu": gpu_name(dev),
            "capability": list(torch.cuda.get_device_capability(dev))
            if dev.type == "cuda"
            else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if dev.type == "cuda" else None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "threads": torch.get_num_threads(),
            "amp": None if trainer.amp is None else str(trainer.amp).removeprefix("torch."),
            **flags,
        },
    }


def start(
    config: Config,
    data: Dataset,
    root: Path,
    tag: str,
    device: str | None = None,
    pilot: dict[str, Any] | None = None,
    prereg: dict[str, Any] | None = None,
) -> Path:
    """Create <root>/jepa/<run_id>/ with its manifest and the untrained twin, then train to the
    end (or to a collapse, when the config stops on one)."""
    dev = torch.device(device or default_device())
    flags = deterministic(dev)
    run_id = engine.run_id(tag)
    run_dir = Path(root) / "jepa" / run_id
    trainer = _Trainer(run_dir, run_id, config, data, dev)
    run_dir.mkdir(parents=True)
    io.write_json(run_dir / "manifest.json", _manifest(trainer, flags, pilot, prereg))
    save_model(run_dir / TWIN, run_id, 0, config, trainer.model, trainer.pca())
    trainer.checkpoint()
    trainer.run()
    return run_dir


def resume(run_dir: Path, data: Dataset | None = None, device: str | None = None) -> Path:
    """Continue a crashed or stopped training from its checkpoint, on the device and torch version
    it started on. Without `data`, the bursts are reloaded from the runs the manifest names,
    which must be byte for byte the ones it read (complete.json sha256)."""
    run_dir = Path(run_dir)
    manifest = io.read_json(run_dir / "manifest.json")
    if (run_dir / "complete.json").exists():
        raise RuntimeError(f"{run_dir.name} is already complete")
    env = manifest["env"]
    if env["torch"] != torch.__version__:
        raise RuntimeError(f"{run_dir.name} needs torch {env['torch']} to continue")
    dev = torch.device(device or env["device"])
    if dev.type != env["device"] or gpu_name(dev) != env["gpu"]:
        raise RuntimeError(f"{run_dir.name} continues only on {env['device']} {env['gpu'] or ''}")
    deterministic(dev)
    config = Config.from_json(manifest["params"]["config"])
    data = data if data is not None else reload(run_dir.parents[1], manifest["data"])
    if sorted(data.train) != manifest["params"]["sizes"]:
        raise RuntimeError(f"{run_dir.name} trained on sizes {manifest['params']['sizes']}")
    trainer = _Trainer(run_dir, manifest["run_id"], config, data, dev)
    trainer.restore(torch.load(run_dir / "checkpoint.pt", weights_only=True, map_location="cpu"))
    io.discard_after(run_dir, trainer.step, TABLES)
    trainer.run()
    return run_dir


def verify(run_dir: Path) -> list[str]:
    """Problems with a finished directory (a JEPA run, a latents set or a simulator run): files
    complete.json lists that are missing or whose sha256 differs. Empty means it checks out."""
    done = Path(run_dir) / "complete.json"
    if not done.is_file():
        return [f"{run_dir}: no complete.json (running, crashed or not a run)"]
    problems = []
    for name, digest in io.read_json(done)["files"].items():
        path = Path(run_dir) / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif sha256(path.read_bytes()) != digest:
            problems.append(f"sha256 differs: {name}")
    return problems
