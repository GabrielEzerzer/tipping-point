# Sequence models on the panel (docs/market.md, "Coin panel"): a JEPA, its untrained twin and a
# supervised model share one encoder over the last 336 hourly input rows of a ladder rung, fitted in
# each quarter fold, probed over a grid of readouts and penalties, and saved for the scores step.

from __future__ import annotations

import copy
import hashlib
import math
import os
import platform
import re
import time
import tomllib
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..ews.scoring import auc
from ..ising import io
from ..jepa.objective import ema, learning_rate, momentum
from . import encoder, labels, ladder, span, store

PLAN = Path(__file__).with_name("sequence.toml")
PINS = (
    "ATEN_CPU_CAPABILITY",
    "ONEDNN_MAX_CPU_ISA",
    "MKL_ENABLE_INSTRUCTIONS",
    "MKL_CBWR",
    "OPENBLAS_CORETYPE",
    "OPENBLAS_NUM_THREADS",
    "NPY_DISABLE_CPU_FEATURES",
)
CURVES = pa.schema(
    [
        ("fold", pa.int64()),
        ("scheme", pa.string()),
        ("seed", pa.int64()),
        ("variant", pa.string()),
        ("model", pa.string()),
        ("step", pa.int64()),
        ("loss", pa.float64()),
        ("prediction", pa.float64()),
        ("variance", pa.float64()),
        ("covariance", pa.float64()),
        ("auc", pa.float64()),
        ("auc_long", pa.float64()),
        ("auc_short", pa.float64()),
        ("rows", pa.int64()),
    ]
)
MODELS = ("jepa", "twin", "supervised")
TARGETS = ("future", "shifted")
OBJECTIVES = ("z", "predictor")
LOG = Callable[[str], None]
READOUTS = {"last": 1, "means": 4, "spreads": 7}
SPANS = (6, 18)


@dataclass(frozen=True)
class Training:
    steps: int
    batch: int
    lr: float
    final_lr: float
    warmup: int
    weight_decay: float
    curve_every: int
    momentum_start: float = 0.0
    momentum_end: float = 0.0
    prediction: float = 0.0
    variance: float = 0.0
    covariance: float = 0.0
    objective: str = "z"
    dropout: float = 0.0


@dataclass(frozen=True)
class Selection:
    share: float
    every: int
    negatives: int


@dataclass(frozen=True)
class Plan:
    ladder: ladder.Plan
    ladder_path: Path
    variants: tuple[str, ...]
    rungs: tuple[str, ...]
    models: tuple[str, ...]
    schemes: tuple[str, ...]
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    target: str
    threads: int
    window: int
    future: int
    shifts: tuple[int, ...]
    shape: encoder.Shape
    predictor_hidden: int
    jepa: Training
    supervised: Training
    selection: Selection
    encode_batch: int
    readouts: tuple[str, ...]
    cs: tuple[float, ...]

    @property
    def pad(self) -> int:
        return self.window - 1

    @property
    def reach(self) -> int:
        return max(self.future, *self.shifts, *self.ladder.horizons)

    @property
    def outputs(self) -> tuple[tuple[str, int], ...]:
        return tuple((side, h) for side in ("long", "short") for h in self.ladder.horizons)

    def rung(self, name: str) -> ladder.Rung:
        return next(r for r in self.ladder.rungs if r.name == name)


def load(
    path: Path = PLAN,
    ladder_path: Path | None = None,
    texts: tuple[str, str] | None = None,
) -> Plan:
    raw = tomllib.loads(path.read_text(encoding="utf-8") if texts is None else texts[0])
    if ladder_path is None:
        ladder_path = (path.parent / str(raw["ladder"])).resolve()
    base = ladder.load(ladder_path, None if texts is None else texts[1])
    nfolds = len(ladder.bounds(base)) - 1
    e = raw["encoder"]
    plan = Plan(
        ladder=base,
        ladder_path=ladder_path,
        variants=tuple(raw["variants"]),
        rungs=tuple(raw["rungs"]),
        models=tuple(raw["models"]),
        schemes=tuple(raw.get("schemes", ["forward"])),
        seeds=tuple(int(s) for s in raw["seeds"]),
        folds=tuple(raw.get("folds") or range(nfolds)),
        target=raw["target"],
        threads=int(raw["threads"]),
        window=int(raw["window"]["length"]),
        future=int(raw["window"]["future"]),
        shifts=tuple(int(s) for s in raw["window"]["shifts"]),
        shape=encoder.Shape(
            int(e["patch"]),
            int(e["width"]),
            int(e["conv_blocks"]),
            int(e["attention_blocks"]),
            int(e["heads"]),
            int(e["kernel"]),
            int(e["expansion"]),
        ),
        predictor_hidden=int(raw["predictor"]["hidden"]),
        jepa=Training(**raw["jepa"]),
        supervised=Training(**raw["supervised"]),
        selection=Selection(**raw["selection"]),
        encode_batch=int(raw["probe"]["batch"]),
        readouts=tuple(raw["probe"]["readouts"]),
        cs=tuple(float(c) for c in raw["probe"]["c"]),
    )
    rungs = {r.name for r in base.rungs}
    problems = [f"variant {v}" for v in plan.variants if v not in rungs]
    problems += [f"rung {r}" for r in plan.rungs if r not in rungs]
    problems += [f"model {m}" for m in plan.models if m not in MODELS]
    problems += [f"scheme {s}" for s in plan.schemes if s not in ladder.SCHEMES]
    problems += [f"fold {f}" for f in plan.folds if not 0 <= f < nfolds]
    problems += [f"readout {r}" for r in plan.readouts if r not in READOUTS]
    problems += [f"c {c}" for c in plan.cs if not (np.isfinite(c) and c > 0)]
    for name in ("variants", "rungs", "models", "schemes", "folds", "seeds", "readouts", "cs"):
        values = getattr(plan, name)
        if not values or len(set(values)) != len(values):
            problems.append(f"{name} must be given and distinct")
    if plan.target not in TARGETS:
        problems.append(f"target {plan.target}")
    if plan.jepa.objective not in OBJECTIVES:
        problems.append(f"objective {plan.jepa.objective}")
    if plan.future <= 0 or any(s <= 0 for s in plan.shifts):
        problems.append("future and every shift must be above 0")
    if plan.target == "shifted" and not plan.shifts:
        problems.append("the shifted target needs shifts")
    if plan.jepa.batch < 2 or plan.supervised.batch < 2:
        problems.append("batches must hold at least 2 rows")
    if any(s < 0 for s in plan.seeds):
        problems.append("seeds must not be negative")
    if not 0 < plan.selection.share < 1 or plan.selection.every <= 0:
        problems.append("the selection share must lie in (0, 1) and its interval be above 0")
    if plan.selection.negatives <= 0:
        problems.append("the selection must draw at least one negative row")
    if plan.jepa.dropout != 0:
        problems.append("the JEPA takes no dropout, so its twin is its encoder at step 0")
    lengths = [plan.window, plan.future]
    if any(n % plan.shape.patch for n in lengths) or plan.shape.width % plan.shape.heads:
        problems.append("window lengths must be whole patches and width whole heads")
    elif plan.shape.width // plan.shape.heads % 2:
        problems.append("each head's width must be even for the rotary positions")
    if plan.window // plan.shape.patch < max(SPANS):
        problems.append(f"the window must hold at least {max(SPANS)} tokens for the readouts")
    if problems:
        raise ValueError(f"{path}: unknown or invalid: {', '.join(problems)}")
    return plan


def training(ts: np.ndarray, plan: Plan, f: int, scheme: str) -> np.ndarray:
    edges = ladder.bounds(plan.ladder)
    fold, _ = ladder.folds(ts, edges, plan.ladder.block_days)
    inside = (ts >= edges[0]) & (ts < edges[-1])
    return inside & ladder.training(ts, fold, f, edges, plan.ladder.gap_days, scheme)


def stream(variant: str, f: int, seed: int, name: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [seed, f, zlib.crc32(variant.encode()), zlib.crc32(name.encode())]
    )


def torch_seed(variant: str, f: int, seed: int, name: str) -> int:
    return int(stream(variant, f, seed, name).generate_state(1, np.uint64)[0])


def layout(grid: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ts, symbol = grid["ts_ns"], grid["symbol"].astype(str)
    hours = np.unique(ts)
    coins = np.unique(symbol)
    expected_ts = np.repeat(hours, len(coins))
    expected_symbol = np.tile(coins, len(hours))
    if not (np.array_equal(ts, expected_ts) and np.array_equal(symbol, expected_symbol)):
        raise ValueError("the grid is not every coin at every hour, ordered by hour then coin")
    if np.any(np.diff(hours) != labels.HOUR_NS):
        raise ValueError("the grid's hours are not consecutive")
    return hours, coins


def by_coin(column: np.ndarray, ncoins: int) -> np.ndarray:
    return column.reshape(-1, ncoins, *column.shape[1:]).swapaxes(0, 1)


def statistics(raw: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    part = raw[:, train].reshape(-1, raw.shape[-1])
    finite = np.isfinite(part)
    count = finite.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(finite, part, 0.0).sum(axis=0) / count
        sd = np.sqrt(np.where(finite, (part - mu) ** 2, 0.0).sum(axis=0) / count)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return mu, sd


def standardised(
    raw: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float, pad: int
) -> np.ndarray:
    ncoins, nhours, nchannels = raw.shape
    z = np.clip((raw - mu) / sd, -clip, clip)
    missing = np.isnan(z)
    out = np.zeros((ncoins, pad + nhours, nchannels + 1), np.float32)
    out[:, :pad, -1] = 1.0
    out[:, pad:, :nchannels] = np.where(missing, 0.0, z)
    out[:, pad:, -1] = missing.mean(axis=-1)
    return out


def windows(x: Tensor, coin: Tensor, last: Tensor, length: int) -> Tensor:
    index = last[:, None] - length + 1 + torch.arange(length)
    return x[coin[:, None], index]


def eligible(train: np.ndarray, back: int, ahead: int) -> np.ndarray:
    n = len(train)
    outside = np.concatenate([[0], np.cumsum(~train)])
    j = np.arange(n)
    lo, hi = j - back, j + ahead + 1
    inside = (lo >= 0) & (hi <= n)
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    return inside & (outside[hi] == outside[lo])


def sample_rows(
    allowed_hours: np.ndarray, ncoins: int, exclude: np.ndarray | None = None
) -> np.ndarray:
    allowed = np.broadcast_to(allowed_hours, (ncoins, len(allowed_hours)))
    if exclude is not None:
        allowed = allowed & ~exclude
    return np.flatnonzero(allowed.ravel())


def batches(
    rows: np.ndarray, nhours: int, batch: int, seed: np.random.SeedSequence
) -> Iterator[Tensor]:
    if not rows.size:
        raise ValueError("no row may be sampled for training in this fold")
    rng = np.random.default_rng(seed)
    while True:
        pick = rows[rng.integers(0, rows.size, batch)]
        yield torch.from_numpy(np.stack([pick // nhours, pick % nhours]))


def vicreg(z: Tensor) -> tuple[Tensor, Tensor]:
    centred = z - z.mean(dim=0)
    cov = centred.T @ centred / (z.shape[0] - 1)
    variance = F.relu(1.0 - torch.sqrt(cov.diagonal() + 1e-4)).mean()
    off = cov - torch.diag(cov.diagonal())
    return variance, off.pow(2).sum() / z.shape[1]


def optimiser(modules: Sequence[nn.Module], t: Training) -> torch.optim.Optimizer:
    params = [p for m in modules for p in m.parameters()]
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": t.weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=t.lr)


def schedule(opt: torch.optim.Optimizer, step: int, t: Training) -> None:
    lr = learning_rate(step, t.steps, t.lr, t.final_lr, t.warmup)
    for group in opt.param_groups:
        group["lr"] = lr


def weights(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


@dataclass
class Jepa:
    online: encoder.Encoder
    target: encoder.Encoder
    predictor: encoder.Predictor


def jepa(channels: int, plan: Plan, seed: int) -> Jepa:
    online = encoder.encoder(channels, plan.shape, seed)
    target = copy.deepcopy(online).requires_grad_(False).eval()
    slots = plan.future // plan.shape.patch if plan.target == "future" else len(plan.shifts)
    return Jepa(online, target, encoder.Predictor(plan.shape.width, plan.predictor_hidden, slots))


def goal(model: Jepa, x: Tensor, coin: Tensor, hour: Tensor, plan: Plan) -> Tensor:
    if plan.target == "future":
        out = model.target.tokens(windows(x, coin, hour + plan.pad + plan.future, plan.future))
    else:
        out = torch.stack(
            [model.target(windows(x, coin, hour + plan.pad + s, plan.window)) for s in plan.shifts],
            dim=1,
        )
    return F.layer_norm(out, out.shape[-1:])


def regulariser(z: Tensor, predicted: Tensor, t: Training) -> tuple[Tensor, Tensor]:
    if t.objective == "z":
        return vicreg(z)
    parts = [vicreg(predicted[:, i]) for i in range(predicted.shape[1])]
    return torch.stack([v for v, _ in parts]).mean(), torch.stack([c for _, c in parts]).mean()


def jepa_step(
    model: Jepa,
    opt: torch.optim.Optimizer,
    x: Tensor,
    rows: Tensor,
    step: int,
    plan: Plan,
) -> dict[str, float]:
    t = plan.jepa
    schedule(opt, step, t)
    coin, hour = rows
    z = model.online(windows(x, coin, hour + plan.pad, plan.window))
    with torch.no_grad():
        target = goal(model, x, coin, hour, plan)
    predicted = model.predictor(z)
    prediction = (predicted - target).pow(2).mean()
    variance, covariance = regulariser(z, predicted, t)
    loss = t.prediction * prediction + t.variance * variance + t.covariance * covariance
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    ema(model.target, model.online, momentum(step, t.steps, t.momentum_start, t.momentum_end))
    return {
        "loss": loss.item(),
        "prediction": prediction.item(),
        "variance": variance.item(),
        "covariance": covariance.item(),
    }


def pretrain(
    x: Tensor,
    fit_hours: np.ndarray,
    plan: Plan,
    init: int,
    feed_seed: np.random.SeedSequence,
    log: LOG,
) -> tuple[encoder.Encoder, list[dict[str, Any]], str]:
    model = jepa(x.shape[-1], plan, init)
    initial = weights(model.online)
    opt = optimiser([model.online, model.predictor], plan.jepa)
    rows = sample_rows(fit_hours, x.shape[0])
    feed = batches(rows, len(fit_hours), plan.jepa.batch, feed_seed)
    curve = []
    for step in range(plan.jepa.steps):
        record = jepa_step(model, opt, x, next(feed), step, plan)
        if step % plan.jepa.curve_every == 0 or step == plan.jepa.steps - 1:
            curve.append({"step": step, **record})
            log(f"      jepa step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in record.items()))
    return model.online, curve, initial


class Supervised(nn.Module):
    def __init__(self, channels: int, plan: Plan, seed: int) -> None:
        super().__init__()
        self.encoder = encoder.encoder(channels, plan.shape, seed, plan.supervised.dropout)
        self.head = nn.Linear(plan.shape.width, len(plan.outputs))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))


def targets(
    grid: Mapping[str, np.ndarray], plan: Plan, ncoins: int
) -> tuple[Tensor, Tensor, np.ndarray]:
    y = np.stack([grid[f"{side}_{h}"] for side, h in plan.outputs], axis=-1)
    keep = np.stack([~grid[f"censored_{h}"] for _, h in plan.outputs], axis=-1)
    return (
        torch.from_numpy(by_coin(y, ncoins).astype(np.float32)),
        torch.from_numpy(by_coin(keep, ncoins).astype(np.float32)),
        by_coin(grid["skip"], ncoins),
    )


def train_supervised(
    x: Tensor,
    y: Tensor,
    keep: Tensor,
    rows: np.ndarray,
    t: Training,
    plan: Plan,
    init: int,
    feed_seed: np.random.SeedSequence,
    log: LOG,
    check: Callable[[Supervised], dict[str, float]] | None = None,
    stop: int | None = None,
) -> tuple[Supervised, list[dict[str, Any]], list[dict[str, Any]]]:
    model = Supervised(x.shape[-1], plan, init)
    opt = optimiser([model], t)
    feed = batches(rows, x.shape[1] - plan.pad, t.batch, feed_seed)
    curve, checked = [], []
    last = t.steps if stop is None else stop
    for step in range(last):
        schedule(opt, step, t)
        coin, hour = next(feed)
        logits = model(windows(x, coin, hour + plan.pad, plan.window))
        mask = keep[coin, hour]
        bce = F.binary_cross_entropy_with_logits(logits, y[coin, hour], reduction="none")
        loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % t.curve_every == 0 or step == last - 1:
            curve.append({"step": step, "loss": loss.item()})
            log(f"      supervised step {step}: loss {loss.item():.4f}")
        if check is not None and (step + 1) % plan.selection.every == 0:
            checked.append({"step": step + 1, **check(model)})
            log(f"      selection step {step + 1}: auc {checked[-1]['auc']:.4f}")
    return model, curve, checked


def validation(
    x: Tensor,
    y: Tensor,
    keep: Tensor,
    skip: np.ndarray,
    hours_ok: np.ndarray,
    plan: Plan,
    seed: np.random.SeedSequence,
) -> Callable[[Supervised], dict[str, float]]:
    h = max(plan.ladder.horizons)
    cols = [plan.outputs.index((side, h)) for side in ("long", "short")]
    allowed = hours_ok[None, :] & ~skip & (keep.numpy()[..., cols] > 0).all(axis=-1)
    coin, hour = np.nonzero(allowed)
    label = y.numpy()[coin, hour][:, cols] > 0.5
    u = np.random.default_rng(seed).random(len(coin))
    both = ~label.any(axis=1)
    sampled = np.ones(len(coin), bool)
    if both.sum() > plan.selection.negatives:
        sampled = u <= np.sort(u[both])[plan.selection.negatives - 1]
    take = np.flatnonzero(~both | sampled)
    coin, hour, label, sampled = coin[take], hour[take], label[take], sampled[take]

    def check(model: Supervised) -> dict[str, float]:
        logits = encode(model, x, coin, hour, plan)
        scores = [
            auc(logits[label[:, i], c], logits[~label[:, i] & sampled, c])
            for i, c in enumerate(cols)
        ]
        finite = [s for s in scores if np.isfinite(s)]
        return {
            "auc_long": scores[0],
            "auc_short": scores[1],
            "auc": float(np.mean(finite)) if finite else float("nan"),
            "rows": len(take),
        }

    return check


def supervise(
    x: Tensor,
    y: Tensor,
    keep: Tensor,
    skip: np.ndarray,
    common: np.ndarray,
    train_hours: np.ndarray,
    plan: Plan,
    key: tuple[str, int, int],
    log: LOG,
) -> tuple[Supervised, list[dict[str, Any]], list[dict[str, Any]], int, str]:
    variant, f, seed = key
    ncoins, nhours = x.shape[0], len(train_hours)
    t = plan.supervised
    init = torch_seed(variant, f, seed, "init")
    hidden = skip | ~common
    exclude = hidden | (keep.numpy() == 0).all(axis=-1)
    fit_hours = eligible(train_hours, plan.pad, plan.reach)
    positions = np.flatnonzero(train_hours)
    start = positions[int(len(positions) * (1 - plan.selection.share))]
    block = train_hours & (np.arange(nhours) >= start)
    early = train_hours & (np.arange(nhours) < start - plan.ladder.gap_days * 24)
    rows = sample_rows(eligible(early, plan.pad, plan.reach), ncoins, exclude)
    steps, checked = t.steps, []
    if rows.size:
        check = validation(x, y, keep, hidden, fit_hours & block, plan, stream(*key, "validation"))
        _, _, checked = train_supervised(
            x, y, keep, rows, t, plan, init, stream(*key, "selection"), log, check
        )
        finite = [c for c in checked if np.isfinite(c["auc"])]
        if finite:
            steps = max(finite, key=lambda c: c["auc"])["step"]
    else:
        log("      no selection: no training row lies a gap before the validation block")
    model, curve, _ = train_supervised(
        x,
        y,
        keep,
        sample_rows(fit_hours, ncoins, exclude),
        t,
        plan,
        init,
        stream(*key, "supervised"),
        log,
        stop=steps,
    )
    return model, curve, checked, steps, weights(Supervised(x.shape[-1], plan, init).encoder)


@torch.inference_mode()
def encode(
    model: nn.Module, x: Tensor, coin: np.ndarray, hour: np.ndarray, plan: Plan
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(coin), plan.encode_batch):
        c = torch.from_numpy(coin[i : i + plan.encode_batch])
        h = torch.from_numpy(hour[i : i + plan.encode_batch])
        out.append(model(windows(x, c, h + plan.pad, plan.window)).double().numpy())
    model.train()
    width = plan.shape.width if not isinstance(model, Supervised) else len(plan.outputs)
    return np.concatenate(out) if out else np.empty((0, width))


def readout(tokens: Tensor) -> Tensor:
    spans = (*SPANS, tokens.shape[1])
    means = [tokens[:, -s:].mean(dim=1) for s in spans]
    spreads = [tokens[:, -s:].std(dim=1, unbiased=False) for s in spans]
    return torch.cat([tokens[:, -1], *means, *spreads], dim=1)


@torch.inference_mode()
def features(
    net: encoder.Encoder, x: Tensor, coin: np.ndarray, hour: np.ndarray, plan: Plan
) -> np.ndarray:
    net.eval()
    out = []
    for i in range(0, len(coin), plan.encode_batch):
        c = torch.from_numpy(coin[i : i + plan.encode_batch])
        h = torch.from_numpy(hour[i : i + plan.encode_batch])
        out.append(readout(net.tokens(windows(x, c, h + plan.pad, plan.window))).numpy())
    net.train()
    width = READOUTS["spreads"] * plan.shape.width
    return np.concatenate(out) if out else np.empty((0, width), np.float32)


def either(long: np.ndarray, short: np.ndarray) -> np.ndarray:
    total = np.logaddexp(0.0, long) + np.logaddexp(0.0, short)
    with np.errstate(divide="ignore"):
        return np.where(total > 30, total, np.log(np.expm1(np.minimum(total, 30))))


def probe(
    train: np.ndarray,
    test: np.ndarray,
    y: Mapping[str, np.ndarray],
    plan: Plan,
    fitted: Any,
    configs: Sequence[tuple[str, float]],
) -> dict[tuple[str, float], dict[str, np.ndarray]]:
    out = {}
    for name in dict.fromkeys(n for n, _ in configs):
        k = READOUTS[name] * plan.shape.width
        a, b = ladder.standardise(
            train[:, :k].astype(np.float64), test[:, :k].astype(np.float64), plan.ladder.clip
        )
        fitted.update(a.tobytes())
        for c in (c for n, c in configs if n == name):
            sides = {}
            for side in ("long", "short"):
                beta = ladder.fit(a, y[side].astype(np.float64), c)
                fitted.update(beta.tobytes())
                sides[side] = b @ beta[:-1] + beta[-1]
            sides["any"] = either(sides["long"], sides["short"])
            out[(name, c)] = sides
    return out


def encoder_name(variant: str, scheme: str, f: int, seed: int, model: str) -> str:
    return f"encoders/{variant}-{scheme}-fold{f}-seed{seed}-{model}.pt"


def save(net: nn.Module) -> bytes:
    buffer = BytesIO()
    torch.save(net.state_dict(), buffer)
    return buffer.getvalue()


def restore(net: nn.Module, data: bytes) -> None:
    net.load_state_dict(torch.load(BytesIO(data), weights_only=True))


def describe(z: np.ndarray) -> dict[str, float]:
    if len(z) < 2:
        return {"z_sd_min": float("nan"), "z_sd_median": float("nan"), "z_effective_rank": 0.0}
    sd = z.std(axis=0)
    eig = np.clip(np.linalg.eigvalsh(np.cov(z, rowvar=False)), 0.0, None)
    rank = 0.0
    if eig.sum() > 0:
        p = eig / eig.sum()
        p = p[p > 0]
        rank = float(np.exp(-(p * np.log(p)).sum()))
    return {
        "z_sd_min": float(sd.min()),
        "z_sd_median": float(np.median(sd)),
        "z_effective_rank": rank,
    }


@dataclass
class Rows:
    keep: np.ndarray
    accounting: list[dict[str, Any]]
    y: dict[str, np.ndarray]
    fold: np.ndarray


def common_rows(
    values: Mapping[str, np.ndarray], grid: Mapping[str, np.ndarray], plan: Plan
) -> dict[int, Rows]:
    edges = ladder.bounds(plan.ladder)
    out = {}
    for h in plan.ladder.horizons:
        keep, accounting = ladder.select(values, grid, plan.ladder, h)
        y = {"long": grid[f"long_{h}"][keep], "short": grid[f"short_{h}"][keep]}
        y["any"] = y["long"] | y["short"]
        fold, _ = ladder.folds(grid["ts_ns"][keep], edges, plan.ladder.block_days)
        out[h] = Rows(keep, accounting, y, fold)
    return out


def sequence(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    plan: Plan,
    log: LOG = lambda line: None,
) -> dict[str, Any]:
    hours, coins = layout(grid)
    ncoins = len(coins)
    edges = ladder.bounds(plan.ladder)
    rows = common_rows(values, grid, plan)
    ts = grid["ts_ns"]
    row_fold, _ = ladder.folds(ts, edges, plan.ladder.block_days)
    y, keep, skip = targets(grid, plan, ncoins)
    kept = np.any([r.keep for r in rows.values()], axis=0)
    common = by_coin(kept, ncoins)
    configs = [(name, c) for name in plan.readouts for c in plan.cs]
    width = plan.shape.width
    diagnostics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    encoders: dict[str, bytes] = {}
    for variant in plan.variants:
        raw = by_coin(np.column_stack([values[c] for c in plan.rung(variant).columns]), ncoins)
        for scheme in plan.schemes:
            for f in plan.folds:
                if f < plan.ladder.first_fold(scheme):
                    continue
                train_hours = training(hours, plan, f, scheme)
                fit_hours = eligible(train_hours, plan.pad, plan.reach)
                if not fit_hours.any():
                    skipped.append(f"{variant} {scheme} fold {f + 1}: no training row")
                    log(f"  {skipped[-1]}")
                    continue
                fit_rows = np.repeat(fit_hours, ncoins)
                mu, sd = statistics(raw, train_hours)
                x = torch.from_numpy(standardised(raw, mu, sd, plan.ladder.clip, plan.pad))
                need_train, need_test = kept & fit_rows, kept & (row_fold == f)
                at_train, at_test = np.cumsum(need_train) - 1, np.cumsum(need_test) - 1
                test_coin, test_hour = _coin_hour(need_test, ncoins)
                for seed in plan.seeds:
                    log(f"  {variant}, {scheme} fold {f + 1}, seed {seed}")
                    tag = {"fold": f, "scheme": scheme, "seed": seed, "variant": variant}
                    init = torch_seed(variant, f, seed, "init")
                    found: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                    fitted: dict[str, Any] = {}
                    mine: list[dict[str, Any]] = []
                    for model_name in ("twin", "jepa"):
                        if model_name not in plan.models:
                            continue
                        began = time.perf_counter()
                        curve: list[dict[str, Any]] = []
                        if model_name == "twin":
                            net = encoder.encoder(x.shape[-1], plan.shape, init)
                            initial = weights(net)
                        else:
                            feed = stream(variant, f, seed, "jepa")
                            net, curve, initial = pretrain(x, fit_hours, plan, init, feed, log)
                        train_f = features(net, x, *_coin_hour(need_train, ncoins), plan)
                        test_f = features(net, x, test_coin, test_hour, plan)
                        found[model_name] = (train_f, test_f)
                        fitted[model_name] = _fitted(mu, sd, net, train_f)
                        encoders[encoder_name(variant, scheme, f, seed, model_name)] = save(net)
                        mine.append(
                            tag
                            | _diagnostic(model_name, plan, curve, test_f[:, :width], net)
                            | {"initial": initial, "weights": weights(net)}
                            | {"seconds": time.perf_counter() - began}
                        )
                        curves += [tag | {"model": model_name, **c} for c in curve]
                    if "supervised" in plan.models:
                        began = time.perf_counter()
                        model, curve, checked, steps, initial = supervise(
                            x, y, keep, skip, common, train_hours, plan, (variant, f, seed), log
                        )
                        logits = encode(model, x, test_coin, test_hour, plan)
                        found["supervised"] = (logits, logits)
                        fitted["supervised"] = _fitted(mu, sd, model, np.array([steps]))
                        encoders[encoder_name(variant, scheme, f, seed, "supervised")] = save(model)
                        z = encode(model.encoder, x, test_coin, test_hour, plan)
                        mine.append(
                            tag
                            | _diagnostic("supervised", plan, curve, z, model.encoder)
                            | {"steps": steps, "initial": initial}
                            | {"weights": weights(model.encoder)}
                            | {"seconds": time.perf_counter() - began}
                        )
                        curves += [tag | {"model": "supervised", **c} for c in curve]
                        curves += [tag | {"model": "selection", **c} for c in checked]
                    for h, r in rows.items():
                        test = r.fold == f
                        train = fit_rows[r.keep]
                        if not test.any():
                            continue
                        if not train.any():
                            skipped.append(f"{variant} {scheme} fold {f + 1} {h} h: no probe row")
                            log(f"  {skipped[-1]}")
                            continue
                        index = np.flatnonzero(r.keep)
                        a, b = at_train[index[train]], at_test[index[test]]
                        y_train = {side: r.y[side][train] for side in ("long", "short")}
                        got: dict[tuple[str, str | None, float | None], dict[str, np.ndarray]] = {}
                        for model_name in ("jepa", "twin"):
                            if model_name in found:
                                train_f, test_f = found[model_name]
                                probed = probe(
                                    train_f[a],
                                    test_f[b],
                                    y_train,
                                    plan,
                                    fitted[model_name],
                                    configs,
                                )
                                for (name, c), sides in probed.items():
                                    got[(model_name, name, c)] = sides
                        if "supervised" in found:
                            p = found["supervised"][1][b]
                            sides = {
                                side: p[:, plan.outputs.index((side, h))]
                                for side in ("long", "short")
                            }
                            sides["any"] = either(sides["long"], sides["short"])
                            got[("supervised", None, None)] = sides
                        for (model_name, name, c), sides in got.items():
                            for side in plan.ladder.sides:
                                label = r.y[side][test]
                                s = sides[side]
                                fold_rows.append(
                                    {"horizon": h, "side": side, "model": model_name}
                                    | {"readout": name, "c": c}
                                    | tag
                                    | {
                                        "auc": auc(s[label], s[~label]),
                                        "rows": int(test.sum()),
                                        "positives": int(label.sum()),
                                    }
                                )
                    diagnostics += [d | {"fitted": fitted[d["model"]].hexdigest()} for d in mine]
    diagnostics.sort(key=cell_order(plan))
    curves.sort(key=cell_order(plan))
    fold_rows.sort(key=cell_order(plan))
    fold_rows += sorted(rung_folds(values, rows, ts, edges, plan, log), key=rung_order(plan))
    return {
        "diagnostics": diagnostics,
        "curves": curves,
        "folds": fold_rows,
        "accounting": {h: r.accounting for h, r in rows.items()},
        "skipped": sorted(set(skipped)),
        "encoders": encoders,
    }


def choose(
    folds: Sequence[Mapping[str, Any]], model: str, variant: str
) -> tuple[str, float, float]:
    if model not in ("jepa", "twin"):
        raise ValueError(f"the {model} model has no readout or c to choose")
    cells: dict[tuple[str, float], dict[str, dict[int, list[float]]]] = {}
    for r in folds:
        mine = (r["model"], r["variant"], r["scheme"], r["horizon"]) == (
            model,
            variant,
            "forward",
            24,
        )
        if mine:
            per = cells.setdefault((r["readout"], r["c"]), {}).setdefault(r["side"], {})
            per.setdefault(r["fold"], []).append(r["auc"])
    order = list(READOUTS)
    best: tuple[str, float, float] | None = None
    for name, c in sorted(cells, key=lambda k: (order.index(k[0]), k[1])):
        means = []
        for side in ("long", "short"):
            per_fold = [float(np.mean(v)) for v in cells[(name, c)].get(side, {}).values()]
            finite = [v for v in per_fold if np.isfinite(v)]
            means.append(float(np.mean(finite)) if finite else float("nan"))
        score = float(np.mean(means))
        if np.isfinite(score) and (best is None or score > best[2]):
            best = (name, c, score)
    if best is None:
        raise ValueError(f"no forward 24 h fold scores of {model} on {variant} to choose from")
    return best


def external(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    plan: Plan,
    choice: tuple[str, str, str | None, float | None],
    fetch: Callable[[str], bytes],
    log: LOG = lambda line: None,
) -> dict[str, pa.Table]:
    model, variant, name, c = choice
    configs: list[tuple[str, float]] = []
    if model != "supervised":
        if name is None or c is None:
            raise ValueError(f"the {model} needs a readout and a c")
        configs.append((name, c))
    hours, coins = layout(grid)
    ncoins = len(coins)
    edges = ladder.bounds(plan.ladder)
    rows = common_rows(values, grid, plan)
    ts, symbol = grid["ts_ns"], grid["symbol"].astype(str)
    row_fold, _ = ladder.folds(ts, edges, plan.ladder.block_days)
    kept = np.any([r.keep for r in rows.values()], axis=0)
    raw = by_coin(np.column_stack([values[k] for k in plan.rung(variant).columns]), ncoins)
    out = {}
    for scheme in plan.schemes:
        first = plan.ladder.first_fold(scheme)
        quarters = sorted({int(f) for r in rows.values() for f in np.unique(r.fold) if f >= first})
        missing = [f + 1 for f in quarters if f not in plan.folds]
        if missing:
            raise ValueError(f"the plan's folds leave {scheme} quarters {missing} unscored")
        total = {h: np.zeros((int(r.keep.sum()), 2)) for h, r in rows.items()}
        count = {h: np.zeros(int(r.keep.sum()), np.int64) for h, r in rows.items()}
        for f in quarters:
            train_hours = training(hours, plan, f, scheme)
            fit_rows = np.repeat(eligible(train_hours, plan.pad, plan.reach), ncoins)
            mu, sd = statistics(raw, train_hours)
            x = torch.from_numpy(standardised(raw, mu, sd, plan.ladder.clip, plan.pad))
            need_train, need_test = kept & fit_rows, kept & (row_fold == f)
            at_train, at_test = np.cumsum(need_train) - 1, np.cumsum(need_test) - 1
            for seed in sorted(plan.seeds):
                key = encoder_name(variant, scheme, f, seed, model)
                data = fetch(key)
                log(f"  {scheme} fold {f + 1}, seed {seed}: {key}")
                if configs:
                    net = encoder.encoder(x.shape[-1], plan.shape, 0)
                    restore(net, data)
                    train_f = features(net, x, *_coin_hour(need_train, ncoins), plan)
                    test_f = features(net, x, *_coin_hour(need_test, ncoins), plan)
                else:
                    model_net = Supervised(x.shape[-1], plan, 0)
                    restore(model_net, data)
                    train_f = test_f = encode(model_net, x, *_coin_hour(need_test, ncoins), plan)
                for h, r in rows.items():
                    test = r.fold == f
                    if not test.any():
                        continue
                    train = fit_rows[r.keep]
                    index = np.flatnonzero(r.keep)
                    b = at_test[index[test]]
                    if not configs:
                        cols = [plan.outputs.index((side, h)) for side in ("long", "short")]
                        total[h][test] += test_f[b][:, cols]
                    else:
                        if not train.any():
                            raise ValueError(f"{scheme} fold {f + 1} {h} h: no probe row")
                        a = at_train[index[train]]
                        y_train = {side: r.y[side][train] for side in ("long", "short")}
                        got = probe(train_f[a], test_f[b], y_train, plan, hashlib.sha256(), configs)
                        total[h][test] += np.column_stack(
                            [got[configs[0]][side] for side in ("long", "short")]
                        )
                    count[h][test] += 1
        parts = []
        for h, r in rows.items():
            scored = r.fold >= first
            if (count[h][scored] != len(plan.seeds)).any():
                raise RuntimeError(f"{scheme} {h} h: a scored row lacks a score from every seed")
            mean = total[h][scored] / len(plan.seeds)
            sides = {"long": mean[:, 0], "short": mean[:, 1]}
            sides["any"] = either(sides["long"], sides["short"])
            n = int(scored.sum())
            for side in plan.ladder.sides:
                if not np.isfinite(sides[side]).all():
                    raise RuntimeError(f"{scheme} {h} h {side}: a score is not finite")
                parts.append(
                    pa.table(
                        {
                            "coin": pa.array(symbol[r.keep][scored]),
                            "hour": pa.array(ts[r.keep][scored], pa.int64()),
                            "horizon": pa.array(np.full(n, h), pa.int16()),
                            "side": pa.array(np.full(n, side)),
                            "score": pa.array(sides[side], pa.float64()),
                            "fold": pa.array(r.fold[scored], pa.int16()),
                            "scheme": pa.array(np.full(n, scheme)),
                        }
                    )
                )
        out[scheme] = pa.concat_tables(parts)
    return out


def _coin_hour(mask: np.ndarray, ncoins: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.flatnonzero(mask)
    return index % ncoins, index // ncoins


def _fitted(mu: np.ndarray, sd: np.ndarray, net: nn.Module, features: np.ndarray) -> Any:
    h = hashlib.sha256(mu.tobytes() + sd.tobytes())
    h.update(weights(net).encode())
    h.update(np.ascontiguousarray(features).tobytes())
    return h


def _diagnostic(
    model: str,
    plan: Plan,
    curve: Sequence[Mapping[str, float]],
    z: np.ndarray,
    net: nn.Module,
) -> dict[str, Any]:
    return {
        "model": model,
        "target": plan.target if model == "jepa" else "",
        "objective": plan.jepa.objective if model == "jepa" else "",
        "steps": plan.jepa.steps if model == "jepa" else 0,
        "loss_start": float(curve[0]["loss"]) if curve else float("nan"),
        "loss_end": float(curve[-1]["loss"]) if curve else float("nan"),
        **describe(z),
        "parameters": encoder.parameters(net),
    }


def rung_folds(
    values: Mapping[str, np.ndarray],
    rows: Mapping[int, Rows],
    ts: np.ndarray,
    edges: np.ndarray,
    plan: Plan,
    log: LOG,
) -> list[dict[str, Any]]:
    out = []
    for h, r in rows.items():
        x = np.column_stack([values[c][r.keep] for c in plan.ladder.columns])
        for scheme in plan.schemes:
            first = plan.ladder.first_fold(scheme)
            scored = ladder.out_of_fold(
                x, r.y, ts[r.keep], r.fold, edges, plan.ladder, scheme, h, log
            )
            fold = r.fold[r.fold >= first]
            for name in plan.rungs:
                for side in plan.ladder.sides:
                    label = r.y[side][r.fold >= first]
                    for f in plan.folds:
                        if f < first:
                            continue
                        test = fold == f
                        if not test.any():
                            continue
                        s = scored[name][side][test]
                        out.append(
                            {
                                "horizon": h,
                                "side": side,
                                "model": "rung",
                                "readout": None,
                                "c": plan.ladder.c,
                                "variant": name,
                                "scheme": scheme,
                                "seed": 0,
                                "fold": f,
                                "auc": auc(s[label[test]], s[~label[test]]),
                                "rows": int(test.sum()),
                                "positives": int(label[test].sum()),
                            }
                        )
    return out


def read(
    inputs_dir: Path, data_dir: Path, plan: Plan
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    manifest = io.read_json(inputs_dir / "manifest.json")
    params = manifest["params"]
    span.refuse(*(date.fromisoformat(params[k]) for k in ("start", "dev_start", "end")))
    complete = io.read_json(inputs_dir / "complete.json")
    if complete.get("clean") is not True:
        raise ValueError(f"{inputs_dir}: complete.json does not mark the run clean")
    labels_dir = data_dir / "labels" / params["labels_run"]
    grid_path = labels_dir / params["labels_grid"]["file"]
    inputs_path = inputs_dir / "inputs.parquet"
    for path in (grid_path, inputs_path):
        store.refuse_held_back(path)
    if store.sha256(grid_path) != params["labels_grid"]["sha256"]:
        raise ValueError(f"{grid_path}: not the grid the inputs run was made from")
    inputs_sha = store.sha256(inputs_path)
    if complete["files"].get(inputs_path.name) != inputs_sha:
        raise ValueError(f"{inputs_path}: its sha256 differs from complete.json's")
    grid_table = pq.read_table(grid_path)
    columns = list(plan.ladder.columns)
    table = pq.read_table(inputs_path, columns=["ts_ns", "symbol", *columns])
    if not (
        table["ts_ns"].equals(grid_table["ts_ns"]) and table["symbol"].equals(grid_table["symbol"])
    ):
        raise ValueError("the inputs' rows are not the grid's rows")
    values = {n: table[n].to_numpy() for n in columns}
    grid = {n: grid_table[n].to_numpy(zero_copy_only=False) for n in grid_table.column_names}
    source = {
        "inputs_run": manifest["run_id"],
        "inputs": {"file": inputs_path.name, "sha256": inputs_sha},
        "labels_run": params["labels_run"],
        "labels_grid": params["labels_grid"],
    }
    return values, grid, source


def cpu_name() -> str:
    info = Path("/proc/cpuinfo")
    if info.is_file():
        found = re.search(r"^model name\s*:\s*(.+)$", info.read_text(), flags=re.MULTILINE)
        if found:
            return found.group(1).strip()
    return platform.processor() or platform.machine()


def runtime() -> dict[str, Any]:
    return {
        "cpu": cpu_name(),
        "capability": torch.backends.cpu.get_cpu_capability(),
        "pins": {name: os.environ.get(name) for name in PINS},
    }


def curve_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    extra = {k for row in rows for k in row} - set(CURVES.names)
    if extra:
        raise ValueError(f"curve rows carry columns the curve table lacks: {sorted(extra)}")
    return pa.Table.from_pylist(list(rows), schema=CURVES)


def snapshot(path: Path) -> tuple[Plan, dict[str, Any], tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    ladder_path = (path.parent / str(tomllib.loads(text)["ladder"])).resolve()
    texts = (text, ladder_path.read_text(encoding="utf-8"))
    plan = load(path, ladder_path, texts)
    records = [
        {"file": p.name, "sha256": hashlib.sha256(t.encode("utf-8")).hexdigest()}
        for p, t in zip((path, ladder_path), texts, strict=True)
    ]
    return plan, {"plan": records[0], "ladder_plan": records[1]}, texts


def run(
    inputs_dir: Path,
    data_dir: Path,
    plan_path: Path,
    tag: str,
    log: LOG = lambda line: None,
) -> tuple[Path, dict[str, Any]]:
    began = time.perf_counter()
    plan, records, (plan_text, ladder_text) = snapshot(plan_path)
    torch.set_num_threads(plan.threads)
    values, grid, source = read(inputs_dir, data_dir, plan)
    out = sequence(values, grid, plan, log)
    seconds = time.perf_counter() - began
    run_params = {
        **source,
        **records,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "runtime": runtime(),
        "threads": plan.threads,
        "code": {"uv_lock_sha256": labels.uv_lock()},
        "encoders": {k: hashlib.sha256(v).hexdigest() for k, v in out["encoders"].items()},
    }
    tables = {
        "diagnostics": pa.Table.from_pylist(out["diagnostics"]),
        "curves": curve_table(out["curves"]),
        "folds": pa.Table.from_pylist(out["folds"]),
        "rows": pa.Table.from_pylist(
            [{"horizon": h, **row} for h, rs in out["accounting"].items() for row in rs]
        ),
    }
    texts = {
        "report.md": report(out, plan, source, seconds),
        "plan.toml": plan_text,
        "ladder.toml": ladder_text,
    }
    checks = {"seconds": round(seconds, 1), "skipped": out["skipped"]}
    run_dir = store.write(
        data_dir, "panel-sequence", tag, tables, run_params, checks, texts, out["encoders"]
    )
    return run_dir, out | {"seconds": seconds}


def scores_run(
    sequence_dir: Path,
    data_dir: Path,
    model: str,
    variant: str,
    tag: str,
    log: LOG = lambda line: None,
) -> tuple[Path, dict[str, Any]]:
    began = time.perf_counter()
    manifest = io.read_json(sequence_dir / "manifest.json")
    if not (sequence_dir / "complete.json").is_file():
        raise ValueError(f"{sequence_dir}: the run has no complete.json")
    complete = io.read_json(sequence_dir / "complete.json")
    if complete.get("clean") is not True:
        raise ValueError(f"{sequence_dir}: complete.json does not mark the run clean")
    files = complete["files"]
    for name in ("plan.toml", "ladder.toml", "folds.parquet"):
        if store.sha256(sequence_dir / name) != files.get(name):
            raise ValueError(f"{sequence_dir / name}: its sha256 differs from complete.json's")
    plan = load(sequence_dir / "plan.toml", sequence_dir / "ladder.toml")
    torch.set_num_threads(plan.threads)
    if model not in plan.models or variant not in plan.variants:
        raise ValueError(f"the sequence run fitted no {model} on {variant}")
    params = manifest["params"]
    values, grid, source = read(data_dir / "panel-inputs" / params["inputs_run"], data_dir, plan)
    if source["inputs"]["sha256"] != params["inputs"]["sha256"]:
        raise ValueError("the inputs table differs from the one the sequence run read")
    if source["labels_grid"]["sha256"] != params["labels_grid"]["sha256"]:
        raise ValueError("the labels grid differs from the one the sequence run read")
    name: str | None = None
    c: float | None = None
    selection = None
    if model != "supervised":
        folds = pq.read_table(sequence_dir / "folds.parquet").to_pylist()
        name, c, value = choose(folds, model, variant)
        selection = {
            "metric": "mean of the long and short 24 h forward fold-mean AUCs, seeds averaged",
            "value": value,
        }
        log(f"  chose readout {name}, c {c:g}: {value:.4f}")
    choice = (model, variant, name, c)
    saved, used = params["encoders"], {}

    def fetch(key: str) -> bytes:
        path = sequence_dir / key
        if key not in saved or not path.is_file():
            raise ValueError(f"{sequence_dir}: no saved encoder {key}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != saved[key] or digest != files.get(key):
            raise ValueError(f"{path}: its sha256 differs from the sequence run's")
        used[key] = digest
        return data

    tables = external(values, grid, plan, choice, fetch, log)
    seconds = time.perf_counter() - began
    run_params = {
        "sequence_run": manifest["run_id"],
        "model": model,
        "variant": variant,
        "readout": name,
        "c": c,
        "selection": selection,
        "inputs_run": source["inputs_run"],
        "labels_run": source["labels_run"],
        "encoders": used,
        "code": {"uv_lock_sha256": labels.uv_lock()},
    }
    checks = {"seconds": round(seconds, 1)}
    run_dir = store.write(data_dir, "panel-scores", tag, tables, run_params, checks)
    return run_dir, {"tables": tables, "seconds": seconds}


VARYING = ("variants", "seeds", "folds")
SHARED = {
    "inputs table": ("inputs_run", "inputs"),
    "labels grid": ("labels_run", "labels_grid"),
    "ladder plan": ("ladder_plan",),
}


def cells(plan: Plan) -> set[str]:
    return {
        encoder_name(variant, scheme, f, seed, model)
        for variant in plan.variants
        for scheme in plan.schemes
        for f in plan.folds
        if f >= plan.ladder.first_fold(scheme)
        for seed in plan.seeds
        for model in plan.models
    }


def cell_order(plan: Plan) -> Callable[[Mapping[str, Any]], tuple[Any, ...]]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (row["variant"], plan.schemes.index(row["scheme"]), row["fold"], row["seed"])

    return key


def rung_order(plan: Plan) -> Callable[[Mapping[str, Any]], tuple[Any, ...]]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            plan.ladder.horizons.index(row["horizon"]),
            plan.schemes.index(row["scheme"]),
            plan.rungs.index(row["variant"]),
            plan.ladder.sides.index(row["side"]),
            row["fold"],
        )

    return key


def _same(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _piece(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Plan]:
    if not (run_dir / "complete.json").is_file():
        raise ValueError(f"{run_dir}: the run has no complete.json")
    complete = io.read_json(run_dir / "complete.json")
    if complete.get("clean") is not True:
        raise ValueError(f"{run_dir}: complete.json does not mark the run clean")
    manifest = io.read_json(run_dir / "manifest.json")
    if manifest.get("domain") != "panel-sequence":
        raise ValueError(f"{run_dir}: not a sequence run")
    if "sources" in manifest.get("params", {}):
        raise ValueError(f"{run_dir}: a merged run; merge takes sequence runs only")
    producer = manifest.get("producer") or {}
    if not producer.get("git_sha"):
        raise ValueError(f"{run_dir}: the run records no commit for its code")
    if producer.get("dirty") is not False:
        raise ValueError(f"{run_dir}: the run's code had uncommitted changes")
    files = complete["files"]
    for name, digest in files.items():
        if store.sha256(run_dir / name) != digest:
            raise ValueError(f"{run_dir / name}: its sha256 differs from complete.json's")
    params = manifest["params"]
    if any(key not in files or files[key] != digest for key, digest in params["encoders"].items()):
        raise ValueError(
            f"{run_dir}: its manifest's model sha256 values differ from complete.json's"
        )
    for library in ("torch", "numpy"):
        if not params.get(library):
            raise ValueError(f"{run_dir}: the run records no {library} version")
    if not params.get("runtime"):
        raise ValueError(f"{run_dir}: the run records no numerical runtime")
    plan = load(run_dir / "plan.toml", run_dir / "ladder.toml")
    if not cells(plan):
        raise ValueError(f"{run_dir}: fits no cell of its plan")
    return manifest, complete, plan


def _bare(plan: Plan) -> Plan:
    return replace(plan, variants=(), seeds=(), folds=(), ladder_path=Path())


def merge_run(
    pieces: Sequence[Path],
    parent_path: Path,
    data_dir: Path,
    tag: str,
    log: LOG = lambda line: None,
) -> tuple[Path, dict[str, Any]]:
    began = time.perf_counter()
    if len(pieces) < 2:
        raise ValueError("a merge takes at least two sequence runs")
    parent, records, (plan_text, ladder_text) = snapshot(parent_path)
    loaded = [(run_dir, *_piece(run_dir)) for run_dir in pieces]
    seen: set[object] = set()
    for run_dir, manifest, _, _ in loaded:
        for key in (run_dir.resolve(), manifest["run_id"]):
            if key in seen:
                raise ValueError(f"{run_dir.name} is listed twice")
            seen.add(key)
    first_dir, first, _, _ = loaded[0]
    head = first["params"]
    for run_dir, manifest, _, _ in loaded[1:]:
        params = manifest["params"]
        for what, keys in SHARED.items():
            if any(params[k] != head[k] for k in keys):
                raise ValueError(f"{run_dir.name} and {first_dir.name} read different {what}s")
        for library in ("torch", "numpy"):
            if params[library] != head[library]:
                raise ValueError(f"{run_dir.name} and {first_dir.name} ran different {library}")
        if manifest["producer"] != first["producer"] or params["code"] != head["code"]:
            raise ValueError(f"{run_dir.name} and {first_dir.name} ran different code")
        mine, theirs = params["runtime"], head["runtime"]
        if mine["capability"] != theirs["capability"]:
            raise ValueError(
                f"{run_dir.name} and {first_dir.name} ran torch at different CPU capabilities"
            )
        if mine["pins"] != theirs["pins"]:
            raise ValueError(
                f"{run_dir.name} and {first_dir.name} set different instruction-set pins"
            )
    if head["ladder_plan"] != records["ladder_plan"]:
        raise ValueError(f"{first_dir.name} read a different ladder plan from the parent plan's")
    owner: dict[str, Path] = {}
    for run_dir, manifest, _, plan in loaded:
        if _bare(plan) != _bare(parent):
            raise ValueError(
                f"{run_dir.name}: its plan differs from the parent plan in more than "
                + ", ".join(VARYING)
            )
        for key in VARYING:
            extra = sorted(set(getattr(plan, key)) - set(getattr(parent, key)))
            if extra:
                raise ValueError(
                    f"{run_dir.name} fitted {key} the parent plan lacks: "
                    + ", ".join(map(str, extra))
                )
        mine = cells(plan)
        if set(manifest["params"]["encoders"]) != mine:
            raise ValueError(f"{run_dir.name}: its saved models are not every cell of its plan")
        for key in sorted(mine):
            if key in owner:
                raise ValueError(f"{key} is in both {owner[key].name} and {run_dir.name}")
            owner[key] = run_dir
    missing = sorted(cells(parent) - set(owner))
    if missing:
        raise ValueError(
            f"no run fitted {len(missing)} cells of the parent plan: {missing[0]}, ..."
        )

    cell, rung = cell_order(parent), rung_order(parent)
    merged: dict[str, list[dict[str, Any]]] = {"diagnostics": [], "curves": [], "folds": []}
    rungs: dict[tuple[Any, ...], dict[str, Any]] = {}
    accounting: list[dict[str, Any]] | None = None
    skipped: list[str] = []
    for run_dir, _, complete, _ in loaded:
        tables = {n: pq.read_table(run_dir / f"{n}.parquet").to_pylist() for n in (*merged, "rows")}
        for name in ("diagnostics", "curves"):
            merged[name] += tables[name]
        for row in tables["folds"]:
            if row["model"] != "rung":
                merged["folds"].append(row)
            elif not _same(rungs.setdefault(rung(row), row), row):
                raise ValueError(f"{run_dir.name}: its rung rows differ from another run's")
        if accounting is None:
            accounting = tables["rows"]
        elif not _same(tables["rows"], accounting):
            raise ValueError(f"{run_dir.name}: its row accounting differs from another run's")
        skipped = sorted({*skipped, *complete["checks"]["skipped"]})
    for name in merged:
        merged[name].sort(key=cell)
    merged["folds"] += [rungs[k] for k in sorted(rungs)]
    encoders = {key: (owner[key] / key).read_bytes() for key in sorted(owner)}
    source = {k: head[k] for k in ("inputs_run", "inputs", "labels_run", "labels_grid")}
    piece_seconds = sum(c["checks"]["seconds"] for _, _, c, _ in loaded)
    run_params = {
        **source,
        **records,
        "torch": head["torch"],
        "numpy": head["numpy"],
        "runtime": {k: head["runtime"][k] for k in ("capability", "pins")},
        "threads": head["threads"],
        "code": head["code"],
        "pieces_producer": first["producer"],
        "encoders": {k: hashlib.sha256(v).hexdigest() for k, v in encoders.items()},
        "sources": [
            {
                "run": m["run_id"],
                "manifest_sha256": store.sha256(d / "manifest.json"),
                "cpu": m["params"]["runtime"]["cpu"],
            }
            for d, m, _, _ in loaded
        ],
    }
    out = {**merged, "skipped": skipped}
    names = ", ".join(f"`{m['run_id']}`" for _, m, _, _ in loaded)
    texts = {
        "report.md": report(out, parent, source, piece_seconds)
        + f"\nMerged from {names}; the wall time is the sum of theirs.\n",
        "plan.toml": plan_text,
        "ladder.toml": ladder_text,
    }
    tables = {name: pa.Table.from_pylist(rows) for name, rows in merged.items()}
    tables["curves"] = curve_table(merged["curves"])
    tables["rows"] = pa.Table.from_pylist(accounting or [])
    seconds = time.perf_counter() - began
    checks = {"seconds": round(seconds, 1), "pieces_seconds": piece_seconds, "skipped": skipped}
    run_dir = store.write(
        data_dir, "panel-sequence", tag, tables, run_params, checks, texts, encoders
    )
    log(f"merged {len(loaded)} runs, {len(encoders)} models")
    return run_dir, out | {"seconds": seconds}


def _f(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{value:.3f}"


def report(out: Mapping[str, Any], plan: Plan, source: Mapping[str, Any], seconds: float) -> str:
    folds = list(plan.folds)
    lines = [
        "# Panel sequence models",
        "",
        f"Inputs run `{source['inputs_run']}`, labels run `{source['labels_run']}`; JEPA target "
        f"`{plan.target}`, objective `{plan.jepa.objective}`, seeds "
        f"{', '.join(map(str, plan.seeds))}, schemes {', '.join(plan.schemes)}, left-out quarters "
        f"{', '.join(str(f + 1) for f in folds)}. AUC inside each left-out quarter on the ladder's "
        f"common rows; rung rows are the ladder plan's rungs on the same rows.",
        "",
        f"Wall time {seconds / 60:.1f} minutes.",
    ]
    if out["skipped"]:
        lines += ["", "Skipped: " + "; ".join(out["skipped"]) + "."]
    for h in plan.ladder.horizons:
        lines += ["", f"## {h} h", ""]
        lines += [
            "| model | variant | readout | c | scheme | seed | side | "
            + " | ".join(f"q{f + 1}" for f in folds)
        ]
        lines[-1] += " | mean |"
        lines.append("|---" * (8 + len(folds)) + "|")
        keyed: dict[tuple[Any, ...], dict[int, float]] = {}
        for row in out["folds"]:
            if row["horizon"] == h:
                key = tuple(
                    row[k] for k in ("model", "variant", "readout", "c", "scheme", "seed", "side")
                )
                keyed.setdefault(key, {})[row["fold"]] = row["auc"]
        for (model, variant, name, c, scheme, seed, side), per in keyed.items():
            cells = [per.get(f, float("nan")) for f in folds]
            mean = float(np.nanmean(cells)) if np.isfinite(cells).any() else float("nan")
            lines.append(
                f"| {model} | {variant} | {name or '-'} | {'-' if c is None else f'{c:g}'} | "
                f"{scheme} | {seed} | {side} | "
                + " | ".join(_f(c) for c in cells)
                + f" | {_f(mean)} |"
            )
    lines += [
        "",
        "## Diagnostics",
        "",
        "| model | variant | scheme | fold | seed | steps | loss start | loss end | z sd min | "
        "z sd median | effective rank | parameters | seconds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in out["diagnostics"]:
        lines.append(
            f"| {d['model']} | {d['variant']} | {d['scheme']} | {d['fold'] + 1} | {d['seed']} | "
            f"{d['steps']} | {_f(d['loss_start'])} | {_f(d['loss_end'])} | {_f(d['z_sd_min'])} | "
            f"{_f(d['z_sd_median'])} | {d['z_effective_rank']:.1f} | {d['parameters']} | "
            f"{d['seconds']:.0f} |"
        )
    return "\n".join(lines) + "\n"
