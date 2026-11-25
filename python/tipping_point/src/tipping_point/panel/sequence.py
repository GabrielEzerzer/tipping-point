# Sequence models on the panel (docs/market.md, "Coin panel"): a JEPA, its untrained twin and a
# supervised model share one encoder over the last 336 hourly input rows of a ladder rung, each
# fitted inside each quarter fold on its training hours, then scored on the ladder's common rows.

from __future__ import annotations

import copy
import hashlib
import time
import tomllib
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
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
MODELS = ("jepa", "twin", "supervised")
TARGETS = ("future", "shifted")
OBJECTIVES = ("z", "predictor")
LOG = Callable[[str], None]
KEYS = ("horizon", "side", "model", "variant", "scheme", "seed")


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


def load(path: Path = PLAN) -> Plan:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    ladder_path = (path.parent / raw["ladder"]).resolve()
    base = ladder.load(ladder_path)
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
    )
    rungs = {r.name for r in base.rungs}
    problems = [f"variant {v}" for v in plan.variants if v not in rungs]
    problems += [f"rung {r}" for r in plan.rungs if r not in rungs]
    problems += [f"model {m}" for m in plan.models if m not in MODELS]
    problems += [f"scheme {s}" for s in plan.schemes if s not in ladder.SCHEMES]
    problems += [f"fold {f}" for f in plan.folds if not 0 <= f < nfolds]
    for name in ("variants", "rungs", "models", "schemes", "folds", "seeds"):
        values = getattr(plan, name)
        if not values or len(set(values)) != len(values):
            problems.append(f"{name} must be given and distinct")
    if plan.target not in TARGETS:
        problems.append(f"target {plan.target}")
    if plan.jepa.objective not in OBJECTIVES:
        problems.append(f"objective {plan.jepa.objective}")
    if "jepa" in plan.models and "full" not in rungs:
        problems.append("full+z needs a rung named full")
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
    lengths = [plan.window, plan.future]
    if any(n % plan.shape.patch for n in lengths) or plan.shape.width % plan.shape.heads:
        problems.append("window lengths must be whole patches and width whole heads")
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
    negative = np.flatnonzero(~label.any(axis=1))
    drawn = np.random.default_rng(seed).choice(
        negative, min(len(negative), plan.selection.negatives), replace=False
    )
    take = np.sort(np.concatenate([np.flatnonzero(label.any(axis=1)), drawn]))
    coin, hour, label = coin[take], hour[take], label[take]

    def check(model: Supervised) -> dict[str, float]:
        logits = encode(model, x, coin, hour, plan)
        scores = [auc(logits[label[:, i], c], logits[~label[:, i], c]) for i, c in enumerate(cols)]
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
    train_hours: np.ndarray,
    plan: Plan,
    key: tuple[str, int, int],
    log: LOG,
) -> tuple[Supervised, list[dict[str, Any]], list[dict[str, Any]], int, str]:
    variant, f, seed = key
    ncoins, nhours = x.shape[0], len(train_hours)
    t = plan.supervised
    init = torch_seed(variant, f, seed, "init")
    exclude = skip | (keep.numpy() == 0).all(axis=-1)
    fit_hours = eligible(train_hours, plan.pad, plan.reach)
    positions = np.flatnonzero(train_hours)
    start = positions[int(len(positions) * (1 - plan.selection.share))]
    block = train_hours & (np.arange(nhours) >= start)
    early = train_hours & (np.arange(nhours) < start - plan.ladder.gap_days * 24)
    rows = sample_rows(eligible(early, plan.pad, plan.reach), ncoins, exclude)
    steps, checked = t.steps, []
    if rows.size:
        check = validation(x, y, keep, skip, fit_hours & block, plan, stream(*key, "validation"))
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


def probe(
    train: np.ndarray,
    test: np.ndarray,
    y: Mapping[str, np.ndarray],
    plan: Plan,
    fitted: Any,
) -> dict[str, np.ndarray]:
    a, b = ladder.standardise(train, test, plan.ladder.clip)
    fitted.update(a.tobytes())
    out = {}
    for side in ("long", "short"):
        beta = ladder.fit(a, y[side].astype(np.float64), plan.ladder.c)
        fitted.update(beta.tobytes())
        out[side] = ladder.predict(b, beta)
    out["any"] = 1 - (1 - out["long"]) * (1 - out["short"])
    return out


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
    full = plan.rung("full").columns if "jepa" in plan.models else ()
    kept = np.any([r.keep for r in rows.values()], axis=0)
    scores: list[pa.Table] = []
    diagnostics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    scored: dict[tuple[str, int], set[int]] = {(s, h): set() for s in plan.schemes for h in rows}
    skipped: list[str] = []
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
                test_rows = row_fold == f
                mu, sd = statistics(raw, train_hours)
                x = torch.from_numpy(standardised(raw, mu, sd, plan.ladder.clip, plan.pad))
                need_train, need_test = kept & fit_rows, kept & test_rows
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
                        z_train = np.full((len(ts), plan.shape.width), np.nan)
                        z_test = np.full((len(ts), plan.shape.width), np.nan)
                        z_train[need_train] = encode(net, x, *_coin_hour(need_train, ncoins), plan)
                        z_test[need_test] = encode(net, x, *_coin_hour(need_test, ncoins), plan)
                        found[model_name] = (z_train, z_test)
                        fitted[model_name] = _fitted(mu, sd, net, z_train[need_train])
                        mine.append(
                            tag
                            | _diagnostic(model_name, plan, curve, z_test[need_test], net)
                            | {"initial": initial, "weights": weights(net)}
                            | {"seconds": time.perf_counter() - began}
                        )
                        curves += [tag | {"model": model_name, **c} for c in curve]
                    if "supervised" in plan.models:
                        began = time.perf_counter()
                        model, curve, checked, steps, initial = supervise(
                            x, y, keep, skip, train_hours, plan, (variant, f, seed), log
                        )
                        p = np.full((len(ts), len(plan.outputs)), np.nan)
                        test_coin, test_hour = _coin_hour(need_test, ncoins)
                        p[need_test] = ladder._probability(
                            encode(model, x, test_coin, test_hour, plan)
                        )
                        found["supervised"] = (p, p)
                        fitted["supervised"] = _fitted(mu, sd, model, np.array([steps]))
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
                        y_train = {side: r.y[side][train] for side in ("long", "short")}
                        got: dict[str, dict[str, np.ndarray]] = {}
                        for model_name in ("jepa", "twin"):
                            if model_name in found:
                                z_train, z_test = found[model_name]
                                got[model_name] = probe(
                                    z_train[r.keep][train],
                                    z_test[r.keep][test],
                                    y_train,
                                    plan,
                                    fitted[model_name],
                                )
                        if "jepa" in found:
                            z_train, z_test = found["jepa"]
                            got["full+z"] = probe(
                                np.column_stack(
                                    [
                                        *(values[c][r.keep][train] for c in full),
                                        z_train[r.keep][train],
                                    ]
                                ),
                                np.column_stack(
                                    [*(values[c][r.keep][test] for c in full), z_test[r.keep][test]]
                                ),
                                y_train,
                                plan,
                                fitted["jepa"],
                            )
                        if "supervised" in found:
                            p = found["supervised"][1][r.keep][test]
                            sides = {
                                side: p[:, plan.outputs.index((side, h))]
                                for side in ("long", "short")
                            }
                            sides["any"] = 1 - (1 - sides["long"]) * (1 - sides["short"])
                            got["supervised"] = sides
                        scored[(scheme, h)].add(f)
                        for model_name, sides in got.items():
                            scores.append(
                                _scores(grid, r.keep, test, h, model_name, tag, sides, plan)
                            )
                            for side in plan.ladder.sides:
                                label = r.y[side][test]
                                s = sides[side]
                                fold_rows.append(
                                    {"horizon": h, "side": side, "model": model_name}
                                    | tag
                                    | {
                                        "auc": auc(s[label], s[~label]),
                                        "rows": int(test.sum()),
                                        "positives": int(label.sum()),
                                    }
                                )
                    diagnostics += [d | {"fitted": fitted[d["model"]].hexdigest()} for d in mine]
    table = pa.concat_tables(scores) if scores else pa.table({})
    check(table, fold_rows, grid, rows, scored, plan, ncoins, hours, coins)
    fold_rows += rung_folds(values, rows, ts, edges, plan, log)
    return {
        "scores": table,
        "diagnostics": diagnostics,
        "curves": curves,
        "folds": fold_rows,
        "accounting": {h: r.accounting for h, r in rows.items()},
        "skipped": skipped,
    }


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


def _scores(
    grid: Mapping[str, np.ndarray],
    keep: np.ndarray,
    test: np.ndarray,
    horizon: int,
    model: str,
    tag: Mapping[str, Any],
    sides: Mapping[str, np.ndarray],
    plan: Plan,
) -> pa.Table:
    n = int(test.sum())
    coin = pa.array(grid["symbol"][keep][test].astype(str)).dictionary_encode()
    hour = pa.array(grid["ts_ns"][keep][test], pa.int64())
    parts = []
    for side in plan.ladder.sides:
        parts.append(
            pa.table(
                {
                    "coin": coin,
                    "hour": hour,
                    "horizon": pa.array(np.full(n, horizon), pa.int16()),
                    "side": _constant(side, n),
                    "model": _constant(model, n),
                    "variant": _constant(tag["variant"], n),
                    "scheme": _constant(tag["scheme"], n),
                    "fold": pa.array(np.full(n, tag["fold"]), pa.int16()),
                    "seed": pa.array(np.full(n, tag["seed"]), pa.int64()),
                    "score": pa.array(sides[side], pa.float64()),
                }
            )
        )
    return pa.concat_tables(parts)


def _constant(value: str, n: int) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(pa.array(np.zeros(n, np.int32)), pa.array([value]))


def _codes(column: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    if pa.types.is_dictionary(column.type):
        column = column.cast(column.type.value_type)
    names, codes = np.unique(column.to_numpy(zero_copy_only=False), return_inverse=True)
    return names, codes


def check(
    table: pa.Table,
    fold_rows: Sequence[Mapping[str, Any]],
    grid: Mapping[str, np.ndarray],
    rows: Mapping[int, Rows],
    scored: Mapping[tuple[str, int], set[int]],
    plan: Plan,
    ncoins: int,
    hours: np.ndarray,
    coins: np.ndarray,
) -> None:
    if not table.num_rows:
        if fold_rows:
            raise RuntimeError("the folds table holds rows the scores table lacks")
        return
    score = table["score"].to_numpy()
    if not np.isfinite(score).all():
        raise RuntimeError("a score is not finite")
    coin = np.searchsorted(coins, table["coin"].cast(pa.string()).to_numpy(zero_copy_only=False))
    hour = (table["hour"].to_numpy() - hours[0]) // labels.HOUR_NS
    row = hour * ncoins + coin
    named = {k: _codes(table[k]) for k in KEYS}
    group = np.zeros(table.num_rows, np.int64)
    for k in KEYS:
        group = group * len(named[k][0]) + named[k][1]
    key = group * len(grid["ts_ns"]) + row
    order = np.argsort(key, kind="stable")
    if np.any(np.diff(key[order]) == 0):
        raise RuntimeError("the scores give a key twice")
    fold = table["fold"].to_numpy()
    aucs: dict[tuple[str, ...], list[float]] = {}
    for d in fold_rows:
        aucs.setdefault((*(str(d[k]) for k in KEYS), str(d["fold"])), []).append(d["auc"])
    starts = np.flatnonzero(np.diff(group[order], prepend=-1))
    found = set()
    for a, b in zip(starts, [*starts[1:], len(order)], strict=True):
        picked = order[a:b]
        i = picked[0]
        values = {k: named[k][0][named[k][1][i]] for k in KEYS}
        h, side, scheme = int(values["horizon"]), str(values["side"]), str(values["scheme"])
        found.add(tuple(str(values[k]) for k in KEYS))
        r = rows[h]
        expected = np.flatnonzero(r.keep)[np.isin(r.fold, sorted(scored[(scheme, h)]))]
        if not np.array_equal(np.sort(row[picked]), expected):
            raise RuntimeError(f"the scores do not cover the common test rows exactly: {values}")
        label = grid[f"long_{h}"] | grid[f"short_{h}"] if side == "any" else grid[f"{side}_{h}"]
        for f in sorted(scored[(scheme, h)]):
            mine = picked[fold[picked] == f]
            want = aucs.get((*(str(values[k]) for k in KEYS), str(f)), [])
            got = auc(score[mine][label[row[mine]]], score[mine][~label[row[mine]]])
            if len(want) != 1 or not (got == want[0] or (np.isnan(got) and np.isnan(want[0]))):
                raise RuntimeError(f"fold {f}'s AUC differs from its scores: {values}")
    if found != {key[:-1] for key in aucs}:
        raise RuntimeError("the scores and the folds table name different groups")


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
                        s = scored[name][side][test]
                        out.append(
                            {
                                "horizon": h,
                                "side": side,
                                "model": "rung",
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


def run(
    inputs_dir: Path,
    data_dir: Path,
    plan_path: Path,
    tag: str,
    log: LOG = lambda line: None,
) -> tuple[Path, dict[str, Any]]:
    began = time.perf_counter()
    plan = load(plan_path)
    torch.set_num_threads(plan.threads)
    values, grid, source = read(inputs_dir, data_dir, plan)
    out = sequence(values, grid, plan, log)
    seconds = time.perf_counter() - began
    run_params = {
        **source,
        "plan": {"file": plan_path.name, "sha256": store.sha256(plan_path)},
        "ladder_plan": {"file": plan.ladder_path.name, "sha256": store.sha256(plan.ladder_path)},
        "torch": torch.__version__,
        "threads": plan.threads,
        "code": {"uv_lock_sha256": labels.uv_lock()},
    }
    tables = {
        "scores": out["scores"],
        "diagnostics": pa.Table.from_pylist(out["diagnostics"]),
        "curves": pa.Table.from_pylist(out["curves"]),
        "folds": pa.Table.from_pylist(out["folds"]),
        "rows": pa.Table.from_pylist(
            [{"horizon": h, **row} for h, rs in out["accounting"].items() for row in rs]
        ),
    }
    texts = {
        "report.md": report(out, plan, source, seconds),
        "plan.toml": plan_path.read_text(encoding="utf-8"),
        "ladder.toml": plan.ladder_path.read_text(encoding="utf-8"),
    }
    checks = {"seconds": round(seconds, 1), "skipped": out["skipped"]}
    run_dir = store.write(data_dir, "panel-sequence", tag, tables, run_params, checks, texts)
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
            "| model | variant | scheme | seed | side | " + " | ".join(f"q{f + 1}" for f in folds)
        ]
        lines[-1] += " | mean |"
        lines.append("|---" * (6 + len(folds)) + "|")
        keyed: dict[tuple[str, str, str, int, str], dict[int, float]] = {}
        for row in out["folds"]:
            if row["horizon"] == h:
                key = (row["model"], row["variant"], row["scheme"], row["seed"], row["side"])
                keyed.setdefault(key, {})[row["fold"]] = row["auc"]
        for (model, variant, scheme, seed, side), per in keyed.items():
            cells = [per.get(f, float("nan")) for f in folds]
            mean = float(np.nanmean(cells)) if np.isfinite(cells).any() else float("nan")
            lines.append(
                f"| {model} | {variant} | {scheme} | {seed} | {side} | "
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
