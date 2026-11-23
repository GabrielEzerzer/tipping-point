# Sequence models on the panel (docs/market.md, "Coin panel"): a JEPA, its untrained twin and a
# supervised model share one encoder over the last 336 hourly input rows of a ladder rung, each
# fitted inside each quarter fold on its training hours, then scored on the ladder's common rows.

from __future__ import annotations

import copy
import time
import tomllib
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
LOG = Callable[[str], None]


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


@dataclass(frozen=True)
class Plan:
    ladder: ladder.Plan
    ladder_path: Path
    variants: tuple[str, ...]
    models: tuple[str, ...]
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
        models=tuple(raw["models"]),
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
        encode_batch=int(raw["probe"]["batch"]),
    )
    rungs = {r.name for r in base.rungs}
    problems = [f"variant {v}" for v in plan.variants if v not in rungs]
    problems += [f"model {m}" for m in plan.models if m not in MODELS]
    problems += [f"fold {f}" for f in plan.folds if not 0 <= f < nfolds]
    if plan.target not in TARGETS:
        problems.append(f"target {plan.target}")
    if "jepa" in plan.models and "full" not in rungs:
        problems.append("full+z needs a rung named full")
    lengths = [plan.window, plan.future]
    if any(n % plan.shape.patch for n in lengths) or plan.shape.width % plan.shape.heads:
        problems.append("window lengths must be whole patches and width whole heads")
    if not plan.seeds or len(set(plan.seeds)) != len(plan.seeds):
        problems.append("seeds must be given and distinct")
    if problems:
        raise ValueError(f"{path}: unknown or invalid: {', '.join(problems)}")
    return plan


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
    lo, hi = np.maximum(j - back, 0), j + ahead + 1
    inside = hi <= n
    hi = np.minimum(hi, n)
    return inside & (outside[hi] == outside[lo])


def sample_rows(
    train: np.ndarray, ncoins: int, back: int, ahead: int, exclude: np.ndarray | None = None
) -> np.ndarray:
    allowed = np.broadcast_to(eligible(train, back, ahead), (ncoins, len(train)))
    if exclude is not None:
        allowed = allowed & ~exclude
    return np.flatnonzero(allowed.ravel())


def batches(rows: np.ndarray, nhours: int, batch: int, seed: Sequence[int]) -> Iterator[Tensor]:
    if not rows.size:
        raise ValueError("no row may be sampled for training in this fold")
    rng = np.random.default_rng(list(seed))
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


@dataclass
class Jepa:
    online: encoder.Encoder
    target: encoder.Encoder
    predictor: encoder.Predictor


def jepa(channels: int, plan: Plan, seed: int) -> Jepa:
    online = encoder.encoder(channels, plan.shape, seed)
    target = copy.deepcopy(online).requires_grad_(False)
    slots = plan.future // plan.shape.patch if plan.target == "future" else len(plan.shifts)
    return Jepa(online, target, encoder.Predictor(plan.shape.width, plan.predictor_hidden, slots))


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
        if plan.target == "future":
            seg = windows(x, coin, hour + plan.pad + plan.future, plan.future)
            goal = model.target.tokens(seg)
        else:
            goal = torch.stack(
                [
                    model.target(windows(x, coin, hour + plan.pad + s, plan.window))
                    for s in plan.shifts
                ],
                dim=1,
            )
        goal = F.layer_norm(goal, goal.shape[-1:])
    prediction = (model.predictor(z) - goal).pow(2).mean()
    variance, covariance = vicreg(z)
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
    x: Tensor, train: np.ndarray, plan: Plan, seed: int, log: LOG
) -> tuple[encoder.Encoder, list[dict[str, Any]]]:
    ncoins, nhours = x.shape[0], len(train)
    model = jepa(x.shape[-1], plan, seed)
    opt = optimiser([model.online, model.predictor], plan.jepa)
    rows = sample_rows(train, ncoins, plan.pad, plan.reach)
    feed = batches(rows, nhours, plan.jepa.batch, (seed, 0))
    curve = []
    for step in range(plan.jepa.steps):
        record = jepa_step(model, opt, x, next(feed), step, plan)
        if step % plan.jepa.curve_every == 0 or step == plan.jepa.steps - 1:
            curve.append({"step": step, **record})
            log(f"      jepa step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in record.items()))
    return model.online, curve


class Supervised(nn.Module):
    def __init__(self, channels: int, plan: Plan, seed: int) -> None:
        super().__init__()
        self.encoder = encoder.encoder(channels, plan.shape, seed)
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


def supervise(
    x: Tensor,
    y: Tensor,
    keep: Tensor,
    skip: np.ndarray,
    train: np.ndarray,
    plan: Plan,
    seed: int,
    log: LOG,
) -> tuple[Supervised, list[dict[str, Any]]]:
    ncoins, nhours = x.shape[0], len(train)
    model = Supervised(x.shape[-1], plan, seed)
    t = plan.supervised
    opt = optimiser([model], t)
    rows = sample_rows(train, ncoins, plan.pad, plan.reach, skip)
    feed = batches(rows, nhours, t.batch, (seed, 1))
    curve = []
    for step in range(t.steps):
        schedule(opt, step, t)
        coin, hour = next(feed)
        logits = model(windows(x, coin, hour + plan.pad, plan.window))
        mask = keep[coin, hour]
        bce = F.binary_cross_entropy_with_logits(logits, y[coin, hour], reduction="none")
        loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % t.curve_every == 0 or step == t.steps - 1:
            curve.append({"step": step, "loss": loss.item()})
            log(f"      supervised step {step}: loss {loss.item():.4f}")
    return model, curve


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
    features: np.ndarray,
    y: Mapping[str, np.ndarray],
    train: np.ndarray,
    test: np.ndarray,
    plan: Plan,
) -> dict[str, np.ndarray]:
    a, b = ladder.standardise(features[train], features[test], plan.ladder.clip)
    out = {
        side: ladder.predict(b, ladder.fit(a, y[side][train].astype(np.float64), plan.ladder.c))
        for side in ("long", "short")
    }
    out["any"] = 1 - (1 - out["long"]) * (1 - out["short"])
    return out


def describe(z: np.ndarray) -> dict[str, float]:
    sd = z.std(axis=0)
    eig = np.clip(np.linalg.eigvalsh(np.cov(z, rowvar=False)), 0.0, None)
    p = eig / eig.sum() if eig.sum() > 0 else np.full(len(eig), 1.0 / len(eig))
    p = p[p > 0]
    return {
        "z_sd_min": float(sd.min()),
        "z_sd_median": float(np.median(sd)),
        "z_effective_rank": float(np.exp(-(p * np.log(p)).sum())),
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
    hour_fold, _ = ladder.folds(hours, edges, plan.ladder.block_days)
    rows = common_rows(values, grid, plan)
    ts = grid["ts_ns"]
    row_fold, _ = ladder.folds(ts, edges, plan.ladder.block_days)
    y, keep, skip = targets(grid, plan, ncoins)
    full = plan.rung("full").columns if "jepa" in plan.models else ()
    scores: list[pa.Table] = []
    diagnostics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for variant in plan.variants:
        columns = plan.rung(variant).columns
        raw = by_coin(np.column_stack([values[c] for c in columns]), ncoins)
        for f in plan.folds:
            train_hours = ladder.training(hours, hour_fold, f, edges, plan.ladder.gap_days)
            train_rows = ladder.training(ts, row_fold, f, edges, plan.ladder.gap_days)
            test_rows = row_fold == f
            mu, sd = statistics(raw, train_hours)
            x = torch.from_numpy(standardised(raw, mu, sd, plan.ladder.clip, plan.pad))
            kept = np.any([r.keep for r in rows.values()], axis=0)
            need = kept & (train_rows | test_rows)
            need_coin, need_hour = np.flatnonzero(need) % ncoins, np.flatnonzero(need) // ncoins
            tested = kept & test_rows
            test_coin, test_hour = np.flatnonzero(tested) % ncoins, np.flatnonzero(tested) // ncoins
            for seed in plan.seeds:
                log(f"  {variant}, fold {f + 1}, seed {seed}")
                found: dict[str, np.ndarray] = {}
                for model_name in ("twin", "jepa"):
                    if model_name not in plan.models:
                        continue
                    began = time.perf_counter()
                    curve: list[dict[str, Any]] = []
                    if model_name == "twin":
                        net = encoder.encoder(x.shape[-1], plan.shape, seed)
                    else:
                        net, curve = pretrain(x, train_hours, plan, seed, log)
                    z = np.full((len(ts), plan.shape.width), np.nan)
                    z[need] = encode(net, x, need_coin, need_hour, plan)
                    found[model_name] = z
                    diagnostics.append(
                        _diagnostic(f, seed, model_name, variant, plan, curve, z[tested], net)
                        | {"seconds": time.perf_counter() - began}
                    )
                    curves += [
                        {"fold": f, "seed": seed, "model": model_name, "variant": variant, **c}
                        for c in curve
                    ]
                if "supervised" in plan.models:
                    began = time.perf_counter()
                    model, curve = supervise(x, y, keep, skip, train_hours, plan, seed, log)
                    p = np.full((len(ts), len(plan.outputs)), np.nan)
                    p[tested] = 1 / (1 + np.exp(-encode(model, x, test_coin, test_hour, plan)))
                    found["supervised"] = p
                    z_test = encode(model.encoder, x, test_coin, test_hour, plan)
                    diagnostics.append(
                        _diagnostic(
                            f, seed, "supervised", variant, plan, curve, z_test, model.encoder
                        )
                        | {"seconds": time.perf_counter() - began}
                    )
                    curves += [
                        {"fold": f, "seed": seed, "model": "supervised", "variant": variant, **c}
                        for c in curve
                    ]
                for h, r in rows.items():
                    test = r.fold == f
                    if not test.any():
                        continue
                    train = ladder.training(ts[r.keep], r.fold, f, edges, plan.ladder.gap_days)
                    got: dict[str, dict[str, np.ndarray]] = {}
                    for model_name in ("jepa", "twin"):
                        if model_name in found:
                            got[model_name] = probe(
                                found[model_name][r.keep], r.y, train, test, plan
                            )
                    if "jepa" in found:
                        both = np.column_stack(
                            [*(values[c][r.keep] for c in full), found["jepa"][r.keep]]
                        )
                        got["full+z"] = probe(both, r.y, train, test, plan)
                    if "supervised" in found:
                        p = found["supervised"][r.keep][test]
                        sides = {
                            side: p[:, plan.outputs.index((side, h))] for side in ("long", "short")
                        }
                        sides["any"] = 1 - (1 - sides["long"]) * (1 - sides["short"])
                        got["supervised"] = sides
                    for model_name, sides in got.items():
                        scores.append(
                            _scores(grid, r.keep, test, h, model_name, variant, seed, sides, plan)
                        )
                        for side in plan.ladder.sides:
                            label = r.y[side][test]
                            s = sides[side]
                            fold_rows.append(
                                {
                                    "horizon": h,
                                    "side": side,
                                    "model": model_name,
                                    "variant": variant,
                                    "seed": seed,
                                    "fold": f,
                                    "auc": auc(s[label], s[~label]),
                                    "rows": int(test.sum()),
                                    "positives": int(label.sum()),
                                }
                            )
    fold_rows += rung_folds(values, rows, ts, edges, plan, log)
    return {
        "scores": pa.concat_tables(scores) if scores else pa.table({}),
        "diagnostics": diagnostics,
        "curves": curves,
        "folds": fold_rows,
        "accounting": {h: r.accounting for h, r in rows.items()},
    }


def _diagnostic(
    f: int,
    seed: int,
    model: str,
    variant: str,
    plan: Plan,
    curve: Sequence[Mapping[str, float]],
    z: np.ndarray,
    net: nn.Module,
) -> dict[str, Any]:
    return {
        "fold": f,
        "seed": seed,
        "model": model,
        "variant": variant,
        "target": plan.target if model == "jepa" else "",
        "steps": {"jepa": plan.jepa.steps, "supervised": plan.supervised.steps}.get(model, 0),
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
    variant: str,
    seed: int,
    sides: Mapping[str, np.ndarray],
    plan: Plan,
) -> pa.Table:
    n = int(test.sum())
    coin = grid["symbol"][keep][test].astype(str)
    hour = grid["ts_ns"][keep][test]
    parts = []
    for side in plan.ladder.sides:
        parts.append(
            pa.table(
                {
                    "coin": pa.array(coin).dictionary_encode(),
                    "hour": pa.array(hour, pa.int64()),
                    "horizon": pa.array(np.full(n, horizon), pa.int8()),
                    "side": pa.array([side] * n).dictionary_encode(),
                    "model": pa.array([model] * n).dictionary_encode(),
                    "variant": pa.array([variant] * n).dictionary_encode(),
                    "seed": pa.array(np.full(n, seed), pa.int64()),
                    "score": pa.array(sides[side], pa.float64()),
                }
            )
        )
    return pa.concat_tables(parts)


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
        scored = ladder.out_of_fold(x, r.y, ts[r.keep], r.fold, edges, plan.ladder, log)
        for name, sides in scored.items():
            for side in plan.ladder.sides:
                for f in plan.folds:
                    test = r.fold == f
                    s, label = sides[side][test], r.y[side][test]
                    out.append(
                        {
                            "horizon": h,
                            "side": side,
                            "model": "rung",
                            "variant": name,
                            "seed": 0,
                            "fold": f,
                            "auc": auc(s[label], s[~label]),
                            "rows": int(test.sum()),
                            "positives": int(label.sum()),
                        }
                    )
    return out


def read(
    inputs_dir: Path, data_dir: Path, plan: Plan
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    manifest = io.read_json(inputs_dir / "manifest.json")
    params = manifest["params"]
    span.refuse(*(date.fromisoformat(params[k]) for k in ("start", "dev_start", "end")))
    labels_dir = data_dir / "labels" / params["labels_run"]
    grid_path = labels_dir / params["labels_grid"]["file"]
    if store.sha256(grid_path) != params["labels_grid"]["sha256"]:
        raise ValueError(f"{grid_path}: not the grid the inputs run was made from")
    grid_table = pq.read_table(grid_path)
    if grid_table.num_rows and max(grid_table["ts_ns"].to_numpy()) >= labels.ns(span.HELD_BACK):
        raise span.HeldBackError("the grid holds an hour in the held-back test span")
    inputs_path = inputs_dir / "inputs.parquet"
    inputs_sha = store.sha256(inputs_path)
    complete = io.read_json(inputs_dir / "complete.json")
    if complete["files"].get(inputs_path.name) != inputs_sha:
        raise ValueError(f"{inputs_path}: its sha256 differs from complete.json's")
    columns = list(plan.ladder.columns)
    table = store.read(inputs_dir, "inputs").select(["ts_ns", "symbol", *columns])
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
    checks = {"seconds": round(seconds, 1)}
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
        f"`{plan.target}`, seeds {', '.join(map(str, plan.seeds))}, left-out quarters "
        f"{', '.join(str(f + 1) for f in folds)}. AUC inside each left-out quarter on the ladder's "
        f"common rows; rung rows are the ladder plan's rungs on the same rows.",
        "",
        f"Wall time {seconds / 60:.1f} minutes.",
    ]
    for h in plan.ladder.horizons:
        lines += ["", f"## {h} h", ""]
        lines += ["| model | variant | seed | side | " + " | ".join(f"q{f + 1}" for f in folds)]
        lines[-1] += " | mean |"
        lines.append("|---" * (5 + len(folds)) + "|")
        keyed: dict[tuple[str, str, int, str], dict[int, float]] = {}
        for row in out["folds"]:
            if row["horizon"] == h:
                key = (row["model"], row["variant"], row["seed"], row["side"])
                keyed.setdefault(key, {})[row["fold"]] = row["auc"]
        for (model, variant, seed, side), per in keyed.items():
            cells = [per.get(f, float("nan")) for f in folds]
            mean = float(np.nanmean(cells)) if np.isfinite(cells).any() else float("nan")
            lines.append(
                f"| {model} | {variant} | {seed} | {side} | "
                + " | ".join(_f(c) for c in cells)
                + f" | {_f(mean)} |"
            )
    lines += [
        "",
        "## Diagnostics",
        "",
        "| model | variant | fold | seed | loss start | loss end | z sd min | z sd median | "
        "effective rank | parameters | seconds |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in out["diagnostics"]:
        lines.append(
            f"| {d['model']} | {d['variant']} | {d['fold'] + 1} | {d['seed']} | "
            f"{_f(d['loss_start'])} | {_f(d['loss_end'])} | {_f(d['z_sd_min'])} | "
            f"{_f(d['z_sd_median'])} | {d['z_effective_rank']:.1f} | {d['parameters']} | "
            f"{d['seconds']:.0f} |"
        )
    return "\n".join(lines) + "\n"
