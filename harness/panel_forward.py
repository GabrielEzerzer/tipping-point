# Checks a panel sequence run's saved JEPA and twin encoders without reading a cascade label:
# ridge from each readout at hour t to the coin's inputs at t+24, alone and beyond the input row at
# t, and to its inputs at t, the last token's 24-hour persistence, and two stated rules.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tipping_point import engine
from tipping_point.ising import io
from tipping_point.panel import encoder, inputs, ladder, sequence, span, store

AHEAD = 24
PRIMARY = ("vol_24h", "oi_24h")
LEVELS = (*PRIMARY, "aret_24h", "volume_24h", "flow_24h", "near_24h")
CHANGES = ("funding", "ls", "oi_vs_28d")
FORWARD = (*LEVELS, *(f"{name}_change" for name in CHANGES))
RETENTION_GROUPS = {
    "fast directional": ("ret_1h", "flow_1h", "ret_6h"),
    "fast volatility and volume": ("vol_1h", "vsurge_1h", "vol_6h"),
    "slow leverage": ("funding", "ls", "oi_vs_28d"),
}
RETAINED = tuple(name for group in RETENTION_GROUPS.values() for name in group)
TARGET_CHANNELS = frozenset((*LEVELS, *CHANGES, *RETAINED))
MODELS = ("jepa", "twin")
SCHEME = "forward"
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
LATE_SHARE = 0.2
ROWS = 4000
RULE_VARIANT = "full"
RULE_READOUT = "last"
RULES = {
    "forward": "raw forward R2",
    "forward_gain": "gain in forward R2 over the inputs at hour t",
}
REPO = Path(__file__).resolve().parents[1]
LOG = Callable[[str], None]
Fit = tuple[float, float, int, int]
TABLE = pa.schema(
    [
        ("variant", pa.string()),
        ("scheme", pa.string()),
        ("fold", pa.int64()),
        ("seed", pa.int64()),
        ("model", pa.string()),
        ("readout", pa.string()),
        ("kind", pa.string()),
        ("target", pa.string()),
        ("metric", pa.string()),
        ("value", pa.float64()),
        ("alpha", pa.float64()),
        ("fit_rows", pa.int64()),
        ("eval_rows", pa.int64()),
    ]
)

DESCRIPTION = f"""\
Does a trained panel encoder's embedding carry forward-looking information its untrained copy
does not, beyond what the last input row gives, and does it keep the inputs? Reads no cascade
label.

  python harness/panel_forward.py --sequence <sequence run folder> --inputs <inputs run folder>
      --out <folder> [--data-dir <store>] [--rows N]

For every variant, forward fold and seed of the sequence run, and for its {" and ".join(MODELS)}
models, the encoder is restored from its saved file (its sha256 checked against the run's manifest
and complete.json) and run on the tensor it was trained on: the variant's inputs standardised by
the fold's training hours, clipped and padded as the sequence step does. Of the labels grid only
ts_ns and symbol are read, and the grid's sha256 is checked against the inputs run.

Fit rows are coin-hours t whose {AHEAD} following hours and whose window lie in the fold's training
span; evaluation rows are the left-out quarter's hours t for which t+{AHEAD} is on the grid. Only
coin-hours whose own-coin inputs at t are not all missing are drawn, at most --rows (default
{ROWS}) of each per variant, fold and seed, the same rows for both models. Each readout (last,
means, spreads) at t is regressed by ridge on:
  forward          the coin's standardised input at t+{AHEAD}: {", ".join(LEVELS)}
                   ({" and ".join(PRIMARY)} primary), and the change from t to t+{AHEAD} of
                   {", ".join(CHANGES)}
  forward_gain     the R2 of the same forward ridge on the variant's standardised input row at t
                   joined with the readout, less forward_inputs: that ridge on the input row alone
                   (model "inputs", readout "none")
  retention        the coin's standardised input at t: {", ".join(RETAINED)}
and persistence is the mean over embedding dimensions of the correlation between the last readout
at t and at t+{AHEAD} on the evaluation rows whose own-coin inputs at t+{AHEAD} are not all missing
either. A channel the variant lacks is skipped and rows with a missing target are dropped. Ridge:
features standardised on the fit rows, the penalty (times the row count) chosen from
{", ".join(f"{a:g}" for a in ALPHAS)} on the last fifth in time of the fit rows, refitted on all
of them; R2 on the evaluation rows against the fit rows' mean.

Writes to --out, which must lie outside the repository and hold no file yet: forward.parquet (one
row per variant, fold, seed, model, readout, kind and target), report.md, ending with two separate
rules on variant {RULE_VARIANT}, readout {RULE_READOUT} (raw forward R2, and its gain over the
inputs), each NOT JUDGED unless both models are present at every seed of the plan in every forward
quarter of the ladder plan, and manifest.json with the commit, the arguments and the sha256 of every
file read. The data dir is --data-dir, else $TP_DATA_DIR, else the engine's per-user data dir; it
holds the labels run the inputs run names.
"""


def refuse_out(out: Path) -> Path:
    out = out.resolve()
    if out == REPO or REPO in out.parents:
        raise ValueError(f"{out}: inside the repository; write the check elsewhere")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out}: already holds files")
    return out


def read_sequence(
    sequence_dir: Path,
) -> tuple[sequence.Plan, dict[str, Any], dict[str, str], dict[str, str]]:
    if not (sequence_dir / "complete.json").is_file():
        raise ValueError(f"{sequence_dir}: the run has no complete.json")
    complete = io.read_json(sequence_dir / "complete.json")
    if complete.get("clean") is not True:
        raise ValueError(f"{sequence_dir}: complete.json does not mark the run clean")
    data = (sequence_dir / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(data).hexdigest()
    if "manifest_sha256" in complete and complete["manifest_sha256"] != manifest_sha:
        raise ValueError(
            f"{sequence_dir / 'manifest.json'}: its sha256 differs from complete.json's"
        )
    manifest = json.loads(data.decode("utf-8"))
    if manifest.get("domain") != "panel-sequence":
        raise ValueError(f"{sequence_dir}: not a sequence run")
    files = complete["files"]
    read = {
        "manifest.json": manifest_sha,
        "complete.json": store.sha256(sequence_dir / "complete.json"),
    }
    for name, key in (("plan.toml", "plan"), ("ladder.toml", "ladder_plan")):
        digest = store.sha256(sequence_dir / name)
        if digest != files.get(name):
            raise ValueError(f"{sequence_dir / name}: its sha256 differs from complete.json's")
        if (manifest["params"].get(key) or {}).get("sha256") != digest:
            raise ValueError(f"{sequence_dir / name}: not the {name} the run's manifest records")
        read[name] = digest
    plan = sequence.load(sequence_dir / "plan.toml", sequence_dir / "ladder.toml")
    if SCHEME not in plan.schemes:
        raise ValueError(f"{sequence_dir}: the run fitted no forward fold")
    if sequence.fit_only(plan):
        raise ValueError(
            f"{sequence_dir}: the run holds only the fit-only fold, with no hour to judge"
        )
    if not set(MODELS) & set(plan.models):
        raise ValueError(f"{sequence_dir}: the run fitted neither a jepa nor a twin")
    return plan, manifest, files, read


def read_inputs(
    inputs_dir: Path, data_dir: Path, columns: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    manifest = io.read_json(inputs_dir / "manifest.json")
    params = manifest["params"]
    span.refuse(*(date.fromisoformat(params[k]) for k in ("start", "dev_start", "end")))
    complete = io.read_json(inputs_dir / "complete.json")
    if complete.get("clean") is not True:
        raise ValueError(f"{inputs_dir}: complete.json does not mark the run clean")
    grid_path = data_dir / "labels" / params["labels_run"] / params["labels_grid"]["file"]
    inputs_path = inputs_dir / "inputs.parquet"
    for path in (grid_path, inputs_path):
        store.refuse_held_back(path)
    grid_sha = store.sha256(grid_path)
    if grid_sha != params["labels_grid"]["sha256"]:
        raise ValueError(f"{grid_path}: not the grid the inputs run was made from")
    inputs_sha = store.sha256(inputs_path)
    if complete["files"].get(inputs_path.name) != inputs_sha:
        raise ValueError(f"{inputs_path}: its sha256 differs from complete.json's")
    grid_table = pq.read_table(grid_path, columns=["ts_ns", "symbol"])
    table = pq.read_table(inputs_path, columns=["ts_ns", "symbol", *columns])
    if not (
        table["ts_ns"].equals(grid_table["ts_ns"]) and table["symbol"].equals(grid_table["symbol"])
    ):
        raise ValueError("the inputs' rows are not the grid's rows")
    values = {n: table[n].to_numpy() for n in columns}
    grid = {n: grid_table[n].to_numpy(zero_copy_only=False) for n in ("ts_ns", "symbol")}
    source = {
        "inputs_run": manifest["run_id"],
        "inputs": {"file": inputs_path.name, "sha256": inputs_sha},
        "labels_run": params["labels_run"],
        "labels_grid": params["labels_grid"],
    }
    read = {
        "inputs/manifest.json": store.sha256(inputs_dir / "manifest.json"),
        "inputs/complete.json": store.sha256(inputs_dir / "complete.json"),
        "inputs/inputs.parquet": inputs_sha,
        f"labels/{params['labels_run']}/{params['labels_grid']['file']}": grid_sha,
    }
    return values, grid, source, read


def sample(
    allowed: np.ndarray, nrows: int, seed: np.random.SeedSequence
) -> tuple[np.ndarray, np.ndarray]:
    index = np.flatnonzero(allowed)
    if len(index) > nrows:
        index = np.sort(np.random.default_rng(seed).choice(index, nrows, replace=False))
    return index // allowed.shape[1], index % allowed.shape[1]


def trading(raw: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    own = [i for i, name in enumerate(columns) if name in inputs.OWN]
    return ~np.isnan(raw[..., own]).all(axis=-1)


def drawn(
    present: np.ndarray,
    fit_hours: np.ndarray,
    eval_hours: np.ndarray,
    nrows: int,
    fit_seed: np.random.SeedSequence,
    eval_seed: np.random.SeedSequence,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray]:
    fit = sample(fit_hours & present, nrows, fit_seed)
    ev = sample(eval_hours & present, nrows, eval_seed)
    return fit, ev, present[ev[0], ev[1] + AHEAD]


def row_hours(
    hours: np.ndarray, plan: sequence.Plan, f: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = sequence.training(hours, plan, f, SCHEME)
    fold, _ = ladder.folds(hours, ladder.bounds(plan.ladder), plan.ladder.block_days)
    fit = sequence.eligible(train, plan.pad, AHEAD)
    evaluation = (fold == f) & (np.arange(len(hours)) + AHEAD < len(hours))
    return train, fit, evaluation


def prepared(
    raw: np.ndarray, train: np.ndarray, columns: Sequence[str], plan: sequence.Plan
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    mu, sd = sequence.statistics(raw, train)
    x = torch.from_numpy(sequence.standardised(raw, mu, sd, plan.ladder.clip, plan.pad))
    clip = plan.ladder.clip
    z = {
        name: np.clip((raw[..., i] - mu[i]) / sd[i], -clip, clip)
        for i, name in enumerate(columns)
        if name in TARGET_CHANNELS
    }
    return x, z


Rows = tuple[np.ndarray, np.ndarray]
Targets = dict[str, tuple[np.ndarray, np.ndarray]]


def forward_targets(z: Mapping[str, np.ndarray], fit: Rows, ev: Rows) -> Targets:
    out: Targets = {}
    for name in (n for n in LEVELS if n in z):
        out[name] = (z[name][fit[0], fit[1] + AHEAD], z[name][ev[0], ev[1] + AHEAD])
    for name in (n for n in CHANGES if n in z):
        out[f"{name}_change"] = (
            z[name][fit[0], fit[1] + AHEAD] - z[name][fit],
            z[name][ev[0], ev[1] + AHEAD] - z[name][ev],
        )
    return out


def retention_targets(z: Mapping[str, np.ndarray], fit: Rows, ev: Rows) -> Targets:
    return {name: (z[name][fit], z[name][ev]) for name in RETAINED if name in z}


def r2(y: np.ndarray, predicted: np.ndarray, reference: float) -> float:
    total = float(((y - reference) ** 2).sum())
    if len(y) == 0 or total <= 0:
        return float("nan")
    return 1.0 - float(((y - predicted) ** 2).sum()) / total


class Solver:
    def __init__(self, a: np.ndarray) -> None:
        self.mean = a.mean(axis=0)
        self.centred = a - self.mean
        self.values, self.vectors = np.linalg.eigh(self.centred.T @ self.centred)
        self.shrink = self.values[:, None] + np.array(ALPHAS) * len(a)

    def predict(self, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
        projected = self.vectors.T @ (self.centred.T @ (y - y.mean()))
        beta = self.vectors @ (projected[:, None] / self.shrink)
        return (rows - self.mean) @ beta + y.mean()


def ridges(a: np.ndarray, hour: np.ndarray, b: np.ndarray, targets: Targets) -> dict[str, Fit]:
    groups: dict[bytes, list[str]] = {}
    for name, (y, _) in targets.items():
        groups.setdefault(np.isfinite(y).tobytes(), []).append(name)
    out: dict[str, Fit] = {}
    for names in groups.values():
        keep = np.isfinite(targets[names[0]][0])
        n_fit = int(keep.sum())
        if n_fit < 2:
            out |= {n: (float("nan"), float("nan"), n_fit, 0) for n in names}
            continue
        fa, fh = a[keep], hour[keep]
        mu, sd = fa.mean(axis=0), fa.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        fa, fb = (fa - mu) / sd, (b - mu) / sd
        cut = np.sort(fh)[int(n_fit * (1 - LATE_SHARE))]
        early, late = fh < cut, fh >= cut
        inner = Solver(fa[early]) if early.sum() >= 2 and late.any() else None
        whole = Solver(fa)
        for name in names:
            y, y_eval = targets[name][0][keep], targets[name][1]
            ok = np.isfinite(y_eval)
            scores = [float("nan")] * len(ALPHAS)
            if inner is not None:
                guesses = inner.predict(y[early], fa[late])
                reference = float(y[early].mean())
                scores = [r2(y[late], guesses[:, i], reference) for i in range(len(ALPHAS))]
            finite = [i for i, s in enumerate(scores) if np.isfinite(s)]
            best = max(finite, key=lambda i: scores[i]) if finite else len(ALPHAS) - 1
            predicted = whole.predict(y, fb[ok])[:, best]
            value = r2(y_eval[ok], predicted, float(y.mean()))
            alpha = ALPHAS[best]
            out[name] = (value, alpha, n_fit, int(ok.sum()))
    return {name: out[name] for name in targets}


def persistence(now: np.ndarray, ahead: np.ndarray) -> float:
    a, b = now - now.mean(axis=0), ahead - ahead.mean(axis=0)
    scale = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(scale > 0, (a * b).sum(axis=0) / scale, np.nan)
    finite = corr[np.isfinite(corr)]
    return float(finite.mean()) if finite.size else float("nan")


def _row(readout: str, kind: str, target: str, fit: Fit, value: float | None = None) -> dict:
    metric = "r2" if value is None else "r2_gain"
    return {"readout": readout, "kind": kind, "target": target, "metric": metric} | {
        "value": fit[0] if value is None else value,
        "alpha": fit[1],
        "fit_rows": fit[2],
        "eval_rows": fit[3],
    }


@dataclass(frozen=True)
class Alone:
    fit: np.ndarray
    eval: np.ndarray
    forward: dict[str, Fit]


def inputs_alone(
    x: torch.Tensor, z: Mapping[str, np.ndarray], plan: sequence.Plan, fit: Rows, ev: Rows
) -> tuple[Alone, list[dict[str, Any]]]:
    rows = x.numpy()
    fit_x = rows[fit[0], fit[1] + plan.pad].astype(np.float64)
    eval_x = rows[ev[0], ev[1] + plan.pad].astype(np.float64)
    found = ridges(fit_x, fit[1], eval_x, forward_targets(z, fit, ev))
    out = [_row("none", "forward_inputs", t, v) for t, v in found.items()]
    return Alone(fit_x, eval_x, found), out


def measure(
    net: encoder.Encoder,
    x: torch.Tensor,
    z: Mapping[str, np.ndarray],
    plan: sequence.Plan,
    fit: Rows,
    ev: Rows,
    alone: Alone,
    later: np.ndarray,
) -> list[dict[str, Any]]:
    width = plan.shape.width
    fit_f = sequence.features(net, x, *fit, plan).astype(np.float64)
    eval_f = sequence.features(net, x, *ev, plan).astype(np.float64)
    ahead = sequence.features(net, x, ev[0][later], ev[1][later] + AHEAD, plan)[:, :width]
    forward, retention = forward_targets(z, fit, ev), retention_targets(z, fit, ev)
    out = []
    for name in sequence.READOUTS:
        k = sequence.READOUTS[name] * width
        a, b = fit_f[:, :k], eval_f[:, :k]
        for target, v in ridges(a, fit[1], b, forward).items():
            out.append(_row(name, "forward", target, v))
        joined = ridges(np.hstack([alone.fit, a]), fit[1], np.hstack([alone.eval, b]), forward)
        for target, v in joined.items():
            out.append(_row(name, "forward_gain", target, v, v[0] - alone.forward[target][0]))
        for target, v in ridges(a, fit[1], b, retention).items():
            out.append(_row(name, "retention", target, v))
    out.append(
        {"readout": "last", "kind": "persistence", "target": "last", "metric": "correlation"}
        | {"value": persistence(eval_f[later, :width], ahead.astype(np.float64))}
        | {"alpha": None, "fit_rows": 0, "eval_rows": int(later.sum())}
    )
    return out


def restored(channels: int, plan: sequence.Plan, data: bytes) -> encoder.Encoder:
    net = encoder.encoder(channels, plan.shape, 0)
    sequence.restore(net, data)
    return net


def check(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    plan: sequence.Plan,
    fetch: Callable[[str], bytes],
    nrows: int,
    log: LOG = lambda line: None,
) -> list[dict[str, Any]]:
    hours, coins = sequence.layout(grid)
    ncoins = len(coins)
    out = []
    for variant in plan.variants:
        columns = plan.rung(variant).columns
        raw = sequence.by_coin(np.column_stack([values[c] for c in columns]), ncoins)
        present = trading(raw, columns)
        for f in plan.folds:
            if f < plan.ladder.first_fold(SCHEME):
                continue
            train, fit_hours, eval_hours = row_hours(hours, plan, f)
            if not sequence.eligible(train, plan.pad, plan.reach).any():
                log(f"  {variant} fold {f + 1}: the sequence run fitted no model")
                continue
            x, z = prepared(raw, train, columns, plan)
            for seed in plan.seeds:
                streams = (sequence.stream(variant, f, seed, n) for n in ("fit", "eval"))
                fit, ev, later = drawn(present, fit_hours, eval_hours, nrows, *streams)
                if not (len(fit[0]) and len(ev[0])):
                    raise ValueError(f"{variant} fold {f + 1}: no fit or no evaluation row")
                tag = {"variant": variant, "scheme": SCHEME, "fold": f, "seed": seed}
                alone, rows = inputs_alone(x, z, plan, fit, ev)
                out += [tag | {"model": "inputs"} | r for r in rows]
                for model in (m for m in MODELS if m in plan.models):
                    key = sequence.encoder_name(variant, SCHEME, f, seed, model)
                    log(f"  {key}")
                    net = restored(x.shape[-1], plan, fetch(key))
                    found = measure(net, x, z, plan, fit, ev, alone, later)
                    out += [tag | {"model": model} | r for r in found]
    return out


def differences(
    rows: Sequence[Mapping[str, Any]], variant: str, readout: str, kind: str, target: str
) -> dict[int, float]:
    cells: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if (r["variant"], r["readout"], r["kind"], r["target"]) == (variant, readout, kind, target):
            cells.setdefault(r["fold"], {}).setdefault(r["model"], []).append(r["value"])
    return {
        f: float(np.mean(m["jepa"]) - np.mean(m["twin"]))
        for f, m in sorted(cells.items())
        if "jepa" in m and "twin" in m
    }


def quarters(plan: sequence.Plan) -> range:
    return range(plan.ladder.first_fold(SCHEME), len(ladder.bounds(plan.ladder)) - 1)


def missing(
    rows: Sequence[Mapping[str, Any]],
    kind: str,
    target: str,
    folds: Sequence[int],
    seeds: Sequence[int],
) -> list[int]:
    found = {
        (r["fold"], r["seed"], r["model"])
        for r in rows
        if (r["variant"], r["readout"], r["kind"], r["target"])
        == (RULE_VARIANT, RULE_READOUT, kind, target)
    }
    return [f for f in folds if any((f, s, m) not in found for s in seeds for m in MODELS)]


def rule(
    rows: Sequence[Mapping[str, Any]], kind: str, folds: Sequence[int], seeds: Sequence[int]
) -> tuple[str, list[str]]:
    judged, passed, parts = bool(folds) and bool(seeds), True, []
    for target in PRIMARY:
        lacking = missing(rows, kind, target, folds, seeds)
        if lacking or not judged:
            judged = False
            named = ", ".join(f"q{f + 1}" for f in lacking) or "the plan names none"
            parts.append(f"{target}: not judged, quarters lacking a seed or model: {named}")
            continue
        d = differences(rows, RULE_VARIANT, RULE_READOUT, kind, target)
        diffs = np.array([d[f] for f in folds], float)
        positive = int((diffs > 0).sum())
        mean = float(diffs.mean())
        passed &= bool(mean > 0 and positive >= len(folds) - 1)
        parts.append(
            f"{target}: mean difference {_f(mean, True)}, "
            f"positive in {positive} of {len(folds)} quarters"
        )
    return ("PASS" if passed else "FAIL") if judged else "NOT JUDGED", parts


def _f(value: float | None, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def report(
    rows: Sequence[Mapping[str, Any]],
    plan: sequence.Plan,
    source: Mapping[str, Any],
    sequence_run: str,
    nrows: int,
    seconds: float,
) -> str:
    folds = sorted({r["fold"] for r in rows})
    lines = [
        "# Panel forward check",
        "",
        f"Sequence run `{sequence_run}`, inputs run `{source['inputs_run']}`. Variants "
        f"{', '.join(plan.variants)}; forward left-out quarters "
        f"{', '.join(f'q{f + 1}' for f in folds)}; seeds {', '.join(map(str, plan.seeds))}; at "
        f"most {nrows} fit and {nrows} evaluation rows per variant, fold and seed, the same for "
        "both models. No cascade label is read: of the labels grid only ts_ns and symbol.",
        "",
        f"Forward: ridge R2 from the readout at hour t to the coin's standardised input at "
        f"t+{AHEAD}, or its change from t, on the left-out quarter's hours. Forward gain: the R2 "
        "of the input row at t joined with the readout less the R2 of the input row alone, so "
        "what the embedding adds about the next day that the last input row does not give "
        "linearly. Retention: R2 to the standardised input at t. Persistence: the mean over "
        f"embedding dimensions of the correlation of the last readout at t and t+{AHEAD}. Fit "
        f"rows are training hours whose window and next {AHEAD} hours lie in the training span.",
        "",
        f"Wall time {seconds / 60:.1f} minutes.",
        "",
        "## Means over seeds and folds",
        "",
        "| variant | kind | target | readout | jepa | twin | jepa - twin |",
        "|---|---|---|---|---|---|---|",
    ]
    keyed: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    for r in rows:
        key = (r["variant"], r["kind"], r["target"], r["readout"])
        keyed.setdefault(key, {}).setdefault(r["model"], []).append(r["value"])
    for (variant, kind, target, readout), per in keyed.items():
        if kind == "forward_inputs":
            continue
        jepa, twin = _mean(per.get("jepa", [])), _mean(per.get("twin", []))
        lines.append(
            f"| {variant} | {kind} | {target} | {readout} | {_f(jepa)} | {_f(twin)} | "
            f"{_f(jepa - twin, True)} |"
        )
    lines += [
        "",
        "## Forward R2 of the input row at t alone",
        "",
        "| variant | target | R2 |",
        "|---|---|---|",
    ]
    for (variant, kind, target, _), per in keyed.items():
        if kind == "forward_inputs":
            lines.append(f"| {variant} | {target} | {_f(_mean(per['inputs']))} |")
    lines += [
        "",
        "## Retention by group (means over targets, seeds and folds)",
        "",
        "| variant | readout | model | " + " | ".join(RETENTION_GROUPS) + " |",
        "|---" * (3 + len(RETENTION_GROUPS)) + "|",
    ]
    for variant in plan.variants:
        for readout in sequence.READOUTS:
            for model in MODELS:
                cells = []
                for group in RETENTION_GROUPS.values():
                    found = [
                        r["value"]
                        for r in rows
                        if (r["variant"], r["readout"], r["model"], r["kind"])
                        == (variant, readout, model, "retention")
                        and r["target"] in group
                    ]
                    cells.append(_f(_mean(found)))
                if any(c != "-" for c in cells):
                    lines.append(f"| {variant} | {readout} | {model} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Forward, jepa minus twin by fold (seeds averaged)",
        "",
        "| variant | readout | kind | target | "
        + " | ".join(f"q{f + 1}" for f in folds)
        + " | mean | positive |",
        "|---" * (6 + len(folds)) + "|",
    ]
    for variant in plan.variants:
        for readout in sequence.READOUTS:
            for kind in RULES:
                for target in FORWARD:
                    d = differences(rows, variant, readout, kind, target)
                    if not d:
                        continue
                    diffs = list(d.values())
                    lines.append(
                        f"| {variant} | {readout} | {kind} | {target} | "
                        + " | ".join(_f(d.get(f, float("nan")), True) for f in folds)
                        + f" | {_f(_mean(diffs), True)} | "
                        + f"{sum(v > 0 for v in diffs)} of {len(diffs)} |"
                    )
    judged = quarters(plan)
    named = ", ".join(f"q{f + 1}" for f in judged)
    lines += [
        "",
        "## Rules",
        "",
        f"Two rules, each judged on its own and never combined. Each reads variant {RULE_VARIANT}, "
        f"readout {RULE_READOUT}, and PASSES when, for both {' and '.join(PRIMARY)}, the mean "
        f"over the {len(judged)} forward left-out quarters of the ladder plan ({named}) of the "
        "seed-averaged difference jepa minus twin is above zero and the difference is positive "
        f"in at least {max(len(judged) - 1, 0)} of the {len(judged)}. A rule is NOT JUDGED unless "
        f"both models are present at every seed of the plan ({', '.join(map(str, plan.seeds))}) "
        "in every one of those quarters. Retention and persistence are reported, not judged.",
    ]
    for number, (kind, what) in enumerate(RULES.items(), start=1):
        verdict, parts = rule(rows, kind, judged, plan.seeds)
        lines += [
            "",
            f"Rule {number}, on the {what} (kind `{kind}`): **{verdict}** ({'; '.join(parts)}).",
        ]
    return "\n".join(lines) + "\n"


def run(
    sequence_dir: Path,
    inputs_dir: Path,
    data_dir: Path,
    out: Path,
    nrows: int = ROWS,
    log: LOG = lambda line: None,
) -> Path:
    began = time.perf_counter()
    if nrows < 2:
        raise ValueError("--rows must be at least 2")
    out = refuse_out(out)
    plan, manifest, files, sequence_read = read_sequence(sequence_dir)
    torch.set_num_threads(plan.threads)
    params = manifest["params"]
    inputs_manifest = io.read_json(inputs_dir / "manifest.json")
    if inputs_manifest.get("run_id") != params["inputs_run"]:
        raise ValueError(f"{inputs_dir}: not the inputs run the sequence run read")
    columns = list(dict.fromkeys(c for v in plan.variants for c in plan.rung(v).columns))
    values, grid, source, inputs_read = read_inputs(inputs_dir, data_dir, columns)
    if source["inputs"]["sha256"] != params["inputs"]["sha256"]:
        raise ValueError("the inputs table differs from the one the sequence run read")
    if source["labels_grid"]["sha256"] != params["labels_grid"]["sha256"]:
        raise ValueError("the labels grid differs from the one the sequence run read")
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

    rows = check(values, grid, plan, fetch, nrows, log)
    table = pa.Table.from_pylist(rows, schema=TABLE)
    seconds = time.perf_counter() - began
    read = {f"sequence/{k}": v for k, v in sequence_read.items()}
    read |= {f"sequence/{k}": v for k, v in sorted(used.items())} | inputs_read
    record = {
        "script": "harness/panel_forward.py",
        "code": io.code_version(),
        "arguments": {
            "sequence": str(sequence_dir),
            "inputs": str(inputs_dir),
            "data_dir": str(data_dir),
            "out": str(out),
            "rows": nrows,
        },
        "sequence_run": manifest["run_id"],
        **source,
        "files": read,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": plan.threads,
        "seconds": round(seconds, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "forward.parquet")
    text = report(rows, plan, source, manifest["run_id"], nrows, seconds)
    (out / "report.md").write_bytes(text.encode("utf-8"))
    io.write_json(out / "manifest.json", record)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sequence", type=Path, required=True, help="the sequence run folder")
    parser.add_argument("--inputs", type=Path, required=True, help="the inputs run folder")
    parser.add_argument("--out", type=Path, required=True, help="a new folder outside the repo")
    parser.add_argument("--data-dir", type=Path, default=engine.data_dir())
    parser.add_argument("--rows", type=int, default=ROWS)
    args = parser.parse_args(argv)
    try:
        out = run(args.sequence, args.inputs, args.data_dir, args.out, args.rows, print)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
