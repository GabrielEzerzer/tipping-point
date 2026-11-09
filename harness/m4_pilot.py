"""The market JEPA's CPU pilot (#30 M4) and its rules, written and committed before any pilot runs.

RUNS. Four trainings: BTCUSDT and ETHUSDT, each at size S (dim 64) and M (dim 96), model seed 100
(never a primary or spare), 20,000 steps, warm-up 1,000 (steps / 20, as on Ising), every other
setting minutes.Config's default, on the CPU in float32. stop_on_collapse is off: the floors are
what this pilot recalibrates, so a run is not cut short by floors set on a toy; the detector's
verdict is still recorded. Each run checkpoints every 500 steps and resumes by the same command.

SPLIT. The training span [2023-01-18, 2023-10-01) (call 5), split as jepa/days.py asks, with the
validation days a block after the training days: training 2023-01-18 .. 2023-08-31 (226 days),
validation 2023-09-01 .. 2023-09-30 (30 days). PROPOSAL: the primaries use the same split.

VALIDATION ERROR (PROPOSAL: a probe, not the JEPA loss). The JEPA's own error is measured against
each model's own target encoder, which moves during training and differs between sizes: on a
1,000-step smoke run (BTC, 10 days) the monitor's val_error fell to 0.44 at step 250 and rose to
0.59 (S) and 0.63 (M) by step 1,000 while effective rank grew, and the richer M latent scored worse
than S. A lower JEPA error can mean a poorer latent, so it can rank neither sizes nor steps. The
validation error is instead a fixed task in input space: a ridge probe (jepa.probes.ridge, alpha
1e-3) from the target encoder's latent of minute t (minutes.readouts) to the mean of each scaled
feature over minute t + 1, over minute pairs both valid, fitted on validation days 1-15 and scored
per day on days 16-30 as the mean over features of the squared error over the fit set's variance
(1 = the fit mean). It is measured for final.pt, for the copy saved at step 10,000, and for the
step-0 twin; the JEPA surprise of final.pt is kept beside it, descriptive. A run is usable when
final.pt's mean probe error is below its twin's.

SIZE (call 4: M unless S reaches the same validation error). S is chosen when, on both assets, both
runs are usable and S matches M: the upper end of the 95% bootstrap interval of the mean over the
scored days of (S - M), paired by day (2,000 draws, seed 0), is at most TOLERANCE = 2% of M's
mean (PROPOSAL: the 2%). Otherwise M. If an M run is not usable, no size is chosen: the owner
decides.

STEPS (research pass 5.2, in the manner of Ising's call 9). On a run of the chosen size, e(s) is
the mean probe error of its model at step s. The primaries train 10,000 steps if, on both assets,
e(10,000) - e(20,000) <= 2% of e(20,000), and 20,000 otherwise: never below 10,000 (about 21
passes over the training minutes), never above 20,000 (the pilot sees no further). The laptop's
seconds per step are recorded but do not set the count: checkpoints make a CPU run of any length
resumable, and the T4 rate the primaries would run at is unmeasured. In a 20,000-step run the
learning rate at step 10,000 is still high, so e(10,000) overstates what a 10,000-step run
reaches: the rule leans to 20,000, which costs time, never validity.

FLOORS. For each floored statistic (spatial_std, here the spread of a latent over the minutes of a
chunk; pooled_std, the spread of chunk means over chunks; effective_rank), h is its minimum over
the usable runs of the chosen size, both assets, at evaluations from step 1,000 on; the new floor
is h / 5 rounded down to one significant figure (PROPOSAL: the 5; Ising's toy had a healthy 0.87
beside its floor 0.1). Patience stays 3. The new floors replace minutes.Config's defaults in a
commit before the primaries, and test_market's collapse mutant must still fire under them; if it
does not, the owner decides. The VICReg weights stay unless a usable pilot run's monitors fire
under the new floors.

  uv run --extra cpu python harness/m4_pilot.py train --asset btc --size M \\
      --data <data>/bybit-book-v3/training --out <data>/m4-pilot      (again to resume)
  uv run --extra cpu python harness/m4_pilot.py decide --out <data>/m4-pilot

`train` writes <out>/<asset>-<size>/jepa/<run_id>/ (the trainer's layout) and, once the training
is complete, <out>/<asset>-<size>/validation.json. --steps and --limit (the last K training and
first K validation days) are for smoke runs only; `decide` refuses a run that used either.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tipping_point.book.features import FEATURES
from tipping_point.ising import io
from tipping_point.jepa import days, minutes, monitor, probes, train

ASSETS = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
SIZES = ("S", "M")
TRAIN_START, VAL_START, END = date(2023, 1, 18), date(2023, 9, 1), date(2023, 10, 1)
SEED = 100
STEPS = 20_000
CHECKPOINT_EVERY = 500
TOLERANCE = 0.02
DRAWS = 2_000
SHORT, LONG = 10_000, 20_000
WINDOW = 1_000  # the warm-up the floors skip
FLOOR_DIVISOR = 5


def split(limit: int | None = None) -> tuple[list[date], list[date]]:
    """Training and validation days; with `limit`, the last `limit` of the first and the first
    `limit` of the second (a smoke run)."""

    def span(a: date, b: date) -> list[date]:
        return [a + timedelta(days=k) for k in range((b - a).days)]

    train_days, val_days = span(TRAIN_START, VAL_START), span(VAL_START, END)
    return (train_days[-limit:], val_days[:limit]) if limit else (train_days, val_days)


def config(asset: str, size: str, steps: int = STEPS, seed: int = SEED) -> minutes.Config:
    return minutes.preset(
        size,
        symbol=ASSETS[asset],
        train_start=TRAIN_START.isoformat(),
        train_end=END.isoformat(),
        steps=steps,
        warmup=max(1, steps // 20),
        seed=seed,
        checkpoint_every=CHECKPOINT_EVERY,
        stop_on_collapse=False,
    )


def _runs(root: Path) -> list[Path]:
    return sorted(p.parent for p in (root / "jepa").glob("*/manifest.json"))


def fit(
    cfg: minutes.Config,
    root: Path,
    tag: str,
    data_dir: Path,
    limit: int | None = None,
    reload_done: bool = False,
) -> tuple[Path, days.Dataset | None]:
    """Start, resume or find finished the one training of `cfg` under `root` (on the SPLIT days);
    returns its folder and its days, loaded when it trains or when `reload_done` asks."""
    found = _runs(root)
    manifest: dict[str, Any] = {}
    if len(found) > 1:
        raise RuntimeError(f"{root}: {len(found)} trainings; keep one")
    if found:
        done = (found[0] / "complete.json").exists()
        if not done and not (found[0] / "checkpoint.pt").exists():
            raise RuntimeError(f"{found[0]}: no checkpoint (stopped while starting); delete it")
        manifest = io.read_json(found[0] / "manifest.json")
        if manifest["params"]["config"] != json.loads(json.dumps(cfg.to_json())):
            raise RuntimeError(f"{found[0]} was started with another config")
    train_days, val_days = split(limit)
    data = None
    if not found or not (found[0] / "complete.json").exists() or reload_done:
        source = days.ParquetSource(data_dir)
        data = days.load(source, cfg.span, cfg.symbol, FEATURES, train_days, val_days)
        # the check minutes.Domain.check_resume makes, here also for a finished run whose
        # days are reloaded (pilot validation): smoke follows the days it trained on, not --limit
        if found and manifest["data"] != json.loads(json.dumps(data.sources)):
            raise RuntimeError(f"{found[0]} was trained on other days (another --limit?)")
    if not found:
        run_dir = train.start(cfg, data, root, tag, "cpu")
    elif not (found[0] / "complete.json").exists():
        run_dir = train.resume(found[0], data, "cpu")
    else:
        run_dir = found[0]
    return run_dir, data


def run(
    asset: str, size: str, data_dir: Path, out: Path, steps: int = STEPS, limit: int | None = None
) -> Path:
    """Start, resume or finish the pilot training of (asset, size); returns its folder."""
    cfg = config(asset, size, steps)
    root = Path(out) / f"{asset}-{size}"
    checkpoint = train._Trainer.checkpoint
    train._Trainer.checkpoint = _keep_short(checkpoint)
    try:
        run_dir, data = fit(
            cfg,
            root,
            f"{cfg.symbol}-{size}-m{SEED}",
            data_dir,
            limit,
            reload_done=not (root / VALIDATION).exists(),
        )
    finally:
        train._Trainer.checkpoint = checkpoint
    if not (root / VALIDATION).exists():
        assert data is not None
        record = validation(run_dir, data)
        record["smoke"] = steps != STEPS or limit is not None
        io.write_json(root / VALIDATION, record)
    return run_dir


VALIDATION = "validation.json"
HALF = Path("ckpt") / f"step_{SHORT:07d}.pt"  # the model at step 10,000, for STEPS


def _keep_short(checkpoint: Any) -> Any:
    """The trainer's checkpoint, which first saves the model once at step SHORT (to HALF)."""

    def wrapped(self: Any) -> None:
        if self.step == SHORT and not (self.run_dir / HALF).exists():
            train.save_model(
                self.run_dir / HALF, self.run_id, self.step, self.config, self.model, self.pca()
            )
        checkpoint(self)

    return wrapped


def _readouts(ckpt: Path, val: days.Minutes) -> tuple[np.ndarray, np.ndarray]:
    """(latent (M, dim), delta = 1 surprise (M,)) over the validation minutes, run by run."""
    model = train.load_market(ckpt).model
    parts = [
        minutes.readouts(model, val.frames[val.run == r], val.valid[val.run == r])
        for r in torch.unique(val.run).tolist()
    ]
    return torch.cat([z for z, _ in parts]).numpy(), torch.cat([s for _, s in parts])[:, 0].numpy()


def probe_errors(latent: np.ndarray, val: days.Minutes) -> list[float]:
    """VALIDATION ERROR per scored day (the second half of the validation days): the ridge probe
    from minute t's latent to minute t + 1's feature means, fitted on the first half."""
    target = val.frames.mean(1).numpy()[1:]
    x = latent[:-1]
    day = np.arange(len(val)) // days.MINUTES
    same = (val.run[:-1] == val.run[1:]).numpy() & (day[:-1] == day[1:])
    pair = (val.valid[:-1] & val.valid[1:]).numpy() & same
    day = day[:-1]
    total = len(val) // days.MINUTES
    first = total // 2
    fit = pair & (day < first)
    var = target[fit].var(0)
    keep = var > 1e-12
    model = probes.ridge(x[fit], target[fit][:, keep])
    err = ((model.predict(x) - target[:, keep]) ** 2 / var[keep]).mean(1)
    scored = [pair & (day == d) for d in range(first, total)]
    return [float(err[m].mean()) if m.any() else math.nan for m in scored]


def surprise_days(surprise: np.ndarray) -> list[float]:
    """Per validation day, the mean delta = 1 surprise over its scored minutes (descriptive)."""
    rows = surprise.reshape(-1, days.MINUTES)
    return [float(np.nanmean(r)) if np.isfinite(r).any() else math.nan for r in rows]


def validation(run_dir: Path, data: days.Dataset) -> dict[str, Any]:
    done = io.read_json(run_dir / "complete.json")
    ledger = run_dir.parents[1] / "jepa" / train.LEDGER
    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    mine = [x for x in lines if x["run_id"] == run_dir.name]
    hours = sum(x["hours"] for x in mine)
    final_z, final_s = _readouts(run_dir / train.FINAL, data.val)
    half = run_dir / HALF
    val_days = [d["day"] for d in data.sources["val"]]
    return {
        "run_id": run_dir.name,
        "days": val_days[len(val_days) // 2 :],  # the scored days
        "final": probe_errors(final_z, data.val),
        "half": probe_errors(_readouts(half, data.val)[0], data.val) if half.exists() else None,
        "twin": probe_errors(_readouts(run_dir / train.TWIN, data.val)[0], data.val),
        "surprise": surprise_days(final_s),  # every validation day, descriptive
        "steps": done["last_key"],
        "collapse": done["collapse"],
        # wall clock between checkpoints, a laptop sleep included
        "seconds_per_step": 3600 * hours / max(1, done["last_key"]),
    }


# ---------------------------------------------------------------- the rules


def usable(record: Mapping[str, Any]) -> bool:
    return bool(np.nanmean(record["final"]) < np.nanmean(record["twin"]))


def matches(s: Sequence[float], m: Sequence[float], seed: int = 0) -> tuple[bool, float]:
    """Whether S reaches M's validation error (SIZE): (verdict, upper end of the 95% interval of
    the mean paired difference S - M over days scored on both)."""
    a, b = np.asarray(s, float), np.asarray(m, float)
    ok = np.isfinite(a) & np.isfinite(b)
    diff = (a - b)[ok]
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(len(diff), size=(DRAWS, len(diff)))].mean(1)
    upper = float(np.quantile(means, 0.975))
    return upper <= TOLERANCE * float(b[ok].mean()), upper


def choose_size(records: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """records[asset][size] -> validation.json; the SIZE rule's choice: S, M or owner."""
    detail: dict[str, Any] = {}
    choice = "S"
    for asset, by in sorted(records.items()):
        if not usable(by["M"]):
            return {"size": "owner", "why": f"{asset} M is not usable", **detail}
        ok, upper = matches(by["S"]["final"], by["M"]["final"])
        detail[asset] = {"usable_S": usable(by["S"]), "S_matches_M": ok, "upper": upper}
        if not (ok and usable(by["S"])):
            choice = "M"
    return {"size": choice, **detail}


def choose_steps(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """records[asset] -> validation.json of the chosen size; the STEPS rule."""
    detail = {
        a: (float(np.nanmean(r["half"])), float(np.nanmean(r["final"])))
        for a, r in sorted(records.items())
    }
    flat = all(short - long <= TOLERANCE * long for short, long in detail.values())
    return {"steps": SHORT if flat else LONG, "e10k_e20k": detail}


def floor_value(h: float) -> float:
    """h / FLOOR_DIVISOR rounded down to one significant figure."""
    x = h / FLOOR_DIVISOR
    unit = 10 ** math.floor(math.log10(x))
    return math.floor(x / unit + 1e-9) * unit


def floors(tables: Sequence[Mapping[str, np.ndarray]]) -> dict[str, float]:
    """The FLOORS rule over the monitor tables of the usable runs of the chosen size."""
    return {
        name: floor_value(min(float(t[name][t["step"] >= WINDOW].min()) for t in tables))
        for name in monitor.FLOORED
    }


def _monitor(run_dir: Path) -> dict[str, np.ndarray]:
    table = pa.concat_tables(pq.read_table(p) for p in io.chunks(run_dir, "monitor"))
    return {c: table[c].to_numpy() for c in ("step", *monitor.FLOORED)}


def decide(out: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for asset in ASSETS:
        for size in SIZES:
            path = Path(out) / f"{asset}-{size}" / VALIDATION
            record = io.read_json(path)
            if record.get("smoke") or record.get("half") is None:
                raise ValueError(f"{path} is a smoke run, or holds no step-{SHORT} model")
            records.setdefault(asset, {})[size] = record
    result = choose_size(records)
    size = result["size"]
    if size == "owner":
        return result
    dirs = {a: _runs(Path(out) / f"{a}-{size}")[0] for a in ASSETS}
    tables = {a: _monitor(d) for a, d in dirs.items()}
    result.update(choose_steps({a: records[a][size] for a in ASSETS}))
    kept = [tables[a] for a in ASSETS if usable(records[a][size])]
    result["floors"] = floors(kept)
    result["seconds_per_step"] = {
        f"{a}-{s}": records[a][s]["seconds_per_step"] for a in ASSETS for s in SIZES
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = parser.add_subparsers(required=True, dest="command")
    p = sub.add_parser("train", help="start, resume or finish one pilot training")
    p.add_argument("--asset", choices=list(ASSETS), required=True)
    p.add_argument("--size", choices=SIZES, required=True)
    p.add_argument("--data", type=Path, required=True, help="the training range folder")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=STEPS, help="smoke runs only")
    p.add_argument("--limit", type=int, help="smoke runs only: K training and K validation days")
    p = sub.add_parser("decide", help="apply the rules to the four finished pilots")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            print(run(args.asset, args.size, args.data, args.out, args.steps, args.limit))
        else:
            print(json.dumps(decide(args.out), indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
