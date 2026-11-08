"""The market JEPA's per-minute latent series (#30 M6, research pass 3.3 and 6.1). NOT
pre-registered: exploratory, nothing here is a trial.

For one symbol and span, each checkpoint's series per UTC minute m (the minute [m, m + 1 min)):
  surprise    the delta = 1 prediction error of z_m from the 95 minutes before it
  surprise5   the delta = 5 error (from the minutes up to m - 5)
  pc_1..pc_8  z_m, the target encoder's latent, on the PCA basis frozen with the checkpoint
  valid       the minute's 60 seconds hold no gap and every column above is finite
keyed by ts_ns, the minute's start. The span is read as one contiguous series: days are read in
blocks, each with the 7 days before it as the relative-depth lead (days.frames), and the last 95
minutes of the block before it as the predictor's context, which gives bitwise what one read of the
whole span gives. The first 95 minutes of the span have no context and are invalid.

Features are scaled by the read symbol's own record, fitted on its training days (the manifest of
--scaling-run, a training on that symbol), never refitted on what is read. A cross-asset read (the
BTC model on ETH) takes ETH's record: the model reads ETH as ETH's own training run would see it.
Days come from <data>/training before 2023-10-01 and <data>/exploratory from it; any day from
2024-01-01 is refused here, in days.frames and in days.ParquetSource, and <data>/confirmatory is
never opened.

Spans: exploratory 2023-10-01 .. 2023-12-31; training 2023-01-18 .. 2023-09-30 (the memorisation
report's inputs: the model reads the days it trained on).

  .venv/Scripts/python.exe harness/explore market_latents --checkpoints <a.pt> [<b.pt> ...]
      --symbol ETHUSDT --scaling-run <a training run on ETHUSDT> --data <bybit-book-v3>
      --span exploratory --out <folder> [--days N]

Writes <out>/<run id>-step<step>/<symbol>-<span>.parquet and .json (checkpoint path and sha256, the
PCA's and the scaling's provenance, every day read, invalid minutes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tipping_point.jepa import days, minutes, train

from explore import common

TRAIN_START = date(2023, 1, 18)
EXPLORATORY = date(2023, 10, 1)
SPANS = {
    "training": (TRAIN_START, EXPLORATORY),
    "exploratory": (EXPLORATORY, days.CONFIRMATORY),
}
PCS = tuple(f"pc_{k}" for k in range(1, 9))
COLUMNS = ("surprise", "surprise5", *PCS)
BLOCK = 16  # days a read holds: about 0.5 GB of frames
MINUTE_NS = 60 * 10**9


class Folders:
    """A days.DaySource over <data>/training and <data>/exploratory, by date; nothing from
    2024-01-01, so <data>/confirmatory is never opened."""

    def __init__(self, root: Path) -> None:
        self.training = days.ParquetSource(Path(root) / "training")
        self.exploratory = days.ParquetSource(Path(root) / "exploratory")

    def read(self, symbol: str, day: date) -> tuple[pa.Table, str]:
        check([day])
        return (self.exploratory if day >= EXPLORATORY else self.training).read(symbol, day)


def check(wanted: Iterable[date]) -> None:
    for day in wanted:
        if day >= days.CONFIRMATORY:
            raise days.DataError(f"{day} is in the confirmatory span: this pass never reads it")
        if day < TRAIN_START:
            raise days.DataError(f"{day} is before the book data starts ({TRAIN_START})")


def span_days(span: str, limit: int | None = None) -> list[date]:
    """The span's days in order; with `limit`, its first `limit` (a smoke run)."""
    start, end = SPANS[span]
    out = [start + timedelta(days=k) for k in range((end - start).days)]
    out = out[:limit] if limit else out
    check(out)
    return out


def blocks(wanted: Sequence[date], size: int = BLOCK) -> Iterator[tuple[list[date], list[date]]]:
    """(lead, block) per read: the block's 7 earlier days (from TRAIN_START) as its lead."""
    for k in range(0, len(wanted), size):
        block = list(wanted[k : k + size])
        first = block[0]
        lead = [
            first - timedelta(days=d)
            for d in range(days.BASELINE_DAYS, 0, -1)
            if first - timedelta(days=d) >= TRAIN_START
        ]
        yield lead, block


def carried(
    model: minutes.MarketJepa, parts: Iterable[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """minutes.readouts over consecutive parts (frames, valid) of one contiguous series, each read
    with the last context - 1 minutes of the part before it: equal to one readout of the whole."""
    keep = model.spec.context - 1
    tail: tuple[torch.Tensor, torch.Tensor] | None = None
    zs, ss = [], []
    for f, v in parts:
        skip = 0
        if tail is not None:
            skip = len(tail[0])
            f, v = torch.cat([tail[0], f]), torch.cat([tail[1], v])
        z, s = minutes.readouts(model, f, v)
        zs.append(z[skip:])
        ss.append(s[skip:])
        tail = f[-keep:], v[-keep:]
    return torch.cat(zs), torch.cat(ss)


def _run_dir(ckpt: Path) -> Path:
    for parent in Path(ckpt).resolve().parents:
        if (parent / "manifest.json").is_file():
            return parent
    raise SystemExit(f"{ckpt}: no training manifest above it")


def scaling_of(run: Path, symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(record, provenance): the scaling a training on `symbol` fitted on its training days."""
    manifest = json.loads((Path(run) / "manifest.json").read_text("utf-8"))
    record = manifest["data"]["scaling"]
    days.Scaling(**record).check(symbol)
    if record["span"] != [TRAIN_START.isoformat(), EXPLORATORY.isoformat()]:
        raise SystemExit(f"{run}: a scaling fitted for span {record['span']}, not the training")
    provenance = {
        "run_id": manifest["run_id"],
        "symbol": record["symbol"],
        "span": record["span"],
        "fitted_days": len(record["days"]),
        "first_day": record["days"][0],
        "last_day": record["days"][-1],
        "sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
        "note": "the read symbol's record from its own training run; never refitted",
    }
    return record, provenance


def _trained_on(run: Path) -> dict[str, Any]:
    data = json.loads((run / "manifest.json").read_text("utf-8"))["data"]
    return {
        "symbol": data["symbol"],
        "train": [data["train"][0]["day"], data["train"][-1]["day"]],
        "val": [data["val"][0]["day"], data["val"][-1]["day"]],
    }


def read(
    source: days.DaySource,
    symbol: str,
    wanted: Sequence[date],
    scaling: dict[str, Any],
    checkpoints: Sequence[train.Checkpoint[minutes.Config, minutes.MarketJepa]],
) -> tuple[dict[Path, dict[str, np.ndarray]], np.ndarray, list[dict[str, Any]]]:
    """Each checkpoint's columns over `wanted`, the minutes' feature validity and every day read.
    The frames are read once, block by block, and every checkpoint reads each block."""
    records: list[dict[str, Any]] = []
    feature_valid: list[torch.Tensor] = []
    parts: list[tuple[torch.Tensor, torch.Tensor]] = []
    for lead, block in blocks(wanted):
        got, rec = days.frames(source, symbol, block, scaling, lead=lead)
        seen = {r["day"] for r in records}  # a later block's lead days were read as days
        records += [r for r in rec if not (r.get("lead") and r["day"] in seen)]
        feature_valid.append(got.valid)
        parts.append((got.frames, got.valid))
    out: dict[Path, dict[str, np.ndarray]] = {}
    for ckpt in checkpoints:
        if ckpt.pca is None:
            raise SystemExit(f"{ckpt.path}: no PCA basis; fit one as training does first")
        z, s = carried(ckpt.model, parts)
        pcs = ckpt.pca.project(z).numpy()
        columns = {"surprise": s[:, 0].double().numpy(), "surprise5": s[:, 4].double().numpy()}
        columns.update({name: pcs[:, k] for k, name in enumerate(PCS)})
        out[ckpt.path] = columns
    return out, torch.cat(feature_valid).numpy(), records


def write(
    out: Path,
    ckpt: train.Checkpoint[minutes.Config, minutes.MarketJepa],
    symbol: str,
    span: str,
    wanted: Sequence[date],
    columns: dict[str, np.ndarray],
    feature_valid: np.ndarray,
    records: list[dict[str, Any]],
    scaling: dict[str, Any],
) -> Path:
    run = _run_dir(ckpt.path)
    folder = Path(out) / f"{ckpt.run_id}-step{ckpt.step}"
    folder.mkdir(parents=True, exist_ok=True)
    start = np.datetime64(wanted[0].isoformat(), "ns").astype(np.int64)
    ts = start + MINUTE_NS * np.arange(len(feature_valid), dtype=np.int64)
    valid = feature_valid & np.all([np.isfinite(columns[c]) for c in COLUMNS], axis=0)
    table = pa.table({"ts_ns": ts, "valid": valid, **{c: columns[c] for c in COLUMNS}})
    target = folder / f"{symbol}-{span}.parquet"
    pq.write_table(table, target)
    pca = ckpt.pca
    assert pca is not None
    manifest = {
        "exploratory": True,
        "pre_registered": False,
        "code_commit": common._git("rev-parse", "HEAD"),
        "code_dirty": bool(common._git("status", "--porcelain", "--untracked-files=no")),
        "checkpoint": str(ckpt.path),
        "checkpoint_sha256": ckpt.sha256,
        "run_id": ckpt.run_id,
        "step": ckpt.step,
        "untrained": ckpt.step == 0,
        "model_symbol": ckpt.config.symbol,
        "model_trained_on": _trained_on(run),
        "symbol": symbol,
        "span": span,
        "pca": {
            "carried_by_checkpoint": True,
            "frames": pca.frames,
            "total": pca.total,
            "variance": pca.variance.tolist(),
            "fitted": "by the trainer when the checkpoint was saved, on its own training minutes "
            "drawn by the run's pca generator stream",
        },
        "scaling": scaling,
        "days": records,
        "minutes": len(valid),
        "invalid_minutes": int((~valid).sum()),
        "feature_invalid_minutes": int((~feature_valid).sum()),
        "parquet": target.name,
        "parquet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    common.save(target.with_suffix(".json"), manifest)
    return target


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"), required=True)
    p.add_argument("--scaling-run", type=Path, required=True, help="a training on --symbol")
    p.add_argument("--data", type=Path, required=True, help="bybit-book-v3")
    p.add_argument("--span", choices=sorted(SPANS), default="exploratory")
    p.add_argument("--days", type=int, help="smoke runs only: the span's first N days")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    wanted = span_days(args.span, args.days)
    common.be_nice()
    torch.set_num_threads(common.THREADS)
    clock = common.Clock()
    record, provenance = scaling_of(args.scaling_run, args.symbol)
    loaded = [train.load_market(c) for c in args.checkpoints]
    for ckpt in loaded:  # a same-asset read takes the model's own record, no other
        own = json.loads((_run_dir(ckpt.path) / "manifest.json").read_text("utf-8"))
        if ckpt.config.symbol == args.symbol and own["data"]["scaling"] != record:
            raise SystemExit(f"{ckpt.path}: --scaling-run is not this model's own record")
    clock(f"{args.symbol} {args.span}: {len(wanted)} days, {len(loaded)} checkpoint(s)")
    columns, feature_valid, records = read(Folders(args.data), args.symbol, wanted, record, loaded)
    for ckpt in loaded:
        write(
            args.out,
            ckpt,
            args.symbol,
            args.span,
            wanted,
            columns[ckpt.path],
            feature_valid,
            records,
            provenance,
        )
    clock("done")
