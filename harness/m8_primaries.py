"""The market JEPA's primaries (#30 M8): their trainings and their pins, before any extraction.

RUNS. Per asset (BTCUSDT, ETHUSDT), 5 primaries and 2 spares (research pass, call 7), each a size S
(d = 64) training of 10,000 steps (decided 2026-11-09 from the M4 rules), with its untrained twin:
the step-0 checkpoint every training writes. Every other setting is the pilot's (m4_pilot.config):
the training span [2023-01-18, 2023-10-01) and its split, training 2023-01-18 .. 2023-08-31 and
validation 2023-09-01 .. 2023-09-30, warm-up steps / 20, a checkpoint every 500 steps, the CPU in
float32, the collapse monitor recorded and never stopping a run.

SEEDS, fixed here before any primary trains: primaries 101, 102, 103, 104, 105 and spares 106, 107,
the same numbers on both assets. `train` refuses any other seed (100 was the pilot's).

  uv run --extra cpu python harness/m8_primaries.py train --asset btc --seed 101 \\
      --data <data>/bybit-book-v3/training --out <data>/m8-primaries   (again to resume)
  uv run --extra cpu python harness/m8_primaries.py pins --out <data>/m8-primaries

`train` runs one training per process (a Kaggle CPU kernel or the laptop) in
<out>/<asset>-s<seed>/jepa/<run_id>/ (the trainer's layout), beside a copy of the training range
folder's manifest.json (data-manifest.json), and resumes after a kill by the same command. The range
folder must be a `training` one of this code's feature format (book.features.FORMAT) naming no day
on or after 2024-01-01, and the one the copy holds; a training whose copy is gone is refused until
its own is put back. --steps is for smoke runs; `pins` refuses them.

`pins` writes <out>/pins.json, what `market-latent-v1`'s training section and
`[[latents.checkpoints]]` take (research pass 7.3): per run the asset, seed, role, run id, the data
manifest's commit and feature format, and the sha256 of the trained model (final.pt, step 10,000),
of its twin, of its PCA basis and of the scaling record. It refuses a run that is not a primary's
configuration, sits outside <out>/<asset>-s<seed>/, is incomplete (fewer than 10,000 steps), fails
train.verify, whose data manifest or scaling record is of another feature format than this code's,
whose data manifest names a day on or after 2024-01-01, or one of whose days does not have, in its
data manifest, the sha256 it read (both are over the bytes of features/<SYMBOL>/<day>.parquet); and
an asset and seed trained twice. It lists every refusal and then writes nothing. Extraction is not
here: it reads the confirmatory span, after `market-latent-v1` is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import m4_pilot
from tipping_point.book.features import FORMAT
from tipping_point.ising import io
from tipping_point.jepa import days, train

PRIMARIES = (101, 102, 103, 104, 105)
SPARES = (106, 107)
SIZE = "S"
STEPS = 10_000
DATA_MANIFEST = "data-manifest.json"
PINS = "pins.json"


def role(seed: int) -> str:
    if seed in PRIMARIES:
        return "primary"
    if seed in SPARES:
        return "spare"
    raise ValueError(f"seed {seed} is neither a primary {PRIMARIES} nor a spare {SPARES}")


def _late_days(manifest: dict[str, Any]) -> list[str]:
    """The days a range manifest names on or after the confirmatory start ("SYMBOL/YYYY-MM-DD")."""
    return sorted(
        k for k in manifest["days"] if date.fromisoformat(k.split("/")[1]) >= days.CONFIRMATORY
    )


def run(asset: str, seed: int, data_dir: Path, out: Path, steps: int = STEPS) -> Path:
    """Start, resume or find finished the training of (asset, seed); returns its folder."""
    tag = role(seed)
    source = Path(data_dir) / "manifest.json"
    manifest = io.read_json(source)
    if manifest.get("range") != "training":
        raise ValueError(f"{source} is the {manifest.get('range')!r} range, not 'training'")
    late = _late_days(manifest)
    if late:
        raise ValueError(f"{source} names days from {days.CONFIRMATORY} on: {late[:3]}")
    found = manifest.get("assembled", {}).get("feature_format")
    if found != FORMAT:
        raise ValueError(f"{source} is feature format {found}; this code reads {FORMAT}")
    root = Path(out) / f"{asset}-s{seed}"
    copy = root / DATA_MANIFEST
    if copy.exists():
        if copy.read_bytes() != source.read_bytes():
            raise RuntimeError(f"{root} was trained from another range manifest than {source}")
    elif any((root / "jepa").glob("*/manifest.json")):
        # a finished training reads no days, so nothing would tie a new copy to the ones it read
        raise RuntimeError(f"{root} holds a training but no {DATA_MANIFEST}: put its own back")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, copy)
    cfg = m4_pilot.config(asset, SIZE, steps, seed)
    print(f"{asset} seed {seed} ({tag}), {steps} steps", file=sys.stderr)
    return m4_pilot.fit(cfg, root, f"{cfg.symbol}-{SIZE}-s{seed}", data_dir)[0]


def _digest(record: Any) -> str:
    """sha256 of a record's canonical JSON (sorted keys, no spaces)."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _pca_digest(path: Path) -> str:
    pca = train.load_market(path).pca
    assert pca is not None  # every saved market model carries its basis
    return _digest({k: v.tolist() if hasattr(v, "tolist") else v for k, v in pca.record().items()})


def pin(run_dir: Path) -> dict[str, Any]:
    """The pin of one training, or ValueError naming why it cannot be pinned."""
    manifest = io.read_json(run_dir / "manifest.json")
    cfg = manifest["params"]["config"]
    asset = next((a for a, s in m4_pilot.ASSETS.items() if s == cfg.get("symbol")), None)
    if asset is None:
        raise ValueError(f"symbol {cfg.get('symbol')} is no asset")
    seed = int(cfg["seed"])
    kind = role(seed)
    if run_dir.parents[1].name != f"{asset}-s{seed}":
        raise ValueError(f"{asset} seed {seed} is not in its folder {asset}-s{seed}")
    done = run_dir / "complete.json"
    if not done.exists():
        raise ValueError("incomplete, still training or killed (no complete.json)")
    steps = io.read_json(done)["last_key"]
    if steps < STEPS:
        raise ValueError(f"incomplete, {steps} of {STEPS} steps")
    expected = json.loads(json.dumps(m4_pilot.config(asset, SIZE, STEPS, seed).to_json()))
    if cfg != expected:
        raise ValueError(f"not a primary's config (size {SIZE}, {STEPS} steps)")
    problems = train.verify(run_dir)
    if problems:
        raise ValueError("; ".join(problems))
    data = io.read_json(run_dir.parents[1] / DATA_MANIFEST)
    late = _late_days(data)
    if late:
        raise ValueError(f"its data manifest names {late[:3]}, on or after 2024")
    formats = (data["assembled"]["feature_format"], manifest["data"]["scaling"]["feature_format"])
    if formats != (FORMAT, FORMAT):
        raise ValueError(f"feature formats {formats} (data manifest, scaling); code {FORMAT}")
    # each day's sha256 is over the bytes of features/<SYMBOL>/<day>.parquet in both: as the
    # training read it (days.ParquetSource) and as it was reduced (book.features.reduce_run)
    files = {k: (e.get("reduced") or {}).get("sha256") for k, e in data["days"].items()}
    read = [*manifest["data"]["train"], *manifest["data"]["val"]]
    other = [d["day"] for d in read if files.get(f"{cfg['symbol']}/{d['day']}") != d["sha256"]]
    if other:
        raise ValueError(f"{len(other)} day(s) read are not its data manifest's files: {other[:3]}")
    return {
        "asset": asset,
        "symbol": cfg["symbol"],
        "seed": seed,
        "role": kind,
        "run_id": manifest["run_id"],
        "data_commit": data["assembled"]["commit"],
        "feature_format": data["assembled"]["feature_format"],
        "trained_sha256": train.sha256((run_dir / train.FINAL).read_bytes()),
        "twin_sha256": train.sha256((run_dir / train.TWIN).read_bytes()),
        "pca_sha256": _pca_digest(run_dir / train.FINAL),
        "scaling_sha256": _digest(manifest["data"]["scaling"]),
    }


def pins(out: Path) -> Path:
    """Write <out>/pins.json over every training under `out`; refuses, listing every training
    that cannot be pinned and every asset and seed trained twice, if there is any."""
    runs = sorted(p.parent for p in Path(out).glob("*/jepa/*/manifest.json"))
    if not runs:
        raise FileNotFoundError(f"no trainings under {out}")
    pinned, problems = [], []
    for run_dir in runs:
        try:
            pinned.append(pin(run_dir))
        except (ValueError, KeyError, OSError) as exc:  # a foreign or half-copied folder too
            problems.append(f"{run_dir}: {type(exc).__name__}: {exc}")
    count = Counter((p["asset"], p["seed"]) for p in pinned)
    problems += [
        f"{a} seed {s}: {n} trainings in {a}-s{s}; keep one" for (a, s), n in count.items() if n > 1
    ]
    if problems:
        raise ValueError("refused:\n  " + "\n  ".join(problems))
    pinned.sort(key=lambda p: (p["asset"], p["seed"]))
    path = Path(out) / PINS
    io.write_json(path, {"format": 1, "size": SIZE, "steps": STEPS, "checkpoints": pinned})
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = parser.add_subparsers(required=True, dest="command")
    p = sub.add_parser("train", help="start, resume or finish one primary or spare training")
    p.add_argument("--asset", choices=list(m4_pilot.ASSETS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data", type=Path, required=True, help="the training range folder")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=STEPS, help="smoke runs only")
    p = sub.add_parser("pins", help="pin every finished training under --out")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            print(run(args.asset, args.seed, args.data, args.out, args.steps))
        else:
            print(pins(args.out))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
