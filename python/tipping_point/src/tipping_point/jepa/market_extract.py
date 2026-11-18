# The market JEPA's per-minute latent series (#30 M8): pinned checkpoints read over a symbol's days
# in independent units (checkpoint x symbol x block of days), then merged and verified. The
# confirmatory span opens only under a gate record made from a market-latent file frozen at its tag.

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..book.features import FORMAT
from ..ews import prereg
from ..ising import io
from . import days, minutes, train
from .readout import Pca

DESCRIPTION = """\
The market JEPA's per-minute latent series for market-latent-v1 (#30 M8, research pass 3.3, 6.1).

Per checkpoint (each pinned run's final.pt and its step-0 twin), symbol and UTC minute m:
  surprise    the delta = 1 prediction error of z_m from the 95 minutes before it
  surprise5   the delta = 5 error
  pc_1..pc_8  z_m on the PCA basis frozen with the checkpoint
  valid       the minute's 60 seconds hold no gap and every column above is finite
keyed by ts_ns, the minute's start. Every checkpoint reads both symbols (its own and the other).
Features are scaled by the read symbol's own record, the one its pinned trainings fitted (one
record per symbol, its sha256 the pins'), never refitted.

Spans: fit 2023-01-18 .. 2023-09-30 (the seasonality fit span, from the book's first day),
exploratory 2023-10-01 .. 2023-12-31, and confirmatory from 514 minutes before 2024-01-01 (the last
514 minutes of 2023-12-31, an exploratory day: 5 x 24 bars + window 60 + length 240 - 1 + context
96 - 1) to 2026-09-30. Days come from <data>/training before 2023-10-01, <data>/exploratory to
2023-12-31 and <data>/confirmatory from 2024-01-01, each checked against its folder's manifest.json
(range, feature format, and the day's sha256). Without --gate the confirmatory span is refused and
<data>/confirmatory is never opened; such output says pre_registered false.

UNITS. A span is cut into blocks of 16 days. A task is one symbol's block and a unit one
checkpoint's read of it: <out>/units/<run id>-step<step>/<symbol>-<span>/<first day>.parquet and
.json. Each task reads its block with the 7 days before the day before it as the relative-depth
lead and that day itself as the predictor's context, so on the CPU every unit is bitwise the
minutes one read of the day before and the whole span gives. The book holds no day before
2023-01-18, so the fit span's first block reads none, as training did: its first day has no depth
baseline and all its minutes are gaps, days 2 to 7 use the 1 to 6 earlier days they have, and a
minute with no valid minute in its context has no value. Tasks are independent:
--shard i/n takes every n-th task from the i-th, so n processes, on one host or several, split a
span with no overlap. A unit appears whole or not at all, and `run` skips a unit already written
and intact whose record names the task's first and last day and minutes before it, the
checkpoint's and the scaling's sha256, the gate stamp and the code commit; any other is redone. So
it resumes after a kill by the same command.

  python -m tipping_point.jepa.market_extract gate --prereg paper/prereg/market-latent-v1.toml \\
      --pins <data>/m8-primaries/pins.json --out <data>/m8-latents/gate.json
  python -m tipping_point.jepa.market_extract run --gate <gate.json> --pins <pins.json> \\
      --primaries <data>/m8-primaries --data <data>/bybit-book-v4 --span confirmatory \\
      --out <units dir> [--shard 0/4] [--runs <run id> ...] [--symbols BTCUSDT] [--threads 1]
  python -m tipping_point.jepa.market_extract merge --gate <gate.json> --pins <pins.json> \\
      --span confirmatory --units <units dir> [<units dir> ...] --out <data>/m8-latents
  python -m tipping_point.jepa.market_extract verify <data>/m8-latents

`gate` refuses unless the file is a market-latent file byte for byte as frozen at its prereg/ tag,
the code that makes the series (HELD) is its blob at the frozen commit, the code it runs (CLEAN) is
committed, the checkout is clean, and its [[latents.checkpoints]] are pins.json's runs with their
trained and twin sha256. Its record holds the file, its stamp (with the tag object's sha, so a tag
made again does not match), the code check and the pins' sha256. `run` refuses a gate record that
is not self-consistent, other pins, or code that is not the record's clean commit; in the
repository it makes the gate again and refuses a record whose frozen part differs, and only on a
host without the repository (a Kaggle kernel) do the record and the installed wheel's commit stand
alone. It loads only
checkpoints whose sha256 (and the trained one's PCA basis) are the pins'. --days N (the span's first
N days, a timing run) is refused with --gate. `merge` runs in the repository: it makes the gate
again, takes only units made by one clean commit whose HELD files are the frozen commit's, under
that gate, from checkpoints of the pins' sha256, covering each block of the span once (a unit found
twice must be byte for byte the same), and writes <out>/<run id>-step<step>/<symbol>-<span>.parquet
and .json: the checkpoint's sha256, its PCA's and the scaling's provenance, the code commit, every
day read with its sha256 and folder, and the invalid minutes. `verify` checks every file under a
folder against the sha256 its record holds; exit 1 on any problem.
"""

TRAIN_START = date(2023, 1, 18)
EXPLORATORY = date(2023, 10, 1)
END = date(2026, 10, 1)
FOLDERS = (
    ("training", TRAIN_START, EXPLORATORY),
    ("exploratory", EXPLORATORY, days.CONFIRMATORY),
    ("confirmatory", days.CONFIRMATORY, END),
)
SPANS = {
    "fit": (TRAIN_START, EXPLORATORY),
    "exploratory": (EXPLORATORY, days.CONFIRMATORY),
    "confirmatory": (days.CONFIRMATORY, END),
}
PRE_MINUTES = {"confirmatory": 5 * 24 + 60 + 240 - 1 + 96 - 1}
SYMBOLS = ("BTCUSDT", "ETHUSDT")
PCS = tuple(f"pc_{k}" for k in range(1, 9))
COLUMNS = ("surprise", "surprise5", *PCS)
MINUTE_NS = 60 * 10**9
BLOCK = 16
KINDS = (("trained", train.FINAL), ("twin", train.TWIN))
DOMAIN = "market-latent"
UNITS = "units"
GATE_FORMAT = 2
SRC = Path(__file__).resolve().parents[1]
HELD = (
    "jepa/market_extract.py",
    "jepa/days.py",
    "jepa/minutes.py",
    "jepa/monitor.py",
    "jepa/objective.py",
    "jepa/readout.py",
    "jepa/lattice.py",
    "jepa/train.py",
    "book/features.py",
)
CLEAN = ("ews/prereg.py", "ising/io.py")


class Refused(ValueError):
    pass


def project(
    ckpt: train.Checkpoint[minutes.Config, minutes.MarketJepa], z: torch.Tensor, s: torch.Tensor
) -> dict[str, np.ndarray]:
    if ckpt.pca is None:
        raise Refused(f"{ckpt.path}: no PCA basis; fit one as training does first")
    pcs = ckpt.pca.project(z).numpy()
    columns = {"surprise": s[:, 0].double().numpy(), "surprise5": s[:, 4].double().numpy()}
    columns.update({name: pcs[:, k] for k, name in enumerate(PCS)})
    return columns


def table(
    first: date, feature_valid: np.ndarray, columns: dict[str, np.ndarray], before: int = 0
) -> pa.Table:
    start = np.datetime64(first.isoformat(), "ns").astype(np.int64) - before * MINUTE_NS
    ts = start + MINUTE_NS * np.arange(len(feature_valid), dtype=np.int64)
    valid = feature_valid & np.all([np.isfinite(columns[c]) for c in COLUMNS], axis=0)
    return pa.table({"ts_ns": ts, "valid": valid, **{c: columns[c] for c in COLUMNS}})


def digest(record: Any) -> str:
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def pca_digest(pca: Pca) -> str:
    return digest({k: v.tolist() if hasattr(v, "tolist") else v for k, v in pca.record().items()})


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_paths(root: Path, paths: Sequence[str]) -> list[str]:
    base = SRC.resolve().relative_to(root.resolve()).as_posix()
    return [f"{base}/{p}" for p in paths]


def load_file(path: Path) -> prereg.Prereg:
    path = Path(path)
    raw = path.read_bytes()
    try:
        doc = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise prereg.PreregError(f"{path}: {exc}") from exc
    data = doc.get("data")
    domain = data.get("domain") if isinstance(data, dict) else None
    if domain != DOMAIN:
        raise prereg.PreregError(f"{path} is a {domain!r} file, not a {DOMAIN} one")
    return prereg.Prereg(path, doc, hashlib.sha256(raw).hexdigest())


def checkpoints_of(doc: dict[str, Any]) -> list[dict[str, str]]:
    latents = doc.get("latents")
    rows = latents.get("checkpoints") if isinstance(latents, dict) else None
    if not isinstance(rows, list) or not rows:
        raise prereg.PreregError("the file has no [[latents.checkpoints]]")
    keys = ("run_id", "trained_sha256", "twin_sha256")
    out = []
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in keys):
            raise prereg.PreregError(f"[[latents.checkpoints]] {row!r}: needs {', '.join(keys)}")
        out.append({k: row[k] for k in keys})
    return sorted(out, key=lambda r: r["run_id"])


def read_pins(path: Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def pinned(pins: dict[str, Any]) -> list[dict[str, str]]:
    keys = ("run_id", "trained_sha256", "twin_sha256")
    return sorted(({k: p[k] for k in keys} for p in pins["checkpoints"]), key=lambda r: r["run_id"])


def _match(listed: list[dict[str, str]], pins: dict[str, Any], what: str) -> None:
    want = pinned(pins)
    if listed == want:
        return
    have = {json.dumps(r, sort_keys=True) for r in listed}
    need = {json.dumps(r, sort_keys=True) for r in want}
    raise Refused(
        f"{what} and pins.json differ: only in {what}: {sorted(have - need)}; "
        f"only in pins.json: {sorted(need - have)}"
    )


def gate(path: Path, pins_path: Path) -> dict[str, Any]:
    pre = load_file(path)
    ref = prereg.frozen(pre)
    code = prereg.code_check(pre, ref, SRC, HELD, CLEAN)
    if code["changed"]:
        raise Refused(
            f"these files differ from {ref.tag}'s commit {ref.commit[:12]}: "
            + ", ".join(code["changed"])
        )
    if code["uncommitted"]:
        raise Refused("changes not committed in " + ", ".join(code["uncommitted"]))
    here = io.code_version()
    if here["git_sha"] != code["head"] or here["dirty"] is not False:
        raise Refused(f"the checkout is not clean at {code['head'][:12]}: commit or remove changes")
    pins, pins_sha = read_pins(pins_path)
    listed = checkpoints_of(pre.doc)
    _match(listed, pins, pre.name)
    root = prereg.repo_root(pre.path)
    return {
        "format": GATE_FORMAT,
        "prereg": {
            **prereg.stamp(pre, ref),
            "tag_object": prereg._git(root, "rev-parse", ref.tag),
            "path": pre.path.resolve().relative_to(root.resolve()).as_posix(),
            "text": pre.path.read_bytes().decode("utf-8"),
        },
        "code": code,
        "pins_sha256": pins_sha,
        "checkpoints": listed,
    }


def admit(record: dict[str, Any], pins: dict[str, Any], pins_sha: str) -> None:
    if not isinstance(record, dict) or record.get("format") != GATE_FORMAT:
        raise Refused(f"not a gate record of format {GATE_FORMAT}: make one with `gate`")
    stamp = record["prereg"]
    if hashlib.sha256(stamp["text"].encode("utf-8")).hexdigest() != stamp["sha256"]:
        raise Refused("the gate record's file is not the one its stamp names")
    doc = tomllib.loads(stamp["text"])
    if (doc.get("data") or {}).get("domain") != DOMAIN:
        raise Refused(f"the gate record's file is not a {DOMAIN} file")
    if checkpoints_of(doc) != record["checkpoints"]:
        raise Refused("the gate record's checkpoints are not its file's")
    if record["pins_sha256"] != pins_sha:
        raise Refused("pins.json is not the one the gate record was made with")
    _match(record["checkpoints"], pins, "the gate record")
    if record["code"]["changed"] or record["code"]["uncommitted"]:
        raise Refused("the gate record names changed or uncommitted code")
    here = io.code_version()
    if here["git_sha"] != record["code"]["head"] or here["dirty"] is not False:
        raise Refused(
            f"this code is commit {here['git_sha']} (dirty {here['dirty']}); the gate record's "
            f"is {record['code']['head']}, clean"
        )


def stamp_of(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    keys = ("id", "version", "sha256", "tag", "tag_object", "commit")
    return {
        **{k: record["prereg"][k] for k in keys},
        "head": record["code"]["head"],
        "pins_sha256": record["pins_sha256"],
    }


class Range:
    def __init__(self, root: Path, name: str, since: date, until: date) -> None:
        self.root, self.name = Path(root), name
        self.source = days.ParquetSource(self.root, since, until)
        self._manifest: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            path = self.root / "manifest.json"
            raw = path.read_bytes()
            record = json.loads(raw)
            if record.get("range") != self.name:
                found_range = record.get("range")
                raise days.DataError(f"{path} is the {found_range!r} range, not {self.name}")
            found = (record.get("assembled") or {}).get("feature_format")
            if found != FORMAT:
                raise days.DataError(f"{path} is feature format {found}; this code reads {FORMAT}")
            self._manifest = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "commit": record["assembled"].get("commit"),
                "days": record["days"],
            }
        return self._manifest

    def read(self, symbol: str, day: date) -> tuple[pa.Table, str]:
        got, sha = self.source.read(symbol, day)
        entry = self.manifest()["days"].get(f"{symbol}/{day.isoformat()}") or {}
        expected = (entry.get("reduced") or {}).get("sha256")
        if expected != sha:
            raise days.DataError(
                f"{symbol} {day}: {self.root} holds a file of sha256 {sha}; its manifest says "
                f"{expected}"
            )
        return got, sha


class Book:
    def __init__(self, ranges: dict[str, days.DaySource], until: date = days.CONFIRMATORY) -> None:
        self.ranges, self.until = ranges, until

    def folder(self, day: date) -> str:
        days.refuse(day, self.until)
        for name, since, end in FOLDERS:
            if since <= day < end:
                return name
        raise days.DataError(f"{day} is outside the book data [{TRAIN_START}, {END})")

    def read(self, symbol: str, day: date) -> tuple[pa.Table, str]:
        return self.ranges[self.folder(day)].read(symbol, day)

    def provenance(self) -> dict[str, Any]:
        return {
            name: {"sha256": r.manifest()["sha256"], "commit": r.manifest()["commit"]}
            for name, r in self.ranges.items()
            if isinstance(r, Range) and r._manifest is not None
        }


def open_book(root: Path, until: date) -> Book:
    ranges: dict[str, days.DaySource] = {
        name: Range(Path(root) / name, name, since, min(end, until))
        for name, since, end in FOLDERS
        if since < until
    }
    return Book(ranges, until)


@dataclass(frozen=True)
class Task:
    symbol: str
    span: str
    days: tuple[date, ...]
    pre: int = 0

    @property
    def first(self) -> date:
        return self.days[0]

    @property
    def eve(self) -> list[date]:
        day = self.days[0] - timedelta(days=1)
        return [day] if day >= TRAIN_START else []

    @property
    def lead(self) -> list[date]:
        before = [self.first - timedelta(days=d) for d in range(days.BASELINE_DAYS + 1, 1, -1)]
        return [d for d in before if d >= TRAIN_START]

    @property
    def keep(self) -> slice:
        return slice(days.MINUTES * len(self.eve) - self.pre, None)


def span_days(span: str, limit: int | None = None) -> list[date]:
    start, end = SPANS[span]
    out = [start + timedelta(days=k) for k in range((end - start).days)]
    return out[:limit] if limit else out


def tasks(
    span: str, symbols: Sequence[str], size: int | None = None, limit: int | None = None
) -> list[Task]:
    wanted = span_days(span, limit)
    size = size or BLOCK
    pre = PRE_MINUTES.get(span, 0)
    return [
        Task(symbol, span, tuple(wanted[k : k + size]), pre if k == 0 else 0)
        for symbol in symbols
        for k in range(0, len(wanted), size)
    ]


def shard(found: Sequence[Task], spec: str) -> list[Task]:
    index, _, count = spec.partition("/")
    if not (index.isdigit() and count.isdigit() and int(index) < int(count)):
        raise Refused(f"--shard {spec!r}: i/n with 0 <= i < n")
    return list(found[int(index) :: int(count)])


def read_task(
    book: Book, task: Task, scaling: dict[str, Any]
) -> tuple[days.Minutes, list[dict[str, Any]]]:
    if task.pre and not task.eve:
        raise Refused(f"{task.first}: {task.pre} minutes before it, but no day before it to read")
    got, records = days.frames(
        book, task.symbol, [*task.eve, *task.days], scaling, lead=task.lead, until=book.until
    )
    for r in records:
        r["folder"] = book.folder(date.fromisoformat(r["day"]))
        if [r["day"]] == [d.isoformat() for d in task.eve]:
            r["context"] = True
            if task.pre:
                r["kept_minutes"] = task.pre
    return got, records


@dataclass
class Model:
    pin: dict[str, Any]
    kind: str
    file: str
    ckpt: train.Checkpoint[minutes.Config, minutes.MarketJepa]

    @property
    def name(self) -> str:
        return f"{self.ckpt.run_id}-step{self.ckpt.step}"


def run_dir(primaries: Path, pin: dict[str, Any]) -> Path:
    return Path(primaries) / f"{pin['asset']}-s{pin['seed']}" / "jepa" / pin["run_id"]


def select(pins: dict[str, Any], runs: Sequence[str] | None) -> list[dict[str, Any]]:
    chosen = [p for p in pins["checkpoints"] if not runs or p["run_id"] in runs]
    missing = sorted(set(runs or ()) - {p["run_id"] for p in chosen})
    if missing:
        raise Refused(f"runs not in pins.json: {missing}")
    return chosen


def load_models(primaries: Path, pins: dict[str, Any], chosen: list[dict[str, Any]]) -> list[Model]:
    out = []
    for pin in chosen:
        for kind, rel in KINDS:
            ckpt = train.load_market(run_dir(primaries, pin) / rel)
            want = pin[f"{kind}_sha256"]
            if ckpt.sha256 != want:
                raise Refused(f"{ckpt.path}: sha256 {ckpt.sha256}, pinned {want}")
            step = pins["steps"] if kind == "trained" else 0
            if (ckpt.run_id, ckpt.step, ckpt.config.symbol) != (pin["run_id"], step, pin["symbol"]):
                raise Refused(
                    f"{ckpt.path}: run {ckpt.run_id} step {ckpt.step} on {ckpt.config.symbol}, "
                    f"pinned {pin['run_id']} step {step} on {pin['symbol']}"
                )
            if ckpt.pca is None:
                raise Refused(f"{ckpt.path}: no PCA basis")
            if kind == "trained" and pca_digest(ckpt.pca) != pin["pca_sha256"]:
                raise Refused(f"{ckpt.path}: its PCA basis is not the pinned one")
            out.append(Model(pin, kind, rel.as_posix(), ckpt))
    return out


def scalings(
    primaries: Path, pins: dict[str, Any], symbols: Sequence[str]
) -> dict[str, dict[str, Any]]:
    out = {}
    for symbol in symbols:
        records = {}
        for pin in (p for p in pins["checkpoints"] if p["symbol"] == symbol):
            manifest = io.read_json(run_dir(primaries, pin) / "manifest.json")
            record = manifest["data"]["scaling"]
            if digest(record) != pin["scaling_sha256"]:
                raise Refused(f"{pin['run_id']}: its scaling record is not the pinned one")
            records[digest(record)] = record
        if len(records) != 1:
            raise Refused(f"{symbol}: {len(records)} pinned scaling records; a read takes one")
        ((sha, record),) = records.items()
        days.Scaling(**record).check(symbol)
        out[symbol] = {
            "record": record,
            "sha256": sha,
            "span": record["span"],
            "fitted_days": len(record["days"]),
            "first_day": record["days"][0],
            "last_day": record["days"][-1],
        }
    return out


def unit_paths(out: Path, name: str, symbol: str, span: str, first: date) -> tuple[Path, Path]:
    folder = Path(out) / UNITS / name / f"{symbol}-{span}"
    return folder / f"{first.isoformat()}.parquet", folder / f"{first.isoformat()}.json"


def unit_key(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "first": record.get("first"),
        "last": record.get("last"),
        "pre_minutes": record.get("pre_minutes"),
        "checkpoint_sha256": (record.get("checkpoint") or {}).get("sha256"),
        "scaling_sha256": record.get("scaling_sha256"),
        "gate": record.get("gate"),
        "git_sha": (record.get("code") or {}).get("git_sha"),
    }


def unit_done(parquet: Path, record_path: Path, want: dict[str, Any]) -> bool:
    if not (parquet.is_file() and record_path.is_file()):
        return False
    record = io.read_json(record_path)
    if not isinstance(record, dict) or unit_key(record) != want:
        return False
    return record.get("parquet_sha256") == file_sha256(parquet)


def environment() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
    }


def _write_parquet(path: Path, result: pa.Table) -> None:
    io.write_file(path, lambda f: pq.write_table(result, f), replace=True)


def extract(
    book: Book,
    models: Sequence[Model],
    scaling: dict[str, dict[str, Any]],
    todo: Sequence[Task],
    out: Path,
    gate_record: dict[str, Any] | None,
    log: Any = None,
) -> int:
    written = 0
    code, env, stamp = io.code_version(), environment(), stamp_of(gate_record)
    for task in todo:
        paths = {
            m.name: unit_paths(out, m.name, task.symbol, task.span, task.first) for m in models
        }
        want = {
            m.name: {
                "first": task.first.isoformat(),
                "last": task.days[-1].isoformat(),
                "pre_minutes": task.pre,
                "checkpoint_sha256": m.ckpt.sha256,
                "scaling_sha256": scaling[task.symbol]["sha256"],
                "gate": stamp,
                "git_sha": code.get("git_sha"),
            }
            for m in models
        }
        left = [m for m in models if not unit_done(*paths[m.name], want[m.name])]
        if not left:
            continue
        clock, cpu = time.perf_counter(), time.process_time()
        got, records = read_task(book, task, scaling[task.symbol]["record"])
        read_s, read_cpu_s = time.perf_counter() - clock, time.process_time() - cpu
        keep = task.keep
        feature_valid = got.valid[keep].numpy()
        for m in left:
            clock, cpu = time.perf_counter(), time.process_time()
            z, s = minutes.readouts(m.ckpt.model, got.frames, got.valid)
            columns = project(m.ckpt, z[keep], s[keep])
            result = table(task.first, feature_valid, columns, task.pre)
            parquet, record_path = paths[m.name]
            parquet.parent.mkdir(parents=True, exist_ok=True)
            _write_parquet(parquet, result)
            valid = result["valid"].to_numpy()
            record = {
                "format": 1,
                "pre_registered": gate_record is not None,
                "gate": stamp,
                "code": code,
                "env": env,
                "checkpoint": {
                    "run_id": m.ckpt.run_id,
                    "kind": m.kind,
                    "step": m.ckpt.step,
                    "file": m.file,
                    "sha256": m.ckpt.sha256,
                    "pca_sha256": pca_digest(m.ckpt.pca) if m.ckpt.pca is not None else None,
                    "symbol": m.ckpt.config.symbol,
                    "asset": m.pin["asset"],
                    "seed": m.pin["seed"],
                    "role": m.pin["role"],
                },
                "symbol": task.symbol,
                "span": task.span,
                "first": task.first.isoformat(),
                "last": task.days[-1].isoformat(),
                "pre_minutes": task.pre,
                "scaling_sha256": scaling[task.symbol]["sha256"],
                "book": book.provenance(),
                "days": records,
                "minutes": len(valid),
                "invalid_minutes": int((~valid).sum()),
                "feature_invalid_minutes": int((~feature_valid).sum()),
                "parquet_sha256": file_sha256(parquet),
                "read_s": round(read_s, 3),
                "read_cpu_s": round(read_cpu_s, 3),
                "model_s": round(time.perf_counter() - clock, 3),
                "model_cpu_s": round(time.process_time() - cpu, 3),
            }
            io.write_json(record_path, record, replace=True)
            written += 1
        if log is not None:
            span = f"{task.first}..{task.days[-1]}"
            log(f"{task.symbol} {span}: {len(left)} unit(s), read {read_s:.1f} s")
    return written


def _units(sources: Sequence[Path], span: str) -> dict[tuple[str, str, str], list[Path]]:
    found: dict[tuple[str, str, str], list[Path]] = {}
    for source in sources:
        for path in sorted((Path(source) / UNITS).glob(f"*/*-{span}/*.json")):
            name, symbol = path.parent.parent.name, path.parent.name.removesuffix(f"-{span}")
            found.setdefault((name, symbol, path.stem), []).append(path)
    return found


def _frozen_part(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in (stamp_of(record) or {}).items() if k != "head"}


def _same_held(root: Path, frozen_commit: str, commit: str) -> bool:
    changed = prereg._git(
        root, "diff", "--name-only", frozen_commit, commit, "--", *_code_paths(root, HELD)
    )
    return changed == ""


def repository() -> Path | None:
    try:
        return prereg.repo_root(SRC)
    except (prereg.PreregError, OSError):
        return None


def regate(record: dict[str, Any], pins_path: Path, root: Path) -> dict[str, Any]:
    fresh = gate(root / record["prereg"]["path"], pins_path)
    if _frozen_part(fresh) != _frozen_part(record):
        raise Refused("the gate made now is not the gate record's")
    return fresh


def admitted(record: dict[str, Any], pins_path: Path) -> str:
    pins, pins_sha = read_pins(pins_path)
    admit(record, pins, pins_sha)
    root = repository()
    if root is None:
        return "the record and the installed code's commit (no repository on this host)"
    regate(record, pins_path, root)
    return f"a gate made again in {root}"


def merge(
    sources: Sequence[Path],
    out: Path,
    span: str,
    pins_path: Path,
    gate_record: dict[str, Any] | None,
    runs: Sequence[str] | None = None,
    symbols: Sequence[str] = SYMBOLS,
) -> list[Path]:
    pins, _ = read_pins(pins_path)
    if gate_record is None and span == "confirmatory":
        raise Refused("the confirmatory span is merged only under a gate record (--gate)")
    if gate_record is not None:
        root = prereg.repo_root(SRC)
        fresh = regate(gate_record, pins_path, root)
        if not _same_held(root, fresh["prereg"]["commit"], gate_record["code"]["head"]):
            raise Refused("the gate record's commit holds other HELD files than the frozen one")
    stamp = stamp_of(gate_record)
    found = _units(sources, span)
    chosen = select(pins, runs)
    plan = tasks(span, symbols)
    written, problems, commits = [], [], set()
    for pin in chosen:
        for kind, _ in KINDS:
            step = pins["steps"] if kind == "trained" else 0
            name = f"{pin['run_id']}-step{step}"
            for symbol in symbols:
                parts, records = [], []
                for task in (t for t in plan if t.symbol == symbol):
                    paths = found.get((name, symbol, task.first.isoformat()), [])
                    if not paths:
                        problems.append(f"{name} {symbol} {task.first}: no unit")
                        continue
                    files = [p.with_suffix(".parquet") for p in paths]
                    shas = {file_sha256(f) for f in files if f.is_file()}
                    if len(shas) != 1 or not all(f.is_file() for f in files):
                        problems.append(f"{name} {symbol} {task.first}: copies differ or lack data")
                        continue
                    path = paths[0]
                    record = io.read_json(path)
                    why = _unit_problems(record, shas.pop(), pin, kind, task, stamp)
                    if why:
                        problems.append(f"{path}: {why}")
                        continue
                    commits.add(record["code"]["git_sha"])
                    parts.append(pq.read_table(path.with_suffix(".parquet")))
                    records.append((path, record))
                if len(records) == len([t for t in plan if t.symbol == symbol]):
                    written.append((name, symbol, parts, records))
    if len(commits) > 1:
        problems.append(f"units made by several commits: {sorted(commits)}")
    if gate_record is not None and commits - {gate_record["code"]["head"]}:
        head = gate_record["code"]["head"]
        problems.append(f"units made by {sorted(commits)}, not the gate's {head}")
    if problems:
        raise Refused(f"{len(problems)} problem(s):\n  " + "\n  ".join(problems[:40]))
    return [_write_merged(out, span, gate_record, *w) for w in written]


def _unit_problems(
    record: dict[str, Any],
    parquet_sha: str,
    pin: dict[str, Any],
    kind: str,
    task: Task,
    stamp: dict[str, Any] | None,
) -> str | None:
    c = record["checkpoint"]
    expected_days = [d.isoformat() for d in task.days]
    read = [r["day"] for r in record["days"] if not r.get("lead") and not r.get("context")]
    checks = (
        (record.get("parquet_sha256") == parquet_sha, "its parquet is not its record's"),
        (c["sha256"] == pin[f"{kind}_sha256"], f"checkpoint sha256 {c['sha256']} is not pinned"),
        ((c["run_id"], c["kind"]) == (pin["run_id"], kind), "another checkpoint's unit"),
        ((record["symbol"], record["span"]) == (task.symbol, task.span), "another symbol or span"),
        (read == expected_days, f"read {read[:1]}..{read[-1:]}, not its block"),
        (record.get("pre_minutes") == task.pre, f"not {task.pre} minutes before its block"),
        (stamp is None or record["code"].get("dirty") is False, "made by uncommitted code"),
        (record["gate"] == stamp, "made under another gate"),
    )
    return next((why for ok, why in checks if not ok), None)


def _write_merged(
    out: Path,
    span: str,
    gate_record: dict[str, Any] | None,
    name: str,
    symbol: str,
    parts: list[pa.Table],
    records: list[tuple[Path, dict[str, Any]]],
) -> Path:
    merged = pa.concat_tables(parts)
    ts = merged["ts_ns"].to_numpy()
    pre = PRE_MINUTES.get(span, 0)
    start = np.datetime64(SPANS[span][0].isoformat(), "ns").astype(np.int64) - pre * MINUTE_NS
    count = len(span_days(span)) * days.MINUTES + pre
    expected = start + MINUTE_NS * np.arange(count, dtype=np.int64)
    if not np.array_equal(ts, expected):
        raise Refused(f"{name} {symbol}: the units' minutes are not the span's, in order")
    folder = Path(out) / name
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{symbol}-{span}.parquet"
    _write_parquet(target, merged)
    first = records[0][1]
    valid = merged["valid"].to_numpy()
    seen: set[str] = set()
    read = []
    for _, r in records:
        for d in r["days"]:
            if d["day"] not in seen:
                seen.add(d["day"])
                read.append(d)
    manifest = {
        "format": 1,
        "pre_registered": gate_record is not None,
        "gate": stamp_of(gate_record),
        "code": first["code"],
        "merged_by": io.code_version(),
        "checkpoint": first["checkpoint"],
        "pca": _pca_provenance(records),
        "scaling_sha256": first["scaling_sha256"],
        "symbol": symbol,
        "span": span,
        "context": "each block reads the day before it as context when the book holds that day",
        "pre_minutes": pre,
        "days": read,
        "units": [
            {
                "first": r["first"],
                "parquet_sha256": r["parquet_sha256"],
                "json_sha256": file_sha256(p),
                "env": r["env"],
                "book": r["book"],
            }
            for p, r in records
        ],
        "minutes": len(valid),
        "invalid_minutes": int((~valid).sum()),
        "feature_invalid_minutes": sum(r["feature_invalid_minutes"] for _, r in records),
        "parquet": target.name,
        "parquet_sha256": file_sha256(target),
    }
    io.write_json(target.with_suffix(".json"), manifest, replace=True)
    return target


def _pca_provenance(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    shas = {r["checkpoint"]["pca_sha256"] for _, r in records}
    return {
        "carried_by_checkpoint": True,
        "sha256": shas.pop() if len(shas) == 1 else sorted(shas),
        "fitted": "by the trainer when the checkpoint was saved, on its own training minutes",
    }


def verify(folder: Path) -> list[str]:
    problems = []
    records = sorted(Path(folder).rglob("*.json"))
    for path in records:
        record = io.read_json(path)
        if not isinstance(record, dict) or "parquet_sha256" not in record:
            continue
        parquet = path.with_suffix(".parquet")
        if not parquet.is_file():
            problems.append(f"{path}: no {parquet.name}")
        elif file_sha256(parquet) != record["parquet_sha256"]:
            problems.append(f"{parquet}: not the file its record names")
    return problems


def _gate_arg(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else io.read_json(Path(path))


def _run(args: argparse.Namespace) -> int:
    pins, _ = read_pins(args.pins)
    gate_record = _gate_arg(args.gate)
    if gate_record is None and args.span == "confirmatory":
        raise Refused("the confirmatory span opens only under a gate record (--gate)")
    if gate_record is not None:
        print(f"gate admitted by {admitted(gate_record, args.pins)}", flush=True)
        if args.days:
            raise Refused("--days is for timing runs, without --gate")
    torch.set_num_threads(args.threads)
    symbols = args.symbols or list(SYMBOLS)
    todo = tasks(args.span, symbols, limit=args.days)
    todo = shard(todo, args.shard) if args.shard else todo
    chosen = select(pins, args.runs)
    models = load_models(args.primaries, pins, chosen)
    scaling = scalings(args.primaries, pins, symbols)
    until = END if gate_record is not None else days.CONFIRMATORY
    book = open_book(args.data, until)
    clock = time.perf_counter()

    def log(message: str) -> None:
        print(f"[{time.perf_counter() - clock:8.1f} s] {message}", flush=True)

    log(f"{args.span}: {len(todo)} task(s) x {len(models)} checkpoint(s)")
    written = extract(book, models, scaling, todo, args.out, gate_record, log)
    log(f"done: {written} unit(s) written")
    return 0


def _merge(args: argparse.Namespace) -> int:
    gate_record = _gate_arg(args.gate)
    symbols = args.symbols or list(SYMBOLS)
    for path in merge(args.units, args.out, args.span, args.pins, gate_record, args.runs, symbols):
        print(path)
    return 0


def _gate(args: argparse.Namespace) -> int:
    record = gate(args.prereg, args.pins)
    io.write_json(args.out, record, replace=True)
    print(args.out)
    return 0


def _verify(args: argparse.Namespace) -> int:
    problems = [p for folder in args.folders for p in verify(folder)]
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tipping_point.jepa.market_extract", description=DESCRIPTION
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("gate", help="check the frozen file and the code; write the gate record")
    p.add_argument("--prereg", type=Path, required=True)
    p.add_argument("--pins", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(command=_gate)
    p = sub.add_parser("run", help="extract the units of a span, or of a shard of it")
    p.add_argument("--gate", type=Path, help="the gate record; without it, exploratory only")
    p.add_argument("--pins", type=Path, required=True)
    p.add_argument("--primaries", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="the book folder (bybit-book-v4)")
    p.add_argument("--span", choices=sorted(SPANS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", help="i/n: every n-th task from the i-th")
    p.add_argument("--runs", nargs="+", help="run ids (default: every pinned run)")
    p.add_argument("--symbols", nargs="+", choices=SYMBOLS)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--days", type=int, help="timing runs only: the span's first N days")
    p.set_defaults(command=_run)
    p = sub.add_parser("merge", help="check the units and write each span's series")
    p.add_argument("--gate", type=Path)
    p.add_argument("--pins", type=Path, required=True)
    p.add_argument("--span", choices=sorted(SPANS), required=True)
    p.add_argument("--units", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", nargs="+")
    p.add_argument("--symbols", nargs="+", choices=SYMBOLS)
    p.set_defaults(command=_merge)
    p = sub.add_parser("verify", help="check every file against its record's sha256")
    p.add_argument("folders", type=Path, nargs="+")
    p.set_defaults(command=_verify)
    args = parser.parse_args(argv)
    try:
        return int(args.command(args))
    except (Refused, prereg.PreregError, days.DataError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
