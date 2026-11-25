# notebooks/kaggle_panel_sequence.py: its twelve pieces are every cell of the shipped sequence plan
# once, its kernel folders are private CPU kernels that carry their commit, piece and data digests,
# and a kernel refuses a changed or held-back dataset and a run that is not exactly its cells.

import ast
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tipping_point.panel import labels, sequence, span

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "notebooks" / "kaggle_panel_sequence.py"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def runner():
    spec = importlib.util.spec_from_file_location("kaggle_panel_sequence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_store(root: Path, k, labels_run: str | None = None) -> Path:
    run_dir = root / "panel-inputs" / k.INPUTS_RUN
    run_dir.mkdir(parents=True)
    grid = {"file": f"grid/{k.GRID_FILE}", "sha256": "b" * 64}
    params = {"labels_run": labels_run or k.LABELS_RUN, "labels_grid": grid}
    manifest = {"run_id": k.INPUTS_RUN, "params": params}
    complete = {"clean": True, "files": {"inputs.parquet": "a" * 64}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    return root


def test_the_pieces_are_every_cell_of_the_shipped_plan_once(tmp_path: Path) -> None:
    k = runner()
    shipped = sequence.load()
    text = sequence.PLAN.read_text(encoding="utf-8")
    shutil.copyfile(shipped.ladder_path, tmp_path / "ladder.toml")
    seen: list[str] = []
    for piece in k.pieces():
        path = tmp_path / f"{k.slug(piece)}.toml"
        path.write_text(k.cut(text, piece), encoding="utf-8")
        plan = sequence.load(path)
        assert sequence._bare(plan) == sequence._bare(shipped)
        seen += sorted(sequence.cells(plan))
    assert len(k.pieces()) == 12 and len({k.slug(p) for p in k.pieces()}) == 12
    assert len(seen) == len(set(seen)) and set(seen) == sequence.cells(shipped)


def test_the_kernel_folders_carry_their_commit_and_piece(tmp_path: Path) -> None:
    k = runner()
    out = tmp_path / "kernels"
    store = fake_store(tmp_path / "store", k)
    k.write(COMMIT, out, store)
    data = {
        "inputs_run": k.INPUTS_RUN,
        "labels_run": k.LABELS_RUN,
        "sha256": {"inputs.parquet": "a" * 64, k.GRID_FILE: "b" * 64},
    }
    folders = sorted(p.name for p in out.iterdir())
    assert folders == sorted(["tp-panel-seq-dry", *(k.slug(p) for p in k.pieces())])
    for name in folders:
        meta = json.loads((out / name / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["id"] == f"{k.OWNER}/{name}" and meta["code_file"] == f"{name}.py"
        assert meta["is_private"] is True and meta["enable_gpu"] is False
        assert meta["enable_tpu"] is False and meta["enable_internet"] is True
        dry = name == "tp-panel-seq-dry"
        assert meta["dataset_sources"] == ([] if dry else [f"{k.OWNER}/{k.DATASET}"])
        script = (out / name / f"{name}.py").read_bytes().decode("utf-8")
        assert "\r" not in script
        (literal,) = re.findall(r"^JOB_JSON: str \| None = (.+)$", script, flags=re.MULTILINE)
        job = json.loads(ast.literal_eval(literal))
        assert job["kernel"] == name and job["commit"] == COMMIT
        assert job["mode"] == ("dry" if dry else "piece")
        if not dry:
            assert k.slug(job["piece"]) == name and job["data"] == data
        else:
            assert "data" not in job
    env = k.child_env(None)
    assert set(k.PINS) == set(sequence.PINS) and all(env[n] == v for n, v in k.PINS.items())
    with pytest.raises(SystemExit, match="not empty"):
        k.write(COMMIT, out, store)
    with pytest.raises(SystemExit, match="40 hex"):
        k.write("main", tmp_path / "other", store)
    with pytest.raises(SystemExit, match="no inputs run"):
        k.write(COMMIT, tmp_path / "other", tmp_path / "nowhere")
    other = fake_store(tmp_path / "other-store", k, labels_run="20260101T000000Z-other")
    with pytest.raises(SystemExit, match="is not inputs run"):
        k.write(COMMIT, tmp_path / "other", other)


def dataset(root: Path, k, end: str = "2023-12-31", last: str = "2023-12-31") -> tuple[Path, dict]:
    root.mkdir()
    for name, value in (("inputs.parquet", 1.0), (k.GRID_FILE, 2.0)):
        stamps = [labels.ns(span.DEV_START), labels.ns(labels.date.fromisoformat(last))]
        pq.write_table(pa.table({"ts_ns": stamps, "x": [value, value]}), root / name)
    digests = {n: k.sha256(root / n) for n in ("inputs.parquet", k.GRID_FILE)}
    params = {"labels_run": k.LABELS_RUN, "start": "2021-10-01", "dev_start": "2022-01-01"}
    manifest = {"run_id": k.INPUTS_RUN, "params": params | {"end": end}}
    complete = {"files": {"inputs.parquet": digests["inputs.parquet"]}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    return root, {"inputs_run": k.INPUTS_RUN, "labels_run": k.LABELS_RUN, "sha256": digests}


def test_the_kernel_refuses_a_changed_dataset(tmp_path: Path) -> None:
    k = runner()
    flat, data = dataset(tmp_path / "dataset", k)
    k.check_dataset(flat, data, k.guard_dataset)
    with pytest.raises(RuntimeError, match="holds runs"):
        k.check_dataset(flat, data | {"labels_run": "20260101T000000Z-other"}, k.guard_dataset)
    with pytest.raises(RuntimeError, match="records another inputs table"):
        changed = data | {"sha256": data["sha256"] | {"inputs.parquet": "0" * 64}}
        k.check_dataset(flat, changed, k.guard_dataset)
    pq.write_table(pa.table({"ts_ns": [labels.ns(span.DEV_START)]}), flat / k.GRID_FILE)
    with pytest.raises(RuntimeError, match=f"{k.GRID_FILE} has changed"):
        k.check_dataset(flat, data, k.guard_dataset)


def test_the_kernel_refuses_a_held_back_dataset_before_hashing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    k = runner()
    hashed: list[Path] = []
    sha256 = k.sha256
    monkeypatch.setattr(k, "sha256", lambda path: hashed.append(path) or sha256(path))
    for name, reach in (("span", {"end": "2024-01-31"}), ("bounds", {"last": "2024-01-02"})):
        flat, data = dataset(tmp_path / name, k, **reach)
        hashed.clear()
        with pytest.raises(span.HeldBackError):
            k.check_dataset(flat, data, k.guard_dataset)
        assert hashed == [], name


def test_a_piece_is_complete_only_with_exactly_its_cells_and_nothing_skipped(
    tmp_path: Path,
) -> None:
    k = runner()
    cells = {"encoders/vol-forward-fold2-seed1-jepa.pt", "encoders/vol-forward-fold2-seed1-twin.pt"}

    def piece(
        name: str, encoders: set[str], skipped: list[str], absent: str = "", null: bool = False
    ) -> Path:
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "table.parquet").write_bytes(b"rows")
        files = {"table.parquet": k.sha256(run_dir / "table.parquet")}
        for e in sorted(encoders - {absent}):
            (run_dir / e).parent.mkdir(parents=True, exist_ok=True)
            (run_dir / e).write_bytes(e.encode())
            files[e] = k.sha256(run_dir / e)
        runtime = {"cpu": "a cpu", "capability": k.CAPABILITY, "pins": k.PINS}
        digests: dict[str, str | None] = {
            e: hashlib.sha256(e.encode()).hexdigest() for e in encoders
        }
        if null:
            digests[absent] = None
        manifest = {
            "domain": "panel-sequence",
            "producer": {"git_sha": COMMIT, "dirty": False},
            "params": {"runtime": runtime, "encoders": digests},
        }
        complete = {"clean": True, "files": files, "checks": {"skipped": skipped}}
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
        return run_dir

    assert k.verify(piece("whole", cells, []), COMMIT, cells)["run"] == "whole"
    with pytest.raises(RuntimeError, match="not exactly its plan's cells"):
        k.verify(piece("short", set(sorted(cells)[:1]), []), COMMIT, cells)
    with pytest.raises(RuntimeError, match="skipped"):
        k.verify(piece("skipping", cells, ["vol forward fold 2 24 h: no probe row"]), COMMIT, cells)
    with pytest.raises(RuntimeError, match="not among its verified files"):
        k.verify(piece("absent", cells, [], absent=sorted(cells)[0]), COMMIT, cells)
    with pytest.raises(RuntimeError, match="not among its verified files"):
        k.verify(piece("null", cells, [], absent=sorted(cells)[0], null=True), COMMIT, cells)


def test_the_pinned_packages_are_uv_lock_s() -> None:
    k = runner()
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {f"{p['name']}=={p['version']}" for p in lock["package"]}
    assert {k.TORCH_CPU, *k.PINNED} <= locked
