# notebooks/kaggle_panel_sequence.py: its twelve pieces are every cell of the shipped sequence plan
# once, its kernel folders are private CPU kernels that carry their commit, piece and data digests,
# and a kernel refuses a changed dataset and a run that is not exactly its plan's cells.

import ast
import importlib.util
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

import pytest
from tipping_point.panel import sequence

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


def test_the_kernel_refuses_a_changed_dataset(tmp_path: Path) -> None:
    k = runner()
    flat = tmp_path / "dataset"
    flat.mkdir()
    (flat / "inputs.parquet").write_bytes(b"inputs")
    (flat / k.GRID_FILE).write_bytes(b"grid")
    digests = {n: k.sha256(flat / n) for n in ("inputs.parquet", k.GRID_FILE)}
    manifest = {"run_id": k.INPUTS_RUN, "params": {"labels_run": k.LABELS_RUN}}
    complete = {"files": {"inputs.parquet": digests["inputs.parquet"]}}
    (flat / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (flat / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    data = {"inputs_run": k.INPUTS_RUN, "labels_run": k.LABELS_RUN, "sha256": digests}
    k.check_dataset(flat, data)
    with pytest.raises(RuntimeError, match="holds runs"):
        k.check_dataset(flat, data | {"labels_run": "20260101T000000Z-other"})
    with pytest.raises(RuntimeError, match="records another inputs table"):
        k.check_dataset(flat, data | {"sha256": digests | {"inputs.parquet": "0" * 64}})
    (flat / k.GRID_FILE).write_bytes(b"another grid")
    with pytest.raises(RuntimeError, match=f"{k.GRID_FILE} has changed"):
        k.check_dataset(flat, data)


def test_a_piece_is_complete_only_with_exactly_its_cells_and_nothing_skipped(
    tmp_path: Path,
) -> None:
    k = runner()
    cells = {"encoders/vol-forward-fold2-seed1-jepa.pt", "encoders/vol-forward-fold2-seed1-twin.pt"}

    def piece(name: str, encoders: set[str], skipped: list[str]) -> Path:
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "table.parquet").write_bytes(b"rows")
        runtime = {"cpu": "a cpu", "capability": k.CAPABILITY, "pins": k.PINS}
        manifest = {
            "domain": "panel-sequence",
            "producer": {"git_sha": COMMIT, "dirty": False},
            "params": {"runtime": runtime, "encoders": {e: "c" * 64 for e in encoders}},
        }
        files = {"table.parquet": k.sha256(run_dir / "table.parquet")}
        complete = {"clean": True, "files": files, "checks": {"skipped": skipped}}
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
        return run_dir

    assert k.verify(piece("whole", cells, []), COMMIT, cells)["run"] == "whole"
    with pytest.raises(RuntimeError, match="not exactly its plan's cells"):
        k.verify(piece("short", set(sorted(cells)[:1]), []), COMMIT, cells)
    with pytest.raises(RuntimeError, match="skipped"):
        k.verify(piece("skipping", cells, ["vol forward fold 2 24 h: no probe row"]), COMMIT, cells)


def test_the_pinned_packages_are_uv_lock_s() -> None:
    k = runner()
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {f"{p['name']}=={p['version']}" for p in lock["package"]}
    assert {k.TORCH_CPU, *k.PINNED} <= locked
