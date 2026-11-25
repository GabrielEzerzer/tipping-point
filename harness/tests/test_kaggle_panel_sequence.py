# notebooks/kaggle_panel_sequence.py: its twelve pieces are every cell of the shipped sequence plan
# once, and the kernel folders it writes are private CPU kernels that carry their commit and piece.

import ast
import importlib.util
import json
import re
import shutil
import sys
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
    k.write(COMMIT, out)
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
            assert k.slug(job["piece"]) == name
    with pytest.raises(SystemExit, match="not empty"):
        k.write(COMMIT, out)
    with pytest.raises(SystemExit, match="40 hex"):
        k.write("main", tmp_path / "other")
