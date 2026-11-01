"""The draft pre-registration the exploratory pass wrote, and the plan that trains under it: its
placeholders are the ones it says, and its numbers are the plan's."""

import json
import tomllib
from pathlib import Path

from tipping_point.ews import prereg

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "paper" / "prereg" / "ising-latent-v1.toml"
PLAN = ROOT / "notebooks" / "plans" / "primary-28.json"

CHECKPOINT_KEYS = {"training", "trained_sha256", "twin_sha256"}


def test_the_draft_passes_as_a_draft_and_names_its_trials() -> None:
    doc = prereg.load(DRAFT).doc
    assert prereg.problems(doc) == []
    assert prereg.cells(doc) == 3132 == doc["trials"]["budget"]


def test_strict_refuses_only_what_cannot_exist_yet() -> None:
    """The checkpoints, because their trainings have not run, and baseline.result, because v1 has
    not been scored: no number stands in for either. Everything the pass was to set is set."""
    found = prereg.problems(prereg.load(DRAFT).doc, strict=True)
    assert found
    for problem in found:
        where = problem.split(":")[0]
        if where == "baseline.result":
            continue
        assert where.startswith("latents.checkpoints["), problem
        assert where.rsplit(".", 1)[1] in CHECKPOINT_KEYS, problem
    assert "<set before freeze>" not in DRAFT.read_text("utf-8")


def test_the_draft_holds_the_passes_values_and_the_plan() -> None:
    doc = prereg.load(DRAFT).doc
    ind, surprise = doc["indicators"], doc["indicators"]["direction_by_observable"]["surprise"]
    assert doc["hypotheses"][2]["direction"] == surprise["level"] == ind["direction"]["level"]
    assert surprise["variance"] in ("greater", "less")
    extremum = {"greater": "max", "less": "min"}[surprise["level"]]
    assert doc["fss"]["dynamic"]["statistics"][2]["extremum"] == extremum
    assert doc["latents"]["steps"] == 25_000
    tolerances = {
        r["quantity"]: r["tolerance"] for r in doc["fss"]["static"] if r["role"] == "reference"
    }
    assert set(tolerances) == {"l2_var_surprise", "one_minus_ac1_pc_1"}
    assert all(isinstance(t, float) and t > 0 for t in tolerances.values())
    v1 = tomllib.loads((ROOT / "paper" / "prereg" / "ising-ews-v1.toml").read_text("utf-8"))
    assert doc["data"]["tau_c"] == v1["data"]["tau_c"]
    assert doc["data"]["staircase"]["tau_sha256"] == v1["data"]["staircase"]["tau_sha256"]
    assert doc["data"]["seeds"] == v1["data"]["seeds"] == [6, 7, 8, 9, 10]


def test_the_disclosure_says_what_was_seen() -> None:
    d = prereg.load(DRAFT).doc["disclosure"]
    text = " ".join(d[k] for k in ("exploratory", "decisions", "confirmatory"))
    for fact in ("seed 100", "10,000 steps", "16 and 32", "seeds 1-5", "stream"):
        assert fact in text, fact


def test_the_primary_plan_is_the_drafts_training() -> None:
    """Five model seeds, one training at a time per T4, on the file's seeds and steps, naming the
    draft by its sha256. The runner itself checks the plan's form (--self-test)."""
    doc = prereg.load(DRAFT).doc
    plan = json.loads(PLAN.read_text("utf-8"))
    runs = plan["runs"]
    assert plan["inputs"]["prereg"]["sha256"] == prereg.load(DRAFT).sha256
    assert sorted(r["args"][r["args"].index("--seed") + 1] for r in runs) == [1, 2, 3, 4, 5]
    assert {r["gpu"] for r in runs} == {0, 1}
    for r in runs:
        args = r["args"]
        assert r["command"] == "jepa train"
        assert args[args.index("--steps") + 1] == doc["latents"]["steps"]
        assert {"input": "prereg"} in args and {"input": "record"} in args
        assert args[args.index("--val-seed") + 1] == doc["latents"]["validation_seed"]
        at = args.index("--train-seeds") + 1
        assert args[at : at + 5] == doc["latents"]["training_seeds"]
        assert "--warmup" not in args  # jepa train's own steps / 20
