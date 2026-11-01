"""The draft pre-registration the exploratory pass wrote, and the plan that trains under it: it
takes the schema slice 5 holds it to, its placeholders are the ones it says, its numbers are the
plan's, and the plan's own `jepa train --prereg` line accepts it as a draft."""

import copy
import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jepa_ising
import pytest
from tipping_point.ews import prereg
from tipping_point.ising import run
from tipping_point.jepa import __main__ as cli
from tipping_point.jepa import train

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "paper" / "prereg" / "ising-latent-v1.toml"
FIXTURE = ROOT / "python" / "tipping_point" / "tests" / "fixtures" / "ising_latent_draft.toml"
PLAN = ROOT / "notebooks" / "plans" / "primary-28.json"
CHECKPOINT_KEYS = {"training", "trained_sha256", "twin_sha256"}
OWNERS = {"baseline.result", "scoring.margins.tolerance"}  # no value exists, or it is the owner's


def test_the_draft_passes_as_a_draft_and_names_its_trials() -> None:
    doc = prereg.load(DRAFT).doc
    assert prereg.problems(doc) == []
    assert prereg.cells(doc) == 3132 == doc["trials"]["budget"]


def test_strict_refuses_only_what_cannot_exist_yet() -> None:
    """The checkpoints, because their trainings have not run; baseline.result, because v1 has not
    been scored; scoring.margins.tolerance, the owner's call. No number stands in for any of them,
    and everything the pass was to set is set."""
    found = prereg.problems(prereg.load(DRAFT).doc, strict=True)
    named = {problem.split(":")[0] for problem in found}
    assert named >= OWNERS
    for where in named - OWNERS:
        assert where.startswith("latents.checkpoints["), where
        assert where.rsplit(".", 1)[1] in CHECKPOINT_KEYS, where
    assert len(named - OWNERS) == 3 * 11
    assert DRAFT.read_text("utf-8").count("<set before freeze>") == 1


def _thin(doc: dict[str, Any]) -> None:
    doc["indicators"]["thin"] = 0.1


def _old_rule(doc: dict[str, Any]) -> None:
    doc["fss"]["dynamic"]["edge_rule"] = (
        "inconclusive when more than half of an L's replicas take their extremum on an edge of the "
        "band"
    )


@pytest.mark.parametrize(
    ("undo", "complaint"),
    [
        (_thin, "indicators.thin"),
        (lambda d: d["scoring"]["margins"].pop("sizes"), "scoring.margins.sizes"),
        (lambda d: d["scoring"]["margins"].pop("tolerance"), "scoring.margins.tolerance"),
        (lambda d: d["fss"]["dynamic"].pop("edge_margin"), "fss.dynamic.edge_margin"),
        (_old_rule, "fss.dynamic.edge_rule"),
    ],
)
def test_undoing_any_edit_slice_5_requires_fails_the_load(
    undo: Callable[[dict[str, Any]], Any], complaint: str
) -> None:
    doc = copy.deepcopy(prereg.load(DRAFT).doc)
    undo(doc)
    assert any(complaint in problem for problem in prereg.problems(doc)), prereg.problems(doc)


@pytest.mark.parametrize(
    ("section", "key"),
    [("latents", "extraction"), ("probes", "fit"), ("probes", "test"), ("probes", "selectivity")],
)
def test_the_draft_keeps_the_fixtures_wording(section: str, key: str) -> None:
    doc = prereg.load(DRAFT).doc
    fixture = tomllib.loads(FIXTURE.read_text("utf-8"))
    assert doc[section][key] == fixture[section][key]


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
    for fact in (
        "seed 100",
        "10,000 steps",
        "16 and 32",
        "seeds 1-5",
        "stream",
        "128-dimensional",
        "edge_margin",
        "No collapse monitor fired",
    ):
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


# ---------------------------------------------------------------- the plan's own training line


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data dir with two tiny staircases, real runs of the simulator, for the line to find."""
    root = tmp_path_factory.mktemp("tiny")
    for tag, seed in (("train", 101), ("val", 102)):
        spec = run.Spec(
            L=jepa_ising.L,
            replicas=jepa_ising.REPLICAS,
            seed=seed,
            checkpoint_every=jepa_ising.FLUSH,
        )
        run.start(spec, jepa_ising.staircase(), root, tag, "cpu")
    return root


def _plan_line(root: Path, train_seed: str = "101") -> list[str]:
    """m1's `jepa train` as the runner would give it: the plan's flags, its inputs resolved, and
    only the sizes and data seeds changed, to name the tiny runs."""
    (m1, *_) = json.loads(PLAN.read_text("utf-8"))["runs"]
    args: list[str] = []
    for arg in m1["args"]:
        if isinstance(arg, dict):
            args.append(str({"record": jepa_ising.PILOT, "prereg": DRAFT}[arg["input"]]))
        else:
            args.append(str(arg))
    planned = [
        "--L",
        "16",
        "32",
        "64",
        "--train-seeds",
        "11",
        "12",
        "13",
        "14",
        "15",
        "--val-seed",
        "16",
    ]
    assert args[: len(planned)] == planned
    data = ["--L", str(jepa_ising.L), "--train-seeds", train_seed, "--val-seed", "102"]
    return ["train", "--data-dir", str(root), "--device", "cpu", *data, *args[len(planned) :]]


def test_the_plans_training_line_accepts_the_draft(
    tiny: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Placeholders and all: `jepa train --prereg` reads the file's id, version, data.seeds and
    bytes, and no key slice 5 changed. The manifest it would write names the file by the sha256 the
    plan pins."""
    calls: list[tuple[Any, ...]] = []

    def capture(*given: Any) -> Path:
        calls.append(given)
        return tiny

    monkeypatch.setattr(train, "start", capture)
    assert cli.main(_plan_line(tiny)) == 0
    ((config, data, _, tag, _, _, recorded),) = calls
    assert recorded == {
        "id": "ising-latent",
        "version": 1,
        "sha256": json.loads(PLAN.read_text("utf-8"))["inputs"]["prereg"]["sha256"],
        "seeds": [6, 7, 8, 9, 10],
    }
    assert config.steps == 25_000 and config.warmup == 1_250 and config.seed == 1
    assert (config.var_weight, config.cov_weight) == (0.025, 0.001)
    assert set(data.sources["refused_seeds"]) >= set(recorded["seeds"])
    assert tag == f"L{jepa_ising.L}-m1"


@pytest.mark.parametrize("seed", ["6", "10"])
def test_the_draft_is_what_refuses_seeds_6_to_10_on_the_plans_wheel(
    tiny: Path,
    seed: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The wheel at 7ba0792 has only seeds 1-5 in its own set, so `--prereg` must do it there."""

    def never(*_: object, **__: object) -> None:
        raise AssertionError("a scored seed reached training")

    monkeypatch.setattr(train, "start", never)
    monkeypatch.setattr(cli, "PREREG_SEEDS", frozenset(range(1, 6)))
    assert cli.main(_plan_line(tiny, seed)) == 1
    assert "scored by a pre-registration" in capsys.readouterr().err
