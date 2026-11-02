"""ising-latent-v1 as rewritten after the m1 and m2 reads (#28), and the plans that trained under
its draft: the file loads, the freeze's strict check names baseline.result alone, its rules and pins
are the ones the rewrite decided, its disclosure says what was read, and the plans still pin the
draft the trainings ran under, whose sha256 the filled file no longer has."""

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
PREREG = ROOT / "paper" / "prereg" / "ising-latent-v1.toml"
FIXTURE = ROOT / "python" / "tipping_point" / "tests" / "fixtures" / "ising_latent_draft.toml"
PLAN = ROOT / "notebooks" / "plans" / "primary-28.json"
ABLATIONS_PLAN = ROOT / "notebooks" / "plans" / "ablations-28.json"
# the draft the trainings ran under; the file's sha256 differs because its placeholders were filled
TRAINED_UNDER = "4d9fd3ed79e02f3e0ec535dd02e2b4f8c6a716ceed5476a6a00fcc72e95552ad"
BASELINE = "<ews result run_id, filled when v1 has been scored>"  # until ising-ews-v1's score ends
PINS = [  # role, model seed, training, final.pt's sha256, ckpt/step_0000000.pt's sha256 (#28)
    (
        "primary",
        1,
        "20261101T051047Z-L16-32-64-m1",
        "3f251147a5072964176cab40122cbf855755d7a7dd77bf946eaaef661b43d149",
        "5e076a53e61d8c359d55c13738f8a2619e4af1002843de0c10b3ca33bc6d4224",
    ),
    (
        "primary",
        2,
        "20261101T051110Z-L16-32-64-m2",
        "7638b436d5ee0cb39f68e6ce62f965b0c0c4f66ff9c402e80e93df0afad9aef4",
        "d22f7e494b6ee40bc6e319173e183c8c21709a3f5d149a0251b03fac84415176",
    ),
    (
        "primary",
        3,
        "20261101T061841Z-L16-32-64-m3",
        "869427f1e895b06159cf4619690df578d2bf73c5259ac4ec2a2d062d240d08cb",
        "0a3836fa6984ce42601cc61b28671ac53768679d7f8ace62793d2e9a9ec32777",
    ),
    (
        "primary",
        4,
        "20261101T061932Z-L16-32-64-m4",
        "1b46a30123ca3e4f861d6ca67d922c3451d857a364a6ec19f2b2a8a3d4671714",
        "3024657c3dd1c3c816c028669e6a333459d17c58dd84cf84affd242771431fff",
    ),
    (
        "primary",
        5,
        "20261101T072708Z-L16-32-64-m5",
        "0f3a58d8350bdae307798481a3152518a0084ab3d83b43f0332aeb0a21aba6b1",
        "c8c0a8999d1b77a078e7034d5643f67e0458b5a163d15c5c6ee5179258d6d76b",
    ),
    (
        "ema-only",
        11,
        "20261102T103007Z-L16-32-64-m11",
        "eb9c53c03427a3f24434b4de326501c215f73652213b4bb1a1f3c3706b4d223d",
        "e3791615d8ca59722ae74e23ceff207f7ceacaf2b524ea070919983884590af2",
    ),
    (
        "ema-only",
        12,
        "20261102T103030Z-L16-32-64-m12",
        "fd4d4c3ed41806b52744debd622223d45e2ada733895a94ac7ffd1fd6d46e1b8",
        "353bfd22ee0ccca4a994cc84daa4cc2462904e55b4c297bb6f7b55516638851f",
    ),
    (
        "ema-only",
        13,
        "20261102T113627Z-L16-32-64-m13",
        "a9f536a28390f0f34186c90e38006a3bf11011b42439be9b510590d88c0b6422",
        "3effa35cda0cdf639943ada961a77342472333e2b0cb355f27af327766427dc0",
    ),
    (
        "held-out",
        21,
        "20261102T113656Z-L16-32-64-m21-heldout",
        "3dc8a07bb342a3cb9215ca95d315420dded10bfe72cd20e0bfffd7af65989296",
        "65a99dd8807b332a490c4907af03e946e2d652c9fdd390fff68e5e08f7b62006",
    ),
    (
        "held-out",
        22,
        "20261102T124232Z-L16-32-64-m22-heldout",
        "add45912a55f4ceb3c3c2b56b1e305f221ecdbb34025adc483cb9c9e16d16541",
        "c4458678b51addb4040ed08decd9daf01125af8cf3b9b61eaba11687ae3d5b93",
    ),
    (
        "held-out",
        23,
        "20261102T124357Z-L16-32-64-m23-heldout",
        "3a800b838824f25394e654b9e7084c32e01e9733485fa5c870d512bca06b16e9",
        "825336443b4f7845e83673429bf7e91cca7c3f6dcd9b47c54b8944e3abdb446a",
    ),
]


def test_the_file_loads_and_names_its_trials() -> None:
    doc = prereg.load(PREREG).doc
    assert prereg.problems(doc) == []
    assert prereg.cells(doc) == 3132 == doc["trials"]["budget"]


def test_strict_refuses_baseline_result_alone() -> None:
    """Every placeholder of the draft is filled but baseline.result: ising-ews-v1's score of these
    runs has not completed, so no run id exists to name, and no value stands in for one. The test
    changes with the file when it is filled."""
    found = prereg.problems(prereg.load(PREREG).doc, strict=True)
    assert found == [f"baseline.result: placeholder {BASELINE!r}"]
    assert PREREG.read_text("utf-8").count(BASELINE) == 1


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
    doc = copy.deepcopy(prereg.load(PREREG).doc)
    undo(doc)
    assert any(complaint in problem for problem in prereg.problems(doc)), prereg.problems(doc)


@pytest.mark.parametrize(
    ("section", "key"),
    [("latents", "extraction"), ("probes", "fit"), ("probes", "test"), ("probes", "selectivity")],
)
def test_the_file_keeps_the_fixtures_wording(section: str, key: str) -> None:
    doc = prereg.load(PREREG).doc
    fixture = tomllib.loads(FIXTURE.read_text("utf-8"))
    assert doc[section][key] == fixture[section][key]


def test_the_checkpoints_are_the_eleven_trainings_pinned_by_sha256() -> None:
    """Five primaries and six ablations, in the draft's order, each training's run id with the
    sha256 of its final.pt and of its step-0 twin, as the trainings' outputs give them."""
    keys = ("role", "model_seed", "training", "trained_sha256", "twin_sha256")
    entries = prereg.load(PREREG).doc["latents"]["checkpoints"]
    assert [tuple(entry[key] for key in keys) for entry in entries] == PINS


def test_the_rules_are_the_ones_rewritten_after_both_reads() -> None:
    """The pipeline checks judged on the twins alone, the surprise variance and the two former
    reference rows descriptive, the slope gone, the margins' sign test at every size beside a
    tolerance of 0.002, and the scoring-time levers of 1 November: alarm scores every 5 tau_c, and
    surrogate tests at the headline cell for h = 0 alone, with its reason beside it."""
    doc = prereg.load(PREREG).doc
    static = {row["quantity"]: row for row in doc["fss"]["static"]}
    assert static["l2_var_pc_1"] == {
        "quantity": "l2_var_pc_1",
        "role": "pipeline-check",
        "expected": 1.75,
        "tolerance": 0.15,
        "judged": "twin",
    }
    for quantity in ("l2_var_surprise", "one_minus_ac1_pc_1"):
        assert static[quantity] == {"quantity": quantity, "role": "descriptive"}, quantity
    assert "reference" not in {row["role"] for row in doc["fss"]["static"]}
    statistics = doc["fss"]["dynamic"]["statistics"]
    peak = {"indicator": "variance", "extremum": "max"}
    assert statistics == [
        {"observable": "pc_1", **peak, "role": "pipeline-check", "judged": "twin"},
        {"observable": "surprise", **peak, "role": "descriptive"},
    ]
    ind = doc["indicators"]
    assert "slope" not in ind["names"]
    assert "slope" not in {name for _, name in ind["pairs"]}
    assert "slope" not in {entry["indicator"] for entry in statistics}
    margins = doc["scoring"]["margins"]
    assert (margins["sizes"], margins["tolerance"]) == ("every", 0.002)
    assert (margins["sign_test"], margins["ablations"]) == (True, "descriptive")
    assert doc["scoring"]["alarms"]["every"] == 5
    assert doc["scoring"]["surrogates"]["cells"] == "headline-h0"
    lines = PREREG.read_text("utf-8").splitlines()
    (cells,) = [line for line in lines if line.startswith("cells = ")]
    assert "the headline cell for h = 0 only" in cells and "1 November" in cells


def test_the_hypotheses_say_what_is_judged_and_what_is_reported() -> None:
    """The same seven names and directions; predictions as the rewrite states them: the twins
    judged where the trained checkpoints are reported, one verdict over five primaries, surprise's
    level descriptive, and "learned" on (a) and (b) alone, positive at every size."""
    hypotheses = prereg.load(PREREG).doc["hypotheses"]
    assert {h["name"]: h["direction"] for h in hypotheses} == {
        "latent-pipeline-check": "match",
        "latent-variance-rises": "greater",
        "surprise-trend": "less",
        "latent-dynamic-fss": "match",
        "learned-beyond-random": "greater",
        "held-out-band": "two-sided",
        "ema-only": "two-sided",
    }
    said = {h["name"]: h["prediction"] for h in hypotheses}
    assert said["latent-pipeline-check"].startswith("kappa within 0.15 of 1.75 for every twin;")
    assert "kappa is reported beside it and not judged" in said["latent-pipeline-check"]
    rises = said["latent-variance-rises"]
    assert "all five primary checkpoints' AUC 95% CIs lie above 0.5" in rises
    assert said["surprise-trend"].startswith("Descriptive:")
    assert said["surprise-trend"].endswith("no verdict.")
    assert "every twin's Var(pc_1)" in said["latent-dynamic-fss"]
    assert "slope" not in said["latent-dynamic-fss"]
    learned = said["learned-beyond-random"]
    assert "all fifteen paired margins (five seeds at three sizes) are positive" in learned
    assert learned.endswith("claimed only if (a) and (b) hold.") and "(c)" not in learned
    assert "10 of the validation seed's 40 plateaus" in said["held-out-band"]
    assert "refuses as collapsed" in said["ema-only"]


def test_the_file_keeps_the_passes_values_and_v1s_runs() -> None:
    doc = prereg.load(PREREG).doc
    ind, surprise = doc["indicators"], doc["indicators"]["direction_by_observable"]["surprise"]
    assert doc["hypotheses"][2]["direction"] == surprise["level"] == ind["direction"]["level"]
    assert surprise == {"variance": "greater", "level": "less"}
    assert doc["latents"]["steps"] == 25_000
    v1 = tomllib.loads((ROOT / "paper" / "prereg" / "ising-ews-v1.toml").read_text("utf-8"))
    assert doc["data"]["tau_c"] == v1["data"]["tau_c"]
    assert doc["data"]["staircase"]["tau_sha256"] == v1["data"]["staircase"]["tau_sha256"]
    assert doc["data"]["seeds"] == v1["data"]["seeds"] == [6, 7, 8, 9, 10]


def test_the_disclosure_says_what_was_seen() -> None:
    """The pilot as before; then both reads, with the figures that moved the rules, the models
    left unread, the ablations' monitors, the baseline scored before the freeze, and the draft the
    trainings name. Nothing the draft said of a value still to be set remains."""
    d = prereg.load(PREREG).doc["disclosure"]
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
        "model seed 1 (m1) on 1 November",
        "model seed 2 (m2) on 2 November",
        "no seeds 1-5 run holds frames at L = 64",
        "+0.265 on m1 and -0.226 on m2, beside +1.644 and +1.673",
        "0.995 [0.988, 0.999] on m1 and 0.026 [0.011, 0.048] on m2",
        "0.495 [0.378, 0.602] on m1 and 0.869 [0.817, 0.914] on m2",
        "81-89%",
        "+0.146 on m1 and +0.252 on m2",
        "-0.681 and -0.344",
        "+0.0047",
        "0.0014 and 0.0016",
        "m3, m4 and m5 were not read before this rewrite",
        "six ablation trainings were read on 2 November",
        "about 38 of 128",
        "about 120",
        "scored before this freeze",
        "scoring.margins.tolerance is 0.002",
        "8-component probe",
        "headline-h0",
        TRAINED_UNDER,
    ):
        assert fact in text, fact
    for stale in ("still to be set", "before any latent number existed", "slope's extremum"):
        assert stale not in text, stale


def test_the_plans_pin_the_draft_the_trainings_ran_under() -> None:
    """Five model seeds, one training at a time per T4, on the file's seeds and steps. Both plans
    name the draft they trained under by its sha256, which the filled file no longer has. The
    runner itself checks the plan's form (--self-test)."""
    doc = prereg.load(PREREG).doc
    plan = json.loads(PLAN.read_text("utf-8"))
    ablations = json.loads(ABLATIONS_PLAN.read_text("utf-8"))
    assert plan["inputs"]["prereg"] == ablations["inputs"]["prereg"]
    assert plan["inputs"]["prereg"]["sha256"] == TRAINED_UNDER != prereg.load(PREREG).sha256
    runs = plan["runs"]
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
            args.append(str({"record": jepa_ising.PILOT, "prereg": PREREG}[arg["input"]]))
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


def test_the_plans_training_line_accepts_the_file(
    tiny: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`jepa train --prereg` reads the file's id, version, data.seeds and bytes, and the rewrite
    changed only the bytes: the manifest it would write names the filled file by its own sha256,
    where the plan pins the draft's."""
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
        "sha256": prereg.load(PREREG).sha256,
        "seeds": [6, 7, 8, 9, 10],
    }
    assert json.loads(PLAN.read_text("utf-8"))["inputs"]["prereg"]["sha256"] == TRAINED_UNDER
    assert recorded["sha256"] != TRAINED_UNDER  # the bytes changed; the plan's pin did not
    assert config.steps == 25_000 and config.warmup == 1_250 and config.seed == 1
    assert (config.var_weight, config.cov_weight) == (0.025, 0.001)
    assert set(data.sources["refused_seeds"]) >= set(recorded["seeds"])
    assert tag == f"L{jepa_ising.L}-m1"


@pytest.mark.parametrize("seed", ["6", "10"])
def test_the_file_is_what_refuses_seeds_6_to_10_on_the_plans_wheel(
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
