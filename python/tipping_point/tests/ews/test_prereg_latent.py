"""The schema of a pre-registration with a [latents] section (#28): the draft of ising-latent-v1 in
tests/fixtures is accepted with its placeholders, each named at the freeze, and its trials count to
3,132; a file without the section is checked exactly as ising-ews-v1 was."""

import copy
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
from tipping_point.ews import prereg
from tipping_point.jepa import probes, train

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DRAFT = FIXTURES / "ising_latent_draft.toml"
V1 = Path(__file__).resolve().parents[4] / "paper" / "prereg" / "ising-ews-v1.toml"
SHA = "a" * 64


@pytest.fixture
def draft() -> dict[str, Any]:
    return prereg.load(DRAFT).doc


def _filled(doc: dict[str, Any]) -> dict[str, Any]:
    """The draft with every placeholder set, as a file ready to freeze."""
    doc = copy.deepcopy(doc)
    doc["hypotheses"][2]["direction"] = "less"
    doc["data"]["tau_c"] |= {
        "pilot_run": "20261025T194846Z-tauc-L16-32-64-128",
        "pilot_sha256": SHA,
    }
    doc["data"]["staircase"]["tau_sha256"] = {str(L): SHA for L in doc["data"]["L"]}
    doc["baseline"]["result"] = "20261201T000000Z-ising-ews-v1"
    for i, entry in enumerate(doc["latents"]["checkpoints"]):
        entry |= {"training": f"training-{i}", "trained_sha256": SHA, "twin_sha256": SHA}
    doc["indicators"]["direction"]["level"] = "less"
    doc["indicators"]["direction_by_observable"]["surprise"] = {
        "variance": "greater",
        "level": "less",
    }
    for row in doc["fss"]["static"]:
        if row["role"] == "reference":
            row["tolerance"] = 0.1
    doc["fss"]["dynamic"]["statistics"][2]["extremum"] = "min"
    doc["disclosure"] |= {"exploratory": "seeds 1-5", "decisions": "those", "confirmatory": "6-10"}
    return doc


def test_the_draft_is_accepted_with_its_placeholders(draft: dict[str, Any]) -> None:
    """Draft mode: load takes the file, placeholders in a direction, an extremum and a reference's
    tolerance included."""
    assert prereg.problems(draft) == []
    assert (draft["id"], draft["version"]) == ("ising-latent", 1)


def test_the_draft_holds_the_decisions_on_the_research_pass(draft: dict[str, Any]) -> None:
    """The owner's calls the schema carries (#28): pc_1 and surprise judged, level and slope on the
    raw series with level's rows on the phase surrogate, N_s = 999 for primaries and twins and 199
    for ablations, the tolerances of call 13, L = 128 out, 5 primaries and 6 ablations."""
    ind, dyn = draft["indicators"], draft["fss"]["dynamic"]
    assert set(ind["observables"]) == set(draft["latents"]["columns"]) == {"pc_1", "surprise"}
    assert ind["pairs"] == [
        ["pc_1", "variance"],
        ["pc_1", "ac"],
        ["surprise", "variance"],
        ["surprise", "level"],
    ]
    assert [(s["observable"], s["indicator"]) for s in dyn["statistics"]] == [
        ("pc_1", "variance"),
        ("surprise", "variance"),
        ("surprise", "slope"),
    ]
    tests = draft["scoring"]["surrogates"]
    assert (tests["n"], tests["n_ablation"], tests["cells"]) == (999, 199, "headline")
    assert (dyn["expected_inverse_nu"], dyn["tolerance"]) == (1.0, 0.5)
    static = {row["quantity"]: row for row in draft["fss"]["static"]}
    assert (static["l2_var_pc_1"]["expected"], static["l2_var_pc_1"]["tolerance"]) == (1.75, 0.15)
    assert static["l2_var_pc_1"]["role"] == "pipeline-check"
    assert draft["data"]["L"] == [16, 32, 64] and draft["data"]["seeds"] == [6, 7, 8, 9, 10]
    roles = [(c["role"], c["model_seed"]) for c in draft["latents"]["checkpoints"]]
    assert roles == [
        *(("primary", seed) for seed in range(1, 6)),
        *(("ema-only", seed) for seed in (11, 12, 13)),
        *(("held-out", seed) for seed in (21, 22, 23)),
    ]
    assert draft["baseline"]["prereg"] == "ising-ews-v1"


def test_the_drafts_trials_are_3132(draft: dict[str, Any]) -> None:
    """10 x 270 + 12 x 36. Each primary checkpoint and its twin: 3 pairs x 3 L x 3 h x 9 cells plus
    level's 3 L x 3 h x 3 windows. Each ablation checkpoint and its twin: 4 pairs x 3 L x 3 h at the
    headline cell. The file's budget is that count."""
    assert prereg.cells(draft) == 3132 == draft["trials"]["budget"]
    primary_only = copy.deepcopy(draft)
    primary_only["latents"]["checkpoints"] = draft["latents"]["checkpoints"][:5]
    assert prereg.cells(primary_only) == 10 * 270
    ablations = copy.deepcopy(draft)
    ablations["latents"]["checkpoints"] = draft["latents"]["checkpoints"][5:]
    assert prereg.cells(ablations) == 12 * 36


def test_a_grown_file_needs_a_budget_to_match(draft: dict[str, Any]) -> None:
    """A twelfth checkpoint, one more primary, spends 540 more trials than the budget allows."""
    doc = _filled(draft)
    assert prereg.problems(doc, strict=True) == []
    doc["latents"]["checkpoints"].append(doc["latents"]["checkpoints"][0] | {"model_seed": 6})
    assert prereg.cells(doc) == 3132 + 540
    assert any("trials.budget" in p for p in prereg.problems(doc, strict=True))


def test_strict_names_every_placeholder(draft: dict[str, Any]) -> None:
    """The freeze's check names each one, the two choices left open as <a|b, ...> among them, and
    passes once they are all set: so no draft value reaches a frozen file."""
    found = prereg.problems(draft, strict=True)
    assert prereg.problems(_filled(draft), strict=True) == []
    named = {p.split(":")[0] for p in found if "placeholder" in p}
    for where in (
        "hypotheses[2].direction",
        "data.tau_c.pilot_run",
        "data.tau_c.pilot_sha256",
        "data.staircase.tau_sha256.16",
        "baseline.result",
        "latents.checkpoints[0].training",
        "latents.checkpoints[10].trained_sha256",
        "latents.checkpoints[10].twin_sha256",
        "indicators.direction.level",
        "indicators.direction_by_observable.surprise.variance",
        "indicators.direction_by_observable.surprise.level",
        "fss.static[2].tolerance",
        "fss.static[4].tolerance",
        "fss.dynamic.statistics[2].extremum",
        "disclosure.exploratory",
    ):
        assert where in named, (where, named)
    assert any("data.staircase.tau_sha256" in p and "16" in p for p in found)


@pytest.mark.parametrize(
    "key",
    ["trained_sha256", "twin_sha256"],
)
def test_strict_wants_a_checkpoint_pinned_by_its_sha256(draft: dict[str, Any], key: str) -> None:
    """A file named by something other than a sha256 pins nothing the scorer can check."""
    doc = _filled(draft)
    assert prereg.problems(doc, strict=True) == []
    doc["latents"]["checkpoints"][3][key] = "final.pt"
    assert prereg.problems(doc) == []  # a draft may still say so
    assert prereg.problems(doc, strict=True) == [
        f"latents.checkpoints[3].{key}: a sha256 (64 hex digits)"
    ]


@pytest.mark.parametrize(
    ("edit", "complaint"),
    [
        (lambda d: d.pop("baseline"), "baseline: missing"),
        (lambda d: d["baseline"].update(prereg="ising ews"), "baseline.prereg"),
        (lambda d: d["latents"].pop("checkpoints"), "latents.checkpoints: missing"),
        (lambda d: d["latents"].update(table="configs"), "latents.table"),
        (lambda d: d["latents"].update(columns=["pc_9"]), "latents.columns"),
        (lambda d: d["latents"].update(columns=["pc_1"]), "indicators.observables"),
        (lambda d: d["latents"].update(steps="many"), "latents.steps: expected int"),
        (lambda d: d["latents"]["checkpoints"][0].update(role="boss"), "checkpoints[0].role"),
        (lambda d: d["latents"]["checkpoints"][0].pop("twin_sha256"), "twin_sha256: missing"),
        (lambda d: d["latents"]["checkpoints"][1].update(model_seed=1), "one entry per model_seed"),
        (
            lambda d: [c.update(training="the-same") for c in d["latents"]["checkpoints"][:2]],
            "one entry per training",
        ),
        (lambda d: d["latents"].update(training_seeds=[6]), "never trained on"),
        (lambda d: d["latents"].update(validation_seed=10), "never trained on"),
        (lambda d: d["indicators"].update(cadences=["dense", "matched"]), "indicators.cadences"),
        (lambda d: d["indicators"].update(observables=["abs_m"]), "indicators.observables"),
        (lambda d: d["indicators"].update(names=["kurtosis"]), "indicators.names"),
        (lambda d: d["indicators"].update(pairs=[["pc_1", "skewness"]]), "indicators.pairs"),
        (lambda d: d["indicators"].update(pairs=[["pc_2", "variance"]]), "indicators.pairs"),
        (lambda d: d["indicators"].update(pairs=[["pc_1"]]), "indicators.pairs"),
        (lambda d: d["indicators"].update(pairs=[]), "indicators.pairs"),
        (lambda d: d["indicators"].update(pairs=[["pc_1", "ac"]] * 2), "indicators.pairs"),
        (lambda d: d["indicators"]["direction"].update(level="up"), "indicators.direction.level"),
        (lambda d: d["indicators"]["direction"].pop("ac"), "indicators.direction.ac"),
        (
            lambda d: d["indicators"]["direction_by_observable"]["surprise"].update(level="up"),
            "direction_by_observable.surprise.level",
        ),
        (
            lambda d: d["indicators"]["direction_by_observable"].update(energy={"ac": "less"}),
            "direction_by_observable.energy",
        ),
        (lambda d: d["hypotheses"][2].update(direction="up"), "hypotheses[2].direction"),
        (lambda d: d["scoring"]["surrogates"].pop("n_ablation"), "n_ablation: missing"),
        (lambda d: d["scoring"]["surrogates"].update(n_ablation=0), "n_ablation"),
        (lambda d: d["scoring"]["margins"].update(sign_test="yes"), "sign_test: expected bool"),
        (lambda d: d["scoring"].pop("margins"), "scoring.margins: missing"),
        (lambda d: d["fss"]["dynamic"].pop("statistics"), "fss.dynamic.statistics: missing"),
        (lambda d: d["fss"]["dynamic"].update(statistics=[]), "statistics: at least one"),
        (lambda d: d["fss"]["dynamic"].pop("search"), "fss.dynamic.search: missing"),
        (lambda d: d["fss"]["dynamic"].pop("min_draw_share"), "min_draw_share: missing"),
        (lambda d: d["fss"]["dynamic"].update(cadence="dense"), "fss.dynamic.cadence: one of"),
        (
            lambda d: d["fss"]["dynamic"]["statistics"][0].update(observable="energy"),
            "statistics[0].observable",
        ),
        (
            lambda d: d["fss"]["dynamic"]["statistics"][0].update(indicator="kurtosis"),
            "statistics[0].indicator",
        ),
        (
            lambda d: d["fss"]["dynamic"]["statistics"][0].update(extremum="mid"),
            "statistics[0].extremum",
        ),
        (lambda d: d["fss"]["dynamic"]["statistics"][0].update(role="x"), "statistics[0].role"),
        (lambda d: d["fss"]["static"][0].update(quantity="l2_var_abs_m"), "fss.static[0]"),
        (lambda d: d["fss"]["static"][0].update(quantity="l2_var_pc_2"), "fss.static[0]"),
        (lambda d: d["fss"]["static"][0].update(quantity="skewness_pc_1"), "fss.static[0]"),
        (lambda d: d["fss"]["static"][0].pop("tolerance"), "a pipeline check needs"),
        (lambda d: d["fss"]["static"][2].pop("tolerance"), "a reference needs its own tolerance"),
        (lambda d: d["fss"]["static"][2].update(tolerance="a tenth"), "fss.static[2].tolerance"),
        (lambda d: d["probes"].pop("ridge_alpha"), "probes.ridge_alpha: missing"),
        (lambda d: d["probes"].update(ridge_alpha=0.0), "probes.ridge_alpha: positive"),
        (lambda d: d["probes"].update(targets=["susceptibility"]), "probes.targets"),
        (lambda d: d["probes"].update(controls=["iid"]), "probes.controls"),
        (lambda d: d.pop("disclosure"), "disclosure: missing"),
    ],
)
def test_malformed_latent_files_are_named(draft: dict[str, Any], edit: Any, complaint: str) -> None:
    edit(draft)
    assert any(complaint in problem for problem in prereg.problems(draft)), prereg.problems(draft)


def test_every_enumerated_placeholder_is_a_choice_only_in_a_latent_file() -> None:
    """A file with no [latents] section is checked as v1 was: a placeholder in a direction is not a
    direction, and it may name neither a latent column nor a latent indicator, role or quantity. The
    latent keys it does not have are not asked of it."""
    v1 = tomllib.loads(V1.read_text("utf-8"))
    assert prereg.problems(v1) == []

    def complaints(edit: Any) -> list[str]:
        doc = copy.deepcopy(v1)
        edit(doc)
        return prereg.problems(doc)

    assert complaints(lambda d: d["hypotheses"][0].update(direction="<set before freeze>"))
    assert complaints(lambda d: d["indicators"]["direction"].update(ac="<set before freeze>"))
    assert complaints(lambda d: d["indicators"].update(observables=["surprise"]))
    assert complaints(lambda d: d["indicators"].update(names=["level"]))
    assert complaints(lambda d: d["indicators"]["direction"].update(level="greater")) == []
    assert complaints(lambda d: d["fss"]["static"][0].update(role="pipeline-check"))
    assert complaints(lambda d: d["fss"]["static"][0].update(quantity="l2_var_pc_1"))


def test_the_placeholder_pattern_leaves_v1s_text_alone() -> None:
    """The freeze's scan looks for "<a|b, ...>" as well as "<words>" now, and v1's long predictions
    hold both |m| and a < sign: none of it reads as a placeholder."""
    raw = V1.read_text("utf-8")
    assert "|m|" in raw and "draws_used <" in raw
    assert prereg.problems(tomllib.loads(raw), strict=True) == []
    for text in (
        "<min|max, set before freeze>",
        "<greater|less, fixed by a pass>",
        "<words>",
        "TBD",
    ):
        assert prereg._PLACEHOLDER.search(text), text
    for text in ("|m| and <|m|> at Tc", "t <tc and |m|> 0", "draws_used < 0.95", "a|b"):
        assert prereg._PLACEHOLDER.search(text) is None, text


def test_the_probe_vocabulary_is_the_probes_own() -> None:
    """ews cannot import jepa, so prereg.py holds the probes' names itself: they must stay its."""
    assert set(probes.TARGETS) == prereg.PROBE_TARGETS
    assert set(probes.COLUMNS) - {"raw"} == prereg.PROBE_CONTROLS


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_a_training_that_names_the_draft_refuses_the_seeds_it_scores() -> None:
    """Call 11: the file exists before training, so `jepa train --prereg` refuses seeds 6-10 by what
    it reads from it, placeholders and all."""
    assert train.prereg_record(DRAFT)["seeds"] == [6, 7, 8, 9, 10]


def test_the_freeze_refuses_the_draft_as_it_is(tmp_path: Path) -> None:
    """Through the path a freeze takes: committed as paper/prereg/ising-latent-v1.toml, the draft is
    loaded, then refused for its placeholders, and no tag is made."""
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Test"),
        ("config", "user.email", "t@example.com"),
        ("config", "core.autocrlf", "false"),
    ):
        _git(tmp_path, *args)
    path = tmp_path / "paper" / "prereg" / "ising-latent-v1.toml"
    path.parent.mkdir(parents=True)
    path.write_bytes(DRAFT.read_bytes())
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "draft")
    assert prereg.load(path).name == "ising-latent-v1"
    with pytest.raises(prereg.PreregError, match="not ready to freeze") as refused:
        prereg.freeze(tmp_path, "ising-latent")
    assert "hypotheses[2].direction: placeholder" in str(refused.value)
    assert not _git(tmp_path, "tag", "--list")


def test_a_classic_file_has_no_baseline_commit() -> None:
    """ising-ews-v1 reads its own runs: nothing about its scoring turns on another tag."""
    assert prereg.baseline_commit(prereg.load(V1)) is None


def test_log_trial_keeps_the_checkpoint_it_is_given(tmp_path: Path) -> None:
    """The line a scoring writes per AUC cell names its checkpoint, and trials keep being counted
    per version whatever a cell holds."""
    latent = prereg.load(DRAFT)
    stamp = {"id": latent.id, "version": latent.version}
    prereg.log_trial(tmp_path, {"prereg": stamp, "cell": {"checkpoint": "t-step0050000", "L": 16}})
    prereg.log_trial(tmp_path, {"prereg": stamp, "cell": {"checkpoint": "t-step0000000", "L": 16}})
    assert prereg.trials(tmp_path, latent) == 2
    lines = prereg.trials_path(tmp_path).read_text("utf-8")
    assert '"checkpoint": "t-step0050000"' in lines and '"checkpoint": "t-step0000000"' in lines
