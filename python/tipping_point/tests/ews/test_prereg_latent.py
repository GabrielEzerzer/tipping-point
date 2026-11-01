"""The schema of a pre-registration with a [latents] section (#28): the draft of ising-latent-v1 in
tests/fixtures is accepted with its placeholders, each named at the freeze, and its trials count to
3,132; a file without the section is checked exactly as ising-ews-v1 was."""

import copy
import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
from tipping_point.ews import prereg
from tipping_point.jepa import probes, train

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DRAFT = FIXTURES / "ising_latent_draft.toml"
REPO = Path(__file__).resolve().parents[4]
V1 = REPO / "paper" / "prereg" / "ising-ews-v1.toml"
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
    doc["scoring"]["margins"]["tolerance"] = 0.01
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
    assert (dyn["edge_margin"], dyn["edge_rule"]) == ("window", prereg.EDGE_RULE)
    assert "or within one headline window of it" in dyn["edge_rule"]
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
        "scoring.margins.tolerance",
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
        (lambda d: d["latents"].update(extraction="every frame"), "latents.extraction: "),
        (lambda d: d["data"]["staircase"].update(temperature=2.3), "data.staircase.temperature"),
        (lambda d: d["scoring"]["margins"].update(sign_test=False), "sign_test: true"),
        (lambda d: d["scoring"]["margins"].update(ablations="judged"), "margins.ablations"),
        (lambda d: d["scoring"]["margins"].update(sizes="max"), "margins.sizes"),
        (lambda d: d["scoring"]["margins"].update(tolerance=0.0), "margins.tolerance: positive"),
        (lambda d: d["scoring"]["margins"].pop("tolerance"), "margins.tolerance: missing"),
        (lambda d: d["scoring"]["margins"].update(tolerance="a tenth"), "margins.tolerance: pos"),
        (lambda d: d["scoring"]["margins"].update(tolerance=True), "margins.tolerance: expected"),
        (lambda d: d["fss"]["dynamic"].pop("edge_margin"), "fss.dynamic.edge_margin: missing"),
        (lambda d: d["fss"]["dynamic"].update(edge_margin="none"), "edge_margin: 'window'"),
        (lambda d: d["fss"]["dynamic"].update(edge_margin=0.4), "edge_margin: expected str"),
        (
            lambda d: d["fss"]["dynamic"].update(
                edge_rule="inconclusive when more than half of an L's replicas take their extremum "
                "on an edge of the band"
            ),
            "fss.dynamic.edge_rule: ",
        ),
        (lambda d: d["scoring"]["margins"].pop("sizes"), "margins.sizes: missing"),
        (lambda d: d["hypotheses"].pop(4), "learned-beyond-random"),
        (lambda d: d["hypotheses"][4].update(direction="match"), "learned-beyond-random"),
        (lambda d: d["probes"].update(fit="some frames"), "probes.fit: "),
        (
            lambda d: d["probes"].update(fit=d["probes"]["fit"].replace("seed 16", "seed 15")),
            "probes.fit: ",
        ),
        (lambda d: d["probes"].update(test="all of them"), "probes.test: "),
        (
            lambda d: d["probes"].update(test=d["probes"]["test"].replace("20000", "10000")),
            "probes.test: ",
        ),
        (lambda d: d["data"].update(seeds=[6, 7, 9]), "data.seeds: consecutive"),
        (lambda d: d["probes"].update(selectivity="R^2"), "probes.selectivity"),
        (lambda d: d["probes"].update(targets=["abs_m"]), "must hold energy"),
        (
            lambda d: d["indicators"].update(names=["variance", "ac", "level", "skewness"]),
            "indicators.names: exactly",
        ),
        (lambda d: d["indicators"]["direction"].update(skewness="less"), "direction.skewness"),
        (
            lambda d: d["indicators"]["direction_by_observable"]["surprise"].update(ac="less"),
            "direction_by_observable.surprise.ac: no pair",
        ),
        (lambda d: d["latents"].update(columns=["surprise", "pc_1", "pc_3"]), "scored by nothing"),
        (lambda d: d["fss"]["static"][1].update(tolerance=0.1), "a descriptive row"),
        (lambda d: d["fss"]["static"][1].update(expected=-0.125), "a descriptive row"),
        (lambda d: d["fss"]["static"][2].update(expected=1.75), "a reference is judged"),
    ],
)
def test_malformed_latent_files_are_named(draft: dict[str, Any], edit: Any, complaint: str) -> None:
    edit(draft)
    assert any(complaint in problem for problem in prereg.problems(draft)), prereg.problems(draft)


@pytest.mark.parametrize(
    ("key", "value"),
    [("indicators", ["variance", "ac"]), ("statistic", "peak"), ("observable", "pc_1")],
)
def test_a_stale_v1_key_under_fss_dynamic_is_refused(
    draft: dict[str, Any], key: str, value: Any
) -> None:
    """Review, round 1: these are ising-ews-v1's keys, and _check passes a key the schema does not
    list. `indicators` beside the statistics went through draft and strict alike and froze, and the
    scorer then judged nothing: no inverse_nu row and no error. A latent file with any of them is
    refused now, draft or ready to freeze."""
    filled = _filled(draft)
    assert prereg.problems(filled, strict=True) == []
    draft["fss"]["dynamic"][key] = filled["fss"]["dynamic"][key] = value
    assert any(p.startswith(f"fss.dynamic.{key}: ") for p in prereg.problems(draft))
    assert any(p.startswith(f"fss.dynamic.{key}: ") for p in prereg.problems(filled, strict=True))


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


def _committed(repo: Path, text: bytes) -> Path:
    """`text` committed as paper/prereg/ising-latent-v1.toml in a scratch repository."""
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Test"),
        ("config", "user.email", "t@example.com"),
        ("config", "core.autocrlf", "false"),
    ):
        _git(repo, *args)
    path = repo / "paper" / "prereg" / "ising-latent-v1.toml"
    path.parent.mkdir(parents=True)
    path.write_bytes(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "draft")
    return path


def test_the_freeze_refuses_a_stale_v1_key_before_it_tags(tmp_path: Path) -> None:
    """The file fails loudly as it loads, so a stale key never reaches a tag."""
    header = b"[fss.dynamic]\n"
    stale = DRAFT.read_bytes().replace(header, header + b'indicators = ["variance", "ac"]\n')
    path = _committed(tmp_path, stale)
    with pytest.raises(prereg.PreregError, match=r"fss\.dynamic\.indicators: "):
        prereg.load(path)
    with pytest.raises(prereg.PreregError, match=r"fss\.dynamic\.indicators: "):
        prereg.freeze(tmp_path, "ising-latent")
    assert not _git(tmp_path, "tag", "--list")


def test_the_draft_and_the_readme_say_how_late_a_position_is_read(draft: dict[str, Any]) -> None:
    """Review, round 1: a trailing window's extremum is read half a window late. That is v1's frozen
    peak path, so positions are not corrected; a file says so instead, with the dead zone it puts
    near a true steepest change of +0.2 eps_L. The figure follows from its window and ramp rate."""
    window = draft["sensitivity"]["headline"]["window"]
    assert window * draft["data"]["ramps"]["x"] / 2 == pytest.approx(0.2)
    for path in (DRAFT, REPO / "README.md"):
        text = " ".join(path.read_text("utf-8").replace("#", " ").split())
        assert "read half the headline window late (0.2 eps_L)" in text, path
        assert "a read position within noise of 0" in text, path


def test_the_freeze_refuses_the_draft_as_it_is(tmp_path: Path) -> None:
    """Through the path a freeze takes: committed as paper/prereg/ising-latent-v1.toml, the draft is
    loaded, then refused for its placeholders, and no tag is made."""
    path = _committed(tmp_path, DRAFT.read_bytes())
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


@pytest.mark.parametrize(
    ("where", "value"),
    [
        (("scoring", "alarms", "indicators"), ["variance", "ac"]),  # v1's: pairs says which
        (("fss", "dynamic", "band"), [6.0, -4.0]),  # the research pass's name for search
        (("fss", "dynamic", "statistic"), "peak"),
        (("indicators", "thin"), 0.1),  # the dense cadence's, which a latents table has not
        (("latents", "subsample"), "tc-plus"),
        (("probes", "limit"), 20000),
        (("probes", "seed"), 0),
        (("scoring", "margins", "rounds"), 2000),
        (("scoring", "surrogates", "kinds"), ["phase"]),
        (("sensitivity", "steps"), 3),
        (("data", "nulls", "seeds"), [6]),
        (("baseline", "commit"), "abc"),
        (("notes",), "anything"),
    ],
)
def test_a_key_nothing_reads_is_refused_draft_or_ready_to_freeze(
    draft: dict[str, Any], where: tuple[str, ...], value: Any
) -> None:
    """Frozen text must not claim what the scorer ignores, and _check passes a key the schema does
    not list: a latent file holding one is refused, wherever it stands and whatever its name, by the
    path _latent_problems takes. The keys a latent file may leave out stay allowed."""
    filled = _filled(draft)
    assert prereg.problems(filled, strict=True) == []
    named = ".".join(where)
    for doc in (draft, filled):
        table = doc
        for key in where[:-1]:
            table = table[key]
        table[where[-1]] = value
    found = prereg.problems(draft)
    assert any(p.startswith(f"{named}: nothing reads") for p in found), found
    assert any(
        p.startswith(f"{named}: nothing reads") for p in prereg.problems(filled, strict=True)
    )


def test_a_key_inside_a_list_of_tables_is_found_too(draft: dict[str, Any]) -> None:
    draft["latents"]["checkpoints"][3]["device"] = "cuda"
    draft["fss"]["dynamic"]["statistics"][1]["band"] = [6.0, -4.0]
    draft["fss"]["static"][0]["band"] = [6.0, -4.0]
    found = prereg.problems(draft)
    for where in (
        "latents.checkpoints[3].device",
        "fss.dynamic.statistics[1].band",
        "fss.static[0].band",
    ):
        assert any(p.startswith(f"{where}: nothing reads") for p in found), (where, found)


def test_the_keys_a_latent_file_may_leave_out_are_not_taken_for_stale(
    draft: dict[str, Any],
) -> None:
    """data.ramps.eps_L, data.staircase.tau_sha256, direction_by_observable and a static row's
    expected and tolerance are read where they stand: the check knows them, and any key under a
    table the schema leaves open."""
    assert prereg.problems(draft) == []
    for key in ("eps_L",):
        assert key in draft["data"]["ramps"]
    assert "tau_sha256" in draft["data"]["staircase"]
    assert "direction_by_observable" in draft["indicators"]
    assert {"expected", "tolerance"} <= draft["fss"]["static"][0].keys()


def test_a_v1_file_keeps_its_own_keys_and_is_never_held_to_the_latent_rule() -> None:
    """scoring.alarms.indicators, fss.dynamic.statistic and indicators.thin are v1's own, and an
    extra key is not refused there: its frozen bytes must keep passing as they are."""
    v1 = tomllib.loads(V1.read_text("utf-8"))
    assert prereg.problems(v1, strict=True) == []
    assert "indicators" in v1["scoring"]["alarms"] and "thin" in v1["indicators"]
    v1["notes"] = "anything"
    assert prereg.problems(v1) == []


def test_the_hypothesis_and_the_level_must_state_one_direction(draft: dict[str, Any]) -> None:
    """The level's direction stands twice: the hypothesis's words and the key the scorer reads, and
    indicators.direction beside its override reads as a third. A file whose texts disagree is
    refused; while they are placeholders there is nothing to compare."""
    assert prereg.problems(draft) == []
    doc = _filled(draft)
    assert prereg.problems(doc, strict=True) == []
    doc["hypotheses"][2]["direction"] = "greater"
    assert any("surprise-trend says greater" in p for p in prereg.problems(doc))
    doc = _filled(draft)
    doc["indicators"]["direction"]["level"] = "greater"
    assert any(p.startswith("indicators.direction.level: no pair") for p in prereg.problems(doc))


def test_the_extraction_sentence_is_the_one_the_scorer_holds_tables_to(
    draft: dict[str, Any],
) -> None:
    """The draft's words, and each word's rule: a primary and its twin keep the plateau and one
    burst in 8 of a staircase, an ablation the plateau, each at the file's own plateau."""
    assert draft["latents"]["extraction"] == prereg.EXTRACTION
    assert set(prereg.SELECTIONS) == prereg.CHECKPOINT_ROLES
    assert prereg.SELECTIONS["primary"]["rule"] == "tc-plus"
    assert {prereg.SELECTIONS[role]["rule"] for role in ("ema-only", "held-out")} == {"tc-only"}
    assert {s["tc_plateau"] for s in prereg.SELECTIONS.values()} == {
        draft["data"]["staircase"]["temperature"]
    }
    assert "2.269231" in prereg.EXTRACTION and "one burst in eight" in prereg.EXTRACTION
    assert prereg.SELECTIONS["primary"]["every"] == 8


def test_the_probe_protocol_is_read_from_the_files_sentence(draft: dict[str, Any]) -> None:
    """The runner's frame limit and probe seed are the numbers the frozen sentence states."""
    assert prereg.probe_protocol(draft) == (20000, 0)
    fit = draft["probes"]["fit"].replace("20000", "500").replace("probe seed 0", "probe seed 7")
    draft["probes"] |= {
        "fit": fit,
        "test": draft["probes"]["test"].replace("20000", "500"),
    }
    assert prereg.problems(draft) == []
    assert prereg.probe_protocol(draft) == (500, 7)
    draft["probes"]["fit"] = "something else"
    with pytest.raises(prereg.PreregError, match=r"probes\.fit"):
        prereg.probe_protocol(draft)


def test_the_margins_tolerance_is_a_placeholder_until_the_freeze(draft: dict[str, Any]) -> None:
    """Typed as a static row's tolerance is: the draft holds the choice still open, a number or the
    placeholder both load, and only the freeze's check refuses the placeholder. Words that are
    neither are refused by any load."""
    assert draft["scoring"]["margins"]["tolerance"] == "<set before freeze>"
    assert prereg.problems(draft) == []
    assert any(
        p.startswith("scoring.margins.tolerance: placeholder")
        for p in prereg.problems(draft, strict=True)
    )
    filled = _filled(draft)
    assert prereg.problems(filled, strict=True) == []
    filled["scoring"]["margins"]["tolerance"] = -0.01
    assert any("margins.tolerance: positive" in p for p in prereg.problems(filled))


def test_edge_margin_is_a_latent_files_alone(draft: dict[str, Any]) -> None:
    """Required in a latent file, and refused in a file without [latents], whatever it says: v1's
    frozen file has no such key and passes as it did, and its dynamic path can never read one."""
    v1 = tomllib.loads(V1.read_text("utf-8"))
    assert "edge_margin" not in v1["fss"]["dynamic"]
    assert prereg.problems(v1, strict=True) == []
    v1["fss"]["dynamic"]["edge_margin"] = "window"
    found = prereg.problems(v1)
    assert len(found) == 1 and found[0].startswith("fss.dynamic.edge_margin: "), found
    filled = _filled(draft)
    del filled["fss"]["dynamic"]["edge_margin"]
    assert "fss.dynamic.edge_margin: missing" in prereg.problems(filled, strict=True)


def test_trials_go_in_one_write_in_the_lines_log_trial_always_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes v1's scoring logs per trial, sorted keys and NaN as JSON has it, whatever came
    before them in the file, under one clock and one sync: one write is what a crash cannot split
    between trials the way one line at a time could."""
    records = [
        {"prereg": {"id": "x", "version": 1}, "cell": {"L": 16, "bandwidth": float("nan")}, "a": i}
        for i in range(3)
    ]
    path = prereg.trials_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{}\n")
    synced: list[int] = []
    real = os.fsync
    monkeypatch.setattr(prereg.os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    monkeypatch.setattr(prereg.time, "time_ns", lambda: 7)
    prereg.log_trials(tmp_path, records)
    expected = [json.dumps({"logged_ns": 7, **r}, sort_keys=True) + "\n" for r in records]
    assert path.read_bytes() == b"{}\n" + "".join(expected).encode()
    assert len(synced) == 1
    prereg.log_trial(tmp_path, records[0])
    assert path.read_bytes().endswith(expected[0].encode()) and len(synced) == 2
