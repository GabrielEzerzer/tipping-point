"""Paired margins of a trained checkpoint over its twin, on planted per-seed quantities: the
nested bootstrap's CI and its coverage, the sign test, the verdicts, and the refusal of
misaligned draws."""

from typing import Any

import numpy as np
import pytest
from tipping_point.ews import fss, margins
from tipping_point.ews.ising import latents_name

SEEDS, DRAWS = 5, 400


def _rng() -> np.random.Generator:
    return np.random.default_rng(28)


def _draws(
    trained: np.ndarray,
    twin: np.ndarray,
    rng: np.random.Generator,
    common: float = 0.0,
    apart: float = 0.0,
    own: float = 0.0,
    columns: int = DRAWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Data-bootstrap draws as margins.paired takes them, column b one resample of the data for
    every checkpoint: it moves them all alike by `common` (which cancels in the margin), every
    trained checkpoint against its twin by `apart` (the data's noise in the margin, the same for
    all seeds), and each checkpoint a little on its own by `own`."""
    shock, split = rng.normal(0.0, common, columns), rng.normal(0.0, apart, columns)
    seeds = len(trained)
    return (
        trained[:, None] + shock + split + rng.normal(0.0, own, (seeds, columns)),
        twin[:, None] + shock + rng.normal(0.0, own, (seeds, columns)),
    )


def test_a_planted_margin_passes() -> None:
    """Every seed gains 0.5 over its twin, though the seeds' levels differ by more than that and
    each data resample moves every checkpoint by twice as much again. Paired draw by draw, the
    resample's shift cancels and the CI excludes 0; against another resample, it spans 0."""
    rng = np.random.default_rng(1)
    twin = rng.normal(1.0, 0.6, SEEDS)
    trained = twin + 0.5 + rng.normal(0.0, 0.05, SEEDS)
    draws = _draws(trained, twin, rng, common=1.0, own=0.05)
    row = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(2), *draws)
    assert row["margins"] == pytest.approx((trained - twin).tolist())
    assert row["margin"] == pytest.approx(float(np.mean(trained - twin)))
    assert 0 < row["lo"] < row["margin"] < row["hi"] < 1.0
    assert row["sign_test"] == fss.PASS and row["verdict"] == fss.PASS
    assert (row["n_seeds"], row["n_draws"], row["n_bootstrap"]) == (SEEDS, DRAWS, 2_000)
    assert (row["direction"], row["level"], row["tolerance"]) == ("greater", 0.95, 0.1)
    again = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(2), *draws)
    assert again == row  # a seeded rng gives the same row
    unpaired = margins.paired(
        trained, twin, "greater", 0.1, np.random.default_rng(2), draws[0], draws[1][:, ::-1]
    )
    assert unpaired["lo"] < 0 < unpaired["hi"]  # draw b against another resample: spans 0


def test_no_margin_covers_zero_and_does_not_pass() -> None:
    """Margins scattered around 0: the CI holds 0 and seeds disagree in sign, so the verdict is
    fail; with the data's noise in the margin too wide to resolve a small margin that every seed
    shows, the CI holds 0 and is wider than the tolerance band, so it is inconclusive
    (fss.excludes)."""
    twin = np.random.default_rng(3).normal(1.0, 0.3, SEEDS)
    trained = twin + np.array([0.2, -0.15, 0.1, -0.25, 0.05])
    row = margins.paired(trained, twin, "greater", 0.05, np.random.default_rng(4))
    assert row["lo"] < 0 < row["hi"] and row["n_draws"] == 0
    assert row["sign_test"] == fss.FAIL and row["verdict"] == fss.FAIL
    small = twin + 0.02
    rng = np.random.default_rng(5)
    draws = _draws(small, twin, rng, apart=1.0)
    row = margins.paired(small, twin, "greater", 0.05, rng, *draws)
    assert row["lo"] < 0 < row["hi"] and (row["hi"] - row["lo"]) / 2 > 0.05
    assert row["sign_test"] == fss.PASS and row["verdict"] == fss.INCONCLUSIVE
    precise = margins.paired(trained, twin, "greater", 1.0, np.random.default_rng(4))
    assert precise["verdict"] == fss.FAIL  # a CI inside the band: precisely no margin


def test_one_seed_of_the_wrong_sign_fails_the_sign_test() -> None:
    """The mean margin is clearly positive and its CI excludes 0, but one seed lost to its twin:
    the sign test fails, and with it the verdict."""
    twin = np.random.default_rng(6).normal(1.0, 0.3, SEEDS)
    trained = twin + np.array([0.5, 0.6, 0.55, 0.45, -0.02])
    row = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(7))
    assert row["lo"] > 0
    assert row["sign_test"] == fss.FAIL and row["verdict"] == fss.FAIL


def test_the_direction_is_the_predicted_sign() -> None:
    """A margin predicted to be negative passes as "less"; read as "greater", its CI lies wholly
    on the wrong side of 0 and fails, however precise. A CI that excludes 0 passes only on the
    predicted side: here every seed's point margin is barely positive, so the sign test holds,
    but the data draws put the whole CI below 0, and that fails too (fss.excludes alone would
    pass it)."""
    twin = np.random.default_rng(8).normal(1.0, 0.3, SEEDS)
    trained = twin - 0.4 + np.random.default_rng(9).normal(0.0, 0.05, SEEDS)
    less = margins.paired(trained, twin, "less", 0.1, np.random.default_rng(10))
    assert less["hi"] < 0 and less["sign_test"] == fss.PASS and less["verdict"] == fss.PASS
    greater = margins.paired(trained, twin, "greater", 1.0, np.random.default_rng(10))
    assert greater["sign_test"] == fss.FAIL and greater["verdict"] == fss.FAIL
    barely = twin + 0.01
    rng = np.random.default_rng(11)
    shifted = barely[:, None] - 0.5 + rng.normal(0.0, 0.05, (SEEDS, DRAWS))
    flat = np.repeat(twin[:, None], DRAWS, axis=1)
    against = margins.paired(barely, twin, "greater", 0.1, rng, shifted, flat)
    assert against["hi"] < 0 and against["sign_test"] == fss.PASS
    assert against["verdict"] == fss.FAIL
    with pytest.raises(ValueError, match="direction"):
        margins.paired(trained, twin, "up", 0.1, np.random.default_rng(10))


def test_the_nested_bootstrap_shares_one_data_resample_across_its_seeds() -> None:
    """Every seed's margin is exactly 1, so the seeds alone give a CI of zero width. One shared
    data resample moves all five margins by N(0, 1), and so their mean by as much: the 95%
    half-width is 1.96. A data resample drawn per resampled seed, as the first version did,
    averages that shared noise down to 1.96 / sqrt(5) = 0.88; ignoring the draws, to nothing."""
    twin = np.random.default_rng(11).normal(1.0, 0.3, SEEDS)
    trained = twin + 1.0
    alone = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(12))
    assert alone["lo"] == pytest.approx(1.0) and alone["hi"] == pytest.approx(1.0)
    rng = np.random.default_rng(13)
    draws = _draws(trained, twin, rng, apart=1.0, columns=4_000)
    nested = margins.paired(trained, twin, "greater", 0.1, rng, *draws, n=4_000)
    assert (nested["hi"] - nested["lo"]) / 2 == pytest.approx(1.96, rel=0.1)


def test_the_ci_covers_the_true_margin_at_its_level_under_shared_data_noise() -> None:
    """400 planted experiments: five model seeds whose true margins scatter by 0.1 around 0.5,
    all scored on one dataset whose noise moves every seed's margin alike (sd 1), resampled 400
    times. The 95% CI covers 0.5 in about 94% of them; a data resample drawn per resampled seed,
    as the first version did, divided the shared noise by sqrt(5) and covered about 63%. Seed
    noise alone caps the coverage near 0.83: the mean of five seeds is t-distributed with four
    degrees of freedom, inherent to the design, which is why the verdict adds the sign test."""
    rng = np.random.default_rng(14)
    trials, covered = 400, 0
    for _ in range(trials):
        twin = rng.normal(1.0, 0.3, SEEDS)
        observed = 0.5 + rng.normal(0.0, 0.1, SEEDS) + rng.normal(0.0, 1.0)  # seeds, then data
        trained = twin + observed
        draws = _draws(trained, twin, rng, apart=1.0, own=0.05)
        row = margins.paired(trained, twin, "greater", 0.1, rng, *draws)
        covered += row["lo"] <= 0.5 <= row["hi"]
    assert 0.9 <= covered / trials <= 0.99, covered / trials


@pytest.mark.parametrize(
    ("trained_draws", "twin_draws", "match"),
    [
        (np.zeros((SEEDS, DRAWS)), np.zeros((SEEDS, DRAWS - 1)), "aligned"),  # draw counts differ
        (np.zeros((SEEDS - 1, DRAWS)), np.zeros((SEEDS - 1, DRAWS)), "aligned"),  # a seed short
        (np.zeros(SEEDS), np.zeros(SEEDS), "aligned"),  # not (seeds, draws)
        (np.zeros((SEEDS, 0)), np.zeros((SEEDS, 0)), "aligned"),  # no draws
        (np.zeros((SEEDS, DRAWS)), None, "pairs"),  # the twin's draws missing
        (None, np.zeros((SEEDS, DRAWS)), "pairs"),
    ],
)
def test_misaligned_draws_are_refused(
    trained_draws: np.ndarray | None, twin_draws: np.ndarray | None, match: str
) -> None:
    trained, twin = np.ones(SEEDS), np.zeros(SEEDS)
    with pytest.raises(ValueError, match=match):
        margins.paired(
            trained, twin, "greater", 0.1, np.random.default_rng(0), trained_draws, twin_draws
        )


@pytest.mark.parametrize(
    ("trained", "twin"),
    [
        (np.ones(SEEDS), np.zeros(SEEDS - 1)),  # seeds unpaired
        (np.ones((SEEDS, 2)), np.zeros((SEEDS, 2))),  # not one value per seed
        (np.ones(1), np.zeros(1)),  # one seed: nothing to bootstrap over
    ],
)
def test_unpaired_seeds_are_refused(trained: np.ndarray, twin: np.ndarray) -> None:
    with pytest.raises(ValueError, match="one value per model seed"):
        margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(0))


def test_an_undefined_margin_is_inconclusive() -> None:
    """A seed whose quantity is undefined leaves the mean undefined: the CI is NaN, never
    computed from the resamples that happened to skip that seed, and nothing passes."""
    trained = np.array([1.5, 1.6, np.nan, 1.4, 1.55])
    twin = np.ones(SEEDS)
    row = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(0))
    assert np.isnan(row["lo"]) and np.isnan(row["hi"]) and np.isnan(row["margin"])
    assert row["sign_test"] == fss.INCONCLUSIVE and row["verdict"] == fss.INCONCLUSIVE
    trained[0] = 0.5  # a finite seed of the wrong sign still fails the sign test
    assert margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(0))["verdict"] == (
        fss.FAIL
    )


# ---------------------------------------------------------------- the table


def _doc(primaries: int = SEEDS, ablations: int = 0, sizes: str = "mean") -> dict[str, Any]:
    """What margins.table reads of a latent file."""
    entries = [("primary", s, f"p{s}") for s in range(primaries)]
    entries += [("held-out", 100 + s, f"h{s}") for s in range(ablations)]
    return {
        "scoring": {
            "bootstrap": 400,
            "ci_level": 0.95,
            "margins": {"tolerance": 0.01, "sizes": sizes},
        },
        "latents": {
            "steps": 50,
            "checkpoints": [
                {"role": role, "model_seed": seed, "training": name} for role, seed, name in entries
            ],
        },
        "hypotheses": [{"name": "learned-beyond-random", "direction": "greater"}],
        "fss": {
            "dynamic": {
                "statistics": [
                    {"observable": "pc_1", "indicator": "variance", "role": "pipeline-check"},
                    {"observable": "surprise", "indicator": "variance", "role": "test"},
                    {"observable": "surprise", "indicator": "slope", "role": "test"},
                ]
            }
        },
    }


def _names(doc: dict[str, Any], role: str = "primary") -> tuple[list[str], list[str]]:
    entries = [e["training"] for e in doc["latents"]["checkpoints"] if e["role"] == role]
    return [latents_name(e, 50) for e in entries], [latents_name(e, 0) for e in entries]


def _verdicts(doc: dict[str, Any], slope: tuple[int, int, int], variance: tuple[int, int, int]):
    """FSS rows: how many trained checkpoints pass and are inconclusive, how many twins pass, per
    surrogate statistic; pc_1's always passes, and must not count for anything here."""
    rows = []
    for role in ("primary", "held-out"):
        trained, twin = _names(doc, role)
        for statistic, (passes, unclear, twins) in (("slope", slope), ("variance", variance)):
            verdicts = (
                [fss.PASS] * passes + [fss.INCONCLUSIVE] * unclear + [fss.FAIL] * len(trained)
            )
            quantity = f"inverse_nu_surprise_{statistic}"
            rows += [
                {"checkpoint": n, "quantity": quantity, "verdict": v}
                for n, v in zip(trained, verdicts, strict=False)
            ]
            twin_verdicts = [fss.PASS] * twins + [fss.FAIL] * len(twin)
            rows += [
                {"checkpoint": n, "quantity": quantity, "verdict": v}
                for n, v in zip(twin, twin_verdicts, strict=False)
            ]
        rows += [
            {"checkpoint": n, "quantity": "inverse_nu_pc_1_variance", "verdict": fss.PASS}
            for n in [*trained, *twin]
        ]
    return rows


def _quantity(
    doc: dict[str, Any],
    name: str,
    margin: np.ndarray,
    role: str = "primary",
    scope: str = margins.ALL,
    noise: float = 0.0,
) -> margins.Quantity:
    """Q per checkpoint, each trained checkpoint `margin` over its twin, with data draws."""
    rng = np.random.default_rng(len(name))
    trained, twin = _names(doc, role)
    level = rng.normal(0.3, 0.1, len(trained))
    values = dict(zip(trained, level + margin, strict=True)) | dict(zip(twin, level, strict=True))
    shared = rng.normal(0.0, noise, 200)
    draws = {
        n: values[n] + (shared if n in trained else 0.0) + rng.normal(0.0, 0.001, 200)
        for n in values
    }
    return margins.Quantity(name, scope, values, draws)


def _judged(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["quantity"]: r for r in rows if r["judged"]}


def test_the_table_judges_the_energy_probe_and_the_surprise_verdicts_and_nothing_else() -> None:
    """A planted positive margin passes on all three; the same large margin on anything else, on one
    L, or read off AUC and the chi alignment has no verdict, whatever it carries."""
    doc = _doc()
    gain = np.array([0.3, 0.25, 0.35, 0.28, 0.32])
    quantities = [
        _quantity(doc, "energy_r2", gain),
        _quantity(doc, "energy_selectivity", gain),
        _quantity(doc, "energy_r2", gain, scope="L=16"),
        _quantity(doc, "l2var_r2", gain),
        _quantity(doc, "abs_m_selectivity", gain),
        _quantity(doc, "auc_pc_1_variance", gain),
        _quantity(doc, "chi_alignment", gain),
    ]
    rows = margins.table(doc, quantities, _verdicts(doc, (5, 0, 0), (0, 0, 0)), _rng())
    judged = _judged(rows)
    assert set(judged) == {"energy_r2", "energy_selectivity", "surprise_fss"}
    assert all(row["verdict"] == fss.PASS and row["role"] == "primary" for row in judged.values())
    for row in rows:
        assert row["judged"] or row["verdict"] == "", row["quantity"]
        assert row["checkpoint"].split(";") == _names(doc)[0]
        assert row["twin"].split(";") == _names(doc)[1]
    assert judged["energy_r2"]["margins"] == pytest.approx(gain.tolist())
    assert judged["energy_r2"]["n_draws"] == 200 and judged["energy_r2"]["lo"] > 0


def test_a_planted_null_passes_nothing() -> None:
    """No margin at all, no statistic that passes: nothing is judged a pass."""
    doc = _doc()
    none = np.array([0.002, -0.001, 0.0015, -0.002, 0.001])
    rows = margins.table(
        doc,
        [_quantity(doc, "energy_r2", none), _quantity(doc, "energy_selectivity", none)],
        _verdicts(doc, (0, 0, 0), (0, 0, 0)),
        _rng(),
    )
    assert all(row["verdict"] != fss.PASS for row in _judged(rows).values())


def test_one_seed_against_the_sign_fails_a_judged_row() -> None:
    """Four of five model seeds gain and the mean is clearly positive; the rule asks all five."""
    doc = _doc()
    rows = margins.table(
        doc,
        [_quantity(doc, "energy_r2", np.array([0.3, 0.3, 0.3, 0.3, -0.01]))],
        _verdicts(doc, (5, 0, 0), (0, 0, 0)),
        _rng(),
    )
    row = _judged(rows)["energy_r2"]
    assert row["sign_test"] == fss.FAIL and row["verdict"] == fss.FAIL


def test_a_margin_the_data_cannot_resolve_is_inconclusive_through_the_draws() -> None:
    """Every seed's margin carries the sign, but the data's noise, shared across seeds through the
    draws, is far wider: the row reads the draws, so its CI holds 0, and no pass."""
    doc = _doc()
    gain = np.full(SEEDS, 0.02)
    rows = margins.table(
        doc,
        [_quantity(doc, "energy_r2", gain, noise=0.2)],
        _verdicts(doc, (5, 0, 0), (0, 0, 0)),
        _rng(),
    )
    row = _judged(rows)["energy_r2"]
    assert row["sign_test"] == fss.PASS and row["lo"] < 0 < row["hi"]
    assert row["verdict"] == fss.INCONCLUSIVE


@pytest.mark.parametrize(
    ("slope", "variance", "expected"),
    [
        ((5, 0, 0), (0, 0, 0), fss.PASS),
        ((4, 0, 0), (0, 0, 0), fss.PASS),  # at least 4 of 5
        ((3, 0, 0), (0, 0, 0), fss.FAIL),
        ((3, 1, 0), (0, 0, 0), fss.INCONCLUSIVE),  # the fourth could still come
        ((2, 1, 0), (0, 0, 0), fss.FAIL),
        ((5, 0, 1), (0, 0, 0), fss.FAIL),  # and for none of their twins
        ((5, 0, 1), (4, 0, 0), fss.PASS),  # at least one statistic
        ((3, 0, 0), (0, 0, 2), fss.FAIL),
        ((3, 1, 0), (5, 0, 0), fss.PASS),
    ],
)
def test_the_surprise_rule_of_the_hypothesis(
    slope: tuple[int, int, int], variance: tuple[int, int, int], expected: str
) -> None:
    """(c) as the file states it, not the margin rule's all-five sign test: an inconclusive twin is
    no pass, and the row counts what decided it. Only surprise statistics of role test count."""
    doc = _doc()
    rows = margins.table(doc, [], _verdicts(doc, slope, variance), _rng())
    row = _judged(rows)["surprise_fss"]
    assert row["verdict"] == expected
    each = {r["quantity"]: r for r in rows if not r["judged"]}
    assert set(each) == {"surprise_fss_surprise_slope", "surprise_fss_surprise_variance"}
    assert each["surprise_fss_surprise_slope"]["trained_passes"] == slope[0]
    assert each["surprise_fss_surprise_slope"]["twin_passes"] == slope[2]
    assert all(r["verdict"] == "" for r in each.values())


def test_an_ablation_reports_its_margin_and_its_ci_without_a_verdict() -> None:
    """Three seeds, a margin that would pass: Delta and its CI, no verdict, and no surprise row
    judged for its role. Each row reads its own role's checkpoints."""
    doc = _doc(ablations=3)
    gain = np.array([0.3, 0.3, 0.3])
    quantities = [
        margins.Quantity(
            "energy_r2",
            margins.ALL,
            {
                **_quantity(doc, "energy_r2", np.full(SEEDS, 0.3)).values,
                **_quantity(doc, "energy_r2", gain, role="held-out").values,
            },
        )
    ]
    rows = margins.table(doc, quantities, _verdicts(doc, (5, 0, 0), (0, 0, 0)), _rng())
    ablation = [r for r in rows if r["role"] == "held-out"]
    assert {r["quantity"] for r in ablation} == {
        "energy_r2",
        "surprise_fss_surprise_slope",
        "surprise_fss_surprise_variance",
    }
    assert all(not r["judged"] and r["verdict"] == "" for r in ablation)
    row = next(r for r in ablation if r["quantity"] == "energy_r2")
    assert row["checkpoint"].split(";") == _names(doc, "held-out")[0] and row["n_seeds"] == 3
    assert row["margin"] == pytest.approx(0.3) and np.isfinite([row["lo"], row["hi"]]).all()
    assert _judged(rows)["energy_r2"]["verdict"] == fss.PASS  # the primaries' are still judged


def test_a_role_with_one_checkpoint_has_no_rows() -> None:
    """A margin needs a spread between seeds to bootstrap over."""
    doc = _doc(primaries=1)
    assert margins.table(doc, [], _verdicts(doc, (1, 0, 0), (0, 0, 0)), _rng()) == []


def test_per_size_margins_join_the_sign_test_and_nothing_else() -> None:
    """paired's per_size: the margins at each L, seed by seed. One of the wrong sign fails the sign
    test, and with it the verdict, though every seed's own margin carries the sign; the CI, read
    off the seeds' margins and the rng alone, is the same. A NaN among them is inconclusive."""
    twin = np.random.default_rng(15).normal(1.0, 0.3, SEEDS)
    trained = twin + np.array([0.5, 0.6, 0.55, 0.45, 0.5])
    good = np.full((3, SEEDS), 0.4)
    alone = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(16))
    sized = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(16), per_size=good)
    assert alone == sized and sized["verdict"] == fss.PASS
    bad = good.copy()
    bad[1, 3] = -0.01
    wrong = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(16), per_size=bad)
    assert wrong["sign_test"] == fss.FAIL and wrong["verdict"] == fss.FAIL
    assert (wrong["lo"], wrong["hi"], wrong["margins"]) == (
        alone["lo"],
        alone["hi"],
        alone["margins"],
    )
    bad[1, 3] = np.nan
    unknown = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(16), per_size=bad)
    assert unknown["sign_test"] == fss.INCONCLUSIVE and unknown["verdict"] == fss.INCONCLUSIVE
    turned = margins.paired(trained, twin, "less", 0.1, np.random.default_rng(16), per_size=-good)
    assert turned["sign_test"] == fss.FAIL  # the seeds' own margins are against "less"


def _over_sizes(
    doc: dict[str, Any], name: str, per_L: dict[str, np.ndarray]
) -> list[margins.Quantity]:
    """Q at each L, each trained checkpoint per_L[scope] over its twin, and the mean over L of
    values and draws alike, as the runner's probe_quantities builds them."""
    each = [_quantity(doc, name, margin, scope=scope) for scope, margin in per_L.items()]
    names = list(each[0].values)
    mean = {c: float(np.mean([q.values[c] for q in each])) for c in names}
    draws = {c: np.mean([q.draws[c] for q in each], axis=0) for c in names}
    return [*each, margins.Quantity(name, margins.ALL, mean, draws)]


def test_under_every_size_one_wrong_signed_margin_at_one_L_fails_the_judged_row() -> None:
    """scoring.margins.sizes = "every" (#28): a judged row's value and CI stay the mean over L, and
    its sign test asks all 5 seeds at all 3 sizes. Every seed's mean margin is clearly positive,
    but the fifth seed loses to its twin at L = 32: under "mean" the row passes, under "every" it
    fails, with the same margins and CI. Each L's rows are reported as before, and a descriptive
    quantity's row keeps the mean's sign test: the rule is the judged rows'."""
    gain = np.array([0.3, 0.25, 0.35, 0.28, 0.32])
    lost = gain.copy()
    lost[4] = -0.05
    per_L = {"L=16": gain, "L=32": lost, "L=64": gain}
    rows = {}
    for sizes in ("mean", "every"):
        doc = _doc(sizes=sizes)
        quantities = [
            *_over_sizes(doc, "energy_r2", per_L),
            *_over_sizes(doc, "energy_selectivity", dict.fromkeys(per_L, gain)),
            *_over_sizes(doc, "l2var_r2", per_L),
        ]
        rows[sizes] = margins.table(doc, quantities, _verdicts(doc, (5, 0, 0), (0, 0, 0)), _rng())
    mean, every = _judged(rows["mean"])["energy_r2"], _judged(rows["every"])["energy_r2"]
    assert all(m > 0.15 for m in mean["margins"]) and mean["lo"] > 0, mean
    assert mean["sign_test"] == fss.PASS and mean["verdict"] == fss.PASS
    assert every["sign_test"] == fss.FAIL and every["verdict"] == fss.FAIL
    same = ("margins", "margin", "lo", "hi", "n_seeds", "n_draws", "scope")
    assert [every[k] for k in same] == [mean[k] for k in same]
    assert _judged(rows["every"])["energy_selectivity"]["verdict"] == fss.PASS
    assert _judged(rows["every"])["surprise_fss"]["verdict"] == fss.PASS

    def reported(found: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        """Every row but the judged ones, as the columns a reader sees."""
        keys = ("quantity", "scope", "margins", "lo", "hi", "sign_test", "verdict")
        return [tuple(r[k] for k in keys) for r in found if not r["judged"]]

    assert reported(rows["every"]) == reported(rows["mean"])
    at_32 = [r for r in rows["every"] if (r["quantity"], r["scope"]) == ("energy_r2", "L=32")]
    assert [(r["sign_test"], r["verdict"]) for r in at_32] == [(fss.FAIL, "")]
    descriptive = [r for r in rows["every"] if (r["quantity"], r["scope"]) == ("l2var_r2", "all")]
    assert [(r["sign_test"], r["verdict"]) for r in descriptive] == [(fss.PASS, "")]


def test_a_file_with_no_test_statistic_has_no_surprise_row() -> None:
    """Every statistic descriptive or a pipeline check (#28): the hypothesis's surprise rule has
    nothing to count, so the table has no surprise_fss row, judged or per statistic, whatever the
    FSS rows say or if there are none, and raises nothing. The energy rows are judged as before."""
    doc = _doc()
    for statistic in doc["fss"]["dynamic"]["statistics"]:
        if statistic["role"] == "test":
            statistic["role"] = "descriptive"
    gain = np.array([0.3, 0.25, 0.35, 0.28, 0.32])
    quantities = [_quantity(doc, "energy_r2", gain), _quantity(doc, "energy_selectivity", gain)]
    for fss_rows in (_verdicts(doc, (5, 0, 0), (0, 0, 0)), []):
        rows = margins.table(doc, quantities, fss_rows, _rng())
        assert not [r for r in rows if r["quantity"].startswith(margins.SURPRISE_FSS)], rows
        assert set(_judged(rows)) == {"energy_r2", "energy_selectivity"}
        assert all(r["verdict"] == fss.PASS for r in _judged(rows).values())


def test_the_table_reads_the_predicted_sign_from_the_file() -> None:
    """A file that predicts the margin falls judges it that way, draw for draw."""
    doc = _doc()
    doc["hypotheses"][0]["direction"] = "less"
    loss = np.array([-0.3, -0.25, -0.35, -0.28, -0.32])
    rows = margins.table(
        doc, [_quantity(doc, "energy_r2", loss)], _verdicts(doc, (5, 0, 0), (0, 0, 0)), _rng()
    )
    assert _judged(rows)["energy_r2"]["verdict"] == fss.PASS
    assert _judged(rows)["energy_r2"]["direction"] == "less"
