"""Paired margins of a trained checkpoint over its twin, on planted per-seed quantities: the
nested bootstrap's CI, the sign test, the verdicts, and the refusal of misaligned draws."""

import numpy as np
import pytest
from tipping_point.ews import fss, margins

SEEDS, DRAWS = 5, 400


def _paired_draws(
    trained: np.ndarray, twin: np.ndarray, shared: float, own: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Data-bootstrap draws around each seed's point: one resample moves the trained checkpoint
    and its twin alike (`shared`), and each a little on its own (`own`)."""
    common = rng.normal(0.0, shared, (SEEDS, DRAWS))
    return (
        trained[:, None] + common + rng.normal(0.0, own, (SEEDS, DRAWS)),
        twin[:, None] + common + rng.normal(0.0, own, (SEEDS, DRAWS)),
    )


def test_a_planted_margin_passes() -> None:
    """Every seed gains 0.5 over its twin, though the seeds' levels differ by more than that and
    each resample moves both checkpoints by twice as much again. Paired draw by draw, the shared
    part cancels and the CI excludes 0; drawn unpaired, it would span 0."""
    rng = np.random.default_rng(1)
    twin = rng.normal(1.0, 0.6, SEEDS)
    trained = twin + 0.5 + rng.normal(0.0, 0.05, SEEDS)
    draws = _paired_draws(trained, twin, shared=1.0, own=0.05, rng=rng)
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
    fail; with draws too wide to resolve a small margin that every seed shows, the CI holds 0 and
    is wider than the tolerance band, so it is inconclusive (fss.excludes)."""
    twin = np.random.default_rng(3).normal(1.0, 0.3, SEEDS)
    trained = twin + np.array([0.2, -0.15, 0.1, -0.25, 0.05])
    row = margins.paired(trained, twin, "greater", 0.05, np.random.default_rng(4))
    assert row["lo"] < 0 < row["hi"] and row["n_draws"] == 0
    assert row["sign_test"] == fss.FAIL and row["verdict"] == fss.FAIL
    small = twin + 0.02
    rng = np.random.default_rng(5)
    draws = _paired_draws(small, twin, shared=0.0, own=1.0, rng=rng)
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


def test_the_nested_bootstrap_takes_one_data_draw_per_resampled_seed() -> None:
    """Every seed has margin exactly 1, so seeds alone give a CI of zero width; data draws with
    sd 1 around it make the mean of five resampled seeds' draws spread with sd 1/sqrt(5), a 95%
    half-width of 1.96/sqrt(5) = 0.88. Averaging each seed's draws instead would shrink it by
    sqrt(DRAWS) = 20; ignoring them, to nothing."""
    twin = np.random.default_rng(11).normal(1.0, 0.3, SEEDS)
    trained = twin + 1.0
    alone = margins.paired(trained, twin, "greater", 0.1, np.random.default_rng(12))
    assert alone["lo"] == pytest.approx(1.0) and alone["hi"] == pytest.approx(1.0)
    rng = np.random.default_rng(13)
    draws = _paired_draws(trained, twin, shared=0.0, own=np.sqrt(0.5), rng=rng)  # sd 1 apart
    nested = margins.paired(trained, twin, "greater", 0.1, rng, *draws, n=4_000)
    half = (nested["hi"] - nested["lo"]) / 2
    assert half == pytest.approx(1.96 / np.sqrt(SEEDS), rel=0.1)


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
