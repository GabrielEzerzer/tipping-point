# The sequence models on a tiny encoder and synthetic panels, on the CPU: timing, the JEPA target's
# hours, the folds' rows and seeds, standardisation, the twin, the step selection, fold isolation,
# unread labels, the scores' checks, skipped folds, the command's run and its held-back refusal.

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.ews.scoring import auc
from tipping_point.jepa.objective import momentum
from tipping_point.panel import encoder, inputs, labels, ladder, sequence, span, store
from tipping_point.panel.__main__ import main
from torch import nn

HOUR = labels.HOUR_NS
COINS = ["AAAUSDT", "BBBUSDT"]

LADDER = """
start = START
end = 2022-12-31
horizons = [24, 1]
sides = ["long", "short", "any"]
without = []

[model]
c = 0.01
clip = 5.0

[folds]
gap_days = 31
forward_first = FIRST

[bootstrap]
block_days = 14
draws = 10
seed = 1

[[rungs]]
name = "full"
groups = ["vol", "price", "flow", "lev"]
own_only = ["lev"]

[[rungs]]
name = "vol"
groups = ["vol"]
"""

PLAN = """
ladder = "ladder.toml"
variants = ["vol"]
rungs = ["full", "vol"]
models = ["jepa", "twin", "supervised"]
schemes = ["loqo", "forward"]
seeds = [3]
folds = [1, 3]
target = "future"
threads = 1

[window]
length = 16
future = 4
shifts = [2, 4]

[encoder]
patch = 2
width = 8
conv_blocks = 1
attention_blocks = 1
heads = 2
kernel = 3
expansion = 2

[predictor]
hidden = 8

[jepa]
steps = 4
batch = 16
lr = 1e-3
final_lr = 1e-4
warmup = 1
weight_decay = 0.05
curve_every = 2
momentum_start = 0.9
momentum_end = 1.0
prediction = 1.0
objective = "z"
variance = 1.0
covariance = 0.04

[supervised]
steps = 4
batch = 16
lr = 1e-3
final_lr = 1e-4
warmup = 1
weight_decay = 0.1
curve_every = 2
dropout = 0.2

[selection]
share = 0.2
every = 2
negatives = 200

[probe]
batch = 4096
"""


def ladder_text(start: str = "2022-01-01", first: int = 2) -> str:
    return LADDER.replace("START", start).replace("FIRST", str(first))


def plans(tmp_path: Path, plan: str = PLAN, base: str | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "ladder.toml").write_text(base or ladder_text(), encoding="utf-8")
    path = tmp_path / "sequence.toml"
    path.write_text(plan, encoding="utf-8")
    return path


def world(
    days: int = 365, seed: int = 0, skip: float = 0.02, censored: float = 0.01
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    hours = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(days * 24, dtype=np.int64)
    n = len(hours) * len(COINS)
    values = {name: rng.normal(size=n) for name in inputs.OWN + inputs.MARKET}
    values["ls"][rng.random(n) < 0.01] = np.nan
    grid: dict[str, np.ndarray] = {
        "ts_ns": np.repeat(hours, len(COINS)),
        "symbol": np.tile(np.array(COINS, dtype=object), len(hours)),
    }
    for h in labels.HORIZONS:
        lift = values["vol_1h"]
        grid[f"long_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 - lift))
        grid[f"short_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 + lift))
        grid[f"censored_{h}"] = rng.random(n) < censored
    grid["skip"] = rng.random(n) < skip
    return values, grid


def tiny(tmp_path: Path, plan: str = PLAN, base: str | None = None) -> sequence.Plan:
    return sequence.load(plans(tmp_path, plan, base))


def test_the_shipped_plan_is_the_scorers_plan() -> None:
    plan = sequence.load()
    assert plan.ladder_path.name == "ladder.toml" and plan.ladder == ladder.load()
    assert plan.ladder.columns == ladder.load().columns
    assert plan.rungs == ("full", "vol", "classic") and plan.schemes == ("forward",)
    assert plan.variants == ("full", "vol") and plan.models == sequence.MODELS
    assert plan.seeds == (1, 2, 3) and plan.folds == tuple(range(8)) and plan.threads == 2
    assert (plan.window, plan.future, plan.pad, plan.reach) == (336, 24, 335, 24)
    assert plan.window // plan.shape.patch == 84 and plan.future // plan.shape.patch == 6
    assert plan.target == "future" and plan.shape.width == 32
    s = plan.supervised
    assert (s.dropout, s.lr, s.weight_decay, plan.jepa.dropout) == (0.2, 3e-4, 0.1, 0.0)
    assert (plan.selection.share, plan.selection.every) == (0.2, 100)


@pytest.mark.parametrize(
    "changes",
    [
        [('variants = ["vol"]', 'variants = ["vol", "vol"]')],
        [("folds = [1, 3]", "folds = [1, 1]")],
        [('models = ["jepa", "twin", "supervised"]', 'models = ["jepa", "jepa"]')],
        [('schemes = ["loqo", "forward"]', 'schemes = ["loqo", "loqo"]')],
        [('rungs = ["full", "vol"]', 'rungs = ["vol", "vol"]')],
        [("future = 4", "future = 0")],
        [("shifts = [2, 4]", "shifts = [0, 4]")],
        [("shifts = [2, 4]", "shifts = []"), ('target = "future"', 'target = "shifted"')],
        [("batch = 16", "batch = 1")],
        [('objective = "z"', 'objective = "y"')],
    ],
)
def test_the_plan_refuses_repeats_and_empty_windows(
    tmp_path: Path, changes: list[tuple[str, str]]
) -> None:
    text = PLAN
    for old, new in changes:
        assert old in text
        text = text.replace(old, new, 1)
    with pytest.raises(ValueError, match="unknown or invalid"):
        tiny(tmp_path, text)


def test_every_token_sees_only_its_own_and_earlier_hours(tmp_path: Path) -> None:
    plan = tiny(tmp_path)
    net = encoder.encoder(6, plan.shape, 0)
    x = torch.randn(3, plan.window, 6)
    later = x.clone()
    later[:, 9:] = torch.randn_like(later[:, 9:]) * 50
    with torch.no_grad():
        before, after = net.tokens(x), net.tokens(later)
    k = 9 // plan.shape.patch
    assert torch.equal(before[:, :k], after[:, :k])
    assert not torch.allclose(before[:, k], after[:, k])


def test_nothing_after_hour_t_reaches_z_or_a_score_at_t(tmp_path: Path) -> None:
    plan = tiny(tmp_path)
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(2, 200, 5))
    mu, sd = sequence.statistics(raw, np.ones(200, bool))
    net = encoder.encoder(6, plan.shape, 0)
    supervised = sequence.Supervised(6, plan, 0)
    coin, hour = np.array([0, 1]), np.array([120, 3])

    def outputs(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(sequence.standardised(r, mu, sd, 5.0, plan.pad))
        return (
            sequence.encode(net, x, coin, hour, plan),
            sequence.encode(supervised, x, coin, hour, plan),
        )

    before = outputs(raw)
    later = raw.copy()
    later[0, 121:] = 1e6
    later[1, 4:] = rng.normal(size=later[1, 4:].shape) * 50
    later[1, 4:7] = np.nan
    after = outputs(later)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    moved = raw.copy()
    moved[0, 120] += 1.0
    assert not np.array_equal(outputs(moved)[0][0], before[0][0])


@pytest.mark.parametrize("target", sequence.TARGETS)
def test_the_jepa_target_reads_exactly_its_hours(tmp_path: Path, target: str) -> None:
    plan = tiny(tmp_path, PLAN.replace('target = "future"', f'target = "{target}"'))
    model = sequence.jepa(6, plan, 0)
    x = torch.randn(1, plan.pad + 100, 6)
    coin, hour, t = torch.tensor([0]), torch.tensor([50]), 50
    base = sequence.goal(model, x, coin, hour, plan)
    if target == "future":
        first, last = t + 1, t + plan.future
    else:
        first, last = t + min(plan.shifts) - plan.window + 1, t + max(plan.shifts)

    def moves(j: int) -> bool:
        y = x.clone()
        y[0, plan.pad + j] += 3.0
        return not torch.equal(sequence.goal(model, y, coin, hour, plan), base)

    assert [moves(j) for j in (first - 1, first, last, last + 1)] == [False, True, True, False]


def test_a_precursor_48_hours_back_is_told_from_one_200_hours_back() -> None:
    plan = sequence.load()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    bump = 3 * np.exp(-0.5 * ((np.arange(12) - 6) / 2.0) ** 2)

    def sample(n: int) -> tuple[torch.Tensor, np.ndarray]:
        x = 0.5 * rng.normal(size=(n, plan.window, 4)).astype(np.float32)
        near = rng.random(n) < 0.5
        for i, end in enumerate(np.where(near, plan.window - 48, plan.window - 200)):
            x[i, end - 12 : end, 0] += bump
        return torch.from_numpy(x), near

    net = encoder.encoder(4, plan.shape, 0)
    head = nn.Linear(plan.shape.width, 1)
    opt = torch.optim.AdamW([*net.parameters(), *head.parameters()], lr=3e-3)
    for _ in range(120):
        x, near = sample(32)
        logits = head(net(x)).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(near.astype(np.float32))
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, near = sample(400)
    with torch.no_grad():
        s = head(net(x)).squeeze(-1).numpy()
    assert auc(s[near], s[~near]) > 0.95


def test_the_folds_keep_to_the_span_and_fitted_rows_off_the_gap() -> None:
    plan = sequence.load()
    base = plan.ladder
    edges = ladder.bounds(base)
    hours = np.arange(edges[0], edges[-1], HOUR)
    wide = np.arange(edges[0] - 400 * HOUR, edges[-1] + 400 * HOUR, HOUR)
    inside = (wide >= edges[0]) & (wide < edges[-1])
    f = 2
    for scheme in ladder.SCHEMES:
        around = sequence.training(wide, plan, f, scheme)
        assert not around[~inside].any()
        assert np.array_equal(around[inside], sequence.training(hours, plan, f, scheme))
    train = sequence.training(hours, plan, f, "loqo")
    back, ahead = plan.pad, plan.reach
    a = int(np.searchsorted(hours, edges[f] - base.gap_days * 24 * HOUR))
    b = int(np.searchsorted(hours, edges[f + 1] + base.gap_days * 24 * HOUR))
    n = len(hours)
    assert train[a - 1] and not train[a] and not train[b - 1] and train[b]
    ok = sequence.eligible(train, back, ahead)
    planted = {
        a - 1 - ahead: True,
        a - ahead: False,
        b + back: True,
        b + back - 1: False,
        n - 1 - ahead: True,
        n - ahead: False,
        back: True,
        back - 1: False,
        0: False,
    }
    assert {j: bool(ok[j]) for j in planted} == planted
    ncoins = 3
    skip = np.zeros((ncoins, n), bool)
    skip[1, a - 1 - ahead] = True
    for exclude in (None, skip):
        rows = sequence.sample_rows(ok, ncoins, exclude)
        feed = sequence.batches(rows, n, 4096, np.random.SeedSequence(0))
        drawn = np.concatenate([next(feed).numpy() for _ in range(40)], axis=1)
        for coin, j in zip(*np.unique(drawn, axis=1), strict=True):
            touched = np.arange(j - back, j + ahead + 1)
            assert touched[0] >= 0 and touched[-1] < n and train[touched].all()
            assert exclude is None or not exclude[coin, j]
        seen = set(map(tuple, drawn.T.tolist()))
        assert (0, a - 1 - ahead) in seen and (2, b + back) in seen
        assert ((1, a - 1 - ahead) in seen) == (exclude is None)
        assert not {(c, j) for c in range(ncoins) for j in planted if not planted[j]} & seen


def test_the_seeds_differ_by_variant_fold_seed_and_stream() -> None:
    keys = [
        (v, f, s, name)
        for v in ("full", "vol")
        for f in (0, 1)
        for s in (1, 2)
        for name in ("init", "jepa")
    ]
    states = {tuple(sequence.stream(*k).generate_state(4)) for k in keys}
    assert len(states) == len(keys)
    assert sequence.torch_seed("vol", 1, 2, "init") == sequence.torch_seed("vol", 1, 2, "init")


def test_the_standardisation_reads_only_training_hours() -> None:
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(3, 500, 4))
    raw[0, :10, 1] = np.nan
    train = np.ones(500, bool)
    train[200:350] = False
    mu, sd = sequence.statistics(raw, train)
    changed = raw.copy()
    changed[:, ~train] = rng.normal(size=changed[:, ~train].shape) * 100 + 7
    changed[1, 210:260] = np.nan
    assert all(
        np.array_equal(u, v)
        for u, v in zip((mu, sd), sequence.statistics(changed, train), strict=True)
    )
    kept = raw[:, train].reshape(-1, 4)
    assert np.allclose(mu, np.nanmean(kept, axis=0)) and np.allclose(sd, np.nanstd(kept, axis=0))


@pytest.mark.parametrize("target", sequence.TARGETS)
def test_the_target_encoder_only_follows_the_moving_average(tmp_path: Path, target: str) -> None:
    plan = tiny(tmp_path, PLAN.replace('target = "future"', f'target = "{target}"'))
    torch.manual_seed(0)
    x = torch.randn(2, plan.pad + 100, 6)
    model = sequence.jepa(6, plan, 5)
    assert not model.target.training
    opt = sequence.optimiser([model.online, model.predictor], plan.jepa)
    rows = torch.tensor([[0, 1, 0, 1], [10, 20, 30, 40]])
    old = [p.detach().clone() for p in model.target.parameters()]
    sequence.jepa_step(model, opt, x, rows, 0, plan)
    m = momentum(0, plan.jepa.steps, plan.jepa.momentum_start, plan.jepa.momentum_end)
    assert m == plan.jepa.momentum_start < 1
    for t, o, s in zip(model.target.parameters(), old, model.online.parameters(), strict=True):
        assert not t.requires_grad and t.grad is None
        assert torch.allclose(t, m * o + (1 - m) * s.detach(), atol=1e-7)
    assert any(not torch.equal(t, o) for t, o in zip(model.target.parameters(), old, strict=True))


def test_describe_survives_constant_empty_and_single_rows() -> None:
    assert sequence.describe(np.ones((50, 4)))["z_effective_rank"] == 0.0
    for n in (0, 1):
        d = sequence.describe(np.zeros((n, 4)))
        assert d["z_effective_rank"] == 0.0 and np.isnan(d["z_sd_min"])
    rank = sequence.describe(np.random.default_rng(0).normal(size=(500, 4)))["z_effective_rank"]
    assert 3.5 < rank <= 4.0


def test_missing_inputs_give_finite_outputs_and_their_share(tmp_path: Path) -> None:
    plan = tiny(tmp_path)
    raw = np.random.default_rng(3).normal(size=(2, 40, 4))
    raw[0, 5, [0, 2]] = np.nan
    raw[1] = np.nan
    mu, sd = sequence.statistics(raw, np.ones(40, bool))
    x = sequence.standardised(raw, mu, sd, 5.0, plan.pad)
    assert x.shape == (2, plan.pad + 40, 5) and np.isfinite(x).all()
    assert (x[:, : plan.pad, :4] == 0).all() and (x[:, : plan.pad, 4] == 1).all()
    assert x[0, plan.pad + 5, 4] == 0.5 and (x[0, plan.pad + 5, [0, 2]] == 0).all()
    assert x[0, plan.pad + 6, 4] == 0 and (x[1, plan.pad :, 4] == 1).all()
    net = encoder.encoder(5, plan.shape, 0)
    z = sequence.encode(net, torch.from_numpy(x), np.array([0, 0, 1]), np.array([0, 5, 39]), plan)
    assert np.isfinite(z).all()
    future = net.tokens(torch.from_numpy(x[:, : plan.future]))
    assert future.shape == (2, plan.future // plan.shape.patch, plan.shape.width)


def test_the_step_selection_reads_no_left_out_or_gap_hour(tmp_path: Path) -> None:
    plan = tiny(tmp_path)
    rng = np.random.default_rng(4)
    ncoins, nhours = 2, 3000
    train = np.ones(nhours, bool)
    train[1000:1800] = False
    raw = rng.normal(size=(ncoins, nhours, 5))
    y = rng.random((ncoins, nhours, len(plan.outputs))) < 1 / (1 + np.exp(2 - raw[..., :1]))
    keep = rng.random((ncoins, nhours, len(plan.outputs))) > 0.1
    skip = rng.random((ncoins, nhours)) < 0.05

    def fit() -> tuple:
        mu, sd = sequence.statistics(raw, train)
        x = torch.from_numpy(sequence.standardised(raw, mu, sd, 5.0, plan.pad))
        return sequence.supervise(
            x,
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(keep.astype(np.float32)),
            skip,
            train,
            plan,
            ("vol", 1, 3),
            lambda line: None,
        )

    first = fit()
    positions = np.flatnonzero(train)
    start = positions[int(len(positions) * (1 - plan.selection.share))]
    before = np.arange(nhours) >= start - plan.ladder.gap_days * 24
    y[:, before & (np.arange(nhours) < start)] ^= True
    near = fit()
    assert json.dumps(first[2]) == json.dumps(near[2]) and first[3] == near[3]
    y[:, before & (np.arange(nhours) < start)] ^= True
    out = ~train
    raw[:, out] = rng.normal(size=raw[:, out].shape) * 100
    y[:, out], keep[:, out], skip[:, out] = ~y[:, out], ~keep[:, out], ~skip[:, out]
    second = fit()
    assert len(first[2]) == plan.supervised.steps // plan.selection.every
    assert all(np.isfinite(c["auc"]) for c in first[2])
    assert json.dumps(first[2]) == json.dumps(second[2]) and first[3] == second[3]
    assert sequence.weights(first[0]) == sequence.weights(second[0])
    assert first[3] in {c["step"] for c in first[2]}


def test_the_refit_follows_the_selection_schedule_and_stops_at_the_chosen_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tiny(tmp_path)
    plan = replace(base, supervised=replace(base.supervised, steps=8, warmup=2))
    rng = np.random.default_rng(5)
    ncoins, nhours = 2, 3000
    train = np.ones(nhours, bool)
    raw = rng.normal(size=(ncoins, nhours, 5))
    y = rng.random((ncoins, nhours, len(plan.outputs))) < 0.2
    keep = np.ones((ncoins, nhours, len(plan.outputs)), bool)
    skip = np.zeros((ncoins, nhours), bool)
    curve = {2: 0.6, 4: 0.9, 6: 0.7, 8: 0.5}

    def validation(*args: object) -> object:
        seen = []

        def check(model: object) -> dict:
            seen.append(None)
            return {"auc": curve[2 * len(seen)]}

        return check

    rates: dict[int, list[float]] = {}
    schedule = sequence.schedule

    def recorded(opt: torch.optim.Optimizer, step: int, t: sequence.Training) -> None:
        schedule(opt, step, t)
        rates.setdefault(id(opt), []).append(opt.param_groups[0]["lr"])

    monkeypatch.setattr(sequence, "validation", validation)
    monkeypatch.setattr(sequence, "schedule", recorded)
    mu, sd = sequence.statistics(raw, train)
    x = torch.from_numpy(sequence.standardised(raw, mu, sd, 5.0, plan.pad))
    _, fitted, _, steps, _ = sequence.supervise(
        x,
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(keep.astype(np.float32)),
        skip,
        train,
        plan,
        ("vol", 1, 3),
        lambda line: None,
    )
    selection, refit = rates.values()
    assert steps == 4 and len(selection) == 8 and len(refit) == 4
    assert refit == selection[:4]
    assert fitted[-1]["step"] == 3


def by_model(out: dict) -> dict:
    return {
        (d["variant"], d["scheme"], d["fold"], d["seed"], d["model"]): d for d in out["diagnostics"]
    }


def test_scores_cover_the_common_rows_and_repeat_with_the_seed(tmp_path: Path) -> None:
    values, grid = world(days=200)
    text = PLAN.replace('variants = ["vol"]', 'variants = ["full", "vol"]')
    text = text.replace("folds = [1, 3]", "folds = [2]")
    plan = tiny(tmp_path, text.replace("seeds = [3]", "seeds = [3, 4]"))
    first = sequence.sequence(values, grid, plan)
    second = sequence.sequence(values, grid, plan)
    assert first["scores"].equals(second["scores"])
    timeless = [
        json.dumps([{k: v for k, v in d.items() if k != "seconds"} for d in out["diagnostics"]])
        for out in (first, second)
    ]
    assert timeless[0] == timeless[1] and json.dumps(first["curves"]) == json.dumps(
        second["curves"]
    )
    scores = first["scores"]
    assert scores.schema.field("horizon").type == pa.int16()
    assert scores.schema.field("fold").type == pa.int16()
    edges = ladder.bounds(plan.ladder)
    fold, _ = ladder.folds(grid["ts_ns"], edges, plan.ladder.block_days)
    column = {
        k: scores[k].cast(pa.string()).to_numpy(zero_copy_only=False)
        if pa.types.is_dictionary(scores[k].type)
        else scores[k].to_numpy()
        for k in ("horizon", "model", "side", "variant", "scheme", "seed")
    }
    for h in plan.ladder.horizons:
        keep, _ = ladder.select(values, grid, plan.ladder, h)
        for scheme in plan.schemes:
            folds = [f for f in plan.folds if f >= plan.ladder.first_fold(scheme)]
            expected = keep & np.isin(fold, folds)
            want = sorted(
                zip(
                    grid["symbol"][expected].tolist(), grid["ts_ns"][expected].tolist(), strict=True
                )
            )
            for model in ("jepa", "twin", "full+z", "supervised"):
                for variant in plan.variants:
                    for seed in plan.seeds:
                        for side in plan.ladder.sides:
                            mask = (
                                (column["horizon"] == h)
                                & (column["model"] == model)
                                & (column["side"] == side)
                                & (column["variant"] == variant)
                                & (column["scheme"] == scheme)
                                & (column["seed"] == seed)
                            )
                            got = scores.filter(pa.array(mask))
                            rows = sorted(
                                zip(got["coin"].to_pylist(), got["hour"].to_pylist(), strict=True)
                            )
                            assert rows == want, (h, scheme, model, variant, seed, side)
                            assert np.isin(got["fold"].to_numpy(), folds).all()
    assert np.isfinite(scores["score"].to_numpy()).all()
    runs = {(v, s, f, seed) for v, s, f, seed, _ in by_model(first)}
    assert runs == {
        (v, s, f, seed)
        for v in plan.variants
        for s in plan.schemes
        for f in plan.folds
        if f >= plan.ladder.first_fold(s)
        for seed in plan.seeds
    }
    rungs = {(r["variant"], r["scheme"], r["fold"]) for r in first["folds"] if r["model"] == "rung"}
    assert rungs == {(n, s, 2) for n in ("full", "vol") for s in plan.schemes}


def test_the_twin_is_the_jepa_encoder_before_its_first_step(tmp_path: Path) -> None:
    values, grid = world(days=200)
    text = PLAN.replace('variants = ["vol"]', 'variants = ["full", "vol"]')
    plan = tiny(tmp_path, text.replace("folds = [1, 3]", "folds = [1, 2]"))
    found = by_model(sequence.sequence(values, grid, plan))
    initials: dict[tuple, set] = {}
    for (v, s, f, seed, model), d in found.items():
        if model != "twin":
            continue
        jepa, supervised = found[(v, s, f, seed, "jepa")], found[(v, s, f, seed, "supervised")]
        assert d["initial"] == d["weights"] == jepa["initial"] == supervised["initial"]
        assert jepa["weights"] != jepa["initial"] != supervised["weights"]
        initials.setdefault((v, f, seed), set()).add(d["initial"])
    assert all(len(v) == 1 for v in initials.values())
    assert len({next(iter(v)) for v in initials.values()}) == len(initials) == 4


def perturbed(values: dict, grid: dict, start: int, end: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(9)
    region = (grid["ts_ns"] >= start) & (grid["ts_ns"] < end)
    moved = {}
    for name, v in values.items():
        noise = rng.normal(size=v.shape) * 10
        moved[name] = np.where(region & np.isfinite(v), v + noise, v)
    flipped = dict(grid)
    for h in labels.HORIZONS:
        for side in ("long", "short"):
            flipped[f"{side}_{h}"] = np.where(region, ~grid[f"{side}_{h}"], grid[f"{side}_{h}"])
    return moved, flipped


def fits(out: dict) -> dict:
    return {
        k: (d["initial"], d["weights"], d["fitted"], d["steps"]) for k, d in by_model(out).items()
    }


def test_the_left_out_quarter_and_its_gaps_reach_no_fitted_parameter(tmp_path: Path) -> None:
    values, grid = world()
    plan = tiny(tmp_path, PLAN.replace("folds = [1, 3]", "folds = [2]"))
    edges = ladder.bounds(plan.ladder)
    gap = plan.ladder.gap_days * 24 * HOUR
    first = sequence.sequence(values, grid, plan)
    moved, flipped = perturbed(values, grid, edges[2] - gap, edges[3] + gap)
    second = sequence.sequence(moved, flipped, plan)
    assert {k[1] for k in fits(first)} == {"loqo", "forward"}
    assert fits(first) == fits(second)
    assert json.dumps(first["curves"]) == json.dumps(second["curves"])
    assert not first["scores"]["score"].equals(second["scores"]["score"])


def test_labels_of_skipped_rows_and_censored_horizons_are_never_read(tmp_path: Path) -> None:
    values, grid = world(skip=0.3, censored=0.3)
    plan = tiny(tmp_path, PLAN.replace("folds = [1, 3]", "folds = [2]"))
    first = sequence.sequence(values, grid, plan)
    flipped = dict(grid)
    for h in labels.HORIZONS:
        hidden = grid["skip"] | grid[f"censored_{h}"]
        for side in ("long", "short"):
            flipped[f"{side}_{h}"] = np.where(hidden, ~grid[f"{side}_{h}"], grid[f"{side}_{h}"])
    second = sequence.sequence(values, flipped, plan)
    assert fits(first) == fits(second)
    assert json.dumps(first["curves"]) == json.dumps(second["curves"])
    assert first["scores"].equals(second["scores"])


def test_a_fold_without_training_or_probe_rows_is_skipped(tmp_path: Path) -> None:
    values, grid = world(days=200)
    base = ladder_text("2022-02-20", first=1)
    text = PLAN.replace('schemes = ["loqo", "forward"]', 'schemes = ["forward"]')
    text = text.replace("folds = [1, 3]", "folds = [1]")
    long = tiny(tmp_path / "a", text.replace("length = 16", "length = 200"), base)
    out = sequence.sequence(values, grid, long)
    assert out["skipped"] == ["vol forward fold 2: no training row"]
    assert out["scores"].num_rows == 0 and out["diagnostics"] == []
    assert {r["model"] for r in out["folds"]} == {"rung"}
    plan = tiny(tmp_path / "b", text, base)
    start = labels.ns(date(2022, 2, 20)) + plan.pad * HOUR
    censored = dict(grid)
    censored["censored_24"] = grid["censored_24"] | (
        (grid["ts_ns"] >= start) & (grid["ts_ns"] < labels.ns(date(2022, 3, 1)))
    )
    out = sequence.sequence(values, censored, plan)
    assert out["skipped"] == ["vol forward fold 2 24 h: no probe row"]
    assert set(out["scores"]["horizon"].to_pylist()) == {1}


def write_runs(data: Path, values: dict, grid: dict, end: str = "2022-12-31") -> Path:
    labels_dir = data / "labels" / "20260101T000000Z-t"
    (labels_dir / "grid").mkdir(parents=True)
    grid_file = labels_dir / "grid" / "0.parquet"
    pq.write_table(pa.table(grid), grid_file)
    sha = store.sha256(grid_file)
    table = pa.table(
        {"ts_ns": grid["ts_ns"], "symbol": grid["symbol"].astype(str), **values},
        schema=inputs.SCHEMA,
    )
    params = {
        "labels_run": labels_dir.name,
        "labels_grid": {"file": "grid/0.parquet", "sha256": sha},
        "start": "2021-10-01",
        "dev_start": "2022-01-01",
        "end": end,
    }
    return store.write(data, "panel-inputs", "inputs", {"inputs": table}, params, {})


def test_the_command_writes_a_sequence_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values, grid = world(days=200)
    run_inputs = write_runs(tmp_path, values, grid)
    plan = plans(tmp_path, PLAN.replace("folds = [1, 3]", "folds = [1]"))
    assert (
        main(
            [
                "sequence",
                "--inputs",
                run_inputs.name,
                "--plan",
                str(plan),
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    (run,) = (tmp_path / "panel-sequence").glob("*-sequence")
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    assert set(complete["files"]) == {
        f"{n}.parquet" for n in ("scores", "diagnostics", "curves", "folds", "rows")
    } | {"report.md", "plan.toml", "ladder.toml"}
    params = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["params"]
    assert params["inputs_run"] == run_inputs.name
    assert params["inputs"]["sha256"] == store.sha256(run_inputs / "inputs.parquet")
    assert params["labels_grid"]["sha256"] == store.sha256(
        tmp_path / "labels" / params["labels_run"] / "grid" / "0.parquet"
    )
    assert params["plan"]["sha256"] == store.sha256(plan)
    assert params["ladder_plan"]["sha256"] == store.sha256(tmp_path / "ladder.toml")
    scores = pq.read_table(run / "scores.parquet")
    assert set(scores.column_names) >= {"coin", "hour", "horizon", "side", "score", "model"}
    assert "| jepa | vol | loqo | 3 | long |" in (run / "report.md").read_text(encoding="utf-8")
    assert set(scores.column_names) >= {"scheme", "fold", "variant", "seed"}
    assert str(run) in capsys.readouterr().out


def test_the_command_refuses_the_held_back_span(tmp_path: Path) -> None:
    late = ladder_text().replace("end = 2022-12-31", "end = 2024-01-01")
    with pytest.raises(span.HeldBackError):
        sequence.load(plans(tmp_path / "p", base=late))
    values, grid = world(days=10)
    plan = plans(tmp_path)
    early = write_runs(tmp_path / "a", values, grid)
    assert (
        main(["sequence", "--inputs", str(early), "--plan", str(tmp_path / "p" / "sequence.toml")])
        == 1
    )
    reaching = write_runs(tmp_path / "b", values, grid, end="2024-01-01")
    (tmp_path / "b" / "labels").rename(tmp_path / "b" / "unread")
    with pytest.raises(span.HeldBackError):
        sequence.run(reaching, tmp_path / "b", plan, "x")
    shift = labels.ns(date(2024, 1, 1)) - labels.ns(date(2022, 1, 1)) - 5 * 24 * HOUR
    moved = write_runs(tmp_path / "c", values, dict(grid, ts_ns=grid["ts_ns"] + shift))
    with pytest.raises(span.HeldBackError):
        sequence.run(moved, tmp_path / "c", plan, "x")
    for folder in ("a", "b", "c"):
        assert not (tmp_path / folder / "panel-sequence").exists()
