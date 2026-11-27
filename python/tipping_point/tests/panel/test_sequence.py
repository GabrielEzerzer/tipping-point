# The sequence models on a tiny encoder and synthetic panels, on the CPU: timing, the JEPA target's
# hours, the folds' rows and seeds, the readouts, the twin, the step selection, fold isolation,
# saved models, the selection rule, the scores the ladder reads, skipped folds and held-back data.

import json
import os
import shutil
from collections.abc import Callable
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
length = 36
future = 4
shifts = [2, 4]

[fresh]
changes = ["oi_vs_28d", "funding", "ls"]

[fresh.windowed]
vol_1h = 1
ret_1h = 1
flow_1h = 1
vsurge_1h = 1
vol_6h = 3
ret_6h = 3
vol_24h = 24
near_24h = 24
volume_24h = 24
ret_24h = 24
aret_24h = 24
flow_24h = 24
oi_24h = 24

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
readouts = ["last", "means", "spreads"]
c = [0.01, 1.0]
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


FRESH = PLAN.replace('target = "future"', 'target = "fresh"')
NAMES = ("vol_1h", "vol_6h", "vol_7d", "ls", "m_vol_6h", "m_ls")
VOL_1H, VOL_6H, VOL_7D, LS, M_VOL_6H, M_LS, MISSING = range(len(NAMES) + 1)


def fresh_world(
    plan: sequence.Plan, raw: np.ndarray | None = None, train: np.ndarray | None = None
) -> tuple[np.ndarray, torch.Tensor, sequence.Fresh | None]:
    if raw is None:
        raw = np.random.default_rng(0).normal(size=(2, 100, len(NAMES)))
    if train is None:
        train = np.ones(raw.shape[1], bool)
    mu, sd = sequence.statistics(raw, train)
    x = torch.from_numpy(sequence.standardised(raw, mu, sd, plan.ladder.clip, plan.pad))
    return raw, x, sequence.fresh(raw, NAMES, train, plan)


def test_the_shipped_plan_is_the_scorers_plan() -> None:
    plan = sequence.load()
    assert plan.ladder_path.name == "ladder.toml" and plan.ladder == ladder.load()
    assert plan.ladder.columns == ladder.load().columns
    assert plan.rungs == ("full", "vol", "classic") and plan.schemes == ("forward",)
    assert plan.variants == ("full", "vol") and plan.models == sequence.MODELS
    assert plan.seeds == (1, 2, 3) and plan.folds == tuple(range(8)) and plan.threads == 2
    assert (plan.window, plan.future, plan.pad, plan.reach) == (336, 24, 335, 24)
    assert plan.window // plan.shape.patch == 84 and plan.future // plan.shape.patch == 6
    assert plan.target == "fresh" and plan.shape.width == 32
    lookbacks = {1: ["vol_1h", "ret_1h", "flow_1h", "vsurge_1h"], 6: ["vol_6h", "ret_6h"]}
    lookbacks[24] = ["vol_24h", "near_24h", "volume_24h", "ret_24h", "aret_24h", "flow_24h"]
    lookbacks[24].append("oi_24h")
    assert dict(plan.windowed) == {n: h for h, names in lookbacks.items() for n in names}
    assert set(plan.changes) == {"oi_vs_28d", "funding", "ls"} and not sequence.blind(plan)
    s = plan.supervised
    assert (s.dropout, s.lr, s.weight_decay, plan.jepa.dropout) == (0.2, 3e-4, 0.1, 0.0)
    assert (plan.selection.share, plan.selection.every) == (0.2, 100)
    j = plan.jepa
    assert (j.objective, j.variance, j.covariance) == ("predictor", 0.5, 0.02)
    assert (j.momentum_start, j.momentum_end) == (0.996, 1.0)
    assert plan.readouts == tuple(sequence.READOUTS) == ("last", "means", "spreads")
    assert plan.cs == (0.0001, 0.0003, 0.001, 0.01, 0.1, 1.0)


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
        [('readouts = ["last", "means", "spreads"]', 'readouts = ["last", "mean"]')],
        [('readouts = ["last", "means", "spreads"]', 'readouts = ["last", "last"]')],
        [('readouts = ["last", "means", "spreads"]', "readouts = []")],
        [("c = [0.01, 1.0]", "c = [0.01, 0.01]")],
        [("c = [0.01, 1.0]", "c = [0.0, 1.0]")],
        [("c = [0.01, 1.0]", "c = [-1.0, 1.0]")],
        [("c = [0.01, 1.0]", 'c = [0.01, "nan"]')],
        [("length = 36", "length = 34")],
        [("width = 8", "width = 6")],
        [("covariance = 0.04", "covariance = 0.04\ndropout = 0.1")],
        [("negatives = 200", "negatives = 0")],
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


SHOWN = PLAN[PLAN.index("[fresh.windowed]\n") : PLAN.index("\n[encoder]")]
CHANGES = 'changes = ["oi_vs_28d", "funding", "ls"]'


@pytest.mark.parametrize("target", ["future", "fresh"])
@pytest.mark.parametrize(
    "old, new, message",
    [
        ("vol_1h = 1\n", "vol_1h = 1\nm_vol_1h = 1\n", "fresh channel m_vol_1h"),
        (CHANGES, CHANGES.replace('"ls"', '"ls", "oi"'), "fresh channel oi($|,)"),
        ("vol_1h = 1\n", "vol_1h = 1\nls = 1\n", "fresh channel ls under both rules"),
        ("[fresh]\n", "[fresh]\nextra = 1\n", "fresh key extra"),
        ("vol_1h = 1\n", "vol_1h = 0\n", "fresh lookback vol_1h"),
        ("vol_1h = 1\n", "vol_1h = true\n", "fresh lookback vol_1h"),
        ("vol_1h = 1\n", "vol_1h = 1.0\n", "fresh lookback vol_1h"),
        (CHANGES, 'changes = ["ls", "ls"]', "fresh changes must be distinct"),
        (CHANGES, 'changes = "ls"', "fresh changes must be a list"),
    ],
)
def test_the_plan_refuses_unknown_or_doubled_fresh_channels(
    tmp_path: Path, target: str, old: str, new: str, message: str
) -> None:
    text = PLAN.replace('target = "future"', f'target = "{target}"')
    assert old in text
    with pytest.raises(ValueError, match=f"unknown or invalid: .*{message}"):
        tiny(tmp_path, text.replace(old, new, 1))


def test_a_rung_the_fresh_target_shows_nothing_of_is_refused(tmp_path: Path) -> None:
    only_lev = FRESH.replace(SHOWN, "[fresh.windowed]\noi_24h = 24\n")
    with pytest.raises(ValueError, match="rung vol shows the fresh target no windowed channel"):
        tiny(tmp_path / "a", only_lev)
    both = only_lev.replace('variants = ["vol"]', 'variants = ["full", "vol"]')
    with pytest.raises(ValueError, match=r"invalid: rung vol shows .* channel$"):
        tiny(tmp_path / "b", both)
    assert tiny(tmp_path / "c", only_lev.replace('variants = ["vol"]', 'variants = ["full"]'))
    future = tiny(tmp_path / "d", PLAN.replace(SHOWN, "[fresh.windowed]\noi_24h = 24\n"))
    assert future.windowed == (("oi_24h", 24),) and sequence.blind(future) == ["vol"]
    plan = tiny(tmp_path / "e", FRESH.replace('variants = ["vol"]', 'variants = ["full", "vol"]'))
    with pytest.raises(ValueError, match="rungs vol show the fresh target no windowed channel"):
        sequence.sequence({}, {}, replace(plan, windowed=(("oi_24h", 24),)))
    with pytest.raises(ValueError, match="rungs full, vol show the fresh target"):
        sequence.sequence({}, {}, replace(plan, windowed=()))
    model = sequence.jepa(len(NAMES) + 1, plan, 0)
    _, x, _ = fresh_world(plan)
    with pytest.raises(ValueError, match="the fresh target needs its fold's rules"):
        sequence.goal(model, x, torch.tensor([0]), torch.tensor([50]), plan)


def test_a_plan_without_a_fresh_table_still_loads(tmp_path: Path) -> None:
    old = PLAN.replace(PLAN[PLAN.index("[fresh]\n") : PLAN.index("[encoder]")], "")
    assert "fresh" not in old
    plan = tiny(tmp_path / "a", old)
    assert plan.target == "future" and plan.windowed == () and plan.changes == ()
    full = tiny(tmp_path / "b")
    assert replace(full, windowed=(), changes=(), ladder_path=plan.ladder_path) == plan
    with pytest.raises(ValueError, match="rung vol shows the fresh target no windowed channel"):
        tiny(tmp_path / "c", old.replace('target = "future"', 'target = "fresh"'))


def net_for(plan: sequence.Plan, trained: bool) -> encoder.Encoder:
    if not trained:
        return encoder.encoder(6, plan.shape, 0)
    torch.manual_seed(0)
    x = torch.randn(2, plan.pad + 300, 6)
    fit = sequence.eligible(np.ones(300, bool), plan.pad, plan.reach)
    shipped = replace(plan, jepa=replace(plan.jepa, objective="predictor", steps=20))
    net, _, initial = sequence.pretrain(
        x, fit, shipped, 0, np.random.SeedSequence(0), lambda line: None
    )
    assert sequence.weights(net) != initial
    return net


@pytest.mark.parametrize("trained", [False, True])
def test_every_token_sees_only_its_own_and_earlier_hours(tmp_path: Path, trained: bool) -> None:
    plan = tiny(tmp_path)
    net = net_for(plan, trained)
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
    _, x, rules = fresh_world(plan)
    model = sequence.jepa(x.shape[-1], plan, 0)
    coin, hour, t = torch.tensor([0]), torch.tensor([50]), 50
    base = sequence.goal(model, x, coin, hour, plan, rules)
    if target in ("future", "fresh"):
        first, last = t + 1, t + plan.future
    else:
        first, last = t + min(plan.shifts) - plan.window + 1, t + max(plan.shifts)

    def moves(j: int) -> bool:
        y = x.clone()
        y[0, plan.pad + j] += 3.0
        return not torch.equal(sequence.goal(model, y, coin, hour, plan, rules), base)

    assert [moves(j) for j in (first - 1, first, last, last + 1)] == [False, True, True, False]
    if rules is None:
        return

    def level_moves(j: int) -> bool:
        levels = rules.levels.clone()
        levels[0, plan.pad + j] += 3.0
        y = replace(rules, levels=levels)
        return not torch.equal(sequence.goal(model, x, coin, hour, plan, y), base)

    hours = (t - 1, t, t + 1, last, last + 1)
    assert [level_moves(j) for j in hours] == [False, True, True, True, False]


def test_the_fresh_target_hides_what_the_context_already_holds(tmp_path: Path) -> None:
    plan = tiny(tmp_path, FRESH)
    _, x, rules = fresh_world(plan)
    assert rules is not None
    model = sequence.jepa(x.shape[-1], plan, 0)
    coin, hour, t = torch.tensor([1]), torch.tensor([50]), 50
    base = sequence.goal(model, x, coin, hour, plan, rules)

    def moves(channel: int, k: int) -> bool:
        y = x.clone()
        y[1, plan.pad + t + k, channel] += 3.0
        return not torch.equal(sequence.goal(model, y, coin, hour, plan, rules), base)

    every = range(1, plan.future + 1)
    hidden = [(c, k) for c in (VOL_7D, MISSING, LS, M_LS) for k in every]
    hidden += [(c, k) for c in (VOL_6H, M_VOL_6H) for k in (1, 2)]
    shown = [(VOL_1H, 1), (VOL_1H, 4), (VOL_6H, 3), (VOL_6H, 4), (M_VOL_6H, 3)]
    assert [c for c in hidden if moves(*c)] == []
    assert all(moves(*c) for c in shown)
    window = sequence.unseen(x, coin, hour, plan, rules)[0]
    ahead = x[1, plan.pad + t + 1 : plan.pad + t + 1 + plan.future]
    assert (window[:, [VOL_7D, MISSING]] == 0).all() and (window[:2, [VOL_6H, M_VOL_6H]] == 0).all()
    assert torch.equal(window[2:, [VOL_6H, M_VOL_6H]], ahead[2:, [VOL_6H, M_VOL_6H]])
    assert torch.equal(window[:, VOL_1H], ahead[:, VOL_1H])


def test_a_change_channel_is_its_move_since_the_context_ends(tmp_path: Path) -> None:
    plan = tiny(tmp_path, FRESH)
    rng = np.random.default_rng(5)
    raw = rng.integers(-40, 40, size=(2, 120, len(NAMES))).astype(np.float64) / 8
    _, x, rules = fresh_world(plan, raw)
    assert rules is not None and rules.columns.tolist() == [LS, M_LS]
    coin, hour, t, n = torch.tensor([0]), torch.tensor([60]), 60, plan.future

    def window(r: sequence.Fresh | None) -> torch.Tensor:
        assert r is not None
        return sequence.unseen(x, coin, hour, plan, r)[0]

    base = window(rules)
    for j, c in enumerate((LS, M_LS)):
        d = raw[:, n:, c] - raw[:, :-n, c]
        assert rules.scale[j].item() == pytest.approx(d.std(), rel=1e-12)
        hand = np.clip((raw[0, t + 1 : t + 1 + n, c] - raw[0, t, c]) / rules.scale[j].item(), -5, 5)
        assert np.array_equal(base[:, c].numpy(), hand.astype(np.float32))
    shifted = raw.copy()
    shifted[..., [LS, M_LS]] += 2.5
    moved = fresh_world(plan, shifted)[2]
    assert moved is not None and torch.equal(moved.scale, rules.scale)
    assert torch.equal(window(moved), base)

    def at_scale(r: np.ndarray) -> torch.Tensor:
        found = sequence.fresh(r, NAMES, np.ones(120, bool), plan)
        assert found is not None and rules is not None
        return window(replace(found, scale=rules.scale))

    later = raw.copy()
    later[0, t + 2, LS] += 1.0
    assert torch.nonzero(at_scale(later) != base).tolist() == [[1, LS]]
    far = later.copy()
    far[0, t + 1, LS] += 1000.0
    assert at_scale(far)[0, LS] == 5.0
    model = sequence.jepa(x.shape[-1], plan, 0)
    targets = [
        sequence.goal(
            model, x, coin, hour, plan, sequence.fresh(r, NAMES, np.ones(120, bool), plan)
        )
        for r in (raw, shifted, later)
    ]
    assert torch.equal(targets[0], targets[1]) and not torch.equal(targets[0], targets[2])
    for gap, zeroed in ((t, list(range(n))), (t + 3, [2])):
        holed = raw.copy()
        holed[0, gap, LS] = np.nan
        got = at_scale(holed)
        keep = [k for k in range(n) if k not in zeroed]
        assert (got[zeroed, LS] == 0).all() and torch.equal(got[keep, LS], base[keep, LS])
        assert torch.equal(got[:, M_LS], base[:, M_LS])


def test_a_change_channels_scale_reads_only_training_hours(tmp_path: Path) -> None:
    plan = tiny(tmp_path, FRESH)
    rng = np.random.default_rng(6)
    raw = rng.normal(size=(2, 400, len(NAMES)))
    raw[0, :20, LS] = np.nan
    train = np.ones(400, bool)
    train[150:230] = False
    _, x, rules = fresh_world(plan, raw, train)
    changed = raw.copy()
    changed[:, ~train] = rng.normal(size=changed[:, ~train].shape) * 100 + 7
    changed[1, 160:170, LS] = np.nan
    _, y, other = fresh_world(plan, changed, train)
    assert rules is not None and other is not None
    assert torch.equal(rules.scale, other.scale)
    n = plan.future
    pairs = train[n:] & train[:-n]
    d = (raw[:, n:, [LS, M_LS]] - raw[:, :-n, [LS, M_LS]])[:, pairs].reshape(-1, 2)
    assert np.allclose(rules.scale.numpy(), np.nanstd(d, axis=0), rtol=1e-12)
    ok = np.flatnonzero(sequence.eligible(train, plan.pad, plan.reach))
    assert ok.size > 200
    coin = torch.from_numpy(np.repeat([0, 1], ok.size))
    hour = torch.from_numpy(np.tile(ok, 2))
    model = sequence.jepa(x.shape[-1], plan, 0)
    first = sequence.goal(model, x, coin, hour, plan, rules)
    assert torch.equal(first, sequence.goal(model, y, coin, hour, plan, other))
    inside = raw.copy()
    inside[0, 300, LS] += 50.0
    moved = sequence.fresh(inside, NAMES, train, plan)
    assert moved is not None and not torch.equal(moved.scale, rules.scale)


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
    _, x, rules = fresh_world(plan)
    model = sequence.jepa(x.shape[-1], plan, 5)
    assert not model.target.training
    opt = sequence.optimiser([model.online, model.predictor], plan.jepa)
    rows = torch.tensor([[0, 1, 0, 1], [10, 20, 30, 40]])
    old = [p.detach().clone() for p in model.target.parameters()]
    with torch.no_grad():
        want = sequence.goal(model, x, rows[0], rows[1], plan, rules).double().numpy()
    record = sequence.jepa_step(model, opt, x, rows, 0, plan, rules)
    assert want.shape == (4, 2, 8)
    assert record["target_sd"] == pytest.approx(float(np.median(want.std(axis=0))), rel=1e-5)
    assert record["target_sd"] > 0
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
            np.ones((ncoins, nhours), bool),
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

    rates: dict[torch.optim.Optimizer, list[float]] = {}
    schedule = sequence.schedule

    def recorded(opt: torch.optim.Optimizer, step: int, t: sequence.Training) -> None:
        schedule(opt, step, t)
        rates.setdefault(opt, []).append(opt.param_groups[0]["lr"])

    monkeypatch.setattr(sequence, "validation", validation)
    monkeypatch.setattr(sequence, "schedule", recorded)
    mu, sd = sequence.statistics(raw, train)
    x = torch.from_numpy(sequence.standardised(raw, mu, sd, 5.0, plan.pad))
    _, fitted, _, steps, _ = sequence.supervise(
        x,
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(keep.astype(np.float32)),
        skip,
        np.ones((ncoins, nhours), bool),
        train,
        plan,
        ("vol", 1, 3),
        lambda line: None,
    )
    selection, refit = rates.values()
    assert steps == 4 and len(selection) == 8 and len(refit) == 4
    assert refit == selection[:4]
    assert fitted[-1]["step"] == 3


def hand_vicreg(a: torch.Tensor) -> tuple[float, float]:
    v = a.detach().numpy().astype(np.float64)
    centred = v - v.mean(axis=0)
    cov = centred.T @ centred / (len(v) - 1)
    variance = np.maximum(0.0, 1.0 - np.sqrt(np.diag(cov) + 1e-4)).mean()
    off = cov - np.diag(np.diag(cov))
    return float(variance), float((off**2).sum() / v.shape[1])


def test_the_predictor_objective_is_its_hand_computation(tmp_path: Path) -> None:
    plan = tiny(tmp_path, PLAN.replace('objective = "z"', 'objective = "predictor"'))
    assert plan.jepa.objective == "predictor"
    g = torch.Generator().manual_seed(0)
    z = torch.randn(40, 8, generator=g, dtype=torch.float64) * torch.linspace(0.2, 2.0, 8)
    predicted = torch.randn(40, 3, 8, generator=g, dtype=torch.float64)
    predicted[:, 1] *= 0.3
    predicted[:, 2, 1] = predicted[:, 2, 0] * 2
    assert np.allclose([float(t) for t in sequence.vicreg(z)], hand_vicreg(z), rtol=1e-10)
    parts = np.array([hand_vicreg(predicted[:, i]) for i in range(3)])
    got = sequence.regulariser(z, predicted, plan.jepa)
    assert np.allclose([float(t) for t in got], parts.mean(axis=0), rtol=1e-10)
    on_z = sequence.regulariser(z, predicted, replace(plan.jepa, objective="z"))
    assert np.allclose([float(t) for t in on_z], hand_vicreg(z), rtol=1e-10)
    torch.manual_seed(1)
    x = torch.randn(2, plan.pad + 300, 6)
    fit = sequence.eligible(np.ones(300, bool), plan.pad, plan.reach)
    _, curve, _ = sequence.pretrain(x, fit, plan, 0, np.random.SeedSequence(1), lambda line: None)
    t = plan.jepa
    assert len(curve) == 3
    for c in curve:
        parts = t.prediction * c["prediction"] + t.variance * c["variance"]
        assert np.isfinite(c["loss"]) and c["loss"] == pytest.approx(
            parts + t.covariance * c["covariance"], rel=1e-5
        )


def test_the_validation_sample_draws_each_sides_negatives_at_one_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tiny(tmp_path)
    plan = replace(base, selection=replace(base.selection, negatives=2000))
    rng = np.random.default_rng(10)
    ncoins, nhours = 2, 20000
    shape = (ncoins, nhours)
    i_long, i_short = plan.outputs.index(("long", 24)), plan.outputs.index(("short", 24))
    long, short = rng.random(shape) < 0.1, rng.random(shape) < 0.1
    y = np.zeros((*shape, len(plan.outputs)), np.float32)
    y[..., i_long], y[..., i_short] = long, short
    score = rng.normal(size=(*shape, len(plan.outputs)))
    score[..., i_long] += np.where(long, 1.0, np.where(short, 3.0, 0.0))
    score[..., i_short] += np.where(short, 1.0, np.where(long, 3.0, 0.0))
    seen: list[int] = []

    def encode(model: object, x: object, coin: np.ndarray, hour: np.ndarray, p: object):
        seen.append(len(coin))
        return score[coin, hour]

    monkeypatch.setattr(sequence, "encode", encode)
    check = sequence.validation(
        torch.zeros(1),
        torch.from_numpy(y),
        torch.ones(y.shape),
        np.zeros(shape, bool),
        np.ones(nhours, bool),
        plan,
        np.random.SeedSequence(2),
    )
    got = check(sequence.Supervised(6, plan, 0))
    full, old = {}, {}
    both = ~long & ~short
    drawn = np.zeros(shape, bool)
    drawn.ravel()[rng.choice(np.flatnonzero(both.ravel()), 2000, replace=False)] = True
    for side, pos, i in (("long", long, i_long), ("short", short, i_short)):
        s = score[..., i]
        full[side] = auc(s[pos], s[~pos])
        old[side] = auc(s[pos], s[~pos & (~both | drawn)])
        assert abs(old[side] - full[side]) > 0.2
        assert abs(got[f"auc_{side}"] - full[side]) < 0.04, (side, got, full)
    assert seen[0] == int((~both).sum() + 2000)
    assert check(sequence.Supervised(6, plan, 0)) == got


def by_model(out: dict) -> dict:
    return {
        (d["variant"], d["scheme"], d["fold"], d["seed"], d["model"]): d for d in out["diagnostics"]
    }


def test_the_fold_table_covers_the_grid_and_repeats_with_the_seed(tmp_path: Path) -> None:
    values, grid = world(days=200)
    text = PLAN.replace('variants = ["vol"]', 'variants = ["full", "vol"]')
    text = text.replace("folds = [1, 3]", "folds = [2]")
    plan = tiny(tmp_path, text.replace("seeds = [3]", "seeds = [3, 4]"))
    first = sequence.sequence(values, grid, plan)
    second = sequence.sequence(values, grid, plan)
    assert json.dumps(first["folds"]) == json.dumps(second["folds"])
    timeless = [
        json.dumps([{k: v for k, v in d.items() if k != "seconds"} for d in out["diagnostics"]])
        for out in (first, second)
    ]
    assert timeless[0] == timeless[1] and json.dumps(first["curves"]) == json.dumps(
        second["curves"]
    )
    rows = sequence.common_rows(values, grid, plan)
    expected = {}
    for v in plan.variants:
        for s in plan.schemes:
            if plan.ladder.first_fold(s) > 2:
                continue
            for seed in plan.seeds:
                for h, r in rows.items():
                    for side in plan.ladder.sides:
                        n = int((r.fold == 2).sum())
                        expected[("supervised", v, None, None, s, seed, h, side)] = n
                        for model in ("jepa", "twin"):
                            for name in plan.readouts:
                                for c in plan.cs:
                                    expected[(model, v, name, c, s, seed, h, side)] = n
    keys = ("model", "variant", "readout", "c", "scheme", "seed", "horizon", "side")
    got = {tuple(r[k] for k in keys): r["rows"] for r in first["folds"] if r["model"] != "rung"}
    assert got == expected and all(n > 0 for n in got.values())
    assert all(np.isfinite(r["auc"]) for r in first["folds"] if r["model"] != "rung")
    assert len({r["auc"] for r in first["folds"] if r["model"] == "jepa"}) > 1
    assert set(first["encoders"]) == {
        sequence.encoder_name(v, s, f, seed, m) for v, s, f, seed, m in by_model(first)
    }
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


def test_each_readout_is_its_hand_computation() -> None:
    tokens = torch.from_numpy(np.random.default_rng(6).normal(size=(3, 20, 4)))
    t = tokens.numpy()
    spans = (6, 18, 20)
    want = np.concatenate(
        [t[:, -1]]
        + [t[:, -n:].mean(axis=1) for n in spans]
        + [
            np.sqrt(((t[:, -n:] - t[:, -n:].mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
            for n in spans
        ],
        axis=1,
    )
    got = sequence.readout(tokens).numpy()
    assert got.shape == (3, 4 * sequence.READOUTS["spreads"])
    assert np.allclose(got, want, rtol=0, atol=1e-12)
    assert sequence.READOUTS == {"last": 1, "means": 4, "spreads": 7}


@pytest.mark.parametrize("trained", [False, True])
def test_the_readouts_read_only_their_rows_window(tmp_path: Path, trained: bool) -> None:
    plan = tiny(tmp_path)
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(2, 200, 5))
    mu, sd = sequence.statistics(raw, np.ones(200, bool))
    net = net_for(plan, trained)
    coin, hour = np.array([0, 1]), np.array([120, 150])

    def features(r: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(sequence.standardised(r, mu, sd, 5.0, plan.pad))
        return sequence.features(net, x, coin, hour, plan)

    before = features(raw)
    assert before.shape == (2, plan.shape.width * sequence.READOUTS["spreads"])

    def moved(j: int) -> np.ndarray:
        r = raw.copy()
        r[0, j] += 4.0
        return features(r)

    start = 120 - plan.window + 1
    for j in (121, start - 1):
        after = moved(j)
        assert np.array_equal(after, before), j
    for j in (120, start):
        after = moved(j)
        assert not np.array_equal(after[0], before[0]), j
        assert np.array_equal(after[1], before[1]), j


def test_the_twin_and_the_jepa_share_one_readout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values, grid = world(days=200)
    plan = tiny(tmp_path, PLAN.replace("folds = [1, 3]", "folds = [2]"))
    seen: list[str] = []
    features = sequence.features

    def recorded(
        net: encoder.Encoder, x: torch.Tensor, coin: np.ndarray, hour: np.ndarray, p: object
    ) -> np.ndarray:
        seen.append(sequence.weights(net))
        return features(net, x, coin, hour, plan)

    monkeypatch.setattr(sequence, "features", recorded)
    out = sequence.sequence(values, grid, plan)
    found = by_model(out)
    twins = [d["weights"] for k, d in found.items() if k[-1] == "twin"]
    jepas = [d["weights"] for k, d in found.items() if k[-1] == "jepa"]
    assert twins and jepas and set(twins).isdisjoint(jepas)
    assert sorted(seen) == sorted(2 * twins + 2 * jepas)


def test_a_saved_model_reloads_to_its_digest_and_features(tmp_path: Path) -> None:
    values, grid = world(days=200)
    plan = tiny(tmp_path, PLAN.replace("folds = [1, 3]", "folds = [2]"))
    out = sequence.sequence(values, grid, plan)
    channels = len(plan.rung("vol").columns) + 1
    x = torch.randn(2, plan.pad + 60, channels)
    coin, hour = np.array([0, 1, 1]), np.array([0, 30, 59])
    for (v, s, f, seed, model), d in by_model(out).items():
        data = out["encoders"][sequence.encoder_name(v, s, f, seed, model)]
        net = (
            sequence.Supervised(channels, plan, 0)
            if model == "supervised"
            else (encoder.encoder(channels, plan.shape, 0))
        )
        sequence.restore(net, data)
        body = net.encoder if isinstance(net, sequence.Supervised) else net
        assert sequence.weights(body) == d["weights"], model
        again = (
            sequence.Supervised(channels, plan, 1)
            if model == "supervised"
            else (encoder.encoder(channels, plan.shape, 1))
        )
        sequence.restore(again, sequence.save(net))
        assert sequence.weights(again) == sequence.weights(net)
        if isinstance(net, encoder.Encoder) and isinstance(again, encoder.Encoder):
            assert np.array_equal(
                sequence.features(net, x, coin, hour, plan),
                sequence.features(again, x, coin, hour, plan),
            )
        else:
            assert np.array_equal(
                sequence.encode(net, x, coin, hour, plan),
                sequence.encode(again, x, coin, hour, plan),
            )


def fold_row(
    model: str,
    readout: str | None,
    c: float | None,
    seed: int,
    fold: int,
    side: str,
    auc: float,
    horizon: int = 24,
    variant: str = "vol",
    scheme: str = "forward",
) -> dict:
    return {
        "model": model,
        "readout": readout,
        "c": c,
        "seed": seed,
        "fold": fold,
        "side": side,
        "auc": auc,
        "horizon": horizon,
        "variant": variant,
        "scheme": scheme,
    }


def test_the_choice_is_the_best_seed_mean_of_long_and_short_at_24_hours() -> None:
    rows = []
    for fold in (2, 3):
        for side in ("long", "short"):
            for seed in (1, 2):
                rows.append(fold_row("jepa", "means", 0.01, seed, fold, side, 0.60))
                rows.append(fold_row("jepa", "last", 1.0, seed, fold, side, [0.70, 0.45][seed - 1]))
                rows.append(fold_row("jepa", "spreads", 0.1, seed, fold, side, 0.58))
                rows.append(fold_row("jepa", "spreads", 0.1, seed, fold, side, 0.99, horizon=1))
                rows.append(fold_row("twin", "last", 0.01, seed, fold, side, 0.99))
                rows.append(fold_row("jepa", "last", 0.01, seed, fold, side, 0.99, variant="full"))
                rows.append(fold_row("jepa", "last", 0.01, seed, fold, side, 0.99, scheme="loqo"))
            rows.append(fold_row("jepa", "spreads", 0.1, 1, fold, "any", 0.99))
            rows.append(fold_row("supervised", None, None, 1, fold, side, 0.99))
    assert sequence.choose(rows, "jepa", "vol") == pytest.approx(("means", 0.01, 0.60))
    assert sequence.choose(rows, "twin", "vol")[:2] == ("last", 0.01)
    tied = [r | {"readout": "spreads"} for r in rows if r["readout"] == "means"]
    tied += [r | {"c": 0.001} for r in rows if r["readout"] == "means"]
    assert sequence.choose(tied, "jepa", "vol")[:2] == ("means", 0.001)
    with pytest.raises(ValueError, match="supervised model has no readout or c to choose"):
        sequence.choose(rows, "supervised", "vol")
    with pytest.raises(ValueError, match="no forward 24 h"):
        sequence.choose(rows, "twin", "full")


def scores_args(run: Path, data: Path, model: str, tag: str) -> list[str]:
    return [
        "scores",
        "--sequence",
        run.name,
        "--model",
        model,
        "--variant",
        "vol",
        "--data-dir",
        str(data),
        "--tag",
        tag,
    ]


def table_side(table: pa.Table, h: int, side: str) -> pa.Table:
    mask = (table["horizon"].to_numpy() == h) & (
        table["side"].to_numpy(zero_copy_only=False) == side
    )
    return table.filter(pa.array(mask))


def softplus(v: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, v)


def test_the_scores_command_writes_a_file_the_ladder_scores_on_its_rows(tmp_path: Path) -> None:
    values, grid = world()
    run_inputs = write_runs(tmp_path, values, grid)
    text = PLAN.replace('schemes = ["loqo", "forward"]', 'schemes = ["forward"]')
    text = text.replace("folds = [1, 3]", "folds = [2, 3]").replace("seeds = [3]", "seeds = [3, 4]")
    plan_path = plans(tmp_path, text)
    command = ["sequence", "--inputs", run_inputs.name, "--plan", str(plan_path)]
    assert main([*command, "--data-dir", str(tmp_path)]) == 0
    (run,) = (tmp_path / "panel-sequence").glob("*-sequence")
    folds = pq.read_table(run / "folds.parquet").to_pylist()
    for model in ("jepa", "twin", "supervised"):
        assert main(scores_args(run, tmp_path, model, model)) == 0
        (out,) = (tmp_path / "panel-scores").glob(f"*-{model}")
        params = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["params"]
        assert params["sequence_run"] == run.name and len(params["encoders"]) == 4
        source = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["params"]
        assert params["sequence_manifest_sha256"] == store.sha256(run / "manifest.json")
        assert params["sequence_folds"] == {
            "file": "folds.parquet",
            "sha256": store.sha256(run / "folds.parquet"),
        }
        assert params["inputs"] == source["inputs"]
        assert params["labels_grid"] == source["labels_grid"]
        if model == "supervised":
            assert (params["readout"], params["c"], params["selection"]) == (None, None, None)
        else:
            name, c, value = sequence.choose(folds, model, "vol")
            assert (params["readout"], params["c"]) == (name, c)
            assert params["selection"]["value"] == value
        if model == "twin":
            continue
        file = out / "forward.parquet"
        store.refuse_held_back(file, "hour")
        meta = pq.ParquetFile(file).metadata
        stats = meta.row_group(0).column(meta.schema.names.index("hour")).statistics
        assert stats is not None and stats.has_min_max
        external = f'\n[external]\nname = "seq"\npath = "{file.as_posix()}"\nscheme = "forward"\n'
        (tmp_path / f"ladder-{model}.toml").write_text(ladder_text() + external, encoding="utf-8")
        ladder_run, _ = ladder.run(run_inputs, tmp_path, tmp_path / f"ladder-{model}.toml", model)
        scores = pq.read_table(ladder_run / "scores.parquet").to_pylist()
        seq = {(r["side"], r["coins"]): r["rows"] for r in scores if r["name"] == "seq"}
        vol = {(r["side"], r["coins"]): r["rows"] for r in scores if r["name"] == "vol"}
        assert seq == vol and seq and all(n > 0 for n in seq.values())
        table = pq.read_table(file)
        assert table.num_rows == 3 * sum(
            int((r.fold >= 2).sum())
            for r in sequence.common_rows(values, grid, sequence.load(plan_path)).values()
        )
    plan = sequence.load(run / "plan.toml", run / "ladder.toml")
    rows = sequence.common_rows(values, grid, plan)
    symbol = grid["symbol"].astype(str)

    def fetch(key: str) -> bytes:
        return (run / key).read_bytes()

    recorded = {
        (r["model"], r["readout"], r["c"], r["horizon"], r["side"], r["fold"]): r["auc"]
        for r in folds
        if r["seed"] == 3 and r["model"] != "rung"
    }
    choices = [("jepa", "means", 0.01), ("twin", "spreads", 1.0), ("supervised", None, None)]
    for model, name, c in choices:
        choice = (model, "vol", name, c)
        one = sequence.external(values, grid, replace(plan, seeds=(3,)), choice, fetch)["forward"]
        checked = 0
        for h, r in rows.items():
            at = {
                (k, t): i
                for i, (k, t) in enumerate(zip(symbol[r.keep], grid["ts_ns"][r.keep], strict=True))
            }
            for side in plan.ladder.sides:
                part = table_side(one, h, side)
                index = np.array(
                    [
                        at[(k, t)]
                        for k, t in zip(
                            part["coin"].to_pylist(), part["hour"].to_pylist(), strict=True
                        )
                    ]
                )
                s = part["score"].to_numpy()
                fold = part["fold"].to_numpy()
                for f in (2, 3):
                    label = r.y[side][index[fold == f]]
                    got = auc(s[fold == f][label], s[fold == f][~label])
                    assert got == recorded[(model, name, c, h, side, f)], (model, h, side, f)
                    checked += 1
        assert checked == 2 * 3 * 2
        tables = {
            seeds: sequence.external(values, grid, replace(plan, seeds=seeds), choice, fetch)[
                "forward"
            ]
            for seeds in ((3,), (4,), (3, 4))
        }
        for h in rows:
            mean = {}
            for side in ("long", "short"):
                per = [table_side(tables[k], h, side)["score"].to_numpy() for k in ((3,), (4,))]
                mean[side] = (per[0] + per[1]) / 2
                both = table_side(tables[(3, 4)], h, side)["score"].to_numpy()
                assert np.allclose(both, mean[side], rtol=1e-12, atol=1e-12), (model, h, side)
            either = np.log(np.expm1(softplus(mean["long"]) + softplus(mean["short"])))
            got = table_side(tables[(3, 4)], h, "any")["score"].to_numpy()
            assert np.allclose(got, either, rtol=1e-10, atol=1e-12), (model, h)
    (run / sequence.encoder_name("vol", "forward", 3, 4, "jepa")).unlink()
    assert main(scores_args(run, tmp_path, "jepa", "gone")) == 1
    assert not list((tmp_path / "panel-scores").glob("*-gone"))


def test_the_scores_command_refuses_a_changed_run_and_writes_nothing(tmp_path: Path) -> None:
    values, grid = world(days=200)
    run_inputs = write_runs(tmp_path, values, grid)
    text = PLAN.replace('schemes = ["loqo", "forward"]', 'schemes = ["forward"]')
    plan_path = plans(tmp_path, text.replace("folds = [1, 3]", "folds = [2]"))
    command = ["sequence", "--inputs", run_inputs.name, "--plan", str(plan_path)]
    assert main([*command, "--data-dir", str(tmp_path)]) == 0
    (run,) = (tmp_path / "panel-sequence").glob("*-sequence")
    shift = labels.ns(date(2024, 1, 1)) - labels.ns(date(2022, 1, 1)) - 5 * 24 * HOUR
    late = write_runs(tmp_path / "late", values, dict(grid, ts_ns=grid["ts_ns"] + shift))

    def params(copy: Path, change: dict, bound: bool = True) -> None:
        manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
        for key, value in change.items():
            manifest["params"][key] = value
        (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if bound:
            record = json.loads((copy / "complete.json").read_text(encoding="utf-8"))
            record["manifest_sha256"] = store.sha256(copy / "manifest.json")
            (copy / "complete.json").write_text(json.dumps(record), encoding="utf-8")

    def complete(copy: Path, clean: bool) -> None:
        record = json.loads((copy / "complete.json").read_text(encoding="utf-8"))
        (copy / "complete.json").write_text(json.dumps(record | {"clean": clean}), "utf-8")

    def relisted(copy: Path, name: str, content: str) -> None:
        (copy / name).write_text(content + "\n", "utf-8")
        record = json.loads((copy / "complete.json").read_text(encoding="utf-8"))
        record["files"][name] = store.sha256(copy / name)
        (copy / "complete.json").write_text(json.dumps(record), encoding="utf-8")

    encoder_file = sequence.encoder_name("vol", "forward", 2, 3, "jepa")
    zero = "0" * 64
    cases = {
        "encoder": (
            lambda d: (d / encoder_file).write_bytes((d / encoder_file).read_bytes() + b"0"),
            "sha256 differs from the sequence run's",
        ),
        "inputs": (
            lambda d: params(d, {"inputs": {"file": "inputs.parquet", "sha256": zero}}),
            "inputs table differs",
        ),
        "grid": (
            lambda d: params(d, {"labels_grid": {"file": "grid/0.parquet", "sha256": zero}}),
            "labels grid differs",
        ),
        "missing": (lambda d: (d / "complete.json").unlink(), "no complete.json"),
        "manifest": (
            lambda d: params(d, {"threads": 2}, bound=False),
            "manifest.json: its sha256 differs from complete.json's",
        ),
        "unclean": (lambda d: complete(d, False), "does not mark the run clean"),
        "plan": (lambda d: (d / "plan.toml").write_text(text + "\n", "utf-8"), "plan.toml"),
        "ladder": (
            lambda d: (d / "ladder.toml").write_text(ladder_text() + "\n", "utf-8"),
            "ladder.toml",
        ),
        "replanned": (
            lambda d: relisted(d, "plan.toml", text),
            "plan.toml: not the plan.toml the run's manifest records",
        ),
        "recorded": (
            lambda d: relisted(d, "ladder.toml", ladder_text()),
            "ladder.toml: not the ladder.toml the run's manifest records",
        ),
    }
    for name, (change, message) in cases.items():
        copy = tmp_path / "copies" / name
        shutil.copytree(run, copy)
        change(copy)
        with pytest.raises(ValueError, match=message):
            sequence.scores_run(copy, tmp_path, "jepa", "vol", name)
    assert not (tmp_path / "panel-scores").exists()
    copy = tmp_path / "copies" / "late"
    shutil.copytree(run, copy)
    params(copy, {"inputs_run": late.name})
    with pytest.raises(span.HeldBackError):
        sequence.scores_run(copy, tmp_path / "late", "jepa", "vol", "late")
    assert not (tmp_path / "late" / "panel-scores").exists()
    sequence.scores_run(run, tmp_path, "jepa", "vol", "kept")
    assert len(list((tmp_path / "panel-scores").glob("*-kept"))) == 1


def test_the_scores_need_every_seed_and_every_scored_quarter(tmp_path: Path) -> None:
    values, grid = world()
    text = PLAN.replace('schemes = ["loqo", "forward"]', 'schemes = ["forward"]')
    plan = tiny(tmp_path, text.replace("folds = [1, 3]", "folds = [2, 3]"))
    out = sequence.sequence(values, grid, plan)
    choice = ("twin", "vol", "last", 1.0)
    tables = sequence.external(values, grid, plan, choice, out["encoders"].__getitem__)
    assert set(tables) == {"forward"} and tables["forward"].num_rows > 0
    more = replace(plan, seeds=(3, 5))
    with pytest.raises(KeyError):
        sequence.external(values, grid, more, choice, out["encoders"].__getitem__)
    fewer = replace(plan, folds=(2,))
    with pytest.raises(ValueError, match=r"quarters \[4\] unscored"):
        sequence.external(values, grid, fewer, choice, out["encoders"].__getitem__)
    with pytest.raises(ValueError, match="needs a readout"):
        sequence.external(values, grid, plan, ("twin", "vol", None, None), out["encoders"].get)


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


@pytest.mark.parametrize(
    "text",
    [PLAN, FRESH.replace('variants = ["vol"]', 'variants = ["full"]')],
    ids=["future", "fresh"],
)
def test_the_left_out_quarter_and_its_gaps_reach_no_fitted_parameter(
    tmp_path: Path, text: str
) -> None:
    values, grid = world()
    plan = tiny(tmp_path, text.replace("folds = [1, 3]", "folds = [2]"))
    edges = ladder.bounds(plan.ladder)
    gap = plan.ladder.gap_days * 24 * HOUR
    first = sequence.sequence(values, grid, plan)
    moved, flipped = perturbed(values, grid, edges[2] - gap, edges[3] + gap)
    second = sequence.sequence(moved, flipped, plan)
    assert {k[1] for k in fits(first)} == {"loqo", "forward"}
    assert fits(first) == fits(second)
    assert json.dumps(first["curves"]) == json.dumps(second["curves"])
    for key in first["encoders"]:
        assert first["encoders"][key] == second["encoders"][key], key
    assert [r["auc"] for r in first["folds"]] != [r["auc"] for r in second["folds"]]


def test_labels_of_skipped_censored_and_uncommon_rows_are_never_read(tmp_path: Path) -> None:
    values, grid = world(skip=0.3, censored=0.3)
    values["ls"][np.random.default_rng(11).random(len(values["ls"])) < 0.2] = np.nan
    plan = tiny(tmp_path, PLAN.replace("folds = [1, 3]", "folds = [2]"))
    first = sequence.sequence(values, grid, plan)
    common = np.any([r.keep for r in sequence.common_rows(values, grid, plan).values()], axis=0)
    flipped = dict(grid)
    for h in labels.HORIZONS:
        hidden = grid["skip"] | grid[f"censored_{h}"] | ~common
        for side in ("long", "short"):
            flipped[f"{side}_{h}"] = np.where(hidden, ~grid[f"{side}_{h}"], grid[f"{side}_{h}"])
    second = sequence.sequence(values, flipped, plan)
    assert fits(first) == fits(second)
    assert json.dumps(first["curves"]) == json.dumps(second["curves"])
    assert json.dumps(first["folds"]) == json.dumps(second["folds"])


def test_a_fold_without_training_or_probe_rows_is_skipped(tmp_path: Path) -> None:
    values, grid = world(days=200)
    base = ladder_text("2022-02-20", first=1)
    text = PLAN.replace('schemes = ["loqo", "forward"]', 'schemes = ["forward"]')
    text = text.replace("folds = [1, 3]", "folds = [1]")
    long = tiny(tmp_path / "a", text.replace("length = 36", "length = 200"), base)
    out = sequence.sequence(values, grid, long)
    assert out["skipped"] == ["vol forward fold 2: no training row"]
    assert out["encoders"] == {} and out["diagnostics"] == []
    assert {r["model"] for r in out["folds"]} == {"rung"}
    plan = tiny(tmp_path / "b", text.replace("seeds = [3]", "seeds = [3, 4]"), base)
    start = labels.ns(date(2022, 2, 20)) + plan.pad * HOUR
    censored = dict(grid)
    censored["censored_24"] = grid["censored_24"] | (
        (grid["ts_ns"] >= start) & (grid["ts_ns"] < labels.ns(date(2022, 3, 1)))
    )
    out = sequence.sequence(values, censored, plan)
    assert out["skipped"] == ["vol forward fold 2 24 h: no probe row"]
    assert {r["horizon"] for r in out["folds"] if r["model"] != "rung"} == {1}
    empty = tiny(tmp_path / "c", PLAN.replace("folds = [1, 3]", "folds = [3]"))
    out = sequence.sequence(values, grid, empty)
    assert out["folds"] == [] and out["skipped"] == [] and out["diagnostics"]


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
    params = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["params"]
    saved = {sequence.encoder_name("vol", "loqo", 1, 3, m) for m in sequence.MODELS}
    assert (
        set(complete["files"])
        == {f"{n}.parquet" for n in ("diagnostics", "curves", "folds", "rows")}
        | {"report.md", "plan.toml", "ladder.toml"}
        | saved
    )
    assert set(params["encoders"]) == saved
    for name in saved:
        assert params["encoders"][name] == complete["files"][name] == store.sha256(run / name)
    assert params["inputs_run"] == run_inputs.name
    assert params["inputs"]["sha256"] == store.sha256(run_inputs / "inputs.parquet")
    assert params["labels_grid"]["sha256"] == store.sha256(
        tmp_path / "labels" / params["labels_run"] / "grid" / "0.parquet"
    )
    assert params["plan"]["sha256"] == complete["files"]["plan.toml"]
    assert params["ladder_plan"]["sha256"] == complete["files"]["ladder.toml"]
    folds = pq.read_table(run / "folds.parquet")
    assert set(folds.column_names) >= {"model", "variant", "readout", "c", "seed", "auc"}
    report = (run / "report.md").read_text(encoding="utf-8")
    assert "| jepa | vol | means | 0.01 | loqo | 3 | long |" in report
    assert "| supervised | vol | - | - | loqo | 3 | long |" in report
    assert str(run) in capsys.readouterr().out


SPLIT = {
    "a": ('["full"]', "[3, 4]", "[2]"),
    "b": ('["full"]', "[3, 4]", "[3]"),
    "c": ('["vol"]', "[3]", "[2, 3]"),
    "d": ('["vol"]', "[4]", "[2, 3]"),
    "e": ('["full", "vol"]', "[5]", "[2, 3]"),
}


def split_plan(variants: str, seeds: str, folds: str) -> str:
    text = FRESH.replace('schemes = ["loqo", "forward"]', 'schemes = ["forward"]')
    text = text.replace('rungs = ["full", "vol"]', 'rungs = ["vol", "full"]')
    text = text.replace('variants = ["vol"]', f"variants = {variants}")
    return text.replace("seeds = [3]", f"seeds = {seeds}").replace(
        "folds = [1, 3]", f"folds = {folds}"
    )


@pytest.fixture(scope="module")
def split(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    data = tmp_path_factory.mktemp("split")
    values, grid = world()
    run_inputs = write_runs(data, values, grid)
    runs = {"inputs": run_inputs, "data": data}
    whole = {
        "full": split_plan('["full", "vol"]', "[3, 4, 5]", "[2, 3]"),
        "reversed": split_plan('["vol", "full"]', "[5, 4, 3]", "[3, 2]"),
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sequence.io, "code_version", lambda: {"git_sha": "c" * 40, "dirty": False})
        for name, text in (whole | {k: split_plan(*v) for k, v in SPLIT.items()}).items():
            path = plans(data / "plans" / name, text)
            runs[name], _ = sequence.run(run_inputs, data, path, name)
    runs["parent"] = data / "plans" / "full" / "sequence.toml"
    return runs


def rebound(run: Path, manifest: dict) -> None:
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    if "manifest_sha256" in complete:
        complete["manifest_sha256"] = store.sha256(run / "manifest.json")
        (run / "complete.json").write_text(json.dumps(complete), encoding="utf-8")


def without_seconds(path: Path) -> str:
    return json.dumps(pq.read_table(path).drop_columns(["seconds"]).to_pylist())


def in_name_order(run: Path) -> None:
    for name in ("diagnostics", "curves", "folds"):
        rows = pq.read_table(run / f"{name}.parquet").to_pylist()
        keys = [(r["variant"], r["fold"], r["seed"]) for r in rows if r.get("model") != "rung"]
        assert keys and keys == sorted(keys), (run.name, name)


def selection_metrics(run: Path) -> None:
    rows = [
        r for r in pq.read_table(run / "curves.parquet").to_pylist() if r["model"] == "selection"
    ]
    assert rows
    for row in rows:
        assert row.get("rows", 0) > 0 and row["loss"] is None, row
        assert all(isinstance(row.get(k), float) for k in ("auc", "auc_long", "auc_short")), row


def test_split_runs_merge_into_the_full_run(split: dict[str, Path]) -> None:
    data, full = split["data"], split["full"]
    full_params = json.loads((full / "manifest.json").read_text(encoding="utf-8"))["params"]
    full_folds = pq.read_table(full / "folds.parquet").to_pylist()
    for name in SPLIT:
        piece = split[name]
        params = json.loads((piece / "manifest.json").read_text(encoding="utf-8"))["params"]
        assert params["encoders"] and all(
            full_params["encoders"][k] == v for k, v in params["encoders"].items()
        )
        rows = [
            r for r in pq.read_table(piece / "folds.parquet").to_pylist() if r["model"] != "rung"
        ]
        cells = {(r["variant"], r["fold"], r["seed"]) for r in rows}
        same = [r for r in full_folds if (r["variant"], r["fold"], r["seed"]) in cells]
        assert json.dumps(rows) == json.dumps([r for r in same if r["model"] != "rung"])
    reverse = split["reversed"]
    for name in ("folds", "curves", "rows"):
        one, two = (pq.read_table(d / f"{name}.parquet") for d in (full, reverse))
        assert one.equals(two), name
    assert without_seconds(reverse / "diagnostics.parquet") == without_seconds(
        full / "diagnostics.parquet"
    )
    order = [split[k] for k in ("d", "a", "e", "c", "b")]
    parent = split["parent"]
    argv = ["merge", "--runs", *map(str, order), "--plan", str(parent), "--data-dir", str(data)]
    assert main(argv) == 0
    (merged,) = (data / "panel-sequence").glob("*-merged")
    params = json.loads((merged / "manifest.json").read_text(encoding="utf-8"))["params"]
    assert params["encoders"] == full_params["encoders"]
    for key in params["encoders"]:
        assert (merged / key).read_bytes() == (full / key).read_bytes()
    for name in ("folds", "curves", "rows"):
        assert pq.read_table(merged / f"{name}.parquet").equals(
            pq.read_table(full / f"{name}.parquet")
        ), name
    assert without_seconds(merged / "diagnostics.parquet") == without_seconds(
        full / "diagnostics.parquet"
    )
    plan = sequence.load(merged / "plan.toml", merged / "ladder.toml")
    assert plan == replace(sequence.load(parent), ladder_path=plan.ladder_path)
    assert (merged / "plan.toml").read_text("utf-8") == parent.read_text("utf-8")
    assert params["plan"] == {"file": parent.name, "sha256": store.sha256(merged / "plan.toml")}
    for run in (full, reverse, merged):
        in_name_order(run)
        selection_metrics(run)
        curves = pq.read_table(run / "curves.parquet").to_pylist()
        assert all((r["target_sd"] is None) == (r["model"] != "jepa") for r in curves)
        assert all(r["target_sd"] > 0 for r in curves if r["model"] == "jepa")
        diagnostics = pq.read_table(run / "diagnostics.parquet").to_pylist()
        assert {(d["model"], d["target"]) for d in diagnostics} == {
            ("jepa", "fresh"),
            ("twin", ""),
            ("supervised", ""),
        }
    runtime = full_params["runtime"]
    assert runtime["capability"] == torch.backends.cpu.get_cpu_capability()
    assert runtime["pins"] == {k: os.environ.get(k) for k in sequence.PINS} and runtime["cpu"]
    assert params["runtime"] == {"capability": runtime["capability"], "pins": runtime["pins"]}
    assert [s["cpu"] for s in params["sources"]] == [runtime["cpu"]] * len(order)
    assert [s["run"] for s in params["sources"]] == [d.name for d in order]
    assert all(
        s["manifest_sha256"] == store.sha256(d / "manifest.json")
        for s, d in zip(params["sources"], order, strict=True)
    )
    for k in ("inputs_run", "inputs", "labels_run", "labels_grid", "ladder_plan", "torch", "numpy"):
        assert params[k] == full_params[k], k
    assert params["pieces_bound"] is True
    complete = json.loads((merged / "complete.json").read_text(encoding="utf-8"))
    assert complete["clean"]
    for model, variant in (("jepa", "full"), ("supervised", "vol")):
        one, _ = sequence.scores_run(full, data, model, variant, f"full-{model}")
        two, _ = sequence.scores_run(merged, data, model, variant, f"merged-{model}")
        assert pq.read_table(one / "forward.parquet").equals(pq.read_table(two / "forward.parquet"))
        three, _ = sequence.scores_run(reverse, data, model, variant, f"reversed-{model}")
        assert pq.read_table(one / "forward.parquet").equals(
            pq.read_table(three / "forward.parquet")
        )
        picked = [
            {
                k: v
                for k, v in json.loads((d / "manifest.json").read_text("utf-8"))["params"].items()
                if k in ("readout", "c", "selection")
            }
            for d in (one, two)
        ]
        assert picked[0] == picked[1]


def test_the_merge_refuses_runs_that_do_not_make_one_plan(
    split: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, parent = split["data"], split["parent"]
    pieces = [split[k] for k in SPLIT]
    whole = split_plan('["full", "vol"]', "[3, 4, 5]", "[2, 3]")
    narrow = plans(tmp_path / "narrow", split_plan('["full"]', "[3, 4, 5]", "[2, 3]"))
    longer = whole.replace("steps = 4\nbatch = 16\nlr", "steps = 5\nbatch = 16\nlr", 1)
    stepped = plans(tmp_path / "stepped", longer)
    relaid = plans(tmp_path / "relaid", whole, ladder_text() + "\n# the same ladder\n")
    monkeypatch.setattr(sequence.io, "code_version", lambda: {"git_sha": "c" * 40, "dirty": False})
    idle = plans(tmp_path / "idle", split_plan('["vol"]', "[3]", "[1]"))
    wide = plans(tmp_path / "wide", split_plan('["full", "vol"]', "[3, 4, 5]", "[1, 2, 3]"))
    empty, _ = sequence.run(split["inputs"], data, idle, "idle")

    def altered(name: str, change: Callable[[Path], None], piece: str = "d") -> list[Path]:
        copy = tmp_path / name / split[piece].name
        shutil.copytree(split[piece], copy)
        change(copy)
        return [copy if d == split[piece] else d for d in pieces]

    def manifest(change: Callable[[dict], None]) -> Callable[[Path], None]:
        def edit(copy: Path) -> None:
            record = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
            change(record)
            rebound(copy, record)

        return edit

    def params(key: str, value: object) -> Callable[[Path], None]:
        return manifest(lambda m: m["params"].__setitem__(key, value))

    def complete(change: Callable[[dict, Path], None]) -> Callable[[Path], None]:
        def edit(copy: Path) -> None:
            record = json.loads((copy / "complete.json").read_text(encoding="utf-8"))
            change(record, copy)
            (copy / "complete.json").write_text(json.dumps(record), encoding="utf-8")

        return edit

    def replan(old: str, new: str) -> Callable[[dict, Path], None]:
        def edit(record: dict, copy: Path) -> None:
            text = (copy / "plan.toml").read_text(encoding="utf-8")
            assert old in text
            (copy / "plan.toml").write_text(text.replace(old, new, 1), "utf-8")
            record["files"]["plan.toml"] = store.sha256(copy / "plan.toml")

        return edit

    stepped_plan = replan("steps = 4\nbatch = 16\nlr", "steps = 5\nbatch = 16\nlr")
    other_rules = {
        "lookback": replan("vol_6h = 3\n", "vol_6h = 2\n"),
        "changes": replan(CHANGES, 'changes = ["oi_vs_28d", "funding"]'),
        "target": replan('target = "fresh"', 'target = "future"'),
    }

    def retable(name: str, change: Callable[[list[dict]], None]) -> Callable[[Path], None]:
        def edit(copy: Path) -> None:
            path = copy / f"{name}.parquet"
            old = pq.read_table(path)
            rows = old.to_pylist()
            change(rows)
            pq.write_table(pa.Table.from_pylist(rows, schema=old.schema), path)
            rehash = complete(lambda r, d: r["files"].__setitem__(path.name, store.sha256(path)))
            rehash(copy)

        return edit

    def other_rung(rows: list[dict]) -> None:
        next(r for r in rows if r["model"] == "rung")["auc"] = 0.123

    def other_count(rows: list[dict]) -> None:
        key = next(k for k, v in rows[0].items() if k != "horizon" and type(v) is int)
        rows[0][key] += 1

    def encoders(change: Callable[[dict], None]) -> Callable[[Path], None]:
        return manifest(lambda m: change(m["params"]["encoders"]))

    def runtime(change: Callable[[dict], None]) -> Callable[[Path], None]:
        return manifest(lambda m: change(m["params"]["runtime"]))

    def unlisted(copy: Path) -> None:
        record = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
        key = next(iter(record["params"]["encoders"]))
        encoders(lambda e: e.update({key: None}))(copy)
        complete(lambda r, d: r["files"].pop(key))(copy)
        (copy / key).unlink()

    def grow(copy: Path) -> None:
        key = next(
            iter(json.loads((copy / "manifest.json").read_text("utf-8"))["params"]["encoders"])
        )
        (copy / key).write_bytes((copy / key).read_bytes() + b"0")

    zero = {"file": "x", "sha256": "0" * 64}
    cases = {
        "plan": (
            altered("plan", complete(stepped_plan)),
            "differs from the parent plan in more than variants, seeds, folds",
        ),
        "inputs": (altered("inputs", params("inputs", zero)), "different inputs tables"),
        "grid": (altered("grid", params("labels_grid", zero)), "different labels grids"),
        "ladder": (altered("ladder", params("ladder_plan", zero)), "different ladder plans"),
        "twice": ([*pieces, split["full"]], "is in both"),
        "same folder": ([*pieces, pieces[0]], "listed twice"),
        "same run id": ([*pieces, altered("same id", lambda d: None, "e")[-1]], "listed twice"),
        "merged": (
            altered("merged", manifest(lambda m: m["params"].__setitem__("sources", []))),
            "a merged run; merge takes sequence runs only",
        ),
        "missing": (pieces[:3], "no run fitted 18 cells"),
        "seed": (pieces[:4], "no run fitted 12 cells of the parent plan"),
        "unclean": (
            altered("unclean", complete(lambda r, d: r.update(clean=False))),
            "does not mark the run clean",
        ),
        "absent": (altered("absent", lambda d: (d / "complete.json").unlink()), "no complete.json"),
        "encoder": (altered("encoder", grow), "sha256 differs from complete.json's"),
        "torch": (altered("torch", params("torch", "0.0")), "ran different torch"),
        "numpy": (altered("numpy", params("numpy", "0.0")), "ran different numpy"),
        "code": (
            altered("code", manifest(lambda m: m["producer"].update(git_sha="0" * 40))),
            "ran different code",
        ),
    }
    for name, edit in other_rules.items():
        cases[name] = (altered(name, complete(edit)), "differs from the parent plan in more than")
    producer = {
        "dirty": ({"dirty": True}, "uncommitted changes"),
        "unknown": ({"git_sha": None}, "records no commit"),
    }
    for name, (change, message) in producer.items():
        edit = manifest(lambda m, change=change: m["producer"].update(change))
        cases[name] = (altered(name, edit), message)
    cases |= {
        "rung rows": (altered("rung", retable("folds", other_rung)), "rung rows differ"),
        "accounting": (altered("rows", retable("rows", other_count)), "row accounting differs"),
        "digests": (
            altered("digests", encoders(lambda e: e.update({next(iter(e)): "0" * 64}))),
            "model sha256 values differ",
        ),
        "unlisted": (altered("unlisted", unlisted), "model sha256 values differ"),
        "models": (
            altered("models", encoders(lambda e: e.pop(next(iter(e))))),
            "saved models are not every cell",
        ),
        "capability": (
            altered("capability", runtime(lambda r: r.update(capability="another"))),
            "different CPU capabilities",
        ),
        "pins": (
            altered("pins", runtime(lambda r: r["pins"].update(ATEN_CPU_CAPABILITY="avx"))),
            "different instruction-set pins",
        ),
        "no runtime": (
            altered("no runtime", manifest(lambda m: m["params"].pop("runtime"))),
            "records no numerical runtime",
        ),
    }
    before = sorted((data / "panel-sequence").iterdir())
    for name, (runs, message) in cases.items():
        with pytest.raises(ValueError, match=message):
            sequence.merge_run(runs, parent, data, name)
    with pytest.raises(ValueError, match="variants the parent plan lacks: vol"):
        sequence.merge_run(pieces, narrow, data, "narrow")
    with pytest.raises(ValueError, match="fits no cell of its plan"):
        sequence.merge_run([*pieces, empty], wide, data, "no cell")
    with pytest.raises(ValueError, match="differs from the parent plan in more than"):
        sequence.merge_run(pieces, stepped, data, "stepped")
    with pytest.raises(ValueError, match="different ladder plan from the parent plan's"):
        sequence.merge_run(pieces, relaid, data, "relaid")
    assert sorted((data / "panel-sequence").iterdir()) == before


def test_pieces_made_before_the_target_spread_merge_with_it_empty(
    split: dict[str, Path], tmp_path: Path
) -> None:
    copies = []
    for name in SPLIT:
        copy = tmp_path / "pieces" / split[name].name
        shutil.copytree(split[name], copy)
        path = copy / "curves.parquet"
        pq.write_table(pq.read_table(path).drop_columns(["target_sd"]), path)
        record = json.loads((copy / "complete.json").read_text(encoding="utf-8"))
        record["files"][path.name] = store.sha256(path)
        (copy / "complete.json").write_text(json.dumps(record), encoding="utf-8")
        copies.append(copy)
    merged, _ = sequence.merge_run(copies, split["parent"], tmp_path / "data", "before")
    curves = pq.read_table(merged / "curves.parquet")
    assert curves.schema == sequence.CURVES and curves["target_sd"].null_count == len(curves)
    full = pq.read_table(split["full"] / "curves.parquet")
    assert curves.drop_columns(["target_sd"]).equals(full.drop_columns(["target_sd"]))


def test_the_merge_takes_a_nan_auc_in_a_shared_rung_row(
    split: dict[str, Path], tmp_path: Path
) -> None:
    def chosen(row: dict) -> bool:
        return (row["model"], row["variant"], row["fold"], row["horizon"], row["side"]) == (
            "rung",
            "vol",
            2,
            1,
            "long",
        )

    pieces, sharing = [], 0
    for name in SPLIT:
        copy = tmp_path / "pieces" / split[name].name
        shutil.copytree(split[name], copy)
        old = pq.read_table(copy / "folds.parquet")
        rows = old.to_pylist()
        hits = [r for r in rows if chosen(r)]
        for row in hits:
            row["auc"] = float("nan")
        sharing += len(hits)
        pq.write_table(pa.Table.from_pylist(rows, schema=old.schema), copy / "folds.parquet")
        record = json.loads((copy / "complete.json").read_text(encoding="utf-8"))
        record["files"]["folds.parquet"] = store.sha256(copy / "folds.parquet")
        (copy / "complete.json").write_text(json.dumps(record), encoding="utf-8")
        pieces.append(copy)
    assert sharing >= 2
    run_dir, _ = sequence.merge_run(pieces, split["parent"], tmp_path, "nan")
    (row,) = [r for r in pq.read_table(run_dir / "folds.parquet").to_pylist() if chosen(r)]
    assert np.isnan(row["auc"])


def test_a_piece_whose_manifest_changed_is_refused_and_unbound_pieces_are_recorded(
    split: dict[str, Path], tmp_path: Path
) -> None:
    def copies(root: str) -> list[Path]:
        out = []
        for name in SPLIT:
            copy = tmp_path / root / split[name].name
            shutil.copytree(split[name], copy)
            out.append(copy)
        return out

    edited = copies("edited")
    for piece in edited:
        record = json.loads((piece / "manifest.json").read_text(encoding="utf-8"))
        record["params"]["inputs_run"] = "20250101T000000Z-other"
        (piece / "manifest.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(
        ValueError, match=r"manifest\.json: its sha256 differs from complete\.json's"
    ):
        sequence.merge_run(edited, split["parent"], tmp_path, "edited")
    assert not (tmp_path / "panel-sequence").exists()
    old = copies("old")
    complete = json.loads((old[2] / "complete.json").read_text(encoding="utf-8"))
    complete.pop("manifest_sha256")
    (old[2] / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    run_dir, _ = sequence.merge_run(old, split["parent"], tmp_path, "old")
    params = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["params"]
    assert params["pieces_bound"] is False
    assert [s["manifest_sha256"] for s in params["sources"]] == [
        store.sha256(d / "manifest.json") for d in old
    ]
    merged = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    assert merged["manifest_sha256"] == store.sha256(run_dir / "manifest.json")


def test_pieces_from_different_cpu_models_merge_and_list_them(
    split: dict[str, Path], tmp_path: Path
) -> None:
    pieces = []
    for name in SPLIT:
        copy = tmp_path / "pieces" / split[name].name
        shutil.copytree(split[name], copy)
        if name == "c":
            record = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
            record["params"]["runtime"]["cpu"] = "Another CPU"
            rebound(copy, record)
        pieces.append(copy)
    run_dir, _ = sequence.merge_run(pieces, split["parent"], tmp_path, "cpus")
    params = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["params"]
    cpus = [s["cpu"] for s in params["sources"]]
    assert cpus[2] == "Another CPU" and len(set(cpus)) == 2


def test_a_plan_replaced_while_it_is_read_configures_and_records_one_text(
    split: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = split["data"]
    path = plans(tmp_path / "plan", split_plan('["vol"]', "[3]", "[2]"))
    parent = plans(tmp_path / "parent", split_plan('["full", "vol"]', "[3, 4, 5]", "[2, 3]"))
    swaps = {
        path.with_name("ladder.toml").resolve(): ("c = 0.01", "c = 0.1"),
        parent.resolve(): ("steps = 4\nbatch = 16\nlr", "steps = 5\nbatch = 16\nlr"),
    }
    reads: dict[Path, int] = {}
    read_text = Path.read_text

    def replaced(self: Path, encoding: str | None = None) -> str:
        text = read_text(self, encoding=encoding)
        key = self.resolve()
        reads[key] = reads.get(key, 0) + 1
        return text.replace(*swaps[key]) if key in swaps and reads[key] > 1 else text

    monkeypatch.setattr(Path, "read_text", replaced)
    run_dir, _ = sequence.run(split["inputs"], data, path, "replaced")
    merged, _ = sequence.merge_run([split[k] for k in SPLIT], parent, tmp_path, "replaced")
    monkeypatch.undo()
    saved = sequence.load(run_dir / "plan.toml", run_dir / "ladder.toml")
    rungs = [
        r for r in pq.read_table(run_dir / "folds.parquet").to_pylist() if r["model"] == "rung"
    ]
    assert rungs and {r["c"] for r in rungs} == {saved.ladder.c} == {0.01}
    assert sequence.load(merged / "plan.toml", merged / "ladder.toml").jepa.steps == 4


def test_a_merge_lists_its_pieces_skipped_cells_sorted(
    split: dict[str, Path], tmp_path: Path
) -> None:
    pieces = []
    for key, line in zip(
        SPLIT,
        (
            "vol forward fold 3 24 h: no probe row",
            "full forward fold 2 1 h: no probe row",
            None,
            None,
            None,
        ),
        strict=True,
    ):
        copy = tmp_path / "pieces" / split[key].name
        shutil.copytree(split[key], copy)
        if line is not None:
            record = json.loads((copy / "complete.json").read_text(encoding="utf-8"))
            record["checks"]["skipped"] = [line]
            (copy / "complete.json").write_text(json.dumps(record), encoding="utf-8")
        pieces.append(copy)
    run_dir, _ = sequence.merge_run(pieces, split["parent"], tmp_path, "skipped")
    complete = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    assert complete["checks"]["skipped"] == [
        "full forward fold 2 1 h: no probe row",
        "vol forward fold 3 24 h: no probe row",
    ]


def test_a_crlf_parent_plan_is_written_and_hashed_as_the_same_bytes(
    split: dict[str, Path], tmp_path: Path
) -> None:
    folder = tmp_path / "crlf"
    folder.mkdir()
    for name in ("sequence.toml", "ladder.toml"):
        text = (split["parent"].parent / name).read_bytes().decode("utf-8")
        crlf = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
        (folder / name).write_bytes(crlf)
    pieces = [split[k] for k in SPLIT]
    run_dir, _ = sequence.merge_run(pieces, folder / "sequence.toml", tmp_path, "crlf")
    params = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["params"]
    files = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))["files"]
    assert params["plan"]["sha256"] == files["plan.toml"] == store.sha256(run_dir / "plan.toml")
    assert params["ladder_plan"]["sha256"] == files["ladder.toml"]


def test_a_merge_without_a_supervised_model_types_its_curves_like_a_run(
    split: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sequence.io, "code_version", lambda: {"git_sha": "c" * 40, "dirty": False})
    data = split["data"]
    unsupervised = '["jepa", "twin", "supervised"]'
    runs = {}
    for name, folds in (("whole", "[2, 3]"), ("two", "[2]"), ("three", "[3]")):
        text = split_plan('["vol"]', "[3]", folds).replace(unsupervised, '["jepa", "twin"]')
        runs[name], _ = sequence.run(split["inputs"], data, plans(tmp_path / name, text), name)
    parent = tmp_path / "whole" / "sequence.toml"
    merged, _ = sequence.merge_run([runs["two"], runs["three"]], parent, data, "untyped")
    one, two = (pq.read_table(d / "curves.parquet") for d in (runs["whole"], merged))
    assert one.schema == two.schema and one.equals(two)


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
