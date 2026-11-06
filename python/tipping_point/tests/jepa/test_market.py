"""The market JEPA (#30 M3): the loader's refusals and input contract, the manifest of input days,
the model's shapes through the generic objective and monitors, an encoder that reads a change in
variance alone, causality of the predictor and the readout, and the market toy
(fixtures/market_toy.py) as the known answer: it is learned, its latent tracks the planted slow
variable where the untrained twin's does not, a collapse mutant is caught, and runs are bitwise
reproducible and resumable on the CPU."""

import copy
import dataclasses
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import market_toy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F
from tipping_point.ising import io
from tipping_point.jepa import days, minutes, monitor, objective, probes, train
from tipping_point.jepa.days import DataError

SHORT = market_toy.config(steps=12, checkpoint_every=4, monitor_every=4, pca_frames=128)
FEATURES = market_toy.FEATURES
SPAN = market_toy.CONFIG.span
VAL = market_toy.VAL_DAYS[0]


@pytest.fixture(scope="module")
def toy_days() -> days.Dataset:
    return market_toy.dataset()


@pytest.fixture(scope="module")
def market_run(tmp_path_factory: pytest.TempPathFactory, toy_days: days.Dataset) -> Path:
    """The market toy trained for its step budget."""
    root = tmp_path_factory.mktemp("market")
    return train.start(market_toy.CONFIG, toy_days, root, "market-toy", "cpu")


def _table(run_dir: Path, name: str) -> dict[str, np.ndarray]:
    table = pa.concat_tables(pq.read_table(p) for p in io.chunks(run_dir, name))
    return {c: table[c].to_numpy() for c in table.column_names if c != "dim_std"}


def _probe_day(day: date) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    frames, valid, _ = days.read_day(market_toy.Source(), market_toy.SYMBOL, day, FEATURES)
    return torch.from_numpy(frames), torch.from_numpy(valid), market_toy.slow(day)


# ---------------------------------------------------------------- the loader (leakage tests 1, 2)


@pytest.mark.parametrize(
    ("symbol", "day", "refusal"),
    [
        ("BTCUSDT", date(2023, 10, 1), "outside the training span"),  # the declared end
        ("BTCUSDT", date(2023, 12, 31), "outside the training span"),
        ("BTCUSDT", date(2024, 1, 1), "confirmatory span"),
        ("BTCUSDT", date(2023, 1, 17), "outside the training span"),  # before the book starts
        ("ETHUSDT", date(2023, 3, 1), "not this model's symbol"),
    ],
)
def test_the_loader_refuses_days_outside_the_span_and_other_symbols(
    symbol: str, day: date, refusal: str
) -> None:
    """Leakage test 1: a day on or after the training end (2023-10-01), any confirmatory day, and
    another symbol are refused before any day is read, in training and validation alike."""
    source = market_toy.Source()
    train_days = [date(2023, 3, 1)] if day != date(2023, 3, 1) else [date(2023, 3, 2)]
    for t, v in (([*train_days, day], [date(2023, 3, 9)]), (train_days, [day])):
        with pytest.raises(DataError, match=refusal):
            days.load(source, SPAN, symbol, FEATURES, sorted(t), v)
    assert source.reads == []


def test_a_model_cannot_declare_a_span_reaching_the_confirmatory_data(
    toy_days: days.Dataset,
) -> None:
    """The span itself is refused if it reaches 2024, and a dataset loaded for another span or
    symbol is refused by the model."""
    late = market_toy.config(train_end="2024-01-02")
    with pytest.raises(DataError, match="must end by 2024-01-01"):
        train._Trainer(Path("unused"), "r", late, toy_days, torch.device("cpu"))
    early = market_toy.config(train_end="2023-09-01")
    with pytest.raises(DataError, match="span"):
        train._Trainer(Path("unused"), "r", early, toy_days, torch.device("cpu"))
    other = market_toy.config(symbol="ETHUSDT")
    with pytest.raises(DataError, match="ETHUSDT days only"):
        train._Trainer(Path("unused"), "r", other, toy_days, torch.device("cpu"))


def test_the_model_refuses_a_hand_built_dataset_listing_a_day_outside_its_span(
    toy_days: days.Dataset,
) -> None:
    """A Dataset made without days.load, whose sources list a training day on the span's end or a
    validation day in the confirmatory span, is refused when the training starts."""
    cases = (
        ("train", "2023-10-01", "outside the training span"),
        ("val", "2024-01-02", "confirmatory span"),
    )
    for role, day, refusal in cases:
        sources = copy.deepcopy(toy_days.sources)
        sources[role][-1]["day"] = day
        forged = dataclasses.replace(toy_days, sources=sources)
        with pytest.raises(DataError, match=refusal):
            train._Trainer(Path("unused"), "r", market_toy.CONFIG, forged, torch.device("cpu"))


def test_the_manifest_lists_every_input_day_with_its_sha256(market_run: Path) -> None:
    """Leakage test 2: every day the training read is named with the sha256 of its source bytes,
    lies inside [2023-01-18, 2023-10-01) and belongs to the model's one symbol."""
    manifest = io.read_json(market_run / "manifest.json")
    assert manifest["domain"] == "jepa-market"
    sources = manifest["data"]
    assert sources["symbol"] == market_toy.SYMBOL and sources["span"] == [
        "2023-01-18",
        "2023-10-01",
    ]
    source = market_toy.Source()
    for role, expected in (("train", market_toy.TRAIN_DAYS), ("val", market_toy.VAL_DAYS)):
        listed = [date.fromisoformat(r["day"]) for r in sources[role]]
        assert listed == list(expected)
        assert all(date(2023, 1, 18) <= d < date(2023, 10, 1) for d in listed)
        for record in sources[role]:
            day = date.fromisoformat(record["day"])
            assert record["sha256"] == source.read(market_toy.SYMBOL, day)[1]
    config = minutes.Config.from_json(manifest["params"]["config"])
    assert config == market_toy.CONFIG and config.span == SPAN
    with pytest.raises(ValueError, match="load_market"):
        train.load_model(market_run / train.FINAL)  # the lattice loader never reads a market model


def test_the_input_contract_is_checked() -> None:
    """Catches a day of the wrong length, a missing or mistyped column, a value that is not finite
    outside a gap, and a gap value reaching the model."""
    good, _ = market_toy.Source().read(market_toy.SYMBOL, date(2023, 3, 1))

    class Fixed:
        def __init__(self, table: pa.Table) -> None:
            self.table = table

        def read(self, symbol: str, day: date) -> tuple[pa.Table, str]:
            return self.table, "0" * 64

    def read(table: pa.Table) -> tuple[np.ndarray, np.ndarray, str]:
        return days.read_day(Fixed(table), market_toy.SYMBOL, date(2023, 3, 1), FEATURES)

    frames, valid, _ = read(good)
    assert frames.shape == (days.MINUTES, days.MINUTE, len(FEATURES)) and frames.dtype == np.float32
    assert set(np.flatnonzero(~valid)) == set(market_toy.GAP_MINUTES)
    assert np.isfinite(frames).all() and (frames[1000] == 0).all()  # NaN behind gaps zeroed
    assert (frames[100, 17] == 0).all() and (frames[100, 16] != 0).any()
    bad_value = good["f03"].to_numpy(zero_copy_only=False).copy()
    bad_value[5] = np.nan  # second 5 is not a gap
    cases = [
        (good.slice(0, days.SECONDS - 1), "86399 rows"),
        (good.drop_columns(["f07"]), "no column"),
        (good.drop_columns([days.GAP]), "no column"),
        (good.set_column(3, "f03", pa.array(bad_value.astype(np.float64))), "not float32"),
        (good.set_column(3, "f03", pa.array(bad_value)), "not finite"),
        (good.set_column(16, days.GAP, pa.array(np.zeros(days.SECONDS, np.int8))), "bool"),
    ]
    for table, refusal in cases:
        with pytest.raises(DataError, match=refusal):
            read(table)


def test_chunks_stay_inside_runs_of_consecutive_days(toy_days: days.Dataset) -> None:
    """Catches a chunk that joins two days with a missing day between them, and a monitor chunk
    holding an invalid minute."""
    frames = torch.zeros(4 * 10, 1, 1)
    valid = torch.ones(40, dtype=torch.bool)
    valid[25] = False
    run = torch.tensor([0] * 20 + [1] * 20)  # two runs (a missing day between them)
    m = days.Minutes(frames, valid, run)
    starts = m.starts(8).tolist()
    assert starts == list(range(0, 13)) + list(range(20, 33))
    assert m.starts(8, whole=True).tolist() == list(range(0, 13)) + list(range(26, 33))
    assert toy_days.train.run.unique().tolist() == [0]  # six consecutive days, one run
    gapped = [date(2023, 3, 1), date(2023, 3, 3)]
    split = days.load(market_toy.Source(), SPAN, market_toy.SYMBOL, FEATURES, gapped, [VAL])
    assert split.train.run.unique().tolist() == [0, 1]
    with pytest.raises(DataError, match="serve training and validation"):
        days.load(market_toy.Source(), SPAN, market_toy.SYMBOL, FEATURES, gapped, gapped[:1])


# ---------------------------------------------------------------- the model


def test_presets_have_the_research_pass_sizes() -> None:
    """Decision 4: S about 0.24M parameters, M about 0.6M. Counted: S 0.245M (encoder 0.113M,
    predictor 0.131M), M 0.651M (encoder 0.250M, predictor 0.401M), the pooling layer included."""
    counts = {}
    for name, spec in minutes.PRESETS.items():
        model = minutes.MarketJepa(spec)
        counts[name] = (minutes.parameters(model.context), minutes.parameters(model.predictor))
    (s_enc, s_pred), (m_enc, m_pred) = counts["S"], counts["M"]
    assert 0.20e6 <= s_enc + s_pred <= 0.28e6
    assert 0.55e6 <= m_enc + m_pred <= 0.70e6
    assert 0.20e6 <= m_enc <= 0.26e6 and 0.36e6 <= m_pred <= 0.44e6


def test_the_encoder_reads_a_change_in_variance_alone() -> None:
    """Catches an encoder blind to second moments, as one that pools its tokens by their mean alone
    is: minutes that differ only in how their variance is shared among the features (f00-f07
    scaled by exp(0.7 s), f08-f15 by exp(-0.7 s), so every mean and the total variance are the
    same for s = +1 and -1) are told apart by a ridge probe on the untrained latent. Measured
    on init seeds 1-5: accuracy 0.80-0.87 with the mean and log-std pooling, 0.52-0.57 with
    the mean alone. A minute whose tokens are all equal reads a finite latent and gradient."""
    model = minutes.MarketJepa(market_toy.SPEC)
    model.initialise(torch.Generator().manual_seed(1))

    def shared_out(seed: int) -> tuple[torch.Tensor, np.ndarray]:
        rng = np.random.default_rng(seed)
        s = np.repeat([1.0, -1.0], 500)
        x = rng.standard_normal((len(s), days.MINUTE, len(FEATURES)))
        x[:, :, :8] *= np.exp(0.7 * s)[:, None, None]
        x[:, :, 8:] *= np.exp(-0.7 * s)[:, None, None]
        return torch.from_numpy(x.astype(np.float32)), s

    (xa, sa), (xb, sb) = shared_out(10), shared_out(11)
    with torch.no_grad():
        za, zb = (model.target(x).double().numpy() for x in (xa, xb))
    guess = probes.ridge(za, sa).predict(zb)
    assert ((guess > 0) == (sb > 0)).mean() >= 0.7
    # no blocks, so a zeroed minute's tokens are exactly equal: their variance is 0
    flat = minutes.MarketJepa(dataclasses.replace(market_toy.SPEC, depth=0))
    flat.initialise(torch.Generator().manual_seed(1))
    z = flat.context(torch.zeros(2, days.MINUTE, len(FEATURES)))
    (z * torch.randn(z.shape, generator=torch.Generator().manual_seed(2))).sum().backward()
    grads = [p.grad for p in flat.context.parameters()]
    assert torch.isfinite(z).all()
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_the_objective_and_monitors_take_the_market_shapes() -> None:
    """objective.py and monitor.py were written for (B, dim, n, n) lattices; a chunk passes as
    (B, dim, C, horizons) and (B, dim, C, 1). The hidden cells are exactly the valid target minutes
    inside the chunk, and every loss term and statistic comes out finite."""
    spec = market_toy.SPEC
    model = minutes.MarketJepa(spec)
    model.initialise(torch.Generator().manual_seed(1))
    B, C, H = 3, spec.context, spec.horizons
    frames = torch.randn(
        B, C, days.MINUTE, spec.features, generator=torch.Generator().manual_seed(2)
    )
    valid = torch.ones(B, C, dtype=torch.bool)
    valid[0, 5] = False
    out = model.chunk(frames, valid)
    assert out["pred"].shape == out["target"].shape == (B, spec.dim, C, H)
    assert out["features"].shape == (B, spec.dim, C, 1) and out["keep"].shape == (B, C, 1)
    hidden = out["hidden"]
    for t in range(C):
        for k in range(H):
            expected = t + k + 1 < C and t + k + 1 != 5
            assert bool(hidden[0, t, k]) == expected, (t, k)
    loss = objective.prediction_loss(out["pred"], out["target"], hidden)
    per_sample = objective.errors(out["pred"], out["target"], hidden)
    variance, covariance = objective.vicreg(out["features"], out["keep"])
    stats = monitor.spread(out["features"])
    values = [loss, *per_sample, variance, covariance]
    assert all(torch.isfinite(v) for v in values)
    assert all(np.isfinite(stats[name]) for name in monitor.FLOORED)
    assert len(stats["dim_std"]) == spec.dim


def test_the_predictor_is_causal_and_ignores_invalid_minutes() -> None:
    """Catches a predictor whose position t sees a later minute, or reads the frame of a minute
    the gap mask marks invalid: changing either leaves every earlier prediction bit for bit."""
    spec = market_toy.SPEC
    model = minutes.MarketJepa(spec)
    model.initialise(torch.Generator().manual_seed(1))
    gen = torch.Generator().manual_seed(3)
    frames = torch.randn(2, spec.context, days.MINUTE, spec.features, generator=gen)
    valid = torch.ones(2, spec.context, dtype=torch.bool)
    valid[:, 4] = False
    with torch.no_grad():
        base = model.chunk(frames, valid)["pred"]
        later = frames.clone()
        later[:, 10:] += torch.randn(later[:, 10:].shape, generator=gen)
        moved = model.chunk(later, valid)["pred"]
        assert torch.equal(moved[:, :, :10], base[:, :, :10])
        assert not torch.equal(moved[:, :, 10:], base[:, :, 10:])
        hidden = frames.clone()
        hidden[:, 4] = 99.0
        assert torch.equal(model.chunk(hidden, valid)["pred"], base)


def test_readouts_have_no_position_effect_and_never_look_ahead() -> None:
    """Catches surprise that depends on where a minute sits in its batch, and any readout that a
    later minute changes (leakage test 3 for the latent): reading a truncated or a perturbed series
    gives bitwise the same values up to the cut. An invalid minute reads NaN in both readouts."""
    spec = market_toy.SPEC
    model = minutes.MarketJepa(spec)
    model.initialise(torch.Generator().manual_seed(1))
    frames, valid, _ = _probe_day(date(2023, 3, 1))
    frames, valid = frames[:300], valid[:300]
    latent, surprise = minutes.readouts(model, frames, valid, windows=64)
    C = spec.context
    assert torch.isnan(surprise[: C - 1]).all() and torch.isfinite(surprise[C - 1 : 100]).all()
    assert torch.nonzero(~valid).flatten().tolist() == [100]  # the toy's gap second
    assert torch.isnan(latent[~valid]).all() and torch.isnan(surprise[~valid]).all()
    assert torch.isfinite(latent[valid]).all()
    t = 150
    with torch.no_grad():
        out = model.chunk(
            frames[t - C + 1 : t + 1].unsqueeze(0), valid[t - C + 1 : t + 1].unsqueeze(0)
        )
    k = 2
    pred = out["pred"][0, :, C - 2 - k, k]
    direct = (pred - model.targets(frames[t : t + 1])[0]).pow(2).mean()
    assert torch.isclose(surprise[t, k], direct, rtol=1e-5)
    cut = 200
    short = minutes.readouts(model, frames[:cut], valid[:cut], windows=32)
    noisy = frames.clone()
    noisy[cut:] += torch.randn(noisy[cut:].shape, generator=torch.Generator().manual_seed(4))
    perturbed = minutes.readouts(model, noisy, valid, windows=64)
    for z, s in (short, perturbed):
        assert torch.equal(z[:cut].nan_to_num(-1.0), latent[:cut].nan_to_num(-1.0))
        assert torch.equal(s[:cut].nan_to_num(-1.0), surprise[:cut].nan_to_num(-1.0))


# ---------------------------------------------------------------- the market toy


def test_the_market_toy_is_learned_within_its_step_budget(market_run: Path) -> None:
    """From no skill (validation error above 1, the score of predicting nothing) to well below
    it, the training loss falling, every floor held throughout."""
    mon, losses = _table(market_run, "monitor"), _table(market_run, "train")["loss"]
    error = dict(zip(mon["step"].tolist(), mon["val_error"].tolist(), strict=True))
    assert error[0] > 1.0 and error[market_toy.CONFIG.steps] < 0.75 * error[0]
    assert losses[-20:].mean() < 0.75 * losses[:5].mean()
    assert not mon["fired"].any()
    assert not io.read_json(market_run / "complete.json")["collapse"]["collapsed"]


def test_the_latent_tracks_the_slow_variable_and_the_twins_does_not(market_run: Path) -> None:
    """The planted slow variable is read off the trained latent by a ridge probe fitted on one
    unseen day and tested on another, and by its first principal component alone; the untrained
    twin's latent reads it far worse. s shows only in second moments, so this fails for an encoder
    that pools its tokens by their mean alone (trained accuracy 0.51-0.52 on seeds 30-32)."""
    (fa, va, sa), (fb, vb, sb) = (_probe_day(d) for d in market_toy.PROBE_DAYS)
    va_np, vb_np = va.numpy(), vb.numpy()
    scores = {}
    for name, path in (("trained", train.FINAL), ("untrained", train.TWIN)):
        ckpt = train.load_market(market_run / path)
        assert ckpt.pca is not None
        za = minutes.readouts(ckpt.model, fa, va)[0].double().numpy()
        zb = minutes.readouts(ckpt.model, fb, vb)[0].double().numpy()
        fit = probes.ridge(za[va_np], sa[va_np].astype(float))
        guess = fit.predict(zb)
        pc1 = ckpt.pca.project(torch.from_numpy(zb))[:, 0].numpy()
        scores[name] = {
            "accuracy": float(((guess[vb_np] > 0) == (sb[vb_np] > 0)).mean()),
            "pc1": abs(float(np.corrcoef(pc1[vb_np], sb[vb_np])[0, 1])),
        }
    trained, untrained = scores["trained"], scores["untrained"]
    # measured on seeds 30 (this run's), 31 and 32: trained accuracy 0.996, 0.847, 0.939 and PC1
    # |r| 0.935, 0.597, 0.792; twin accuracy 0.595, 0.567, 0.608 and PC1 |r| 0.038, 0.046, 0.044.
    # Each bound clears the weakest trained seed and the strongest twin by 0.09 or more.
    assert trained["accuracy"] >= 0.75 and trained["accuracy"] >= untrained["accuracy"] + 0.15
    assert trained["pc1"] >= 0.45 and untrained["accuracy"] <= 0.7, scores
    assert untrained["pc1"] <= 0.15, scores


def test_the_detector_catches_a_market_collapse_on_its_floors(
    tmp_path: Path, toy_days: days.Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberately broken loss: VICReg off and no EMA target (targets from the context encoder
    itself, the gradient through both branches). The latents shrink towards a constant while their
    effective rank holds; the detector must fire on the absolute spread floors."""

    def leaky(self: minutes.MarketJepa, frames: torch.Tensor) -> torch.Tensor:
        z = self.context(frames)
        return F.layer_norm(z, (z.shape[-1],))

    monkeypatch.setattr(minutes.MarketJepa, "targets", leaky)
    config = market_toy.config(
        var_weight=0.0, cov_weight=0.0, steps=120, monitor_every=10, checkpoint_every=60
    )
    run_dir = train.start(config, toy_days, tmp_path, "mutant", "cpu")
    done = io.read_json(run_dir / "complete.json")
    assert done["collapse"]["collapsed"] and done["stopped_early"]
    fired = done["collapse"]["first_fired"]["fired"]
    assert fired and not any("effective_rank" in f for f in fired)
    assert all(not f.startswith("L") for f in fired)  # one group: no size in the message
    mon, floors = _table(run_dir, "monitor"), config.floors
    assert (mon["effective_rank"] > 4 * floors.effective_rank).all()
    assert mon["spatial_std"][0] > 5 * floors.spatial_std
    assert mon["spatial_std"][-1] < floors.spatial_std


def _losses(run_dir: Path) -> np.ndarray:
    return _table(run_dir, "train")["loss"]


def _chunks(run_dir: Path) -> dict[str, str]:
    files = io.read_json(run_dir / "complete.json")["files"]
    return {k: v for k, v in files.items() if k.split("/")[0] in train.TABLES}


def _same_weights(a: Path, b: Path) -> bool:
    first, second = train.load_market(a), train.load_market(b)
    left, right = first.model.state_dict(), second.model.state_dict()
    assert first.pca is not None and second.pca is not None
    pcs = torch.equal(first.pca.components, second.pca.components)
    return pcs and all(torch.equal(left[k], right[k]) for k in left)


def test_one_market_seed_twice_gives_bitwise_identical_runs(
    tmp_path: Path, toy_days: days.Dataset
) -> None:
    first = train.start(SHORT, toy_days, tmp_path / "a", "m", "cpu")
    second = train.start(SHORT, toy_days, tmp_path / "b", "m", "cpu")
    assert len(_losses(first)) == SHORT.steps
    assert np.array_equal(_losses(first), _losses(second))
    assert _chunks(first) == _chunks(second)
    assert _same_weights(first / train.FINAL, second / train.FINAL)
    other = train.start(dataclasses.replace(SHORT, seed=31), toy_days, tmp_path / "c", "m", "cpu")
    assert (_losses(first) != _losses(other)).all()


def _crash_after(calls: int, real: Callable[..., Any]) -> Callable[..., Any]:
    count = 0

    def crashing(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        if count > calls:
            raise RuntimeError("injected crash")
        return real(*args, **kwargs)

    return crashing


def test_a_crashed_market_run_resumes_bit_for_bit(
    tmp_path: Path, toy_days: days.Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a resume that loses optimiser or generator state, and one that trains on other
    days than the manifest names."""
    straight = train.start(SHORT, toy_days, tmp_path / "straight", "m", "cpu")
    monkeypatch.setattr(train._Trainer, "update", _crash_after(7, train._Trainer.update))
    with pytest.raises(RuntimeError, match="injected crash"):
        train.start(SHORT, toy_days, tmp_path / "crashed", "m", "cpu")
    monkeypatch.undo()
    (run_dir,) = [p for p in (tmp_path / "crashed" / "jepa").iterdir() if p.is_dir()]
    with pytest.raises(RuntimeError, match="hand its days"):
        train.resume(run_dir)
    shifted = market_toy.dataset(market_toy.Source(seed=1))  # same days, other bytes
    with pytest.raises(DataError, match="differ"):
        train.resume(run_dir, shifted)
    train.resume(run_dir, toy_days)
    assert _chunks(run_dir) == _chunks(straight)
    assert np.array_equal(_losses(run_dir), _losses(straight))
    assert _same_weights(run_dir / train.FINAL, straight / train.FINAL)
