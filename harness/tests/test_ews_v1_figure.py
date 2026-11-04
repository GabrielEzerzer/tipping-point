"""harness/ews_v1_figure.py draws the README's figure from three tables, built here in memory with
the column names of an ews result. matplotlib is no project dependency and the gate's environment
has none, so the test skips there; with matplotlib layered on, it runs:

  uv run --no-sync --with matplotlib pytest -c pyproject.toml -q harness/tests/test_ews_v1_figure.py
"""

import math
from pathlib import Path

import pyarrow as pa
import pytest

NAN = math.nan
SIZES = (16, 32)


def _fss() -> pa.Table:
    def row(
        quantity: str, cadence: str, role: str, estimate: float, expected: float, tolerance: float,
        verdict: str,
    ) -> dict[str, object]:  # fmt: skip
        return {"quantity": quantity, "cadence": cadence, "role": role, "estimate": estimate,
                "lo": estimate - 0.05, "hi": estimate + 0.05, "corrected": NAN, "draws": 20,
                "draws_used": 1.0, "expected": expected, "tolerance": tolerance,
                "verdict": verdict, "checkpoint": ""}  # fmt: skip

    rows = [
        row("l2_var_abs_m", "dense", "positive-control", 1.7, 1.75, 0.1, "pass"),
        row("mean_abs_m", "dense", "positive-control", -0.12, -0.125, 0.02, "pass"),
        row("skewness_abs_m", "dense", "positive-control", 0.02, 0.0, 0.1, "pass"),
        row("tau_int_abs_m", "dense", "descriptive", 2.2, 2.17, NAN, ""),
        row("one_minus_ac1", "matched", "reference", -1.7, NAN, 0.1, "pass"),
        row("inverse_nu_variance", "dense", "dynamic", 0.9, 1.0, 0.5, "pass"),
        row("inverse_nu_variance", "matched", "dynamic-robustness", 1.1, 1.0, 0.5, ""),
    ]
    return pa.Table.from_pylist([{"cell": cell, **r} for cell, r in enumerate(rows)])


def _peaks() -> pa.Table:
    """Var|m| on both cadences, and AC1 rows beside them that the figure must leave out."""
    rows = [
        {"indicator": indicator, "observable": "abs_m", "cadence": cadence, "L": L,
         "direction": "greater", "peak_eps_median": 0.6, "peak_eps_q1": 0.3, "peak_eps_q3": 0.9,
         "n": 8, "edge_fraction": 0.0, "checkpoint": ""}
        for indicator in ("variance", "ac") for cadence in ("dense", "matched") for L in SIZES
    ]  # fmt: skip
    return pa.Table.from_pylist([{"cell": cell, **r} for cell, r in enumerate(rows)])


def _auc() -> pa.Table:
    """The headline cell (window 20 tau_c) and a sensitivity cell beside it, every L and h."""
    rows = [
        {"indicator": indicator, "observable": "abs_m", "cadence": "dense", "L": L, "h": h,
         "window_taus": window, "bandwidth_taus": 10.0, "headline": window == 20.0,
         "direction": "less" if indicator == "skewness" else "greater",
         "auc": 0.02 * h if indicator == "skewness" else 1.0, "auc_lo": 0.0, "auc_hi": 1.0,
         "n_pos": 8, "n_neg": 8, "undefined": 0, "window": 10 * int(window), "bandwidth": 100.0,
         "length": 3001, "sweeps_per_sample": 5.0, "tau_c": 50.0, "checkpoint": ""}
        for indicator in ("variance", "ac", "skewness") for L in SIZES for h in (0.0, 1.0)
        for window in (10.0, 20.0)
    ]  # fmt: skip
    return pa.Table.from_pylist([{"cell": cell, **r} for cell, r in enumerate(rows)])


def test_the_figure_is_drawn_from_the_three_tables_and_refuses_a_missing_one(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")  # the gate's environment has none
    import ews_v1_figure

    out = tmp_path / "figures" / "ising-ews-v1.png"  # its folder does not exist yet
    ews_v1_figure.draw(_fss(), _peaks(), _auc(), out)
    data = out.read_bytes()
    assert data[1:4] == b"PNG"
    size = int.from_bytes(data[16:20]), int.from_bytes(data[20:24])  # the IHDR chunk
    assert size == (1500, 600)  # 10 by 4 inches at 150 dpi

    with pytest.raises(ValueError, match="the fss table has no column tolerance"):
        ews_v1_figure.draw(_fss().drop_columns(["tolerance"]), _peaks(), _auc(), out)
    with pytest.raises(FileNotFoundError, match="no peaks table"):
        ews_v1_figure.read(tmp_path, "peaks")


# ---------- a missing or doubled row, and a footer that claims cells the table lacks (#135)


def _figure():
    pytest.importorskip("matplotlib")  # the gate's environment has none
    import ews_v1_figure

    return ews_v1_figure


def _without(table: pa.Table, **match: object) -> pa.Table:
    rows = table.to_pylist()
    kept = [r for r in rows if not all(r[key] == value for key, value in match.items())]
    assert len(kept) < len(rows), match  # the fixture holds what is taken out
    return pa.Table.from_pylist(kept)


def _doubled(table: pa.Table, **match: object) -> pa.Table:
    rows = table.to_pylist()
    extra = [r for r in rows if all(r[key] == value for key, value in match.items())]
    assert extra, match
    return pa.Table.from_pylist(rows + extra)


# every fss row the figure reads: the five exponents, and 1/nu on the matched cadence for the peaks
FSS_READ = [
    ("l2_var_abs_m", "dense"),
    ("mean_abs_m", "dense"),
    ("skewness_abs_m", "dense"),
    ("one_minus_ac1", "matched"),
    ("inverse_nu_variance", "dense"),
    ("inverse_nu_variance", "matched"),
]


@pytest.mark.parametrize(("quantity", "cadence"), FSS_READ, ids=[f"{q}-{c}" for q, c in FSS_READ])
def test_a_missing_fss_row_is_refused_by_name(tmp_path: Path, quantity: str, cadence: str) -> None:
    figure = _figure()
    out = tmp_path / "figure.png"
    fss = _without(_fss(), quantity=quantity, cadence=cadence)
    with pytest.raises(ValueError, match="rows with") as refused:
        figure.draw(fss, _peaks(), _auc(), out)
    assert "the fss table has 0 rows with" in str(refused.value)
    assert f"'quantity': '{quantity}', 'cadence': '{cadence}'" in str(refused.value)
    assert not out.exists()


@pytest.mark.parametrize(("quantity", "cadence"), FSS_READ, ids=[f"{q}-{c}" for q, c in FSS_READ])
def test_a_doubled_fss_row_is_refused_by_name(tmp_path: Path, quantity: str, cadence: str) -> None:
    # two rows of one exponent, and the figure would draw whichever came first
    figure = _figure()
    out = tmp_path / "figure.png"
    fss = _doubled(_fss(), quantity=quantity, cadence=cadence)
    with pytest.raises(ValueError, match="the fss table has 2 rows with") as refused:
        figure.draw(fss, _peaks(), _auc(), out)
    assert f"'quantity': '{quantity}', 'cadence': '{cadence}'" in str(refused.value)
    assert not out.exists()


def test_the_footer_of_a_whole_table_says_every_l_and_h() -> None:
    figure = _figure()
    assert figure.headline_auc(figure.rows(_auc(), "auc")) == (
        "AUC against the fixed-T nulls at the headline cell (window 20 τc, bandwidth 10 τc), dense "
        "cadence, |m|, every L and h: Var 1.000, lag-1 AC 1.000, skewness 0.000 to 0.020."
    )


@pytest.mark.parametrize(
    "broken",
    [
        lambda auc: _without(auc, headline=True, h=1.0),
        lambda auc: _without(auc, headline=True, L=32),
        lambda auc: _without(auc, headline=True, indicator="skewness", L=32, h=1.0),
        lambda auc: _without(auc, headline=True, indicator="variance", L=16, h=0.0),
        lambda auc: _without(auc, headline=True, indicator="ac"),
        lambda auc: _doubled(auc, headline=True, indicator="skewness", L=16, h=1.0),
    ],
    ids=[
        "no-headline-rows-at-one-h",
        "no-headline-rows-at-one-L",
        "one-skewness-cell",
        "one-variance-cell",
        "no-ac-rows",
        "a-doubled-cell",
    ],
)
def test_a_headline_cell_missing_or_doubled_is_refused(tmp_path: Path, broken) -> None:
    # #135: with the headline rows at one h gone, the footer still read "every L and h"
    figure = _figure()
    auc = broken(_auc())
    with pytest.raises(ValueError, match=r"headline \w+ rows of \|m\|, dense, hold \(L, h\)"):
        figure.headline_auc(figure.rows(auc, "auc"))
    out = tmp_path / "figure.png"
    with pytest.raises(ValueError, match="not one each of"):
        figure.draw(_fss(), _peaks(), auc, out)
    assert not out.exists()


def test_the_refusal_names_the_grid_and_what_the_indicator_holds() -> None:
    figure = _figure()
    auc = _without(_auc(), headline=True, indicator="skewness", L=32, h=1.0)
    with pytest.raises(ValueError) as refused:
        figure.headline_auc(figure.rows(auc, "auc"))
    assert str(refused.value) == (
        "the auc table's headline skewness rows of |m|, dense, hold (L, h) "
        "[(16, 0.0), (16, 1.0), (32, 0.0)], not one each of "
        "[(16, 0.0), (16, 1.0), (32, 0.0), (32, 1.0)]"
    )


def test_the_auc_table_needs_its_l_and_h_columns() -> None:
    figure = _figure()
    for column in ("L", "h"):
        with pytest.raises(ValueError, match=f"the auc table has no column {column}$"):
            figure.rows(_auc().drop_columns([column]), "auc")
