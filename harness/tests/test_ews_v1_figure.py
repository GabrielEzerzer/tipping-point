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
