"""The circularity floor's GARCH(1,1)-t, fitted once on the 2023 pilot year (#29 S5).

Maximum likelihood on each symbol's 1-minute close-to-close log returns from 2023-01-01 to
2023-12-31, the pilot year and nothing later: r_t = sqrt(h_t) e_t, h_t = omega + alpha r_t-1^2 +
beta h_t-1, e_t Student-t with nu degrees of freedom scaled to unit variance, and h_0 the
unconditional variance, as nulls.simulate starts it. A return must span exactly one minute, so a gap
in the klines gives no return. The mean is taken as zero, as the simulator's is. The returns keep
their intraday seasonality, which the labeller removes and the simulated process does not have.

The fit is per minute, and so is the floor's process: nulls.simulate bridges each minute return to
10-second sub-steps (nulls.py says why). The command writes the report `garch` into the run's
reports/ folder: per symbol the four parameters, the unconditional variance against the sample's,
the persistence and its half-life. harness/market_floor.py holds the numbers frozen; the run only
records where they came from. This lives outside market/ because the optimiser is scipy's, a dev
dependency the package does not carry.

  uv run python harness/market_garch.py --run <labels run folder> [--archive <archive folder>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
from scipy.optimize import minimize
from scipy.signal import lfilter
from tipping_point.market import archive, bars, nulls, reports
from tipping_point.market.__main__ import ARCHIVE

PILOT = (date(2023, 1, 1), date(2023, 12, 31))
NU_EDGE = 1e-3  # nu within this of 2 is the fit running into the shocks' infinite variance
START = (0.9, 0.5, 10.0)  # persistence, alpha's share of it, nu: generic, far from any fit


class Fit(NamedTuple):
    garch: nulls.Garch
    loglik: float  # mean log-likelihood per return
    returns: int
    sample_variance: float  # mean r^2


def loglik(r: np.ndarray, garch: nulls.Garch) -> np.ndarray:
    """Each return's log-likelihood under `garch`, h_0 at its unconditional variance."""
    omega, alpha, beta, nu = garch.omega, garch.alpha, garch.beta, garch.nu
    h = np.empty(len(r))
    h[0] = garch.variance
    # h_t = (omega + alpha r_t-1^2) + beta h_t-1; zi carries beta h_0 into h_1
    h[1:] = lfilter([1.0], [1.0, -beta], omega + alpha * r[:-1] ** 2, zi=[beta * h[0]])[0]
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    return c - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(r * r / (h * (nu - 2)))


def unpack(x: np.ndarray) -> tuple[float, float, float, float]:
    """(variance, persistence s = alpha + beta, alpha's share of s, nu) from the optimiser's free
    coordinates, which hold each inside its range: variance > 0, 0 < s < 1, 0 < share < 1, nu > 2.
    The logistic of a large coordinate rounds to 1, which the caller must refuse."""
    log_var, logit_s, logit_share, log_nu = (float(v) for v in x)
    return (
        math.exp(log_var),
        1.0 / (1.0 + math.exp(-logit_s)),
        1.0 / (1.0 + math.exp(-logit_share)),
        2.0 + math.exp(log_nu),
    )


def _free(variance: float, s: float, share: float, nu: float) -> np.ndarray:
    return np.array([math.log(variance), _logit(s), _logit(share), math.log(nu - 2.0)])


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _garch(variance: float, s: float, share: float, nu: float) -> nulls.Garch:
    alpha = share * s
    return nulls.Garch(variance * (1.0 - s), alpha, s - alpha, nu)


def fit(r: np.ndarray, start: tuple[float, float, float] = START) -> Fit:
    """The maximum-likelihood GARCH(1,1)-t of the returns r, from `start` (persistence, alpha's
    share of it, nu) and the sample variance. Refused (edges): a fit that does not converge, and one
    on the edge of having no finite variance."""
    r = np.asarray(r, np.float64)
    sample = float(np.mean(r * r))

    def cost(x: np.ndarray) -> float:
        variance, s, share, nu = unpack(x)
        if not s < 1.0:
            return math.inf
        return -float(np.mean(loglik(r, _garch(variance, s, share, nu))))

    found = minimize(
        cost, _free(sample, *start), method="L-BFGS-B", options={"ftol": 1e-12, "gtol": 1e-8}
    )
    if not found.success:
        raise ValueError(f"the GARCH fit did not converge: {found.message}")
    variance, s, share, nu = unpack(found.x)
    edges(s, nu, len(r))
    return Fit(_garch(variance, s, share, nu), -float(found.fun), len(r), sample)


def edges(s: float, nu: float, n: int) -> None:
    """The fit's refusal. alpha + beta = s must be below 1 with a half-life shorter than the n
    returns: a longer one is no finite variance this sample can show. nu must stay clear of 2,
    where the shocks have no finite variance."""
    if not half_life(s) < n:
        raise ValueError(
            f"alpha + beta = {s!r} has a half-life of {half_life(s):g} steps, not shorter than the "
            f"{n} returns: no finite variance this sample can show"
        )
    if not nu - 2.0 > NU_EDGE:
        raise ValueError(f"nu = {nu!r} runs into 2: the shocks have no finite variance")


def half_life(s: float) -> float:
    """Steps for a shock to h to halve, ln 1/2 / ln s; infinite for s >= 1."""
    return math.log(0.5) / math.log(s) if 0.0 < s < 1.0 else math.inf


def minute_returns(klines: dict[str, np.ndarray]) -> np.ndarray:
    """Close-to-close log returns between klines exactly one minute apart."""
    one = np.diff(klines["open_time"]) == bars.MINUTE_MS
    return np.diff(np.log(klines["close"]))[one]


def pilot_returns(symbol: str, root: Path) -> np.ndarray:
    """A symbol's minute returns over the pilot year, refused if any kline reaches 2024."""
    klines = archive.collect([symbol], *PILOT, root).columns[(symbol, "klines")]
    nulls.refuse_confirmatory(int(klines["open_time"].max()) * 1_000_000)
    return minute_returns(klines)


FITS = pa.schema(
    [
        ("symbol", pa.string()),
        ("omega", pa.float64()),
        ("alpha", pa.float64()),
        ("beta", pa.float64()),
        ("nu", pa.float64()),
        ("persistence", pa.float64()),
        ("half_life_minutes", pa.float64()),
        ("variance", pa.float64()),
        ("sample_variance", pa.float64()),
        ("annual_vol", pa.float64()),
        ("sample_annual_vol", pa.float64()),
        ("loglik", pa.float64()),
        ("returns", pa.int64()),
    ]
)
MINUTES_PER_YEAR = 1440 * 365


def row(symbol: str, found: Fit) -> dict[str, Any]:
    g = found.garch
    s = g.alpha + g.beta
    return {
        "symbol": symbol,
        "omega": g.omega,
        "alpha": g.alpha,
        "beta": g.beta,
        "nu": g.nu,
        "persistence": s,
        "half_life_minutes": half_life(s),
        "variance": g.variance,
        "sample_variance": found.sample_variance,
        "annual_vol": math.sqrt(g.variance * MINUTES_PER_YEAR),
        "sample_annual_vol": math.sqrt(found.sample_variance * MINUTES_PER_YEAR),
        "loglik": found.loglik,
        "returns": found.returns,
    }


def report(found: dict[str, Fit]) -> reports.Report:
    table = pa.Table.from_pylist([row(s, f) for s, f in found.items()], schema=FITS)
    header = [
        "# GARCH(1,1)-t of the pilot year",
        f"Maximum likelihood on 1-minute close-to-close log returns, {PILOT[0]} to {PILOT[1]}, "
        "per symbol (harness/market_garch.py). The floor simulates this minute process and bridges "
        "each return to 10-second sub-steps. `variance` is the fit's unconditional variance per "
        "minute and `sample_variance` the year's mean squared return; the half-life is that of a "
        "shock to h.",
    ]
    lines = ["| " + " | ".join(table.column_names) + " |", "|" + "---|" * table.num_columns]
    for values in table.to_pylist():
        cells = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in values.values()]
        lines.append("| " + " | ".join(cells) + " |")
    summary = "\n\n".join([*header, "\n".join(lines), ""])
    return reports.Report("garch", {"fits": table}, summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="the labels run's folder")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args(argv)
    folder = args.run / "reports" / "garch"
    if folder.exists():
        sys.exit(f"error: {folder} exists")
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if "labeller" not in manifest["params"]:
        sys.exit(f"error: {args.run} is not a labels run")
    found = {s: fit(pilot_returns(s, args.archive)) for s in manifest["params"]["symbols"]}
    print(reports.write(report(found), args.run / "reports"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
