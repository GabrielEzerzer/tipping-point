"""Physics gates: does a run sample the Boltzmann distribution it claims to?

A  exact enumeration at L = 4: the final snapshot's P(E) and P(|M|) (G-test), and the acceptance
   rate, against their exact equilibrium values
B  local Boltzmann identity on saved configs, any L: flipping a site with n unsatisfied bonds gives
   one with 4 - n at energy cost 8 - 4n, so P(n=4) = e^(-8/T) P(n=0) and P(n=3) = e^(-4/T) P(n=1)
C  detailed balance of each half-sweep: in equilibrium accepted flips costing +d and -d balance
   (d = 4, 8), and moves that do not raise the energy are always accepted
independence  replicas share no randomness: their |M| series are uncorrelated

Error bars come from the spread across replicas, which are independent by construction, so time
correlation inside a chain can never fake a pass.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import exact, io, protocol, run

Z_MAX = 5.0  # a replica t-statistic beyond 5 sigma fails
P_MIN = 1e-6  # a G-test p-value below this fails
MAX_CORRELATION = 0.01  # far above the noise once a temperature has ~1000 replicas
VALIDATION_TEMPERATURES = (1.5, exact.TC, 3.5)


@dataclass(frozen=True)
class Check:
    gate: str
    temperature: float
    passed: bool
    detail: str


def validate(
    root: Path,
    seed: int = 1,
    replicas: int = 4096,
    therm: int = 400,
    sweeps: int = 400,
    device: str | None = None,
) -> tuple[Path, list[Check]]:
    """A small L = 4 run at 1.5, Tc and 3.5, judged by every gate; results go to gates.json."""
    schedule = protocol.Fixed(VALIDATION_TEMPERATURES, therm, sweeps, stats_stride=8, saves=5)
    spec = run.Spec(L=4, replicas=replicas, seed=seed, tally=True)
    run_dir = run.start(spec, schedule, root, f"L4-val-s{seed}", device)
    checks = gate_a(run_dir) + gate_b(run_dir) + gate_c(run_dir) + independence(run_dir)
    io.write_json(run_dir / "gates.json", [asdict(c) for c in checks])
    return run_dir, checks


def t_statistic(values: np.ndarray, mean: float = 0.0) -> float:
    """(average - mean) / standard error, across independent replicas."""
    diff = float(values.mean()) - mean
    se = float(values.std(ddof=1)) / math.sqrt(len(values))
    if diff == 0:
        return 0.0
    return diff / se if se > 0 else math.copysign(math.inf, diff)


def g_test(observed: np.ndarray, expected: np.ndarray) -> float:
    """p-value of G = 2 sum O ln(O/E) on chi^2(cells - 1); cells expecting < 5 are pooled."""
    small = expected < 5
    if small.any():
        observed = np.append(observed[~small], observed[small].sum())
        expected = np.append(expected[~small], expected[small].sum())
    if len(observed) < 2:
        return 1.0
    seen = observed > 0
    g = 2 * float(np.sum(observed[seen] * np.log(observed[seen] / expected[seen])))
    df = torch.tensor((len(observed) - 1) / 2, dtype=torch.float64)
    return float(torch.special.gammaincc(df, torch.tensor(g / 2, dtype=torch.float64)))


def _histogram_test(sample: np.ndarray, values: np.ndarray, pi: np.ndarray) -> float:
    """G-test of a sample against the exact distribution of `values` under weights pi."""
    levels = np.unique(values)
    observed = np.array([(sample == v).sum() for v in levels])
    if observed.sum() < len(sample):
        return 0.0  # the sample holds a value no configuration can have
    expected = np.array([pi[values == v].sum() for v in levels]) * len(sample)
    return g_test(observed, expected)


def _columns(table: pa.Table) -> dict[str, np.ndarray]:
    return {name: table[name].to_numpy() for name in table.column_names if name != "spins"}


def _per_replica(rows: dict[str, np.ndarray], mask: np.ndarray, column: str) -> np.ndarray:
    """Sum of `column` over the masked rows, per replica, in replica order."""
    replicas = rows["replica"][mask]
    return np.bincount(replicas, weights=rows[column][mask])[np.unique(replicas)]


def gate_a(run_dir: Path) -> list[Check]:
    L = io.read_json(run_dir / "manifest.json")["params"]["L"]
    if L > 4:
        raise ValueError(f"Gate A enumerates 2^(L*L) states and needs L <= 4, got {L}")
    _, M_all, E_all = exact.enumerate_states(L)
    stats = _columns(io.read_table(run_dir, "stats"))
    balance = _columns(io.read_table(run_dir, "balance"))
    final = stats["step"] == stats["step"].max()  # one snapshot: replicas are i.i.d. samples
    checks = []
    for T in np.unique(stats["temperature"][final]):
        mine = final & (stats["temperature"] == T)
        pi = exact.boltzmann_weights(E_all, T)
        p_e = _histogram_test(stats["energy"][mine], E_all, pi)
        p_m = _histogram_test(np.abs(stats["magnetisation"][mine]), np.abs(M_all), pi)
        here = balance["temperature"] == T
        rate = _per_replica(balance, here, "accepted") / _per_replica(balance, here, "attempted")
        expected = exact.acceptance_rate(T, L)
        t = t_statistic(rate, expected)
        passed = min(p_e, p_m) >= P_MIN and abs(t) <= Z_MAX
        detail = (
            f"p(E)={p_e:.3g} p(|M|)={p_m:.3g} "
            f"acceptance {rate.mean():.5f} vs exact {expected:.5f} (t={t:+.2f})"
        )
        checks.append(Check("A", float(T), bool(passed), detail))
    return checks


def _unsatisfied(spins: np.ndarray) -> np.ndarray:
    """(configs, 5): how many sites have n = 0..4 unsatisfied bonds, n = (4 - s h) / 2."""
    h = np.roll(spins, 1, 1) + np.roll(spins, -1, 1) + np.roll(spins, 1, 2) + np.roll(spins, -1, 2)
    n = (4 - spins * h) // 2
    return np.stack([(n == k).sum(axis=(1, 2)) for k in range(5)], axis=1)


def gate_b(run_dir: Path) -> list[Check]:
    params = io.read_json(run_dir / "manifest.json")["params"]
    L, batch_size = params["L"], params["replicas"] * params["blocks"]
    sums: dict[float, np.ndarray] = {}  # temperature -> (replica, n) site counts over saved configs
    for path in io.chunks(run_dir, "configs"):
        with pq.ParquetFile(path) as f:
            for batch in f.iter_batches(batch_size=4096):  # bounded memory at any L
                counts = _unsatisfied(io.unpack_column(batch.column("spins"), L))
                temperature = batch.column("temperature").to_numpy()
                replica = batch.column("replica").to_numpy()
                for T in np.unique(temperature):
                    mine = temperature == T
                    total = sums.setdefault(float(T), np.zeros((batch_size, 5)))
                    np.add.at(total, replica[mine], counts[mine])
    checks = []
    for T, total in sorted(sums.items()):
        seen = total[total.sum(axis=1) > 0]  # the replicas that visited T
        t4 = t_statistic(seen[:, 4] - math.exp(-8 / T) * seen[:, 0])
        t3 = t_statistic(seen[:, 3] - math.exp(-4 / T) * seen[:, 1])
        n0, n4 = seen[:, 0].sum(), seen[:, 4].sum()
        ratio = f"{n4 / n0:.5f}" if n0 else "undefined"
        detail = f"P(n=4)/P(n=0) {ratio} vs {math.exp(-8 / T):.5f} (t={t4:+.2f}), n=3/1 t={t3:+.2f}"
        checks.append(Check("B", T, bool(max(abs(t4), abs(t3)) <= Z_MAX), detail))
    return checks


def gate_c(run_dir: Path) -> list[Check]:
    balance = _columns(io.read_table(run_dir, "balance"))
    checks = []
    for T in np.unique(balance["temperature"]):
        here = balance["temperature"] == T

        def accepted(colour: int, delta: int, here: np.ndarray = here) -> np.ndarray:
            mask = here & (balance["colour"] == colour) & (balance["delta_e"] == delta)
            return _per_replica(balance, mask, "accepted")

        # accepted(+d) - accepted(-d) per replica, for black d = 4, 8 then white d = 4, 8
        ts = [t_statistic(accepted(c, d) - accepted(c, -d)) for c in (0, 1) for d in (4, 8)]
        free = here & (balance["delta_e"] <= 0)
        always = bool(np.array_equal(balance["accepted"][free], balance["attempted"][free]))
        passed = max(abs(t) for t in ts) <= Z_MAX and always
        detail = (
            f"accepted(+d) - accepted(-d) t = {', '.join(f'{t:+.2f}' for t in ts)}; "
            f"dE <= 0 always accepted: {always}"
        )
        checks.append(Check("C", float(T), bool(passed), detail))
    return checks


def independence(run_dir: Path) -> list[Check]:
    stats = _columns(io.read_table(run_dir, "stats"))
    measured = stats["phase"] == 1
    checks = []
    for T in np.unique(stats["temperature"][measured]):
        mine = measured & (stats["temperature"] == T)
        steps = len(np.unique(stats["step"][mine]))
        m = np.abs(stats["magnetisation"][mine]).reshape(steps, -1).astype(np.float64)
        R = m.shape[1]
        spread = m.var(axis=0).mean()  # one replica's variance over time
        # Var(batch mean) = spread / R * (1 + (R - 1) * mean pairwise correlation)
        corr = (R * m.mean(axis=1).var() / spread - 1) / (R - 1) if spread > 0 else 0.0
        detail = f"mean inter-replica correlation of |M| {corr:+.5f} over {R} replicas"
        checks.append(Check("independence", float(T), bool(corr <= MAX_CORRELATION), detail))
    return checks
