"""Simulations against exact infinite-lattice results: Onsager's energy, Yang's magnetisation, Tc.

Away from Tc (|T - Tc| >~ 5 Tc / L) finite-size corrections fall like exp(-L / xi), so at L = 16
the measured <e> at T = 1.5 and 3.5, and <|m|> at 1.5, must equal u(T) and m0(T) within
statistical error. Error bars come from the spread of per-replica time averages.
"""

from pathlib import Path

import numpy as np
from tipping_point.ising import exact, gates, io, protocol, run


def _measured(run_dir: Path, L: int) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Per temperature: (replicas, sweeps) arrays of m = M / L^2 and e = E / L^2 in phase 1."""
    table = io.read_table(run_dir, "stats")
    c = {name: table[name].to_numpy() for name in table.column_names}
    out = {}
    for T in np.unique(c["temperature"]):
        mine = (c["phase"] == 1) & (c["temperature"] == T)
        steps = len(np.unique(c["step"][mine]))
        m = c["magnetisation"][mine].reshape(steps, -1).T / L**2
        e = c["energy"][mine].reshape(steps, -1).T / L**2
        out[float(T)] = (m, e)
    return out


def test_energy_and_magnetisation_match_onsager_and_yang(tmp_path: Path) -> None:
    """Catches a wrong sign, factor or double count in the Hamiltonian, or a wrong temperature
    scale: each moves <e> or <|m|> by far more than 5 standard errors."""
    schedule = protocol.Fixed((1.5, 3.5), therm=300, sweeps=1500)
    run_dir = run.start(run.Spec(16, 32, seed=8), schedule, tmp_path, "onsager", "cpu")
    measured = _measured(run_dir, 16)
    for T, (_, e) in measured.items():
        t = gates.t_statistic(e.mean(axis=1), exact.energy_per_site(T))
        assert abs(t) <= gates.Z_MAX, (T, e.mean(), exact.energy_per_site(T), t)
    m = measured[1.5][0]
    t = gates.t_statistic(np.abs(m).mean(axis=1), exact.magnetisation(1.5))
    assert abs(t) <= gates.Z_MAX, (np.abs(m).mean(), exact.magnetisation(1.5), t)


def _binder(m: np.ndarray) -> tuple[float, float]:
    """U4 = 1 - <m^4> / (3 <m^2>^2) and its leave-one-replica-out jackknife error."""
    m2, m4 = (m**2).mean(axis=1), (m**4).mean(axis=1)
    R = len(m)
    loo = 1 - ((m4.sum() - m4) / (R - 1)) / (3 * ((m2.sum() - m2) / (R - 1)) ** 2)
    u4 = 1 - m4.mean() / (3 * m2.mean() ** 2)
    return float(u4), float(np.sqrt((R - 1) / R * ((loo - loo.mean()) ** 2).sum()))


def test_binder_cumulants_cross_between_2_15_and_2_40(tmp_path: Path) -> None:
    """Locates Tc ~ 2.269 from the simulation alone: below Tc a larger lattice is more ordered
    (U4 grows with L), above it less, so U4(8) and U4(16) must swap order inside the bracket.
    Catches a wrong temperature scale or coupling, which moves the crossing out of it."""
    schedule = protocol.Fixed((2.15, 2.40), therm=300, sweeps=1500)
    u4 = {}
    for L in (8, 16):
        run_dir = run.start(run.Spec(L, 64, seed=9), schedule, tmp_path, f"binder{L}", "cpu")
        u4[L] = {T: _binder(m) for T, (m, _) in _measured(run_dir, L).items()}
    for T, larger_is_more_ordered in ((2.15, True), (2.40, False)):
        (small, err_small), (big, err_big) = u4[8][T], u4[16][T]
        gap = (big - small) if larger_is_more_ordered else (small - big)
        assert gap > 3 * np.hypot(err_small, err_big), (T, u4)
    assert 2.15 < exact.TC < 2.40
