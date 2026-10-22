"""Exact results: Onsager's u(T), Yang's m0(T), Tc, and the L = 4 enumeration behind Gate A."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ellipk
from tipping_point.ising import exact

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "ising_exact.json").read_text("utf-8")
)
TOL = FIXTURE["decimals_tolerance"]


def test_energy_and_magnetisation_match_the_fixture_values() -> None:
    """Catches a wrong Onsager/Yang formula or a modulus-vs-parameter mix-up in K (>> 1e-4 off)."""
    for T, u in FIXTURE["energy_per_site"].items():
        assert exact.energy_per_site(float(T)) == pytest.approx(u, abs=TOL)
    for T, m in FIXTURE["magnetisation"].items():
        assert exact.magnetisation(float(T)) == pytest.approx(m, abs=TOL)


def test_critical_temperature() -> None:
    """Catches a wrong Tc constant: sinh(2/Tc) = 1 is Onsager's self-duality condition."""
    assert abs(exact.TC - FIXTURE["Tc"]) < 1e-9
    assert math.sinh(2 / exact.TC) == pytest.approx(1.0, abs=1e-12)


def test_energy_at_tc_is_minus_root_two_and_finite_around_it() -> None:
    """Catches a missing guard: at Tc, q = 1 and K diverges (NaN from sqrt of a negative)."""
    near = [exact.TC, math.nextafter(exact.TC, 0.0), math.nextafter(exact.TC, 4.0), exact.TC + 1e-9]
    for T in near:
        u = exact.energy_per_site(T)
        assert math.isfinite(u)
        assert u == pytest.approx(-math.sqrt(2), abs=1e-6)


def test_elliptic_k_quadrature_matches_scipy() -> None:
    """Catches a quadrature or convention error: scipy's ellipk takes the parameter m = k^2 too."""
    for q in (0.0, 0.1, 0.5, 0.738, 0.9, 0.99, 0.999):
        assert exact.elliptic_k(q) == pytest.approx(float(ellipk(q)), rel=1e-10)


def test_magnetisation_is_zero_from_tc_up_and_decreasing_below() -> None:
    """Catches a wrong branch: Yang's m0 vanishes at and above Tc."""
    assert exact.magnetisation(exact.TC) == 0.0
    assert exact.magnetisation(3.0) == 0.0
    below = [exact.magnetisation(T) for T in (1.0, 1.5, 2.0, 2.2, 2.26)]
    assert all(a > b > 0 for a, b in itertools.pairwise(below))


def test_l4_enumeration_reproduces_the_known_density_of_states() -> None:
    """Catches a wrong Hamiltonian or wrap in the enumeration Gate A trusts as its reference."""
    s, M, E = exact.enumerate_states(4)
    assert s.shape == (2**16, 4, 4)
    levels, counts = np.unique(E, return_counts=True)
    expected = {int(k): v for k, v in FIXTURE["density_of_states_L4"].items()}
    assert dict(zip(levels.tolist(), counts.tolist(), strict=True)) == expected
    assert np.all(M % 2 == 0)
    assert np.array_equal(np.sort(M), np.sort(-M))  # spin-flip symmetry


def test_boltzmann_weights_and_acceptance_limits() -> None:
    """Catches unnormalised weights, a sign error in the expected acceptance rate, or a rate
    that forgets the kernel accepts dE = 0 moves half the time."""
    _, _, energies = exact.enumerate_states(4)
    pi = exact.boltzmann_weights(energies, 2.0)
    assert pi.sum() == pytest.approx(1.0)
    assert pi[energies == energies.min()].sum() > pi[energies == energies.max()].sum()
    rates = [exact.acceptance_rate(T) for T in (1.5, exact.TC, 3.5, 1e6)]
    assert rates == sorted(rates)
    # infinite T accepts every move but half the dE = 0 ones, which are 6/16 of sites at random
    assert rates[-1] == pytest.approx(1 - 6 / 16 / 2, abs=1e-4)
