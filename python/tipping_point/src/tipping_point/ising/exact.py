"""Exact results for the 2D square-lattice Ising model with J = k_B = 1.

Onsager's energy per site u(T), Yang's spontaneous magnetisation m0(T), the critical temperature,
and brute-force enumeration of a small periodic lattice (the reference distribution for Gate A).
"""

from __future__ import annotations

import functools
import math

import numpy as np

TC = 2.0 / math.log(1.0 + math.sqrt(2.0))  # 2.2691853...

# 256 Gauss-Legendre nodes resolve K(q) to near machine precision for q up to ~0.9999.
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(256)


def elliptic_k(q: float) -> float:
    """K(q) = integral over [0, pi/2] of (1 - q sin^2 t)^(-1/2) dt, with q the parameter (k^2)."""
    t = np.pi / 4 * (_NODES + 1)  # map [-1, 1] onto [0, pi/2]
    return float(np.pi / 4 * np.sum(_WEIGHTS / np.sqrt(1 - q * np.sin(t) ** 2)))


def energy_per_site(T: float) -> float:
    """Onsager: u = -coth(2b) [1 + (2/pi)(2 tanh^2(2b) - 1) K(q)], q = 4 sinh^2(2b) / cosh^4(2b)."""
    y = 2.0 / T  # 2b
    q = 4 * math.sinh(y) ** 2 / math.cosh(y) ** 4  # <= 1, equal to 1 only at Tc
    if q >= 1.0:  # Tc: K diverges but its prefactor vanishes faster, so u = -coth(2bc) = -sqrt(2)
        return -1 / math.tanh(y)
    return -(1 / math.tanh(y)) * (1 + 2 / math.pi * (2 * math.tanh(y) ** 2 - 1) * elliptic_k(q))


def magnetisation(T: float) -> float:
    """Yang: m0 = (1 - sinh(2b)^-4)^(1/8) below Tc, zero at and above it."""
    if T >= TC:
        return 0.0
    return (1 - math.sinh(2.0 / T) ** -4) ** 0.125


@functools.cache
def enumerate_states(L: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every configuration of a periodic L x L lattice: spins (2^(L*L), L, L) int8, M and E."""
    n = L * L
    bits = (np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n)) & 1
    s = (2 * bits - 1).astype(np.int8).reshape(-1, L, L)
    M = s.sum(axis=(1, 2), dtype=np.int64)
    E = -(s * (np.roll(s, 1, 1) + np.roll(s, 1, 2))).sum(axis=(1, 2), dtype=np.int64)
    for array in (s, M, E):
        array.flags.writeable = False  # cached and shared between callers
    return s, M, E


def boltzmann_weights(E: np.ndarray, T: float) -> np.ndarray:
    """Normalised pi(x) proportional to exp(-E(x)/T), shifted by min E so nothing overflows."""
    w = np.exp(-(E - E.min()) / T)
    return w / w.sum()


def acceptance_rate(T: float, L: int = 4) -> float:
    """Expected Metropolis acceptance per attempted flip in equilibrium: E_pi[min(1, e^(-dE/T))]."""
    s, _, E = enumerate_states(L)
    h = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
    p = np.minimum(1.0, np.exp(-2.0 * s * h / T)).mean(axis=(1, 2))
    return float(boltzmann_weights(E, T) @ p)
