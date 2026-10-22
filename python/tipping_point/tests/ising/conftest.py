"""Shared runs for the Ising harness. Everything runs on the CPU, in seconds."""

from pathlib import Path

import pytest
from tipping_point.ising import gates, run


@pytest.fixture(scope="session")
def validation(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[gates.Check]]:
    """The L = 4 validation run: 4096 replicas at each of 1.5, Tc and 3.5, judged by every gate."""
    root = tmp_path_factory.mktemp("validation")
    return run.validate(root, seed=1, replicas=4096, therm=200, sweeps=200, device="cpu")
