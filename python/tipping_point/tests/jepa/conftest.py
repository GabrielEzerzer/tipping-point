"""Shared runs for the JEPA harness, all on the CPU: the planted-stripe toy trained once
(harness/jepa_toy.py), and tiny simulator runs with a small model trained on them
(harness/jepa_ising.py). Step budgets, never wall-clock time, so every number is deterministic."""

from pathlib import Path

import jepa_ising
import jepa_toy
import pytest
from tipping_point.jepa import extract, train


@pytest.fixture(scope="session")
def toy_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The toy trained for its step budget: the run every learning test reads."""
    root = tmp_path_factory.mktemp("toy")
    return train.start(jepa_toy.CONFIG, jepa_toy.dataset(), root, "toy", "cpu")


@pytest.fixture(scope="session")
def ising(tmp_path_factory: pytest.TempPathFactory) -> jepa_ising.Ising:
    return jepa_ising.build(tmp_path_factory.mktemp("jepa-ising"))


@pytest.fixture(scope="session")
def ramp_latents(ising: jepa_ising.Ising) -> Path:
    """The trained checkpoint's latents table for the ramp run, extracted once."""
    return extract.extract(ising.training / train.FINAL, ising.runs["ramp"], "cpu")
