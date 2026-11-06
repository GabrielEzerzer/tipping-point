"""No test here reaches the network, even when the code under test breaks (tests/conftest.py)."""

import pytest


@pytest.fixture(autouse=True)
def offline(no_network: None) -> None:
    """Every test runs with the shared network block."""
