# Every panel test runs with the shared network block (tests/conftest.py).

import pytest


@pytest.fixture(autouse=True)
def offline(no_network: None) -> None:
    pass
