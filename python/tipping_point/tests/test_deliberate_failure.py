"""Deliberately failing test: proves the gate goes red. Removed in the next commit."""

import tipping_point


def test_deliberately_fails() -> None:
    assert tipping_point.__version__ == "not-the-version", "deliberate failure"
