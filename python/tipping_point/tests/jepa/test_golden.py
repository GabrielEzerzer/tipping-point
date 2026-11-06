"""The Ising training path is bit-identical to the trainer the frozen Ising runs came from (#30 M3):
the golden (fixtures/jepa_golden.py) was recorded before train.py was split into a loop and its
domains, and the same fixed-seed CPU training must reproduce every loss term, every monitor value,
the weights and the PCA bases exactly."""

import json
from pathlib import Path

import jepa_golden
import pytest


def test_the_ising_training_reproduces_its_golden_bit_for_bit(tmp_path: Path) -> None:
    """Catches any change to the Ising code path: an op reordered, a random draw moved between
    streams, a schedule or monitor computed differently, a table or manifest field renamed."""
    record = json.loads(jepa_golden.OUT.read_text("utf-8"))
    here = jepa_golden.environment()
    if record["environment"] != here:
        pytest.skip(f"golden recorded on {record['environment']}, this is {here}")
    got = jepa_golden.compute(tmp_path)
    golden = record["golden"]
    assert got.keys() == golden.keys()
    for key in golden:
        assert got[key] == golden[key], key
