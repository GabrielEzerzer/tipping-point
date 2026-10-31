"""What the ising-ews v1 scorer gives on planted data at aa29cbc, the freeze lane's merge, so that
a later commit can be held to it: the same rows, the same values, in the same order.

score() runs ising.evaluate on the planted v1 dataset (fss mode) under the committed
paper/prereg/ising-ews-v1.toml. The file's settings change only as far as the planted data need
(seeds, replicas, the tau_c record, the tau.json pins) and to run in seconds: 40 bootstrap draws,
one surrogate, alarm scores every 200 tau_c, one window, two bandwidths and two horizons of the
grid. Both observables, cadences and surrogate tests stay, so every loop of the scorer turns; and
every draw of the one RNG feeds the CIs and surrogate rates of the rows after it, so a scorer that
draws once more, once less or in another order moves them.

The golden file was written by this script from aa29cbc, before any latent code existed. To check
it again, run the script on that commit's tree (nothing here needs anything newer), never on a later
one: rewriting the golden from the code it checks would prove nothing.

  uv run python harness/ews_v1_golden.py [--out FILE]
"""

import argparse
import hashlib
import json
import math
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from ews_planted import V1, V1_PILOT_RUN, V1_REPLICAS, V1_SEEDS, V1_SIZES, tau_source, v1_dataset
from tipping_point.ews import ising
from tipping_point.ising import io

REPO = Path(__file__).resolve().parents[1]
V1_TOML = REPO / "paper" / "prereg" / "ising-ews-v1.toml"
GOLDEN = REPO / "python" / "tipping_point" / "tests" / "fixtures" / "ews_v1_golden.json"
SEED = 26  # the RNG seed `ews score` defaults to
TOLERANCE = 1e-9  # relative: rounding between machines, far below any value a draw could move


def v1_doc(data: V1) -> dict[str, Any]:
    """The committed v1 file, set to score `data` as the module docstring says."""
    doc = tomllib.loads(V1_TOML.read_text("utf-8"))
    record = hashlib.sha256(data.pilot.read_bytes()).hexdigest()
    doc["data"] |= {"seeds": list(V1_SEEDS), "replicas": V1_REPLICAS}
    doc["data"]["tau_c"] |= {"pilot_run": V1_PILOT_RUN, "pilot_sha256": record}
    doc["data"]["staircase"]["tau_sha256"] = {str(L): tau_source(L) for L in V1_SIZES}
    doc["scoring"]["bootstrap"] = 40
    doc["scoring"]["surrogates"]["n"] = 1
    doc["scoring"]["alarms"]["every"] = 200.0
    doc["sensitivity"] |= {"window": [20.0], "bandwidth": [5.0, 10.0]}
    doc["event"]["horizons"] = [0.0, 2.0]
    return doc


def plain(value: Any) -> Any:
    """A row's value as JSON holds it: numpy scalars as Python's, NaN as null."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and math.isnan(value) else value


def score(data: V1) -> dict[str, Any]:
    """Everything a v1 scoring computes on `data`, in plain values, every row as it comes."""
    rng = np.random.default_rng(SEED)
    evaluation = ising.evaluate(v1_doc(data), data.root, rng, say=lambda _: None)
    return plain(
        {
            "tables": evaluation.tables,
            "dropped": evaluation.dropped,
            "inputs": [r.run_id for r in evaluation.inputs],
        }
    )


def differences(golden: Any, found: Any, where: str = "") -> list[str]:
    """Where `found` departs from `golden`: a float must be close, anything else equal."""
    if isinstance(golden, dict) and isinstance(found, dict):
        missing = [f"{where}.{k}: one side lacks it" for k in golden.keys() ^ found.keys()]
        both = sorted(golden.keys() & found.keys())
        return missing + [d for k in both for d in differences(golden[k], found[k], f"{where}.{k}")]
    if isinstance(golden, list) and isinstance(found, list):
        if len(golden) != len(found):
            return [f"{where}: {len(found)} rows, the golden holds {len(golden)}"]
        pairs = zip(golden, found, strict=True)
        return [d for i, (g, f) in enumerate(pairs) for d in differences(g, f, f"{where}[{i}]")]
    if isinstance(golden, float) and isinstance(found, float):
        same = math.isclose(golden, found, rel_tol=TOLERANCE, abs_tol=TOLERANCE * 1e-3)
    else:
        same = golden == found and type(golden) is type(found)
    return [] if same else [f"{where}: {found!r}, the golden holds {golden!r}"]


def write(path: Path) -> None:
    """One table row per line, so that a diff of the file reads row by row."""
    with tempfile.TemporaryDirectory() as root:
        scored = score(v1_dataset(Path(root), "fss"))
    scored["made_by"] = io.code_version()["git_sha"]
    lines = ["{"]
    for key in ("made_by", "dropped", "inputs"):
        lines.append(f"{json.dumps(key)}: {json.dumps(scored[key], allow_nan=False)},")
    lines.append('"tables": {')
    tables = []
    for name, rows in scored["tables"].items():
        body = ",\n".join(json.dumps(row, allow_nan=False) for row in rows)
        tables.append(f"{json.dumps(name)}: [\n{body}\n]")
    lines.append(",\n".join(tables))
    lines += ["}", "}", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=GOLDEN)
    args = parser.parse_args(argv)
    write(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
