# Tests for tipping-point-layers.yml: each import and layer rule fires on a bad sample and not on a good one.
import re
import subprocess
import unittest
from pathlib import Path

import test_rules
from test_rules import PACK, scan_matches

setUpModule = test_rules.setUpModule
REPO = PACK.parent.parent
RULES = PACK / "rules" / "tipping-point-layers.yml"
SRC = "python/tipping_point/src/tipping_point"
TOOL_DIRS = {"harness", "harness/engine-harness/tests", "notebooks", "python/tipping_point/tests/fixtures"}


def layer_hits(files):
    found = set()
    for match in scan_matches(files):
        if match["ruleId"].startswith("tp-"):
            found.add((match["file"].replace("\\", "/").removeprefix("./"), match["range"]["start"]["line"] + 1, match["ruleId"]))
    return found


class PythonLayers(unittest.TestCase):
    def assert_hits(self, files, expected):
        self.assertEqual(layer_hits(files), expected)

    def test_root_modules_import_no_subpackage(self):
        bad = "from . import ising\nfrom .ews import scoring\nimport tipping_point.jepa\nfrom tipping_point import market\n"
        good = "from . import _git\nimport os\n"
        self.assert_hits({f"{SRC}/engine.py": bad, f"{SRC}/_git.py": good}, {
            (f"{SRC}/engine.py", line, "tp-layer-root") for line in (1, 2, 3, 4)
        })

    def test_ising_imports_only_root(self):
        bad = "from ..book import features\nfrom .. import ews\nimport tipping_point.market.archive\nfrom tipping_point.jepa import train\n"
        good = "from .. import __version__, engine\nfrom .exact import TC\nfrom . import io\n"
        self.assert_hits({f"{SRC}/ising/run.py": bad, f"{SRC}/ising/pilot.py": good}, {
            (f"{SRC}/ising/run.py", line, "tp-layer-ising") for line in (1, 2, 3, 4)
        })

    def test_book_imports_below_itself(self):
        bad = "from ..market import archive\nfrom .. import (\n    __version__,\n    ews,\n)\n"
        good = "from .. import __version__\nfrom ..ising import io\n"
        self.assert_hits({f"{SRC}/book/features.py": bad, f"{SRC}/book/other.py": good}, {
            (f"{SRC}/book/features.py", 1, "tp-layer-book"),
            (f"{SRC}/book/features.py", 2, "tp-layer-book"),
        })

    def test_market_library_never_imports_ews_but_its_cli_may(self):
        cli = "from ..ews import prereg\n"
        good = "from ..ising import io\nfrom . import archive\n"
        self.assert_hits({
            f"{SRC}/market/archive.py": cli,
            f"{SRC}/market/__main__.py": cli,
            f"{SRC}/market/book.py": good,
        }, {(f"{SRC}/market/archive.py", 1, "tp-layer-market")})

    def test_ews_never_imports_jepa(self):
        bad = "from ..jepa.train import run\n"
        good = "from ..market import archive\nfrom ..book.features import DAY\nfrom .ising import Run\n"
        self.assert_hits({f"{SRC}/ews/scoring.py": bad, f"{SRC}/ews/same_input.py": good}, {
            (f"{SRC}/ews/scoring.py", 1, "tp-layer-ews")
        })

    def test_core_modules_import_no_io(self):
        bad = "\n".join([
            "import os, subprocess",
            "import pyarrow as pa",
            "from pathlib import Path",
            "from urllib.request import urlopen",
            "from .. import __version__, engine",
            "from ..ising import io",
            "from ..market.archive import fetch",
            "import tipping_point.ising.io",
            "",
        ])
        good = "import math\nimport os\nimport numpy as np\nfrom ..ising.exact import TC\nfrom .scoring import oriented\nfrom . import indicators\n"
        self.assert_hits({f"{SRC}/ews/fss.py": bad, f"{SRC}/ews/trend.py": good, f"{SRC}/ews/prereg.py": bad}, {
            (f"{SRC}/ews/fss.py", line, "tp-core-no-io") for line in range(1, 9)
        })

    def test_core_modules_in_ising_and_market_import_no_sibling_io(self):
        self.assert_hits({
            f"{SRC}/ising/autocorr.py": "from .io import read\nfrom . import exact, io\n",
            f"{SRC}/market/bars.py": "from .archive import days\nfrom . import book\n",
        }, {
            (f"{SRC}/ising/autocorr.py", 1, "tp-core-no-io"),
            (f"{SRC}/ising/autocorr.py", 2, "tp-core-no-io"),
            (f"{SRC}/market/bars.py", 1, "tp-core-no-io"),
            (f"{SRC}/market/bars.py", 2, "tp-core-no-io"),
        })

    def test_library_imports_no_tools(self):
        bad = "import pytest\nfrom jepa_toy import toy\nimport numpy as np, strays\nfrom explore.common import load\n"
        self.assert_hits({
            f"{SRC}/jepa/train.py": bad,
            "harness/m4_pilot.py": bad,
            f"{SRC}/jepa/data.py": "import numpy as np\nfrom ..ews.ising import Run\n",
        }, {(f"{SRC}/jepa/train.py", line, "tp-library-no-tools") for line in (1, 2, 3, 4)})


class RustLayers(unittest.TestCase):
    BAD = "\n".join([
        "use std::fs;",
        "use tokio::sync::mpsc;",
        "use std::{collections::BTreeMap, io::Write};",
        "use std::{env, fmt};",
        "pub fn f() -> u32 { std::process::id() }",
        "pub fn g() { let _ = reqwest::Client::new(); }",
        "",
    ])
    GOOD = "use std::collections::BTreeMap;\nuse crate::venue::Venue;\npub fn f(a: &[u8], b: &[u8]) -> usize { a.iter().zip(b).count() }\n"

    def test_core_modules_touch_no_io(self):
        hits = layer_hits({
            "engine/src/market/book.rs": self.BAD,
            "engine/src/store/run.rs": self.BAD,
            "engine/src/market/decimal.rs": self.GOOD,
        })
        self.assertEqual({(f, line) for f, line, _ in hits}, {("engine/src/market/book.rs", line) for line in range(1, 7)})
        self.assertEqual({rule for _, _, rule in hits}, {"tp-engine-core-no-io"})

    def test_engine_never_uses_the_harness(self):
        sample = "use engine_harness::skipped;\npub fn f() { engine_harness::transcript::open(); }\n"
        self.assertEqual(layer_hits({
            "engine/tests/archive.rs": sample,
            "harness/engine-harness/src/lib.rs": sample,
        }), {
            ("engine/tests/archive.rs", 1, "tp-engine-no-harness"),
            ("engine/tests/archive.rs", 2, "tp-engine-no-harness"),
        })


class RuleLists(unittest.TestCase):
    def tracked(self, *patterns):
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", *patterns], capture_output=True, text=True, check=True)
        return out.stdout.split()

    def test_every_listed_file_exists(self):
        text = RULES.read_text(encoding="utf-8")
        listed = [path for path in re.findall(r'^\s+- "([^"]+)"$', text, re.MULTILINE) if "*" not in path]
        missing = [path for path in listed if not (REPO / path).is_file()]
        self.assertEqual(missing, [], "tipping-point-layers.yml names files that do not exist; update its files lists")

    def test_tool_list_names_every_harness_notebook_and_fixture_module(self):
        text = RULES.read_text(encoding="utf-8")
        block = text[text.index("id: tp-library-no-tools"):]
        alternation = re.search(r"\(\?:(pytest\|[^)]*)\)", block)
        assert alternation is not None
        names = set(alternation.group(1).split("|"))
        stems = {Path(path).stem for path in self.tracked("*.py") if Path(path).parent.as_posix() in TOOL_DIRS}
        stems.add("explore")
        missing = sorted(stems - names)
        self.assertEqual(missing, [], "add these importable tool modules to tp-library-no-tools in tipping-point-layers.yml")


if __name__ == "__main__":
    unittest.main()
