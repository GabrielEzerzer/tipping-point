# Tests for tipping-point-held-back.yml: only panel/heldback.py and tests pass a limit to the held-back
# refusals and data stages; elsewhere a limit is only forwarded from a parameter defaulting to HELD_BACK.
import subprocess
import unittest

import test_rules
from test_rules import PACK, scan_matches

setUpModule = test_rules.setUpModule
REPO = PACK.parent.parent
PANEL = "python/tipping_point/src/tipping_point/panel"
RULES = ("tp-held-back-limit", "tp-held-back-limit-default")

PASSING = """\
from datetime import date
from . import fetch, inputs, labels, span, store
LATER = date(2027, 1, 1)
def held(root, data_dir):
    fetch.fetch([], LATER, LATER, root, None, limit=LATER)
    labels.run(root, data_dir, [], LATER, LATER, LATER, "t", limit=LATER)
    inputs.run(root, root, data_dir, "t", limit=date(2027, 1, 1))
    store.refuse_held_back(root, limit=LATER)
    span.refuse(LATER, limit=LATER)
    span.refuse_url("x", limit=LATER)
    span.refuse_names(root, [], "holds", limit=LATER)
    span.refuse_archive(root, limit=LATER)
    span.manifest_files(root, limit=LATER)
    fetch.guarded(None, limit=LATER)
    inputs.ratios(root, [], LATER, LATER, limit=LATER)
    inputs.labels_run(root, limit=LATER)
"""

NESTED = """\
from typing import Any
from . import span, store
def run(root, *, limit: date = span.HELD_BACK):
    def helper(limit=date(2027, 1, 1)):
        return span.manifest_files(root, limit=limit)
    def typed(limit: Any = span.LATER):
        return span.refuse_archive(root, limit=limit)
    def bare(limit):
        return span.refuse(span.START, limit=limit)
    def closure(path):
        return store.refuse_held_back(path, limit=limit)
    check = lambda path, limit=span.LATER: store.refuse_held_back(path, limit=limit)
    return helper, typed, bare, closure, check
def counted(x, limit=12):
    return x
"""

WRAPPER = """\
from . import fetch, span
def stage(root, *, limit: date = span.HELD_BACK):
    return fetch.fetch([], None, None, root, None, limit=limit)
check = lambda root, limit=span.HELD_BACK: span.refuse_archive(root, limit=limit)
"""

STAGES = ("refuse", "refuse_url", "refuse_names", "refuse_archive", "manifest_files",
          "refuse_held_back", "guarded", "fetch", "run", "ratios", "labels_run")

FORWARDING = """\
from datetime import date
from . import span, store
def run(root, *, limit: date = span.HELD_BACK):
    span.refuse(date(2023, 1, 1), limit=limit)
    files = span.manifest_files(root, limit=limit)
    check = lambda path: store.refuse_held_back(path, limit=limit)
    return files, check
def refuse(*days: date, limit: date = HELD_BACK):
    return refuse_url("x", limit=limit)
def carried(x, limit: int = 12):
    return smooth(x, limit=limit)
"""


def hits(files):
    found = set()
    for match in scan_matches(files):
        if match["ruleId"] in RULES:
            found.add((match["file"].replace("\\", "/").removeprefix("./"), match["range"]["start"]["line"] + 1, match["ruleId"]))
    return found


class HeldBackLimit(unittest.TestCase):
    def test_a_development_module_passing_a_limit_is_refused(self):
        found = hits({f"{PANEL}/ladder.py": PASSING, "harness/panel_tool.py": PASSING})
        self.assertEqual(found, {
            (name, line, "tp-held-back-limit")
            for name in (f"{PANEL}/ladder.py", "harness/panel_tool.py")
            for line in range(5, 17)
        })

    def test_heldback_and_tests_may_pass_a_limit(self):
        self.assertEqual(hits({
            f"{PANEL}/heldback.py": PASSING,
            "python/tipping_point/tests/panel/test_held_back.py": PASSING,
            "harness/tests/test_panel.py": PASSING,
        }), set())

    def test_a_limit_is_forwarded_only_from_a_parameter_defaulting_to_held_back(self):
        raised = FORWARDING.replace("limit: date = span.HELD_BACK", "limit: date = date(2027, 1, 1)")
        local = "from . import span\nlimit = span.LATER\nspan.refuse(span.START, limit=limit)\n"
        bare = "from . import span\ndef stage(root, limit):\n    span.manifest_files(root, limit=limit)\n"
        self.assertEqual(hits({
            f"{PANEL}/fetch.py": FORWARDING,
            f"{PANEL}/labels.py": raised,
            f"{PANEL}/inputs.py": local,
            f"{PANEL}/store.py": bare,
        }), {
            (f"{PANEL}/labels.py", 3, "tp-held-back-limit-default"),
            (f"{PANEL}/labels.py", 4, "tp-held-back-limit"),
            (f"{PANEL}/labels.py", 5, "tp-held-back-limit"),
            (f"{PANEL}/labels.py", 6, "tp-held-back-limit"),
            (f"{PANEL}/inputs.py", 3, "tp-held-back-limit"),
            (f"{PANEL}/store.py", 3, "tp-held-back-limit"),
        })

    def test_every_listed_stage_may_default_its_limit_to_held_back(self):
        source = "from . import span\n" + "".join(
            f"def {name}(x, *, limit: date = span.HELD_BACK):\n    return x\n" for name in STAGES
        )
        self.assertEqual(hits({f"{PANEL}/span.py": source}), set())

    def test_a_wrapper_defaulting_its_limit_to_held_back_is_refused(self):
        self.assertEqual(hits({"harness/panel_tool.py": WRAPPER}), {
            ("harness/panel_tool.py", 2, "tp-held-back-limit-default"),
            ("harness/panel_tool.py", 4, "tp-held-back-limit-default"),
            ("harness/panel_tool.py", 4, "tp-held-back-limit"),
        })

    def test_the_nearest_function_binding_limit_decides_what_is_forwarded(self):
        self.assertEqual(hits({f"{PANEL}/inputs.py": NESTED}), {
            (f"{PANEL}/inputs.py", 4, "tp-held-back-limit-default"),
            (f"{PANEL}/inputs.py", 5, "tp-held-back-limit"),
            (f"{PANEL}/inputs.py", 6, "tp-held-back-limit-default"),
            (f"{PANEL}/inputs.py", 7, "tp-held-back-limit"),
            (f"{PANEL}/inputs.py", 9, "tp-held-back-limit"),
            (f"{PANEL}/inputs.py", 12, "tp-held-back-limit-default"),
            (f"{PANEL}/inputs.py", 12, "tp-held-back-limit"),
            (f"{PANEL}/inputs.py", 14, "tp-held-back-limit-default"),
        })

    def test_the_repository_passes(self):
        proc = subprocess.run(
            ["ast-grep", "scan", "-c", str(PACK / "sgconfig.yml"), "--json=compact", "python", "harness", "notebooks"],
            cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        )
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        self.assertNotIn('"ruleId":"tp-held-back-limit', proc.stdout)


if __name__ == "__main__":
    unittest.main()
