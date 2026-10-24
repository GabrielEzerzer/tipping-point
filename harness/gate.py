"""Local gate: run every leg independently and judge each from its own evidence.

Run as `uv run python harness/gate.py [--leg NAME ...]`.

A leg is GREEN only when its exit code is 0 AND its own summary evidence is positive.
Every leg always runs; the gate exits nonzero if any leg is RED. Stdlib only.

The cargo legs cover only the root workspace's members; test_gate fails engine/ or any crate
under harness/ that is outside it, so app/src-tauri must join members and that check when it lands.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "harness" / "out"
JUNIT = OUT / "pytest-junit.xml"


@dataclass(frozen=True)
class Verdict:
    green: bool
    source: str  # the evidence line the verdict was judged on, or the reason it is red


# ---------------------------------------------------------------- pure verdict functions


def _last_line(output: str) -> str:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    return lines[-1] if lines else "(no output)"


def judge_missing_tool(tool: str) -> Verdict:
    return Verdict(False, f"tool not found on PATH: {tool}")


def judge_cargo_fmt(returncode: int, output: str) -> Verdict:
    if "Diff in" in output:
        return Verdict(False, next(ln for ln in output.splitlines() if "Diff in" in ln).strip())
    if returncode != 0:
        return Verdict(False, f"exit {returncode}: {_last_line(output)}")
    return Verdict(True, "no diff reported")


_CARGO_ERROR = re.compile(r"^(error|warning)(\[\w+\])?:", re.MULTILINE)


def judge_clippy(returncode: int, output: str) -> Verdict:
    marker = _CARGO_ERROR.search(output)
    if marker:
        line = output[marker.start() :].splitlines()[0]
        return Verdict(False, line.strip())
    if returncode != 0:
        return Verdict(False, f"exit {returncode}: {_last_line(output)}")
    finished = [ln.strip() for ln in output.splitlines() if ln.strip().startswith("Finished")]
    if not finished:
        return Verdict(False, "no 'Finished' line from cargo")
    return Verdict(True, finished[-1])


_TEST_RESULT = re.compile(
    r"test result: (?P<status>\w+)\. (?P<passed>\d+) passed; (?P<failed>\d+) failed"
)
# The line cargo prints above each test binary's output: "Running unittests src\lib.rs (...)",
# "Running tests\smoke.rs (...)", "Doc-tests engine".
_TEST_HEADER = re.compile(r"^\s*(?:Running (?P<binary>.+?)|(?P<doctests>Doc-tests \S+))\s*$")


def judge_cargo_test(returncode: int, output: str) -> Verdict:
    """Credit each 'test result:' line to the cargo header above it. Every binary must report
    ok, and every test binary except doc-tests must pass >=1 test: headers name binaries, not
    crates, so any exemption would let a crate of exempt binaries ride on another's tests. A
    binary may opt out with `test = false` only while its crate keeps a tested target: an
    opted-out binary prints no header at all, so test_gate checks that against cargo metadata."""
    # (name, is cargo's own Doc-tests header, results): the flag records which branch matched,
    # because a [[test]] at path "Doc-tests x.rs" prints "Running Doc-tests x.rs (...)"
    binaries: list[tuple[str, bool, list[re.Match[str]]]] = []
    for line in output.splitlines():
        if header := _TEST_HEADER.match(line):
            doctests = header["doctests"] is not None
            binaries.append((header["binary"] or header["doctests"], doctests, []))
        elif result := _TEST_RESULT.search(line):
            if not binaries:
                return Verdict(False, f"unparseable: no cargo header above {line.strip()!r}")
            binaries[-1][2].append(result)  # doc-tests can print two (merged + standalone)
    results = [m for *_, found in binaries for m in found]
    if not results:
        return Verdict(False, f"no 'test result:' line (exit {returncode})")
    passed = sum(int(m["passed"]) for m in results)
    failed = sum(int(m["failed"]) for m in results)
    summary = f"{len(results)} test result lines: {passed} passed; {failed} failed"
    if any(m["status"] != "ok" for m in results) or failed:
        return Verdict(False, summary)
    for name, doctests, found in binaries:
        if not found:
            return Verdict(False, f"{summary} (no 'test result:' line for {name})")
        if sum(int(m["passed"]) for m in found) < 1 and not doctests:
            return Verdict(False, f"{summary} (zero tests ran in {name})")
    if passed < 1:
        return Verdict(False, f"{summary} (zero tests ran)")
    if returncode != 0:
        return Verdict(False, f"exit {returncode} despite: {summary}")
    return Verdict(True, summary)


_FAILURE_HEADER = re.compile(r"^---- .+ ----$")  # libtest's "---- <test> stdout ----"


def cargo_test_evidence(output: str) -> dict[str, list[str]]:
    """What the record keeps of a cargo-test run besides its tail: every 'test result:' line and
    every failing test's output header, each under the binary that printed it, so a red run
    names what failed however far above the tail it sits."""
    binary, evidence = "(no header)", {"results": [], "failures": []}
    for line in output.splitlines():
        if header := _TEST_HEADER.match(line):
            binary = (header["binary"] or header["doctests"]).split(" (")[0]
        elif _TEST_RESULT.search(line):
            evidence["results"].append(f"{binary}: {line.strip()}")
        elif _FAILURE_HEADER.match(line.strip()):
            evidence["failures"].append(f"{binary}: {line.strip()}")
    return evidence


def judge_ruff_check(returncode: int, output: str) -> Verdict:
    found = re.search(r"^Found \d+ errors?.*$", output, re.MULTILINE)
    if found:
        return Verdict(False, found.group(0).strip())
    if returncode != 0:
        return Verdict(False, f"exit {returncode}: {_last_line(output)}")
    if "All checks passed!" not in output:
        return Verdict(False, f"no 'All checks passed!' line: {_last_line(output)}")
    return Verdict(True, "All checks passed!")


def judge_ruff_format(returncode: int, output: str) -> Verdict:
    marker = re.search(r"^.*(Would reformat|would be reformatted).*$", output, re.MULTILINE)
    if marker:
        return Verdict(False, marker.group(0).strip())
    if returncode != 0:
        return Verdict(False, f"exit {returncode}: {_last_line(output)}")
    ok = re.search(r"^\d+ files? already formatted$", output, re.MULTILINE)
    if not ok:
        return Verdict(False, f"no 'already formatted' line: {_last_line(output)}")
    return Verdict(True, ok.group(0))


_PYRIGHT_SUMMARY = re.compile(r"^(?P<errors>\d+) errors?, \d+ warnings?, \d+ informations?", re.M)


def judge_pyright(returncode: int, output: str) -> Verdict:
    summaries = list(_PYRIGHT_SUMMARY.finditer(output))
    if not summaries:
        return Verdict(False, f"no pyright summary line (exit {returncode}): {_last_line(output)}")
    line = output[summaries[-1].start() :].splitlines()[0].strip()
    if int(summaries[-1]["errors"]) != 0:
        return Verdict(False, line)
    if returncode != 0:
        return Verdict(False, f"exit {returncode} despite: {line}")
    return Verdict(True, line)


def judge_pytest(returncode: int, junit_xml: str | None, dirs: Sequence[str] = ()) -> Verdict:
    """Judge pytest from the JUnit XML it wrote, never from its console output. Each of `dirs`
    (repo-relative) must contribute >=1 test that ran, so one package's tests cannot hide
    another's that silently vanished."""
    if junit_xml is None:
        return Verdict(False, f"JUnit XML missing (exit {returncode})")
    try:
        root = ET.fromstring(junit_xml)
    except ET.ParseError as exc:
        return Verdict(False, f"JUnit XML unparseable: {exc}")
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        return Verdict(False, f"JUnit XML has no <testsuite> (root <{root.tag}>)")
    try:
        counts = {
            key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    except ValueError as exc:
        return Verdict(False, f"JUnit XML has non-integer counts: {exc}")
    summary = (
        f"junit: tests={counts['tests']} failures={counts['failures']} "
        f"errors={counts['errors']} skipped={counts['skipped']}"
    )
    if counts["failures"] or counts["errors"]:
        return Verdict(False, summary)
    if counts["tests"] - counts["skipped"] < 1:
        return Verdict(False, f"{summary} (zero tests ran)")
    ran = [c.get("classname", "") for c in root.iter("testcase") if c.find("skipped") is None]
    for d in dirs:
        # pytest's classname is the test file's path, dotted: python/a/test_x.py -> python.a.test_x
        dotted = d.strip("/").removesuffix(".py").replace("/", ".")
        if not any(name == dotted or name.startswith(dotted + ".") for name in ran):
            return Verdict(False, f"{summary} (zero tests ran from {d})")
    if returncode != 0:
        return Verdict(False, f"exit {returncode} despite: {summary}")
    return Verdict(True, summary)


def pytest_dirs(pyproject: Path) -> list[str]:
    """The dirs that must each contribute a test, read from the root pyproject at gate time:
    every pytest testpath, then every uv workspace member (a glob match with a pyproject.toml),
    so a package under a shared testpath like python/ cannot hide behind its siblings."""
    tool = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("tool", {})
    cfg = tool.get("pytest", {})  # pytest 9 reads native [tool.pytest] as well as ini_options
    dirs = list(cfg.get("testpaths") or cfg.get("ini_options", {}).get("testpaths", []))
    for pattern in tool.get("uv", {}).get("workspace", {}).get("members", []):
        dirs += sorted(
            p.relative_to(pyproject.parent).as_posix()
            for p in pyproject.parent.glob(pattern)
            if (p / "pyproject.toml").is_file()
        )
    return dirs


# ---------------------------------------------------------------- legs and runner


@dataclass(frozen=True)
class Leg:
    name: str
    argv: list[str]
    judge: Callable[[int, str], Verdict]
    evidence: Callable[[str], dict[str, list[str]]] | None = None  # kept beside the tail


def _judge_pytest_from_disk(returncode: int, _output: str) -> Verdict:
    text = JUNIT.read_text(encoding="utf-8") if JUNIT.is_file() else None
    return judge_pytest(returncode, text, pytest_dirs(ROOT / "pyproject.toml"))


LEGS: list[Leg] = [
    Leg("cargo-fmt", ["cargo", "fmt", "--all", "--check"], judge_cargo_fmt),
    Leg(
        "cargo-clippy",
        ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
        judge_clippy,
    ),
    # --no-fail-fast: one failing binary must not hide whether the ones after it pass
    Leg(
        "cargo-test",
        ["cargo", "test", "--workspace", "--no-fail-fast"],
        judge_cargo_test,
        cargo_test_evidence,
    ),
    Leg("ruff-check", ["ruff", "check"], judge_ruff_check),
    Leg("ruff-format", ["ruff", "format", "--check"], judge_ruff_format),
    Leg("pyright", ["pyright"], judge_pyright),
    # -c pins the config pytest_dirs reads: a pytest.toml or pytest.ini would otherwise
    # outrank pyproject.toml, and pytest would collect from dirs the floor never checks
    Leg(
        "pytest",
        ["pytest", "-c", "pyproject.toml", f"--junitxml={JUNIT}", "-q"],
        _judge_pytest_from_disk,
    ),
]


TAIL_LINES = 200  # of each leg's output, kept in the record so a red leg reads without a re-run


@dataclass(frozen=True)
class LegResult:
    leg: str
    returncode: int | None
    duration: float
    verdict: str
    verdict_source: str
    output_tail: list[str]
    evidence: dict[str, list[str]]


def run_leg(leg: Leg) -> LegResult:
    start = time.perf_counter()
    output = ""
    try:
        if leg.name == "pytest":
            JUNIT.unlink(missing_ok=True)  # a stale report must never vouch for this run
        proc = subprocess.run(
            leg.argv,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # one stream in write order: cargo's headers go to stderr
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        verdict, returncode = judge_missing_tool(leg.argv[0]), None
    except OSError as exc:  # also a stale JUnit report that cannot be deleted (held open)
        verdict, returncode = Verdict(False, f"tool could not start: {exc}"), None
    else:
        returncode, output = proc.returncode, proc.stdout
        try:
            verdict = leg.judge(proc.returncode, proc.stdout)
        except Exception as exc:  # a crashing judge is a RED leg, never a dead gate
            verdict = Verdict(False, f"judge crashed: {exc!r}")
    return LegResult(
        leg=leg.name,
        returncode=returncode,
        duration=round(time.perf_counter() - start, 2),
        verdict="GREEN" if verdict.green else "RED",
        verdict_source=verdict.source,
        output_tail=output.splitlines()[-TAIL_LINES:],
        evidence=leg.evidence(output) if leg.evidence else {},
    )


def main(argv: list[str] | None = None) -> int:
    names = [leg.name for leg in LEGS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", action="append", choices=names, help="run only this leg")
    args = parser.parse_args(argv)
    selected = [leg for leg in LEGS if not args.leg or leg.name in args.leg]

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for leg in selected:
        result = run_leg(leg)
        results.append(result)
        print(
            f"{result.leg:<13} {result.verdict:<5} {result.duration:>7.2f}s  "
            f"{result.verdict_source}",
            flush=True,
        )

    record = OUT / f"gate-{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    record.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    red = [r.leg for r in results if r.verdict != "GREEN"]
    print(f"gate {'RED: ' + ', '.join(red) if red else 'GREEN'}  (record: {record})")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
