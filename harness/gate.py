"""Local gate: run every leg independently and judge each from its own evidence.

Run as `uv run python harness/gate.py [--leg NAME ...]`.

A leg is GREEN only when its exit code is 0 AND its own summary evidence is positive.
Every leg always runs; the gate exits nonzero if any leg is RED. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
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


def judge_cargo_test(returncode: int, output: str) -> Verdict:
    results = list(_TEST_RESULT.finditer(output))
    if not results:
        return Verdict(False, f"no 'test result:' line (exit {returncode})")
    passed = sum(int(m["passed"]) for m in results)
    failed = sum(int(m["failed"]) for m in results)
    summary = f"{len(results)} test result lines: {passed} passed; {failed} failed"
    if any(m["status"] != "ok" for m in results) or failed:
        return Verdict(False, summary)
    if passed < 1:
        return Verdict(False, f"{summary} (zero tests ran)")
    if returncode != 0:
        return Verdict(False, f"exit {returncode} despite: {summary}")
    return Verdict(True, summary)


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


def judge_pytest(returncode: int, junit_xml: str | None) -> Verdict:
    """Judge pytest from the JUnit XML it wrote, never from its console output."""
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
    if returncode != 0:
        return Verdict(False, f"exit {returncode} despite: {summary}")
    return Verdict(True, summary)


# ---------------------------------------------------------------- legs and runner


@dataclass(frozen=True)
class Leg:
    name: str
    argv: list[str]
    judge: Callable[[int, str], Verdict]


def _judge_pytest_from_disk(returncode: int, _output: str) -> Verdict:
    text = JUNIT.read_text(encoding="utf-8") if JUNIT.is_file() else None
    return judge_pytest(returncode, text)


LEGS: list[Leg] = [
    Leg("cargo-fmt", ["cargo", "fmt", "--all", "--check"], judge_cargo_fmt),
    Leg(
        "cargo-clippy",
        ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
        judge_clippy,
    ),
    Leg("cargo-test", ["cargo", "test", "--workspace"], judge_cargo_test),
    Leg("ruff-check", ["ruff", "check"], judge_ruff_check),
    Leg("ruff-format", ["ruff", "format", "--check"], judge_ruff_format),
    Leg("pyright", ["pyright"], judge_pyright),
    Leg("pytest", ["pytest", f"--junitxml={JUNIT}", "-q"], _judge_pytest_from_disk),
]


@dataclass(frozen=True)
class LegResult:
    leg: str
    returncode: int | None
    duration: float
    verdict: str
    verdict_source: str


def run_leg(leg: Leg) -> LegResult:
    if leg.name == "pytest":
        JUNIT.unlink(missing_ok=True)  # a stale report must never vouch for this run
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            leg.argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        verdict, returncode = judge_missing_tool(leg.argv[0]), None
    else:
        verdict, returncode = leg.judge(proc.returncode, proc.stdout + proc.stderr), proc.returncode
    return LegResult(
        leg=leg.name,
        returncode=returncode,
        duration=round(time.perf_counter() - start, 2),
        verdict="GREEN" if verdict.green else "RED",
        verdict_source=verdict.source,
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
