"""The gate must not lie: every lie case judges RED, the honest case judges GREEN.

Fixtures in fixtures/ are real tool output captured on this repo (green = clean tree,
red = deliberately broken files).
"""

import json
import sys
from pathlib import Path

import gate
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# (leg judge, fixture stem) for every text-judged leg
TEXT_LEGS = [
    (gate.judge_cargo_fmt, "cargo-fmt"),
    (gate.judge_clippy, "cargo-clippy"),
    (gate.judge_cargo_test, "cargo-test"),
    (gate.judge_ruff_check, "ruff-check"),
    (gate.judge_ruff_format, "ruff-format"),
    (gate.judge_pyright, "pyright"),
]
IDS = [stem for _, stem in TEXT_LEGS]


# ---------- the honest cases


@pytest.mark.parametrize(("judge", "stem"), TEXT_LEGS, ids=IDS)
def test_honest_green_is_green(judge, stem) -> None:
    verdict = judge(0, fixture(f"{stem}-green.txt"))
    assert verdict.green, verdict.source


@pytest.mark.parametrize(("judge", "stem"), TEXT_LEGS, ids=IDS)
def test_honest_red_is_red(judge, stem) -> None:
    assert not judge(1, fixture(f"{stem}-red.txt")).green


def test_honest_pytest_green_is_green() -> None:
    verdict = gate.judge_pytest(0, fixture("junit-green.xml"))
    assert verdict.green, verdict.source
    assert "tests=1" in verdict.source


def test_green_evidence_is_the_judged_line() -> None:
    assert gate.judge_cargo_test(0, fixture("cargo-test-green.txt")).source == (
        "2 test result lines: 1 passed; 0 failed"
    )
    assert gate.judge_pyright(0, fixture("pyright-green.txt")).source == (
        "0 errors, 0 warnings, 0 informations"
    )


# ---------- lie: exit 0, failing summary


@pytest.mark.parametrize(("judge", "stem"), TEXT_LEGS, ids=IDS)
def test_exit_zero_with_failing_summary_is_red(judge, stem) -> None:
    assert not judge(0, fixture(f"{stem}-red.txt")).green


def test_pytest_exit_zero_with_failing_junit_is_red() -> None:
    verdict = gate.judge_pytest(0, fixture("junit-red.xml"))
    assert not verdict.green
    assert "failures=1" in verdict.source


# ---------- lie: failing exit, passing summary


@pytest.mark.parametrize(("judge", "stem"), TEXT_LEGS, ids=IDS)
def test_failing_exit_with_passing_summary_is_red(judge, stem) -> None:
    verdict = judge(1, fixture(f"{stem}-green.txt"))
    assert not verdict.green
    assert "exit 1" in verdict.source


def test_pytest_failing_exit_with_passing_junit_is_red() -> None:
    assert not gate.judge_pytest(1, fixture("junit-green.xml")).green


# ---------- lie: zero tests


ZERO_CARGO = """\
running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests engine

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"""


def test_cargo_zero_tests_is_red() -> None:
    verdict = gate.judge_cargo_test(0, ZERO_CARGO)
    assert not verdict.green
    assert "zero tests" in verdict.source


def test_cargo_one_failed_line_among_ok_lines_is_red() -> None:
    output = fixture("cargo-test-green.txt") + (
        "test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    assert not gate.judge_cargo_test(0, output).green


JUNIT_ZERO = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
    'errors="0" failures="0" skipped="0" tests="0" time="0.01"/></testsuites>'
)
JUNIT_ALL_SKIPPED = JUNIT_ZERO.replace('skipped="0" tests="0"', 'skipped="2" tests="2"')


@pytest.mark.parametrize("xml", [JUNIT_ZERO, JUNIT_ALL_SKIPPED], ids=["none", "all-skipped"])
def test_pytest_zero_tests_ran_is_red(xml: str) -> None:
    verdict = gate.judge_pytest(0, xml)
    assert not verdict.green
    assert "zero tests" in verdict.source


def test_pytest_collection_error_is_red() -> None:
    xml = JUNIT_ZERO.replace('errors="0"', 'errors="1"').replace('tests="0"', 'tests="1"')
    assert not gate.judge_pytest(0, xml).green


# ---------- lie: missing or garbage JUnit


@pytest.mark.parametrize(
    "xml",
    [
        None,
        "",
        "this is not xml",
        "<testsuites></testsuites>",
        "<html><body>ok</body></html>",
        '<testsuites><testsuite tests="many" failures="0" errors="0"/></testsuites>',
    ],
    ids=["missing", "empty", "garbage", "no-suites", "wrong-root", "non-integer"],
)
def test_bad_junit_is_red(xml: str | None) -> None:
    assert not gate.judge_pytest(0, xml).green


def test_pytest_leg_reads_junit_from_disk_and_missing_file_is_red(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "JUNIT", tmp_path / "absent.xml")
    # console output claiming success is ignored: only the file on disk counts
    assert not gate._judge_pytest_from_disk(0, "1 passed in 0.01s").green


# ---------- lie: missing cargo test result line


@pytest.mark.parametrize(
    "output",
    ["", fixture("cargo-clippy-red.txt"), "running 1 test\ntest tests::x ... ok\n"],
    ids=["empty", "compile-error", "truncated"],
)
def test_cargo_test_without_result_line_is_red(output: str) -> None:
    verdict = gate.judge_cargo_test(0, output)
    assert not verdict.green
    assert "no 'test result:' line" in verdict.source


def test_pyright_without_summary_line_is_red() -> None:
    assert not gate.judge_pyright(0, "").green


@pytest.mark.parametrize(
    ("judge", "output"),
    [(gate.judge_ruff_check, ""), (gate.judge_ruff_format, ""), (gate.judge_clippy, "")],
    ids=["ruff-check", "ruff-format", "clippy"],
)
def test_silent_success_without_positive_evidence_is_red(judge, output: str) -> None:
    assert not judge(0, output).green


# ---------- lie: tool not found


def test_missing_tool_is_red_not_a_crash() -> None:
    leg = gate.Leg("ghost", ["definitely-not-a-real-tool-tp20"], gate.judge_ruff_check)
    result = gate.run_leg(leg)
    assert result.verdict == "RED"
    assert result.returncode is None
    assert "tool not found" in result.verdict_source


# ---------- the runner


def _python_leg(name: str, code: str) -> gate.Leg:
    return gate.Leg(name, [sys.executable, "-c", code], gate.judge_ruff_check)


def test_all_legs_run_after_a_failure_and_gate_exits_nonzero(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(
        gate,
        "LEGS",
        [
            _python_leg("first", "raise SystemExit(1)"),
            gate.Leg("ghost", ["definitely-not-a-real-tool-tp20"], gate.judge_ruff_check),
            _python_leg("last", "print('All checks passed!')"),
        ],
    )
    assert gate.main([]) == 1
    (record,) = tmp_path.glob("gate-*.json")
    rows = json.loads(record.read_text(encoding="utf-8"))
    assert [(r["leg"], r["verdict"]) for r in rows] == [
        ("first", "RED"),
        ("ghost", "RED"),
        ("last", "GREEN"),
    ]
    assert set(rows[0]) == {"leg", "returncode", "duration", "verdict", "verdict_source"}


def test_gate_exits_zero_only_when_all_green(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "LEGS", [_python_leg("ok", "print('All checks passed!')")])
    assert gate.main([]) == 0


# ---------- lie: a tool that cannot start, a judge that crashes


def test_unstartable_tool_is_red_and_the_rest_still_run(tmp_path, monkeypatch) -> None:
    broken = tmp_path / "broken.exe"
    broken.write_bytes(b"")  # on PATH-equivalent but not a runnable program
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(
        gate,
        "LEGS",
        [
            gate.Leg("broken", [str(broken)], gate.judge_ruff_check),
            _python_leg("last", "print('All checks passed!')"),
        ],
    )
    assert gate.main([]) == 1
    (record,) = tmp_path.glob("gate-*.json")
    rows = json.loads(record.read_text(encoding="utf-8"))
    assert [(r["leg"], r["verdict"]) for r in rows] == [("broken", "RED"), ("last", "GREEN")]
    assert rows[0]["returncode"] is None
    assert "could not start" in rows[0]["verdict_source"]


def test_non_utf8_junit_is_red_not_a_crash(monkeypatch) -> None:
    monkeypatch.setattr(gate, "JUNIT", FIXTURES / "junit-non-utf8.xml")
    leg = gate.Leg("junit", [sys.executable, "-c", "pass"], gate._judge_pytest_from_disk)
    result = gate.run_leg(leg)
    assert result.verdict == "RED"
    assert "judge crashed" in result.verdict_source
