"""The gate must not lie: every lie case judges RED, the honest case judges GREEN.

Fixtures in fixtures/ are real tool output captured on this repo (green = clean tree,
red = deliberately broken files; any other name = files deliberately edited into that shape). The
self-test-* ones have their local repo and temp prefixes replaced, because the repo is public.
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import gate
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REAL_MACHINE = gate.machine


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def no_real_sweep_or_cpu_sample(monkeypatch) -> None:
    """main() kills strays and samples the CPU for a second: here it does neither, so no test
    kills a real process."""
    monkeypatch.setattr(gate.strays, "sweep", lambda root: [])
    monkeypatch.setattr(gate, "machine", lambda: {"cores": 8, "cpu_busy_percent": 12.5})


def _record(out: Path) -> dict:
    (record,) = out.glob("gate-*.json")
    return json.loads(record.read_text(encoding="utf-8"))


# (leg judge, fixture stem) for every text-judged leg
TEXT_LEGS = [
    (gate.judge_cargo_fmt, "cargo-fmt"),
    (gate.judge_clippy, "cargo-clippy"),
    (gate.judge_cargo_test, "cargo-test"),
    (gate.judge_ruff_check, "ruff-check"),
    (gate.judge_ruff_format, "ruff-format"),
    (gate.judge_pyright, "pyright"),
    (gate.judge_self_test, "self-test"),
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
    assert gate.judge_self_test(0, fixture("self-test-green.txt")).source == (
        "245/245 checks passed"
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
     Running unittests src/lib.rs (target/debug/deps/engine-0)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests engine

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"""
ZERO_CARGO_BINS_ONLY = """\
     Running unittests src/bin/record.rs (target/debug/deps/record-0)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"""


@pytest.mark.parametrize("output", [ZERO_CARGO, ZERO_CARGO_BINS_ONLY], ids=["lib", "bins-only"])
def test_cargo_zero_tests_is_red(output: str) -> None:
    verdict = gate.judge_cargo_test(0, output)
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


# ---------- lie: one crate's or package's tests vanish behind another's (#39)


def test_cargo_every_test_binary_passing_is_green() -> None:
    verdict = gate.judge_cargo_test(0, fixture("cargo-test-two-binaries-green.txt"))
    assert verdict.green, verdict.source


def test_cargo_lib_binary_with_zero_tests_is_red() -> None:
    # engine's unit tests deleted; its integration test alone keeps the workspace at 1 passed
    verdict = gate.judge_cargo_test(0, fixture("cargo-test-lib-zero.txt"))
    assert not verdict.green
    assert "zero tests ran in unittests src" in verdict.source
    assert "lib.rs" in verdict.source


WORKSPACE = fixture("cargo-test-workspace-green.txt")  # every crate after the #21 merge
WORKSPACE_BINARIES = [
    ln.strip().removeprefix("Running ")
    for ln in WORKSPACE.splitlines()
    if ln.strip().startswith("Running ")
]


def test_cargo_whole_workspace_passing_is_green() -> None:
    verdict = gate.judge_cargo_test(0, WORKSPACE)
    assert verdict.green, verdict.source


@pytest.mark.parametrize(
    "binary", WORKSPACE_BINARIES, ids=lambda b: b.split(" (")[0].replace("\\", "/")
)
def test_cargo_any_one_binary_with_zero_tests_is_red(binary: str) -> None:
    # headers name binaries, not crates: no binary may ride on another's tests, so a bins-only
    # crate (engine-harness: record, soak) cannot hide behind engine's
    head, header, tail = WORKSPACE.partition(binary)
    tail = re.sub(r"ok\. \d+ passed", "ok. 0 passed", tail, count=1)
    verdict = gate.judge_cargo_test(0, head + header + tail)
    assert not verdict.green
    assert f"zero tests ran in {binary}" in verdict.source


def test_cargo_test_binary_named_like_doc_tests_is_not_exempt() -> None:
    # real header of a [[test]] at path "Doc-tests sweep.rs": only cargo's own Doc-tests line
    # is exempt from the floor, not a binary whose name merely reads like it
    spoof = (
        "\n     Running Doc-tests sweep.rs (target\\debug\\deps\\sweep-a70bfbc7b205fadb.exe)\n\n"
        "running 0 tests\n\n"
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    verdict = gate.judge_cargo_test(0, fixture("cargo-test-two-binaries-green.txt") + spoof)
    assert not verdict.green
    assert "zero tests ran in Doc-tests sweep.rs" in verdict.source


def test_cargo_doc_tests_may_print_two_result_lines() -> None:
    # edition 2024 runs merged doctests, then standalone (compile_fail) ones, under one header
    verdict = gate.judge_cargo_test(0, fixture("cargo-test-doctests-green.txt"))
    assert verdict.green, verdict.source
    assert verdict.source == "2 test result lines: 2 passed; 0 failed"


def test_cargo_test_result_without_its_header_is_red() -> None:
    lines = fixture("cargo-test-two-binaries-green.txt").splitlines(keepends=True)
    verdict = gate.judge_cargo_test(0, "".join(ln for ln in lines if "Running unittests" not in ln))
    assert not verdict.green
    assert "unparseable" in verdict.source


def test_cargo_integration_binary_with_zero_tests_is_red() -> None:
    output = fixture("cargo-test-two-binaries-green.txt")
    head, header, tail = output.partition("Running tests\\smoke.rs")
    tail = tail.replace("ok. 1 passed", "ok. 0 passed", 1)  # smoke's own result line
    verdict = gate.judge_cargo_test(0, head + header + tail)
    assert not verdict.green
    assert "zero tests ran in tests\\smoke.rs" in verdict.source


def test_cargo_binary_without_test_result_is_red() -> None:
    output = fixture("cargo-test-two-binaries-green.txt")
    cut = output.index("\n", output.index("Running tests")) + 1  # stops after smoke's header
    verdict = gate.judge_cargo_test(0, output[:cut])
    assert not verdict.green
    assert "no 'test result:' line for tests" in verdict.source


def test_cargo_headers_on_stderr_stay_above_their_results() -> None:
    # cargo prints each binary's header on stderr; libtest prints its result on stdout
    code = (
        "import sys; "
        "print('     Running tests/smoke.rs (target/debug/deps/smoke-0)', file=sys.stderr, "
        "flush=True); "
        "print('test result: ok. 1 passed; 0 failed; 0 ignored', flush=True)"
    )
    leg = gate.Leg("cargo-test", [sys.executable, "-c", code], gate.judge_cargo_test)
    result = gate.run_leg(leg)
    assert result.verdict == "GREEN", result.verdict_source


JUNIT_HARNESS_ALL_SKIPPED = (
    fixture("junit-green.xml")
    .replace('skipped="0" tests="1"', 'skipped="1" tests="2"')
    .replace(
        "</testsuite>",
        '<testcase classname="harness.tests.test_gate" name="test_x" time="0.0">'
        '<skipped type="pytest.skip" message="no torch">skipped</skipped></testcase></testsuite>',
    )
)


@pytest.mark.parametrize(
    "xml", [fixture("junit-green.xml"), JUNIT_HARNESS_ALL_SKIPPED], ids=["none", "all-skipped"]
)
def test_pytest_dir_contributing_zero_tests_is_red(xml: str) -> None:
    verdict = gate.judge_pytest(0, xml, ["python", "harness/tests"])
    assert not verdict.green
    assert "zero tests ran from harness/tests" in verdict.source


def test_pytest_sibling_sharing_a_name_prefix_does_not_count() -> None:
    # python/tipping_point_extra's tests must not vouch for python/tipping_point
    xml = fixture("junit-green.xml").replace(
        'classname="python.tipping_point.tests.test_smoke"',
        'classname="python.tipping_point_extra.tests.test_x"',
    )
    verdict = gate.judge_pytest(0, xml, ["python/tipping_point"])
    assert not verdict.green
    assert "zero tests ran from python/tipping_point" in verdict.source


def test_pytest_every_dir_contributing_is_green() -> None:
    verdict = gate.judge_pytest(0, fixture("junit-green.xml"), ["python", "python/tipping_point"])
    assert verdict.green, verdict.source


def test_pytest_dirs_are_testpaths_then_uv_members(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    for name in ("b", "a"):
        (tmp_path / "libs" / name).mkdir(parents=True)
        (tmp_path / "libs" / name / "pyproject.toml").touch()
    (tmp_path / "libs" / "notes").mkdir()  # no pyproject.toml: not a uv member
    members = '[tool.uv.workspace]\nmembers = ["libs/*"]\n'
    pyproject.write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["libs", "harness/tests"]\n' + members,
        encoding="utf-8",
    )
    assert gate.pytest_dirs(pyproject) == ["libs", "harness/tests", "libs/a", "libs/b"]
    pyproject.write_text(members, encoding="utf-8")
    assert gate.pytest_dirs(pyproject) == ["libs/a", "libs/b"]


def test_pytest_dirs_read_native_toml_testpaths(tmp_path) -> None:
    # pytest 9 also takes [tool.pytest]; moving testpaths there must not empty the list
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.pytest]\ntestpaths = ["libs", "harness/tests"]\n', encoding="utf-8")
    assert gate.pytest_dirs(pyproject) == ["libs", "harness/tests"]


def test_pytest_leg_pins_the_config_pytest_dirs_reads(tmp_path) -> None:
    # pytest.toml outranks pyproject.toml: unpinned, pytest collects b/ while the floor checks a/
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["a"]\n', encoding="utf-8"
    )
    (tmp_path / "pytest.toml").write_text('[pytest]\ntestpaths = ["b"]\n', encoding="utf-8")
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        test_file = tmp_path / name / f"test_{name}.py"
        test_file.write_text(f"def test_{name}():\n    pass\n", encoding="utf-8")
    (leg,) = [leg for leg in gate.LEGS if leg.name == "pytest"]
    junit = tmp_path / "junit.xml"
    argv = [f"--junitxml={junit}" if a.startswith("--junitxml=") else a for a in leg.argv]
    proc = subprocess.run(argv, cwd=tmp_path, capture_output=True, check=False)
    dirs = gate.pytest_dirs(tmp_path / "pyproject.toml")
    assert dirs == ["a"]
    verdict = gate.judge_pytest(proc.returncode, junit.read_text(encoding="utf-8"), dirs)
    assert verdict.green, verdict.source
    assert leg.argv[leg.argv.index("-c") + 1] == "pyproject.toml"


def test_pytest_leg_reads_the_dirs_from_the_real_pyproject(monkeypatch) -> None:
    # real JUnit from when only the python package had tests: harness/tests contributed none
    monkeypatch.setattr(gate, "JUNIT", FIXTURES / "junit-green.xml")
    verdict = gate._judge_pytest_from_disk(0, "")
    assert not verdict.green
    assert "zero tests ran from harness/tests" in verdict.source


# ---------- what pyright reads (#117)


def test_pyright_reads_every_python_file_the_repository_tracks() -> None:
    # notebooks/ sat outside include, so a type error in the Kaggle runner first showed on Kaggle;
    # an exclude entry over it brings that back, with pyright still reporting 0 errors
    pyproject = tomllib.loads((gate.ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=gate.ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.splitlines()
    assert "notebooks/kaggle_ising_runs.py" in tracked
    include, exclude = (pyproject["tool"]["pyright"][key] for key in ("include", "exclude"))
    assert [f for f in tracked if not any(Path(f).is_relative_to(d) for d in include)] == []
    # shortcut: is_relative_to reads a glob entry (notebooks/**) as no path, so it would go unseen;
    # use full_match once an entry other than **/.venv holds a wildcard
    assert [f for f in tracked if any(Path(f).is_relative_to(d) for d in exclude)] == []


# ---------- the Cargo workspace, as cargo itself resolves it


def _cargo_packages(root: Path) -> list[dict]:
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--offline"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["packages"]


def _workspace_problems(root: Path) -> list[str]:
    """Ask cargo, never re-derive its rules: engine/ and every crate dir under harness/ must be a
    package it builds, every package must keep a target that runs tests (a `test = false` binary
    prints no header, so the cargo-test judge cannot see a crate whose targets all opted out),
    and no target may need features that `cargo test --workspace` never enables."""
    packages = _cargo_packages(root)
    built = {Path(p["manifest_path"]).resolve().parent for p in packages}
    manifests = [root / "engine" / "Cargo.toml", *(root / "harness").rglob("Cargo.toml")]
    crates = sorted(m.resolve().parent for m in manifests)
    problems = [
        f"{c.relative_to(root.resolve()).as_posix()} is not a workspace member"
        for c in crates
        if c not in built
    ]
    problems += [
        f"{p['name']} has no target that runs tests"
        for p in packages
        if not any(t["test"] for t in p["targets"])
    ]
    problems += [
        f"{p['name']}: {t['name']} needs features {t['required-features']}, never enabled"
        for p in packages
        for t in p["targets"]
        if t.get("required-features")
    ]
    return problems


def _scratch_workspace(root: Path) -> Path:
    """The real manifests with empty sources: enough for cargo metadata, nothing to build."""
    for manifest in ("Cargo.toml", "engine/Cargo.toml", "harness/engine-harness/Cargo.toml"):
        (root / manifest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gate.ROOT / manifest, root / manifest)
    for source in (
        "engine/src/lib.rs",
        "harness/engine-harness/src/bin/record.rs",
        "harness/engine-harness/src/bin/soak.rs",
    ):
        (root / source).parent.mkdir(parents=True, exist_ok=True)
        (root / source).touch()
    return root


def _add_crate(crate: Path, source: str = "lib.rs", targets: str = "") -> None:
    (crate / "src").mkdir(parents=True)
    (crate / "src" / source).touch()
    (crate / "Cargo.toml").write_text(
        f'[package]\nname = "{crate.name}"\nversion = "0.1.0"\nedition = "2024"\n{targets}',
        encoding="utf-8",
    )


def test_every_harness_crate_is_a_workspace_package_that_runs_tests() -> None:
    assert _workspace_problems(gate.ROOT) == []


def test_cargo_members_skip_non_crate_dirs_under_harness(tmp_path) -> None:
    # record writes harness/fixtures/recorded/; dot-dirs, out/ and tests/ sit there too
    root = _scratch_workspace(tmp_path)
    for path in ("fixtures/recorded/x.jsonl", ".hidden/x", "out/x", "tests/x"):
        (root / "harness" / path).parent.mkdir(parents=True, exist_ok=True)
        (root / "harness" / path).touch()
    assert sorted(p["name"] for p in _cargo_packages(root)) == ["engine", "engine-harness"]
    assert _workspace_problems(root) == []


def test_an_excluded_harness_crate_is_caught(tmp_path) -> None:
    root = _scratch_workspace(tmp_path)
    _add_crate(root / "harness" / "ising-harness")
    manifest = root / "Cargo.toml"
    excluded = '[workspace]\nexclude = ["harness/ising-harness"]\n'
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("[workspace]\n", excluded, 1),
        encoding="utf-8",
    )
    assert _workspace_problems(root) == ["harness/ising-harness is not a workspace member"]


def test_a_nested_harness_crate_is_caught(tmp_path) -> None:
    root = _scratch_workspace(tmp_path)
    _add_crate(root / "harness" / "tools" / "ising-harness")
    assert _workspace_problems(root) == ["harness/tools/ising-harness is not a workspace member"]


def test_a_crate_whose_every_target_skips_tests_is_caught(tmp_path) -> None:
    root = _scratch_workspace(tmp_path)
    no_tests = '\n[[bin]]\nname = "ising"\npath = "src/main.rs"\ntest = false\n'
    _add_crate(root / "harness" / "ising-harness", "main.rs", no_tests)
    assert _workspace_problems(root) == ["ising-harness has no target that runs tests"]


def test_a_crate_whose_only_target_needs_features_is_caught(tmp_path) -> None:
    # cargo test --workspace enables no features, so this target is never built or run
    root = _scratch_workspace(tmp_path)
    gated = (
        '\n[features]\nsim = []\n\n[[bin]]\nname = "ising"\npath = "src/main.rs"\n'
        'required-features = ["sim"]\n'
    )
    _add_crate(root / "harness" / "ising-harness", "main.rs", gated)
    assert _workspace_problems(root) == [
        "ising-harness: ising needs features ['sim'], never enabled"
    ]


def test_engine_outside_the_workspace_is_caught(tmp_path) -> None:
    # exclude alone keeps an explicitly listed member, and engine-harness's path dependency
    # pulls engine back in, so engine leaves members and is excluded
    root = _scratch_workspace(tmp_path)
    (root / "engine" / "Cargo.toml").write_text(
        '[package]\nname = "engine"\nversion = "0.1.0"\nedition = "2024"\n', encoding="utf-8"
    )
    manifest = root / "Cargo.toml"
    listed = 'members = ["engine", "harness/*-harness"]'
    text = manifest.read_text(encoding="utf-8")
    assert listed in text
    dropped = 'members = ["harness/*-harness"]\nexclude = ["engine"]'
    manifest.write_text(text.replace(listed, dropped), encoding="utf-8")
    assert _workspace_problems(root) == ["engine is not a workspace member"]


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
    [
        (gate.judge_ruff_check, ""),
        (gate.judge_ruff_format, ""),
        (gate.judge_clippy, ""),
        (gate.judge_self_test, ""),
    ],
    ids=["ruff-check", "ruff-format", "clippy", "self-test"],
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
    rows = _record(tmp_path)["legs"]
    assert [(r["leg"], r["verdict"]) for r in rows] == [
        ("first", "RED"),
        ("ghost", "RED"),
        ("last", "GREEN"),
    ]
    assert set(rows[0]) == {
        "leg",
        "returncode",
        "duration",
        "verdict",
        "verdict_source",
        "output_tail",
        "evidence",
    }


def test_gate_exits_zero_only_when_all_green(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "LEGS", [_python_leg("ok", "print('All checks passed!')")])
    assert gate.main([]) == 0


# ---------- the record diagnoses a red leg without a re-run (#59)


def test_cargo_test_runs_every_binary_past_a_failing_one() -> None:
    # without it cargo stops at the first failing binary: the RED run of 2026-10-22 never ran
    # api_parity, api_python, crash_recovery, replay_equals_live or the doc tests
    (leg,) = [leg for leg in gate.LEGS if leg.name == "cargo-test"]
    assert "--no-fail-fast" in leg.argv


def test_a_red_cargo_test_record_names_the_failing_test_and_keeps_its_panic(
    tmp_path, monkeypatch
) -> None:
    # 2026-10-22: the RED record held only "151 passed; 1 failed", so the test that failed
    # (api_load, #60) could not be named without re-running the suite
    (real,) = [leg for leg in gate.LEGS if leg.name == "cargo-test"]
    path = FIXTURES / "cargo-test-no-fail-fast-red.txt"
    code = f"import sys; sys.stdout.buffer.write(open({str(path)!r}, 'rb').read())"
    fake = gate.Leg("cargo-test", [sys.executable, "-c", code], real.judge, real.evidence)
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "LEGS", [fake])
    assert gate.main([]) == 1
    (row,) = _record(tmp_path)["legs"]
    assert row["verdict"] == "RED"
    lib = "unittests src\\lib.rs (target\\debug\\deps\\engine-ceeaaad3673c0ec7.exe)"
    smoke = "tests\\smoke.rs (target\\debug\\deps\\smoke-5a66701626c5dd9a.exe)"
    assert row["evidence"] == {
        "results": [
            f"{lib}: test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 0.01s",
            f"{smoke}: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 0.00s",
            "Doc-tests engine: test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 0.00s",
        ],
        "failures": [[f"{lib}: ---- deliberate::deliberately_fails stdout ----", *PANIC]],
        "skipped": [],
    }
    assert row["output_tail"] == fixture("cargo-test-no-fail-fast-red.txt").splitlines()
    assert "assertion `left == right` failed: deliberate" in row["output_tail"]


# the panic of the real red engine run (cargo-test-red.txt), as its failure section holds it
PANIC = [
    "thread 'deliberate::deliberately_fails' (22964) panicked at engine\\src\\lib.rs:23:9:",
    "assertion `left == right` failed: deliberate",
    "  left: 2",
    " right: 3",
    "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace",
]


def _planted_early_failure() -> str:
    """The real workspace run with the real red engine run's failure planted in its first
    binary, as --no-fail-fast prints it: every binary after the failing one still reports."""
    red = fixture("cargo-test-red.txt")
    failure = red[red.index("failures:") : red.index("test result: FAILED")]
    first = WORKSPACE.index("test result: ok. 38 passed; 0 failed")
    rest = WORKSPACE[first:].replace("ok. 38 passed; 0 failed", "FAILED. 37 passed; 1 failed", 1)
    return WORKSPACE[:first] + failure + rest


def test_an_early_failure_keeps_its_whole_section_after_it_leaves_the_tail(
    tmp_path, monkeypatch
) -> None:
    # --no-fail-fast runs every binary after a failing one, so on the whole workspace an early
    # failure's panic scrolls past the 200-line tail; this real workspace run is shorter, so the
    # tail is cut to match
    log = tmp_path / "cargo-test.txt"
    log.write_bytes(_planted_early_failure().encode("utf-8"))
    code = f"import sys; sys.stdout.buffer.write(open({str(log)!r}, 'rb').read())"
    (real,) = [leg for leg in gate.LEGS if leg.name == "cargo-test"]
    monkeypatch.setattr(gate, "TAIL_LINES", 40)
    leg = gate.Leg("cargo-test", [sys.executable, "-c", code], real.judge, real.evidence)
    result = gate.run_leg(leg)
    assert result.verdict == "RED"
    assert not set(PANIC) & set(result.output_tail)
    lib = "unittests src\\lib.rs (target\\debug\\deps\\engine-9f819c0ddae3f0f5.exe)"
    assert result.evidence["failures"] == [
        [f"{lib}: ---- deliberate::deliberately_fails stdout ----", *PANIC]
    ]
    assert len(result.evidence["results"]) == 9  # and every binary after it reported


def test_two_binaries_of_one_name_are_told_apart_by_their_exe() -> None:
    # engine's and engine-harness's lib tests both run as "unittests src\lib.rs"
    engine = fixture("cargo-test-green.txt")
    harness = engine.replace("engine-ceeaaad3673c0ec7.exe", "engine_harness-0123456789abcdef.exe")
    results = gate.cargo_test_evidence(engine + harness)["results"]
    labels = [r.split(": test result")[0] for r in results if r.startswith("unittests")]
    assert labels == [
        "unittests src\\lib.rs (target\\debug\\deps\\engine-ceeaaad3673c0ec7.exe)",
        "unittests src\\lib.rs (target\\debug\\deps\\engine_harness-0123456789abcdef.exe)",
    ]


def test_the_record_keeps_the_last_lines_of_a_long_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "OUT", tmp_path)
    code = "for i in range(500): print(i)\nprint('All checks passed!')"
    monkeypatch.setattr(gate, "LEGS", [_python_leg("long", code)])
    assert gate.main([]) == 0
    (row,) = _record(tmp_path)["legs"]
    assert len(row["output_tail"]) == gate.TAIL_LINES == 200
    assert row["output_tail"][-2:] == ["499", "All checks passed!"]
    assert row["evidence"] == {}  # a leg with no evidence function keeps only its tail


# ---------- a check a test skipped is counted, never passed silently (#60)


def test_a_check_a_test_skipped_is_counted_in_the_summary_and_the_record() -> None:
    # api_load's real output on a loaded machine (2026-10-24), a line of its own past libtest's
    # capture of a passing test; the line is edited into the excuse api_load prints now, for a p99
    # over the bound under load, since no run here missed the bound
    output = fixture("cargo-test-skipped-green.txt")
    verdict = gate.judge_cargo_test(0, output)
    assert verdict.green, verdict.source
    assert verdict.source == "1 test result lines: 1 passed; 0 failed; 1 skipped"
    (skip,) = gate.cargo_test_evidence(output)["skipped"]
    api_load = "tests\\api_load.rs (target\\debug\\deps\\api_load-8652aba3279d7f40.exe)"
    assert skip.startswith(f"{api_load}: SKIPPED: api_load's arrow p99 131.0442ms over its")


def test_only_a_line_that_starts_with_skipped_counts() -> None:
    output = fixture("cargo-test-green.txt").replace(
        "test tests::version_is_set ... ok", "test tests::version_is_set ... ok (SKIPPED: no)"
    )
    assert gate.judge_cargo_test(0, output).source == "2 test result lines: 1 passed; 0 failed"


@pytest.mark.parametrize(
    ("stem", "last"),
    [
        ("cargo-test-skipped-green.txt", "gate GREEN (1 skipped)"),
        ("cargo-test-green.txt", "gate GREEN"),
    ],
    ids=["skipped", "none-skipped"],
)
def test_the_gate_line_counts_the_checks_its_legs_skipped(
    tmp_path, monkeypatch, capsys, stem: str, last: str
) -> None:
    # a green gate whose legs skipped a check says so on its last line, not only in a leg's
    (real,) = [leg for leg in gate.LEGS if leg.name == "cargo-test"]
    code = f"import sys; sys.stdout.buffer.write(open({str(FIXTURES / stem)!r}, 'rb').read())"
    fake = gate.Leg("cargo-test", [sys.executable, "-c", code], real.judge, real.evidence)
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "LEGS", [fake])
    assert gate.main([]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith(f"{last}  (record: ")


def test_a_red_gate_names_the_leg_that_skipped(tmp_path, monkeypatch, capsys) -> None:
    # "gate RED: ruff-check (1 skipped)" would read as the red leg's skip
    (real,) = [leg for leg in gate.LEGS if leg.name == "cargo-test"]
    path = FIXTURES / "cargo-test-skipped-green.txt"
    code = f"import sys; sys.stdout.buffer.write(open({str(path)!r}, 'rb').read())"
    fake = gate.Leg("cargo-test", [sys.executable, "-c", code], real.judge, real.evidence)
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "LEGS", [_python_leg("ruff-check", "raise SystemExit(1)"), fake])
    assert gate.main([]) == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("gate RED: ruff-check; skipped: cargo-test 1  (record: ")


# ---------- a shared laptop: strays, build jobs, priority and load (#61)


def test_cargo_builds_on_half_the_cores_unless_told_otherwise() -> None:
    assert gate.leg_env({}, 8)["CARGO_BUILD_JOBS"] == "4"
    assert gate.leg_env({"CARGO_BUILD_JOBS": "6"}, 8)["CARGO_BUILD_JOBS"] == "6"
    assert gate.leg_env({}, 1)["CARGO_BUILD_JOBS"] == "1"
    assert gate.leg_env({}, None)["CARGO_BUILD_JOBS"] == "1"
    assert gate.leg_env({"PATH": "bin"}, 8)["PATH"] == "bin"


def test_every_leg_runs_with_the_capped_build_jobs(monkeypatch) -> None:
    monkeypatch.delenv("CARGO_BUILD_JOBS", raising=False)
    leg = _python_leg("env", "import os; print(os.environ['CARGO_BUILD_JOBS'])")
    assert gate.run_leg(leg).output_tail == [str(max(1, (os.cpu_count() or 2) // 2))]


# (HANDLE)-1 is the process's own pseudo-handle, whole: GetCurrentProcess() through ctypes'
# default int return comes back truncated, and GetPriorityClass then fails with 0
PRIORITY_CLASS = (
    "import ctypes; print(ctypes.windll.kernel32.GetPriorityClass(ctypes.c_void_p(-1)))"
)
BELOW_NORMAL, NORMAL = 0x4000, 0x20


@pytest.mark.skipif(sys.platform != "win32", reason="priority classes are Windows'")
def test_every_leg_starts_at_below_normal_priority() -> None:
    assert gate.run_leg(_python_leg("priority", PRIORITY_CLASS)).output_tail == [str(BELOW_NORMAL)]


@pytest.mark.skipif(sys.platform != "win32", reason="priority classes are Windows'")
@pytest.mark.parametrize(
    ("choice", "expected"), [("normal", NORMAL), ("belownormal", BELOW_NORMAL)], ids=str
)
def test_a_leg_runs_at_the_priority_asked_for_even_from_a_below_normal_gate(
    choice: str, expected: int
) -> None:
    # with no class given, a BelowNormal process's child is BelowNormal: a gate whose own process
    # runs there (this test, in the gate's pytest leg) must still start a leg at Normal when told
    leg = f"gate.Leg('priority', [sys.executable, '-c', {PRIORITY_CLASS!r}], gate.judge_ruff_check)"
    code = f"import gate, sys; print(gate.run_leg({leg}, {choice!r}).output_tail)"
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=gate.ROOT / "harness",
        capture_output=True,
        text=True,
        check=True,
        creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS,
    )
    assert done.stdout.strip() == str([str(expected)])


@pytest.mark.parametrize(
    ("set_to", "priority"),
    [(None, "belownormal"), ("belownormal", "belownormal"), ("normal", "normal")],
    ids=["default", "belownormal", "normal"],
)
def test_the_priority_is_on_the_machine_line_and_in_the_record(
    tmp_path, monkeypatch, capsys, set_to: str | None, priority: str
) -> None:
    # a timing test that goes red under load must say which priority it starved at
    if set_to is None:
        monkeypatch.delenv("TP_GATE_PRIORITY", raising=False)
    else:
        monkeypatch.setenv("TP_GATE_PRIORITY", set_to)
    code = PRIORITY_CLASS if sys.platform == "win32" else "pass"
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "LEGS", [_python_leg("ok", f"{code}; print('All checks passed!')")])
    assert gate.main([]) == 0
    record = _record(tmp_path)
    assert record["priority"] == priority
    assert f"machine at start: cores 8, cpu_busy_percent 12.5, priority {priority}\n" in (
        capsys.readouterr().out
    )
    if sys.platform == "win32":
        expected = {"normal": NORMAL, "belownormal": BELOW_NORMAL}[priority]
        assert record["legs"][0]["output_tail"][0] == str(expected)


@pytest.mark.parametrize(
    "value",
    ["high", "", "Normal", "BelowNormal", "below normal", " normal"],
    ids=["high", "empty", "capital", "camel", "a-space", "padded"],
)
def test_an_invalid_priority_is_refused_before_anything_runs(
    tmp_path, monkeypatch, capsys, value: str
) -> None:
    # ignored, it would run the gate at a priority nobody chose and record that as if chosen
    def never(root: Path) -> None:
        raise AssertionError("swept strays for a gate it should have refused")

    ran = tmp_path / "leg-ran"
    monkeypatch.setenv("TP_GATE_PRIORITY", value)
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate.strays, "sweep", never)
    code = f"open({str(ran)!r}, 'w').close(); print('All checks passed!')"
    monkeypatch.setattr(gate, "LEGS", [_python_leg("ok", code)])
    with pytest.raises(SystemExit) as refused:
        gate.main([])
    assert refused.value.code == 2
    message = capsys.readouterr().err
    assert "TP_GATE_PRIORITY must be one of ['belownormal', 'normal']" in message
    assert repr(value) in message
    assert not ran.exists()
    assert not list(tmp_path.glob("gate-*.json"))


def test_the_record_says_how_busy_the_cpu_was_and_what_was_swept(
    tmp_path, monkeypatch, capsys
) -> None:
    # a leg that goes red under load (#60) must be explainable from the record alone
    killed = {"pid": 7, "exe": "serve.exe", "minutes": 41.0, "action": "SUCCESS"}
    sweeps = iter([[killed], []])
    monkeypatch.setattr(gate.strays, "sweep", lambda root: next(sweeps))
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "LEGS", [_python_leg("ok", "print('All checks passed!')")])
    assert gate.main([]) == 0
    record = _record(tmp_path)
    assert record["machine"] == {"cores": 8, "cpu_busy_percent": 12.5}
    assert record["strays"] == {"start": [killed], "end": []}
    printed = capsys.readouterr().out
    assert "strays at start: serve.exe (pid 7, 41.0 min): SUCCESS" in printed
    assert "machine at start: cores 8, cpu_busy_percent 12.5" in printed
    assert "strays at end: none" in printed


def test_a_failing_sweep_costs_the_legs_nothing(tmp_path, monkeypatch) -> None:
    def denied(root: Path) -> None:
        raise PermissionError("access denied")

    monkeypatch.setattr(gate.strays, "sweep", denied)
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "LEGS", [_python_leg("ok", "print('All checks passed!')")])
    assert gate.main([]) == 0
    record = _record(tmp_path)
    assert record["strays"]["start"] == "sweep failed: PermissionError('access denied')"
    assert [row["verdict"] for row in record["legs"]] == ["GREEN"]


def test_the_machine_sample_is_cores_and_cpu_load() -> None:
    sample = REAL_MACHINE(interval=0.05)
    assert sample["cores"] == os.cpu_count()
    load = sample["cpu_busy_percent" if sys.platform == "win32" else "load_average_1m"]
    assert isinstance(load, float) and load >= 0
    assert sys.platform != "win32" or load <= 100


# ---------- the Kaggle runner's self-test: the gate runs it, and it must have run (#116)


def _self_test_leg() -> gate.Leg:
    (leg,) = [leg for leg in gate.LEGS if leg.name == "self-test"]
    return leg


def _run_self_test_leg(
    root: Path, monkeypatch, selftest: str, limit: float | None = None
) -> gate.LegResult:
    """The real leg on a copy of the real runner, with `selftest` as its checks: the whole path
    from the leg's argv to its verdict, in a second rather than five minutes."""
    (root / "notebooks").mkdir()
    shutil.copy(gate.ROOT / "notebooks" / "kaggle_ising_runs.py", root / "notebooks")
    (root / "notebooks" / "kaggle_ising_runs_selftest.py").write_text(selftest, encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", root)
    leg = _self_test_leg()
    return gate.run_leg(leg if limit is None else replace(leg, limit=limit))


@pytest.mark.parametrize(
    ("selftest", "verdict", "source"),
    [
        ("import no_such_module_tp116\n", "RED", "(exit 1): ModuleNotFoundError"),
        (
            "def run_all(k):\n    raise RuntimeError('died')\n",
            "RED",
            "(exit 1): RuntimeError: died",
        ),
        ("def run_all(k):\n    return 0\n", "RED", "no 'N/M checks passed' line (exit 0)"),
        ("def run_all(k):\n    print('0/0 checks passed')\n    return 0\n", "RED", "zero checks"),
        (
            "def run_all(k):\n    print('FAIL  a plan')\n    print('1/2 checks passed')\n"
            "    return 1\n",
            "RED",
            "1/2 checks passed; first: FAIL  a plan",
        ),
        (
            "def run_all(k):\n    print('PASS  a plan')\n    print('1/1 checks passed')\n"
            "    return 0\n",
            "GREEN",
            "1/1 checks passed",
        ),
    ],
    ids=["crash-at-import", "crash-in-a-section", "silent", "no-checks", "a-failed-check", "green"],
)
def test_the_self_test_leg_from_argv_to_verdict(
    tmp_path, monkeypatch, selftest: str, verdict: str, source: str
) -> None:
    result = _run_self_test_leg(tmp_path, monkeypatch, selftest)
    assert result.verdict == verdict, result.verdict_source
    assert source in result.verdict_source


def test_a_runner_that_crashes_at_import_is_red(tmp_path, monkeypatch) -> None:
    (tmp_path / "notebooks").mkdir()
    runner = tmp_path / "notebooks" / "kaggle_ising_runs.py"
    runner.write_text("import no_such_module_tp116\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    result = gate.run_leg(_self_test_leg())
    assert result.verdict == "RED"
    assert "(exit 1): ModuleNotFoundError" in result.verdict_source


def test_a_runner_that_is_not_there_is_red(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    result = gate.run_leg(_self_test_leg())
    assert result.verdict == "RED"
    assert "no 'N/M checks passed' line (exit 2)" in result.verdict_source


def test_the_self_test_leg_runs_the_real_runner_on_this_interpreter() -> None:
    # a bare "python" finds the base install beside a venv's interpreter, without tipping_point
    argv = _self_test_leg().argv
    assert argv[0] == sys.executable
    assert (gate.ROOT / argv[1]).is_file()
    assert argv[2:] == ["--self-test"]


@pytest.mark.parametrize(
    "output",
    ["PASS  1/1 checks passed\n", "[2026-11-01 05:00:00 UTC +0s] 1/1 checks passed\n"],
    ids=["a-check-name", "a-log-line"],
)
def test_only_the_self_tests_own_closing_line_counts(output: str) -> None:
    verdict = gate.judge_self_test(0, output)
    assert not verdict.green
    assert "no 'N/M checks passed' line" in verdict.source


def test_the_last_summary_line_decides() -> None:
    assert not gate.judge_self_test(0, "1/1 checks passed\nFAIL  x\n1/2 checks passed\n").green


def test_a_failed_check_stays_in_the_record_after_it_leaves_the_tail(tmp_path, monkeypatch) -> None:
    # the plans are checked first and a re-run takes minutes, so the record must name the plan
    real = _self_test_leg()
    path = FIXTURES / "self-test-red.txt"
    code = f"import sys; sys.stdout.buffer.write(open({str(path)!r}, 'rb').read())"
    fake = gate.Leg("self-test", [sys.executable, "-c", code], real.judge, real.evidence)
    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "LEGS", [fake])
    assert gate.main([]) == 1
    (row,) = _record(tmp_path)["legs"]
    assert row["verdict"] == "RED"
    (first, *_) = row["evidence"]["failures"]
    assert first.startswith("FAIL  plan primary-28.json loads:")
    assert "no-such-input" in first
    assert first not in row["output_tail"]
    assert first in row["verdict_source"]


# ---------- a section deleted from the self-test must not read as green (#124)


# run_all's sections, each with the fewest `check(` calls it may reach (#132). The leg trusts the
# closing "N/M checks passed" line, which counts only the checks that ran: delete real_runs, the
# only real train, resume and extract path and most of the leg's time, and it reads 223/223, exit
# 0. A total would need bumping on nearly every change and could not say which section went. A new
# section joins this list with its count; a check deleted on purpose lowers its section's.
SELF_TEST = gate.ROOT / "notebooks" / "kaggle_ising_runs_selftest.py"
SELF_TEST_SECTIONS = {
    "plans": 8,
    "refusals": 3,
    "cli": 3,
    "shrinking": 7,
    "queues": 13,
    "failed_extractions": 7,
    "resolution": 7,
    "previous": 9,
    "inputs": 8,
    "input_checkpoints": 6,
    "wheel": 14,
    "verification": 18,
    "kernel_folder": 3,
    "jepa_entries": 43,
    "extraction_rates": 7,
    "kernel_acceptance": 7,
    "invalid_plans": 6,
    "pilot_28": 20,
    "primary_28": 13,
    "first_reads": 21,
    "v1_confirmatory": 22,
    "ablations_28": 20,
    "real_runs": 21,
}


def _calls_that_always_run(body: list[ast.stmt]) -> Iterator[str]:
    """The name each bare call among these statements and through their `with` blocks calls. A call
    behind an `if`, in a loop or in a function defined here may never run."""
    for statement in body:
        if isinstance(statement, ast.With):
            yield from _calls_that_always_run(statement.body)
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
        ):
            yield statement.value.func.id


def _sections_missing(source: str) -> list[str]:
    bodies = [
        node.body
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_all"
    ]
    called = {name for body in bodies for name in _calls_that_always_run(body)}
    return [section for section in SELF_TEST_SECTIONS if section not in called]


def test_run_all_calls_every_named_section() -> None:
    assert _sections_missing(SELF_TEST.read_text("utf-8")) == []


# run_all's shape with every section in it: what the check can catch, whatever the real one holds
WHOLE_RUN_ALL = (
    "def run_all(k):\n"
    "    with tempfile.TemporaryDirectory() as tmp:\n"
    "        root = Path(tmp)\n"
    + "".join(f"        {section}(k, root)\n" for section in SELF_TEST_SECTIONS)
    + "    return 0\n"
)


def test_a_run_all_with_every_section_has_none_missing() -> None:
    assert _sections_missing(WHOLE_RUN_ALL) == []


@pytest.mark.parametrize("section", SELF_TEST_SECTIONS)
def test_a_section_deleted_from_run_all_is_named(section: str) -> None:
    deleted = WHOLE_RUN_ALL.replace(f"        {section}(k, root)\n", "")
    assert _sections_missing(deleted) == [section]


@pytest.mark.parametrize(
    "guarded",
    [
        "        if os.environ.get('TP_FULL'):\n            real_runs(k, root)\n",
        "        for _ in ():\n            real_runs(k, root)\n",
        "        def later():\n            real_runs(k, root)\n",
        "        later = lambda: real_runs(k, root)\n",
    ],
    ids=["behind-an-if", "in-a-loop", "in-a-function", "in-a-lambda"],
)
def test_a_section_that_may_never_run_counts_as_deleted(guarded: str) -> None:
    source = WHOLE_RUN_ALL.replace("        real_runs(k, root)\n", guarded)
    assert _sections_missing(source) == ["real_runs"]


def test_no_run_all_leaves_every_section_missing() -> None:
    assert _sections_missing("def something_else(k):\n    plans(k)\n") == list(SELF_TEST_SECTIONS)


# ---------- a section emptied in the self-test must not read as green (#132)


def _live(body: list[ast.stmt]) -> list[ast.stmt]:
    """These statements up to the first return or raise among them: what follows never runs."""
    for i, statement in enumerate(body):
        if isinstance(statement, ast.Return | ast.Raise):
            return body[: i + 1]
    return body


def _checks_reached(source: str, section: str) -> int:
    """The `check(` calls written in the section and in each function of the file it calls, through
    their calls in turn, each function once. Call sites, not checks run: a check in a loop counts
    once. A call behind an `if` counts, so a guarded check is not caught here; #124's test holds
    run_all, and this holds the sections under it."""
    functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    seen, todo, sites = set(), [section], 0
    while todo:
        name = todo.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        for statement in _live(functions[name].body):
            for node in ast.walk(statement):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "check":
                        sites += 1
                    else:
                        todo.append(node.func.id)
    return sites


def _sections_short(source: str, floors: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Each section that reaches fewer checks than its floor: (reached, floor)."""
    reached = {section: _checks_reached(source, section) for section in floors}
    return {s: (n, floors[s]) for s, n in reached.items() if n < floors[s]}


def test_every_section_reaches_its_count_of_checks() -> None:
    # a red here names the section: restore what went, or lower its count if that was the point
    assert _sections_short(SELF_TEST.read_text("utf-8"), SELF_TEST_SECTIONS) == {}


def test_every_section_reaches_at_least_one_check() -> None:
    assert [section for section, floor in SELF_TEST_SECTIONS.items() if floor < 1] == []


# a self-test's shape: three sections, the last reaching its checks through a helper in a loop
WHOLE_SELF_TEST = """\
def check(name, ok, detail=""):
    results.append(bool(ok))

def plans(k, root):
    check("a", True)
    check("b", True)

def queues(k, root):
    with open(root) as f:
        if f:
            check("c", True)

def helper(k, root, model):
    check("d", True)
    check("e", True)

def first_reads(k, root):
    check("f", True)
    for model in MODELS:
        helper(k, root, model)
"""
WHOLE_FLOORS = {"plans": 2, "queues": 1, "first_reads": 3}


def test_a_whole_self_test_has_no_section_short() -> None:
    assert _sections_short(WHOLE_SELF_TEST, WHOLE_FLOORS) == {}


@pytest.mark.parametrize(
    ("section", "emptied", "reached"),
    [
        ("plans", "def plans(k, root):\n    return\n", 0),
        ("plans", "def plans(k, root): pass\n", 0),
        ("plans", 'def plans(k, root):\n    return\n    check("a", True)\n', 0),
        ("plans", 'def plans(k, root):\n    raise SystemExit\n    check("a", True)\n', 0),
        ("plans", 'def plans(k, root):\n    check("a", True)\n', 1),
        ("queues", "def queues(k, root):\n    return\n", 0),
        ("first_reads", 'def first_reads(k, root):\n    check("f", True)\n', 1),
    ],
    ids=[
        "body-gone",
        "pass",
        "return-first",
        "raise-first",
        "one-check-gone",
        "nested-check-gone",
        "helper-loop-gone",
    ],
)
def test_an_emptied_section_is_named(section: str, emptied: str, reached: int) -> None:
    # #132: `def real_runs(k, root): return` kept #124's call from run_all and ran no checks
    start = WHOLE_SELF_TEST.index(f"def {section}(")
    end = WHOLE_SELF_TEST.find("\ndef ", start + 1)
    end = len(WHOLE_SELF_TEST) if end == -1 else end + 1
    source = WHOLE_SELF_TEST[:start] + emptied + WHOLE_SELF_TEST[end:]
    assert _sections_short(source, WHOLE_FLOORS) == {section: (reached, WHOLE_FLOORS[section])}


def test_a_helper_emptied_shorts_the_section_that_calls_it() -> None:
    source = WHOLE_SELF_TEST.replace('    check("d", True)\n    check("e", True)\n', "    pass\n")
    assert _sections_short(source, WHOLE_FLOORS) == {"first_reads": (1, 3)}


def test_a_helper_counts_once_in_a_section_and_in_every_section_that_calls_it() -> None:
    twice = "\ndef twice(k, root):\n    helper(k, root, 1)\n    helper(k, root, 2)\n"
    shared = WHOLE_SELF_TEST + twice
    assert _checks_reached(shared, "twice") == 2
    assert _checks_reached(shared, "first_reads") == 3


def test_a_section_that_calls_itself_ends() -> None:
    source = "def check(n, ok):\n    pass\n"
    source += "def plans(k, root):\n    check('a', 1)\n    plans(k, root)\n"
    assert _checks_reached(source, "plans") == 1


def test_a_section_not_in_the_file_reaches_none() -> None:
    assert _sections_short("def check(n, ok):\n    pass\n", {"plans": 1}) == {"plans": (0, 1)}


# ---------- a hung leg ends in a red row, a last line and a record (#125)


# The slowest each leg has taken on this laptop under load, in seconds: every record and gate log
# across the checkouts, to 2026-11-01. A new leg has to bring its own. The limits allow for more
# than three times these where a leg's time is mostly CPU: on 2026-11-01, at BelowNormal with every
# core taken, pyright took nine times its median and a bare python stand-in over a minute.
SLOWEST = {
    "cargo-fmt": 55.15,  # .worktrees/gateprune.log, CPU 100% busy as it started
    "cargo-clippy": 833.48,  # .worktrees/gate-freeze3.log, 83.6%
    "cargo-test": 961.12,  # .worktrees/gateprune.log, 100%
    "ruff-check": 25.78,  # .worktrees/gate-fix-runner.log, 100%
    "ruff-format": 16.94,  # the real leg through run_leg, 2026-11-01, 100%
    "pyright": 419.37,  # .worktrees/gate-fix-runner.log, 100%
    "pytest": 2455.43,  # .worktrees/gater3.log, 54.3%
    "self-test": 646.78,  # .worktrees/gate-fix-runner.log, 100%
}
# Seconds a stand-in holds on for: past every limit below, however long a starved kill takes (at
# BelowNormal under load a bare python can take a minute to start), and over by itself, so that a
# gate that has lost its limit fails these tests instead of hanging them
HOLD = 900


def _hanging(name: str, limit: float, code: str = f"import time; time.sleep({HOLD})") -> gate.Leg:
    return gate.Leg(name, [sys.executable, "-c", code], gate.judge_ruff_check, limit=limit)


def _gone(pid: int, within: float = 120.0) -> bool:
    """Whether the process has left the system's list, soon: a kill takes effect a moment after
    taskkill says so, and longer on a starved laptop."""
    deadline = time.monotonic() + within
    while any(p.pid == pid for p in gate.strays.list_processes() or []):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def test_every_limit_sits_well_above_the_slowest_that_leg_has_taken() -> None:
    assert [leg.name for leg in gate.LEGS if leg.limit < 3 * SLOWEST[leg.name]] == []


def test_a_hung_leg_is_red_and_the_gate_still_reports(tmp_path, monkeypatch, capsys) -> None:
    # #125: run_leg waited with no timeout, so a gate in a background shell never reported at all
    monkeypatch.setattr(gate, "OUT", tmp_path)
    last = _python_leg("last", "print('All checks passed!')")
    monkeypatch.setattr(gate, "LEGS", [_hanging("hung", 2.0), last])
    assert gate.main([]) == 1
    rows = _record(tmp_path)["legs"]
    assert [(r["leg"], r["verdict"]) for r in rows] == [("hung", "RED"), ("last", "GREEN")]
    assert rows[0]["returncode"] is None
    assert "hung: still running at its 2s limit" in rows[0]["verdict_source"]
    assert 2.0 <= rows[0]["duration"] < HOLD
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1].startswith("gate RED: hung  (record: ")


@pytest.mark.skipif(sys.platform != "win32", reason="lists processes on Windows alone")
def test_a_hung_leg_takes_its_whole_tree_and_keeps_what_it_printed(tmp_path) -> None:
    # killing only the leg leaves the process doing the work: a venv's python.exe and pytest.exe
    # are launchers, cargo sits above the test binary, and subprocess.run(timeout=) kills the leg
    # alone, then on Windows reads what the survivors hold open until they end. This stand-in has
    # that shape: leg -> grandchild, each through the venv's launcher
    pid_file = tmp_path / "grandchild.pid"
    grandchild = (
        "import os, time\n"
        "print('grandchild up', flush=True)\n"
        f"open({f'{pid_file}.tmp'!r}, 'w').write(str(os.getpid()))\n"
        f"os.replace({f'{pid_file}.tmp'!r}, {str(pid_file)!r})\n"
        f"time.sleep({HOLD})\n"
    )
    code = (
        "import subprocess, sys\n"
        "print('leg started', flush=True)\n"
        f"subprocess.run([sys.executable, '-c', {grandchild!r}])\n"
    )
    for limit in (2.0, 8.0, 32.0, 128.0, 512.0):  # seconds alone, unless starting it takes minutes
        result = gate.run_leg(_hanging("hung", limit, code))
        if pid_file.is_file():
            break
    assert pid_file.is_file(), "the limit fell every time before the tree was up"
    assert result.verdict == "RED"
    assert result.duration < HOLD
    assert f"hung: still running at its {limit:g}s limit" in result.verdict_source
    assert result.output_tail == ["leg started", "grandchild up"]
    assert _gone(int(pid_file.read_text())), result.verdict_source


def test_a_leg_that_ended_is_judged_though_a_process_it_started_lives_on() -> None:
    # that process inherited the leg's output: reading a pipe to its end would wait for it, so a
    # finished leg sat there until the stray ended, and with no limit for ever
    code = (
        "import subprocess, sys\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep({HOLD})'])\n"
        "print(child.pid)\n"
        "print('All checks passed!')\n"
    )
    result = gate.run_leg(_python_leg("leaves-one-behind", code))
    if sys.platform == "win32":  # elsewhere it ends by itself, in HOLD seconds
        gate.strays.kill_tree(int(result.output_tail[0]))
    assert result.verdict == "GREEN", result.verdict_source
    assert result.returncode == 0
    assert result.duration < HOLD


@pytest.mark.skipif(sys.platform != "win32", reason="taskkill is Windows'")
def test_a_tree_kill_that_fails_is_said_and_the_gate_goes_on(monkeypatch) -> None:
    def denied(pid: int) -> str:
        raise PermissionError("access denied")

    monkeypatch.setattr(gate.strays, "kill_tree", denied)
    result = gate.run_leg(_hanging("hung", 1.0, "import time; time.sleep(10)"))
    assert result.verdict == "RED"
    assert "hung" in result.verdict_source and "access denied" in result.verdict_source


def test_the_self_test_leg_hung_in_a_section_is_red(tmp_path, monkeypatch) -> None:
    # #125's failing case, on the leg's own argv: a runner edit that stops Scheduler.stop setting
    # halt makes the hold loop in the self-test's drive spin for ever
    selftest = f"def run_all(k):\n    import time\n    time.sleep({HOLD})\n    return 0\n"
    result = _run_self_test_leg(tmp_path, monkeypatch, selftest, limit=2.0)
    assert result.verdict == "RED"
    assert result.returncode is None
    assert "hung" in result.verdict_source
    assert result.duration < HOLD


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
    rows = _record(tmp_path)["legs"]
    assert [(r["leg"], r["verdict"]) for r in rows] == [("broken", "RED"), ("last", "GREEN")]
    assert rows[0]["returncode"] is None
    assert "could not start" in rows[0]["verdict_source"]


def test_non_utf8_junit_is_red_not_a_crash(monkeypatch) -> None:
    monkeypatch.setattr(gate, "JUNIT", FIXTURES / "junit-non-utf8.xml")
    leg = gate.Leg("junit", [sys.executable, "-c", "pass"], gate._judge_pytest_from_disk)
    result = gate.run_leg(leg)
    assert result.verdict == "RED"
    assert "judge crashed" in result.verdict_source


def test_undeletable_stale_junit_is_red_and_the_rest_still_run(tmp_path, monkeypatch) -> None:
    def held_open(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(13, "held open by another process", str(self))

    monkeypatch.setattr(gate, "OUT", tmp_path)
    monkeypatch.setattr(gate, "JUNIT", tmp_path / "pytest-junit.xml")
    monkeypatch.setattr(Path, "unlink", held_open)
    monkeypatch.setattr(
        gate,
        "LEGS",
        [
            gate.Leg("pytest", [sys.executable, "-c", "pass"], gate._judge_pytest_from_disk),
            _python_leg("last", "print('All checks passed!')"),
        ],
    )
    assert gate.main([]) == 1
    rows = _record(tmp_path)["legs"]
    assert [(r["leg"], r["verdict"]) for r in rows] == [("pytest", "RED"), ("last", "GREEN")]
    assert "held open by another process" in rows[0]["verdict_source"]
