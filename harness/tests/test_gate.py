"""The gate must not lie: every lie case judges RED, the honest case judges GREEN.

Fixtures in fixtures/ are real tool output captured on this repo (green = clean tree,
red = deliberately broken files; any other name = files deliberately edited into that shape).
"""

import json
import re
import shutil
import subprocess
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
    (record,) = tmp_path.glob("gate-*.json")
    rows = json.loads(record.read_text(encoding="utf-8"))
    assert [(r["leg"], r["verdict"]) for r in rows] == [("pytest", "RED"), ("last", "GREEN")]
    assert "held open by another process" in rows[0]["verdict_source"]
