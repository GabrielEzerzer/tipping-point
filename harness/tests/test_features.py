"""features.json, the feature map: every entry point (Cargo bin, guarded package module, guarded
script, harness/explore script) is mapped, every mapped command's target exists, an argparse
command's --help exits 0 naming only mapped subcommands, and every mapped test file exists."""

import ast
import json
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = json.loads((ROOT / "features.json").read_text(encoding="utf-8"))
FEATURES: list[dict[str, object]] = MAP["features"]
INTERNAL: list[dict[str, str]] = MAP["internal"]
FIELDS = {"name", "description", "command", "inputs", "outputs", "test"}
MAIN_GUARD = re.compile(r'^if __name__ == "__main__":', re.MULTILINE)
SUBCOMMANDS = re.compile(r"^  \{([\w,-]+)\}", re.MULTILINE)
HELP_SECONDS = 120
EXPLORE = "harness/explore"


def parse(command: str) -> tuple[str, str | None]:
    words = shlex.split(command)
    if words[0] == "cargo":
        target = f"{words[words.index('-p') + 1]}:{words[words.index('--bin') + 1]}"
        rest = words[words.index("--") + 1 :] if "--" in words else []
    else:
        at = words.index("python") + 1
        at += words[at] == "-m"
        target, rest = words[at], words[at + 1 :]
    sub = rest[0] if rest and not rest[0].startswith(("-", "<")) else None
    return target, sub


def scripts() -> dict[str, Path]:
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    found: dict[str, Path] = {}
    for name in tracked:
        path, parts = ROOT / name, Path(name).parts
        if "tests" in parts or not MAIN_GUARD.search(path.read_text(encoding="utf-8")):
            continue
        if parts[0] == "python":
            module = parts[3:-1] if path.name == "__main__.py" else (*parts[3:-1], path.stem)
            found[".".join(module)] = path
        else:
            found[Path(name).parent.as_posix() if path.name == "__main__.py" else name] = path
    return found


def binaries() -> set[str]:
    metadata = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--offline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        f"{package['name']}:{target['name']}"
        for package in json.loads(metadata)["packages"]
        for target in package["targets"]
        if "bin" in target["kind"]
    }


def mapped() -> dict[str, set[str | None]]:
    commands: dict[str, set[str | None]] = {}
    for command in [str(f["command"]) for f in FEATURES] + [i["command"] for i in INTERNAL]:
        target, sub = parse(command)
        commands.setdefault(target, set()).add(sub)
    return commands


def explore_scripts() -> tuple[str, ...]:
    door = ast.parse((ROOT / EXPLORE / "__main__.py").read_text(encoding="utf-8"))
    (listed,) = [
        ast.literal_eval(node.value)
        for node in door.body
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == ["SCRIPTS"]
    ]
    return listed


def help_of(key: str, *target: str) -> tuple[str, subprocess.CompletedProcess[str]]:
    return key, subprocess.run(
        [sys.executable, *target, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=HELP_SECONDS,
    )


def test_every_feature_names_its_command_inputs_outputs_and_test() -> None:
    names = [f.get("name") for f in FEATURES]
    assert len(names) == len(set(names)), "give every feature in features.json a unique name"
    for feature in FEATURES:
        untested = feature.get("test") == "none"
        want = FIELDS | {"issue"} if untested else FIELDS
        assert set(feature) == want, (
            f"{feature.get('name')}: give exactly the fields {sorted(want)}"
        )
        assert all(isinstance(feature[k], str) and feature[k] for k in FIELDS), (
            f"{feature['name']}: every field but issue is a non-empty string"
        )
        assert not untested or isinstance(feature["issue"], int), (
            f"{feature['name']}: a feature with no test names its open test-gap issue number"
        )
    assert all(set(i) == {"command", "reason"} for i in INTERNAL), (
        "each internal entry of features.json gives its command and the reason it is not a feature"
    )


def test_every_entry_point_is_in_the_map() -> None:
    tests = {str(f["test"]) for f in FEATURES}
    missing = sorted((set(scripts()) | binaries()) - set(mapped()) - tests)
    assert not missing, (
        f"add these entry points to features.json, as features or internal: {missing}"
    )


def test_every_explore_script_is_in_the_map() -> None:
    want, named = set(explore_scripts()), mapped().get(EXPLORE, set())
    assert named == want, (
        f"{EXPLORE}/__main__.py SCRIPTS and features.json differ: add "
        f"{sorted(want - named)} to features.json, as features or internal, and remove "
        f"{sorted(map(str, named - want))}"
    )


def test_every_mapped_command_runs_something_that_exists() -> None:
    bins = binaries()
    for target in mapped():
        if ":" in target:
            ok = target in bins
        elif "/" in target:
            ok = (ROOT / target).is_file() or (ROOT / target / "__main__.py").is_file()
        else:
            ok = find_spec(target) is not None
        assert ok, (
            f"features.json maps a command to {target}, which does not exist: fix the command"
        )


def test_every_mapped_test_exists() -> None:
    for feature in FEATURES:
        test = str(feature["test"])
        assert test == "none" or (ROOT / test).is_file(), (
            f"{feature['name']}: its test {test} does not exist; name the test file that covers it"
        )


def test_every_argparse_command_has_help_and_maps_each_subcommand() -> None:
    tests = {str(f["test"]) for f in FEATURES}
    runnable = {
        key: path
        for key, path in scripts().items()
        if key not in tests and "ArgumentParser(" in path.read_text(encoding="utf-8")
    }
    commands = mapped()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [(key, ["-m", key] if "/" not in key else [key]) for key in runnable]
        for key, done in pool.map(lambda job: help_of(job[0], *job[1]), jobs):
            assert done.returncode == 0, (
                f"{key} --help exits {done.returncode}: {done.stderr[-500:]}"
            )
            subs = {s for group in SUBCOMMANDS.findall(done.stdout) for s in group.split(",")}
            named = commands.get(key, set())
            assert named == (subs or {None}), (
                f"map every subcommand of {key} in features.json, and no other: "
                f"mapped {sorted(map(str, named))}, --help lists {sorted(subs)}"
            )


def test_every_explore_script_has_help() -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, done in pool.map(lambda name: help_of(name, EXPLORE, name), explore_scripts()):
            assert done.returncode == 0, (
                f"{EXPLORE} {name} --help exits {done.returncode}: {done.stderr[-500:]}"
            )
