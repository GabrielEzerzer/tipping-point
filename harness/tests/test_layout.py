# The repository layout: every tracked file sits under a known top-level folder or is a
# known top-level file.

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {".cargo", ".github", "engine", "harness", "notebooks", "paper", "python", "setup"}
FILES = {
    ".gitattributes",
    ".gitignore",
    ".lint-baseline.json",
    "Cargo.lock",
    "Cargo.toml",
    "LICENSE",
    "README.md",
    "features.json",
    "pyproject.toml",
    "rustfmt.toml",
    "uv.lock",
}


def stray(paths: list[str]) -> list[str]:
    return sorted(
        path
        for path in paths
        if (path.split("/", 1)[0] not in FOLDERS if "/" in path else path not in FILES)
    )


def test_every_tracked_file_sits_in_a_known_place() -> None:
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout
    found = stray([path for path in listed.split("\0") if path])
    assert not found, (
        f"tracked files outside the layout: {found}. Move each under one of the top-level folders "
        f"{sorted(FOLDERS)}, or, for a new top-level file, add it to FILES in harness/tests/"
        f"test_layout.py (top-level files: {sorted(FILES)})."
    )


def test_a_file_outside_the_layout_is_reported() -> None:
    assert stray(["engine/src/lib.rs", "README.md", "docs/notes.md", "notes.txt", "harness"]) == [
        "docs/notes.md",
        "harness",
        "notes.txt",
    ]
