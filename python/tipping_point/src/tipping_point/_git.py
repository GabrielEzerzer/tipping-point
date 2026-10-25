"""The git checkout this package's source tree sits in (#109).

Standard library only: the build hook (python/tipping_point/hatch_build.py) loads this file by
path to stamp a build, and ising.io.code_version reads a source checkout through it, both asking
about the project's pyproject.toml, so a stamp and a checkout mean the same thing by "dirty".
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def head(path: Path) -> tuple[str, bool]:
    """The HEAD commit of the git repository that tracks the file `path`, and whether that
    checkout has any uncommitted change (untracked files count). Raises OSError without git, and
    CalledProcessError when no repository tracks `path`: one that only encloses it (a venv or an
    unpacked sdist inside another project) is not its checkout."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=path.parent, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("ls-files", "--error-unmatch", "--", path.name)
    return git("rev-parse", "HEAD"), git("status", "--porcelain") != ""
