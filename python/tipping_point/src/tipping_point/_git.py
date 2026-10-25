"""The git checkout this package's source tree sits in (#109).

Standard library only: the build hook (python/tipping_point/hatch_build.py) loads this file by
path to stamp a build, and ising.io.code_version reads a source checkout through it, both asking
about the project's pyproject.toml, so a stamp and a checkout mean the same thing by "dirty".
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Each points git at a repository other than the one around the folder it runs in. A git hook
# exports some of them, and so may a shell; with GIT_DIR set, ls-files and HEAD answer for that
# repository wherever git runs.
_REDIRECTS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")


def head(path: Path) -> tuple[str, bool]:
    """The HEAD commit of the git repository that tracks the file `path`, and whether that
    checkout has any uncommitted change (untracked files count). Raises OSError without git, and
    CalledProcessError when no repository tracks `path`: one that only encloses it (a venv or an
    unpacked sdist inside another project) is not its checkout. Git finds that repository from
    `path` alone, whatever GIT_DIR and its kin say."""
    env = {key: value for key, value in os.environ.items() if key not in _REDIRECTS}

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=path.parent, env=env, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("ls-files", "--error-unmatch", "--", path.name)
    return git("rev-parse", "HEAD"), git("status", "--porcelain") != ""
