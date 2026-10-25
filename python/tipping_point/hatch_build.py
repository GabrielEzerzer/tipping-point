"""Hatch build hook (#109): stamps tipping_point/_build_info.py with the commit a build comes from,
so an installed copy, which sits in no checkout, still names its code (ising.io.code_version).

Built from a git checkout of this source, every sdist and wheel ships a generated _build_info.py
in place of the committed placeholder: HEAD and whether the checkout had uncommitted changes (read
through tipping_point/_git.py, as code_version reads a checkout), the sha256 of ising/kernel.py and
the UTC build time. Built where no repository tracks the source, as when uv builds the wheel from
the sdist in a temporary folder, it changes nothing, so the sdist's stamp carries through. An
editable install is left alone: it reads git from the checkout itself.
"""

from __future__ import annotations

import datetime
import hashlib
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import (  # pyright: ignore[reportMissingImports]
    BuildHookInterface,
)

PACKAGE = "src/tipping_point"


class StampBuildInfo(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            return
        package = Path(self.root) / PACKAGE
        head = runpy.run_path(str(package / "_git.py"))["head"]
        try:
            commit, dirty = head(Path(self.root) / "pyproject.toml")
        except (OSError, subprocess.CalledProcessError):
            if runpy.run_path(str(package / "_build_info.py"))["commit"] is None:
                self.app.display_warning(
                    "tipping_point: no git checkout tracks this source, so the build records no "
                    "commit and its runs cannot make a pilot record (#109)"
                )
            return
        kernel = hashlib.sha256((package / "ising" / "kernel.py").read_bytes()).hexdigest()
        built_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.stamp_dir = tempfile.mkdtemp(prefix="tp-build-info-")
        stamp = Path(self.stamp_dir) / "_build_info.py"
        stamp.write_bytes(
            (
                '"""Build provenance (#109), stamped by hatch_build.py at build time."""\n\n'
                f"commit: str | None = {commit!r}\n"
                f"dirty: bool | None = {dirty!r}\n"
                f"kernel_sha256: str | None = {kernel!r}\n"
                f"built_at: str | None = {built_at!r}\n"
            ).encode()
        )
        # outside the project root, so hatchling ships it in place of the placeholder, not beside it
        build_data["force_include"][str(stamp)] = f"{PACKAGE}/_build_info.py"

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        if hasattr(self, "stamp_dir"):
            shutil.rmtree(self.stamp_dir, ignore_errors=True)
