"""Build provenance (#109): the commit this copy of the package was built from.

The checkout holds this placeholder. Every sdist and wheel built from a git checkout ships these
fields filled in by the build hook (python/tipping_point/hatch_build.py). ising.io.code_version
reads them when the package is not imported from its own checkout.
"""

commit: str | None = None  # the checkout's HEAD at build time
dirty: bool | None = None  # whether that checkout had uncommitted changes (untracked files count)
kernel_sha256: str | None = None  # sha256 of ising/kernel.py as built
built_at: str | None = None  # UTC, ISO 8601
