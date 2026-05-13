"""Rewrite pyproject.toml from an rc candidate to its stable release version."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_RE = re.compile(r'^(version\s*=\s*)"([^"]+)"', re.MULTILINE)
PACKAGE_VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)(?:rc\d+)?$")


def _release_version(candidate_version: str) -> str:
    match = PACKAGE_VERSION_RE.match(candidate_version)
    if not match:
        raise ValueError(f"Unsupported package version: {candidate_version}")
    return match.group(1)


def _resolve_release_version(current_version: str) -> str:
    release_version = os.environ.get("US_DATA_RELEASE_VERSION", "")
    derived_release_version = _release_version(current_version)
    if not release_version:
        return derived_release_version
    explicit_release_version = _release_version(release_version)
    if explicit_release_version != derived_release_version:
        raise ValueError(
            "US_DATA_RELEASE_VERSION must match the current package candidate: "
            f"{explicit_release_version} != {derived_release_version}"
        )
    return explicit_release_version


def main() -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    text = pyproject.read_text()
    match = VERSION_RE.search(text)
    if not match:
        print("Could not find project version in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    current_version = match.group(2)
    release_version = _resolve_release_version(current_version)
    if current_version == release_version:
        print(f"pyproject.toml already uses final version {release_version}.")
        return

    updated = VERSION_RE.sub(rf'\1"{release_version}"', text, count=1)
    pyproject.write_text(updated)
    print(f"Finalized package version: {current_version} -> {release_version}")


if __name__ == "__main__":
    main()
