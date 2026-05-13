"""Rewrite pyproject.toml from an rc candidate to its stable release version."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_RE = re.compile(r'^(version\s*=\s*)"([^"]+)"', re.MULTILINE)


def _release_version(candidate_version: str) -> str:
    match = re.match(r"^(\d+\.\d+\.\d+)(?:rc\d+)?$", candidate_version)
    if not match:
        raise ValueError(f"Unsupported package version: {candidate_version}")
    return match.group(1)


def main() -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    text = pyproject.read_text()
    match = VERSION_RE.search(text)
    if not match:
        print("Could not find project version in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    current_version = match.group(2)
    release_version = os.environ.get("US_DATA_RELEASE_VERSION") or _release_version(
        current_version
    )
    if current_version == release_version:
        print(f"pyproject.toml already uses final version {release_version}.")
        return

    updated = VERSION_RE.sub(rf'\1"{release_version}"', text, count=1)
    pyproject.write_text(updated)
    print(f"Finalized package version: {current_version} -> {release_version}")


if __name__ == "__main__":
    main()
