"""Synchronize pyproject.toml with the latest finalized HF data release."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from urllib.error import URLError

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from check_data_release_version import (
    BEHIND,
    DEFAULT_VERSION_MANIFEST_URL,
    REPO_ROOT,
    check_repository_state,
)


VERSION_RE = re.compile(r'^(version\s*=\s*)"([^"]+)"', re.MULTILINE)


def update_pyproject_version(pyproject: Path, release_version: str) -> str:
    text = pyproject.read_text()
    match = VERSION_RE.search(text)
    if not match:
        raise ValueError("Could not find project version in pyproject.toml")

    current_version = match.group(2)
    if current_version == release_version:
        return current_version

    updated = VERSION_RE.sub(rf'\1"{release_version}"', text, count=1)
    pyproject.write_text(updated)
    return current_version


def sync_finalized_data_release_version(
    root: Path | None = None,
    *,
    finalized_release_version: str | None = None,
    version_manifest_url: str = DEFAULT_VERSION_MANIFEST_URL,
) -> bool:
    root = root or REPO_ROOT
    state = check_repository_state(
        root,
        finalized_release_version=finalized_release_version,
        version_manifest_url=version_manifest_url,
    )
    if state.release_version_relation != BEHIND:
        print(
            "No finalized data release version sync needed: "
            f"package={state.package_version}, "
            f"finalized={state.finalized_release_version}, "
            f"relation={state.release_version_relation}."
        )
        return False

    previous_version = update_pyproject_version(
        root / "pyproject.toml",
        state.finalized_release_version,
    )
    print(
        "Synchronized pyproject.toml with finalized HF data release: "
        f"{previous_version} -> {state.finalized_release_version}."
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version-manifest-url",
        default=os.environ.get(
            "US_DATA_VERSION_MANIFEST_URL", DEFAULT_VERSION_MANIFEST_URL
        ),
    )
    parser.add_argument(
        "--finalized-release-version",
        default=os.environ.get("US_DATA_FINALIZED_RELEASE_VERSION"),
        help="Already-resolved finalized HF release version to sync to.",
    )
    args = parser.parse_args(argv)

    try:
        sync_finalized_data_release_version(
            finalized_release_version=args.finalized_release_version,
            version_manifest_url=args.version_manifest_url,
        )
    except (URLError, OSError, ValueError) as exc:
        print(
            f"Could not synchronize finalized HF data release version: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
