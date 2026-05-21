"""Require pyproject.toml to track the latest finalized HF data release."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERSION_MANIFEST_URL = (
    "https://huggingface.co/policyengine/policyengine-us-data/"
    "resolve/main/version_manifest.json"
)
VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:rc\d+)?$")
CURRENT = "current"
BEHIND = "behind"
AHEAD = "ahead"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReleaseVersionState:
    package_version: str
    finalized_release_version: str
    release_version_relation: str


def stable_version_tuple(version: str) -> tuple[int, int, int]:
    match = SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"Unsupported version format: {version}")
    return tuple(int(part) for part in match.groups())


def release_version_relation(
    *,
    package_version: str,
    finalized_release_version: str,
) -> str:
    package_tuple = stable_version_tuple(package_version)
    finalized_tuple = stable_version_tuple(finalized_release_version)
    if package_tuple < finalized_tuple:
        return BEHIND
    if package_tuple > finalized_tuple:
        return AHEAD
    return CURRENT


def pyproject_version(root: Path | None = None) -> str:
    root = root or REPO_ROOT
    text = (root / "pyproject.toml").read_text()
    match = VERSION_RE.search(text)
    if not match:
        raise ValueError("Could not find project version in pyproject.toml")
    return match.group(1)


def latest_hf_release_version(
    url: str = DEFAULT_VERSION_MANIFEST_URL,
) -> str:
    with urlopen(url, timeout=30) as response:
        payload = json.load(response)
    current = payload.get("current")
    if isinstance(current, str) and current:
        return current
    versions = payload.get("versions")
    if not isinstance(versions, list) or not versions:
        raise ValueError("HF version_manifest.json has no current version")
    latest = versions[-1].get("version")
    if not isinstance(latest, str) or not latest:
        raise ValueError("HF version_manifest.json latest entry has no version")
    return latest


def version_violations(
    *,
    package_version: str,
    finalized_release_version: str,
) -> list[str]:
    if (
        release_version_relation(
            package_version=package_version,
            finalized_release_version=finalized_release_version,
        )
        != BEHIND
    ):
        return []
    return [
        "pyproject.toml version "
        f"{package_version} is behind finalized HF data release "
        f"{finalized_release_version}. Finalize the package version before "
        "creating another publication candidate."
    ]


def check_repository_state(
    root: Path | None = None,
    *,
    finalized_release_version: str | None = None,
    version_manifest_url: str = DEFAULT_VERSION_MANIFEST_URL,
) -> ReleaseVersionState:
    root = root or REPO_ROOT
    package_version = pyproject_version(root)
    finalized_release_version = finalized_release_version or latest_hf_release_version(
        version_manifest_url
    )
    relation = release_version_relation(
        package_version=package_version,
        finalized_release_version=finalized_release_version,
    )
    return ReleaseVersionState(
        package_version=package_version,
        finalized_release_version=finalized_release_version,
        release_version_relation=relation,
    )


def check_repository(
    root: Path | None = None,
    *,
    finalized_release_version: str | None = None,
    version_manifest_url: str = DEFAULT_VERSION_MANIFEST_URL,
) -> list[str]:
    state = check_repository_state(
        root,
        finalized_release_version=finalized_release_version,
        version_manifest_url=version_manifest_url,
    )
    return version_violations(
        package_version=state.package_version,
        finalized_release_version=state.finalized_release_version,
    )


def write_github_outputs(state: ReleaseVersionState) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a") as output:
        output.write(f"package_version={state.package_version}\n")
        output.write(f"finalized_release_version={state.finalized_release_version}\n")
        output.write(f"release_version_relation={state.release_version_relation}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("warn", "fail"),
        default="fail",
        help="Whether stale versions should fail the command.",
    )
    parser.add_argument(
        "--version-manifest-url",
        default=os.environ.get(
            "US_DATA_VERSION_MANIFEST_URL", DEFAULT_VERSION_MANIFEST_URL
        ),
    )
    args = parser.parse_args(argv)

    try:
        package_version = pyproject_version()
        stable_version_tuple(package_version)
    except (OSError, ValueError) as exc:
        write_github_outputs(
            ReleaseVersionState(
                package_version="",
                finalized_release_version="",
                release_version_relation=UNKNOWN,
            )
        )
        print(f"Could not read data package version: {exc}", file=sys.stderr)
        return 1

    try:
        finalized_release_version = latest_hf_release_version(args.version_manifest_url)
        state = ReleaseVersionState(
            package_version=package_version,
            finalized_release_version=finalized_release_version,
            release_version_relation=release_version_relation(
                package_version=package_version,
                finalized_release_version=finalized_release_version,
            ),
        )
    except (URLError, OSError, ValueError) as exc:
        write_github_outputs(
            ReleaseVersionState(
                package_version=package_version,
                finalized_release_version="",
                release_version_relation=UNKNOWN,
            )
        )
        print(
            f"Could not check finalized HF data release version: {exc}", file=sys.stderr
        )
        return 1 if args.mode == "fail" else 0

    write_github_outputs(state)
    violations = version_violations(
        package_version=state.package_version,
        finalized_release_version=state.finalized_release_version,
    )
    if not violations:
        if state.release_version_relation == AHEAD:
            print(
                "Data package version "
                f"{state.package_version} is ahead of finalized HF release "
                f"{state.finalized_release_version}."
            )
        else:
            print(
                "Data package version is current with the latest finalized HF release."
            )
        return 0

    for violation in violations:
        print(violation, file=sys.stderr)
    return 1 if args.mode == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
