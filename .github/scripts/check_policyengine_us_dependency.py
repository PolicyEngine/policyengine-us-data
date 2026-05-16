"""Check whether the locked PolicyEngine US dependency is current."""

from __future__ import annotations

import argparse
import json
import os
from itertools import zip_longest
from pathlib import Path
import re
import sys
import tomllib
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPI_JSON_TIMEOUT_SECONDS = 20
POLICYENGINE_US = "policyengine-us"
STALE_LOCK_PREFIX = "uv.lock has policyengine-us "


def _annotation(level: str, message: str) -> str:
    escaped = message.replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
    return f"::{level} title=PolicyEngine US dependency::{escaped}"


def _version_key(version: str) -> tuple[int, ...]:
    release = version.split("+", 1)[0].split("-", 1)[0]
    if not re.fullmatch(r"\d+(?:\.\d+)*", release):
        raise ValueError(f"Unsupported version format: {version}")
    return tuple(int(part) for part in release.split("."))


def _compare_versions(left: str, right: str) -> int:
    for left_part, right_part in zip_longest(
        _version_key(left), _version_key(right), fillvalue=0
    ):
        if left_part < right_part:
            return -1
        if left_part > right_part:
            return 1
    return 0


def _locked_policyengine_us(root: Path) -> tuple[str, dict[str, object]]:
    with (root / "uv.lock").open("rb") as file:
        lock = tomllib.load(file)

    for package in lock.get("package", []):
        if package.get("name") == POLICYENGINE_US:
            version = package.get("version")
            if not isinstance(version, str):
                raise ValueError("uv.lock entry for policyengine-us has no version")
            source = package.get("source", {})
            return version, source if isinstance(source, dict) else {}

    raise ValueError("uv.lock does not contain policyengine-us")


def _project_policyengine_us_dependency(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)

    dependencies = pyproject.get("project", {}).get("dependencies", [])
    for dependency in dependencies:
        if isinstance(dependency, str) and dependency.startswith(POLICYENGINE_US):
            return dependency

    raise ValueError("pyproject.toml does not declare policyengine-us")


def _latest_pypi_version() -> str:
    with urlopen(
        f"https://pypi.org/pypi/{POLICYENGINE_US}/json",
        timeout=PYPI_JSON_TIMEOUT_SECONDS,
    ) as response:
        payload = json.load(response)

    version = payload.get("info", {}).get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("PyPI response does not include info.version")
    return version


def check_dependency(root: Path, latest_version: str | None = None) -> list[str]:
    locked_version, source = _locked_policyengine_us(root)
    project_dependency = _project_policyengine_us_dependency(root)

    violations: list[str] = []
    if (
        latest_version is not None
        and _compare_versions(locked_version, latest_version) < 0
    ):
        violations.append(
            f"{STALE_LOCK_PREFIX}{locked_version}, but PyPI latest is "
            f"{latest_version}. Update pyproject.toml and run 'uv lock', or "
            "explicitly document why the older model is required."
        )

    expected_dependency = f"{POLICYENGINE_US}=={locked_version}"
    if project_dependency != expected_dependency:
        violations.append(
            f"pyproject.toml must pin {expected_dependency} to match uv.lock; "
            f"found {project_dependency!r}."
        )

    if "git" in source:
        violations.append(
            "uv.lock resolves policyengine-us from a Git ref. Prefer an exact "
            f"PyPI release pin once policyengine-us {locked_version} is published."
        )

    if "@" in project_dependency and "git+" in project_dependency:
        violations.append(
            "pyproject.toml pins policyengine-us to a Git ref. Prefer an exact "
            "PyPI release pin for production data builds."
        )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("warn", "fail"),
        default="warn",
        help="Whether stale dependency findings should fail the command.",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Report stale dependency findings but exit successfully.",
    )
    args = parser.parse_args()
    allow_stale = args.allow_stale or os.environ.get(
        "POLICYENGINE_US_ALLOW_STALE", ""
    ).lower() in {"1", "true", "yes"}

    latest_version = None
    try:
        latest_version = _latest_pypi_version()
    except (OSError, URLError, ValueError) as exc:
        message = (
            "Unable to fetch the latest policyengine-us version from PyPI; "
            f"continuing with local dependency checks only: {exc}"
        )
        print(_annotation("warning", message))

    try:
        violations = check_dependency(REPO_ROOT, latest_version=latest_version)
    except (OSError, ValueError) as exc:
        message = f"Unable to check policyengine-us freshness: {exc}"
        if args.mode == "fail":
            print(_annotation("error", message), file=sys.stderr)
            return 1
        print(_annotation("warning", message))
        return 0

    if not violations:
        locked_version, _source = _locked_policyengine_us(REPO_ROOT)
        print(f"policyengine-us dependency is current at {locked_version}.")
        return 0

    has_blocking_violation = False
    allowed_stale_version = False
    for violation in violations:
        stale_version_violation = violation.startswith(STALE_LOCK_PREFIX)
        allowed_by_override = allow_stale and stale_version_violation
        level = "warning" if args.mode == "warn" or allowed_by_override else "error"
        print(_annotation(level, violation))
        if args.mode == "fail" and not allowed_by_override:
            has_blocking_violation = True
        if allowed_by_override:
            allowed_stale_version = True

    if allowed_stale_version:
        print(
            _annotation(
                "warning",
                "POLICYENGINE_US_ALLOW_STALE is set; continuing despite "
                "policyengine-us lagging the latest PyPI release.",
            )
        )

    if has_blocking_violation:
        return 1
    if allowed_stale_version:
        return 0

    return 1 if args.mode == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
