"""Wait until the package version in pyproject.toml is visible on PyPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPI_JSON_TIMEOUT_SECONDS = 20


def _package_metadata() -> tuple[str, str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)
    project = pyproject["project"]
    return project["name"], project["version"]


def _normalize_package_name(package_name: str) -> str:
    return re.sub(r"[-_.]+", "-", package_name).lower()


def _pypi_version_exists(package_name: str, version: str) -> bool:
    normalized_name = _normalize_package_name(package_name)
    url = f"https://pypi.org/pypi/{normalized_name}/{version}/json"
    try:
        with urlopen(url, timeout=PYPI_JSON_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    except URLError:
        return False
    return payload.get("info", {}).get("version") == version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=int, default=15)
    args = parser.parse_args()

    package_name, version = _package_metadata()
    for attempt in range(1, args.attempts + 1):
        if _pypi_version_exists(package_name, version):
            print(f"PyPI has {package_name}=={version}.")
            return
        print(
            f"PyPI does not have {package_name}=={version} yet "
            f"(attempt {attempt}/{args.attempts})."
        )
        if attempt < args.attempts:
            time.sleep(args.sleep_seconds)

    print(
        f"Timed out waiting for PyPI to expose {package_name}=={version}.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
