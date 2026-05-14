"""Print the stable release version corresponding to pyproject.toml."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)(?:rc\d+)?$")


def main() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    match = VERSION_RE.match(version)
    if not match:
        print(f"Unsupported version format: {version}", file=sys.stderr)
        sys.exit(1)
    print(match.group(1))


if __name__ == "__main__":
    main()
