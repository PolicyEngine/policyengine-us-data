"""Print the stable release version corresponding to pyproject.toml."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


def main() -> None:
    with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    match = re.match(r"^(\d+\.\d+\.\d+)(?:rc\d+)?$", version)
    if not match:
        print(f"Unsupported version format: {version}", file=sys.stderr)
        sys.exit(1)
    print(match.group(1))


if __name__ == "__main__":
    main()
