"""Print one field from the generated publication candidate scope file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_SCOPE_PATH = REPO_ROOT / ".github" / "publication_scope.json"
VALID_FIELDS = frozenset(
    {
        "base_release_version",
        "release_bump",
        "candidate_scope",
        "would_release_as_at_build_time",
    }
)


def read_publication_scope(path: Path = PUBLICATION_SCOPE_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing publication scope file: {path}")
    payload = json.loads(path.read_text())
    missing = sorted(VALID_FIELDS.difference(payload))
    if missing:
        raise ValueError(
            "Publication scope file is missing required field(s): " + ", ".join(missing)
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("field", choices=sorted(VALID_FIELDS))
    args = parser.parse_args()

    try:
        value = read_publication_scope(PUBLICATION_SCOPE_PATH)[args.field]
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(value)


if __name__ == "__main__":
    main()
