"""Infer release candidate scope from towncrier fragment types."""

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policyengine_us_data.utils.run_context import (  # noqa: E402
    build_candidate_scope,
    release_version_from_bump,
)


VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$")
PUBLICATION_SCOPE_PATH = Path(".github/publication_scope.json")


def get_current_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text()
    match = VERSION_RE.search(text)
    if not match:
        print(
            "Could not find version in pyproject.toml",
            file=sys.stderr,
        )
        sys.exit(1)
    return match.group(1)


def infer_bump(changelog_dir: Path) -> str:
    fragments = [
        f for f in changelog_dir.iterdir() if f.is_file() and f.name != ".gitkeep"
    ]
    if not fragments:
        print("No changelog fragments found", file=sys.stderr)
        sys.exit(1)

    categories = {f.suffix.lstrip(".") for f in fragments}
    for f in fragments:
        parts = f.stem.split(".")
        if len(parts) >= 2:
            categories.add(parts[-1])

    if "breaking" in categories:
        return "major"
    if "added" in categories or "removed" in categories:
        return "minor"
    return "patch"


def bump_version(version: str, bump: str) -> str:
    match = SEMVER_RE.match(version)
    if not match:
        print(f"Unsupported version format: {version}", file=sys.stderr)
        sys.exit(1)
    return release_version_from_bump(version, bump)


def write_publication_scope(path: Path, payload: dict[str, str]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  Updated {path}")


def main():
    pyproject = _REPO_ROOT / "pyproject.toml"
    changelog_dir = _REPO_ROOT / "changelog.d"

    current = get_current_version(pyproject)
    bump = infer_bump(changelog_dir)
    would_release_as = bump_version(current, bump)
    candidate_scope = build_candidate_scope(current, bump)

    print(f"Base release version: {current}")
    print(f"Candidate scope: {candidate_scope}")
    print(f"Release bump: {bump}")
    print(f"Would release as at build time: {would_release_as}")

    write_publication_scope(
        _REPO_ROOT / PUBLICATION_SCOPE_PATH,
        {
            "base_release_version": current,
            "release_bump": bump,
            "candidate_scope": candidate_scope,
            "would_release_as_at_build_time": would_release_as,
        },
    )


if __name__ == "__main__":
    main()
