"""Infer release candidate scope from towncrier fragment types."""

import json
import re
import os
import shutil
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
PUBLICATION_CANDIDATES_DIR = Path(".github/publication_candidates")
CHANGELOG_KEEP_FILE = ".gitkeep"


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
    fragments = changelog_fragments(changelog_dir)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  Updated {path}")


def changelog_fragments(changelog_dir: Path) -> list[Path]:
    return sorted(
        f
        for f in changelog_dir.iterdir()
        if f.is_file() and f.name != CHANGELOG_KEEP_FILE
    )


def snapshot_changelog_fragments(
    *,
    run_id: str,
    changelog_dir: Path,
    publication_candidates_dir: Path,
) -> Path:
    fragments = changelog_fragments(changelog_dir)
    if not run_id:
        print(
            "US_DATA_RUN_ID is required to snapshot changelog fragments",
            file=sys.stderr,
        )
        sys.exit(1)
    if not fragments:
        print("No changelog fragments found", file=sys.stderr)
        sys.exit(1)

    snapshot_dir = publication_candidates_dir / run_id / "changelog.d"
    if snapshot_dir.exists() and changelog_fragments(snapshot_dir):
        print(
            f"Candidate changelog snapshot already exists: {snapshot_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for fragment in fragments:
        destination = snapshot_dir / fragment.name
        shutil.copy2(fragment, destination)
        fragment.unlink()
        print(f"  Snapshotted {fragment} -> {destination}")
    return snapshot_dir


def main():
    pyproject = _REPO_ROOT / "pyproject.toml"
    changelog_dir = _REPO_ROOT / "changelog.d"
    run_id = os.environ.get("US_DATA_RUN_ID", "")

    current = get_current_version(pyproject)
    bump = infer_bump(changelog_dir)
    would_release_as = bump_version(current, bump)
    candidate_scope = build_candidate_scope(current, bump)

    print(f"Base release version: {current}")
    print(f"Candidate scope: {candidate_scope}")
    print(f"Release bump: {bump}")
    print(f"Would release as at build time: {would_release_as}")

    snapshot_changelog_fragments(
        run_id=run_id,
        changelog_dir=changelog_dir,
        publication_candidates_dir=_REPO_ROOT / PUBLICATION_CANDIDATES_DIR,
    )
    payload = {
        "run_id": run_id,
        "base_release_version": current,
        "release_bump": bump,
        "candidate_scope": candidate_scope,
        "would_release_as_at_build_time": would_release_as,
    }
    write_publication_scope(
        _REPO_ROOT / PUBLICATION_SCOPE_PATH,
        payload,
    )
    write_publication_scope(
        _REPO_ROOT / PUBLICATION_CANDIDATES_DIR / run_id / PUBLICATION_SCOPE_PATH.name,
        payload,
    )


if __name__ == "__main__":
    main()
