"""Restore candidate-scoped changelog fragments for final promotion."""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CHANGELOG_DIR = REPO_ROOT / "changelog.d"
PUBLICATION_SCOPE_PATH = REPO_ROOT / ".github" / "publication_scope.json"
PUBLICATION_CANDIDATES_DIR = REPO_ROOT / ".github" / "publication_candidates"
CHANGELOG_KEEP_FILE = ".gitkeep"


def _fragments(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.name != CHANGELOG_KEEP_FILE
    )


def _validate_root_fragments_match_snapshot(
    *,
    root_fragments: list[Path],
    snapshot_fragments: list[Path],
) -> None:
    snapshot_by_name = {fragment.name: fragment for fragment in snapshot_fragments}
    root_by_name = {fragment.name: fragment for fragment in root_fragments}
    extra = sorted(set(root_by_name).difference(snapshot_by_name))
    missing = sorted(set(snapshot_by_name).difference(root_by_name))
    changed = sorted(
        name
        for name in set(root_by_name).intersection(snapshot_by_name)
        if not filecmp.cmp(root_by_name[name], snapshot_by_name[name], shallow=False)
    )
    if extra or missing or changed:
        details = []
        if extra:
            details.append(f"extra root fragments: {', '.join(extra)}")
        if missing:
            details.append(f"missing root fragments: {', '.join(missing)}")
        if changed:
            details.append(f"changed root fragments: {', '.join(changed)}")
        raise RuntimeError(
            "Root changelog fragments do not match the candidate snapshot; "
            + "; ".join(details)
        )


def _publication_scope() -> dict[str, str]:
    if not PUBLICATION_SCOPE_PATH.exists():
        return {}
    return json.loads(PUBLICATION_SCOPE_PATH.read_text())


def _scope_matches_environment(scope: dict[str, str]) -> bool:
    expected = {
        "candidate_scope": os.environ.get("US_DATA_CANDIDATE_VERSION")
        or os.environ.get("CANDIDATE_VERSION"),
        "base_release_version": os.environ.get("US_DATA_BASE_RELEASE_VERSION")
        or os.environ.get("BASE_RELEASE_VERSION"),
        "release_bump": os.environ.get("US_DATA_RELEASE_BUMP")
        or os.environ.get("RELEASE_BUMP"),
    }
    return all(
        not value or not scope.get(key) or scope[key] == value
        for key, value in expected.items()
    )


def _snapshot_dir(run_id: str) -> Path:
    return PUBLICATION_CANDIDATES_DIR / run_id / "changelog.d"


def _resolve_snapshot_dir(run_id: str) -> Path:
    requested_dir = _snapshot_dir(run_id)
    if _fragments(requested_dir):
        return requested_dir

    scope = _publication_scope()
    scope_run_id = scope.get("run_id", "")
    if scope_run_id and scope_run_id != run_id and _scope_matches_environment(scope):
        scope_dir = _snapshot_dir(scope_run_id)
        if _fragments(scope_dir):
            print(
                "No changelog snapshot found for promotion run "
                f"{run_id}; using publication scope snapshot {scope_run_id}.",
            )
            return scope_dir

    details = (
        f"No candidate changelog fragments found for run {run_id}: {requested_dir}"
    )
    if scope_run_id and scope_run_id != run_id:
        details += f"; publication scope points to {scope_run_id}: {_snapshot_dir(scope_run_id)}"
    raise RuntimeError(details)


def restore_candidate_changelog(run_id: str) -> Path:
    if not run_id:
        raise RuntimeError("US_DATA_RUN_ID is required to restore changelog fragments.")

    snapshot_dir = _resolve_snapshot_dir(run_id)
    snapshot_fragments = _fragments(snapshot_dir)

    ROOT_CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)
    root_fragments = _fragments(ROOT_CHANGELOG_DIR)
    if root_fragments:
        _validate_root_fragments_match_snapshot(
            root_fragments=root_fragments,
            snapshot_fragments=snapshot_fragments,
        )

    for fragment in snapshot_fragments:
        destination = ROOT_CHANGELOG_DIR / fragment.name
        shutil.copy2(fragment, destination)
        print(f"Restored {destination} from {fragment}")

    return snapshot_dir


def main() -> None:
    try:
        snapshot_dir = restore_candidate_changelog(os.environ.get("US_DATA_RUN_ID", ""))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"Restored candidate changelog fragments from {snapshot_dir}")


if __name__ == "__main__":
    main()
