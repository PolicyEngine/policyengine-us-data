"""Restore candidate-scoped changelog fragments for final promotion."""

from __future__ import annotations

import filecmp
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CHANGELOG_DIR = REPO_ROOT / "changelog.d"
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


def restore_candidate_changelog(run_id: str) -> Path:
    if not run_id:
        raise RuntimeError("US_DATA_RUN_ID is required to restore changelog fragments.")

    snapshot_dir = PUBLICATION_CANDIDATES_DIR / run_id / "changelog.d"
    snapshot_fragments = _fragments(snapshot_dir)
    if not snapshot_fragments:
        raise RuntimeError(
            f"No candidate changelog fragments found for run {run_id}: {snapshot_dir}"
        )

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
