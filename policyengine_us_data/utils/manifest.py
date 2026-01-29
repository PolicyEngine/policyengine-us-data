"""
Manifest utilities for atomic deployment of local area H5 files.

Provides checksum computation, manifest generation, and verification
for ensuring data integrity during uploads.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def compute_file_checksum(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 hash string
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_manifest(
    staging_dir: Path,
    version: str,
    categories: Optional[List[str]] = None,
) -> Dict:
    """
    Generate manifest.json for all H5 files in staging directory.

    Args:
        staging_dir: Root staging directory (contains version subdirs)
        version: Version string (e.g., "1.56.0")
        categories: List of categories to include (default: states, districts,
            cities)

    Returns:
        Manifest dictionary with structure:
        {
            "version": "1.56.0",
            "created_at": "2026-01-29T12:00:00Z",
            "files": {
                "states/AL.h5": {"sha256": "...", "size_bytes": 12345},
                ...
            },
            "totals": {
                "states": 50,
                "districts": 435,
                "cities": 1,
                "total_size_bytes": 987654321
            }
        }
    """
    if categories is None:
        categories = ["states", "districts", "cities"]

    manifest = {
        "version": version,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "files": {},
        "totals": {cat: 0 for cat in categories},
    }
    manifest["totals"]["total_size_bytes"] = 0

    version_dir = staging_dir / version

    for category in categories:
        category_dir = version_dir / category
        if not category_dir.exists():
            continue

        for h5_file in sorted(category_dir.glob("*.h5")):
            rel_path = f"{category}/{h5_file.name}"
            file_size = h5_file.stat().st_size

            manifest["files"][rel_path] = {
                "sha256": compute_file_checksum(h5_file),
                "size_bytes": file_size,
            }
            manifest["totals"][category] += 1
            manifest["totals"]["total_size_bytes"] += file_size

    return manifest


def verify_manifest(staging_dir: Path, manifest: Dict) -> Dict:
    """
    Verify all files in manifest exist and have correct checksums.

    Args:
        staging_dir: Root staging directory
        manifest: Manifest dictionary to verify against

    Returns:
        Verification result:
        {
            "valid": True/False,
            "missing": ["states/AL.h5", ...],
            "checksum_mismatch": ["districts/CA-01.h5", ...],
            "verified": 486
        }
    """
    version = manifest["version"]
    version_dir = staging_dir / version

    result = {
        "valid": True,
        "missing": [],
        "checksum_mismatch": [],
        "verified": 0,
    }

    for rel_path, file_info in manifest["files"].items():
        file_path = version_dir / rel_path

        if not file_path.exists():
            result["missing"].append(rel_path)
            result["valid"] = False
            continue

        actual_checksum = compute_file_checksum(file_path)
        if actual_checksum != file_info["sha256"]:
            result["checksum_mismatch"].append(rel_path)
            result["valid"] = False
            continue

        result["verified"] += 1

    return result


def save_manifest(manifest: Dict, output_path: Path) -> None:
    """Save manifest to JSON file."""
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(manifest_path: Path) -> Dict:
    """Load manifest from JSON file."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def create_latest_pointer(
    version: str,
    previous_version: Optional[str] = None,
    previous_versions: Optional[List[Dict]] = None,
) -> Dict:
    """
    Create latest.json pointer structure.

    Args:
        version: Current version to point to
        previous_version: The version being replaced (will be added to history)
        previous_versions: Existing version history (from old latest.json)

    Returns:
        Latest pointer dictionary:
        {
            "current_version": "1.56.0",
            "updated_at": "2026-01-29T12:00:00Z",
            "manifest_url": "v1.56.0/manifest.json",
            "previous_versions": [...]
        }
    """
    now = datetime.utcnow().isoformat() + "Z"

    history = []
    if previous_version:
        history.append({"version": previous_version, "deprecated_at": now})
    if previous_versions:
        history.extend(previous_versions[:9])

    return {
        "current_version": version,
        "updated_at": now,
        "manifest_url": f"v{version}/manifest.json",
        "previous_versions": history,
    }
