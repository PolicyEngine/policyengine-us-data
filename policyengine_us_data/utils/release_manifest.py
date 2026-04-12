from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Optional, Sequence, Tuple

from policyengine_us_data.utils.manifest import compute_file_checksum

RELEASE_MANIFEST_SCHEMA_VERSION = 1


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact_key(path_in_repo: str) -> str:
    return str(PurePosixPath(path_in_repo).with_suffix(""))


def _artifact_kind(path_in_repo: str) -> str:
    suffix = PurePosixPath(path_in_repo).suffix.lower()
    if suffix == ".h5":
        return "microdata"
    if suffix == ".db":
        return "database"
    if suffix == ".npz":
        return "geography"
    if suffix == ".npy":
        return "weights"
    return "auxiliary"


def _base_manifest(
    *,
    version: str,
    data_package_name: str,
    compatible_model_packages: Sequence[Dict[str, str]],
    created_at: str,
) -> Dict:
    manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": data_package_name,
            "version": version,
        },
        "compatible_model_packages": list(compatible_model_packages),
        "default_datasets": {},
        "created_at": created_at,
        "artifacts": {},
    }
    return manifest


def _compatible_model_packages(
    model_package_name: str,
    model_package_version: str | None,
) -> list[Dict[str, str]]:
    if not model_package_version:
        return []
    return [{"name": model_package_name, "version": model_package_version}]


def _normalize_existing_manifest(
    existing_manifest: Mapping | None,
    *,
    version: str,
    data_package_name: str,
) -> Dict | None:
    if existing_manifest is None:
        return None
    package = existing_manifest.get("data_package", {})
    if package.get("name") != data_package_name or package.get("version") != version:
        return None
    return deepcopy(dict(existing_manifest))


def build_release_manifest(
    *,
    files_with_repo_paths: Sequence[Tuple[Path | str, str]],
    version: str,
    repo_id: str,
    data_package_name: str = "policyengine-us-data",
    model_package_name: str = "policyengine-us",
    model_package_version: str | None = None,
    existing_manifest: Mapping | None = None,
    default_datasets: Optional[Mapping[str, str]] = None,
    created_at: str | None = None,
) -> Dict:
    manifest = _normalize_existing_manifest(
        existing_manifest,
        version=version,
        data_package_name=data_package_name,
    )
    manifest_timestamp = created_at or _utc_timestamp()

    if manifest is None:
        manifest = _base_manifest(
            version=version,
            data_package_name=data_package_name,
            compatible_model_packages=_compatible_model_packages(
                model_package_name,
                model_package_version,
            ),
            created_at=manifest_timestamp,
        )
    else:
        manifest["schema_version"] = RELEASE_MANIFEST_SCHEMA_VERSION
        manifest["created_at"] = manifest.get("created_at") or manifest_timestamp
        manifest["compatible_model_packages"] = _compatible_model_packages(
            model_package_name,
            model_package_version,
        )

    if default_datasets:
        manifest.setdefault("default_datasets", {}).update(default_datasets)

    for local_path, path_in_repo in files_with_repo_paths:
        local_path = Path(local_path)
        manifest["artifacts"][_artifact_key(path_in_repo)] = {
            "kind": _artifact_kind(path_in_repo),
            "path": path_in_repo,
            "repo_id": repo_id,
            "revision": version,
            "sha256": compute_file_checksum(local_path),
            "size_bytes": local_path.stat().st_size,
        }

    if (
        "national" not in manifest["default_datasets"]
        and "enhanced_cps_2024" in manifest["artifacts"]
    ):
        manifest["default_datasets"]["national"] = "enhanced_cps_2024"

    return manifest


def serialize_release_manifest(manifest: Mapping) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
