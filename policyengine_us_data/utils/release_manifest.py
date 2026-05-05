from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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


def _without_none_values(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _runtime_component_metadata(
    *,
    name: str,
    version: str | None,
    git_sha: str | None = None,
    data_build_fingerprint: str | None = None,
    core_package_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if version is None:
        return None

    metadata = _without_none_values(
        {
            "name": name,
            "version": version,
            "git_sha": git_sha,
            "data_build_fingerprint": data_build_fingerprint,
        }
    )
    if core_package_metadata is not None:
        metadata["core"] = dict(core_package_metadata)
    return metadata


def _build_metadata(
    *,
    pipeline_run_id: str | None,
    data_package_git_sha: str | None,
    run_context: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    metadata = _without_none_values(
        {
            "pipeline_run_id": pipeline_run_id,
            "data_package_git_sha": data_package_git_sha,
        }
    )
    if run_context:
        metadata["run_context"] = dict(run_context)
    return metadata


def _core_version(core_package_metadata: Mapping[str, Any] | None) -> str | None:
    if core_package_metadata is None:
        return None
    version = core_package_metadata.get("version")
    return version if isinstance(version, str) and version else None


def _base_manifest(
    *,
    version: str,
    data_package_name: str,
    model_package_name: str,
    model_package_version: str | None,
    model_package_git_sha: str | None,
    model_package_data_build_fingerprint: str | None,
    core_package_metadata: Mapping[str, Any] | None,
    run_context: Mapping[str, Any] | None,
    build_id: str,
    created_at: str,
    additional_compatible_specifiers: Sequence[str] | None = None,
    pipeline_run_id: str | None,
    data_package_git_sha: str | None,
) -> Dict:
    build_metadata = _build_metadata(
        pipeline_run_id=pipeline_run_id,
        data_package_git_sha=data_package_git_sha,
        run_context=run_context,
    )
    manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": data_package_name,
            "version": version,
        },
        "compatible_model_packages": [],
        "compatible_core_packages": [],
        "default_datasets": {},
        "build": {
            "build_id": build_id,
            "built_at": created_at,
        },
        "artifacts": {},
    }
    if build_metadata:
        manifest["build"]["metadata"] = build_metadata
    model_package_metadata = _runtime_component_metadata(
        name=model_package_name,
        version=model_package_version,
        git_sha=model_package_git_sha,
        data_build_fingerprint=model_package_data_build_fingerprint,
        core_package_metadata=core_package_metadata,
    )
    if model_package_metadata is not None:
        manifest["build"]["built_with_model_package"] = model_package_metadata
    if core_package_metadata is not None:
        manifest["build"]["built_with_core_package"] = dict(core_package_metadata)
    if model_package_version:
        manifest["compatible_model_packages"].append(
            {
                "name": model_package_name,
                "specifier": f"=={model_package_version}",
            }
        )
    for specifier in additional_compatible_specifiers or ():
        manifest["compatible_model_packages"].append(
            {"name": model_package_name, "specifier": specifier}
        )
    core_version = _core_version(core_package_metadata)
    if core_version:
        manifest["compatible_core_packages"].append(
            {
                "name": core_package_metadata.get("name", "policyengine-core"),
                "specifier": f"=={core_version}",
            }
        )
    return manifest


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
    manifest = deepcopy(dict(existing_manifest))
    manifest.pop("created_at", None)
    build = manifest.get("build")
    if isinstance(build, dict):
        legacy_run = build.pop("run", None)
        if legacy_run:
            build.setdefault("metadata", {}).setdefault("run_context", legacy_run)
    return manifest


def build_release_manifest(
    *,
    files_with_repo_paths: Sequence[Tuple[Path | str, str]],
    version: str,
    repo_id: str,
    data_package_name: str = "policyengine-us-data",
    model_package_name: str = "policyengine-us",
    model_package_version: str | None = None,
    model_package_git_sha: str | None = None,
    model_package_data_build_fingerprint: str | None = None,
    core_package_metadata: Optional[Mapping[str, Any]] = None,
    run_context: Mapping[str, Any] | None = None,
    build_id: str | None = None,
    pipeline_run_id: str | None = None,
    data_package_git_sha: str | None = None,
    existing_manifest: Mapping | None = None,
    default_datasets: Optional[Mapping[str, str]] = None,
    created_at: str | None = None,
    additional_compatible_specifiers: Sequence[str] | None = None,
) -> Dict:
    manifest = _normalize_existing_manifest(
        existing_manifest,
        version=version,
        data_package_name=data_package_name,
    )
    manifest_timestamp = created_at or _utc_timestamp()
    resolved_build_id = build_id or f"{data_package_name}-{version}"

    if manifest is None:
        manifest = _base_manifest(
            version=version,
            data_package_name=data_package_name,
            model_package_name=model_package_name,
            model_package_version=model_package_version,
            model_package_git_sha=model_package_git_sha,
            model_package_data_build_fingerprint=model_package_data_build_fingerprint,
            core_package_metadata=core_package_metadata,
            run_context=run_context,
            build_id=resolved_build_id,
            created_at=manifest_timestamp,
            additional_compatible_specifiers=additional_compatible_specifiers,
            pipeline_run_id=pipeline_run_id,
            data_package_git_sha=data_package_git_sha,
        )
    else:
        manifest["schema_version"] = RELEASE_MANIFEST_SCHEMA_VERSION
        manifest.setdefault("compatible_core_packages", [])
        manifest.setdefault("build", {})
        manifest["build"].setdefault("build_id", resolved_build_id)
        manifest["build"].setdefault("built_at", manifest_timestamp)
        build_metadata = _build_metadata(
            pipeline_run_id=pipeline_run_id,
            data_package_git_sha=data_package_git_sha,
            run_context=run_context,
        )
        if build_metadata:
            manifest["build"].setdefault("metadata", {}).update(build_metadata)
        model_package_metadata = _runtime_component_metadata(
            name=model_package_name,
            version=model_package_version,
            git_sha=model_package_git_sha,
            data_build_fingerprint=model_package_data_build_fingerprint,
            core_package_metadata=core_package_metadata,
        )
        if model_package_metadata is not None:
            manifest["build"]["built_with_model_package"] = model_package_metadata
        if core_package_metadata is not None:
            manifest["build"]["built_with_core_package"] = dict(core_package_metadata)
        compat = []
        if model_package_version:
            compat.append(
                {
                    "name": model_package_name,
                    "specifier": f"=={model_package_version}",
                }
            )
        for specifier in additional_compatible_specifiers or ():
            compat.append({"name": model_package_name, "specifier": specifier})
        if compat:
            manifest["compatible_model_packages"] = compat
        core_version = _core_version(core_package_metadata)
        if core_version:
            manifest["compatible_core_packages"] = [
                {
                    "name": core_package_metadata.get("name", "policyengine-core"),
                    "specifier": f"=={core_version}",
                }
            ]

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
