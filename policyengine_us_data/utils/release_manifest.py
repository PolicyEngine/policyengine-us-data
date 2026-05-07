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


def _model_package_compatibility(
    *,
    model_package_name: str,
    model_package_version: str | None,
    additional_compatible_specifiers: Sequence[str] | None = None,
) -> list[Dict[str, str]]:
    compatible_packages = []
    if model_package_version:
        compatible_packages.append(
            {
                "name": model_package_name,
                "specifier": f"=={model_package_version}",
            }
        )
    for specifier in additional_compatible_specifiers or ():
        compatible_packages.append({"name": model_package_name, "specifier": specifier})
    return compatible_packages


def _core_package_compatibility(
    core_package_metadata: Mapping[str, Any] | None,
) -> list[Dict[str, str]]:
    core_version = _core_version(core_package_metadata)
    if not core_version:
        return []
    return [
        {
            "name": core_package_metadata.get("name", "policyengine-core"),
            "specifier": f"=={core_version}",
        }
    ]


def _build_section(
    *,
    model_package_name: str,
    model_package_version: str | None,
    model_package_git_sha: str | None,
    model_package_data_build_fingerprint: str | None,
    core_package_metadata: Mapping[str, Any] | None,
    run_context: Mapping[str, Any] | None,
    build_id: str,
    created_at: str,
    pipeline_run_id: str | None,
    data_package_git_sha: str | None,
) -> Dict:
    build = {
        "build_id": build_id,
        "built_at": created_at,
    }
    build_metadata = _build_metadata(
        pipeline_run_id=pipeline_run_id,
        data_package_git_sha=data_package_git_sha,
        run_context=run_context,
    )
    if build_metadata:
        build["metadata"] = build_metadata
    model_package_metadata = _runtime_component_metadata(
        name=model_package_name,
        version=model_package_version,
        git_sha=model_package_git_sha,
        data_build_fingerprint=model_package_data_build_fingerprint,
        core_package_metadata=core_package_metadata,
    )
    if model_package_metadata is not None:
        build["built_with_model_package"] = model_package_metadata
    if core_package_metadata is not None:
        build["built_with_core_package"] = dict(core_package_metadata)
    return build


def _new_manifest(
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
    return {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": data_package_name,
            "version": version,
        },
        "compatible_model_packages": _model_package_compatibility(
            model_package_name=model_package_name,
            model_package_version=model_package_version,
            additional_compatible_specifiers=additional_compatible_specifiers,
        ),
        "compatible_core_packages": _core_package_compatibility(core_package_metadata),
        "default_datasets": {},
        "build": _build_section(
            model_package_name=model_package_name,
            model_package_version=model_package_version,
            model_package_git_sha=model_package_git_sha,
            model_package_data_build_fingerprint=model_package_data_build_fingerprint,
            core_package_metadata=core_package_metadata,
            run_context=run_context,
            build_id=build_id,
            created_at=created_at,
            pipeline_run_id=pipeline_run_id,
            data_package_git_sha=data_package_git_sha,
        ),
        "artifacts": {},
    }


def _update_existing_build_section(
    manifest: Dict,
    *,
    model_package_name: str,
    model_package_version: str | None,
    model_package_git_sha: str | None,
    model_package_data_build_fingerprint: str | None,
    core_package_metadata: Mapping[str, Any] | None,
    run_context: Mapping[str, Any] | None,
    build_id: str,
    created_at: str,
    pipeline_run_id: str | None,
    data_package_git_sha: str | None,
) -> None:
    build = manifest.setdefault("build", {})
    build.setdefault("build_id", build_id)
    build.setdefault("built_at", created_at)

    build_metadata = _build_metadata(
        pipeline_run_id=pipeline_run_id,
        data_package_git_sha=data_package_git_sha,
        run_context=run_context,
    )
    if build_metadata:
        build.setdefault("metadata", {}).update(build_metadata)

    model_package_metadata = _runtime_component_metadata(
        name=model_package_name,
        version=model_package_version,
        git_sha=model_package_git_sha,
        data_build_fingerprint=model_package_data_build_fingerprint,
        core_package_metadata=core_package_metadata,
    )
    if model_package_metadata is not None:
        build["built_with_model_package"] = model_package_metadata
    if core_package_metadata is not None:
        build["built_with_core_package"] = dict(core_package_metadata)


def _update_existing_manifest_metadata(
    manifest: Dict,
    *,
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
) -> None:
    manifest["schema_version"] = RELEASE_MANIFEST_SCHEMA_VERSION
    manifest.setdefault("compatible_core_packages", [])
    _update_existing_build_section(
        manifest,
        model_package_name=model_package_name,
        model_package_version=model_package_version,
        model_package_git_sha=model_package_git_sha,
        model_package_data_build_fingerprint=model_package_data_build_fingerprint,
        core_package_metadata=core_package_metadata,
        run_context=run_context,
        build_id=build_id,
        created_at=created_at,
        pipeline_run_id=pipeline_run_id,
        data_package_git_sha=data_package_git_sha,
    )

    compatible_model_packages = _model_package_compatibility(
        model_package_name=model_package_name,
        model_package_version=model_package_version,
        additional_compatible_specifiers=additional_compatible_specifiers,
    )
    if compatible_model_packages:
        manifest["compatible_model_packages"] = compatible_model_packages

    compatible_core_packages = _core_package_compatibility(core_package_metadata)
    if compatible_core_packages:
        manifest["compatible_core_packages"] = compatible_core_packages


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


def _artifact_entry(
    *,
    local_path: Path,
    path_in_repo: str,
    repo_id: str,
    version: str,
) -> Dict[str, Any]:
    return {
        "kind": _artifact_kind(path_in_repo),
        "path": path_in_repo,
        "repo_id": repo_id,
        "revision": version,
        "sha256": compute_file_checksum(local_path),
        "size_bytes": local_path.stat().st_size,
    }


def _update_artifacts(
    manifest: Dict,
    *,
    files_with_repo_paths: Sequence[Tuple[Path | str, str]],
    repo_id: str,
    version: str,
    preservation_mirrors_by_artifact: (
        Mapping[str, Sequence[Mapping[str, Any]]] | None
    ) = None,
) -> None:
    artifacts = manifest.setdefault("artifacts", {})
    for local_path, path_in_repo in files_with_repo_paths:
        local_path = Path(local_path)
        artifact_key = _artifact_key(path_in_repo)
        artifacts[artifact_key] = _artifact_entry(
            local_path=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            version=version,
        )
        if preservation_mirrors_by_artifact:
            mirrors = preservation_mirrors_by_artifact.get(artifact_key)
            if mirrors:
                artifacts[artifact_key]["preservation_mirrors"] = [
                    dict(mirror) for mirror in mirrors
                ]


def _update_default_datasets(
    manifest: Dict,
    default_datasets: Optional[Mapping[str, str]] = None,
) -> None:
    defaults = manifest.setdefault("default_datasets", {})
    if default_datasets:
        defaults.update(default_datasets)
    if "national" not in defaults and "enhanced_cps_2024" in manifest["artifacts"]:
        defaults["national"] = "enhanced_cps_2024"


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
    preservation_mirrors_by_artifact: Optional[
        Mapping[str, Sequence[Mapping[str, Any]]]
    ] = None,
    preservation_dois: Optional[Sequence[str]] = None,
) -> Dict:
    manifest = _normalize_existing_manifest(
        existing_manifest,
        version=version,
        data_package_name=data_package_name,
    )
    manifest_timestamp = created_at or _utc_timestamp()
    resolved_build_id = build_id or f"{data_package_name}-{version}"

    if manifest is None:
        manifest = _new_manifest(
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
        _update_existing_manifest_metadata(
            manifest,
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

    _update_artifacts(
        manifest,
        files_with_repo_paths=files_with_repo_paths,
        repo_id=repo_id,
        version=version,
        preservation_mirrors_by_artifact=preservation_mirrors_by_artifact,
    )
    _update_default_datasets(manifest, default_datasets)

    if preservation_dois:
        manifest["preservation_dois"] = list(preservation_dois)
    else:
        manifest.pop("preservation_dois", None)

    return manifest


def serialize_release_manifest(manifest: Mapping) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
