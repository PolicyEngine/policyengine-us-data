"""Completion marker for certified US data releases."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Mapping, Sequence

RELEASE_COMPLETION_SCHEMA_VERSION = 1
RELEASE_COMPLETE_FILENAME = "release-complete.json"
VERSION_MANIFEST_PATH = "version_manifest.json"


def release_completion_marker_path(version: str) -> str:
    return f"releases/{version}/{RELEASE_COMPLETE_FILENAME}"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _release_artifact_paths(release_manifest: Mapping[str, Any]) -> list[str]:
    artifacts = release_manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return []
    return sorted(
        artifact["path"]
        for artifact in artifacts.values()
        if isinstance(artifact, Mapping) and isinstance(artifact.get("path"), str)
    )


def _artifacts_have_checksums(release_manifest: Mapping[str, Any]) -> bool:
    artifacts = release_manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return False
    return all(
        isinstance(artifact, Mapping) and isinstance(artifact.get("sha256"), str)
        for artifact in artifacts.values()
    )


def build_release_completion_marker(
    *,
    version: str,
    run_id: str,
    hf_repo_name: str,
    hf_repo_type: str,
    release_manifest: Mapping[str, Any],
    released_paths: Sequence[str],
    validation_report_paths: Sequence[str],
    promoted_hf: int,
    uploaded_gcs: int,
    created_at: str | None = None,
) -> dict[str, Any]:
    release_manifest_paths = _release_artifact_paths(release_manifest)
    missing_manifest_paths = sorted(set(released_paths) - set(release_manifest_paths))
    if missing_manifest_paths:
        raise ValueError(
            "Release manifest is missing released artifacts: "
            + ", ".join(missing_manifest_paths)
        )
    if not _artifacts_have_checksums(release_manifest):
        raise ValueError(
            "Release manifest artifacts must all include sha256 checksums."
        )
    if not validation_report_paths:
        raise ValueError("A release completion marker requires validation reports.")

    marker_path = release_completion_marker_path(version)
    return {
        "schema_version": RELEASE_COMPLETION_SCHEMA_VERSION,
        "status": "complete",
        "version": version,
        "run_id": run_id,
        "completed_at": created_at or _utc_timestamp(),
        "marker_path": marker_path,
        "hf": {
            "repo_id": hf_repo_name,
            "repo_type": hf_repo_type,
            "revision": version,
        },
        "required_paths": {
            "release_manifest": [
                "release_manifest.json",
                f"releases/{version}/release_manifest.json",
            ],
            "trace_tro": [
                "trace.tro.jsonld",
                f"releases/{version}/trace.tro.jsonld",
            ],
            "version_manifest": VERSION_MANIFEST_PATH,
            "validation_reports": sorted(validation_report_paths),
            "artifacts": sorted(released_paths),
        },
        "checks": {
            "hf_artifacts_promoted": promoted_hf,
            "gcs_artifacts_uploaded": uploaded_gcs,
            "release_manifest_written": True,
            "trace_tro_written": True,
            "version_manifest_written": True,
            "validation_reports_present": True,
            "release_manifest_checksums_present": True,
        },
    }


def serialize_release_completion_marker(marker: Mapping[str, Any]) -> bytes:
    return (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode("utf-8")
