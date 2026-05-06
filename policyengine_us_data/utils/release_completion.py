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


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Release completion marker {field} must be an object.")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Release completion marker {field} must be a string.")
    return value


def _require_string_sequence(value: Any, field: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Release completion marker {field} must be a list.")
    strings = list(value)
    if not all(isinstance(item, str) for item in strings):
        raise ValueError(
            f"Release completion marker {field} must contain only strings."
        )
    return strings


def _require_nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"Release completion marker {field} must be a non-negative integer."
        )
    return value


def _require_equal(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise ValueError(
            f"Release completion marker {field} must be {expected!r}; got {value!r}."
        )


def _require_paths(
    *,
    actual_paths: Sequence[str],
    expected_paths: Sequence[str],
    field: str,
) -> None:
    missing_paths = sorted(set(expected_paths) - set(actual_paths))
    if missing_paths:
        raise ValueError(
            f"Release completion marker {field} is missing paths: "
            + ", ".join(missing_paths)
        )


def validate_release_completion_marker(
    marker: Mapping[str, Any],
    *,
    version: str,
    hf_repo_name: str | None = None,
    hf_repo_type: str | None = None,
) -> Mapping[str, Any]:
    """Validate that a marker certifies a complete release at a version tag."""
    marker = _require_mapping(marker, "root")
    _require_equal(
        marker.get("schema_version"),
        RELEASE_COMPLETION_SCHEMA_VERSION,
        "schema_version",
    )
    _require_equal(marker.get("status"), "complete", "status")
    _require_equal(marker.get("version"), version, "version")
    _require_equal(
        marker.get("marker_path"),
        release_completion_marker_path(version),
        "marker_path",
    )
    _require_string(marker.get("run_id"), "run_id")
    _require_string(marker.get("completed_at"), "completed_at")

    hf = _require_mapping(marker.get("hf"), "hf")
    if hf_repo_name is not None:
        _require_equal(hf.get("repo_id"), hf_repo_name, "hf.repo_id")
    else:
        _require_string(hf.get("repo_id"), "hf.repo_id")
    if hf_repo_type is not None:
        _require_equal(hf.get("repo_type"), hf_repo_type, "hf.repo_type")
    else:
        _require_string(hf.get("repo_type"), "hf.repo_type")
    _require_equal(hf.get("revision"), version, "hf.revision")

    required_paths = _require_mapping(
        marker.get("required_paths"),
        "required_paths",
    )
    release_manifest_paths = _require_string_sequence(
        required_paths.get("release_manifest"),
        "required_paths.release_manifest",
    )
    _require_paths(
        actual_paths=release_manifest_paths,
        expected_paths=[
            "release_manifest.json",
            f"releases/{version}/release_manifest.json",
        ],
        field="required_paths.release_manifest",
    )
    trace_tro_paths = _require_string_sequence(
        required_paths.get("trace_tro"),
        "required_paths.trace_tro",
    )
    _require_paths(
        actual_paths=trace_tro_paths,
        expected_paths=[
            "trace.tro.jsonld",
            f"releases/{version}/trace.tro.jsonld",
        ],
        field="required_paths.trace_tro",
    )
    _require_equal(
        required_paths.get("version_manifest"),
        VERSION_MANIFEST_PATH,
        "required_paths.version_manifest",
    )
    validation_report_paths = _require_string_sequence(
        required_paths.get("validation_reports"),
        "required_paths.validation_reports",
    )
    if not validation_report_paths:
        raise ValueError(
            "Release completion marker required_paths.validation_reports "
            "must not be empty."
        )
    artifact_paths = _require_string_sequence(
        required_paths.get("artifacts"),
        "required_paths.artifacts",
    )
    if not artifact_paths:
        raise ValueError(
            "Release completion marker required_paths.artifacts must not be empty."
        )

    checks = _require_mapping(marker.get("checks"), "checks")
    _require_nonnegative_int(
        checks.get("hf_artifacts_promoted"),
        "checks.hf_artifacts_promoted",
    )
    _require_nonnegative_int(
        checks.get("gcs_artifacts_uploaded"),
        "checks.gcs_artifacts_uploaded",
    )
    for field in (
        "release_manifest_written",
        "trace_tro_written",
        "version_manifest_written",
        "validation_reports_present",
        "release_manifest_checksums_present",
    ):
        _require_equal(checks.get(field), True, f"checks.{field}")

    return marker


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
    marker = {
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
    validate_release_completion_marker(
        marker,
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    return marker


def serialize_release_completion_marker(marker: Mapping[str, Any]) -> bytes:
    return (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode("utf-8")
