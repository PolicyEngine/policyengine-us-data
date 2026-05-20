"""Run-scoped diagnostics helpers for Stage 5 release candidates."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    DiagnosticRef,
    StageContract,
)

from .artifact_uris import validate_uri_repo
from .artifacts import normalize_release_path, strip_staging_prefix
from .context import ReleasePromotionContext


def diagnostics_manifest_path(
    output_contract: StageContract,
    *,
    context: ReleasePromotionContext,
) -> str | None:
    """Return the diagnostics manifest path from a Stage 4 contract."""

    for diagnostic in diagnostic_refs(output_contract):
        artifact = diagnostic.artifact
        if artifact is None:
            continue
        if not is_diagnostics_manifest_ref(diagnostic.name, diagnostic.kind, artifact):
            continue
        path = diagnostic_artifact_path(artifact, context)
        if path is not None:
            return path
    for artifact in output_contract.outputs:
        if not is_diagnostics_artifact(artifact):
            continue
        path = diagnostic_artifact_path(artifact, context)
        if path is not None:
            return path
    return None


def diagnostics_manifest_identity(
    output_contract: StageContract,
    *,
    context: ReleasePromotionContext,
) -> dict[str, object] | None:
    """Return checksum identity for the diagnostics manifest when available."""

    for diagnostic in diagnostic_refs(output_contract):
        artifact = diagnostic.artifact
        if artifact is None:
            continue
        if not is_diagnostics_manifest_ref(diagnostic.name, diagnostic.kind, artifact):
            continue
        path = diagnostic_artifact_path(artifact, context)
        if path is not None:
            return diagnostic_artifact_identity(path, artifact)
    for artifact in output_contract.outputs:
        if not is_diagnostics_artifact(artifact):
            continue
        path = diagnostic_artifact_path(artifact, context)
        if path is not None:
            return diagnostic_artifact_identity(path, artifact)
    return None


def diagnostic_artifact_identity(
    path: str,
    artifact: ArtifactRef,
) -> dict[str, object]:
    """Return stable checksum identity for one diagnostic artifact."""

    return {
        "path": path,
        "logical_name": artifact.logical_name,
        "uri": artifact.uri,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def validation_report_ref_path(
    ref: DiagnosticRef,
    context: ReleasePromotionContext,
) -> str:
    """Return the normalized run diagnostics path for a validation report ref."""

    artifact = ref.artifact
    if artifact is None:
        raise ValueError("validation_report_refs entries must include artifacts")
    path = diagnostic_artifact_path(artifact, context)
    if path is None:
        raise ValueError(
            "validation_report_refs artifacts must live under run diagnostics"
        )
    return path


def diagnostic_refs(output_contract: StageContract) -> Iterable[DiagnosticRef]:
    """Yield unique diagnostics from contract-level and validation report refs."""

    seen: set[tuple[str, str, str | None]] = set()
    refs = list(output_contract.diagnostics)
    if output_contract.validation is not None:
        refs.extend(output_contract.validation.diagnostics)
    for diagnostic in refs:
        artifact_uri = diagnostic.artifact.uri if diagnostic.artifact else None
        key = (diagnostic.name, diagnostic.kind, artifact_uri)
        if key in seen:
            continue
        seen.add(key)
        yield diagnostic


def diagnostic_artifact_path(
    artifact: ArtifactRef,
    context: ReleasePromotionContext,
) -> str | None:
    """Return a normalized run diagnostics path for a diagnostic artifact."""

    path = artifact.metadata.get("relative_path") or artifact.metadata.get(
        "output_relative_path"
    )
    metadata_path = (
        normalize_run_diagnostic_path(path, context)
        if isinstance(path, str) and path
        else None
    )
    uri_path = diagnostic_path_from_uri(artifact.uri, context)
    if metadata_path is not None and uri_path is not None and metadata_path != uri_path:
        raise ValueError("Diagnostic artifact metadata path must match artifact.uri")
    return metadata_path or uri_path


def diagnostic_path_from_uri(
    uri: str,
    context: ReleasePromotionContext,
) -> str | None:
    """Return a normalized run diagnostics path from a diagnostic URI."""

    parsed = urlparse(uri)
    if parsed.scheme or parsed.netloc:
        validate_uri_repo(parsed, context)
    raw_path = parsed.path.lstrip("/") if parsed.scheme else uri
    candidate_paths = [raw_path]
    if parsed.netloc:
        candidate_paths.append(f"{parsed.netloc}/{raw_path}")
    marker = f"calibration/runs/{context.run_id}/diagnostics/"
    for candidate in candidate_paths:
        if marker in candidate:
            return normalize_run_diagnostic_path(
                candidate[candidate.index(marker) :],
                context,
            )
        if "calibration/runs/" in candidate:
            raise ValueError("diagnostic artifact URI must match context.run_id")
    return None


def normalize_run_contract_path(
    path: str,
    context: ReleasePromotionContext,
) -> str:
    """Normalize and run-scope a Stage 4 source contract path."""

    normalized = strip_staging_prefix(path, context.hf_staging_prefix)
    required_prefix = f"calibration/runs/{context.run_id}/"
    if not normalized.startswith(required_prefix):
        raise ValueError(
            "source_output_contract_path must live under "
            f"{required_prefix} for context.run_id"
        )
    return normalized


def normalize_run_diagnostic_path(
    path: str,
    context: ReleasePromotionContext,
) -> str:
    """Normalize and run-scope a diagnostics or validation report path."""

    normalized = strip_staging_prefix(path, context.hf_staging_prefix)
    required_prefix = f"calibration/runs/{context.run_id}/diagnostics/"
    if not normalized.startswith(required_prefix):
        raise ValueError(
            "diagnostic and validation report paths must live under "
            f"{required_prefix} for context.run_id"
        )
    return normalized


def is_diagnostics_artifact(artifact: ArtifactRef) -> bool:
    """Return whether an artifact ref points at a diagnostics artifact."""

    path = artifact.metadata.get("relative_path") or artifact.metadata.get(
        "output_relative_path"
    )
    return (
        artifact.logical_name == "diagnostics_manifest"
        or artifact.metadata.get("artifact_family") == "diagnostics"
        or (isinstance(path, str) and is_diagnostics_path(path))
    )


def is_diagnostics_path(path: str) -> bool:
    """Return whether a repo-relative path lives under run diagnostics."""

    normalized = normalize_release_path(path)
    parts = normalized.split("/")
    return (
        len(parts) >= 5
        and parts[:2] == ["calibration", "runs"]
        and parts[3] == "diagnostics"
    )


def is_diagnostics_manifest_ref(
    name: str,
    kind: str,
    artifact: ArtifactRef,
) -> bool:
    """Return whether a diagnostic ref is the diagnostics manifest."""

    return (
        name == "diagnostics_manifest"
        or kind == "diagnostics_manifest"
        or artifact.logical_name == "diagnostics_manifest"
    )
