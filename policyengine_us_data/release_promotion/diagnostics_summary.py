"""Compact release diagnostics summary for Stage 5 promotion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import ArtifactRef
from policyengine_us_data.stage_contracts._coercion import (
    freeze_mapping,
    jsonable_value,
    mapping_value,
    optional_string,
    optional_string_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_optional_float,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION
from policyengine_us_data.stage_contracts.stages import (
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
)
from policyengine_us_data.utils.canonical_json import (
    canonical_json_dumps,
    canonical_json_loads,
)

from .context import ReleasePromotionContext
from .results import FullPromotionResult

RELEASE_DIAGNOSTICS_SUMMARY_FILENAME = "release_diagnostics_summary.json"
RELEASE_DIAGNOSTICS_SUMMARY_MEDIA_TYPE = "application/json"

DEFAULT_RELEASE_DIAGNOSTICS_SOURCES = (
    ("run_manifest", "run_manifest", "run_manifest.json"),
    (
        "stage_2_calibration_package",
        "step_manifest",
        "steps/2_build_calibration_package.json",
    ),
    (
        "stage_3_weight_fitting_regional",
        "step_manifest",
        "steps/3a_weight_fitting_regional.json",
    ),
    (
        "stage_3_weight_fitting_national",
        "step_manifest",
        "steps/3b_weight_fitting_national.json",
    ),
    (
        "stage_4_local_area_h5_regional",
        "step_manifest",
        "steps/4a_local_area_h5_regional.json",
    ),
    (
        "stage_4_local_area_h5_national",
        "step_manifest",
        "steps/4b_local_area_h5_national.json",
    ),
    (
        "stage_4_upload_diagnostics",
        "step_manifest",
        "steps/4d_upload_diagnostics.json",
    ),
    (
        "stage_4_output_contract",
        "stage_contract",
        "diagnostics/contracts/output_build_contract.json",
    ),
)


def release_diagnostics_summary_repo_path(run_id: str) -> str:
    """Return the run-scoped repository path for the release diagnostics summary."""

    return (
        f"calibration/runs/{run_id}/diagnostics/{RELEASE_DIAGNOSTICS_SUMMARY_FILENAME}"
    )


def release_diagnostics_summary_path(run_dir: str | Path) -> Path:
    """Return the run-local diagnostics path for the release diagnostics summary."""

    return Path(run_dir) / "diagnostics" / RELEASE_DIAGNOSTICS_SUMMARY_FILENAME


def release_diagnostics_summary_artifact_ref(
    context: ReleasePromotionContext,
    summary: "ReleaseDiagnosticsSummary",
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> ArtifactRef:
    """Return a stage-contract reference to the release diagnostics summary."""

    return ArtifactRef(
        logical_name="release_diagnostics_summary",
        uri=(
            f"hf://{context.hf_repo_name}/"
            f"{release_diagnostics_summary_repo_path(context.run_id)}"
        ),
        sha256=sha256,
        size_bytes=size_bytes,
        media_type=RELEASE_DIAGNOSTICS_SUMMARY_MEDIA_TYPE,
        metadata={
            "artifact_family": "release_diagnostics_summary",
            "source_stage_id": STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
            "relative_path": release_diagnostics_summary_repo_path(context.run_id),
            "summary_status": summary.status,
            "source_count": len(summary.sources),
            "missing_source_count": len(summary.missing_sources),
        },
    )


@dataclass(frozen=True, kw_only=True)
class ReleaseDiagnosticsSource:
    """Availability and compact facts for one summary source artifact."""

    name: str
    source_kind: str
    status: str
    path: str
    stage_id: str | None = None
    facts: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        for field_name in ("name", "source_kind", "status", "path"):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        if self.status not in {"available", "missing", "invalid"}:
            raise ValueError("status must be available, missing, or invalid")
        object.__setattr__(
            self,
            "stage_id",
            optional_string_value(self.stage_id, "stage_id"),
        )
        object.__setattr__(
            self,
            "facts",
            freeze_mapping(self.facts, "facts"),
        )
        object.__setattr__(
            self,
            "message",
            optional_string_value(self.message, "message"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this source summary to JSON-compatible primitives."""

        return {
            "name": self.name,
            "source_kind": self.source_kind,
            "status": self.status,
            "path": self.path,
            "stage_id": self.stage_id,
            "facts": jsonable_value(self.facts),
            "message": self.message,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseDiagnosticsSource":
        """Restore a source summary from serialized data."""

        return cls(
            name=required_string(data, "name"),
            source_kind=required_string(data, "source_kind"),
            status=required_string(data, "status"),
            path=required_string(data, "path"),
            stage_id=optional_string(data, "stage_id"),
            facts=mapping_value(data, "facts"),
            message=optional_string(data, "message"),
            schema_version=schema_version(data),
        )


@pipeline_node(
    id="release_diagnostics_summary",
    label="ReleaseDiagnosticsSummary",
    node_type="library",
    description="Compact final Stage 5 summary over existing run diagnostics.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    artifacts_in=["step manifests", "stage contracts", "typed promotion result"],
    artifacts_out=["release_diagnostics_summary.json"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_diagnostics_summary.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class ReleaseDiagnosticsSummary:
    """Compact dashboard/API summary for one promoted release run."""

    run_id: str
    candidate_version: str
    release_version: str
    generated_at: str
    status: str
    sources: Mapping[str, ReleaseDiagnosticsSource]
    missing_sources: tuple[str, ...] = ()
    invalid_sources: tuple[str, ...] = ()
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    release_promotion: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        for field_name in (
            "run_id",
            "candidate_version",
            "release_version",
            "generated_at",
            "status",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        if self.status not in {"complete", "partial", "invalid"}:
            raise ValueError("status must be complete, partial, or invalid")
        object.__setattr__(self, "sources", _coerce_sources(self.sources))
        object.__setattr__(
            self,
            "missing_sources",
            _string_tuple(self.missing_sources, "missing_sources"),
        )
        object.__setattr__(
            self,
            "invalid_sources",
            _string_tuple(self.invalid_sources, "invalid_sources"),
        )
        object.__setattr__(
            self, "artifacts", freeze_mapping(self.artifacts, "artifacts")
        )
        object.__setattr__(
            self,
            "release_promotion",
            freeze_mapping(self.release_promotion, "release_promotion"),
        )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this release diagnostics summary."""

        return {
            "run_id": self.run_id,
            "candidate_version": self.candidate_version,
            "release_version": self.release_version,
            "generated_at": self.generated_at,
            "status": self.status,
            "sources": {
                name: self.sources[name].to_dict() for name in sorted(self.sources)
            },
            "missing_sources": list(self.missing_sources),
            "invalid_sources": list(self.invalid_sources),
            "artifacts": jsonable_value(self.artifacts),
            "release_promotion": jsonable_value(self.release_promotion),
            "metadata": jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseDiagnosticsSummary":
        """Restore a release diagnostics summary from serialized data."""

        return cls(
            run_id=required_string(data, "run_id"),
            candidate_version=required_string(data, "candidate_version"),
            release_version=required_string(data, "release_version"),
            generated_at=required_string(data, "generated_at"),
            status=required_string(data, "status"),
            sources=mapping_value(data, "sources"),
            missing_sources=_string_tuple(
                data.get("missing_sources", ()),
                "missing_sources",
            ),
            invalid_sources=_string_tuple(
                data.get("invalid_sources", ()),
                "invalid_sources",
            ),
            artifacts=mapping_value(data, "artifacts"),
            release_promotion=mapping_value(data, "release_promotion"),
            metadata=mapping_value(data, "metadata"),
            schema_version=schema_version(data),
        )


def build_release_diagnostics_summary(
    *,
    context: ReleasePromotionContext,
    promotion_result: FullPromotionResult,
    generated_at: str,
    source_payloads: Mapping[str, Mapping[str, Any] | None] | None = None,
    source_errors: Mapping[str, str] | None = None,
    artifact_refs: Sequence[ArtifactRef] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ReleaseDiagnosticsSummary:
    """Build a compact summary from existing structured run artifacts."""

    _validate_result_matches_context(promotion_result, context)
    source_payloads = source_payloads or {}
    source_errors = source_errors or {}
    sources = {
        name: _source_from_payload(
            name=name,
            source_kind=source_kind,
            path=path,
            payload=source_payloads.get(name),
            error=source_errors.get(name),
        )
        for name, source_kind, path in DEFAULT_RELEASE_DIAGNOSTICS_SOURCES
    }
    sources["stage_5_promotion_result"] = ReleaseDiagnosticsSource(
        name="stage_5_promotion_result",
        source_kind="typed_result",
        status="available",
        path="in_memory:FullPromotionResult",
        stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
        facts=_promotion_facts(promotion_result),
    )
    missing_sources = tuple(
        name for name, source in sources.items() if source.status == "missing"
    )
    invalid_sources = tuple(
        name for name, source in sources.items() if source.status == "invalid"
    )
    status = (
        "invalid" if invalid_sources else "partial" if missing_sources else "complete"
    )
    return ReleaseDiagnosticsSummary(
        run_id=context.run_id,
        candidate_version=context.candidate_version,
        release_version=context.release_version,
        generated_at=generated_at,
        status=status,
        sources=sources,
        missing_sources=missing_sources,
        invalid_sources=invalid_sources,
        artifacts=_artifact_refs_by_name(artifact_refs),
        release_promotion=promotion_result.to_dict(),
        metadata=metadata or {},
    )


def build_release_diagnostics_summary_from_run_dir(
    *,
    run_dir: str | Path,
    context: ReleasePromotionContext,
    promotion_result: FullPromotionResult,
    generated_at: str,
    artifact_refs: Sequence[ArtifactRef] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ReleaseDiagnosticsSummary:
    """Read structured run artifacts and build the release diagnostics summary."""

    source_payloads, source_errors = _load_sources_from_run_dir(Path(run_dir))
    return build_release_diagnostics_summary(
        context=context,
        promotion_result=promotion_result,
        generated_at=generated_at,
        source_payloads=source_payloads,
        source_errors=source_errors,
        artifact_refs=artifact_refs,
        metadata=metadata,
    )


def release_diagnostics_summary_to_json(
    summary: ReleaseDiagnosticsSummary,
) -> str:
    """Serialize the release diagnostics summary deterministically."""

    return canonical_json_dumps(summary.to_dict())


def release_diagnostics_summary_from_json(
    payload: str,
) -> ReleaseDiagnosticsSummary:
    """Restore the release diagnostics summary from JSON text."""

    return ReleaseDiagnosticsSummary.from_dict(canonical_json_loads(payload))


def write_release_diagnostics_summary(
    summary: ReleaseDiagnosticsSummary,
    path: str | Path,
) -> ReleaseDiagnosticsSummary:
    """Write the release diagnostics summary to disk."""

    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        release_diagnostics_summary_to_json(summary),
        encoding="utf-8",
    )
    return summary


def read_release_diagnostics_summary(
    path: str | Path,
) -> ReleaseDiagnosticsSummary:
    """Read a release diagnostics summary from disk."""

    return release_diagnostics_summary_from_json(Path(path).read_text(encoding="utf-8"))


def _load_sources_from_run_dir(
    run_dir: Path,
) -> tuple[dict[str, Mapping[str, Any] | None], dict[str, str]]:
    source_payloads: dict[str, Mapping[str, Any] | None] = {}
    source_errors: dict[str, str] = {}
    for name, _, relative_path in DEFAULT_RELEASE_DIAGNOSTICS_SOURCES:
        path = run_dir / relative_path
        if not path.exists():
            source_payloads[name] = None
            continue
        try:
            payload = canonical_json_loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            source_payloads[name] = None
            source_errors[name] = type(exc).__name__
            continue
        if not isinstance(payload, Mapping):
            source_payloads[name] = None
            source_errors[name] = "not_a_json_object"
            continue
        source_payloads[name] = payload
    return source_payloads, source_errors


def _source_from_payload(
    *,
    name: str,
    source_kind: str,
    path: str,
    payload: Mapping[str, Any] | None,
    error: str | None,
) -> ReleaseDiagnosticsSource:
    if error is not None:
        return ReleaseDiagnosticsSource(
            name=name,
            source_kind=source_kind,
            status="invalid",
            path=path,
            message=error,
        )
    if payload is None:
        return ReleaseDiagnosticsSource(
            name=name,
            source_kind=source_kind,
            status="missing",
            path=path,
            message="source artifact not present",
        )
    return ReleaseDiagnosticsSource(
        name=name,
        source_kind=source_kind,
        status="available",
        path=path,
        stage_id=_stage_id_from_payload(source_kind, payload),
        facts=_facts_from_payload(source_kind, payload),
    )


def _facts_from_payload(source_kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if source_kind == "run_manifest":
        return {
            "status": payload.get("status"),
            "known_step_count": len(payload.get("known_step_ids", [])),
            "completed_at": payload.get("completed_at"),
            "branch": payload.get("branch"),
            "sha": payload.get("sha"),
        }
    if source_kind == "step_manifest":
        reuse_measurement = payload.get("reuse_measurement", {})
        if not isinstance(reuse_measurement, Mapping):
            reuse_measurement = {}
        facts = {
            "step_id": payload.get("step_id"),
            "parent_step_id": payload.get("parent_step_id"),
            "status": payload.get("status"),
            "attempt": payload.get("attempt"),
            "completed_at": payload.get("completed_at"),
            "duration_s": _optional_float(payload.get("duration_s")),
            "output_count": len(payload.get("outputs", [])),
            "diagnostic_count": len(payload.get("diagnostics", [])),
            "reuse_decision": payload.get("reuse_decision"),
            "reuse_reason": payload.get("reuse_reason"),
            "expected_outputs": reuse_measurement.get("expected_outputs"),
            "recomputed_outputs": reuse_measurement.get("recomputed_outputs"),
            "valid_reused_outputs": reuse_measurement.get("valid_reused_outputs"),
        }
        return {key: value for key, value in facts.items() if value is not None}
    if source_kind == "stage_contract":
        execution = payload.get("execution", {})
        validation = payload.get("validation")
        if not isinstance(execution, Mapping):
            execution = {}
        if not isinstance(validation, Mapping):
            validation = {}
        facts = {
            "contract_type": payload.get("contract_type"),
            "stage_id": payload.get("stage_id"),
            "fingerprint": payload.get("fingerprint"),
            "execution_status": execution.get("status"),
            "reuse_decision": execution.get("reuse_decision"),
            "validation_status": validation.get("status"),
            "input_count": len(payload.get("inputs", [])),
            "output_count": len(payload.get("outputs", [])),
            "diagnostic_count": len(payload.get("diagnostics", [])),
        }
        return {key: value for key, value in facts.items() if value is not None}
    return {"keys": sorted(str(key) for key in payload)}


def _stage_id_from_payload(
    source_kind: str,
    payload: Mapping[str, Any],
) -> str | None:
    if source_kind == "step_manifest":
        value = payload.get("parent_step_id") or payload.get("step_id")
    elif source_kind == "stage_contract":
        value = payload.get("stage_id")
    else:
        value = None
    return value if isinstance(value, str) and value else None


def _promotion_facts(result: FullPromotionResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "candidate_version": result.candidate_version,
        "release_version": result.release_version,
        "artifact_count": result.artifact_count,
        "hf_promoted_count": result.hf.promoted_count,
        "gcs_uploaded_count": result.gcs.uploaded_count,
        "release_manifest_artifacts": result.release_manifest.artifact_count,
        "version_manifest_updated": result.version_manifest.updated,
        "completion_marker_path": result.completion_marker.marker_path,
        "cleanup": result.cleanup.to_dict(),
        "already_finalized": result.already_finalized,
    }


def _artifact_refs_by_name(
    artifact_refs: Sequence[ArtifactRef],
) -> Mapping[str, Any]:
    return {
        artifact.logical_name: artifact.to_dict()
        for artifact in sorted(artifact_refs, key=lambda item: item.logical_name)
    }


def _validate_result_matches_context(
    result: FullPromotionResult,
    context: ReleasePromotionContext,
) -> None:
    if result.run_id != context.run_id:
        raise ValueError("promotion_result.run_id must match context.run_id")
    if result.candidate_version != context.candidate_version:
        raise ValueError(
            "promotion_result.candidate_version must match context.candidate_version"
        )
    if result.release_version != context.release_version:
        raise ValueError(
            "promotion_result.release_version must match context.release_version"
        )


def _coerce_sources(
    value: Mapping[str, ReleaseDiagnosticsSource | Mapping[str, Any]],
) -> Mapping[str, ReleaseDiagnosticsSource]:
    if not isinstance(value, Mapping):
        raise ValueError("sources must be a mapping")
    sources: dict[str, ReleaseDiagnosticsSource] = {}
    for name, source in value.items():
        if isinstance(source, ReleaseDiagnosticsSource):
            sources[str(name)] = source
        elif isinstance(source, Mapping):
            sources[str(name)] = ReleaseDiagnosticsSource.from_dict(source)
        else:
            raise ValueError(
                "sources entries must be ReleaseDiagnosticsSource mappings"
            )
    for name, source in sources.items():
        if name != source.name:
            raise ValueError("sources keys must match source.name")
    return freeze_mapping(sources, "sources")


def _optional_float(value: Any) -> float | None:
    validate_optional_float(value, "duration_s")
    return float(value) if value is not None else None


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise ValueError(f"{field_name} must be a tuple or list")
    return tuple(require_non_empty(item, field_name) for item in value)
