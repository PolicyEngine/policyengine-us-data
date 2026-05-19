"""Stage 5 release candidate bundle schemas and readers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    StageContract,
    read_contract,
)
from policyengine_us_data.stage_contracts._coercion import (
    freeze_mapping,
    freeze_sequence,
    jsonable_value,
    optional_string,
    optional_string_value,
    required_string,
    schema_version,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION
from policyengine_us_data.stage_contracts.fingerprints import fingerprint_material
from policyengine_us_data.stage_contracts.stages import (
    STAGE_4_BUILD_OUTPUTS,
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
)

from .artifacts import (
    BASE_RELEASE_ARTIFACT_PATHS,
    ReleaseArtifactSpec,
    dedupe_normalized_release_paths,
    infer_artifact_identity,
    infer_release_artifact_spec,
    logical_name_for_release_path,
    normalize_release_path,
    strip_staging_prefix,
)
from .context import ReleasePromotionContext

_INVENTORY_PATH_KEYS = (
    "expected_release_path",
    "relative_path",
    "output_relative_path",
    "repo_path",
    "path",
    "destination_path",
    "staging_path",
)
RELEASE_CANDIDATE_BUNDLE_TYPE = "release_candidate_input_bundle"
RELEASE_SAFE_STAGE4_EXECUTION_STATUSES = frozenset(
    {"completed", "reused", "partially_reused"}
)


@pipeline_node(
    id="release_candidate_input_bundle",
    label="ReleaseCandidateInputBundle",
    node_type="library",
    description="Typed Stage 5 input bundle describing artifacts eligible for release promotion.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class ReleaseCandidateInputBundle:
    """Typed Stage 5 input bundle describing a candidate ready for promotion."""

    context: ReleasePromotionContext
    artifacts: tuple[ReleaseArtifactSpec, ...]
    source_output_contract_path: str | None = None
    release_candidate_fingerprint: str | None = None
    validation_report_paths: tuple[str, ...] = ()
    diagnostics_manifest_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    bundle_type: str = RELEASE_CANDIDATE_BUNDLE_TYPE
    stage_id: str = STAGE_5_VALIDATE_AND_PROMOTE_RELEASE
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        if self.bundle_type != RELEASE_CANDIDATE_BUNDLE_TYPE:
            raise ValueError(f"bundle_type must be {RELEASE_CANDIDATE_BUNDLE_TYPE!r}")
        if self.stage_id != STAGE_5_VALIDATE_AND_PROMOTE_RELEASE:
            raise ValueError(
                f"stage_id must be {STAGE_5_VALIDATE_AND_PROMOTE_RELEASE!r}"
            )
        if not isinstance(self.context, ReleasePromotionContext):
            raise ValueError("context must be ReleasePromotionContext")
        object.__setattr__(
            self,
            "artifacts",
            freeze_sequence(self.artifacts, "artifacts", ReleaseArtifactSpec),
        )
        if not self.artifacts:
            raise ValueError("artifacts must include at least one release artifact")
        object.__setattr__(
            self,
            "source_output_contract_path",
            (
                _normalize_run_contract_path(
                    self.source_output_contract_path,
                    self.context,
                )
                if self.source_output_contract_path is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "release_candidate_fingerprint",
            optional_string_value(
                self.release_candidate_fingerprint,
                "release_candidate_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "validation_report_paths",
            tuple(
                _normalize_run_diagnostic_path(path, self.context)
                for path in self.validation_report_paths
            ),
        )
        object.__setattr__(
            self,
            "diagnostics_manifest_path",
            (
                _normalize_run_diagnostic_path(
                    self.diagnostics_manifest_path,
                    self.context,
                )
                if self.diagnostics_manifest_path is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate bundle to JSON-compatible primitives."""

        return {
            "bundle_type": self.bundle_type,
            "stage_id": self.stage_id,
            "schema_version": self.schema_version,
            "context": self.context.to_dict(),
            "source_output_contract_path": self.source_output_contract_path,
            "release_candidate_fingerprint": self.release_candidate_fingerprint,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "validation_report_paths": list(self.validation_report_paths),
            "diagnostics_manifest_path": self.diagnostics_manifest_path,
            "metadata": jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseCandidateInputBundle":
        """Restore a release candidate bundle from serialized data."""

        return cls(
            context=ReleasePromotionContext.from_dict(data["context"]),
            source_output_contract_path=optional_string(
                data,
                "source_output_contract_path",
            ),
            release_candidate_fingerprint=optional_string(
                data,
                "release_candidate_fingerprint",
            ),
            artifacts=tuple(
                ReleaseArtifactSpec.from_dict(item)
                for item in data.get("artifacts", ())
            ),
            validation_report_paths=tuple(
                required_string({"path": item}, "path")
                for item in data.get("validation_report_paths", ())
            ),
            diagnostics_manifest_path=optional_string(
                data,
                "diagnostics_manifest_path",
            ),
            metadata=data.get("metadata", {}),
            bundle_type=data.get("bundle_type", RELEASE_CANDIDATE_BUNDLE_TYPE),
            stage_id=data.get("stage_id", STAGE_5_VALIDATE_AND_PROMOTE_RELEASE),
            schema_version=schema_version(data),
        )


@pipeline_node(
    id="legacy_release_candidate_bundle_builder",
    label="Legacy Release Candidate Bundle Builder",
    node_type="library",
    description="Compatibility builder for Stage 5 candidates from legacy staged path sets.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
def build_legacy_release_candidate_bundle(
    *,
    context: ReleasePromotionContext,
    rel_paths: Sequence[str],
    artifact_metadata_by_path: Mapping[str, Mapping[str, Any]] | None = None,
    validation_report_paths: Sequence[str] = (),
    source_output_contract_path: str | None = None,
    diagnostics_manifest_path: str | None = None,
) -> ReleaseCandidateInputBundle:
    """Build a candidate bundle from the current legacy staged relative paths."""

    artifact_metadata_by_path = _normalize_artifact_metadata_by_path(
        artifact_metadata_by_path or {},
        staging_prefix=context.hf_staging_prefix,
    )
    artifacts = tuple(
        _legacy_artifact_spec(
            path,
            artifact_metadata_by_path=artifact_metadata_by_path,
        )
        for path in dedupe_normalized_release_paths(
            rel_paths,
            staging_prefix=context.hf_staging_prefix,
        )
    )
    return _candidate_bundle_with_fingerprint(
        context=context,
        artifacts=artifacts,
        source_output_contract_path=source_output_contract_path,
        validation_report_paths=validation_report_paths,
        diagnostics_manifest_path=diagnostics_manifest_path,
        reader="legacy_staged_paths",
    )


@pipeline_node(
    id="stage4_release_candidate_bundle_builder",
    label="Stage 4 Release Candidate Bundle Builder",
    node_type="library",
    description="Build a Stage 5 candidate bundle from Stage 4 contract and inventory records.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
def build_release_candidate_bundle_from_stage4_contract(
    *,
    context: ReleasePromotionContext,
    output_contract: StageContract,
    inventory_records: Iterable[Mapping[str, Any]] = (),
    source_output_contract_path: str | None = None,
    validation_report_paths: Sequence[str] = (),
    diagnostics_manifest_path: str | None = None,
) -> ReleaseCandidateInputBundle:
    """Build a candidate bundle from a Stage 4 output contract shape."""

    _validate_stage4_contract_context(output_contract, context)

    inventory_specs = tuple(
        _artifact_spec_from_inventory_record(
            record,
            context=context,
        )
        for record in inventory_records
    )
    contract_specs = tuple(
        spec
        for artifact in output_contract.outputs
        if (
            spec := _artifact_spec_from_contract_artifact(
                artifact,
                context=context,
            )
        )
        is not None
    )
    artifacts = _merge_artifact_specs(contract_specs, inventory_specs)
    if not artifacts:
        raise ValueError(
            "Stage 4 candidate reader needs inventory records or output artifacts "
            "with release-relative paths"
        )

    derived_diagnostics_manifest_path = (
        diagnostics_manifest_path
        or _diagnostics_manifest_path(output_contract, context=context)
    )
    extra_fingerprint_material: dict[str, Any] = {
        "source_output_contract_fingerprint": output_contract.fingerprint.value,
        "source_output_contract_stage_id": output_contract.stage_id,
    }
    diagnostics_manifest_identity = _diagnostics_manifest_identity(
        output_contract,
        context=context,
    )
    if diagnostics_manifest_identity is not None:
        extra_fingerprint_material["diagnostics_manifest_identity"] = (
            diagnostics_manifest_identity
        )

    return _candidate_bundle_with_fingerprint(
        context=context,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
        source_output_contract_path=source_output_contract_path,
        validation_report_paths=validation_report_paths,
        diagnostics_manifest_path=derived_diagnostics_manifest_path,
        reader="stage4_contract",
        extra_fingerprint_material=extra_fingerprint_material,
    )


@pipeline_node(
    id="stage4_release_candidate_bundle_reader",
    label="Stage 4 Release Candidate Bundle Reader",
    node_type="library",
    description="Read Stage 4 output contract and inventory files into a Stage 5 candidate bundle.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
def read_stage4_release_candidate_bundle(
    *,
    context: ReleasePromotionContext,
    output_contract_path: str | Path,
    output_inventory_path: str | Path | None = None,
    source_output_contract_path: str | None = None,
    validation_report_paths: Sequence[str] = (),
    diagnostics_manifest_path: str | None = None,
) -> ReleaseCandidateInputBundle:
    """Read a candidate bundle from Stage 4 contract and optional inventory files."""

    output_contract = read_contract(output_contract_path)
    inventory_records = (
        tuple(_read_jsonl(output_inventory_path)) if output_inventory_path else ()
    )
    return build_release_candidate_bundle_from_stage4_contract(
        context=context,
        output_contract=output_contract,
        inventory_records=inventory_records,
        source_output_contract_path=source_output_contract_path,
        validation_report_paths=validation_report_paths,
        diagnostics_manifest_path=diagnostics_manifest_path,
    )


def _candidate_bundle_with_fingerprint(
    *,
    context: ReleasePromotionContext,
    artifacts: tuple[ReleaseArtifactSpec, ...],
    source_output_contract_path: str | None,
    validation_report_paths: Sequence[str],
    diagnostics_manifest_path: str | None,
    reader: str,
    extra_fingerprint_material: Mapping[str, Any] | None = None,
) -> ReleaseCandidateInputBundle:
    sorted_artifacts = tuple(sorted(artifacts, key=lambda item: item.relative_path))
    normalized_source_output_contract_path = (
        _normalize_run_contract_path(source_output_contract_path, context)
        if source_output_contract_path is not None
        else None
    )
    normalized_validation_report_paths = tuple(
        _normalize_run_diagnostic_path(path, context)
        for path in validation_report_paths
    )
    normalized_diagnostics_manifest_path = (
        _normalize_run_diagnostic_path(diagnostics_manifest_path, context)
        if diagnostics_manifest_path is not None
        else None
    )
    fingerprint_status, missing_identity_paths = _fingerprint_identity_status(
        sorted_artifacts
    )
    fingerprint = None
    if fingerprint_status == "complete":
        fingerprint = fingerprint_material(
            {
                "reader": reader,
                "context": _context_fingerprint_material(context),
                "artifacts": [
                    _artifact_fingerprint_material(artifact)
                    for artifact in sorted_artifacts
                ],
                "source_output_contract_path": normalized_source_output_contract_path,
                "validation_report_paths": sorted(normalized_validation_report_paths),
                "diagnostics_manifest_path": normalized_diagnostics_manifest_path,
                **(extra_fingerprint_material or {}),
            }
        ).value
    return ReleaseCandidateInputBundle(
        context=context,
        artifacts=sorted_artifacts,
        source_output_contract_path=normalized_source_output_contract_path,
        release_candidate_fingerprint=fingerprint,
        validation_report_paths=normalized_validation_report_paths,
        diagnostics_manifest_path=normalized_diagnostics_manifest_path,
        metadata={
            "reader": reader,
            "fingerprint_status": fingerprint_status,
            "missing_fingerprint_identity_paths": missing_identity_paths,
        },
    )


def _normalize_artifact_metadata_by_path(
    artifact_metadata_by_path: Mapping[str, Mapping[str, Any]],
    *,
    staging_prefix: str | None,
) -> dict[str, Mapping[str, Any]]:
    return {
        strip_staging_prefix(path, staging_prefix): metadata
        for path, metadata in artifact_metadata_by_path.items()
    }


def _legacy_artifact_spec(
    path: str,
    *,
    artifact_metadata_by_path: Mapping[str, Mapping[str, Any]],
) -> ReleaseArtifactSpec:
    metadata = artifact_metadata_by_path.get(path, {})
    return infer_release_artifact_spec(
        path,
        sha256=_optional_record_string(metadata, "sha256"),
        size_bytes=_optional_record_int(metadata, "size_bytes"),
        metadata={
            key: value
            for key, value in jsonable_value(metadata).items()
            if key not in {"sha256", "size_bytes"}
        },
    )


def _merge_artifact_specs(
    contract_specs: Sequence[ReleaseArtifactSpec],
    inventory_specs: Sequence[ReleaseArtifactSpec],
) -> tuple[ReleaseArtifactSpec, ...]:
    merged = _artifact_specs_by_path(contract_specs, source="Stage 4 contract")
    for spec in inventory_specs:
        previous = merged.get(spec.relative_path)
        if previous is None:
            merged[spec.relative_path] = spec
            continue
        merged[spec.relative_path] = _merge_duplicate_artifact_spec(
            contract_spec=previous,
            inventory_spec=spec,
        )
    return tuple(merged.values())


def _artifact_specs_by_path(
    specs: Sequence[ReleaseArtifactSpec],
    *,
    source: str,
) -> dict[str, ReleaseArtifactSpec]:
    by_path: dict[str, ReleaseArtifactSpec] = {}
    for spec in specs:
        previous = by_path.get(spec.relative_path)
        if previous is None:
            by_path[spec.relative_path] = spec
            continue
        by_path[spec.relative_path] = _merge_duplicate_artifact_spec(
            contract_spec=previous,
            inventory_spec=spec,
            source=source,
        )
    return by_path


def _merge_duplicate_artifact_spec(
    *,
    contract_spec: ReleaseArtifactSpec,
    inventory_spec: ReleaseArtifactSpec,
    source: str = "Stage 4 contract/inventory",
) -> ReleaseArtifactSpec:
    comparable_fields = (
        "logical_name",
        "artifact_family",
        "source_stage_id",
        "area_type",
        "area_id",
    )
    for field_name in comparable_fields:
        if getattr(contract_spec, field_name) != getattr(inventory_spec, field_name):
            raise ValueError(
                "Conflicting Stage 4 artifact identity for "
                f"{contract_spec.relative_path}: {field_name}"
            )
    if (
        contract_spec.sha256 is not None
        and inventory_spec.sha256 is not None
        and contract_spec.sha256 != inventory_spec.sha256
    ):
        raise ValueError(
            f"Conflicting {source} sha256 for {contract_spec.relative_path}"
        )
    if (
        contract_spec.size_bytes is not None
        and inventory_spec.size_bytes is not None
        and contract_spec.size_bytes != inventory_spec.size_bytes
    ):
        raise ValueError(
            f"Conflicting {source} size_bytes for {contract_spec.relative_path}"
        )
    return ReleaseArtifactSpec(
        logical_name=contract_spec.logical_name,
        relative_path=contract_spec.relative_path,
        artifact_family=contract_spec.artifact_family,
        source_stage_id=contract_spec.source_stage_id,
        area_type=contract_spec.area_type,
        area_id=contract_spec.area_id,
        sha256=inventory_spec.sha256 or contract_spec.sha256,
        size_bytes=(
            inventory_spec.size_bytes
            if inventory_spec.size_bytes is not None
            else contract_spec.size_bytes
        ),
        required=contract_spec.required or inventory_spec.required,
        metadata={
            "source_contract": jsonable_value(contract_spec.metadata),
            "stage4_inventory": jsonable_value(inventory_spec.metadata),
        },
    )


def _fingerprint_identity_status(
    artifacts: Sequence[ReleaseArtifactSpec],
) -> tuple[str, tuple[str, ...]]:
    missing_identity_paths = tuple(
        artifact.relative_path
        for artifact in artifacts
        if artifact.required
        and (artifact.sha256 is None or artifact.size_bytes is None)
    )
    if missing_identity_paths:
        return "path_only_missing_artifact_identity", missing_identity_paths
    return "complete", ()


def _artifact_spec_from_inventory_record(
    record: Mapping[str, Any],
    *,
    context: ReleasePromotionContext,
) -> ReleaseArtifactSpec:
    _validate_inventory_record_context(record, context)
    relative_path = _inventory_record_path(record, context=context)
    return _artifact_spec_from_stage4_mapping(
        record,
        relative_path=relative_path,
        metadata={"stage4_inventory": jsonable_value(record)},
    )


def _validate_stage4_contract_context(
    output_contract: StageContract,
    context: ReleasePromotionContext,
) -> None:
    if output_contract.stage_id != STAGE_4_BUILD_OUTPUTS:
        raise ValueError("output_contract must be a Stage 4 output contract")
    if output_contract.run_id and output_contract.run_id != context.run_id:
        raise ValueError(
            "output_contract.run_id must match release promotion context.run_id"
        )
    if output_contract.execution.status not in RELEASE_SAFE_STAGE4_EXECUTION_STATUSES:
        raise ValueError(
            "output_contract.execution.status must be completed, reused, or "
            "partially_reused"
        )


def _validate_inventory_record_context(
    record: Mapping[str, Any],
    context: ReleasePromotionContext,
) -> None:
    run_id = _optional_nested_record_string(record, "run_id")
    if run_id is not None and run_id != context.run_id:
        raise ValueError("inventory record run_id must match context.run_id")
    stage_id = _optional_nested_record_string(record, "stage_id")
    if stage_id is not None and stage_id != STAGE_4_BUILD_OUTPUTS:
        raise ValueError("inventory record stage_id must be 4_build_outputs")


def _inventory_record_path(
    record: Mapping[str, Any],
    *,
    context: ReleasePromotionContext,
) -> str:
    paths = _inventory_record_paths(record)
    if not paths:
        raise ValueError("inventory record must include a release path")
    normalized_paths = tuple(
        strip_staging_prefix(path, context.hf_staging_prefix) for path in paths
    )
    if len(set(normalized_paths)) != 1:
        raise ValueError("inventory record path fields must agree")
    return normalized_paths[0]


def _inventory_record_paths(record: Mapping[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []
    for key in _INVENTORY_PATH_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    artifact = record.get("artifact")
    if isinstance(artifact, Mapping):
        for key in _INVENTORY_PATH_KEYS:
            value = artifact.get(key)
            if isinstance(value, str) and value:
                paths.append(value)
    return tuple(paths)


def _artifact_spec_from_contract_artifact(
    artifact: ArtifactRef,
    *,
    context: ReleasePromotionContext,
) -> ReleaseArtifactSpec | None:
    path = artifact.metadata.get("relative_path") or artifact.metadata.get(
        "output_relative_path"
    )
    metadata_path = (
        strip_staging_prefix(path, context.hf_staging_prefix)
        if isinstance(path, str) and path
        else None
    )
    uri_path = _release_path_from_artifact_uri(artifact.uri, context=context)
    metadata_path_is_diagnostic = metadata_path is not None and _is_diagnostics_path(
        metadata_path
    )
    if metadata_path_is_diagnostic or _is_diagnostics_artifact(artifact):
        _diagnostic_artifact_path(artifact, context)
        return None
    if metadata_path is not None and uri_path is not None and metadata_path != uri_path:
        raise ValueError("ArtifactRef metadata path must match artifact.uri")
    path = metadata_path or uri_path
    if path is None:
        return None
    return _artifact_spec_from_stage4_mapping(
        artifact.metadata,
        relative_path=path,
        default_logical_name=artifact.logical_name,
        default_sha256=artifact.sha256,
        default_size_bytes=artifact.size_bytes,
        allow_inferred_semantics=True,
        metadata={
            "source_contract_artifact": artifact.to_dict(),
        },
    )


def _diagnostics_manifest_path(
    output_contract: StageContract,
    *,
    context: ReleasePromotionContext,
) -> str | None:
    for diagnostic in _diagnostic_refs(output_contract):
        artifact = diagnostic.artifact
        if artifact is None:
            continue
        if not _is_diagnostics_manifest_ref(diagnostic.name, diagnostic.kind, artifact):
            continue
        path = _diagnostic_artifact_path(artifact, context)
        if path is not None:
            return path
    for artifact in output_contract.outputs:
        if not _is_diagnostics_artifact(artifact):
            continue
        path = _diagnostic_artifact_path(artifact, context)
        if path is not None:
            return path
    return None


def _diagnostics_manifest_identity(
    output_contract: StageContract,
    *,
    context: ReleasePromotionContext,
) -> dict[str, Any] | None:
    for diagnostic in _diagnostic_refs(output_contract):
        artifact = diagnostic.artifact
        if artifact is None:
            continue
        if not _is_diagnostics_manifest_ref(diagnostic.name, diagnostic.kind, artifact):
            continue
        path = _diagnostic_artifact_path(artifact, context)
        if path is not None:
            return _diagnostic_artifact_identity(path, artifact)
    for artifact in output_contract.outputs:
        if not _is_diagnostics_artifact(artifact):
            continue
        path = _diagnostic_artifact_path(artifact, context)
        if path is not None:
            return _diagnostic_artifact_identity(path, artifact)
    return None


def _diagnostic_artifact_identity(
    path: str,
    artifact: ArtifactRef,
) -> dict[str, Any]:
    return {
        "path": path,
        "logical_name": artifact.logical_name,
        "uri": artifact.uri,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def _optional_record_string(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) and value else None


def _optional_nested_record_string(
    record: Mapping[str, Any],
    key: str,
) -> str | None:
    value = _record_value(record, key)
    return value if isinstance(value, str) and value else None


def _optional_record_int(record: Mapping[str, Any], key: str) -> int | None:
    value = _record_value(record, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"inventory record {key} must be an integer")
    return value


def _artifact_spec_from_stage4_mapping(
    record: Mapping[str, Any],
    *,
    relative_path: str,
    metadata: Mapping[str, Any],
    default_logical_name: str | None = None,
    default_sha256: str | None = None,
    default_size_bytes: int | None = None,
    allow_inferred_semantics: bool = False,
) -> ReleaseArtifactSpec:
    inferred_family, inferred_area_type, inferred_area_id, inferred_stage_id = (
        infer_artifact_identity(relative_path)
    )
    size_bytes = _optional_record_int(record, "size_bytes")
    return ReleaseArtifactSpec(
        logical_name=_stage4_string(
            record,
            "logical_name",
            default=default_logical_name,
            inferred=(
                logical_name_for_release_path(relative_path)
                if allow_inferred_semantics
                else None
            ),
        ),
        relative_path=relative_path,
        artifact_family=_stage4_string(
            record,
            "artifact_family",
            inferred=inferred_family if allow_inferred_semantics else None,
        ),
        source_stage_id=_stage4_string(
            record,
            "source_stage_id",
            inferred=inferred_stage_id if allow_inferred_semantics else None,
        ),
        area_type=(
            _optional_nested_record_string(record, "area_type")
            or (inferred_area_type if allow_inferred_semantics else None)
        ),
        area_id=(
            _optional_nested_record_string(record, "area_id")
            or (inferred_area_id if allow_inferred_semantics else None)
        ),
        sha256=_optional_nested_record_string(record, "sha256") or default_sha256,
        size_bytes=size_bytes if size_bytes is not None else default_size_bytes,
        required=_record_value(record, "required", default=True),
        metadata=metadata,
    )


def _stage4_string(
    record: Mapping[str, Any],
    key: str,
    *,
    default: str | None = None,
    inferred: str | None = None,
) -> str:
    value = _record_value(record, key, default=default or inferred)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Stage 4 candidate records must include {key}")
    return value


def _release_path_from_artifact_uri(
    uri: str,
    *,
    context: ReleasePromotionContext,
) -> str | None:
    parsed = urlparse(uri)
    if parsed.scheme or parsed.netloc:
        _validate_uri_repo(parsed, context)
    raw_path = parsed.path.lstrip("/") if parsed.scheme else uri
    candidate_paths = [raw_path]
    if parsed.netloc:
        candidate_paths.append(f"{parsed.netloc}/{raw_path}")
    for candidate in candidate_paths:
        if context.hf_staging_prefix and context.hf_staging_prefix in candidate:
            return strip_staging_prefix(
                candidate[candidate.index(context.hf_staging_prefix) :],
                context.hf_staging_prefix,
            )
        if parsed.scheme or parsed.netloc:
            if _contains_release_artifact_path(candidate):
                raise ValueError(
                    "external artifact URI must point under the expected staging prefix"
                )
            continue
        if candidate.startswith("staging/"):
            return strip_staging_prefix(candidate, context.hf_staging_prefix)
        for prefix in ("states/", "districts/", "cities/", "national/"):
            if prefix in candidate:
                return normalize_release_path(candidate[candidate.index(prefix) :])
        for path in BASE_RELEASE_ARTIFACT_PATHS:
            if candidate == path or candidate.endswith(f"/{path}"):
                return normalize_release_path(
                    candidate[candidate.rindex(path) :],
                )
    return None


def _contains_release_artifact_path(candidate: str) -> bool:
    if any(
        prefix in candidate
        for prefix in ("states/", "districts/", "cities/", "national/")
    ):
        return True
    return any(
        candidate.endswith(f"/{path}") or candidate == path
        for path in BASE_RELEASE_ARTIFACT_PATHS
    )


def _validate_uri_repo(
    parsed_uri,
    context: ReleasePromotionContext,
) -> None:
    if not parsed_uri.scheme or not parsed_uri.netloc:
        return
    path_parts = parsed_uri.path.strip("/").split("/")
    if not path_parts or not path_parts[0]:
        return
    repo_name = f"{parsed_uri.netloc}/{path_parts[0]}"
    if repo_name != context.hf_repo_name:
        raise ValueError("external artifact URI repo must match context.hf_repo_name")


def _is_diagnostics_manifest_ref(
    name: str,
    kind: str,
    artifact: ArtifactRef,
) -> bool:
    return (
        name == "diagnostics_manifest"
        or kind == "diagnostics_manifest"
        or artifact.logical_name == "diagnostics_manifest"
    )


def _diagnostic_refs(output_contract: StageContract):
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


def _diagnostic_artifact_path(
    artifact: ArtifactRef,
    context: ReleasePromotionContext,
) -> str | None:
    path = artifact.metadata.get("relative_path") or artifact.metadata.get(
        "output_relative_path"
    )
    metadata_path = (
        _normalize_run_diagnostic_path(path, context)
        if isinstance(path, str) and path
        else None
    )
    uri_path = _diagnostic_path_from_uri(artifact.uri, context)
    if metadata_path is not None and uri_path is not None and metadata_path != uri_path:
        raise ValueError("Diagnostic artifact metadata path must match artifact.uri")
    return metadata_path or uri_path


def _diagnostic_path_from_uri(
    uri: str,
    context: ReleasePromotionContext,
) -> str | None:
    parsed = urlparse(uri)
    if parsed.scheme or parsed.netloc:
        _validate_uri_repo(parsed, context)
    raw_path = parsed.path.lstrip("/") if parsed.scheme else uri
    candidate_paths = [raw_path]
    if parsed.netloc:
        candidate_paths.append(f"{parsed.netloc}/{raw_path}")
    marker = f"calibration/runs/{context.run_id}/diagnostics/"
    for candidate in candidate_paths:
        if marker in candidate:
            return _normalize_run_diagnostic_path(
                candidate[candidate.index(marker) :],
                context,
            )
        if "calibration/runs/" in candidate:
            raise ValueError("diagnostic artifact URI must match context.run_id")
    return None


def _normalize_run_contract_path(
    path: str,
    context: ReleasePromotionContext,
) -> str:
    normalized = strip_staging_prefix(path, context.hf_staging_prefix)
    required_prefix = f"calibration/runs/{context.run_id}/"
    if not normalized.startswith(required_prefix):
        raise ValueError(
            "source_output_contract_path must live under "
            f"{required_prefix} for context.run_id"
        )
    return normalized


def _normalize_run_diagnostic_path(
    path: str,
    context: ReleasePromotionContext,
) -> str:
    normalized = strip_staging_prefix(path, context.hf_staging_prefix)
    required_prefix = f"calibration/runs/{context.run_id}/diagnostics/"
    if not normalized.startswith(required_prefix):
        raise ValueError(
            "diagnostic and validation report paths must live under "
            f"{required_prefix} for context.run_id"
        )
    return normalized


def _is_diagnostics_artifact(artifact: ArtifactRef) -> bool:
    path = artifact.metadata.get("relative_path") or artifact.metadata.get(
        "output_relative_path"
    )
    return (
        artifact.logical_name == "diagnostics_manifest"
        or artifact.metadata.get("artifact_family") == "diagnostics"
        or (isinstance(path, str) and _is_diagnostics_path(path))
    )


def _is_diagnostics_path(path: str) -> bool:
    normalized = normalize_release_path(path)
    parts = normalized.split("/")
    return (
        len(parts) >= 5
        and parts[:2] == ["calibration", "runs"]
        and parts[3] == "diagnostics"
    )


def _context_fingerprint_material(
    context: ReleasePromotionContext,
) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "candidate_version": context.candidate_version,
        "release_version": context.release_version,
        "hf_repo_name": context.hf_repo_name,
        "hf_repo_type": context.hf_repo_type,
        "gcs_bucket_name": context.gcs_bucket_name,
        "base_release_version": context.base_release_version,
        "release_bump": context.release_bump,
        "modal_app_name": context.modal_app_name,
        "modal_environment": context.modal_environment,
        "hf_staging_prefix": context.hf_staging_prefix,
        "schema_version": context.schema_version,
    }


def _artifact_fingerprint_material(
    artifact: ReleaseArtifactSpec,
) -> dict[str, Any]:
    return {
        "logical_name": artifact.logical_name,
        "relative_path": artifact.relative_path,
        "artifact_family": artifact.artifact_family,
        "source_stage_id": artifact.source_stage_id,
        "area_type": artifact.area_type,
        "area_id": artifact.area_id,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "required": artifact.required,
        "schema_version": artifact.schema_version,
    }


def _record_value(
    record: Mapping[str, Any],
    key: str,
    *,
    default: Any = None,
) -> Any:
    if key in record:
        return record[key]
    artifact = record.get("artifact")
    if isinstance(artifact, Mapping) and key in artifact:
        return artifact[key]
    return default


def _read_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    with Path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError("output inventory JSONL rows must be mappings")
            yield payload
