"""Stage 4 artifact conversion helpers for release candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from policyengine_us_data.stage_contracts import ArtifactRef
from policyengine_us_data.stage_contracts._coercion import jsonable_value

from .artifact_uris import release_path_from_artifact_uri
from .artifacts import (
    ReleaseArtifactSpec,
    infer_artifact_identity,
    logical_name_for_release_path,
    strip_staging_prefix,
)
from .context import ReleasePromotionContext
from .diagnostics import (
    diagnostic_artifact_path,
    is_diagnostics_artifact,
    is_diagnostics_path,
)
from .stage4_inventory import (
    inventory_record_path,
    optional_nested_record_string,
    optional_record_int,
    record_value,
    validate_inventory_record_context,
)


def merge_artifact_specs(
    contract_specs: Sequence[ReleaseArtifactSpec],
    inventory_specs: Sequence[ReleaseArtifactSpec],
) -> tuple[ReleaseArtifactSpec, ...]:
    """Merge Stage 4 contract and inventory artifact specs."""

    merged = artifact_specs_by_path(contract_specs, source="Stage 4 contract")
    for spec in inventory_specs:
        previous = merged.get(spec.relative_path)
        if previous is None:
            merged[spec.relative_path] = spec
            continue
        merged[spec.relative_path] = merge_duplicate_artifact_spec(
            contract_spec=previous,
            inventory_spec=spec,
        )
    return tuple(merged.values())


def artifact_specs_by_path(
    specs: Sequence[ReleaseArtifactSpec],
    *,
    source: str,
) -> dict[str, ReleaseArtifactSpec]:
    """Index artifact specs by path and merge duplicate records."""

    by_path: dict[str, ReleaseArtifactSpec] = {}
    for spec in specs:
        previous = by_path.get(spec.relative_path)
        if previous is None:
            by_path[spec.relative_path] = spec
            continue
        by_path[spec.relative_path] = merge_duplicate_artifact_spec(
            contract_spec=previous,
            inventory_spec=spec,
            source=source,
        )
    return by_path


def merge_duplicate_artifact_spec(
    *,
    contract_spec: ReleaseArtifactSpec,
    inventory_spec: ReleaseArtifactSpec,
    source: str = "Stage 4 contract/inventory",
) -> ReleaseArtifactSpec:
    """Merge duplicate artifact specs while rejecting identity conflicts."""

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


def artifact_spec_from_inventory_record(
    record: Mapping[str, Any],
    *,
    context: ReleasePromotionContext,
) -> ReleaseArtifactSpec:
    """Build a release artifact spec from a Stage 4 inventory record."""

    validate_inventory_record_context(record, context)
    relative_path = inventory_record_path(record, context=context)
    return artifact_spec_from_stage4_mapping(
        record,
        relative_path=relative_path,
        metadata={"stage4_inventory": jsonable_value(record)},
    )


def artifact_spec_from_contract_artifact(
    artifact: ArtifactRef,
    *,
    context: ReleasePromotionContext,
) -> ReleaseArtifactSpec | None:
    """Build a release artifact spec from a Stage 4 contract output."""

    path = artifact.metadata.get("relative_path") or artifact.metadata.get(
        "output_relative_path"
    )
    metadata_path = (
        strip_staging_prefix(path, context.hf_staging_prefix)
        if isinstance(path, str) and path
        else None
    )
    uri_path = release_path_from_artifact_uri(artifact.uri, context=context)
    metadata_path_is_diagnostic = metadata_path is not None and is_diagnostics_path(
        metadata_path
    )
    if metadata_path_is_diagnostic or is_diagnostics_artifact(artifact):
        diagnostic_artifact_path(artifact, context)
        return None
    if metadata_path is not None and uri_path is not None and metadata_path != uri_path:
        raise ValueError("ArtifactRef metadata path must match artifact.uri")
    path = metadata_path or uri_path
    if path is None:
        return None
    return artifact_spec_from_stage4_mapping(
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


def artifact_spec_from_stage4_mapping(
    record: Mapping[str, Any],
    *,
    relative_path: str,
    metadata: Mapping[str, Any],
    default_logical_name: str | None = None,
    default_sha256: str | None = None,
    default_size_bytes: int | None = None,
    allow_inferred_semantics: bool = False,
) -> ReleaseArtifactSpec:
    """Build an artifact spec from a Stage 4 mapping-shaped record."""

    inferred_family, inferred_area_type, inferred_area_id, inferred_stage_id = (
        infer_artifact_identity(relative_path)
    )
    size_bytes = optional_record_int(record, "size_bytes")
    return ReleaseArtifactSpec(
        logical_name=stage4_string(
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
        artifact_family=stage4_string(
            record,
            "artifact_family",
            inferred=inferred_family if allow_inferred_semantics else None,
        ),
        source_stage_id=stage4_string(
            record,
            "source_stage_id",
            inferred=inferred_stage_id if allow_inferred_semantics else None,
        ),
        area_type=(
            optional_nested_record_string(record, "area_type")
            or (inferred_area_type if allow_inferred_semantics else None)
        ),
        area_id=(
            optional_nested_record_string(record, "area_id")
            or (inferred_area_id if allow_inferred_semantics else None)
        ),
        sha256=optional_nested_record_string(record, "sha256") or default_sha256,
        size_bytes=size_bytes if size_bytes is not None else default_size_bytes,
        required=record_value(record, "required", default=True),
        metadata=metadata,
    )


def stage4_string(
    record: Mapping[str, Any],
    key: str,
    *,
    default: str | None = None,
    inferred: str | None = None,
) -> str:
    """Return a required Stage 4 string field."""

    value = record_value(record, key, default=default or inferred)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Stage 4 candidate records must include {key}")
    return value
