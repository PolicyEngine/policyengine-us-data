"""Stage 4 contract and inventory readers for Stage 5 candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import (
    DiagnosticRef,
    StageContract,
    read_contract,
)
from policyengine_us_data.stage_contracts.stages import STAGE_4_BUILD_OUTPUTS

from .candidate import ReleaseCandidateInputBundle
from .context import ReleasePromotionContext
from .diagnostics import (
    diagnostics_manifest_identity,
    diagnostics_manifest_path,
)
from .fingerprints import candidate_bundle_with_fingerprint
from .stage4_artifacts import (
    artifact_spec_from_contract_artifact,
    artifact_spec_from_inventory_record,
    merge_artifact_specs,
)
from .stage4_inventory import read_jsonl

RELEASE_SAFE_STAGE4_EXECUTION_STATUSES = frozenset(
    {"completed", "reused", "partially_reused"}
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
    validation_report_refs: Sequence[DiagnosticRef] = (),
    diagnostics_manifest_path: str | None = None,
) -> ReleaseCandidateInputBundle:
    """Build a candidate bundle from a Stage 4 output contract shape."""

    validate_stage4_contract_context(output_contract, context)

    inventory_specs = tuple(
        artifact_spec_from_inventory_record(
            record,
            context=context,
        )
        for record in inventory_records
    )
    contract_specs = tuple(
        spec
        for artifact in output_contract.outputs
        if (
            spec := artifact_spec_from_contract_artifact(
                artifact,
                context=context,
            )
        )
        is not None
    )
    artifacts = merge_artifact_specs(contract_specs, inventory_specs)
    if not artifacts:
        raise ValueError(
            "Stage 4 candidate reader needs inventory records or output artifacts "
            "with release-relative paths"
        )

    derived_diagnostics_manifest_path = (
        diagnostics_manifest_path
        or diagnostics_manifest_path_from_contract(output_contract, context=context)
    )
    extra_fingerprint_material: dict[str, Any] = {
        "source_output_contract_fingerprint": output_contract.fingerprint.value,
        "source_output_contract_stage_id": output_contract.stage_id,
    }
    manifest_identity = diagnostics_manifest_identity(
        output_contract,
        context=context,
    )
    if manifest_identity is not None:
        extra_fingerprint_material["diagnostics_manifest_identity"] = manifest_identity

    return candidate_bundle_with_fingerprint(
        context=context,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
        source_output_contract_path=source_output_contract_path,
        validation_report_paths=validation_report_paths,
        validation_report_refs=validation_report_refs,
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
    validation_report_refs: Sequence[DiagnosticRef] = (),
    diagnostics_manifest_path: str | None = None,
) -> ReleaseCandidateInputBundle:
    """Read a candidate bundle from Stage 4 contract and optional inventory files."""

    output_contract = read_contract(output_contract_path)
    inventory_records = (
        tuple(read_jsonl(output_inventory_path)) if output_inventory_path else ()
    )
    return build_release_candidate_bundle_from_stage4_contract(
        context=context,
        output_contract=output_contract,
        inventory_records=inventory_records,
        source_output_contract_path=source_output_contract_path,
        validation_report_paths=validation_report_paths,
        validation_report_refs=validation_report_refs,
        diagnostics_manifest_path=diagnostics_manifest_path,
    )


def validate_stage4_contract_context(
    output_contract: StageContract,
    context: ReleasePromotionContext,
) -> None:
    """Validate that a Stage 4 contract is safe for candidate construction."""

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


def diagnostics_manifest_path_from_contract(
    output_contract: StageContract,
    *,
    context: ReleasePromotionContext,
) -> str | None:
    """Return the diagnostics manifest path from a Stage 4 output contract."""

    return diagnostics_manifest_path(output_contract, context=context)
