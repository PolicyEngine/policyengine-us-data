"""Stage 1 dataset-build contract assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policyengine_us_data.utils.step_manifest import sha256_file

from .artifacts import ArtifactRef
from .contracts import StageContract
from .execution import ExecutionRecord, ReuseSummary
from .fingerprints import fingerprint_material
from .stages import STAGE_1_BUILD_DATASETS, contract_type_for_stage
from .substages import SubstageRecord

DATASET_BUILD_OUTPUT_CONTRACT_FILENAME = "dataset_build_output.json"
DATASET_BUILD_OUTPUT_CONTRACT_TYPE = contract_type_for_stage(STAGE_1_BUILD_DATASETS)


@dataclass(frozen=True, kw_only=True)
class _Stage1ArtifactSpec:
    filename: str
    logical_name: str
    artifact_family: str
    substage_id: str
    period: int | None = None
    required: bool = True
    required_for_stage_2: bool = False
    yearless_alias: bool = False
    skip_when_enhanced_cps_skipped: bool = False


_STAGE_1_ARTIFACTS: tuple[_Stage1ArtifactSpec, ...] = (
    _Stage1ArtifactSpec(
        filename="acs_2022.h5",
        logical_name="acs_2022",
        artifact_family="dataset",
        period=2022,
        substage_id="1b_base_dataset_construction",
    ),
    _Stage1ArtifactSpec(
        filename="irs_puf_2015.h5",
        logical_name="irs_puf_2015",
        artifact_family="dataset",
        period=2015,
        substage_id="1b_base_dataset_construction",
    ),
    _Stage1ArtifactSpec(
        filename="cps_2024.h5",
        logical_name="cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1b_base_dataset_construction",
    ),
    _Stage1ArtifactSpec(
        filename="puf_2024.h5",
        logical_name="puf_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1b_base_dataset_construction",
    ),
    _Stage1ArtifactSpec(
        filename="extended_cps_2024.h5",
        logical_name="extended_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1c_extended_cps_puf_clone",
    ),
    _Stage1ArtifactSpec(
        filename="enhanced_cps_2024.h5",
        logical_name="enhanced_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1d_enhanced_cps_reweighting",
        skip_when_enhanced_cps_skipped=True,
    ),
    _Stage1ArtifactSpec(
        filename="small_enhanced_cps_2024.h5",
        logical_name="small_enhanced_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1d_enhanced_cps_reweighting",
        skip_when_enhanced_cps_skipped=True,
    ),
    _Stage1ArtifactSpec(
        filename="stratified_extended_cps_2024.h5",
        logical_name="stratified_extended_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1e_stratified_cps",
    ),
    _Stage1ArtifactSpec(
        filename="source_imputed_stratified_extended_cps_2024.h5",
        logical_name="source_imputed_stratified_extended_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1f_source_imputation",
        required_for_stage_2=True,
    ),
    _Stage1ArtifactSpec(
        filename="source_imputed_stratified_extended_cps.h5",
        logical_name="source_imputed_stratified_extended_cps",
        artifact_family="dataset",
        period=2024,
        substage_id="1f_source_imputation",
        required_for_stage_2=True,
        yearless_alias=True,
    ),
    _Stage1ArtifactSpec(
        filename="policy_data.db",
        logical_name="policy_data_db",
        artifact_family="target_database",
        substage_id="1g_stage_base_datasets",
        required_for_stage_2=True,
    ),
    _Stage1ArtifactSpec(
        filename="build_log.txt",
        logical_name="build_log",
        artifact_family="log",
        substage_id="1g_stage_base_datasets",
    ),
    _Stage1ArtifactSpec(
        filename="data_build_checkpoint_stats.json",
        logical_name="data_build_checkpoint_stats",
        artifact_family="execution_metadata",
        substage_id="1g_stage_base_datasets",
    ),
)

_SUBSTAGE_IDS = (
    "1a_raw_data_download",
    "1b_base_dataset_construction",
    "1c_extended_cps_puf_clone",
    "1d_enhanced_cps_reweighting",
    "1e_stratified_cps",
    "1f_source_imputation",
    "1g_stage_base_datasets",
)


def build_dataset_build_output_contract(
    *,
    artifacts_dir: Path,
    run_id: str,
    code_sha: str,
    package_version: str | None,
    checkpoint_stats: Mapping[str, int],
    started_at: str | None,
    completed_at: str,
    duration_s: float | None = None,
    upload_requested: bool = False,
    stage_only: bool = False,
    skip_enhanced_cps: bool = False,
) -> StageContract:
    """Build the Stage 1 handoff contract from copied pipeline artifacts."""

    artifacts_dir = Path(artifacts_dir)
    parameters = {
        "period": 2024,
        "skip_enhanced_cps": skip_enhanced_cps,
        "stage_only": stage_only,
        "upload_requested": upload_requested,
    }
    outputs = _stage_1_outputs(
        artifacts_dir=artifacts_dir,
        skip_enhanced_cps=skip_enhanced_cps,
    )
    execution = _execution_record(
        checkpoint_stats=checkpoint_stats,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
    )
    fingerprint = _fingerprint_for_dataset_build(
        code_sha=code_sha,
        package_version=package_version,
        parameters=parameters,
        outputs=outputs,
    )
    return StageContract(
        contract_type=DATASET_BUILD_OUTPUT_CONTRACT_TYPE,
        stage_id=STAGE_1_BUILD_DATASETS,
        run_id=run_id,
        created_at=completed_at,
        code_sha=code_sha,
        package_version=package_version,
        outputs=outputs,
        parameters=parameters,
        fingerprint=fingerprint,
        substages=_stage_1_substages(
            outputs=outputs,
            skip_enhanced_cps=skip_enhanced_cps,
        ),
        execution=execution,
        metadata={
            "artifact_count": len(outputs),
            "artifact_directory": str(artifacts_dir),
            "contract_file": DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
        },
    )


def _stage_1_outputs(
    *,
    artifacts_dir: Path,
    skip_enhanced_cps: bool,
) -> tuple[ArtifactRef, ...]:
    outputs: list[ArtifactRef] = []
    missing_required: list[str] = []
    for spec in _STAGE_1_ARTIFACTS:
        if skip_enhanced_cps and spec.skip_when_enhanced_cps_skipped:
            continue
        artifact_path = artifacts_dir / spec.filename
        if not artifact_path.exists():
            if spec.required:
                missing_required.append(spec.filename)
            continue
        metadata = {
            "artifact_family": spec.artifact_family,
            "substage_id": spec.substage_id,
        }
        if spec.period is not None:
            metadata["period"] = spec.period
        if spec.required_for_stage_2:
            metadata["required_for_stage_2"] = True
        if spec.yearless_alias:
            metadata["yearless_alias"] = True
        outputs.append(
            _artifact_ref_from_path(
                logical_name=spec.logical_name,
                path=artifact_path,
                metadata=metadata,
            )
        )
    if missing_required:
        raise FileNotFoundError(
            "Missing Stage 1 handoff artifact(s): "
            + ", ".join(sorted(missing_required))
        )
    return tuple(outputs)


def _artifact_ref_from_path(
    *,
    logical_name: str,
    path: Path,
    metadata: Mapping[str, Any],
) -> ArtifactRef:
    return ArtifactRef(
        logical_name=logical_name,
        uri=path.resolve().as_uri(),
        sha256=f"sha256:{sha256_file(path)}",
        size_bytes=path.stat().st_size,
        media_type=_media_type_for_path(path),
        metadata=metadata,
    )


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".h5":
        return "application/x-hdf5"
    if suffix == ".db":
        return "application/vnd.sqlite3"
    if suffix == ".json":
        return "application/json"
    if suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


def _stage_1_substages(
    *,
    outputs: tuple[ArtifactRef, ...],
    skip_enhanced_cps: bool,
) -> tuple[SubstageRecord, ...]:
    output_by_substage: dict[str, list[ArtifactRef]] = {
        substage_id: [] for substage_id in _SUBSTAGE_IDS
    }
    for artifact in outputs:
        substage_id = artifact.metadata.get("substage_id")
        if isinstance(substage_id, str) and substage_id in output_by_substage:
            output_by_substage[substage_id].append(artifact)

    records: list[SubstageRecord] = []
    for substage_id in _SUBSTAGE_IDS:
        status = "completed"
        if substage_id == "1d_enhanced_cps_reweighting" and skip_enhanced_cps:
            status = "skipped"
        reuse_mode = "checkpointable"
        if substage_id in {
            "1a_raw_data_download",
        }:
            reuse_mode = "observed_only"
        elif substage_id in {
            "1e_stratified_cps",
            "1f_source_imputation",
            "1g_stage_base_datasets",
        }:
            reuse_mode = "handoff"
        records.append(
            SubstageRecord(
                substage_id=substage_id,
                status=status,
                outputs=tuple(output_by_substage[substage_id]),
                reuse_mode=reuse_mode,
            )
        )
    return tuple(records)


def _execution_record(
    *,
    checkpoint_stats: Mapping[str, int],
    started_at: str | None,
    completed_at: str,
    duration_s: float | None,
) -> ExecutionRecord:
    reuse_summary = ReuseSummary(
        expected_outputs=int(checkpoint_stats.get("expected_outputs", 0)),
        valid_reused_outputs=int(checkpoint_stats.get("valid_reused_outputs", 0)),
        recomputed_outputs=int(checkpoint_stats.get("recomputed_outputs", 0)),
        invalid_outputs=int(checkpoint_stats.get("invalid_outputs", 0)),
    )
    return ExecutionRecord(
        status="completed",
        started_at=started_at,
        completed_at=completed_at,
        duration_s=float(duration_s) if duration_s is not None else None,
        reuse_decision=_reuse_decision(checkpoint_stats),
        reuse_summary=reuse_summary,
    )


def _reuse_decision(checkpoint_stats: Mapping[str, int]) -> str:
    reused = int(checkpoint_stats.get("valid_reused_outputs", 0))
    recomputed = int(checkpoint_stats.get("recomputed_outputs", 0))
    if reused and recomputed:
        return "partially_reused"
    if reused:
        return "reused"
    return "computed"


def _fingerprint_for_dataset_build(
    *,
    code_sha: str,
    package_version: str | None,
    parameters: Mapping[str, Any],
    outputs: tuple[ArtifactRef, ...],
):
    material = {
        "stage_id": STAGE_1_BUILD_DATASETS,
        "contract_type": DATASET_BUILD_OUTPUT_CONTRACT_TYPE,
        "code_sha": code_sha,
        "package_version": package_version,
        "parameters": parameters,
        "outputs": [
            {
                "logical_name": output.logical_name,
                "sha256": output.sha256,
                "size_bytes": output.size_bytes,
            }
            for output in sorted(outputs, key=lambda item: item.logical_name)
        ],
    }
    return fingerprint_material(material)
