"""Artifact specifications for the Stage 1 dataset build."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from policyengine_us_data.pipeline_metadata import pipeline_node


ScriptOutput = str | list[str]


@dataclass(frozen=True, kw_only=True)
class DatasetArtifactSpec:
    """A durable or checkpointed output produced by Stage 1."""

    filename: str
    logical_name: str
    artifact_family: str
    substage_id: str
    period: int | None = None
    storage_path: str | None = None
    script_path: str | None = None
    required: bool = True
    required_for_stage_2: bool = False
    yearless_alias: bool = False
    contract_output: bool = True
    skip_when_enhanced_cps_skipped: bool = False
    skip_when_stage_5_skipped: bool = False


_UPRATING_SCRIPT = "policyengine_us_data/utils/uprating.py"
_ACS_SCRIPT = "policyengine_us_data/datasets/acs/acs.py"
_IRS_PUF_SCRIPT = "policyengine_us_data/datasets/puf/irs_puf.py"
_CPS_SCRIPT = "policyengine_us_data/datasets/cps/cps.py"
_PUF_SCRIPT = "policyengine_us_data/datasets/puf/puf.py"
_EXTENDED_CPS_SCRIPT = "policyengine_us_data/datasets/cps/extended_cps.py"
_ENHANCED_CPS_SCRIPT = "policyengine_us_data/datasets/cps/enhanced_cps.py"
_STRATIFIED_CPS_SCRIPT = "policyengine_us_data/calibration/create_stratified_cps.py"
_SOURCE_IMPUTED_CPS_SCRIPT = (
    "policyengine_us_data/calibration/create_source_imputed_cps.py"
)
_SMALL_ENHANCED_CPS_SCRIPT = "policyengine_us_data/datasets/cps/small_enhanced_cps.py"


STAGE_1_ARTIFACT_SPECS: tuple[DatasetArtifactSpec, ...] = (
    DatasetArtifactSpec(
        filename="uprating_factors.csv",
        logical_name="uprating_factors",
        artifact_family="uprating_table",
        substage_id="1a_raw_data_download",
        storage_path="policyengine_us_data/storage/uprating_factors.csv",
        script_path=_UPRATING_SCRIPT,
        contract_output=False,
    ),
    DatasetArtifactSpec(
        filename="acs_2022.h5",
        logical_name="acs_2022",
        artifact_family="dataset",
        period=2022,
        substage_id="1b_base_dataset_construction",
        storage_path="policyengine_us_data/storage/acs_2022.h5",
        script_path=_ACS_SCRIPT,
    ),
    DatasetArtifactSpec(
        filename="irs_puf_2015.h5",
        logical_name="irs_puf_2015",
        artifact_family="dataset",
        period=2015,
        substage_id="1b_base_dataset_construction",
        storage_path="policyengine_us_data/storage/irs_puf_2015.h5",
        script_path=_IRS_PUF_SCRIPT,
    ),
    DatasetArtifactSpec(
        filename="cps_2024.h5",
        logical_name="cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1b_base_dataset_construction",
        storage_path="policyengine_us_data/storage/cps_2024.h5",
        script_path=_CPS_SCRIPT,
    ),
    DatasetArtifactSpec(
        filename="puf_2024.h5",
        logical_name="puf_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1b_base_dataset_construction",
        storage_path="policyengine_us_data/storage/puf_2024.h5",
        script_path=_PUF_SCRIPT,
    ),
    DatasetArtifactSpec(
        filename="extended_cps_2024.h5",
        logical_name="extended_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1c_extended_cps_puf_clone",
        storage_path="policyengine_us_data/storage/extended_cps_2024.h5",
        script_path=_EXTENDED_CPS_SCRIPT,
    ),
    DatasetArtifactSpec(
        filename="enhanced_cps_2024.h5",
        logical_name="enhanced_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1d_enhanced_cps_reweighting",
        storage_path="policyengine_us_data/storage/enhanced_cps_2024.h5",
        script_path=_ENHANCED_CPS_SCRIPT,
        skip_when_enhanced_cps_skipped=True,
    ),
    DatasetArtifactSpec(
        filename="enhanced_cps_2024.clone_diagnostics.json",
        logical_name="enhanced_cps_2024_clone_diagnostics",
        artifact_family="diagnostic",
        period=2024,
        substage_id="1d_enhanced_cps_reweighting",
        storage_path=(
            "policyengine_us_data/storage/enhanced_cps_2024.clone_diagnostics.json"
        ),
        script_path=_ENHANCED_CPS_SCRIPT,
        contract_output=False,
        skip_when_enhanced_cps_skipped=True,
    ),
    DatasetArtifactSpec(
        filename="calibration_log.csv",
        logical_name="enhanced_cps_calibration_log",
        artifact_family="log",
        substage_id="1d_enhanced_cps_reweighting",
        storage_path="calibration_log.csv",
        script_path=_ENHANCED_CPS_SCRIPT,
        contract_output=False,
        skip_when_enhanced_cps_skipped=True,
    ),
    DatasetArtifactSpec(
        filename="stratified_extended_cps_2024.h5",
        logical_name="stratified_extended_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1e_stratified_cps",
        storage_path="policyengine_us_data/storage/stratified_extended_cps_2024.h5",
        script_path=_STRATIFIED_CPS_SCRIPT,
    ),
    DatasetArtifactSpec(
        filename="source_imputed_stratified_extended_cps_2024.h5",
        logical_name="source_imputed_stratified_extended_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1f_source_imputation",
        storage_path=(
            "policyengine_us_data/storage/"
            "source_imputed_stratified_extended_cps_2024.h5"
        ),
        script_path=_SOURCE_IMPUTED_CPS_SCRIPT,
        required_for_stage_2=True,
        skip_when_stage_5_skipped=True,
    ),
    DatasetArtifactSpec(
        filename="small_enhanced_cps_2024.h5",
        logical_name="small_enhanced_cps_2024",
        artifact_family="dataset",
        period=2024,
        substage_id="1d_enhanced_cps_reweighting",
        storage_path="policyengine_us_data/storage/small_enhanced_cps_2024.h5",
        script_path=_SMALL_ENHANCED_CPS_SCRIPT,
        skip_when_enhanced_cps_skipped=True,
        skip_when_stage_5_skipped=True,
    ),
    DatasetArtifactSpec(
        filename="source_imputed_stratified_extended_cps.h5",
        logical_name="source_imputed_stratified_extended_cps",
        artifact_family="dataset",
        period=2024,
        substage_id="1f_source_imputation",
        required_for_stage_2=True,
        yearless_alias=True,
        skip_when_stage_5_skipped=True,
    ),
    DatasetArtifactSpec(
        filename="policy_data.db",
        logical_name="policy_data_db",
        artifact_family="target_database",
        substage_id="1g_stage_base_datasets",
        storage_path="policyengine_us_data/storage/calibration/policy_data.db",
        required_for_stage_2=True,
    ),
    DatasetArtifactSpec(
        filename="build_log.txt",
        logical_name="build_log",
        artifact_family="log",
        substage_id="1g_stage_base_datasets",
    ),
    DatasetArtifactSpec(
        filename="data_build_checkpoint_stats.json",
        logical_name="data_build_checkpoint_stats",
        artifact_family="execution_metadata",
        substage_id="1g_stage_base_datasets",
    ),
)

_STAGE_1_CONTRACT_OUTPUT_FILENAMES = (
    "acs_2022.h5",
    "irs_puf_2015.h5",
    "cps_2024.h5",
    "puf_2024.h5",
    "extended_cps_2024.h5",
    "enhanced_cps_2024.h5",
    "small_enhanced_cps_2024.h5",
    "stratified_extended_cps_2024.h5",
    "source_imputed_stratified_extended_cps_2024.h5",
    "source_imputed_stratified_extended_cps.h5",
    "policy_data.db",
    "build_log.txt",
    "data_build_checkpoint_stats.json",
)


@pipeline_node(
    id="stage_1_dataset_artifact_specs",
    label="Stage 1 Dataset Artifact Specs",
    node_type="library",
    description="Canonical artifact inventory for Stage 1 dataset-build outputs.",
    source_file="policyengine_us_data/build_datasets/artifacts.py",
    status="current",
    stability="stable",
    pathways=["data_build", "stage_contracts", "pipeline_docs"],
    artifacts_out=[
        "acs_2022.h5",
        "irs_puf_2015.h5",
        "cps_2024.h5",
        "puf_2024.h5",
        "extended_cps_2024.h5",
        "enhanced_cps_2024.h5",
        "stratified_extended_cps_2024.h5",
        "source_imputed_stratified_extended_cps_2024.h5",
        "source_imputed_stratified_extended_cps.h5",
        "policy_data.db",
    ],
    validation_commands=["uv run pytest tests/unit/test_build_dataset_specs.py"],
)
def stage_1_artifact_specs() -> tuple[DatasetArtifactSpec, ...]:
    """Return all artifact specs known to the Stage 1 dataset build."""

    return STAGE_1_ARTIFACT_SPECS


def stage_1_contract_artifact_specs() -> tuple[DatasetArtifactSpec, ...]:
    """Return artifact specs emitted in the Stage 1 handoff contract."""

    specs_by_filename = {spec.filename: spec for spec in STAGE_1_ARTIFACT_SPECS}
    return tuple(
        specs_by_filename[filename] for filename in _STAGE_1_CONTRACT_OUTPUT_FILENAMES
    )


def stage_1_script_outputs() -> Mapping[str, ScriptOutput]:
    """Return the checkpoint output mapping consumed by Modal data-build."""

    outputs: dict[str, list[str]] = {}
    for spec in STAGE_1_ARTIFACT_SPECS:
        if spec.script_path is None or spec.storage_path is None:
            continue
        outputs.setdefault(spec.script_path, []).append(spec.storage_path)

    return {
        script_path: paths[0] if len(paths) == 1 else paths
        for script_path, paths in outputs.items()
    }


__all__ = [
    "DatasetArtifactSpec",
    "STAGE_1_ARTIFACT_SPECS",
    "ScriptOutput",
    "stage_1_artifact_specs",
    "stage_1_contract_artifact_specs",
    "stage_1_script_outputs",
]
