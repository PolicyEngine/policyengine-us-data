"""Canonical Stage 1 dataset-build specifications."""

from .artifacts import (
    DatasetArtifactSpec,
    STAGE_1_ARTIFACT_SPECS,
    stage_1_artifact_specs,
    stage_1_contract_artifact_specs,
    stage_1_diagnostic_artifact_specs,
    stage_1_pipeline_artifact_specs,
    stage_1_script_outputs,
)
from .context import DatasetBuildContext
from .contracts import DatasetBuildOutputContractBuilder
from .diagnostics import (
    ARTIFACT_SCHEMA_VERSION,
    DatasetInventoryWriter,
    SourceDatasetSchemaSummaryWriter,
    TargetDatabaseSchemaSummaryWriter,
    write_stage_1_diagnostics,
)
from .specs import (
    DatasetBuildStepSpec,
    STAGE_1_BUILD_DATASETS,
    STAGE_1_BUILD_STEP_SPECS,
    stage_1_step_specs,
)
from .staging import PipelineArtifactStager

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DatasetArtifactSpec",
    "DatasetBuildContext",
    "DatasetBuildOutputContractBuilder",
    "DatasetBuildStepSpec",
    "DatasetInventoryWriter",
    "PipelineArtifactStager",
    "STAGE_1_ARTIFACT_SPECS",
    "STAGE_1_BUILD_DATASETS",
    "STAGE_1_BUILD_STEP_SPECS",
    "SourceDatasetSchemaSummaryWriter",
    "TargetDatabaseSchemaSummaryWriter",
    "stage_1_artifact_specs",
    "stage_1_contract_artifact_specs",
    "stage_1_diagnostic_artifact_specs",
    "stage_1_pipeline_artifact_specs",
    "stage_1_script_outputs",
    "stage_1_step_specs",
    "write_stage_1_diagnostics",
]
