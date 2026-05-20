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
from .commands import (
    CommandRunner,
    DatasetCommand,
    DatasetCommandError,
    SubprocessLogCapture,
)
from .context import DatasetBuildContext
from .contracts import DatasetBuildOutputContractBuilder
from .coordinator import (
    CommandBackedSubstepRunner,
    Stage1Coordinator,
    Stage1SubstepRunner,
    stage_1_substep_id_for_script,
    stage_1_substep_title,
)
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
from .results import DatasetCommandResult, DatasetSubstepResult
from .staging import PipelineArtifactStager
from .status import Stage1ErrorRecord, Stage1StatusEvent

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CommandBackedSubstepRunner",
    "CommandRunner",
    "DatasetArtifactSpec",
    "DatasetBuildContext",
    "DatasetBuildOutputContractBuilder",
    "DatasetBuildStepSpec",
    "DatasetCommand",
    "DatasetCommandError",
    "DatasetCommandResult",
    "DatasetInventoryWriter",
    "DatasetSubstepResult",
    "PipelineArtifactStager",
    "STAGE_1_ARTIFACT_SPECS",
    "STAGE_1_BUILD_DATASETS",
    "STAGE_1_BUILD_STEP_SPECS",
    "SourceDatasetSchemaSummaryWriter",
    "Stage1Coordinator",
    "Stage1ErrorRecord",
    "Stage1StatusEvent",
    "Stage1SubstepRunner",
    "SubprocessLogCapture",
    "TargetDatabaseSchemaSummaryWriter",
    "stage_1_artifact_specs",
    "stage_1_contract_artifact_specs",
    "stage_1_diagnostic_artifact_specs",
    "stage_1_pipeline_artifact_specs",
    "stage_1_script_outputs",
    "stage_1_substep_id_for_script",
    "stage_1_substep_title",
    "stage_1_step_specs",
    "write_stage_1_diagnostics",
]
