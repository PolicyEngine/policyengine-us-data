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
from .checkpoints import (
    CheckpointDecision,
    CheckpointReuseSummary,
    CheckpointStore,
)
from .commands import CommandRunner, DatasetCommand, DatasetCommandError
from .context import DatasetBuildContext
from .contracts import DatasetBuildOutputContractBuilder
from .coordinator import (
    CommandBackedSubstepRunner,
    Stage1Coordinator,
    Stage1SubstepRunner,
    Stage1ValidationAdapter,
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
from .rerun import (
    Stage1IdentityMaterial,
    Stage1RerunPlanner,
    Stage1ReuseDecision,
)
from .staging import PipelineArtifactStager
from .status import Stage1ErrorRecord, Stage1StatusEvent
from .validation import (
    Stage1ValidationContext,
    Stage1ValidationError,
    Stage1ValidationRunner,
    Stage1Validator,
    Stage1ValidatorSpec,
    iter_stage_1_validators,
    run_stage_1_validators,
    validators_for_substage,
)
from .validation_results import Stage1ValidationResultWriter, Stage1ValidationSummary
from .validation_targets import ValidationTarget, ValidationTargetCatalog

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CheckpointDecision",
    "CheckpointReuseSummary",
    "CheckpointStore",
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
    "Stage1IdentityMaterial",
    "Stage1RerunPlanner",
    "Stage1ReuseDecision",
    "Stage1StatusEvent",
    "Stage1SubstepRunner",
    "Stage1ValidationAdapter",
    "Stage1ValidationContext",
    "Stage1ValidationError",
    "Stage1ValidationResultWriter",
    "Stage1ValidationRunner",
    "Stage1ValidationSummary",
    "Stage1Validator",
    "Stage1ValidatorSpec",
    "TargetDatabaseSchemaSummaryWriter",
    "ValidationTarget",
    "ValidationTargetCatalog",
    "iter_stage_1_validators",
    "run_stage_1_validators",
    "stage_1_artifact_specs",
    "stage_1_contract_artifact_specs",
    "stage_1_diagnostic_artifact_specs",
    "stage_1_pipeline_artifact_specs",
    "stage_1_script_outputs",
    "stage_1_substep_id_for_script",
    "stage_1_substep_title",
    "stage_1_step_specs",
    "validators_for_substage",
    "write_stage_1_diagnostics",
]
