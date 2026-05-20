"""Canonical Stage 1 dataset-build specifications."""

from importlib import import_module
from typing import Any

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
    Stage1StatusSink,
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
from .status_store import (
    Stage1StatusRecorder,
    Stage1StatusReadError,
    Stage1StatusSnapshot,
    Stage1StoredStatusEvent,
    empty_stage_1_status_snapshot,
    read_stage_1_status_snapshot,
)

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
    "Stage1ReuseManifest",
    "Stage1ReuseManifestRecord",
    "Stage1StatusRecorder",
    "Stage1StatusReadError",
    "Stage1StatusEvent",
    "Stage1StatusSink",
    "Stage1StatusSnapshot",
    "Stage1StoredStatusEvent",
    "Stage1SubstepRunner",
    "SubprocessLogCapture",
    "TargetDatabaseSchemaSummaryWriter",
    "empty_stage_1_status_snapshot",
    "read_stage_1_status_snapshot",
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

_LAZY_EXPORTS = {
    "CheckpointDecision": (".checkpoints", "CheckpointDecision"),
    "CheckpointReuseSummary": (".checkpoints", "CheckpointReuseSummary"),
    "CheckpointStore": (".checkpoints", "CheckpointStore"),
    "Stage1IdentityMaterial": (".rerun", "Stage1IdentityMaterial"),
    "Stage1RerunPlanner": (".rerun", "Stage1RerunPlanner"),
    "Stage1ReuseDecision": (".rerun", "Stage1ReuseDecision"),
    "Stage1ReuseManifest": (".rerun", "Stage1ReuseManifest"),
    "Stage1ReuseManifestRecord": (".rerun", "Stage1ReuseManifestRecord"),
}


def __getattr__(name: str) -> Any:
    """Load checkpoint and rerun exports without package-import cycles."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
