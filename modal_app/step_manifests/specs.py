"""Canonical pipeline step and sub-step specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class PipelineSubstepSpec:
    """A named operation nested under a top-level pipeline step."""

    id: str
    title: str
    parent_id: str


@dataclass(frozen=True)
class PipelineStepSpec:
    """A top-level publication pipeline step."""

    id: str
    title: str
    substeps: tuple[PipelineSubstepSpec, ...] = ()


PipelineStepRef: TypeAlias = PipelineStepSpec | PipelineSubstepSpec | str


def _substep(id: str, title: str, parent_id: str) -> PipelineSubstepSpec:
    return PipelineSubstepSpec(id=id, title=title, parent_id=parent_id)


BUILD_DATASETS = PipelineStepSpec(
    id="1_build_datasets",
    title="Build datasets",
    substeps=(
        _substep(
            "1a_raw_data_download",
            "Raw data download",
            "1_build_datasets",
        ),
        _substep(
            "1b_base_dataset_construction",
            "Base dataset construction",
            "1_build_datasets",
        ),
        _substep(
            "1c_extended_cps_puf_clone",
            "Extended CPS PUF clone",
            "1_build_datasets",
        ),
        _substep(
            "1d_enhanced_cps_reweighting",
            "Enhanced CPS reweighting",
            "1_build_datasets",
        ),
        _substep(
            "1e_stratified_cps",
            "Stratified CPS",
            "1_build_datasets",
        ),
        _substep(
            "1f_source_imputation",
            "Source imputation",
            "1_build_datasets",
        ),
        _substep(
            "1g_stage_base_datasets",
            "Stage base datasets",
            "1_build_datasets",
        ),
    ),
)
RAW_DATA_DOWNLOAD = BUILD_DATASETS.substeps[0]
BASE_DATASET_CONSTRUCTION = BUILD_DATASETS.substeps[1]
EXTENDED_CPS_PUF_CLONE = BUILD_DATASETS.substeps[2]
ENHANCED_CPS_REWEIGHTING = BUILD_DATASETS.substeps[3]
STRATIFIED_CPS = BUILD_DATASETS.substeps[4]
SOURCE_IMPUTATION = BUILD_DATASETS.substeps[5]
STAGE_BASE_DATASETS = BUILD_DATASETS.substeps[6]

BUILD_CALIBRATION_PACKAGE = PipelineStepSpec(
    id="2_build_calibration_package",
    title="Build calibration package",
    substeps=(
        _substep(
            "2a_matrix_build_calibration_target_construction",
            "Matrix build and calibration target construction",
            "2_build_calibration_package",
        ),
    ),
)
MATRIX_BUILD_CALIBRATION_TARGET_CONSTRUCTION = BUILD_CALIBRATION_PACKAGE.substeps[0]

FIT_WEIGHTS = PipelineStepSpec(
    id="3_fit_weights",
    title="Fit weights",
    substeps=(
        _substep(
            "3a_weight_fitting_regional",
            "Weight fitting regional",
            "3_fit_weights",
        ),
        _substep(
            "3b_weight_fitting_national",
            "Weight fitting national",
            "3_fit_weights",
        ),
    ),
)
WEIGHT_FITTING_REGIONAL = FIT_WEIGHTS.substeps[0]
WEIGHT_FITTING_NATIONAL = FIT_WEIGHTS.substeps[1]

BUILD_OUTPUTS = PipelineStepSpec(
    id="4_build_outputs",
    title="Build outputs",
    substeps=(
        _substep(
            "4a_local_area_h5_regional",
            "Local area H5 regional",
            "4_build_outputs",
        ),
        _substep(
            "4b_local_area_h5_national",
            "Local area H5 national",
            "4_build_outputs",
        ),
        _substep(
            "4d_upload_diagnostics",
            "Upload diagnostics",
            "4_build_outputs",
        ),
    ),
)
LOCAL_AREA_H5_REGIONAL = BUILD_OUTPUTS.substeps[0]
LOCAL_AREA_H5_NATIONAL = BUILD_OUTPUTS.substeps[1]
UPLOAD_DIAGNOSTICS = BUILD_OUTPUTS.substeps[2]

VALIDATE_AND_PROMOTE_RELEASE = PipelineStepSpec(
    id="5_validate_and_promote_release",
    title="Validate and promote release",
    substeps=(
        _substep(
            "5a_validate_outputs",
            "Validate outputs",
            "5_validate_and_promote_release",
        ),
        _substep(
            "5b_promote_huggingface",
            "Promote Hugging Face",
            "5_validate_and_promote_release",
        ),
        _substep(
            "5c_promote_gcs",
            "Promote GCS",
            "5_validate_and_promote_release",
        ),
        _substep(
            "5d_write_version_manifest",
            "Write version manifest",
            "5_validate_and_promote_release",
        ),
    ),
)
VALIDATE_OUTPUTS = VALIDATE_AND_PROMOTE_RELEASE.substeps[0]
PROMOTE_HUGGINGFACE = VALIDATE_AND_PROMOTE_RELEASE.substeps[1]
PROMOTE_GCS = VALIDATE_AND_PROMOTE_RELEASE.substeps[2]
WRITE_VERSION_MANIFEST = VALIDATE_AND_PROMOTE_RELEASE.substeps[3]

PIPELINE_STEPS = (
    BUILD_DATASETS,
    BUILD_CALIBRATION_PACKAGE,
    FIT_WEIGHTS,
    BUILD_OUTPUTS,
    VALIDATE_AND_PROMOTE_RELEASE,
)

PIPELINE_SUBSTEPS = tuple(
    substep for step in PIPELINE_STEPS for substep in step.substeps
)

PIPELINE_STEP_IDS = tuple(step.id for step in PIPELINE_STEPS)
PIPELINE_SUBSTEP_IDS = tuple(substep.id for substep in PIPELINE_SUBSTEPS)

# These are the step manifest records the current pipeline actually emits.
# PIPELINE_STEPS and PIPELINE_SUBSTEPS define the intended taxonomy; not every
# declared sub-step has a separate runtime manifest yet.
RUN_MANIFEST_STEP_IDS = tuple(
    step.id
    for step in (
        BUILD_DATASETS,
        STAGE_BASE_DATASETS,
        BUILD_CALIBRATION_PACKAGE,
        WEIGHT_FITTING_REGIONAL,
        WEIGHT_FITTING_NATIONAL,
        LOCAL_AREA_H5_REGIONAL,
        LOCAL_AREA_H5_NATIONAL,
        UPLOAD_DIAGNOSTICS,
        VALIDATE_AND_PROMOTE_RELEASE,
    )
)

_STEP_BY_ID = {step.id: step for step in PIPELINE_STEPS}
_SUBSTEP_BY_ID = {substep.id: substep for substep in PIPELINE_SUBSTEPS}


def step_id(step: PipelineStepRef) -> str:
    """Return the canonical ID for a step spec, sub-step spec, or raw ID."""
    if isinstance(step, str):
        return step
    return step.id


def parent_step_id(step: PipelineStepRef) -> str | None:
    """Return the parent step ID for a sub-step, if known."""
    if isinstance(step, PipelineSubstepSpec):
        return step.parent_id
    if isinstance(step, PipelineStepSpec):
        return None
    substep = _SUBSTEP_BY_ID.get(step)
    return substep.parent_id if substep is not None else None


def step_title(step: PipelineStepRef) -> str:
    """Return a human-readable title for a step or sub-step."""
    if isinstance(step, str):
        spec = _STEP_BY_ID.get(step) or _SUBSTEP_BY_ID.get(step)
        return spec.title if spec is not None else step
    return step.title
