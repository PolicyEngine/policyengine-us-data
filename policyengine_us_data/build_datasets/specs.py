"""Substage specifications for the Stage 1 dataset build."""

from __future__ import annotations

from dataclasses import dataclass

from policyengine_us_data.pipeline_metadata import pipeline_node


STAGE_1_BUILD_DATASETS = "1_build_datasets"


@dataclass(frozen=True)
class DatasetBuildStepSpec:
    """A canonical substage in the Stage 1 dataset-build workflow."""

    id: str
    title: str
    parent_id: str = STAGE_1_BUILD_DATASETS
    legacy_stage_id: str | None = None
    manifest_step_ids: tuple[str, ...] = ("01_build_datasets",)
    reuse_mode: str = "checkpointable"
    skip_when_enhanced_cps_skipped: bool = False
    skip_when_stage_5_skipped: bool = False


STAGE_1_BUILD_STEP_SPECS: tuple[DatasetBuildStepSpec, ...] = (
    DatasetBuildStepSpec(
        id="1a_raw_data_download",
        title="Raw data download",
        legacy_stage_id="0",
        reuse_mode="observed_only",
    ),
    DatasetBuildStepSpec(
        id="1b_base_dataset_construction",
        title="Base dataset construction",
        legacy_stage_id="1",
    ),
    DatasetBuildStepSpec(
        id="1c_extended_cps_puf_clone",
        title="Extended CPS PUF clone",
        legacy_stage_id="2",
    ),
    DatasetBuildStepSpec(
        id="1d_enhanced_cps_reweighting",
        title="Enhanced CPS reweighting",
        legacy_stage_id="3a",
        skip_when_enhanced_cps_skipped=True,
    ),
    DatasetBuildStepSpec(
        id="1e_stratified_cps",
        title="Stratified CPS",
        legacy_stage_id="3b",
        reuse_mode="handoff",
    ),
    DatasetBuildStepSpec(
        id="1f_source_imputation",
        title="Source imputation",
        legacy_stage_id="4",
        reuse_mode="handoff",
        skip_when_stage_5_skipped=True,
    ),
    DatasetBuildStepSpec(
        id="1g_stage_base_datasets",
        title="Stage base datasets",
        legacy_stage_id="7",
        manifest_step_ids=("04_stage_base_datasets",),
        reuse_mode="handoff",
    ),
)


@pipeline_node(
    id="stage_1_dataset_build_specs",
    label="Stage 1 Dataset Build Specs",
    node_type="library",
    description=(
        "Canonical substage taxonomy for Stage 1 dataset-build contracts, "
        "step manifests, and pipeline documentation."
    ),
    source_file="policyengine_us_data/build_datasets/specs.py",
    status="current",
    stability="stable",
    pathways=["data_build", "stage_contracts", "pipeline_docs"],
    validation_commands=["uv run pytest tests/unit/test_build_dataset_specs.py"],
)
def stage_1_step_specs() -> tuple[DatasetBuildStepSpec, ...]:
    """Return the canonical Stage 1 dataset-build substage specs."""

    return STAGE_1_BUILD_STEP_SPECS


__all__ = [
    "DatasetBuildStepSpec",
    "STAGE_1_BUILD_DATASETS",
    "STAGE_1_BUILD_STEP_SPECS",
    "stage_1_step_specs",
]
