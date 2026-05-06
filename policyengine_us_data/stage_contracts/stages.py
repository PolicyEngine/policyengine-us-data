"""Canonical stage and substage identifiers for US data contracts."""

from __future__ import annotations

from types import MappingProxyType

STAGE_1_BUILD_DATASETS = "1_build_datasets"
STAGE_2_BUILD_CALIBRATION_PACKAGE = "2_build_calibration_package"
STAGE_3_FIT_WEIGHTS = "3_fit_weights"
STAGE_4_BUILD_OUTPUTS = "4_build_outputs"
STAGE_5_VALIDATE_AND_PROMOTE_RELEASE = "5_validate_and_promote_release"

CANONICAL_STAGE_IDS = (
    STAGE_1_BUILD_DATASETS,
    STAGE_2_BUILD_CALIBRATION_PACKAGE,
    STAGE_3_FIT_WEIGHTS,
    STAGE_4_BUILD_OUTPUTS,
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
)

CONTRACT_TYPE_BY_STAGE_ID = MappingProxyType(
    {
        STAGE_1_BUILD_DATASETS: "dataset_build_output",
        STAGE_2_BUILD_CALIBRATION_PACKAGE: "calibration_package",
        STAGE_3_FIT_WEIGHTS: "fitted_weights",
        STAGE_4_BUILD_OUTPUTS: "output_build",
        STAGE_5_VALIDATE_AND_PROMOTE_RELEASE: "release_promotion",
    }
)

SUBSTAGE_IDS_BY_STAGE_ID = MappingProxyType(
    {
        STAGE_1_BUILD_DATASETS: (
            "1a_raw_data_download",
            "1b_base_dataset_construction",
            "1f_source_imputation",
            "1g_stage_base_datasets",
        ),
        STAGE_2_BUILD_CALIBRATION_PACKAGE: (
            "2a_build_target_matrix",
            "2b_package_calibration_inputs",
        ),
        STAGE_3_FIT_WEIGHTS: (
            "3a_weight_fitting_regional",
            "3b_weight_fitting_national",
        ),
        STAGE_4_BUILD_OUTPUTS: (
            "4a_local_area_h5_regional",
            "4b_local_area_h5_national",
            "4d_upload_diagnostics",
        ),
        STAGE_5_VALIDATE_AND_PROMOTE_RELEASE: (
            "5a_validate_outputs",
            "5b_promote_huggingface",
            "5c_promote_gcs",
            "5d_write_version_manifest",
        ),
    }
)


def contract_type_for_stage(stage_id: str) -> str:
    """Return the canonical contract type for a canonical stage."""

    return CONTRACT_TYPE_BY_STAGE_ID[stage_id]


def substage_ids_for_stage(stage_id: str) -> tuple[str, ...]:
    """Return the canonical substage IDs for a canonical stage."""

    return SUBSTAGE_IDS_BY_STAGE_ID[stage_id]


def is_canonical_stage_id(stage_id: str) -> bool:
    """Return whether a stage ID is one of the canonical stage IDs."""

    return stage_id in CONTRACT_TYPE_BY_STAGE_ID


def is_canonical_substage_id(stage_id: str, substage_id: str) -> bool:
    """Return whether a substage ID belongs to a canonical stage."""

    return substage_id in SUBSTAGE_IDS_BY_STAGE_ID.get(stage_id, ())
