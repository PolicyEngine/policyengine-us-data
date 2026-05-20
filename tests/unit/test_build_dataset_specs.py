from modal_app.step_manifests.specs import BUILD_DATASETS
from policyengine_us_data.build_datasets import (
    STAGE_1_ARTIFACT_SPECS,
    STAGE_1_BUILD_DATASETS,
    STAGE_1_BUILD_STEP_SPECS,
    stage_1_contract_artifact_specs,
    stage_1_script_outputs,
)
from policyengine_us_data.stage_contracts import (
    STAGE_1_BUILD_DATASETS as CONTRACT_STAGE_1_BUILD_DATASETS,
    substage_ids_for_stage,
)


def _script_output_paths() -> set[str]:
    paths: set[str] = set()
    for output in stage_1_script_outputs().values():
        if isinstance(output, str):
            paths.add(output)
        else:
            paths.update(output)
    return paths


def test_stage_1_step_specs_match_stage_contract_taxonomy():
    assert STAGE_1_BUILD_DATASETS == CONTRACT_STAGE_1_BUILD_DATASETS
    assert tuple(
        spec.id for spec in STAGE_1_BUILD_STEP_SPECS
    ) == substage_ids_for_stage(STAGE_1_BUILD_DATASETS)
    assert STAGE_1_BUILD_STEP_SPECS[-1].id == "1g_stage_base_datasets"
    assert STAGE_1_BUILD_STEP_SPECS[-1].manifest_step_ids == ("04_stage_base_datasets",)


def test_stage_1_artifacts_have_known_substages():
    substage_ids = {spec.id for spec in STAGE_1_BUILD_STEP_SPECS}

    assert {spec.substage_id for spec in STAGE_1_ARTIFACT_SPECS} <= substage_ids


def test_stage_1_script_outputs_are_generated_from_artifact_specs():
    assert stage_1_script_outputs() == {
        "policyengine_us_data/utils/uprating.py": (
            "policyengine_us_data/storage/uprating_factors.csv"
        ),
        "policyengine_us_data/datasets/acs/acs.py": (
            "policyengine_us_data/storage/acs_2022.h5"
        ),
        "policyengine_us_data/datasets/puf/irs_puf.py": (
            "policyengine_us_data/storage/irs_puf_2015.h5"
        ),
        "policyengine_us_data/datasets/cps/cps.py": (
            "policyengine_us_data/storage/cps_2024.h5"
        ),
        "policyengine_us_data/datasets/puf/puf.py": (
            "policyengine_us_data/storage/puf_2024.h5"
        ),
        "policyengine_us_data/datasets/cps/extended_cps.py": (
            "policyengine_us_data/storage/extended_cps_2024.h5"
        ),
        "policyengine_us_data/datasets/cps/enhanced_cps.py": [
            "policyengine_us_data/storage/enhanced_cps_2024.h5",
            "policyengine_us_data/storage/enhanced_cps_2024.clone_diagnostics.json",
            "calibration_log.csv",
        ],
        "policyengine_us_data/calibration/create_stratified_cps.py": (
            "policyengine_us_data/storage/stratified_extended_cps_2024.h5"
        ),
        "policyengine_us_data/calibration/create_source_imputed_cps.py": (
            "policyengine_us_data/storage/"
            "source_imputed_stratified_extended_cps_2024.h5"
        ),
        "policyengine_us_data/datasets/cps/small_enhanced_cps.py": (
            "policyengine_us_data/storage/small_enhanced_cps_2024.h5"
        ),
    }


def test_stage_1_script_output_paths_have_artifact_specs():
    spec_paths = {
        spec.storage_path
        for spec in STAGE_1_ARTIFACT_SPECS
        if spec.storage_path is not None
    }

    assert _script_output_paths() <= spec_paths


def test_stage_1_contract_outputs_are_explicit_subset():
    contract_specs = stage_1_contract_artifact_specs()
    contract_names = {spec.logical_name for spec in contract_specs}

    assert contract_specs == tuple(
        spec for spec in STAGE_1_ARTIFACT_SPECS if spec.contract_output
    )
    assert "uprating_factors" not in contract_names
    assert "enhanced_cps_2024_clone_diagnostics" not in contract_names
    assert "enhanced_cps_calibration_log" not in contract_names
    assert "source_imputed_stratified_extended_cps" in contract_names
    assert all(spec.contract_output for spec in contract_specs)


def test_step_manifest_stage_1_substeps_match_dataset_build_specs():
    assert tuple(substep.id for substep in BUILD_DATASETS.substeps) == tuple(
        spec.id for spec in STAGE_1_BUILD_STEP_SPECS
    )
    assert tuple(substep.title for substep in BUILD_DATASETS.substeps) == tuple(
        spec.title for spec in STAGE_1_BUILD_STEP_SPECS
    )
    assert {substep.parent_id for substep in BUILD_DATASETS.substeps} == {
        STAGE_1_BUILD_DATASETS
    }


def test_stage_1_skip_flags_identify_expected_artifacts():
    enhanced_cps_skipped = {
        spec.filename
        for spec in STAGE_1_ARTIFACT_SPECS
        if spec.skip_when_enhanced_cps_skipped
    }
    stage_5_skipped = {
        spec.filename
        for spec in STAGE_1_ARTIFACT_SPECS
        if spec.skip_when_stage_5_skipped
    }

    assert {
        "enhanced_cps_2024.h5",
        "enhanced_cps_2024.clone_diagnostics.json",
        "calibration_log.csv",
        "small_enhanced_cps_2024.h5",
    } <= enhanced_cps_skipped
    assert {
        "small_enhanced_cps_2024.h5",
        "source_imputed_stratified_extended_cps_2024.h5",
        "source_imputed_stratified_extended_cps.h5",
    } == stage_5_skipped
