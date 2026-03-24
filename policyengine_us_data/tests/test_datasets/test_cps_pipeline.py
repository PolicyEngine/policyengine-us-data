from pathlib import Path


def test_source_imputed_dataset_matches_calibration_artifact_paths():
    from policyengine_us_data.calibration.create_source_imputed_cps import (
        INPUT_PATH,
        OUTPUT_PATH,
    )
    from policyengine_us_data.datasets.cps import (
        SourceImputedStratifiedExtendedCPS_2024,
        StratifiedExtendedCPS_2024,
    )

    assert StratifiedExtendedCPS_2024.file_path == Path(INPUT_PATH)
    assert SourceImputedStratifiedExtendedCPS_2024.file_path == Path(OUTPUT_PATH)
    assert (
        SourceImputedStratifiedExtendedCPS_2024.input_dataset
        is StratifiedExtendedCPS_2024
    )


def test_source_imputed_cps_uses_base_cps_input():
    from policyengine_us_data.datasets.cps import CPS_2024, SourceImputedCPS_2024

    assert SourceImputedCPS_2024.input_dataset is CPS_2024


def test_enhanced_cps_uses_source_imputed_stratified_input():
    from policyengine_us_data.datasets.cps import (
        EnhancedCPS_2024,
        SourceImputedStratifiedExtendedCPS_2024,
    )

    assert EnhancedCPS_2024.input_dataset is SourceImputedStratifiedExtendedCPS_2024
