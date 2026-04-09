from policyengine_us_data.datasets import EnhancedCPS_2025
from policyengine_us_data.datasets.cps.extended_cps import (
    ExtendedCPS_2025,
    ExtendedCPS_2025_Half,
)
from policyengine_us_data.storage.artifact_paths import (
    PRODUCTION_ECPS_YEAR,
    enhanced_cps_path,
    extended_cps_half_path,
    extended_cps_path,
    small_enhanced_cps_path,
    source_imputed_stratified_extended_cps_path,
    sparse_enhanced_cps_path,
    stratified_extended_cps_path,
)


def test_2025_artifact_paths_are_production_defaults():
    assert PRODUCTION_ECPS_YEAR == 2025
    assert extended_cps_path(PRODUCTION_ECPS_YEAR).name == "extended_cps_2025.h5"
    assert (
        extended_cps_half_path(PRODUCTION_ECPS_YEAR).name == "extended_cps_2025_half.h5"
    )
    assert (
        stratified_extended_cps_path(PRODUCTION_ECPS_YEAR).name
        == "stratified_extended_cps_2025.h5"
    )
    assert (
        source_imputed_stratified_extended_cps_path(PRODUCTION_ECPS_YEAR).name
        == "source_imputed_stratified_extended_cps_2025.h5"
    )
    assert enhanced_cps_path(PRODUCTION_ECPS_YEAR).name == "enhanced_cps_2025.h5"
    assert (
        small_enhanced_cps_path(PRODUCTION_ECPS_YEAR).name
        == "small_enhanced_cps_2025.h5"
    )
    assert (
        sparse_enhanced_cps_path(PRODUCTION_ECPS_YEAR).name
        == "sparse_enhanced_cps_2025.h5"
    )


def test_extended_cps_2025_metadata_uses_2025_time_period():
    assert ExtendedCPS_2025.time_period == 2025
    assert ExtendedCPS_2025.file_path == extended_cps_path(2025)
    assert ExtendedCPS_2025_Half.time_period == 2025
    assert ExtendedCPS_2025_Half.file_path == extended_cps_half_path(2025)


def test_enhanced_cps_2025_metadata_disables_legacy_aca_override():
    assert EnhancedCPS_2025.time_period == 2025
    assert EnhancedCPS_2025.start_year == 2025
    assert EnhancedCPS_2025.end_year == 2025
    assert EnhancedCPS_2025.file_path == enhanced_cps_path(2025)
    assert EnhancedCPS_2025.input_dataset is ExtendedCPS_2025_Half
    assert EnhancedCPS_2025.apply_aca_2025_takeup_override is False
