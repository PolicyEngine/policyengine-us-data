from pathlib import Path

from policyengine_us_data.storage import STORAGE_FOLDER


RAW_CPS_BASE_YEAR = 2024
PRODUCTION_ECPS_YEAR = 2025


def extended_cps_path(year: int) -> Path:
    return STORAGE_FOLDER / f"extended_cps_{year}.h5"


def extended_cps_half_path(year: int) -> Path:
    return STORAGE_FOLDER / f"extended_cps_{year}_half.h5"


def stratified_extended_cps_path(year: int) -> Path:
    return STORAGE_FOLDER / f"stratified_extended_cps_{year}.h5"


def source_imputed_stratified_extended_cps_path(year: int) -> Path:
    return STORAGE_FOLDER / f"source_imputed_stratified_extended_cps_{year}.h5"


def enhanced_cps_path(year: int) -> Path:
    return STORAGE_FOLDER / f"enhanced_cps_{year}.h5"


def small_enhanced_cps_path(year: int) -> Path:
    return STORAGE_FOLDER / f"small_enhanced_cps_{year}.h5"


def sparse_enhanced_cps_path(year: int) -> Path:
    return STORAGE_FOLDER / f"sparse_enhanced_cps_{year}.h5"
