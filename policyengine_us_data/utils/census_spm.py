from pathlib import Path

import pandas as pd

from policyengine_us_data.storage import STORAGE_FOLDER

# These totals are derived from raw public-use Census CPS ASEC files by
# summing SPM_CAPHOUSESUB * SPM_WEIGHT / 100 at the SPM-unit level.
# They represent the Census SPM capped housing subsidy concept, not HUD
# spending or outlays.
CENSUS_SPM_CAPPED_HOUSING_SUBSIDY_TOTALS = {
    2022: 29_549_204_420.92,
    2023: 31_844_144_470.85,
    2024: 33_649_114_150.37,
}


def get_census_spm_capped_housing_subsidy_total(
    year: int,
    storage_folder: Path | str | None = None,
) -> float:
    """
    Return the Census CPS ASEC total for SPM_CAPHOUSESUB.

    If a storage folder is provided, recompute directly from the local
    raw Census CPS HDF file. Otherwise, use the checked-in year-specific
    totals above so callers do not depend on local data files at import
    time.
    """

    if storage_folder is None:
        if year not in CENSUS_SPM_CAPPED_HOUSING_SUBSIDY_TOTALS:
            raise ValueError(
                f"No published Census SPM capped housing subsidy total for {year}."
            )
        return CENSUS_SPM_CAPPED_HOUSING_SUBSIDY_TOTALS[year]

    storage_path = Path(storage_folder) / f"census_cps_{year}.h5"
    if not storage_path.exists():
        raise FileNotFoundError(
            f"Missing raw Census CPS file for {year}: {storage_path}"
        )

    with pd.HDFStore(storage_path, mode="r") as store:
        spm_unit = store["spm_unit"][["SPM_CAPHOUSESUB", "SPM_WEIGHT"]]

    return float((spm_unit.SPM_CAPHOUSESUB * spm_unit.SPM_WEIGHT).sum() / 100)


def build_census_spm_capped_housing_subsidy_target(
    year: int,
    storage_folder: Path | str | None = None,
) -> dict:
    return {
        "variable": "spm_unit_capped_housing_subsidy",
        "value": get_census_spm_capped_housing_subsidy_total(
            year, storage_folder=storage_folder
        ),
        "source": "Census CPS ASEC public-use SPM_CAPHOUSESUB",
        "notes": (
            "Capped SPM housing subsidy total from raw Census CPS ASEC "
            "SPM_CAPHOUSESUB; not HUD spending or outlays."
        ),
        "year": year,
    }
