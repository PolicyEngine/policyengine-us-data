import pandas as pd
import numpy as np
from policyengine_us_data.storage import CALIBRATION_FOLDER
from policyengine_us_data.utils.census_spm import (
    get_census_spm_capped_housing_subsidy_total,
)

"""
Hardcoded targets for the year 2024 from CPS-derived statistics and other
sources. Housing uses the Census SPM capped subsidy concept, not HUD spending.
"""


def get_hard_coded_totals(
    year: int = 2024,
    storage_folder=None,
) -> dict[str, float]:
    return {
        "health_insurance_premiums_without_medicare_part_b": 385e9,
        "other_medical_expenses": 278e9,
        "medicare_part_b_premiums": 112e9,
        "over_the_counter_health_expenses": 72e9,
        "spm_unit_spm_threshold": 3_945e9,
        "child_support_expense": 33e9,
        "child_support_received": 33e9,
        "spm_unit_capped_work_childcare_expenses": 348e9,
        "spm_unit_capped_housing_subsidy": get_census_spm_capped_housing_subsidy_total(
            year, storage_folder=storage_folder
        ),
        "tanf": 9e9,
        # Alimony could be targeted via SOI
        "alimony_income": 13e9,
        "alimony_expense": 13e9,
        # Rough estimate, not CPS derived
        "real_estate_taxes": 500e9,  # Rough estimate between 350bn and 600bn total property tax collections
        "rent": 735e9,  # ACS total uprated by CPI
        # Table 5A from https://www.irs.gov/statistics/soi-tax-stats-individual-information-return-form-w2-statistics
        # shows $38,316,190,000 in Box 7: Social security tips (2018)
        # Wages and salaries grew 32% from 2018 to 2023: https://fred.stlouisfed.org/graph/?g=1J0CC
        # Assume 40% through 2024
        "tip_income": 38e9 * 1.4,
    }


def pull_hardcoded_targets(year: int = 2024, storage_folder=None):
    """
    Returns a DataFrame with hardcoded targets for various CPS-derived statistics and other sources.
    """
    hard_coded_totals = get_hard_coded_totals(
        year=year,
        storage_folder=storage_folder,
    )
    data = {
        "DATA_SOURCE": ["hardcoded"] * len(hard_coded_totals),
        "GEO_ID": ["0000000US"] * len(hard_coded_totals),
        "GEO_NAME": ["national"] * len(hard_coded_totals),
        "VARIABLE": list(hard_coded_totals.keys()),
        "VALUE": list(hard_coded_totals.values()),
        "IS_COUNT": [0.0]
        * len(hard_coded_totals),  # All values are monetary amounts, not counts
        "BREAKDOWN_VARIABLE": [np.nan]
        * len(hard_coded_totals),  # No breakdown variable for hardcoded targets
        "LOWER_BOUND": [np.nan] * len(hard_coded_totals),
        "UPPER_BOUND": [np.nan] * len(hard_coded_totals),
    }

    df = pd.DataFrame(data)
    return df[
        [
            "DATA_SOURCE",
            "GEO_ID",
            "GEO_NAME",
            "VARIABLE",
            "VALUE",
            "IS_COUNT",
            "BREAKDOWN_VARIABLE",
            "LOWER_BOUND",
            "UPPER_BOUND",
        ]
    ]


def main() -> None:
    out_dir = CALIBRATION_FOLDER
    df = pull_hardcoded_targets()
    df.to_csv(out_dir / "national_hardcoded_targets.csv", index=False)


if __name__ == "__main__":
    main()
