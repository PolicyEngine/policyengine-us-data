import pandas as pd
import numpy as np
from policyengine_us_data.storage import CALIBRATION_FOLDER

"""
Hardcoded targets for the year 2024 from CPS-derived statistics and other sources. Include medical expenses, sum of SPM thresholds, and child support expenses.
"""

HARD_CODED_TOTALS = {
    "health_insurance_premiums_without_medicare_part_b": 385e9,
    "other_medical_expenses": 278e9,
    "medicare_part_b_premiums": 112e9,
    "over_the_counter_health_expenses": 72e9,
    "spm_unit_spm_threshold": 3_945e9,
    "child_support_expense": 33e9,
    "child_support_received": 33e9,
    "spm_unit_capped_work_childcare_expenses": 348e9,
    "spm_unit_capped_housing_subsidy": 35e9,
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


def pull_hardcoded_targets():
    """
    Returns a DataFrame with hardcoded targets for various CPS-derived statistics and other sources.
    """
    data = {
        "DATA_SOURCE": ["hardcoded"] * len(HARD_CODED_TOTALS),
        "GEO_ID": ["0000000US"] * len(HARD_CODED_TOTALS),
        "GEO_NAME": ["US"] * len(HARD_CODED_TOTALS),
        "VARIABLE": list(HARD_CODED_TOTALS.keys()),
        "VALUE": list(HARD_CODED_TOTALS.values()),
        "IS_COUNT": [0.0]
        * len(
            HARD_CODED_TOTALS
        ),  # All values are monetary amounts, not counts
        "BREAKDOWN_VARIABLE": [np.nan]
        * len(
            HARD_CODED_TOTALS
        ),  # No breakdown variable for hardcoded targets
        "LOWER_BOUND": [np.nan] * len(HARD_CODED_TOTALS),
        "UPPER_BOUND": [np.nan] * len(HARD_CODED_TOTALS),
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
