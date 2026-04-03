"""Shared metadata for IRS SOI-backed calibration targets."""

LATEST_PUBLISHED_NATIONAL_SOI_YEAR = 2023
LATEST_PUBLISHED_GEOGRAPHIC_SOI_YEAR = 2022
LATEST_PUBLISHED_IRA_ACCUMULATION_YEAR = 2022

RETIREMENT_CONTRIBUTION_TARGETS = {
    "traditional_ira_contributions": {
        "value": 13.771289e9,
        "source": "https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income",
        "notes": (
            "SOI 1304 Table 1.4 (TY 2023) 'IRA payments' deduction, "
            "col DU, row 'All returns, total'"
        ),
        "source_year": 2023,
    },
    "self_employed_pension_contribution_ald": {
        "value": 30.130848e9,
        "source": "https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income",
        "notes": (
            "SOI 1304 Table 1.4 (TY 2023) 'Payments to a Keogh plan', "
            "col DM, row 'All returns, total'"
        ),
        "source_year": 2023,
    },
    "roth_ira_contributions": {
        "value": 34.951077e9,
        "source": "https://www.irs.gov/statistics/soi-tax-stats-accumulation-and-distribution-of-individual-retirement-arrangements",
        "notes": (
            "IRS SOI IRA Accumulation Table 6 (TY 2022), latest published "
            "Roth IRA contribution total"
        ),
        "source_year": 2022,
    },
}
