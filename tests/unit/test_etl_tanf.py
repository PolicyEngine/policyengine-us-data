import pandas as pd
import pytest

from policyengine_us_data.db.etl_tanf import (
    _validate_supported_year,
    transform_tanf_caseload_data,
    transform_tanf_financial_data,
)


def test_transform_tanf_caseload_data_extracts_national_and_state_rows():
    raw_df = pd.DataFrame(
        {
            "State": [
                "Alabama",
                "California",
                "District of Columbia",
                "U.S. Totals",
                "Footnote",
            ],
            "FY2024": [8_500.0, 290_247.75, 5_056.25, 841_208.666667, None],
        }
    )

    national_families, state_df = transform_tanf_caseload_data(raw_df)

    assert national_families == pytest.approx(841_208.666667)
    assert list(state_df["state_fips"]) == [1, 6, 11]
    assert state_df.loc[state_df["state_fips"] == 6, "recipient_families"].iloc[
        0
    ] == pytest.approx(290_247.75)
    assert state_df.loc[state_df["state_fips"] == 11, "recipient_families"].iloc[
        0
    ] == pytest.approx(5_056.25)


def test_transform_tanf_financial_data_extracts_cash_assistance_totals():
    national_df = pd.DataFrame(
        {
            "Spending Category": [
                "Basic Assistance",
                (
                    "Basic Assistance (excluding Relative Foster Care Maintenance "
                    "Payments and Adoption and Guardianship Subsidies)"
                ),
                "Work Supports",
            ],
            "All Funds": ["8,186,013,422.99", "7,788,317,474.55", "123.45"],
        }
    )
    state_sheets = {
        "California": pd.DataFrame(
            {
                "Spending Category": [
                    "Basic Assistance",
                    (
                        "Basic Assistance (excluding Relative Foster Care Maintenance "
                        "Payments and Adoption and Guardianship Subsidies)"
                    ),
                ],
                "All Funds": ["3,908,497,323.43", "3,742,540,224.36"],
            }
        ),
        "District of Columbia": pd.DataFrame(
            {
                "Spending Category": [
                    "Basic Assistance",
                    (
                        "Basic Assistance (excluding Relative Foster Care Maintenance "
                        "Payments and Adoption and Guardianship Subsidies)"
                    ),
                ],
                "All Funds": ["51,920,224.78", "45,666,113.50"],
            }
        ),
    }

    national_spending, state_df = transform_tanf_financial_data(
        national_df,
        state_sheets,
    )

    assert national_spending == pytest.approx(7_788_317_474.55)
    assert state_df.loc[state_df["state_fips"] == 6, "tanf"].iloc[0] == pytest.approx(
        3_742_540_224.36
    )
    assert state_df.loc[state_df["state_fips"] == 11, "tanf"].iloc[0] == pytest.approx(
        45_666_113.50
    )


def test_validate_supported_year_rejects_non_2024():
    with pytest.raises(ValueError, match="FY2024"):
        _validate_supported_year(2025)
