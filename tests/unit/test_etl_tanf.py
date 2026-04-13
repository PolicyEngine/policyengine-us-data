import pandas as pd
import pytest

from policyengine_us_data.db.etl_tanf import (
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


def test_transform_tanf_financial_data_extracts_basic_assistance_totals():
    national_df = pd.DataFrame(
        {
            "Spending Category": ["Basic Assistance", "Work Supports"],
            "All Funds": ["8,186,013,422.99", "123.45"],
        }
    )
    state_sheets = {
        "California": pd.DataFrame(
            {
                "Spending Category": ["Basic Assistance"],
                "All Funds": ["3,908,497,323.43"],
            }
        ),
        "District of Columbia": pd.DataFrame(
            {
                "Spending Category": ["Basic Assistance"],
                "All Funds": ["51,920,224.78"],
            }
        ),
    }

    national_spending, state_df = transform_tanf_financial_data(
        national_df,
        state_sheets,
    )

    assert national_spending == pytest.approx(8_186_013_422.99)
    assert state_df.loc[state_df["state_fips"] == 6, "tanf"].iloc[0] == pytest.approx(
        3_908_497_323.43
    )
    assert state_df.loc[state_df["state_fips"] == 11, "tanf"].iloc[
        0
    ] == pytest.approx(51_920_224.78)
