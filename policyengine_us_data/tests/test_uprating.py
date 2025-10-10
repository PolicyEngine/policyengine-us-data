"""
Unit tests for calibration target uprating functionality.
"""

import pytest
import pandas as pd
import numpy as np
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import (
    uprate_targets_df,
)


@pytest.fixture(scope="module")
def sim():
    """Create a microsimulation instance for testing."""
    return Microsimulation(
        dataset="hf://policyengine/test/extended_cps_2023.h5"
    )


@pytest.fixture
def test_targets_2023():
    """Create test data with various source years to uprate to 2023."""
    return pd.DataFrame(
        [
            # Income values from 2022 (should use CPI-U)
            {"variable": "income_tax", "value": 1000000, "period": 2022},
            {"variable": "wages", "value": 5000000, "period": 2022},
            # Count values from 2022 (should use Population)
            {"variable": "person_count", "value": 100000, "period": 2022},
            {"variable": "household_count", "value": 40000, "period": 2022},
            # Values from 2023 (should NOT be uprated)
            {"variable": "income_tax", "value": 1100000, "period": 2023},
            {"variable": "person_count", "value": 101000, "period": 2023},
            # Values from 2024 (should be DOWNRATED to 2023)
            {"variable": "income_tax", "value": 1200000, "period": 2024},
            {"variable": "person_count", "value": 102000, "period": 2024},
        ]
    )


def test_uprating_adds_tracking_columns(test_targets_2023, sim):
    """Test that uprating adds the expected tracking columns."""
    uprated = uprate_targets_df(test_targets_2023, target_year=2023, sim=sim)

    assert "original_value" in uprated.columns
    assert "uprating_factor" in uprated.columns
    assert "uprating_source" in uprated.columns


def test_no_uprating_for_target_year(test_targets_2023, sim):
    """Test that values from the target year are not uprated."""
    uprated = uprate_targets_df(test_targets_2023, target_year=2023, sim=sim)

    # Filter for 2023 data
    target_year_data = uprated[uprated["period"] == 2023]

    # Check that 2023 data was not modified
    assert (target_year_data["uprating_factor"] == 1.0).all()
    assert (target_year_data["uprating_source"] == "None").all()
    assert (
        target_year_data["value"] == target_year_data["original_value"]
    ).all()


def test_cpi_uprating_for_monetary_values(test_targets_2023, sim):
    """Test that monetary values use CPI-U uprating."""
    uprated = uprate_targets_df(test_targets_2023, target_year=2023, sim=sim)

    # Check income tax from 2022
    income_2022 = uprated[
        (uprated["variable"] == "income_tax") & (uprated["period"] == 2022)
    ].iloc[0]
    assert income_2022["uprating_source"] == "CPI-U"
    assert (
        income_2022["uprating_factor"] > 1.0
    )  # Should be inflated from 2022 to 2023
    assert (
        abs(income_2022["uprating_factor"] - 1.0641) < 0.001
    )  # Expected CPI factor

    # Check wages from 2022
    wages_2022 = uprated[
        (uprated["variable"] == "wages") & (uprated["period"] == 2022)
    ].iloc[0]
    assert wages_2022["uprating_source"] == "CPI-U"
    assert (
        wages_2022["uprating_factor"] == income_2022["uprating_factor"]
    )  # Same CPI factor


def test_population_uprating_for_counts(test_targets_2023, sim):
    """Test that count variables use population uprating."""
    uprated = uprate_targets_df(test_targets_2023, target_year=2023, sim=sim)

    # Check person count from 2022
    person_2022 = uprated[
        (uprated["variable"] == "person_count") & (uprated["period"] == 2022)
    ].iloc[0]
    assert person_2022["uprating_source"] == "Population"
    assert (
        person_2022["uprating_factor"] > 1.0
    )  # Population grew from 2022 to 2023
    assert (
        abs(person_2022["uprating_factor"] - 1.0094) < 0.001
    )  # Expected population factor

    # Check household count from 2022
    household_2022 = uprated[
        (uprated["variable"] == "household_count")
        & (uprated["period"] == 2022)
    ].iloc[0]
    assert household_2022["uprating_source"] == "Population"
    assert (
        household_2022["uprating_factor"] == person_2022["uprating_factor"]
    )  # Same population factor


def test_downrating_from_future_years(test_targets_2023, sim):
    """Test that values from future years are correctly downrated."""
    uprated = uprate_targets_df(test_targets_2023, target_year=2023, sim=sim)

    # Check income tax from 2024 (should be downrated)
    income_2024 = uprated[
        (uprated["variable"] == "income_tax") & (uprated["period"] == 2024)
    ].iloc[0]
    assert income_2024["uprating_source"] == "CPI-U"
    assert (
        income_2024["uprating_factor"] < 1.0
    )  # Should be deflated from 2024 to 2023
    assert (
        abs(income_2024["uprating_factor"] - 0.9700) < 0.001
    )  # Expected CPI factor

    # Check person count from 2024
    person_2024 = uprated[
        (uprated["variable"] == "person_count") & (uprated["period"] == 2024)
    ].iloc[0]
    assert person_2024["uprating_source"] == "Population"
    assert (
        person_2024["uprating_factor"] < 1.0
    )  # Population was higher in 2024
    assert (
        abs(person_2024["uprating_factor"] - 0.9892) < 0.001
    )  # Expected population factor


def test_values_are_modified_correctly(test_targets_2023, sim):
    """Test that values are actually modified by the uprating factors."""
    uprated = uprate_targets_df(test_targets_2023, target_year=2023, sim=sim)

    for _, row in uprated.iterrows():
        if row["uprating_factor"] != 1.0:
            # Check that value was modified
            expected_value = row["original_value"] * row["uprating_factor"]
            assert (
                abs(row["value"] - expected_value) < 1.0
            )  # Allow for rounding


def test_no_double_uprating(test_targets_2023, sim):
    """Test that calling uprate_targets_df twice doesn't double-uprate."""
    uprated_once = uprate_targets_df(
        test_targets_2023, target_year=2023, sim=sim
    )
    uprated_twice = uprate_targets_df(uprated_once, target_year=2023, sim=sim)

    # Values should be identical after second call
    pd.testing.assert_series_equal(
        uprated_once["value"], uprated_twice["value"]
    )
    pd.testing.assert_series_equal(
        uprated_once["uprating_factor"], uprated_twice["uprating_factor"]
    )


def test_numpy_int_compatibility(sim):
    """Test that numpy int64 types work correctly (regression test)."""
    # Create data with numpy int64 period column
    data = pd.DataFrame(
        {
            "variable": ["income_tax"],
            "value": [1000000],
            "period": np.array([2022], dtype=np.int64),
        }
    )

    # This should not raise an exception
    uprated = uprate_targets_df(data, target_year=2023, sim=sim)

    # And should actually uprate
    assert uprated["uprating_factor"].iloc[0] > 1.0
    assert uprated["value"].iloc[0] > uprated["original_value"].iloc[0]


def test_missing_period_column():
    """Test that missing period column is handled gracefully."""
    data = pd.DataFrame({"variable": ["income_tax"], "value": [1000000]})

    result = uprate_targets_df(data, target_year=2023)

    # Should return unchanged
    pd.testing.assert_frame_equal(result, data)
