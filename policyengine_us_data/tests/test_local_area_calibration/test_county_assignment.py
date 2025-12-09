"""Tests for county assignment functionality."""

import pytest
import numpy as np

from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
    County,
)
from policyengine_us_data.datasets.cps.local_area_calibration.county_assignment import (
    assign_counties_for_cd,
    get_county_index,
    _build_state_counties,
)


class TestCountyAssignment:
    """Test county assignment for CDs."""

    def test_assign_returns_correct_shape(self):
        """Verify assign_counties_for_cd returns correct shape."""
        n_households = 100
        result = assign_counties_for_cd("3610", n_households, seed=42)
        assert result.shape == (n_households,)
        assert result.dtype == np.int32

    def test_assign_is_deterministic(self):
        """Verify same seed produces same results."""
        result1 = assign_counties_for_cd("3610", 50, seed=42)
        result2 = assign_counties_for_cd("3610", 50, seed=42)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_different_results(self):
        """Verify different seeds produce different results."""
        result1 = assign_counties_for_cd("3610", 100, seed=42)
        result2 = assign_counties_for_cd("3610", 100, seed=99)
        # With 100 samples, should be different
        assert not np.array_equal(result1, result2)

    def test_ny_cd_gets_ny_counties(self):
        """Verify NY CDs get NY counties."""
        # NY-10 (Manhattan/Brooklyn area)
        result = assign_counties_for_cd("3610", 100, seed=42)

        # All indices should be valid County enum indices
        for idx in result:
            county_name = County._member_names_[idx]
            # Should end with _NY
            assert county_name.endswith(
                "_NY"
            ), f"Got non-NY county: {county_name}"

    def test_ca_cd_gets_ca_counties(self):
        """Verify CA CDs get CA counties."""
        # CA-12 (San Francisco area)
        result = assign_counties_for_cd("612", 100, seed=42)

        for idx in result:
            county_name = County._member_names_[idx]
            assert county_name.endswith(
                "_CA"
            ), f"Got non-CA county: {county_name}"


class TestCountyIndex:
    """Test county index conversion."""

    def test_get_county_index_known_county(self):
        """Verify known county returns valid index."""
        idx = get_county_index("NEW_YORK_COUNTY_NY")
        assert isinstance(idx, int)
        assert idx >= 0
        assert County._member_names_[idx] == "NEW_YORK_COUNTY_NY"

    def test_get_county_index_unknown(self):
        """Verify UNKNOWN county returns valid index."""
        idx = get_county_index("UNKNOWN")
        assert isinstance(idx, int)
        assert idx >= 0


class TestStateCuntiesMapping:
    """Test state to counties mapping."""

    def test_build_state_counties_excludes_unknown(self):
        """Verify UNKNOWN is not in any state's county list."""
        state_counties = _build_state_counties()
        for state, counties in state_counties.items():
            assert "UNKNOWN" not in counties

    def test_all_50_states_plus_dc(self):
        """Verify we have counties for all 50 states + DC."""
        state_counties = _build_state_counties()
        # Check for some known states
        assert "NY" in state_counties
        assert "CA" in state_counties
        assert "TX" in state_counties
        assert "DC" in state_counties
        # Should have 51 (50 states + DC)
        assert len(state_counties) >= 51

    def test_ny_has_nyc_counties(self):
        """Verify NY includes NYC counties."""
        state_counties = _build_state_counties()
        ny_counties = state_counties["NY"]

        nyc_counties = [
            "QUEENS_COUNTY_NY",
            "BRONX_COUNTY_NY",
            "RICHMOND_COUNTY_NY",
            "NEW_YORK_COUNTY_NY",
            "KINGS_COUNTY_NY",
        ]
        for county in nyc_counties:
            assert county in ny_counties, f"Missing NYC county: {county}"
