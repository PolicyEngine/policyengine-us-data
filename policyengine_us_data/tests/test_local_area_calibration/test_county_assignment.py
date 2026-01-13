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
    get_county_filter_probability,
    get_filtered_county_distribution,
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


class TestInvalidCountyExclusion:
    """Test that invalid counties are properly excluded."""

    def test_delaware_has_exactly_3_counties(self):
        """Delaware should have exactly 3 counties (no DORCHESTER)."""
        state_counties = _build_state_counties()
        de_counties = state_counties.get("DE", [])

        assert len(de_counties) == 3
        assert "DORCHESTER_COUNTY_DE" not in de_counties

        expected = {
            "KENT_COUNTY_DE",
            "NEW_CASTLE_COUNTY_DE",
            "SUSSEX_COUNTY_DE",
        }
        assert set(de_counties) == expected

    def test_suffolk_county_ct_excluded(self):
        """Suffolk County, CT should be excluded (doesn't exist)."""
        state_counties = _build_state_counties()
        ct_counties = state_counties.get("CT", [])
        assert "SUFFOLK_COUNTY_CT" not in ct_counties


class TestCountyFilterProbability:
    """Test probability calculations for city datasets."""

    NYC_COUNTIES = {
        "QUEENS_COUNTY_NY",
        "BRONX_COUNTY_NY",
        "RICHMOND_COUNTY_NY",
        "NEW_YORK_COUNTY_NY",
        "KINGS_COUNTY_NY",
    }

    def test_fully_nyc_cd_has_probability_one(self):
        """NY-05 (fully in NYC) should have P(NYC|CD) = 1.0."""
        prob = get_county_filter_probability("3605", self.NYC_COUNTIES)
        assert prob == pytest.approx(1.0, abs=0.001)

    def test_mixed_cd_has_partial_probability(self):
        """NY-03 (mixed NYC/suburbs) should have 0 < P(NYC|CD) < 1."""
        prob = get_county_filter_probability("3603", self.NYC_COUNTIES)
        assert 0 < prob < 1
        # Should be approximately 24% based on Census data
        assert prob == pytest.approx(0.24, abs=0.05)

    def test_non_nyc_cd_has_zero_probability(self):
        """Non-NY CD should have P(NYC|CD) = 0."""
        # CA-12 (San Francisco)
        prob = get_county_filter_probability("612", self.NYC_COUNTIES)
        assert prob == 0.0

    def test_filtered_distribution_sums_to_one(self):
        """Filtered distribution should sum to 1.0."""
        dist = get_filtered_county_distribution("3603", self.NYC_COUNTIES)
        if dist:  # Only if CD has overlap
            assert sum(dist.values()) == pytest.approx(1.0)

    def test_filtered_distribution_only_target_counties(self):
        """Filtered distribution should only contain target counties."""
        dist = get_filtered_county_distribution("3603", self.NYC_COUNTIES)
        for county in dist:
            assert county in self.NYC_COUNTIES

    def test_filtered_distribution_empty_for_no_overlap(self):
        """Non-overlapping CD should return empty distribution."""
        dist = get_filtered_county_distribution("612", self.NYC_COUNTIES)
        assert dist == {}
