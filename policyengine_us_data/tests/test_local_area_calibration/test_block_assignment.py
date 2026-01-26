"""Tests for census block assignment functionality.

This tests the new block-based geographic assignment that replaces county-only
assignment, allowing consistent lookup of all geographic variables from a
single census block GEOID.
"""

import pytest
import numpy as np


class TestBlockAssignment:
    """Test census block assignment for CDs."""

    def test_assign_returns_correct_shape(self):
        """Verify assign_blocks_for_cd returns correct shape."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_blocks_for_cd,
        )

        n_households = 100
        result = assign_blocks_for_cd("3610", n_households, seed=42)
        assert result.shape == (n_households,)
        # Block GEOIDs are 15-character strings
        assert all(len(geoid) == 15 for geoid in result)

    def test_assign_is_deterministic(self):
        """Verify same seed produces same results."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_blocks_for_cd,
        )

        result1 = assign_blocks_for_cd("3610", 50, seed=42)
        result2 = assign_blocks_for_cd("3610", 50, seed=42)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_different_results(self):
        """Verify different seeds produce different results."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_blocks_for_cd,
        )

        result1 = assign_blocks_for_cd("3610", 100, seed=42)
        result2 = assign_blocks_for_cd("3610", 100, seed=99)
        assert not np.array_equal(result1, result2)

    def test_ny_cd_gets_ny_blocks(self):
        """Verify NY CDs get NY blocks."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_blocks_for_cd,
        )

        # NY-10 (Manhattan/Brooklyn area)
        result = assign_blocks_for_cd("3610", 100, seed=42)

        # All block GEOIDs should start with NY state FIPS (36)
        for geoid in result:
            assert geoid.startswith("36"), f"Got non-NY block: {geoid}"

    def test_ca_cd_gets_ca_blocks(self):
        """Verify CA CDs get CA blocks."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_blocks_for_cd,
        )

        # CA-12 (San Francisco area)
        result = assign_blocks_for_cd("612", 100, seed=42)

        # All block GEOIDs should start with CA state FIPS (06)
        for geoid in result:
            assert geoid.startswith("06"), f"Got non-CA block: {geoid}"


class TestGeographyLookup:
    """Test looking up geographic variables from block GEOID."""

    def test_get_county_from_block(self):
        """Verify county FIPS extraction from block GEOID."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_county_fips_from_block,
        )

        # New York County (Manhattan) block example
        # Block GEOID format: SSCCCTTTTTBBBBB (state, county, tract, block)
        # 36061 = NY state (36) + New York County (061)
        block_geoid = "360610001001000"  # Example Manhattan block
        county_fips = get_county_fips_from_block(block_geoid)
        assert county_fips == "36061"

    def test_get_tract_from_block(self):
        """Verify tract GEOID extraction from block GEOID."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_tract_geoid_from_block,
        )

        block_geoid = "360610001001000"
        tract_geoid = get_tract_geoid_from_block(block_geoid)
        # Tract is positions 0-10 (state + county + tract)
        assert tract_geoid == "36061000100"

    def test_get_state_fips_from_block(self):
        """Verify state FIPS extraction from block GEOID."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_state_fips_from_block,
        )

        block_geoid = "360610001001000"
        state_fips = get_state_fips_from_block(block_geoid)
        assert state_fips == "36"


class TestCBSALookup:
    """Test CBSA/metro area lookup from county."""

    def test_manhattan_in_nyc_metro(self):
        """Verify Manhattan (New York County) is in NYC metro area."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_cbsa_from_county,
        )

        # New York County FIPS = 36061
        cbsa_code = get_cbsa_from_county("36061")
        # NYC metro area CBSA code = 35620
        assert cbsa_code == "35620"

    def test_sf_county_in_sf_metro(self):
        """Verify San Francisco County is in SF metro area."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_cbsa_from_county,
        )

        # San Francisco County FIPS = 06075
        cbsa_code = get_cbsa_from_county("06075")
        # SF-Oakland-Berkeley metro area CBSA code = 41860
        assert cbsa_code == "41860"

    def test_rural_county_no_cbsa(self):
        """Verify rural county not in any metro area returns None."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_cbsa_from_county,
        )

        # Wheeler County, NE (FIPS 31183) - rural county not in CBSA
        cbsa_code = get_cbsa_from_county("31183")
        assert cbsa_code is None


class TestIntegratedAssignment:
    """Test integrated assignment that returns all geography."""

    def test_assign_geography_returns_all_fields(self):
        """Verify assign_geography returns dict with all geography fields."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_geography_for_cd,
        )

        n_households = 50
        result = assign_geography_for_cd("3610", n_households, seed=42)

        # Should return dict with arrays for each geography
        expected_fields = [
            "block_geoid",
            "county_fips",
            "tract_geoid",
            "state_fips",
            "cbsa_code",
            "sldu",
            "sldl",
            "place_fips",
            "vtd",
            "puma",
            "county_index",
        ]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

        # All arrays should have same length
        for key, arr in result.items():
            assert len(arr) == n_households, f"{key} has wrong length"

    def test_geography_is_consistent(self):
        """Verify all geography fields are consistent with each other."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_geography_for_cd,
        )

        result = assign_geography_for_cd("3610", 100, seed=42)

        # County should be derived from block
        for i in range(100):
            block = result["block_geoid"][i]
            county = result["county_fips"][i]
            tract = result["tract_geoid"][i]
            state = result["state_fips"][i]

            # Block starts with county
            assert block[:5] == county
            # Tract is first 11 chars of block
            assert block[:11] == tract
            # State is first 2 chars
            assert block[:2] == state


class TestStateLegislativeDistricts:
    """Test state legislative district lookups."""

    def test_get_sldu_from_block(self):
        """Verify SLDU lookup from block GEOID."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_sldu_from_block,
        )

        # Alabama block from crosswalk
        sldu = get_sldu_from_block("010010201001000")
        # Should return a 3-character district code or None
        assert sldu is None or (isinstance(sldu, str) and len(sldu) <= 3)

    def test_get_sldl_from_block(self):
        """Verify SLDL lookup from block GEOID."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_sldl_from_block,
        )

        # Alabama block from crosswalk
        sldl = get_sldl_from_block("010010201001000")
        # Should return a 3-character district code or None
        assert sldl is None or (isinstance(sldl, str) and len(sldl) <= 3)

    def test_assign_geography_includes_state_leg(self):
        """Verify assign_geography includes SLDU and SLDL."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_geography_for_cd,
        )

        result = assign_geography_for_cd("3610", 50, seed=42)

        assert "sldu" in result
        assert "sldl" in result
        assert len(result["sldu"]) == 50
        assert len(result["sldl"]) == 50


class TestPlaceLookup:
    """Test place/city lookup from block."""

    def test_get_place_fips_from_block(self):
        """Verify place FIPS lookup from block GEOID."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_place_fips_from_block,
        )

        # Alabama block that's in a place
        place = get_place_fips_from_block("010010201001000")
        # Should return 5-char place FIPS or None
        assert place is None or (isinstance(place, str) and len(place) == 5)

    def test_assign_geography_includes_place(self):
        """Verify assign_geography includes place_fips."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_geography_for_cd,
        )

        result = assign_geography_for_cd("3610", 50, seed=42)

        assert "place_fips" in result
        assert len(result["place_fips"]) == 50


class TestPUMALookup:
    """Test PUMA lookup from block."""

    def test_get_puma_from_block(self):
        """Verify PUMA lookup from block GEOID."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_puma_from_block,
        )

        # Alabama block
        puma = get_puma_from_block("010010201001000")
        # Should return 5-char PUMA code or None
        assert puma is None or (isinstance(puma, str) and len(puma) == 5)

    def test_assign_geography_includes_puma(self):
        """Verify assign_geography includes PUMA."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_geography_for_cd,
        )

        result = assign_geography_for_cd("3610", 50, seed=42)

        assert "puma" in result
        assert len(result["puma"]) == 50


class TestVTDLookup:
    """Test VTD (Voting Tabulation District) lookup from block."""

    def test_get_vtd_from_block(self):
        """Verify VTD lookup from block GEOID."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_vtd_from_block,
        )

        # Alabama block
        vtd = get_vtd_from_block("010010201001000")
        # Should return VTD code string or None
        assert vtd is None or isinstance(vtd, str)

    def test_assign_geography_includes_vtd(self):
        """Verify assign_geography includes VTD."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_geography_for_cd,
        )

        result = assign_geography_for_cd("3610", 50, seed=42)

        assert "vtd" in result
        assert len(result["vtd"]) == 50


class TestAllGeographyLookup:
    """Test bulk lookup of all geography from block."""

    def test_get_all_geography_returns_all_fields(self):
        """Verify get_all_geography_from_block returns all expected fields."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_all_geography_from_block,
        )

        result = get_all_geography_from_block("010010201001000")

        expected_keys = ["sldu", "sldl", "place_fips", "vtd", "puma"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_get_all_geography_unknown_block(self):
        """Verify get_all_geography handles unknown block gracefully."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_all_geography_from_block,
        )

        result = get_all_geography_from_block("999999999999999")

        # Should return dict with all None values
        for key, val in result.items():
            assert val is None, f"{key} should be None for unknown block"


class TestCountyEnumIntegration:
    """Test integration with existing County enum."""

    def test_get_county_enum_from_block(self):
        """Verify we can get County enum index from block GEOID."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            get_county_enum_index_from_block,
        )
        from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
            County,
        )

        # Manhattan block
        block_geoid = "360610001001000"
        county_idx = get_county_enum_index_from_block(block_geoid)

        # Should map to NEW_YORK_COUNTY_NY
        assert County._member_names_[county_idx] == "NEW_YORK_COUNTY_NY"

    def test_assign_geography_includes_county_index(self):
        """Verify assign_geography includes county_index for backwards compat."""
        from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
            assign_geography_for_cd,
        )
        from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
            County,
        )

        result = assign_geography_for_cd("3610", 50, seed=42)

        # Should include county_index for backwards compatibility
        assert "county_index" in result
        assert result["county_index"].dtype == np.int32

        # All indices should be valid NY counties
        for idx in result["county_index"]:
            county_name = County._member_names_[idx]
            assert county_name.endswith("_NY")
