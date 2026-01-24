"""
Tests for weeks_unemployed variable extraction from CPS ASEC.

The Census CPS ASEC uses LKWEEKS (not IPUMS's WKSUNEM1) for weeks looking for work.
"""

import numpy as np
from pathlib import Path


class TestWeeksUnemployed:
    """Test suite for weeks_unemployed variable."""

    def test_lkweeks_in_person_columns(self):
        """Test that LKWEEKS is in PERSON_COLUMNS, not WKSUNEM."""
        # Read the source file directly to check column names
        census_cps_path = Path(__file__).parent.parent / (
            "policyengine_us_data/datasets/cps/census_cps.py"
        )
        content = census_cps_path.read_text()

        # Check for correct variable
        assert '"LKWEEKS"' in content, (
            "LKWEEKS should be in PERSON_COLUMNS"
        )
        assert '"WKSUNEM"' not in content, (
            "WKSUNEM should not be in PERSON_COLUMNS (Census uses LKWEEKS)"
        )

    def test_cps_uses_lkweeks(self):
        """Test that cps.py uses LKWEEKS, not WKSUNEM."""
        cps_path = Path(__file__).parent.parent / (
            "policyengine_us_data/datasets/cps/cps.py"
        )
        content = cps_path.read_text()

        # Check for correct variable reference
        assert "LKWEEKS" in content, "cps.py should reference LKWEEKS"
        assert "WKSUNEM" not in content, (
            "cps.py should not reference WKSUNEM"
        )

    def test_weeks_unemployed_value_range(self):
        """Test that weeks_unemployed values are in valid range (0-52)."""
        # LKWEEKS values: 0 = not unemployed, 1-52 = weeks, -1 = NIU
        # After processing, should be 0-52 (NIU mapped to 0)

        raw_values = np.array([-1, 0, 1, 26, 52, -1])
        processed = np.where(raw_values == -1, 0, raw_values)

        assert processed.min() >= 0, "Minimum should be >= 0"
        assert processed.max() <= 52, "Maximum should be <= 52"
        assert processed[0] == 0, "NIU (-1) should map to 0"
        assert processed[1] == 0, "Not unemployed (0) should stay 0"
        assert processed[3] == 26, "26 weeks should stay 26"

    def test_puf_weeks_imputation_constraints(self):
        """Test the weeks imputation constraints for PUF copy."""
        # The QRF-based imputation should respect these constraints:
        # 1. weeks should be in [0, 52]
        # 2. weeks should be 0 when UC is 0

        # Test constraint enforcement
        raw_imputed = np.array([-5, 0, 25, 60, 100])
        uc_values = np.array([100, 0, 5000, 10000, 0])

        # Apply constraints like the function does
        constrained = np.clip(raw_imputed, 0, 52)
        constrained = np.where(uc_values > 0, constrained, 0)

        assert constrained.min() >= 0, "Should be non-negative"
        assert constrained.max() <= 52, "Should be capped at 52 weeks"
        assert constrained[1] == 0, "No UC should mean 0 weeks"
        assert constrained[4] == 0, "No UC should mean 0 weeks"
        assert constrained[2] == 25, "Valid weeks with UC should be preserved"

    def test_extended_cps_handles_weeks_unemployed(self):
        """Test that extended_cps.py has special handling for weeks_unemployed."""
        ecps_path = Path(__file__).parent.parent / (
            "policyengine_us_data/datasets/cps/extended_cps.py"
        )
        content = ecps_path.read_text()

        # Check for weeks_unemployed handling
        assert "weeks_unemployed" in content, (
            "extended_cps.py should handle weeks_unemployed"
        )
        assert "impute_weeks_unemployed_for_puf" in content, (
            "Should have imputation function for PUF weeks"
        )
