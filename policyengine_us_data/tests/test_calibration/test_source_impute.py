"""
Tests for source_impute module — ACS/SIPP/SCF imputations
with state_fips as QRF predictor.

Uses mocks to avoid loading real donor data (ACS, SIPP, SCF).
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# -------------------------------------------------------------------
# Constants tests
# -------------------------------------------------------------------


class TestConstants:
    """Test that module constants are defined correctly."""

    def test_acs_variables_defined(self):
        from policyengine_us_data.calibration.source_impute import (
            ACS_IMPUTED_VARIABLES,
        )

        assert "rent" in ACS_IMPUTED_VARIABLES
        assert "real_estate_taxes" in ACS_IMPUTED_VARIABLES

    def test_sipp_variables_defined(self):
        from policyengine_us_data.calibration.source_impute import (
            SIPP_IMPUTED_VARIABLES,
        )

        assert "tip_income" in SIPP_IMPUTED_VARIABLES
        assert "bank_account_assets" in SIPP_IMPUTED_VARIABLES
        assert "stock_assets" in SIPP_IMPUTED_VARIABLES
        assert "bond_assets" in SIPP_IMPUTED_VARIABLES

    def test_scf_variables_defined(self):
        from policyengine_us_data.calibration.source_impute import (
            SCF_IMPUTED_VARIABLES,
        )

        assert "net_worth" in SCF_IMPUTED_VARIABLES
        assert "auto_loan_balance" in SCF_IMPUTED_VARIABLES
        assert "auto_loan_interest" in SCF_IMPUTED_VARIABLES

    def test_all_source_variables_defined(self):
        from policyengine_us_data.calibration.source_impute import (
            ALL_SOURCE_VARIABLES,
            ACS_IMPUTED_VARIABLES,
            SIPP_IMPUTED_VARIABLES,
            SCF_IMPUTED_VARIABLES,
        )

        expected = (
            ACS_IMPUTED_VARIABLES
            + SIPP_IMPUTED_VARIABLES
            + SCF_IMPUTED_VARIABLES
        )
        assert ALL_SOURCE_VARIABLES == expected


# -------------------------------------------------------------------
# impute_source_variables tests
# -------------------------------------------------------------------


class TestImputeSourceVariables:
    """Test main entry point."""

    def _make_data_dict(self, n_persons=100, time_period=2024):
        """Create a minimal data dict for testing."""
        n_hh = n_persons // 2
        rng = np.random.default_rng(42)
        return {
            "person_id": {
                time_period: np.arange(n_persons),
            },
            "household_id": {
                time_period: np.arange(n_hh),
            },
            "person_household_id": {
                time_period: np.repeat(np.arange(n_hh), 2),
            },
            "age": {
                time_period: rng.integers(18, 80, n_persons).astype(
                    np.float32
                ),
            },
            "employment_income": {
                time_period: rng.uniform(0, 100000, n_persons).astype(
                    np.float32
                ),
            },
            # Variables that will be overwritten
            "rent": {
                time_period: np.zeros(n_persons),
            },
            "real_estate_taxes": {
                time_period: np.zeros(n_persons),
            },
            "tip_income": {
                time_period: np.zeros(n_persons),
            },
            "bank_account_assets": {
                time_period: np.zeros(n_persons),
            },
            "stock_assets": {
                time_period: np.zeros(n_persons),
            },
            "bond_assets": {
                time_period: np.zeros(n_persons),
            },
            "net_worth": {
                time_period: np.zeros(n_persons),
            },
            "auto_loan_balance": {
                time_period: np.zeros(n_persons),
            },
            "auto_loan_interest": {
                time_period: np.zeros(n_persons),
            },
        }

    def test_function_exists(self):
        """impute_source_variables is importable."""
        from policyengine_us_data.calibration.source_impute import (
            impute_source_variables,
        )

        assert callable(impute_source_variables)

    def test_returns_dict(self):
        """Returns a dict with same keys as input."""
        from policyengine_us_data.calibration.source_impute import (
            impute_source_variables,
        )

        data = self._make_data_dict(n_persons=20)
        state_fips = np.ones(10, dtype=np.int32) * 6

        result = impute_source_variables(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_acs=True,
            skip_sipp=True,
            skip_scf=True,
        )
        assert isinstance(result, dict)
        # All original keys preserved
        for key in data:
            assert key in result

    def test_skip_flags_work(self):
        """When all skip flags True, data unchanged."""
        from policyengine_us_data.calibration.source_impute import (
            impute_source_variables,
        )

        data = self._make_data_dict(n_persons=20)
        state_fips = np.ones(10, dtype=np.int32) * 6

        result = impute_source_variables(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_acs=True,
            skip_sipp=True,
            skip_scf=True,
        )

        for var in [
            "rent",
            "real_estate_taxes",
            "tip_income",
            "net_worth",
        ]:
            np.testing.assert_array_equal(result[var][2024], data[var][2024])

    def test_state_fips_added_to_data(self):
        """state_fips is added to the returned data dict."""
        from policyengine_us_data.calibration.source_impute import (
            impute_source_variables,
        )

        data = self._make_data_dict(n_persons=20)
        state_fips = np.ones(10, dtype=np.int32) * 6

        result = impute_source_variables(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_acs=True,
            skip_sipp=True,
            skip_scf=True,
        )

        assert "state_fips" in result


# -------------------------------------------------------------------
# Individual source imputation tests
# -------------------------------------------------------------------


class TestACSImputation:
    """Test _impute_acs function."""

    def test_function_exists(self):
        from policyengine_us_data.calibration.source_impute import (
            _impute_acs,
        )

        assert callable(_impute_acs)


class TestSIPPImputation:
    """Test _impute_sipp function."""

    def test_function_exists(self):
        from policyengine_us_data.calibration.source_impute import (
            _impute_sipp,
        )

        assert callable(_impute_sipp)


class TestSCFImputation:
    """Test _impute_scf function."""

    def test_function_exists(self):
        from policyengine_us_data.calibration.source_impute import (
            _impute_scf,
        )

        assert callable(_impute_scf)
