"""Tests for source_impute module.

Uses skip flags to avoid loading real donor data.
"""

import numpy as np

from policyengine_us_data.calibration.source_impute import (
    ACS_IMPUTED_VARIABLES,
    ACS_PREDICTORS,
    ALL_SOURCE_VARIABLES,
    SCF_IMPUTED_VARIABLES,
    SCF_PREDICTORS,
    SIPP_ASSETS_PREDICTORS,
    SIPP_IMPUTED_VARIABLES,
    SIPP_TIPS_PREDICTORS,
    _impute_acs,
    _impute_scf,
    _impute_sipp,
    _person_state_fips,
    impute_source_variables,
)


def _make_data_dict(n_persons=20, time_period=2024):
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
            time_period: rng.integers(18, 80, n_persons).astype(np.float32),
        },
        "employment_income": {
            time_period: rng.uniform(0, 100000, n_persons).astype(np.float32),
        },
        "rent": {time_period: np.zeros(n_persons)},
        "real_estate_taxes": {time_period: np.zeros(n_persons)},
        "tip_income": {time_period: np.zeros(n_persons)},
        "bank_account_assets": {time_period: np.zeros(n_persons)},
        "stock_assets": {time_period: np.zeros(n_persons)},
        "bond_assets": {time_period: np.zeros(n_persons)},
        "net_worth": {time_period: np.zeros(n_persons)},
        "auto_loan_balance": {time_period: np.zeros(n_persons)},
        "auto_loan_interest": {time_period: np.zeros(n_persons)},
    }


class TestConstants:
    def test_acs_variables_defined(self):
        assert "rent" in ACS_IMPUTED_VARIABLES
        assert "real_estate_taxes" in ACS_IMPUTED_VARIABLES

    def test_sipp_variables_defined(self):
        assert "tip_income" in SIPP_IMPUTED_VARIABLES
        assert "bank_account_assets" in SIPP_IMPUTED_VARIABLES
        assert "stock_assets" in SIPP_IMPUTED_VARIABLES
        assert "bond_assets" in SIPP_IMPUTED_VARIABLES

    def test_scf_variables_defined(self):
        assert "net_worth" in SCF_IMPUTED_VARIABLES
        assert "auto_loan_balance" in SCF_IMPUTED_VARIABLES
        assert "auto_loan_interest" in SCF_IMPUTED_VARIABLES

    def test_all_source_variables_defined(self):
        expected = (
            ACS_IMPUTED_VARIABLES
            + SIPP_IMPUTED_VARIABLES
            + SCF_IMPUTED_VARIABLES
        )
        assert ALL_SOURCE_VARIABLES == expected


class TestPredictorLists:
    def test_acs_uses_state(self):
        # ACS has state identifiers, so state_fips is added at
        # call time in _impute_acs (predictors + ["state_fips"]).
        assert "state_fips" not in ACS_PREDICTORS  # added dynamically

    def test_sipp_tips_has_income(self):
        assert "employment_income" in SIPP_TIPS_PREDICTORS

    def test_sipp_assets_has_income(self):
        assert "employment_income" in SIPP_ASSETS_PREDICTORS

    def test_scf_has_income(self):
        assert "employment_income" in SCF_PREDICTORS

    def test_sipp_and_scf_exclude_state(self):
        # SIPP and SCF lack state identifiers.
        for predictor_list in [
            SIPP_TIPS_PREDICTORS,
            SIPP_ASSETS_PREDICTORS,
            SCF_PREDICTORS,
        ]:
            assert "state_fips" not in predictor_list


class TestImputeSourceVariables:
    def test_function_exists(self):
        assert callable(impute_source_variables)

    def test_returns_dict(self):
        data = _make_data_dict(n_persons=20)
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
        for key in data:
            assert key in result

    def test_skip_flags_preserve_data(self):
        data = _make_data_dict(n_persons=20)
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
        data = _make_data_dict(n_persons=20)
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
        np.testing.assert_array_equal(result["state_fips"][2024], state_fips)


class TestPersonStateFips:
    def test_maps_correctly(self):
        data = {
            "household_id": {2024: np.array([10, 20, 30])},
            "person_household_id": {2024: np.array([10, 10, 20, 20, 30])},
            "person_id": {2024: np.arange(5)},
        }
        state_fips = np.array([1, 2, 3])

        result = _person_state_fips(data, state_fips, 2024)
        expected = np.array([1, 1, 2, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_fallback_equal_sizes(self):
        data = {
            "household_id": {2024: np.array([10, 20])},
            "person_id": {2024: np.arange(4)},
        }
        state_fips = np.array([1, 2])

        result = _person_state_fips(data, state_fips, 2024)
        expected = np.array([1, 1, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_fallback_unequal_sizes(self):
        # Without person_household_id, the fallback must still
        # produce the right length (one state per person).
        data = {
            "household_id": {2024: np.array([10, 20, 30])},
            "person_id": {2024: np.arange(5)},
        }
        state_fips = np.array([1, 2, 3])

        result = _person_state_fips(data, state_fips, 2024)
        assert len(result) == 5


class TestSubfunctions:
    def test_impute_acs_exists(self):
        assert callable(_impute_acs)

    def test_impute_sipp_exists(self):
        assert callable(_impute_sipp)

    def test_impute_scf_exists(self):
        assert callable(_impute_scf)
