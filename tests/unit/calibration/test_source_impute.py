"""Tests for source_impute module.

Uses skip flags to avoid loading real donor data.
"""

import numpy as np
import pandas as pd

from policyengine_us_data.calibration.source_impute import (
    ACS_IMPUTED_VARIABLES,
    ACS_PREDICTORS,
    ALL_SOURCE_VARIABLES,
    SCF_IMPUTED_VARIABLES,
    SCF_PREDICTORS,
    SIPP_ASSETS_PREDICTORS,
    SIPP_IMPUTED_VARIABLES,
    SSI_DISABILITY_MODEL_VARIABLE,
    SIPP_TIPS_PREDICTORS,
    _add_cps_asset_predictors,
    _impute_acs,
    _impute_org,
    _impute_scf,
    _impute_sipp,
    _person_is_married,
    _person_state_fips,
    impute_source_variables,
    preserve_under_65_ssi_disability_criteria,
)
from policyengine_us_data.datasets.sipp.sipp import ASSET_PREDICTORS
from policyengine_us_data.datasets.cps.tipped_occupation import (
    derive_any_treasury_tipped_occupation_code,
    derive_is_tipped_occupation,
)
from policyengine_us_data.datasets.org import ORG_IMPUTED_VARIABLES


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
        "treasury_tipped_occupation_code": {
            time_period: np.zeros(n_persons, dtype=np.int16),
        },
        "rent": {time_period: np.zeros(n_persons)},
        "real_estate_taxes": {time_period: np.zeros(n_persons)},
        "tip_income": {time_period: np.zeros(n_persons)},
        "bank_account_assets": {time_period: np.zeros(n_persons)},
        "stock_assets": {time_period: np.zeros(n_persons)},
        "bond_assets": {time_period: np.zeros(n_persons)},
        "household_vehicles_owned": {time_period: np.zeros(n_hh, dtype=np.int32)},
        "household_vehicles_value": {time_period: np.zeros(n_hh, dtype=np.float32)},
        "hourly_wage": {time_period: np.zeros(n_persons)},
        "is_paid_hourly": {time_period: np.zeros(n_persons, dtype=bool)},
        "is_union_member_or_covered": {
            time_period: np.zeros(n_persons, dtype=bool),
        },
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
        assert SSI_DISABILITY_MODEL_VARIABLE in SIPP_IMPUTED_VARIABLES
        assert "household_vehicles_owned" in SIPP_IMPUTED_VARIABLES
        assert "household_vehicles_value" in SIPP_IMPUTED_VARIABLES

    def test_scf_variables_defined(self):
        assert "net_worth" in SCF_IMPUTED_VARIABLES
        assert "auto_loan_balance" in SCF_IMPUTED_VARIABLES
        assert "auto_loan_interest" in SCF_IMPUTED_VARIABLES
        assert "scf_retirement_assets" in SCF_IMPUTED_VARIABLES
        assert "scf_vehicle_installment_debt" in SCF_IMPUTED_VARIABLES
        assert "scf_mortgage_debt" in SCF_IMPUTED_VARIABLES

    def test_org_variables_defined(self):
        assert "hourly_wage" in ORG_IMPUTED_VARIABLES
        assert "is_paid_hourly" in ORG_IMPUTED_VARIABLES
        assert "is_union_member_or_covered" in ORG_IMPUTED_VARIABLES

    def test_all_source_variables_defined(self):
        expected = (
            ACS_IMPUTED_VARIABLES
            + SIPP_IMPUTED_VARIABLES
            + ORG_IMPUTED_VARIABLES
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

    def test_sipp_tips_uses_tipped_occupation_status(self):
        assert "is_tipped_occupation" in SIPP_TIPS_PREDICTORS

    def test_sipp_assets_has_income(self):
        assert "employment_income" in SIPP_ASSETS_PREDICTORS

    def test_sipp_assets_use_shared_asset_predictors(self):
        assert SIPP_ASSETS_PREDICTORS == ASSET_PREDICTORS

    def test_sipp_assets_exclude_circular_and_noncomparable_predictors(self):
        assert "ssi" not in SIPP_ASSETS_PREDICTORS
        assert "ssi_reported" not in SIPP_ASSETS_PREDICTORS
        assert "RSSI_YRYN" not in SIPP_ASSETS_PREDICTORS
        assert not any("disab" in pred.lower() for pred in SIPP_ASSETS_PREDICTORS)

    def test_sipp_assets_include_comparable_income_and_household_predictors(self):
        expected = {
            "employment_income",
            "interest_income",
            "dividend_income",
            "rental_income",
            "social_security",
            "retirement_income",
            "non_ssi_income",
            "count_under_18",
            "count_under_6",
            "household_size",
        }
        assert expected <= set(SIPP_ASSETS_PREDICTORS)

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
            skip_org=True,
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
            skip_org=True,
            skip_scf=True,
        )

        for var in [
            "rent",
            "real_estate_taxes",
            "tip_income",
            "hourly_wage",
            "is_union_member_or_covered",
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
            skip_org=True,
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


class TestAssetPredictorHelpers:
    def test_person_is_married_uses_existing_flag(self):
        data = {"is_married": {2024: np.array([1, 0, 1], dtype=bool)}}

        result = _person_is_married(data, 2024, 3)

        np.testing.assert_array_equal(result, np.array([1.0, 0.0, 1.0]))

    def test_person_is_married_falls_back_to_marital_unit_id(self):
        data = {
            "person_marital_unit_id": {
                2024: np.array([10, 10, 20, 30, 30]),
            }
        }

        result = _person_is_married(data, 2024, 5)

        np.testing.assert_array_equal(
            result,
            np.array([1.0, 1.0, 0.0, 1.0, 1.0]),
        )

    def test_add_cps_asset_predictors_builds_non_ssi_income(self):
        data = {
            "person_household_id": {2024: np.array([1, 1, 2])},
            "age": {2024: np.array([40, 6, 70], dtype=np.float32)},
            "person_marital_unit_id": {2024: np.array([1, 1, 2])},
            "social_security": {2024: np.array([100.0, 0.0, 500.0])},
            "retirement_distributions": {2024: np.array([10.0, 0.0, 20.0])},
            "pension_income": {2024: np.array([5.0, 0.0, 30.0])},
        }
        cps = pd.DataFrame(
            {
                "employment_income": [1000.0, 0.0, 200.0],
                "interest_income": [1.0, 0.0, 2.0],
                "dividend_income": [3.0, 0.0, 4.0],
                "rental_income": [0.0, 0.0, 5.0],
                "age": [40.0, 6.0, 70.0],
                "is_male": [True, False, False],
            }
        )

        result = _add_cps_asset_predictors(cps, data, 2024)

        assert set(SIPP_ASSETS_PREDICTORS) <= set(result.columns)
        np.testing.assert_array_equal(result["count_under_18"], [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result["count_under_6"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result["household_size"], [2.0, 2.0, 1.0])
        np.testing.assert_array_equal(result["is_married"], [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result["retirement_income"], [15.0, 0.0, 50.0])
        np.testing.assert_array_equal(
            result["non_ssi_income"],
            [1115.0, 0.0, 750.0],
        )


class TestSubfunctions:
    def test_impute_acs_exists(self):
        assert callable(_impute_acs)

    def test_impute_sipp_exists(self):
        assert callable(_impute_sipp)

    def test_impute_org_exists(self):
        assert callable(_impute_org)

    def test_impute_scf_exists(self):
        assert callable(_impute_scf)

    def test_source_impute_preserves_existing_under_65_ssi_criteria(self):
        fake_model_predictions = np.array([False, False, False])

        result = preserve_under_65_ssi_disability_criteria(
            fake_model_predictions,
            age=np.array([40, 64, 70]),
            existing_meets_ssi_disability_criteria=np.array([True, False, True]),
        )

        np.testing.assert_array_equal(result, np.array([True, False, False]))


class TestTippedOccupationHelpers:
    def test_derive_any_treasury_tipped_occupation_code(self):
        occupations = pd.DataFrame(
            {
                "TJB1_OCC": [4040, 1021, np.nan],
                "TJB2_OCC": [np.nan, 4110, 9620],
            }
        )
        derived = derive_any_treasury_tipped_occupation_code(occupations)
        np.testing.assert_array_equal(derived, np.array([101, 102, 809]))

    def test_derive_is_tipped_occupation(self):
        derived = derive_is_tipped_occupation(np.array([0, 101, 809]))
        np.testing.assert_array_equal(derived, np.array([False, True, True]))
