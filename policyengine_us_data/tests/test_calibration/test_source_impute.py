"""Tests for source_impute module.

Uses skip flags to avoid loading real donor data.
"""

import numpy as np
import pandas as pd

from policyengine_us_data.calibration.source_impute import (
    ACS_IMPUTED_VARIABLES,
    ACS_PREDICTORS,
    ALL_SOURCE_VARIABLES,
    NET_WORTH_TOTAL_TARGETS,
    SCF_IMPUTED_VARIABLES,
    SCF_PREDICTORS,
    SCF_DONOR_UPRATING_MAP,
    SIPP_ASSETS_PREDICTORS,
    SIPP_IMPUTED_VARIABLES,
    SIPP_TIPS_PREDICTORS,
    _build_household_scf_receiver,
    _household_values_from_data,
    _align_weighted_total,
    _impute_acs,
    _impute_scf,
    _impute_sipp,
    _person_state_fips,
    _uprate_scf_donor_frame,
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
        "taxable_interest_income": {
            time_period: rng.uniform(0, 5000, n_persons).astype(np.float32),
        },
        "qualified_dividend_income": {
            time_period: rng.uniform(0, 4000, n_persons).astype(np.float32),
        },
        "taxable_private_pension_income": {
            time_period: rng.uniform(0, 6000, n_persons).astype(np.float32),
        },
        "social_security_retirement": {
            time_period: rng.uniform(0, 8000, n_persons).astype(np.float32),
        },
        "is_male": {
            time_period: rng.integers(0, 2, n_persons).astype(np.float32),
        },
        "cps_race": {
            time_period: rng.integers(1, 5, n_persons).astype(np.float32),
        },
        "is_married": {
            time_period: rng.integers(0, 2, n_persons).astype(np.float32),
        },
        "own_children_in_household": {
            time_period: rng.integers(0, 3, n_persons).astype(np.float32),
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
            ACS_IMPUTED_VARIABLES + SIPP_IMPUTED_VARIABLES + SCF_IMPUTED_VARIABLES
        )
        assert ALL_SOURCE_VARIABLES == expected

    def test_scf_uprating_map_covers_scf_money_columns(self):
        expected = {
            "employment_income",
            "interest_dividend_income",
            "social_security_pension_income",
            "net_worth",
            "auto_loan_balance",
            "auto_loan_interest",
        }
        assert expected == set(SCF_DONOR_UPRATING_MAP)

    def test_net_worth_total_targets_defined_for_2024(self):
        assert NET_WORTH_TOTAL_TARGETS[2024] == 160e12


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


class TestHouseholdReceiverHelpers:
    def test_household_values_from_data_aggregates_person_arrays(self):
        data = {
            "household_id": {2024: np.array([10, 20])},
            "person_household_id": {2024: np.array([10, 10, 20, 20])},
            "employment_income": {2024: np.array([1, 2, 3, 4], dtype=np.float32)},
            "age": {2024: np.array([30, 31, 40, 41], dtype=np.float32)},
        }

        summed = _household_values_from_data(
            data,
            "employment_income",
            2024,
            data["household_id"][2024],
            data["person_household_id"][2024],
            how="sum",
        )
        first = _household_values_from_data(
            data,
            "age",
            2024,
            data["household_id"][2024],
            data["person_household_id"][2024],
            how="first",
        )

        np.testing.assert_array_equal(summed, np.array([3, 7], dtype=np.float32))
        np.testing.assert_array_equal(first, np.array([30, 40], dtype=np.float32))

    def test_build_household_scf_receiver_uses_household_level_predictors(self):
        data = {
            "household_id": {2024: np.array([10, 20])},
            "person_household_id": {2024: np.array([10, 10, 20, 20])},
            "age": {2024: np.array([30, 31, 40, 41], dtype=np.float32)},
            "is_male": {2024: np.array([1, 0, 0, 0], dtype=np.float32)},
            "cps_race": {2024: np.array([1, 1, 3, 3], dtype=np.float32)},
            "is_married": {2024: np.array([1, 1, 0, 0], dtype=np.float32)},
            "own_children_in_household": {
                2024: np.array([2, 2, 1, 1], dtype=np.float32)
            },
            "employment_income": {
                2024: np.array([10_000, 20_000, 30_000, 40_000], dtype=np.float32)
            },
            "taxable_interest_income": {
                2024: np.array([100, 150, 200, 250], dtype=np.float32)
            },
            "qualified_dividend_income": {
                2024: np.array([50, 50, 60, 60], dtype=np.float32)
            },
            "taxable_private_pension_income": {
                2024: np.array([500, 500, 700, 700], dtype=np.float32)
            },
            "social_security_retirement": {
                2024: np.array([250, 250, 300, 300], dtype=np.float32)
            },
        }

        receiver = _build_household_scf_receiver(data, 2024)

        np.testing.assert_array_equal(receiver["household_id"], np.array([10, 20]))
        np.testing.assert_array_equal(
            receiver["employment_income"],
            np.array([30_000, 70_000], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            receiver["interest_dividend_income"],
            np.array([350, 570], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            receiver["social_security_pension_income"],
            np.array([1_500, 2_000], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            receiver["is_female"],
            np.array([0, 1], dtype=np.float32),
        )


class TestScfDonorUprating:
    def test_align_weighted_total_hits_target(self):
        values = np.array([10.0, 30.0], dtype=np.float32)
        weights = np.array([2.0, 1.0], dtype=np.float32)

        aligned = _align_weighted_total(values, weights, target_total=100.0)

        assert np.isclose(np.dot(aligned, weights), 100.0)

    def test_uprate_scf_donor_frame_noops_same_year(self):
        donor = pd.DataFrame(
            {
                "employment_income": [10_000.0],
                "net_worth": [50_000.0],
                "wgt": [1.0],
            }
        )

        result = _uprate_scf_donor_frame(donor, from_year=2022, to_year=2022)

        pd.testing.assert_frame_equal(result, donor)

    def test_uprate_scf_donor_frame_changes_monetary_columns(self):
        donor = pd.DataFrame(
            {
                "employment_income": [10_000.0],
                "interest_dividend_income": [2_000.0],
                "social_security_pension_income": [3_000.0],
                "net_worth": [50_000.0],
                "auto_loan_balance": [12_000.0],
                "auto_loan_interest": [900.0],
                "age": [55.0],
                "wgt": [1.0],
            }
        )

        result = _uprate_scf_donor_frame(donor, from_year=2022, to_year=2024)

        for column in [
            "employment_income",
            "interest_dividend_income",
            "social_security_pension_income",
            "net_worth",
            "auto_loan_balance",
            "auto_loan_interest",
        ]:
            assert result[column].iloc[0] > donor[column].iloc[0]
        assert result["age"].iloc[0] == donor["age"].iloc[0]
        assert result["wgt"].iloc[0] == donor["wgt"].iloc[0]


class TestSubfunctions:
    def test_impute_acs_exists(self):
        assert callable(_impute_acs)

    def test_impute_sipp_exists(self):
        assert callable(_impute_sipp)

    def test_impute_scf_exists(self):
        assert callable(_impute_scf)
