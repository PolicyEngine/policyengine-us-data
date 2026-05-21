"""Tests for source_impute module.

Uses skip flags to avoid loading real donor data.
"""

import numpy as np
import pandas as pd
import h5py

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
    _impute_org,
    _impute_scf,
    _impute_sipp,
    _person_state_fips,
    impute_source_variables,
)
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
        "primary_residence_value": {time_period: np.zeros(n_persons)},
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
        assert "primary_residence_value" in ACS_IMPUTED_VARIABLES

    def test_sipp_variables_defined(self):
        assert "tip_income" in SIPP_IMPUTED_VARIABLES
        assert "bank_account_assets" in SIPP_IMPUTED_VARIABLES
        assert "stock_assets" in SIPP_IMPUTED_VARIABLES
        assert "bond_assets" in SIPP_IMPUTED_VARIABLES
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
            "primary_residence_value",
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


class TestSubfunctions:
    def test_impute_acs_exists(self):
        assert callable(_impute_acs)

    def test_impute_acs_sets_primary_residence_value_only_for_owner_heads(
        self, monkeypatch, tmp_path
    ):
        import microimpute.models.qrf as qrf_module
        import policyengine_us
        import policyengine_us_data.datasets.acs.acs as acs_module

        fake_acs_path = tmp_path / "acs.h5"
        rows = 10_050
        with h5py.File(fake_acs_path, mode="w") as fake_acs:
            fake_acs.create_dataset(
                "primary_residence_value",
                data=np.full(rows, 300_000, dtype=np.float32),
            )

        class FakeStateValues:
            values = np.ones(rows, dtype=np.float32) * 6

        class FakeMicrosimulation:
            def __init__(self, dataset):
                self.dataset = dataset

            def calculate_dataframe(self, variables, map_to=None, use_weights=True):
                if self.dataset is acs_module.ACS_2022:
                    return pd.DataFrame(
                        {
                            "is_household_head": np.ones(rows, dtype=bool),
                            "age": np.full(rows, 55, dtype=np.float32),
                            "is_male": np.zeros(rows, dtype=bool),
                            "tenure_type": ["OWNED_WITH_MORTGAGE"] * rows,
                            "employment_income": np.full(
                                rows, 75_000, dtype=np.float32
                            ),
                            "self_employment_income": np.zeros(rows, dtype=np.float32),
                            "social_security": np.zeros(rows, dtype=np.float32),
                            "pension_income": np.zeros(rows, dtype=np.float32),
                            "household_size": np.full(rows, 2, dtype=np.float32),
                            "rent": np.zeros(rows, dtype=np.float32),
                            "real_estate_taxes": np.full(rows, 4_000, dtype=np.float32),
                        }
                    )
                return pd.DataFrame(
                    {
                        "is_household_head": [True, False, True],
                        "age": [55, 53, 31],
                        "is_male": [True, False, False],
                        "tenure_type": [
                            "OWNED_WITH_MORTGAGE",
                            "OWNED_WITH_MORTGAGE",
                            "RENTED",
                        ],
                        "employment_income": [80_000, 30_000, 45_000],
                        "self_employment_income": [0, 0, 0],
                        "social_security": [0, 0, 0],
                        "pension_income": [0, 0, 0],
                        "household_size": [2, 2, 1],
                    }
                )

            def calculate(self, variable, map_to=None):
                assert variable == "state_fips"
                return FakeStateValues()

        class FakeQRFModel:
            def predict(self, X_test):
                assert len(X_test) == 2
                return pd.DataFrame(
                    {
                        "rent": [0, 1_200],
                        "real_estate_taxes": [4_000, 0],
                        "primary_residence_value": [500_000, 700_000],
                    }
                )

        class FakeQRF:
            def fit(self, X_train, predictors, imputed_variables):
                assert len(X_train) == 10_000
                assert "primary_residence_value" in X_train
                assert imputed_variables == ACS_IMPUTED_VARIABLES
                return FakeQRFModel()

        monkeypatch.setattr(acs_module.ACS_2022, "file_path", fake_acs_path)
        monkeypatch.setattr(policyengine_us, "Microsimulation", FakeMicrosimulation)
        monkeypatch.setattr(qrf_module, "QRF", FakeQRF)

        data = {
            "person_id": {2024: np.arange(3)},
            "household_id": {2024: np.array([0, 1])},
            "person_household_id": {2024: np.array([0, 0, 1])},
        }

        result = _impute_acs(
            data,
            state_fips=np.array([6, 48], dtype=np.int32),
            time_period=2024,
            dataset_path="fake-cps.h5",
        )

        np.testing.assert_array_equal(
            result["rent"][2024],
            np.array([0, 0, 1_200], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result["real_estate_taxes"][2024],
            np.array([4_000, 0, 0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result["primary_residence_value"][2024],
            np.array([500_000, 0, 0], dtype=np.float32),
        )

    def test_impute_sipp_exists(self):
        assert callable(_impute_sipp)

    def test_impute_org_exists(self):
        assert callable(_impute_org)

    def test_impute_scf_exists(self):
        assert callable(_impute_scf)


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
