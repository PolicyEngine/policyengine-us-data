"""Tests for source_impute module.

Uses skip flags to avoid loading real donor data.
"""

import numpy as np
import pandas as pd
import huggingface_hub
import pytest

from policyengine_us_data.calibration import source_impute
from policyengine_us_data.calibration.source_impute import (
    ACS_IMPUTED_VARIABLES,
    ACS_PREDICTORS,
    ACS_TARGET_ALLOCATION_COLUMNS,
    ALL_SOURCE_VARIABLES,
    SCF_IMPUTED_VARIABLES,
    SCF_PREDICTORS,
    SIPP_ASSETS_PREDICTORS,
    SIPP_IMPUTED_VARIABLES,
    SSI_DISABILITY_EXPORT_VARIABLES,
    SOURCE_IMPUTATION_CONSTRUCTION_ONLY_VARIABLES,
    SIPP_TIPS_PREDICTORS,
    _add_cps_asset_predictors,
    _impute_acs,
    _impute_org,
    _impute_scf,
    _impute_sipp,
    _person_is_married,
    _person_state_fips,
    drop_source_imputation_construction_variables,
    impute_source_variables,
)
from policyengine_us_data.datasets.sipp.sipp import (
    ASSET_PREDICTORS,
    preserve_under_65_ssi_disability_criteria,
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
        assert ACS_TARGET_ALLOCATION_COLUMNS == {
            "rent": ["rent_is_allocated"],
            "real_estate_taxes": ["real_estate_taxes_is_allocated"],
        }

    def test_sipp_variables_defined(self):
        assert "tip_income" in SIPP_IMPUTED_VARIABLES
        assert "bank_account_assets" in SIPP_IMPUTED_VARIABLES
        assert "stock_assets" in SIPP_IMPUTED_VARIABLES
        assert "bond_assets" in SIPP_IMPUTED_VARIABLES
        assert set(SSI_DISABILITY_EXPORT_VARIABLES) <= set(SIPP_IMPUTED_VARIABLES)
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

    def test_source_impute_construction_only_variables_defined(self):
        assert "difficulty_hearing" in SOURCE_IMPUTATION_CONSTRUCTION_ONLY_VARIABLES


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

    def test_drop_source_imputation_construction_variables_removes_difficulty_flags(
        self,
    ):
        data = {
            "difficulty_hearing": {2024: np.array([True, False])},
            "meets_ssi_disability_criteria": {2024: np.array([True, False])},
        }

        result = drop_source_imputation_construction_variables(data)

        assert "difficulty_hearing" not in result
        assert "meets_ssi_disability_criteria" in result


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

    def test_calibration_sipp_tip_counts_use_reference_month(self, monkeypatch):
        captured = {}

        columns = {
            "SSUID": [1, 1, 1, 2],
            "MONTHCODE": [1, 12, 12, 12],
            "TAGE": [5, 40, 10, 30],
            "TPTOTINC": [1_000.0, 2_000.0, 0.0, 3_000.0],
            "WPFINWGT": [1.0, 1.0, 1.0, 1.0],
        }
        for column in source_impute.SIPP_TIP_AMOUNT_COLUMNS:
            columns[column] = [0.0, 10.0, 0.0, 5.0]
        for column in source_impute.SIPP_TIP_AMOUNT_TO_ALLOCATION_COLUMN.values():
            columns[column] = [0, 0, 0, 0]
        for column in source_impute.SIPP_JOB_OCCUPATION_COLUMNS:
            columns[column] = [0, 0, 0, 0]
        tip_source = pd.DataFrame(columns)

        read_count = {"count": 0}

        def fake_read_csv(*args, **kwargs):
            read_count["count"] += 1
            if read_count["count"] == 1:
                return tip_source.copy()
            raise FileNotFoundError("stop after tip imputation")

        class FakeQRF:
            def __init__(self, *args, **kwargs):
                captured["init_kwargs"] = kwargs

            def fit(self, X_train, **kwargs):
                captured["train"] = X_train.copy()
                captured["fit_kwargs"] = kwargs
                return self

            def predict(self, X_test):
                return pd.DataFrame({"tip_income": np.zeros(len(X_test))})

        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(source_impute.pd, "read_csv", fake_read_csv)
        monkeypatch.setattr(source_impute, "QRF", FakeQRF)

        data = _make_data_dict(n_persons=4)
        _impute_sipp(
            data=data,
            state_fips=np.array([1, 1], dtype=np.int32),
            time_period=2024,
        )

        household_one = captured["train"][captured["train"]["household_id"] == 1]
        np.testing.assert_array_equal(household_one["count_under_18"], [1, 1])
        np.testing.assert_array_equal(household_one["count_under_6"], [0, 0])
        assert captured["init_kwargs"] == {}
        assert set(captured["fit_kwargs"]["target_filters"]) == {"tip_income"}
        assert len(captured["fit_kwargs"]["target_filters"]["tip_income"]) == len(
            captured["train"]
        )

    def test_calibration_sipp_qrf_passes_target_filters(self, monkeypatch):
        fit_calls = []
        captured_ssi_receiver = {}

        tip_columns = {
            "SSUID": [1, 2, 3],
            "MONTHCODE": [12, 12, 12],
            "TAGE": [40, 30, 10],
            "TPTOTINC": [1_000.0, 2_000.0, 0.0],
            "WPFINWGT": [1.0, 1.0, 1.0],
        }
        for column in source_impute.SIPP_TIP_AMOUNT_COLUMNS:
            tip_columns[column] = [10.0, 5.0, 0.0]
        for column in source_impute.SIPP_TIP_AMOUNT_TO_ALLOCATION_COLUMN.values():
            tip_columns[column] = [1, 2, 0]
        for column in source_impute.SIPP_JOB_OCCUPATION_COLUMNS:
            tip_columns[column] = [0, 0, 0]
        tip_source = pd.DataFrame(tip_columns)

        asset_columns = {
            "SSUID": [1, 2, 3],
            "PNUM": [1, 1, 1],
            "MONTHCODE": [12, 12, 12],
            "WPFINWGT": [1.0, 1.0, 1.0],
            "TAGE": [40, 30, 10],
            "ESEX": [1, 2, 1],
            "EMS": [1, 2, 2],
            "TSSSAMT": [0.0, 0.0, 0.0],
            "TRETINCAMT": [0.0, 0.0, 0.0],
            "TVAL_BANK": [100.0, 200.0, 300.0],
            "TVAL_STMF": [10.0, 20.0, 30.0],
            "TVAL_BOND": [1.0, 2.0, 3.0],
            "TINC_BANK": [0.0, 0.0, 0.0],
            "TINC_STMF": [0.0, 0.0, 0.0],
            "TINC_BOND": [0.0, 0.0, 0.0],
            "TINC_RENT": [0.0, 0.0, 0.0],
        }
        for column in source_impute.ASSET_JOB_EARNINGS_COLUMNS:
            asset_columns[column] = [1_000.0, 2_000.0, 0.0]
        for column in source_impute.SIPP_ASSET_ALLOCATION_COLUMNS:
            asset_columns[column] = [0, 0, 0]
        asset_columns["AJSSAVVAL"] = [0, 2, 0]
        asset_columns["AJSSTVAL"] = [0, 0, 6]
        asset_source = pd.DataFrame(asset_columns)

        vehicle_train = pd.DataFrame(
            {
                **{
                    predictor: [0.0, 1.0, 2.0]
                    for predictor in source_impute.VEHICLE_MODEL_PREDICTORS
                },
                "household_vehicles_owned": [1.0, 2.0, 3.0],
                "household_vehicles_value": [5_000.0, 10_000.0, 15_000.0],
                "AVEH_NUM": [1, 2, 1],
                "AVEH1VAL": [1, 1, 5],
                "AVEH2VAL": [0, 0, 0],
                "AVEH3VAL": [0, 0, 0],
                "household_weight": [1.0, 1.0, 1.0],
            }
        )

        def fake_read_csv(path, *args, **kwargs):
            if str(path).endswith("pu2023_slim.csv"):
                return tip_source.copy()
            if str(path).endswith("pu2023.csv"):
                return asset_source.copy()
            raise AssertionError(f"Unexpected read_csv path: {path}")

        class FakeQRF:
            def __init__(self, *args, **kwargs):
                self.init_kwargs = kwargs

            def fit(self, X_train, **kwargs):
                self.imputed_variables = kwargs["imputed_variables"]
                fit_calls.append(
                    {
                        "init_kwargs": self.init_kwargs,
                        "train": X_train.copy(),
                        "kwargs": kwargs,
                    }
                )
                return self

            def predict(self, X_test):
                return pd.DataFrame(
                    {
                        variable: np.zeros(len(X_test), dtype=np.float32)
                        for variable in self.imputed_variables
                    }
                )

        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(source_impute.pd, "read_csv", fake_read_csv)
        monkeypatch.setattr(source_impute, "QRF", FakeQRF)
        monkeypatch.setattr(
            source_impute,
            "get_ssi_disability_model",
            lambda time_period: object(),
        )

        def fake_predict_ssi_disability_criteria(model, receiver):
            captured_ssi_receiver["receiver"] = receiver.copy()
            return np.zeros(len(receiver), dtype=bool)

        monkeypatch.setattr(
            source_impute,
            "predict_ssi_disability_criteria",
            fake_predict_ssi_disability_criteria,
        )
        monkeypatch.setattr(
            source_impute,
            "build_vehicle_training_frame",
            lambda: vehicle_train.copy(),
        )

        data = _make_data_dict(n_persons=6)
        data["difficulty_hearing"] = {
            2024: np.array([False, True, False, False, True, False])
        }

        _impute_sipp(
            data=data,
            state_fips=np.array([1, 1, 1], dtype=np.int32),
            time_period=2024,
        )

        by_targets = {
            tuple(call["kwargs"]["imputed_variables"]): call for call in fit_calls
        }
        assert set(by_targets) == {
            ("tip_income",),
            ("bank_account_assets", "stock_assets", "bond_assets"),
            ("household_vehicles_owned", "household_vehicles_value"),
        }
        for call in fit_calls:
            assert call["init_kwargs"] == {}
            filters = call["kwargs"]["target_filters"]
            assert set(filters) == set(call["kwargs"]["imputed_variables"])
            assert all(len(mask) == len(call["train"]) for mask in filters.values())

        tip_filters = by_targets[("tip_income",)]["kwargs"]["target_filters"]
        assert len(by_targets[("tip_income",)]["train"]) == 2
        np.testing.assert_array_equal(tip_filters["tip_income"].values, [True, True])

        asset_filters = by_targets[
            ("bank_account_assets", "stock_assets", "bond_assets")
        ]["kwargs"]["target_filters"]
        np.testing.assert_array_equal(
            asset_filters["bank_account_assets"].values,
            [True, False, True],
        )
        np.testing.assert_array_equal(
            asset_filters["stock_assets"].values,
            [True, True, False],
        )

        vehicle_filters = by_targets[
            ("household_vehicles_owned", "household_vehicles_value")
        ]["kwargs"]["target_filters"]
        np.testing.assert_array_equal(
            vehicle_filters["household_vehicles_owned"].values,
            [True, False, True],
        )
        np.testing.assert_array_equal(
            vehicle_filters["household_vehicles_value"].values,
            [True, True, False],
        )
        np.testing.assert_array_equal(
            captured_ssi_receiver["receiver"]["difficulty_hearing"],
            [False, True, False, False, True, False],
        )

    def test_calibration_sipp_tip_requires_allocation_flags(self, monkeypatch):
        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            source_impute.pd,
            "read_csv",
            lambda *args, **kwargs: pd.DataFrame({"TJB1_TXAMT": [10.0]}),
        )

        with pytest.raises(KeyError, match="AJB1_TXAMT"):
            _impute_sipp(
                data=_make_data_dict(n_persons=4),
                state_fips=np.array([1, 1], dtype=np.int32),
                time_period=2024,
            )

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
