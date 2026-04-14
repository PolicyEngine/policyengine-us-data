"""Tests for puf_impute module.

Verifies PUF clone + QRF imputation logic using mock data
so tests don't require real CPS/PUF datasets.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from policyengine_us_data.calibration.puf_impute import (
    DEMOGRAPHIC_PREDICTORS,
    IMPUTED_VARIABLES,
    OVERRIDDEN_IMPUTED_VARIABLES,
    SELF_EMPLOYMENT_QRF_WINSOR_LOWER_PERCENTILE,
    SELF_EMPLOYMENT_QRF_WINSOR_UPPER_PERCENTILE,
    _impute_retirement_contributions,
    _log_stratified_subsample,
    _stratified_subsample_index,
    puf_clone_dataset,
)


def _make_mock_data(n_persons=20, n_households=5, time_period=2024):
    """Build a minimal mock CPS data dict."""
    person_ids = np.arange(1, n_persons + 1)
    household_ids_person = np.repeat(
        np.arange(1, n_households + 1),
        n_persons // n_households,
    )
    tax_unit_ids_person = household_ids_person.copy()
    spm_unit_ids_person = household_ids_person.copy()

    rng = np.random.default_rng(42)
    ages = rng.integers(18, 80, size=n_persons)
    is_male = rng.integers(0, 2, size=n_persons)

    data = {
        "person_id": {time_period: person_ids},
        "household_id": {time_period: np.arange(1, n_households + 1)},
        "tax_unit_id": {time_period: np.arange(1, n_households + 1)},
        "spm_unit_id": {time_period: np.arange(1, n_households + 1)},
        "person_household_id": {time_period: household_ids_person},
        "person_tax_unit_id": {time_period: tax_unit_ids_person},
        "person_spm_unit_id": {time_period: spm_unit_ids_person},
        "age": {time_period: ages.astype(np.float32)},
        "is_male": {time_period: is_male.astype(np.float32)},
        "household_weight": {time_period: np.ones(n_households) * 1000},
        "employment_income": {
            time_period: rng.uniform(0, 100000, n_persons).astype(np.float32)
        },
    }
    return data


class TestPufCloneDataset:
    def test_doubles_records(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        assert len(result["household_id"][2024]) == 10
        assert len(result["person_id"][2024]) == 40

    def test_ids_are_unique(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        person_ids = result["person_id"][2024]
        household_ids = result["household_id"][2024]
        assert len(np.unique(person_ids)) == len(person_ids)
        assert len(np.unique(household_ids)) == len(household_ids)

    def test_string_id_like_variables_are_duplicated_without_numeric_offset(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        data["taxpayer_id_type"] = {
            2024: np.array([b"VALID_SSN", b"NONE"] * 10, dtype="S9")
        }
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        values = result["taxpayer_id_type"][2024]
        n = len(values) // 2
        np.testing.assert_array_equal(values[:n], values[n:])

    def test_puf_half_weight_zero(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        weights = result["household_weight"][2024]
        assert np.all(weights[:5] > 0)
        assert np.all(weights[5:] == 0)

    def test_state_fips_preserved(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        result_states = result["state_fips"][2024]
        np.testing.assert_array_equal(result_states[:5], state_fips)
        np.testing.assert_array_equal(result_states[5:], state_fips)

    def test_demographics_shared(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        ages = result["age"][2024]
        n = len(ages) // 2
        np.testing.assert_array_equal(ages[:n], ages[n:])

    def test_demographic_predictors_excludes_state(self):
        # PUF has no state identifier, so state_fips must not
        # be a predictor for PUF imputation.
        assert "state_fips" not in DEMOGRAPHIC_PREDICTORS

    def test_imputed_variables_not_empty(self):
        assert len(IMPUTED_VARIABLES) > 0

    def test_overridden_subset_of_imputed(self):
        for var in OVERRIDDEN_IMPUTED_VARIABLES:
            assert var in IMPUTED_VARIABLES

    def test_sstb_qbi_split_variables_imputed(self):
        expected = {
            "sstb_self_employment_income",
            "sstb_w2_wages_from_qualified_business",
            "sstb_unadjusted_basis_qualified_property",
        }
        for var in expected:
            assert var in IMPUTED_VARIABLES

    def test_sstb_allocable_wage_and_ubia_are_overridden(self):
        expected = {
            "sstb_w2_wages_from_qualified_business",
            "sstb_unadjusted_basis_qualified_property",
        }
        for var in expected:
            assert var in OVERRIDDEN_IMPUTED_VARIABLES

    def test_self_employment_qrf_outliers_are_clipped_before_publish(self):
        n = 20
        data = _make_mock_data(n_persons=n, n_households=5)
        rng = np.random.default_rng(1)
        for predictor in DEMOGRAPHIC_PREDICTORS:
            if predictor not in data:
                data[predictor] = {2024: rng.integers(0, 2, n).astype(np.float32)}
        data["self_employment_income"] = {
            2024: np.zeros(n, dtype=np.float32),
        }

        train = pd.DataFrame(
            {
                predictor: rng.integers(0, 2, n).astype(float)
                for predictor in DEMOGRAPHIC_PREDICTORS
            }
        )
        train["adjusted_gross_income"] = np.linspace(0, 100_000, n)
        train["self_employment_income"] = np.linspace(-2_000, 50_000, n)
        for variable in IMPUTED_VARIABLES:
            if variable not in train:
                train[variable] = 0.0
        train_override = train[
            DEMOGRAPHIC_PREDICTORS + OVERRIDDEN_IMPUTED_VARIABLES
        ].copy()

        lower, upper = np.percentile(
            train["self_employment_income"],
            [
                SELF_EMPLOYMENT_QRF_WINSOR_LOWER_PERCENTILE,
                SELF_EMPLOYMENT_QRF_WINSOR_UPPER_PERCENTILE,
            ],
        )

        puf_sim = MagicMock()
        puf_sim.calculate.return_value.values = train["adjusted_gross_income"].values

        def calculate_dataframe(columns):
            if set(columns) == set(DEMOGRAPHIC_PREDICTORS + IMPUTED_VARIABLES):
                return train[columns].copy()
            return train_override[columns].copy()

        puf_sim.calculate_dataframe.side_effect = calculate_dataframe

        qrf_instance = MagicMock()

        def fit_predict(
            X_train,
            X_test,
            predictors,
            imputed_variables,
            n_jobs,
        ):
            predictions = pd.DataFrame(
                {
                    variable: np.zeros(len(X_test), dtype=np.float32)
                    for variable in imputed_variables
                }
            )
            if "self_employment_income" in imputed_variables:
                predictions["self_employment_income"] = np.linspace(
                    -15_000_000,
                    15_000_000,
                    len(X_test),
                    dtype=np.float32,
                )
            return predictions

        qrf_instance.fit_predict.side_effect = fit_predict

        with (
            patch("policyengine_us.Microsimulation", return_value=puf_sim),
            patch("microimpute.models.qrf.QRF", return_value=qrf_instance),
        ):
            result = puf_clone_dataset(
                data=data,
                state_fips=np.array([1, 2, 36, 6, 48]),
                time_period=2024,
                puf_dataset="mock-puf",
                skip_qrf=False,
            )

        puf_half = result["self_employment_income"][2024][n:]
        assert puf_half.min() >= lower
        assert puf_half.max() <= upper


class TestStratifiedSubsample:
    def test_noop_when_small(self):
        income = np.random.default_rng(0).normal(50000, 20000, size=100)
        idx = _stratified_subsample_index(income, target_n=200)
        assert len(idx) == 100

    def test_reduces_to_target(self):
        rng = np.random.default_rng(0)
        income = np.concatenate(
            [
                rng.normal(50000, 20000, size=50_000),
                rng.uniform(500_000, 5_000_000, size=250),
            ]
        )
        idx = _stratified_subsample_index(income, target_n=10_000, top_pct=99.5)
        assert len(idx) == 10_000

    def test_preserves_top_earners(self):
        rng = np.random.default_rng(0)
        income = np.concatenate(
            [
                rng.normal(50000, 20000, size=50_000),
                rng.uniform(500_000, 5_000_000, size=250),
            ]
        )
        threshold = np.percentile(income, 99.5)
        n_top = (income >= threshold).sum()

        idx = _stratified_subsample_index(income, target_n=10_000, top_pct=99.5)
        selected_income = income[idx]
        n_top_selected = (selected_income >= threshold).sum()
        assert n_top_selected == n_top

    def test_indices_sorted(self):
        income = np.random.default_rng(0).normal(50000, 20000, size=50_000)
        idx = _stratified_subsample_index(income, target_n=10_000)
        assert np.all(idx[1:] >= idx[:-1])


def test_retirement_imputation_caps_se_pension_using_sstb_income(monkeypatch):
    class FakeMicrosimulation:
        def __init__(self, dataset):
            self.dataset = dataset

        def calculate_dataframe(self, columns):
            if "self_employed_pension_contributions" in columns:
                return pd.DataFrame(
                    {
                        "age": [40, 55],
                        "is_male": [0, 1],
                        "tax_unit_is_joint": [0, 1],
                        "tax_unit_count_dependents": [0, 1],
                        "is_tax_unit_head": [1, 1],
                        "is_tax_unit_spouse": [0, 0],
                        "is_tax_unit_dependent": [0, 0],
                        "employment_income": [0.0, 0.0],
                        "self_employment_income": [0.0, 100.0],
                        "taxable_interest_income": [0.0, 0.0],
                        "qualified_dividend_income": [0.0, 0.0],
                        "taxable_pension_income": [0.0, 0.0],
                        "social_security": [0.0, 0.0],
                        "traditional_401k_contributions": [0.0, 0.0],
                        "roth_401k_contributions": [0.0, 0.0],
                        "traditional_ira_contributions": [0.0, 0.0],
                        "roth_ira_contributions": [0.0, 0.0],
                        "self_employed_pension_contributions": [0.0, 0.0],
                    }
                )
            return pd.DataFrame(
                {
                    "age": [40, 55],
                    "is_male": [0, 1],
                    "tax_unit_is_joint": [0, 1],
                    "tax_unit_count_dependents": [0, 1],
                    "is_tax_unit_head": [1, 1],
                    "is_tax_unit_spouse": [0, 0],
                    "is_tax_unit_dependent": [0, 0],
                }
            )

        def calculate(self, variable):
            return pd.Series(np.zeros(2))

    class FakeQRF:
        def __init__(self, **kwargs):
            pass

        def fit_predict(
            self,
            X_train,
            X_test,
            predictors,
            imputed_variables,
            n_jobs,
        ):
            np.testing.assert_array_equal(
                X_test["self_employment_income"].to_numpy(),
                np.array([100.0, 100.0]),
            )
            return pd.DataFrame(
                {
                    "traditional_401k_contributions": [0.0, 0.0],
                    "roth_401k_contributions": [0.0, 0.0],
                    "traditional_ira_contributions": [0.0, 0.0],
                    "roth_ira_contributions": [0.0, 0.0],
                    "self_employed_pension_contributions": [50_000.0, 50_000.0],
                }
            )

    monkeypatch.setattr("policyengine_us.Microsimulation", FakeMicrosimulation)
    monkeypatch.setattr("microimpute.models.qrf.QRF", FakeQRF)

    result = _impute_retirement_contributions(
        data={"person_id": {2024: np.array([1, 2])}},
        puf_imputations={
            "employment_income": np.array([0.0, 0.0]),
            "self_employment_income": np.array([0.0, 100.0]),
            "sstb_self_employment_income": np.array([100.0, 0.0]),
            "taxable_interest_income": np.array([0.0, 0.0]),
            "qualified_dividend_income": np.array([0.0, 0.0]),
            "taxable_pension_income": np.array([0.0, 0.0]),
            "social_security": np.array([0.0, 0.0]),
        },
        time_period=2024,
        dataset_path="ignored.h5",
    )

    np.testing.assert_array_equal(
        result["self_employed_pension_contributions"],
        np.array([25.0, 25.0]),
    )


def test_log_handles_grouped_currency_threshold(caplog):
    threshold = np.float32(8.934329e7)
    caplog.set_level(
        "INFO",
        logger="policyengine_us_data.calibration.puf_impute",
    )

    _log_stratified_subsample(484_015, 20_000, 0.5, threshold)

    assert "Stratified PUF subsample: 484015 -> 20000 records" in caplog.text
    assert f"${threshold:,.0f}" in caplog.text
