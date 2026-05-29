"""Tests for puf_impute module.

Verifies PUF clone + QRF imputation logic using mock data
so tests don't require real CPS/PUF datasets.
"""

import numpy as np
import pandas as pd

from policyengine_us_data.calibration import puf_impute as puf_impute_module
from policyengine_us_data.calibration.puf_impute import (
    DEMOGRAPHIC_PREDICTORS,
    IMPUTED_VARIABLES,
    OVERRIDDEN_IMPUTED_VARIABLES,
    _forbes_person_training_mask,
    _impute_retirement_contributions,
    _impute_weeks_unemployed,
    _log_stratified_subsample,
    _run_qrf_imputation,
    _stratified_subsample_index,
    puf_clone_dataset,
)
from policyengine_us_data.datasets.puf.variable_roles import (
    PUF_REPORTED_CALCULATED_TAX_OUTPUT_VARIABLES,
    PUF_SOURCE_VARIABLE_ROLES,
    REPORTED_CALCULATED_TAX_OUTPUT_ROLE,
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
        "family_id": {time_period: np.arange(1, n_households + 1)},
        "person_household_id": {time_period: household_ids_person},
        "person_tax_unit_id": {time_period: tax_unit_ids_person},
        "person_spm_unit_id": {time_period: spm_unit_ids_person},
        "person_family_id": {time_period: household_ids_person},
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

    def test_geography_fields_preserved(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])
        block_geoid = np.array(
            [
                "010010001001001",
                "020010001001001",
                "360610001001000",
                "060010001001001",
                "480010001001001",
            ]
        )
        cd_geoid = np.array(["101", "202", "3610", "601", "4801"])
        county_fips = np.array(["01001", "02001", "36061", "06001", "48001"])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            block_geoid=block_geoid,
            cd_geoid=cd_geoid,
            county_fips=county_fips,
            time_period=2024,
            skip_qrf=True,
        )

        np.testing.assert_array_equal(
            result["block_geoid"][2024][:5].astype(str),
            block_geoid,
        )
        np.testing.assert_array_equal(
            result["block_geoid"][2024][5:].astype(str),
            block_geoid,
        )
        np.testing.assert_array_equal(
            result["congressional_district_geoid"][2024][:5],
            cd_geoid.astype(np.int32),
        )
        np.testing.assert_array_equal(
            result["county_fips"][2024][:5].astype(str),
            county_fips,
        )
        np.testing.assert_array_equal(
            result["tract_geoid"][2024][:5].astype(str),
            np.array([b[:11] for b in block_geoid]),
        )

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

    def test_reported_calculated_tax_outputs_not_imputed(self):
        expected = {
            "taxable_unemployment_compensation",
            "foreign_tax_credit",
            "american_opportunity_credit",
            "general_business_credit",
            "energy_efficient_home_improvement_credit",
            "amt_foreign_tax_credit",
            "excess_withheld_payroll_tax",
            "savers_credit",
            "early_withdrawal_penalty",
            "prior_year_minimum_tax_credit",
            "other_credits",
            "unreported_payroll_tax",
            "recapture_of_investment_credit",
        }
        blocked = PUF_REPORTED_CALCULATED_TAX_OUTPUT_VARIABLES
        assert blocked == expected
        assert all(
            PUF_SOURCE_VARIABLE_ROLES[var] == REPORTED_CALCULATED_TAX_OUTPUT_ROLE
            for var in blocked
        )
        assert blocked.isdisjoint(IMPUTED_VARIABLES)
        assert blocked.isdisjoint(OVERRIDDEN_IMPUTED_VARIABLES)

    def test_reported_calculated_tax_outputs_not_emitted(self, monkeypatch):
        data = _make_mock_data(n_persons=20, n_households=5)
        data["general_business_credit"] = {2024: np.ones(20, dtype=np.float32)}
        y_full = {var: np.ones(20, dtype=np.float32) for var in IMPUTED_VARIABLES}
        y_full.update(
            {
                var: np.ones(20, dtype=np.float32)
                for var in PUF_REPORTED_CALCULATED_TAX_OUTPUT_VARIABLES
            }
        )

        def fake_run_qrf_imputation(*args, **kwargs):
            return y_full, {}

        monkeypatch.setattr(
            puf_impute_module,
            "_run_qrf_imputation",
            fake_run_qrf_imputation,
        )

        result = puf_clone_dataset(
            data=data,
            state_fips=np.array([1, 2, 36, 6, 48]),
            time_period=2024,
            puf_dataset=object(),
            skip_qrf=False,
        )

        for var in PUF_REPORTED_CALCULATED_TAX_OUTPUT_VARIABLES:
            assert var not in result

    def test_puf_only_variables_are_imputed_onto_cps_half(self, monkeypatch):
        data = _make_mock_data(n_persons=20, n_households=5)
        assert "partnership_s_corp_income" not in data

        predictions = np.arange(20, dtype=np.float32) + 100
        y_full = {var: np.ones(20, dtype=np.float32) for var in IMPUTED_VARIABLES}
        y_full["partnership_s_corp_income"] = predictions
        y_full["employment_income"] = np.full(20, 999_999, dtype=np.float32)

        def fake_run_qrf_imputation(*args, **kwargs):
            return y_full, {}

        monkeypatch.setattr(
            puf_impute_module,
            "_run_qrf_imputation",
            fake_run_qrf_imputation,
        )

        result = puf_clone_dataset(
            data=data,
            state_fips=np.array([1, 2, 36, 6, 48]),
            time_period=2024,
            puf_dataset=object(),
            skip_qrf=False,
        )

        partnership = result["partnership_s_corp_income"][2024]
        np.testing.assert_array_equal(partnership[:20], predictions)
        np.testing.assert_array_equal(partnership[20:], predictions)

        employment = result["employment_income"][2024]
        np.testing.assert_array_equal(employment[:20], data["employment_income"][2024])
        np.testing.assert_array_equal(employment[20:], y_full["employment_income"])

    def test_sstb_qbi_split_variables_imputed(self):
        expected = {
            "sstb_self_employment_income",
            "sstb_self_employment_income_would_be_qualified",
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

    def test_clone_origin_flags_are_added(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        expected_lengths = {
            "person_is_puf_clone": 20,
            "tax_unit_is_puf_clone": 5,
            "spm_unit_is_puf_clone": 5,
            "family_is_puf_clone": 5,
            "household_is_puf_clone": 5,
        }

        for variable_name, half_length in expected_lengths.items():
            values = result[variable_name][2024]
            assert values.dtype == np.int8
            np.testing.assert_array_equal(values[:half_length], 0)
            np.testing.assert_array_equal(values[half_length:], 1)


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


class TestForbesTrainingExclusion:
    def test_maps_forbes_tax_units_to_person_records(self):
        data = {
            "tax_unit_id": {2024: np.array([10, 20, 30])},
            "person_tax_unit_id": {2024: np.array([10, 20, 20, 30])},
            "forbes_unit_id": {2024: np.array([-1, 0, -1])},
            "forbes_replicate_id": {2024: np.array([-1, 3, -1])},
            "forbes_rank": {2024: np.array([0, 42, 0])},
        }

        result = _forbes_person_training_mask(data, 2024, n_persons=4)

        np.testing.assert_array_equal(
            result,
            np.array([False, True, True, False]),
        )

    def test_missing_forbes_metadata_keeps_all_records(self):
        data = {
            "tax_unit_id": {2024: np.array([10, 20])},
            "person_tax_unit_id": {2024: np.array([10, 20])},
        }

        result = _forbes_person_training_mask(data, 2024, n_persons=2)

        np.testing.assert_array_equal(result, np.array([False, False]))

    def test_qrf_training_filters_forbes_person_records(self, monkeypatch):
        class FakeDataset:
            def load_dataset(self):
                return {
                    "tax_unit_id": {2024: np.array([10, 20, 30])},
                    "person_tax_unit_id": {2024: np.array([10, 20, 30, 30])},
                    "forbes_unit_id": {2024: np.array([-1, 0, -1])},
                    "forbes_rank": {2024: np.array([0, 1, 0])},
                    "forbes_replicate_id": {2024: np.array([-1, 0, -1])},
                }

        class FakeMicrosimulation:
            def __init__(self, dataset):
                self.dataset = FakeDataset()

            def calculate(self, variable, map_to=None):
                assert map_to == "person"
                assert variable == "adjusted_gross_income"
                return pd.Series([10.0, 30_000_000.0, 20.0, 30.0])

            def calculate_dataframe(self, columns):
                frame = pd.DataFrame({"age": [40.0, 99.0, 50.0, 55.0]})
                for column in columns:
                    if column not in frame:
                        frame[column] = 0.0
                return frame[list(columns)]

        train_frames = []

        def fake_sequential_qrf(X_train, X_test, predictors, output_vars):
            train_frames.append(X_train.copy())
            return {variable: np.zeros(len(X_test)) for variable in output_vars}

        monkeypatch.setattr("policyengine_us.Microsimulation", FakeMicrosimulation)
        monkeypatch.setattr(
            puf_impute_module,
            "_sequential_qrf",
            fake_sequential_qrf,
        )

        data = {
            predictor: {2024: np.array([0.0, 1.0])}
            for predictor in DEMOGRAPHIC_PREDICTORS
        }
        _run_qrf_imputation(
            data=data,
            time_period=2024,
            puf_dataset=object(),
            dataset_path=None,
        )

        assert len(train_frames) == 2
        assert all(len(frame) == 3 for frame in train_frames)
        assert all(99.0 not in set(frame["age"]) for frame in train_frames)

    def test_qrf_training_filters_synthetic_top_tail_without_metadata(
        self, monkeypatch
    ):
        class FakeDataset:
            def load_dataset(self):
                return {
                    "tax_unit_id": {2024: np.array([10, 20, 30])},
                    "person_tax_unit_id": {2024: np.array([10, 20, 30, 30])},
                }

        class FakeMicrosimulation:
            def __init__(self, dataset):
                self.dataset = FakeDataset()

            def calculate(self, variable, map_to=None):
                assert map_to == "person"
                if variable == "person_weight":
                    return pd.Series([100.0, 0.13, 100.0, 100.0])
                assert variable == "adjusted_gross_income"
                return pd.Series([10.0, 1_000_000_000.0, 20.0, 30.0])

            def calculate_dataframe(self, columns):
                frame = pd.DataFrame({"age": [40.0, 99.0, 50.0, 55.0]})
                for column in columns:
                    if column not in frame:
                        frame[column] = 0.0
                return frame[list(columns)]

        train_frames = []

        def fake_sequential_qrf(X_train, X_test, predictors, output_vars):
            train_frames.append(X_train.copy())
            return {variable: np.zeros(len(X_test)) for variable in output_vars}

        monkeypatch.setattr("policyengine_us.Microsimulation", FakeMicrosimulation)
        monkeypatch.setattr(
            puf_impute_module,
            "_sequential_qrf",
            fake_sequential_qrf,
        )

        data = {
            predictor: {2024: np.array([0.0, 1.0])}
            for predictor in DEMOGRAPHIC_PREDICTORS
        }
        _run_qrf_imputation(
            data=data,
            time_period=2024,
            puf_dataset=object(),
            dataset_path=None,
        )

        assert len(train_frames) == 2
        assert all(len(frame) == 3 for frame in train_frames)
        assert all(99.0 not in set(frame["age"]) for frame in train_frames)

    def test_qrf_training_filters_synthetic_top_tail_components_without_metadata(
        self, monkeypatch
    ):
        class FakeDataset:
            def load_dataset(self):
                return {
                    "tax_unit_id": {2024: np.array([10, 20, 30])},
                    "person_tax_unit_id": {2024: np.array([10, 20, 30, 30])},
                }

        class FakeMicrosimulation:
            def __init__(self, dataset):
                self.dataset = FakeDataset()

            def calculate(self, variable, map_to=None):
                assert map_to == "person"
                assert variable == "adjusted_gross_income"
                return pd.Series([10.0, 20.0, 30.0, 40.0])

            def calculate_dataframe(self, columns):
                frame = pd.DataFrame(
                    {
                        "age": [40.0, 99.0, 50.0, 55.0],
                        "long_term_capital_gains": [
                            0.0,
                            30_000_000.0,
                            0.0,
                            0.0,
                        ],
                    }
                )
                for column in columns:
                    if column not in frame:
                        frame[column] = 0.0
                return frame[list(columns)]

        train_frames = []

        def fake_sequential_qrf(X_train, X_test, predictors, output_vars):
            train_frames.append(X_train.copy())
            return {variable: np.zeros(len(X_test)) for variable in output_vars}

        monkeypatch.setattr("policyengine_us.Microsimulation", FakeMicrosimulation)
        monkeypatch.setattr(
            puf_impute_module,
            "_sequential_qrf",
            fake_sequential_qrf,
        )

        data = {
            predictor: {2024: np.array([0.0, 1.0])}
            for predictor in DEMOGRAPHIC_PREDICTORS
        }
        _run_qrf_imputation(
            data=data,
            time_period=2024,
            puf_dataset=object(),
            dataset_path=None,
        )

        assert len(train_frames) == 2
        assert all(len(frame) == 3 for frame in train_frames)
        assert all(99.0 not in set(frame["age"]) for frame in train_frames)

    def test_qrf_training_filters_normal_weight_top_tail_without_metadata(
        self, monkeypatch
    ):
        class FakeDataset:
            def load_dataset(self):
                return {
                    "tax_unit_id": {2024: np.array([10, 20, 30])},
                    "person_tax_unit_id": {2024: np.array([10, 20, 30, 30])},
                }

        class FakeMicrosimulation:
            def __init__(self, dataset):
                self.dataset = FakeDataset()

            def calculate(self, variable, map_to=None):
                assert map_to == "person"
                assert variable == "adjusted_gross_income"
                return pd.Series([10.0, 30_000_000.0, 20.0, 30.0])

            def calculate_dataframe(self, columns):
                frame = pd.DataFrame({"age": [40.0, 99.0, 50.0, 55.0]})
                for column in columns:
                    if column not in frame:
                        frame[column] = 0.0
                return frame[list(columns)]

        train_frames = []

        def fake_sequential_qrf(X_train, X_test, predictors, output_vars):
            train_frames.append(X_train.copy())
            return {variable: np.zeros(len(X_test)) for variable in output_vars}

        monkeypatch.setattr("policyengine_us.Microsimulation", FakeMicrosimulation)
        monkeypatch.setattr(
            puf_impute_module,
            "_sequential_qrf",
            fake_sequential_qrf,
        )

        data = {
            predictor: {2024: np.array([0.0, 1.0])}
            for predictor in DEMOGRAPHIC_PREDICTORS
        }
        _run_qrf_imputation(
            data=data,
            time_period=2024,
            puf_dataset=object(),
            dataset_path=None,
        )

        assert len(train_frames) == 2
        assert all(len(frame) == 3 for frame in train_frames)
        assert all(99.0 not in set(frame["age"]) for frame in train_frames)

    def test_qrf_training_filters_all_default_metadata_top_tail(self, monkeypatch):
        class FakeDataset:
            def load_dataset(self):
                return {
                    "tax_unit_id": {2024: np.array([10, 20, 30])},
                    "person_tax_unit_id": {2024: np.array([10, 20, 30, 30])},
                    "forbes_unit_id": {2024: np.array([-1, -1, -1])},
                    "forbes_rank": {2024: np.array([0, 0, 0])},
                    "forbes_replicate_id": {2024: np.array([-1, -1, -1])},
                }

        class FakeMicrosimulation:
            def __init__(self, dataset):
                self.dataset = FakeDataset()

            def calculate(self, variable, map_to=None):
                assert map_to == "person"
                assert variable == "adjusted_gross_income"
                return pd.Series([10.0, 30_000_000.0, 20.0, 30.0])

            def calculate_dataframe(self, columns):
                frame = pd.DataFrame({"age": [40.0, 99.0, 50.0, 55.0]})
                for column in columns:
                    if column not in frame:
                        frame[column] = 0.0
                return frame[list(columns)]

        train_frames = []

        def fake_sequential_qrf(X_train, X_test, predictors, output_vars):
            train_frames.append(X_train.copy())
            return {variable: np.zeros(len(X_test)) for variable in output_vars}

        monkeypatch.setattr("policyengine_us.Microsimulation", FakeMicrosimulation)
        monkeypatch.setattr(
            puf_impute_module,
            "_sequential_qrf",
            fake_sequential_qrf,
        )

        data = {
            predictor: {2024: np.array([0.0, 1.0])}
            for predictor in DEMOGRAPHIC_PREDICTORS
        }
        _run_qrf_imputation(
            data=data,
            time_period=2024,
            puf_dataset=object(),
            dataset_path=None,
        )

        assert len(train_frames) == 2
        assert all(len(frame) == 3 for frame in train_frames)
        assert all(99.0 not in set(frame["age"]) for frame in train_frames)

    def test_qrf_training_keeps_non_forbes_top_tail_with_metadata(self, monkeypatch):
        class FakeDataset:
            def load_dataset(self):
                return {
                    "tax_unit_id": {2024: np.array([10, 20, 30])},
                    "person_tax_unit_id": {2024: np.array([10, 20, 30, 30])},
                    "forbes_unit_id": {2024: np.array([-1, -1, 0])},
                    "forbes_rank": {2024: np.array([0, 0, 1])},
                    "forbes_replicate_id": {2024: np.array([-1, -1, 0])},
                }

        class FakeMicrosimulation:
            def __init__(self, dataset):
                self.dataset = FakeDataset()

            def calculate(self, variable, map_to=None):
                assert map_to == "person"
                assert variable == "adjusted_gross_income"
                return pd.Series([10.0, 20.0, 30.0, 40.0])

            def calculate_dataframe(self, columns):
                frame = pd.DataFrame(
                    {
                        "age": [40.0, 99.0, 50.0, 55.0],
                        "long_term_capital_gains": [
                            0.0,
                            30_000_000.0,
                            0.0,
                            0.0,
                        ],
                    }
                )
                for column in columns:
                    if column not in frame:
                        frame[column] = 0.0
                return frame[list(columns)]

        train_frames = []

        def fake_sequential_qrf(X_train, X_test, predictors, output_vars):
            train_frames.append(X_train.copy())
            return {variable: np.zeros(len(X_test)) for variable in output_vars}

        monkeypatch.setattr("policyengine_us.Microsimulation", FakeMicrosimulation)
        monkeypatch.setattr(
            puf_impute_module,
            "_sequential_qrf",
            fake_sequential_qrf,
        )

        data = {
            predictor: {2024: np.array([0.0, 1.0])}
            for predictor in DEMOGRAPHIC_PREDICTORS
        }
        _run_qrf_imputation(
            data=data,
            time_period=2024,
            puf_dataset=object(),
            dataset_path=None,
        )

        assert len(train_frames) == 2
        assert all(len(frame) == 2 for frame in train_frames)
        assert all(99.0 in set(frame["age"]) for frame in train_frames)


def test_retirement_imputation_uses_sstb_income_for_se_eligibility(monkeypatch):
    class FakeMicrosimulation:
        def __init__(self, dataset):
            self.dataset = dataset

        def calculate_dataframe(self, columns):
            if "self_employed_pension_contributions_desired" in columns:
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
                        "traditional_401k_contributions_desired": [0.0, 0.0],
                        "roth_401k_contributions_desired": [0.0, 0.0],
                        "traditional_ira_contributions_desired": [0.0, 0.0],
                        "roth_ira_contributions_desired": [0.0, 0.0],
                        "self_employed_pension_contributions_desired": [0.0, 0.0],
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
                    "traditional_401k_contributions_desired": [0.0, 0.0],
                    "roth_401k_contributions_desired": [0.0, 0.0],
                    "traditional_ira_contributions_desired": [0.0, 0.0],
                    "roth_ira_contributions_desired": [0.0, 0.0],
                    "self_employed_pension_contributions_desired": [50_000.0, 50_000.0],
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
        result["self_employed_pension_contributions_desired"],
        np.array([50_000.0, 50_000.0]),
    )


def test_weeks_imputation_uses_unemployment_compensation_input(monkeypatch):
    class FakeMicrosimulation:
        def __init__(self, dataset):
            self.dataset = dataset

        def calculate(self, variable):
            if variable == "weeks_unemployed":
                return pd.Series([0.0, 12.0, 0.0])
            if variable == "unemployment_compensation":
                return pd.Series([0.0, 500.0, 0.0])
            raise ValueError(variable)

        def calculate_dataframe(self, columns):
            return pd.DataFrame({column: [0.0, 1.0, 0.0] for column in columns})

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
            assert "unemployment_compensation" in predictors
            assert "unemployment_compensation" in X_train
            np.testing.assert_array_equal(
                X_test["unemployment_compensation"].to_numpy(),
                np.array([0.0, 100.0, 0.0]),
            )
            return pd.DataFrame({"weeks_unemployed": [5.0, 6.0, 7.0]})

    monkeypatch.setattr("policyengine_us.Microsimulation", FakeMicrosimulation)
    monkeypatch.setattr("microimpute.models.qrf.QRF", FakeQRF)

    result = _impute_weeks_unemployed(
        data={
            "person_id": {2024: np.array([1, 2, 3])},
            "unemployment_compensation": {2024: np.array([0.0, 100.0, 0.0])},
        },
        puf_imputations={},
        time_period=2024,
        dataset_path="ignored.h5",
    )

    np.testing.assert_array_equal(result, np.array([0.0, 6.0, 0.0]))


def test_log_handles_grouped_currency_threshold(caplog):
    threshold = np.float32(8.934329e7)
    caplog.set_level(
        "INFO",
        logger="policyengine_us_data.calibration.puf_impute",
    )

    _log_stratified_subsample(484_015, 20_000, 0.5, threshold)

    assert "Stratified PUF subsample: 484015 -> 20000 records" in caplog.text
    assert f"${threshold:,.0f}" in caplog.text
