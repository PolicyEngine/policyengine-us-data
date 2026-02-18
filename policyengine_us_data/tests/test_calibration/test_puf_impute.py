"""Tests for puf_impute module.

Verifies PUF clone + QRF imputation logic using mock data
so tests don't require real CPS/PUF datasets.
"""

import numpy as np

from policyengine_us_data.calibration.puf_impute import (
    DEMOGRAPHIC_PREDICTORS,
    IMPUTED_VARIABLES,
    OVERRIDDEN_IMPUTED_VARIABLES,
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
        idx = _stratified_subsample_index(
            income, target_n=10_000, top_pct=99.5
        )
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

        idx = _stratified_subsample_index(
            income, target_n=10_000, top_pct=99.5
        )
        selected_income = income[idx]
        n_top_selected = (selected_income >= threshold).sum()
        assert n_top_selected == n_top

    def test_indices_sorted(self):
        income = np.random.default_rng(0).normal(50000, 20000, size=50_000)
        idx = _stratified_subsample_index(income, target_n=10_000)
        assert np.all(idx[1:] >= idx[:-1])
