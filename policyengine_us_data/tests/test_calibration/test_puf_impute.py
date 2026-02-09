"""Tests for puf_impute module.

Verifies PUF clone + QRF imputation logic using mock data
so tests don't require real CPS/PUF datasets.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from policyengine_us_data.calibration.puf_impute import (
    puf_clone_dataset,
    DEMOGRAPHIC_PREDICTORS,
    IMPUTED_VARIABLES,
)


# ------------------------------------------------------------------
# Mock helpers
# ------------------------------------------------------------------


def _make_mock_data(n_persons=20, n_households=5, time_period=2024):
    """Build a minimal mock CPS data dict.

    Returns a dict of {variable: {time_period: array}} matching
    the Dataset.TIME_PERIOD_ARRAYS format.
    """
    # Person-level IDs and demographics
    person_ids = np.arange(1, n_persons + 1)
    # 4 persons per household
    household_ids_person = np.repeat(
        np.arange(1, n_households + 1), n_persons // n_households
    )
    tax_unit_ids_person = household_ids_person.copy()
    spm_unit_ids_person = household_ids_person.copy()

    ages = np.random.default_rng(42).integers(18, 80, size=n_persons)
    is_male = np.random.default_rng(42).integers(0, 2, size=n_persons)

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
            time_period: np.random.default_rng(42).uniform(
                0, 100000, n_persons
            )
        },
    }
    return data


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestPufCloneDataset:
    """Tests for puf_clone_dataset."""

    def test_doubles_records(self):
        """Output has 2x the records of input."""
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])  # per household

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        # Households should double
        assert len(result["household_id"][2024]) == 10
        # Persons should double
        assert len(result["person_id"][2024]) == 40

    def test_ids_are_unique(self):
        """Person and household IDs are unique across both halves."""
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
        """PUF half has zero household weights."""
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        weights = result["household_weight"][2024]
        # First half: original weights
        assert np.all(weights[:5] > 0)
        # Second half: zero weights (PUF copy)
        assert np.all(weights[5:] == 0)

    def test_state_fips_preserved(self):
        """State FIPS doubles along with records."""
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        result_states = result["state_fips"][2024]
        # Both halves should have the same state assignments
        np.testing.assert_array_equal(result_states[:5], state_fips)
        np.testing.assert_array_equal(result_states[5:], state_fips)

    def test_demographics_shared(self):
        """Both halves share the same demographic values."""
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

    def test_n_records_output(self):
        """Returns correct n_households_out count."""
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        # Should have 10 households total
        assert len(result["household_id"][2024]) == 10

    def test_demographic_predictors_list(self):
        """DEMOGRAPHIC_PREDICTORS includes state_fips."""
        assert "state_fips" in DEMOGRAPHIC_PREDICTORS

    def test_imputed_variables_not_empty(self):
        """IMPUTED_VARIABLES list is populated."""
        assert len(IMPUTED_VARIABLES) > 0
