from types import SimpleNamespace

import numpy as np
import pytest

from policyengine_us_data.utils.downsample import downsample_dataset_arrays


class _FakeArrayResult:
    def __init__(self, values):
        self.values = values


class _FakeMicrosimulation:
    def __init__(self, variable_entities, calculated_values):
        self.tax_benefit_system = SimpleNamespace(
            variables={
                variable_name: SimpleNamespace(entity=SimpleNamespace(key=entity_key))
                for variable_name, entity_key in variable_entities.items()
            }
        )
        self._calculated_values = calculated_values

    def calculate(self, variable_name):
        return _FakeArrayResult(self._calculated_values[variable_name])


def test_downsample_dataset_arrays_preserves_original_dtypes():
    original_data = {
        "person_id": np.array([101, 102], dtype=np.int32),
        "household_id": np.array([201], dtype=np.int32),
        "employment_income": np.array([100.0, 200.0], dtype=np.float32),
    }
    sim = _FakeMicrosimulation(
        variable_entities={
            "person_id": "person",
            "household_id": "household",
            "employment_income": "person",
        },
        calculated_values={
            "person_id": np.array([101], dtype=np.int64),
            "household_id": np.array([201], dtype=np.int64),
            "employment_income": np.array([150.0], dtype=np.float64),
        },
    )

    resampled = downsample_dataset_arrays(
        original_data=original_data,
        sim=sim,
        dataset_name="cps",
    )

    assert resampled["person_id"].dtype == np.int32
    assert resampled["household_id"].dtype == np.int32
    assert resampled["employment_income"].dtype == np.float32
    np.testing.assert_array_equal(
        resampled["employment_income"], np.array([150.0], dtype=np.float32)
    )


def test_downsample_dataset_arrays_resamples_auxiliary_variables():
    original_data = {
        "person_id": np.array([101, 102], dtype=np.int32),
        "household_id": np.array([202], dtype=np.int32),
        "employment_income": np.array([100.0, 200.0], dtype=np.float32),
        "hourly_wage": np.array([25.0, 30.0], dtype=np.float32),
        "count_under_18": np.array([0], dtype=np.int32),
    }
    sim = _FakeMicrosimulation(
        variable_entities={
            "person_id": "person",
            "household_id": "household",
            "employment_income": "person",
        },
        calculated_values={
            "person_id": np.array([102], dtype=np.int64),
            "household_id": np.array([202], dtype=np.int64),
            "employment_income": np.array([200.0], dtype=np.float64),
        },
    )

    resampled = downsample_dataset_arrays(
        original_data=original_data,
        sim=sim,
        dataset_name="cps",
    )

    np.testing.assert_array_equal(
        resampled["hourly_wage"], np.array([30.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        resampled["count_under_18"], np.array([0], dtype=np.int32)
    )


def test_downsample_dataset_arrays_rejects_ambiguous_auxiliary_variable_lengths():
    original_data = {
        "person_id": np.array([101], dtype=np.int32),
        "household_id": np.array([201], dtype=np.int32),
        "mystery_variable": np.array([5.0], dtype=np.float32),
    }
    sim = _FakeMicrosimulation(
        variable_entities={
            "person_id": "person",
            "household_id": "household",
        },
        calculated_values={
            "person_id": np.array([101], dtype=np.int64),
            "household_id": np.array([201], dtype=np.int64),
        },
    )

    with pytest.raises(ValueError, match="matches multiple entity sizes"):
        downsample_dataset_arrays(
            original_data=original_data,
            sim=sim,
            dataset_name="cps",
        )


def test_downsample_dataset_arrays_rejects_entity_length_mismatches():
    original_data = {
        "person_id": np.array([101, 102], dtype=np.int32),
        "household_id": np.array([201], dtype=np.int32),
        "employment_income": np.array([100.0, 200.0], dtype=np.float32),
    }
    sim = _FakeMicrosimulation(
        variable_entities={
            "person_id": "person",
            "household_id": "household",
            "employment_income": "person",
        },
        calculated_values={
            "person_id": np.array([101], dtype=np.int64),
            "household_id": np.array([201], dtype=np.int64),
            "employment_income": np.array([150.0, 250.0], dtype=np.float64),
        },
    )

    with pytest.raises(ValueError, match="entity lengths are inconsistent"):
        downsample_dataset_arrays(
            original_data=original_data,
            sim=sim,
            dataset_name="cps",
        )
