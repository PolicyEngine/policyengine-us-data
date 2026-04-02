from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import policyengine_us_data.storage.upload_completed_datasets as upload_module
from policyengine_us_data.storage.upload_completed_datasets import (
    DatasetValidationError,
    validate_dataset,
)
from policyengine_us_data.utils.dataset_validation import validate_dataset_contract
from policyengine_us_data.utils.policyengine import PolicyEngineUSBuildInfo


class _FakeArrayResult:
    def __init__(self, values):
        self.values = values


class _FakeMicrosimulation:
    def __init__(self, dataset=None):
        self.dataset = dataset

    def calculate(self, variable_name):
        return _FakeArrayResult(np.array([1.0], dtype=np.float32))


class _AggregateResult:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def sum(self):
        return float(self.values.sum())


class _TimePeriodCheckingAggregateMicrosimulation:
    last_dataset = None

    def __init__(self, dataset=None):
        _TimePeriodCheckingAggregateMicrosimulation.last_dataset = dataset
        if getattr(dataset, "time_period", None) is None:
            raise ValueError(
                "Expected a period (eg. '2017', '2017-01', '2017-01-01', ...); got: 'None'."
            )

    def calculate(self, variable_name, period=None):
        if variable_name == "employment_income":
            return _AggregateResult([6e12])
        if variable_name == "household_weight":
            return _AggregateResult([1.5e8])
        raise KeyError(variable_name)


def _fake_tax_benefit_system():
    variable_entities = {
        "person_id": "person",
        "household_id": "household",
        "employment_income": "person",
        "household_weight": "household",
    }
    return SimpleNamespace(
        variables={
            variable_name: SimpleNamespace(entity=SimpleNamespace(key=entity_key))
            for variable_name, entity_key in variable_entities.items()
        }
    )


def _write_h5(path, datasets: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as h5_file:
        for name, values in datasets.items():
            h5_file.create_dataset(name, data=values)


@pytest.fixture(autouse=True)
def patch_contract_validation(monkeypatch):
    monkeypatch.setitem(upload_module.MIN_FILE_SIZES, "cps_2024.h5", 0)
    monkeypatch.setattr(
        "policyengine_us_data.utils.dataset_validation.assert_locked_policyengine_us_version",
        lambda: PolicyEngineUSBuildInfo(version="1.587.0", locked_version="1.587.0"),
    )
    monkeypatch.setattr(
        upload_module,
        "validate_dataset_contract",
        lambda file_path: validate_dataset_contract(
            file_path,
            tax_benefit_system=_fake_tax_benefit_system(),
            microsimulation_cls=_FakeMicrosimulation,
            dataset_loader=lambda path: path,
        ),
    )


def test_validate_dataset_rejects_unalignable_auxiliary_variables(tmp_path):
    file_path = tmp_path / "cps_2024.h5"
    _write_h5(
        file_path,
        {
            "person_id": np.array([101], dtype=np.int32),
            "household_id": np.array([201], dtype=np.int32),
            "employment_income": np.array([50_000.0], dtype=np.float32),
            "household_weight": np.array([1.0], dtype=np.float32),
            "mystery_variable": np.array([1.0, 2.0], dtype=np.float32),
        },
    )

    with pytest.raises(
        DatasetValidationError,
        match="does not match any entity count",
    ):
        validate_dataset(file_path)


def test_validate_dataset_rejects_entity_length_mismatches(tmp_path):
    file_path = tmp_path / "cps_2024.h5"
    _write_h5(
        file_path,
        {
            "person_id": np.array([101], dtype=np.int32),
            "household_id": np.array([201], dtype=np.int32),
            "employment_income": np.array([50_000.0, 60_000.0], dtype=np.float32),
            "household_weight": np.array([1.0], dtype=np.float32),
        },
    )

    with pytest.raises(
        DatasetValidationError,
        match="inconsistent entity lengths",
    ):
        validate_dataset(file_path)


def test_validate_dataset_infers_time_period_for_flat_h5(tmp_path, monkeypatch):
    file_path = tmp_path / "cps_2024.h5"
    _write_h5(
        file_path,
        {
            "person_id": np.array([101], dtype=np.int32),
            "household_id": np.array([201], dtype=np.int32),
            "employment_income": np.array([50_000.0], dtype=np.float32),
            "household_weight": np.array([1.0], dtype=np.float32),
        },
    )

    monkeypatch.setattr(
        "policyengine_us.Microsimulation",
        _TimePeriodCheckingAggregateMicrosimulation,
    )

    validate_dataset(file_path)

    assert _TimePeriodCheckingAggregateMicrosimulation.last_dataset.time_period == 2024
