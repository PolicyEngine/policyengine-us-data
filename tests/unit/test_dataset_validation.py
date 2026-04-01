from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from policyengine_us_data.utils.dataset_validation import (
    DatasetContractError,
    validate_dataset_contract,
)
from policyengine_us_data.utils.policyengine import PolicyEngineUSBuildInfo


class _FakeArrayResult:
    def __init__(self, values):
        self.values = values


class _FakeMicrosimulation:
    last_dataset = None
    calculate_calls = []

    def __init__(self, dataset=None):
        _FakeMicrosimulation.last_dataset = dataset

    def calculate(self, variable_name):
        _FakeMicrosimulation.calculate_calls.append(variable_name)
        return _FakeArrayResult(np.array([1.0], dtype=np.float32))


def _write_test_h5(path, datasets: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as h5_file:
        for name, values in datasets.items():
            h5_file.create_dataset(name, data=values)


def _fake_tax_benefit_system():
    variable_entities = {
        "person_id": "person",
        "tax_unit_id": "tax_unit",
        "family_id": "family",
        "spm_unit_id": "spm_unit",
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


@pytest.fixture(autouse=True)
def reset_fake_microsim():
    _FakeMicrosimulation.last_dataset = None
    _FakeMicrosimulation.calculate_calls = []


def test_validate_dataset_contract_passes(tmp_path, monkeypatch):
    file_path = tmp_path / "valid.h5"
    _write_test_h5(
        file_path,
        {
            "person_id": np.array([101, 102], dtype=np.int32),
            "tax_unit_id": np.array([201], dtype=np.int32),
            "family_id": np.array([301], dtype=np.int32),
            "spm_unit_id": np.array([401], dtype=np.int32),
            "household_id": np.array([501], dtype=np.int32),
            "employment_income": np.array([10_000.0, 20_000.0], dtype=np.float32),
            "household_weight": np.array([1.5], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "policyengine_us_data.utils.dataset_validation.assert_locked_policyengine_us_version",
        lambda: PolicyEngineUSBuildInfo(
            version="1.587.0",
            locked_version="1.587.0",
            git_commit="abc123",
        ),
    )

    summary = validate_dataset_contract(
        file_path,
        tax_benefit_system=_fake_tax_benefit_system(),
        microsimulation_cls=_FakeMicrosimulation,
        dataset_loader=lambda path: f"dataset::{path.name}",
    )

    assert summary.variable_count == 7
    assert summary.entity_counts == {
        "person": 2,
        "tax_unit": 1,
        "family": 1,
        "spm_unit": 1,
        "household": 1,
    }
    assert summary.policyengine_us.version == "1.587.0"
    assert _FakeMicrosimulation.last_dataset == "dataset::valid.h5"
    assert _FakeMicrosimulation.calculate_calls == ["household_weight"]


def test_validate_dataset_contract_warns_on_unknown_variables(
    tmp_path, monkeypatch, caplog
):
    file_path = tmp_path / "unknown.h5"
    _write_test_h5(
        file_path,
        {
            "person_id": np.array([101], dtype=np.int32),
            "mystery_variable": np.array([1.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "policyengine_us_data.utils.dataset_validation.assert_locked_policyengine_us_version",
        lambda: PolicyEngineUSBuildInfo(version="1.587.0"),
    )

    import logging

    with caplog.at_level(logging.WARNING):
        summary = validate_dataset_contract(
            file_path,
            tax_benefit_system=_fake_tax_benefit_system(),
            microsimulation_cls=_FakeMicrosimulation,
            dataset_loader=lambda path: path,
        )
    assert "mystery_variable" in caplog.text
    assert summary.variable_count == 2


def test_validate_dataset_contract_rejects_entity_length_mismatch(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "mismatch.h5"
    _write_test_h5(
        file_path,
        {
            "person_id": np.array([101], dtype=np.int32),
            "tax_unit_id": np.array([201], dtype=np.int32),
            "family_id": np.array([301], dtype=np.int32),
            "spm_unit_id": np.array([401], dtype=np.int32),
            "household_id": np.array([501], dtype=np.int32),
            "employment_income": np.array([10_000.0, 20_000.0], dtype=np.float32),
            "household_weight": np.array([1.5], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "policyengine_us_data.utils.dataset_validation.assert_locked_policyengine_us_version",
        lambda: PolicyEngineUSBuildInfo(version="1.587.0"),
    )

    with pytest.raises(DatasetContractError, match="inconsistent entity lengths"):
        validate_dataset_contract(
            file_path,
            tax_benefit_system=_fake_tax_benefit_system(),
            microsimulation_cls=_FakeMicrosimulation,
            dataset_loader=lambda path: path,
        )
