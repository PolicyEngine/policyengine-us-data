from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import policyengine_us_data.storage.upload_completed_datasets as upload_module
from policyengine_us_data.storage.upload_completed_datasets import (
    DatasetValidationError,
    validate_dataset,
)
import policyengine_us_data.utils.dataset_validation as _dv_mod
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
    periods = []

    def __init__(self, dataset=None):
        _TimePeriodCheckingAggregateMicrosimulation.last_dataset = dataset
        _TimePeriodCheckingAggregateMicrosimulation.periods = []
        if getattr(dataset, "time_period", None) is None:
            raise ValueError(
                "Expected a period (eg. '2017', '2017-01', '2017-01-01', ...); got: 'None'."
            )

    def calculate(self, variable_name, period=None):
        _TimePeriodCheckingAggregateMicrosimulation.periods.append(period)
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
        _dv_mod,
        "assert_locked_policyengine_us_version",
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


def test_validate_dataset_uses_dataset_time_period_for_aggregate_checks(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "enhanced_cps_2025.h5"
    _write_h5(
        file_path,
        {
            "person_id": np.array([101], dtype=np.int32),
            "household_id": np.array([201], dtype=np.int32),
            "employment_income": np.array([50_000.0], dtype=np.float32),
            "household_weight": np.array([1.0], dtype=np.float32),
        },
    )

    monkeypatch.setitem(upload_module.MIN_FILE_SIZES, "enhanced_cps_2025.h5", 0)
    monkeypatch.setattr(
        "policyengine_us.Microsimulation",
        _TimePeriodCheckingAggregateMicrosimulation,
    )

    validate_dataset(file_path)

    assert _TimePeriodCheckingAggregateMicrosimulation.last_dataset.time_period == 2025
    assert _TimePeriodCheckingAggregateMicrosimulation.periods == [2025, 2025]


def test_validate_dataset_skips_aggregate_checks_for_small_enhanced(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "small_enhanced_cps_2025.h5"
    _write_h5(
        file_path,
        {
            "person_id": np.array([101], dtype=np.int32),
            "household_id": np.array([201], dtype=np.int32),
            "employment_income": np.array([50_000.0], dtype=np.float32),
            "household_weight": np.array([1.0], dtype=np.float32),
        },
    )

    class _FailingMicrosimulation:
        def __init__(self, *args, **kwargs):
            raise AssertionError("aggregate checks should not run")

    monkeypatch.setattr("policyengine_us.Microsimulation", _FailingMicrosimulation)

    validate_dataset(file_path)


def test_validate_built_datasets_checks_calibration_and_small_artifacts(
    tmp_path, monkeypatch
):
    cps_path = tmp_path / "cps_2024.h5"
    enhanced_path = tmp_path / "enhanced_cps_2025.h5"
    source_imputed_path = tmp_path / "source_imputed_stratified_extended_cps_2025.h5"
    small_path = tmp_path / "small_enhanced_cps_2025.h5"
    for path in [cps_path, enhanced_path, source_imputed_path, small_path]:
        path.touch()

    monkeypatch.setattr(upload_module.CPS_2024, "file_path", cps_path, raising=False)
    monkeypatch.setattr(
        upload_module.EnhancedCPS_2025,
        "file_path",
        enhanced_path,
        raising=False,
    )
    monkeypatch.setattr(
        upload_module,
        "source_imputed_stratified_extended_cps_path",
        lambda year: source_imputed_path,
    )
    monkeypatch.setattr(
        upload_module,
        "small_enhanced_cps_path",
        lambda year: small_path,
    )

    validated = []
    monkeypatch.setattr(
        upload_module,
        "validate_dataset",
        lambda path: validated.append(path.name),
    )

    upload_module.validate_built_datasets(require_enhanced_cps=True)

    assert validated == [
        "cps_2024.h5",
        "source_imputed_stratified_extended_cps_2025.h5",
        "enhanced_cps_2025.h5",
        "small_enhanced_cps_2025.h5",
    ]


def test_upload_calibration_dataset_validates_before_upload(tmp_path, monkeypatch):
    calibration_dataset_path = (
        tmp_path / "source_imputed_stratified_extended_cps_2025.h5"
    )
    calibration_dataset_path.touch()

    monkeypatch.setattr(
        upload_module,
        "source_imputed_stratified_extended_cps_path",
        lambda year: calibration_dataset_path,
    )

    calls = []
    monkeypatch.setattr(
        upload_module,
        "validate_dataset",
        lambda path: calls.append(("validate", Path(path).name)),
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.utils.huggingface",
        SimpleNamespace(
            upload=lambda local_file_path, repo, repo_file_path: calls.append(
                ("upload", Path(local_file_path).name, repo, repo_file_path)
            )
        ),
    )

    upload_module.upload_calibration_dataset()

    assert calls == [
        ("validate", "source_imputed_stratified_extended_cps_2025.h5"),
        (
            "upload",
            "source_imputed_stratified_extended_cps_2025.h5",
            "policyengine/policyengine-us-data",
            "calibration/source_imputed_stratified_extended_cps.h5",
        ),
    ]
