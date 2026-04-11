from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

import policyengine_us_data.storage.upload_completed_datasets as upload_module
from policyengine_us_data.storage.upload_completed_datasets import (
    DatasetValidationError,
    upload_datasets,
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


def _prepare_release_files(tmp_path, monkeypatch):
    cps_path = tmp_path / "cps_2024.h5"
    cps_path.write_bytes(b"cps")
    enhanced_path = tmp_path / "enhanced_cps_2024.h5"
    enhanced_path.write_bytes(b"enhanced")
    small_path = tmp_path / "small_enhanced_cps_2024.h5"
    small_path.write_bytes(b"small")
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    db_path = calibration_dir / "policy_data.db"
    db_path.write_bytes(b"db")

    monkeypatch.setattr(upload_module.CPS_2024, "file_path", cps_path)
    monkeypatch.setattr(upload_module.EnhancedCPS_2024, "file_path", enhanced_path)
    monkeypatch.setattr(upload_module, "STORAGE_FOLDER", tmp_path)

    return {
        "cps": cps_path,
        "enhanced": enhanced_path,
        "small": small_path,
        "db": db_path,
    }


def test_upload_datasets_stages_then_promotes_release(tmp_path, monkeypatch):
    _prepare_release_files(tmp_path, monkeypatch)
    validated = []
    stage_calls = []
    promote_calls = []

    monkeypatch.setattr(
        upload_module,
        "validate_dataset",
        lambda file_path: validated.append(Path(file_path).name),
    )
    monkeypatch.setattr(
        upload_module.metadata,
        "version",
        lambda _: "1.73.0",
    )
    monkeypatch.setattr(
        upload_module,
        "upload_to_staging_hf",
        lambda files_with_paths, **kwargs: stage_calls.append(
            ([(Path(path), repo_path) for path, repo_path in files_with_paths], kwargs)
        ),
    )
    monkeypatch.setattr(
        upload_module,
        "promote_staging_to_production_hf",
        lambda rel_paths, **kwargs: promote_calls.append(("hf", rel_paths, kwargs)),
    )
    monkeypatch.setattr(
        upload_module,
        "upload_from_hf_staging_to_gcs",
        lambda rel_paths, **kwargs: promote_calls.append(("gcs", rel_paths, kwargs)),
    )
    publish_calls = []
    monkeypatch.setattr(
        upload_module,
        "publish_release_manifest_to_hf",
        lambda files_with_paths, **kwargs: publish_calls.append(
            ([(Path(path), repo_path) for path, repo_path in files_with_paths], kwargs)
        ),
    )
    monkeypatch.setattr(
        upload_module,
        "should_finalize_local_area_release",
        lambda **kwargs: (False, ["national/", "states/", "districts/", "cities/"]),
    )
    cleanup_calls = []
    monkeypatch.setattr(
        upload_module,
        "cleanup_staging_hf",
        lambda rel_paths, **kwargs: cleanup_calls.append((rel_paths, kwargs)),
    )

    build_manifest_calls = []
    upload_manifest_calls = []
    monkeypatch.setattr(
        upload_module,
        "build_manifest",
        lambda **kwargs: build_manifest_calls.append(kwargs),
    )
    monkeypatch.setattr(
        upload_module,
        "upload_manifest",
        lambda manifest: upload_manifest_calls.append(manifest),
    )

    upload_datasets(version="1.73.0")

    expected_repo_paths = [
        "cps_2024.h5",
        "policy_data.db",
        "enhanced_cps_2024.h5",
        "small_enhanced_cps_2024.h5",
    ]
    assert validated == [
        "cps_2024.h5",
        "policy_data.db",
        "enhanced_cps_2024.h5",
        "small_enhanced_cps_2024.h5",
    ]
    assert [repo_path for _, repo_path in stage_calls[0][0]] == expected_repo_paths
    assert stage_calls[0][1]["run_id"] == ""
    assert promote_calls == [
        (
            "hf",
            expected_repo_paths,
            {
                "version": "1.73.0",
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "run_id": "",
            },
        ),
        (
            "gcs",
            expected_repo_paths,
            {
                "version": "1.73.0",
                "gcs_bucket_name": upload_module.GCS_BUCKET_NAME,
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "run_id": "",
            },
        ),
    ]
    assert [repo_path for _, repo_path in publish_calls[0][0]] == expected_repo_paths
    assert publish_calls[0][1]["create_tag"] is False
    assert build_manifest_calls == []
    assert upload_manifest_calls == []
    assert cleanup_calls == [
        (
            expected_repo_paths,
            {
                "version": "1.73.0",
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "run_id": "",
            },
        )
    ]


def test_upload_datasets_stage_only_skips_promote(tmp_path, monkeypatch):
    _prepare_release_files(tmp_path, monkeypatch)
    stage_calls = []
    promote_calls = []

    monkeypatch.setattr(upload_module, "validate_dataset", lambda file_path: None)
    monkeypatch.setattr(upload_module.metadata, "version", lambda _: "1.73.0")
    monkeypatch.setattr(
        upload_module,
        "upload_to_staging_hf",
        lambda files_with_paths, **kwargs: stage_calls.append(kwargs),
    )
    monkeypatch.setattr(
        upload_module,
        "promote_staging_to_production_hf",
        lambda *args, **kwargs: promote_calls.append((args, kwargs)),
    )

    upload_datasets(stage_only=True, run_id="sha123", version="1.73.0")

    assert stage_calls == [
        {
            "version": "1.73.0",
            "hf_repo_name": upload_module.HF_REPO_NAME,
            "hf_repo_type": upload_module.HF_REPO_TYPE,
            "run_id": "sha123",
        }
    ]
    assert promote_calls == []


def test_upload_datasets_promote_only_uses_staged_artifacts(tmp_path, monkeypatch):
    downloaded_dir = tmp_path / "downloaded"
    downloaded_dir.mkdir()
    expected_repo_paths = [
        "cps_2024.h5",
        "policy_data.db",
        "enhanced_cps_2024.h5",
        "small_enhanced_cps_2024.h5",
    ]

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = [
        f"staging/run-123/{repo_path}" for repo_path in expected_repo_paths
    ]
    monkeypatch.setattr(upload_module, "HfApi", lambda: mock_api)
    monkeypatch.setattr(upload_module.metadata, "version", lambda _: "1.73.0")
    monkeypatch.setattr(
        upload_module,
        "hf_hub_download",
        lambda **kwargs: str(downloaded_dir / Path(kwargs["filename"]).name),
    )
    for repo_path in expected_repo_paths:
        (downloaded_dir / Path(repo_path).name).write_bytes(repo_path.encode())

    validate_calls = []
    monkeypatch.setattr(
        upload_module,
        "validate_dataset",
        lambda file_path: validate_calls.append(file_path),
    )
    promote_calls = []
    monkeypatch.setattr(
        upload_module,
        "promote_staging_to_production_hf",
        lambda rel_paths, **kwargs: promote_calls.append(("hf", rel_paths, kwargs)),
    )
    monkeypatch.setattr(
        upload_module,
        "upload_from_hf_staging_to_gcs",
        lambda rel_paths, **kwargs: promote_calls.append(("gcs", rel_paths, kwargs)),
    )
    publish_calls = []
    monkeypatch.setattr(
        upload_module,
        "publish_release_manifest_to_hf",
        lambda files_with_paths, **kwargs: publish_calls.append(
            ([repo_path for _, repo_path in files_with_paths], kwargs)
        ),
    )
    monkeypatch.setattr(
        upload_module,
        "should_finalize_local_area_release",
        lambda **kwargs: (False, ["national/", "states/", "districts/", "cities/"]),
    )
    upload_manifest_calls = []
    monkeypatch.setattr(
        upload_module,
        "upload_manifest",
        lambda manifest: upload_manifest_calls.append(manifest),
    )
    cleanup_calls = []
    monkeypatch.setattr(
        upload_module,
        "cleanup_staging_hf",
        lambda rel_paths, **kwargs: cleanup_calls.append((rel_paths, kwargs)),
    )

    upload_datasets(promote_only=True, run_id="run-123", version="1.73.0")

    assert validate_calls == []
    assert promote_calls == [
        (
            "hf",
            expected_repo_paths,
            {
                "version": "1.73.0",
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "run_id": "run-123",
            },
        ),
        (
            "gcs",
            expected_repo_paths,
            {
                "version": "1.73.0",
                "gcs_bucket_name": upload_module.GCS_BUCKET_NAME,
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "run_id": "run-123",
            },
        ),
    ]
    assert publish_calls == [
        (
            expected_repo_paths,
            {
                "version": "1.73.0",
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "create_tag": False,
            },
        )
    ]
    assert upload_manifest_calls == []
    assert cleanup_calls == [
        (
            expected_repo_paths,
            {
                "version": "1.73.0",
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "run_id": "run-123",
            },
        )
    ]
