import json
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
    validate_built_datasets,
)
import policyengine_us_data.utils.dataset_validation as _dv_mod
from policyengine_us_data.utils.dataset_validation import validate_dataset_contract
from policyengine_us_data.utils.policyengine import PolicyEngineUSBuildInfo


ORIGINAL_MICROSIMULATION_AGGREGATE_CHECKS_BY_FILENAME = (
    upload_module.MICROSIMULATION_AGGREGATE_CHECKS_BY_FILENAME
)

VALID_CLONE_DIAGNOSTICS = {
    "period": 2024,
    "clone_household_weight_share_pct": 5.0,
    "clone_person_weight_share_pct": 5.0,
    "clone_poor_person_weight_share_pct": 1.0,
    "clone_childcare_exceeds_pre_subsidy_share_pct": 0.0,
    "clone_childcare_above_5000_share_pct": 0.0,
    "clone_taxes_exceed_market_income_share_pct": 0.0,
}


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

    def mean(self):
        return float(self.values.mean())

    def __ge__(self, other):
        return self.values >= other

    def __getitem__(self, key):
        return _AggregateResult(self.values[key])


class _TimePeriodCheckingAggregateMicrosimulation:
    last_dataset = None
    calls = []
    overrides = {}

    def __init__(self, dataset=None):
        _TimePeriodCheckingAggregateMicrosimulation.last_dataset = dataset
        if getattr(dataset, "time_period", None) is None:
            raise ValueError(
                "Expected a period (eg. '2017', '2017-01', '2017-01-01', ...); got: 'None'."
            )

    def calculate(self, variable_name, period=None, map_to=None):
        _TimePeriodCheckingAggregateMicrosimulation.calls.append(
            (variable_name, period, map_to)
        )
        if variable_name in _TimePeriodCheckingAggregateMicrosimulation.overrides:
            return _AggregateResult(
                _TimePeriodCheckingAggregateMicrosimulation.overrides[variable_name]
            )
        if variable_name == "employment_income":
            return _AggregateResult([6e12])
        if variable_name == "household_weight":
            return _AggregateResult([1.5e8])
        raise KeyError(variable_name)


def _fake_tax_benefit_system():
    variable_entities = {
        "person_id": "person",
        "spm_unit_id": "spm_unit",
        "household_id": "household",
        "employment_income": "person",
        "household_weight": "household",
        "takes_up_snap_if_eligible": "spm_unit",
        "takes_up_ssi_if_eligible": "person",
        "takes_up_tanf_if_eligible": "spm_unit",
        "takes_up_housing_assistance_if_eligible": "spm_unit",
        "would_claim_wic": "person",
        "is_wic_at_nutritional_risk": "person",
    }
    return SimpleNamespace(
        variables={
            variable_name: SimpleNamespace(entity=SimpleNamespace(key=entity_key))
            for variable_name, entity_key in variable_entities.items()
        }
    )


def _fake_variable(entity_key, *, formulas=None, adds=None, subtracts=None):
    return SimpleNamespace(
        entity=SimpleNamespace(key=entity_key),
        formulas=formulas or {},
        adds=adds,
        subtracts=subtracts,
    )


def _write_h5(path, datasets: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as h5_file:
        for name, values in datasets.items():
            h5_file.create_dataset(name, data=values)


def _minimal_enhanced_cps_contract_datasets(**overrides):
    datasets = {
        "person_id": np.array([101, 102, 103], dtype=np.int32),
        "spm_unit_id": np.array([201, 202], dtype=np.int32),
        "household_id": np.array([301, 302], dtype=np.int32),
        "employment_income": np.array([50_000.0, 60_000.0, 0.0], dtype=np.float32),
        "household_weight": np.array([1.0, 1.0], dtype=np.float32),
    }
    for key, value in overrides.items():
        if value is None:
            datasets.pop(key, None)
        else:
            datasets[key] = value
    return datasets


@pytest.fixture(autouse=True)
def patch_contract_validation(monkeypatch):
    monkeypatch.setitem(upload_module.MIN_FILE_SIZES, "cps_2024.h5", 0)
    monkeypatch.setitem(upload_module.MIN_FILE_SIZES, "enhanced_cps_2024.h5", 0)
    monkeypatch.setattr(upload_module, "H5_SUM_CHECKS_BY_FILENAME", {})
    monkeypatch.setattr(
        upload_module,
        "MICROSIMULATION_AGGREGATE_CHECKS_BY_FILENAME",
        {},
    )
    _TimePeriodCheckingAggregateMicrosimulation.calls = []
    _TimePeriodCheckingAggregateMicrosimulation.overrides = {}
    monkeypatch.setattr(
        _dv_mod,
        "assert_locked_policyengine_us_version",
        lambda: PolicyEngineUSBuildInfo(version="1.587.0", locked_version="1.587.0"),
    )
    monkeypatch.setattr(
        upload_module,
        "validate_dataset_contract",
        lambda file_path, **kwargs: validate_dataset_contract(
            file_path,
            tax_benefit_system=_fake_tax_benefit_system(),
            microsimulation_cls=_FakeMicrosimulation,
            dataset_loader=lambda path: path,
            **kwargs,
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


def test_validate_cps_allows_source_computed_policyengine_variables(
    tmp_path,
    monkeypatch,
):
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
    tbs = _fake_tax_benefit_system()
    tbs.variables["employment_income"] = _fake_variable(
        "person",
        adds=["employment_income_before_lsr"],
    )
    monkeypatch.setattr(
        upload_module,
        "validate_dataset_contract",
        lambda file_path, **kwargs: validate_dataset_contract(
            file_path,
            tax_benefit_system=tbs,
            microsimulation_cls=_FakeMicrosimulation,
            dataset_loader=lambda path: path,
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        "policyengine_us.Microsimulation",
        _TimePeriodCheckingAggregateMicrosimulation,
    )

    validate_dataset(file_path)


def test_validate_enhanced_cps_rejects_computed_policyengine_variables(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "enhanced_cps_2024.h5"
    _write_h5(file_path, _minimal_enhanced_cps_contract_datasets())
    tbs = _fake_tax_benefit_system()
    tbs.variables["employment_income"] = _fake_variable(
        "person",
        adds=["employment_income_before_lsr"],
    )
    monkeypatch.setattr(
        upload_module,
        "REQUIRED_VARIABLES_BY_FILENAME",
        {},
    )
    monkeypatch.setattr(
        upload_module,
        "validate_dataset_contract",
        lambda file_path, **kwargs: validate_dataset_contract(
            file_path,
            tax_benefit_system=tbs,
            microsimulation_cls=_FakeMicrosimulation,
            dataset_loader=lambda path: path,
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        "policyengine_us.Microsimulation",
        _TimePeriodCheckingAggregateMicrosimulation,
    )

    with pytest.raises(DatasetValidationError, match="employment_income"):
        validate_dataset(file_path)


def test_validate_dataset_rejects_temporary_reported_source_variables(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "cps_2024.h5"
    _write_h5(
        file_path,
        {
            "person_id": np.array([101], dtype=np.int32),
            "household_id": np.array([201], dtype=np.int32),
            "employment_income": np.array([50_000.0], dtype=np.float32),
            "household_weight": np.array([1.0], dtype=np.float32),
            "snap_reported": np.array([1_200.0], dtype=np.float32),
            "ssi_reported": np.array([600.0], dtype=np.float32),
        },
    )

    monkeypatch.setattr(
        "policyengine_us.Microsimulation",
        _TimePeriodCheckingAggregateMicrosimulation,
    )

    with pytest.raises(
        DatasetValidationError,
        match="temporary or retired variables: snap_reported, ssi_reported",
    ):
        validate_dataset(file_path)


def test_validate_enhanced_cps_rejects_missing_critical_leaf_inputs(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "enhanced_cps_2024.h5"
    _write_h5(
        file_path,
        _minimal_enhanced_cps_contract_datasets(),
    )
    monkeypatch.setattr(
        upload_module,
        "REQUIRED_VARIABLES_BY_FILENAME",
        {"enhanced_cps_2024.h5": ("required_leaf_input",)},
    )

    monkeypatch.setattr(
        "policyengine_us.Microsimulation",
        _TimePeriodCheckingAggregateMicrosimulation,
    )

    with pytest.raises(
        DatasetValidationError,
        match="Required group 'required_leaf_input' missing or empty",
    ):
        validate_dataset(file_path)


def test_validate_dataset_rejects_configured_implausible_mapped_aggregate(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "enhanced_cps_2024.h5"
    _write_h5(file_path, _minimal_enhanced_cps_contract_datasets())
    _TimePeriodCheckingAggregateMicrosimulation.overrides = {
        "generic_population_share": [0, 1, 1, 0, 1],
        "generic_age": [30, 67, 66, 12, 78],
    }
    monkeypatch.setattr(
        upload_module,
        "REQUIRED_VARIABLES_BY_FILENAME",
        {},
    )
    monkeypatch.setattr(
        upload_module,
        "MICROSIMULATION_AGGREGATE_CHECKS_BY_FILENAME",
        {
            "enhanced_cps_2024.h5": (
                upload_module.MicrosimulationAggregateCheck(
                    variable="generic_population_share",
                    label="generic senior share",
                    statistic="mean",
                    max_value=0.5,
                    map_to="person",
                    filter_variable="generic_age",
                    filter_map_to="person",
                    filter_min_value=65,
                ),
            )
        },
    )

    monkeypatch.setattr(
        "policyengine_us.Microsimulation",
        _TimePeriodCheckingAggregateMicrosimulation,
    )

    with pytest.raises(
        DatasetValidationError,
        match="generic senior share",
    ):
        validate_dataset(file_path)

    assert (
        "generic_population_share",
        2024,
        "person",
    ) in _TimePeriodCheckingAggregateMicrosimulation.calls


def test_enhanced_cps_employment_income_gate_rejects_missing_nipa_target(
    monkeypatch,
):
    target = upload_module.BEA_NIPA_WAGES_AND_SALARIES_2024

    assert upload_module.MIN_ENHANCED_CPS_EMPLOYMENT_INCOME_SUM == pytest.approx(
        target * 0.9
    )
    assert upload_module.MAX_ENHANCED_CPS_EMPLOYMENT_INCOME_SUM == pytest.approx(
        target * 1.1
    )

    monkeypatch.setattr(
        upload_module,
        "MICROSIMULATION_AGGREGATE_CHECKS_BY_FILENAME",
        ORIGINAL_MICROSIMULATION_AGGREGATE_CHECKS_BY_FILENAME,
    )
    _TimePeriodCheckingAggregateMicrosimulation.overrides = {
        "employment_income": [8_805_350_912_424.707],
        "social_security_retirement": [1.0e12],
        "person_in_poverty": [0.2, 0.2, 0.2],
        "age": [20, 67, 80],
    }
    errors = []
    results = upload_module._run_microsimulation_aggregate_checks(
        _TimePeriodCheckingAggregateMicrosimulation(
            dataset=SimpleNamespace(time_period=2024)
        ),
        filename="enhanced_cps_2024.h5",
        period=2024,
        errors=errors,
    )

    assert (
        "employment_income sum vs NIPA wages target",
        8_805_350_912_424.707,
    ) in results
    assert errors == [
        (
            "employment_income sum vs NIPA wages target = "
            "8,805,350,912,425, expected >= 11,149,136,100,000."
        )
    ]


def _prepare_release_files(tmp_path, monkeypatch):
    cps_path = tmp_path / "cps_2024.h5"
    cps_path.write_bytes(b"cps")
    enhanced_path = tmp_path / "enhanced_cps_2024.h5"
    enhanced_path.write_bytes(b"enhanced")
    diagnostics_path = enhanced_path.with_suffix(".clone_diagnostics.json")
    diagnostics_path.write_text(json.dumps(VALID_CLONE_DIAGNOSTICS))
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
        "diagnostics": diagnostics_path,
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
        upload_module,
        "DATA_PACKAGE_VERSION",
        "1.73.0",
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
        "preflight_release_manifest_publish",
        lambda *args, **kwargs: (
            False,
            ["national/", "states/", "districts/", "cities/"],
        ),
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
        "enhanced_cps_2024.clone_diagnostics.json",
        "small_enhanced_cps_2024.h5",
    ]
    assert validated == [
        "cps_2024.h5",
        "policy_data.db",
        "enhanced_cps_2024.h5",
        "enhanced_cps_2024.clone_diagnostics.json",
        "small_enhanced_cps_2024.h5",
    ]
    assert [repo_path for _, repo_path in stage_calls[0][0]] == expected_repo_paths
    assert stage_calls[0][1]["run_id"] == ""
    assert promote_calls == [
        (
            "hf",
            expected_repo_paths,
            {
                "candidate_version": "1.73.0",
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "run_id": "",
            },
        ),
        (
            "gcs",
            expected_repo_paths,
            {
                "candidate_version": "1.73.0",
                "release_version": "1.73.0",
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
                "candidate_version": "1.73.0",
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
    monkeypatch.setattr(upload_module, "DATA_PACKAGE_VERSION", "1.73.0")
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
            "candidate_version": "1.73.0",
            "hf_repo_name": upload_module.HF_REPO_NAME,
            "hf_repo_type": upload_module.HF_REPO_TYPE,
            "run_id": "sha123",
        }
    ]
    assert promote_calls == []


def test_upload_datasets_can_stage_enhanced_cps_without_small(
    tmp_path,
    monkeypatch,
):
    _prepare_release_files(tmp_path, monkeypatch)
    staged_files = []

    monkeypatch.setattr(upload_module, "validate_dataset", lambda file_path: None)
    monkeypatch.setattr(upload_module, "DATA_PACKAGE_VERSION", "1.73.0")
    monkeypatch.setattr(
        upload_module,
        "upload_to_staging_hf",
        lambda files_with_paths, **kwargs: staged_files.append(
            ([(Path(path), repo_path) for path, repo_path in files_with_paths], kwargs)
        ),
    )

    upload_datasets(
        require_small_enhanced_cps=False,
        stage_only=True,
        run_id="ecps-only",
        version="1.73.0",
    )

    assert [repo_path for _, repo_path in staged_files[0][0]] == [
        "cps_2024.h5",
        "policy_data.db",
        "enhanced_cps_2024.h5",
        "enhanced_cps_2024.clone_diagnostics.json",
    ]
    assert staged_files[0][1]["run_id"] == "ecps-only"


def test_upload_datasets_promote_only_uses_staged_artifacts(tmp_path, monkeypatch):
    downloaded_dir = tmp_path / "downloaded"
    downloaded_dir.mkdir()
    expected_repo_paths = [
        "cps_2024.h5",
        "policy_data.db",
        "enhanced_cps_2024.h5",
        "enhanced_cps_2024.clone_diagnostics.json",
        "small_enhanced_cps_2024.h5",
    ]

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = [
        f"staging/1.73.0-run-123/{repo_path}" for repo_path in expected_repo_paths
    ]
    monkeypatch.setattr(upload_module, "HfApi", lambda: mock_api)
    monkeypatch.setattr(upload_module, "DATA_PACKAGE_VERSION", "1.73.0")
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
        "preflight_release_manifest_publish",
        lambda *args, **kwargs: (
            False,
            ["national/", "states/", "districts/", "cities/"],
        ),
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

    assert [Path(path).name for path in validate_calls] == [
        Path(repo_path).name for repo_path in expected_repo_paths
    ]
    assert promote_calls == [
        (
            "hf",
            expected_repo_paths,
            {
                "candidate_version": "1.73.0",
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "run_id": "run-123",
            },
        ),
        (
            "gcs",
            expected_repo_paths,
            {
                "candidate_version": "1.73.0",
                "release_version": "1.73.0",
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
                "pipeline_run_id": "run-123",
            },
        )
    ]
    assert upload_manifest_calls == []
    assert cleanup_calls == [
        (
            expected_repo_paths,
            {
                "candidate_version": "1.73.0",
                "hf_repo_name": upload_module.HF_REPO_NAME,
                "hf_repo_type": upload_module.HF_REPO_TYPE,
                "run_id": "run-123",
            },
        )
    ]


def test_promote_datasets_preflight_failure_stops_before_production_writes(
    tmp_path, monkeypatch
):
    files = _prepare_release_files(tmp_path, monkeypatch)
    files_with_repo_paths = [
        (files["cps"], "cps_2024.h5"),
        (files["db"], "policy_data.db"),
    ]
    promote_calls = []

    monkeypatch.setattr(upload_module, "DATA_PACKAGE_VERSION", "1.73.0")
    monkeypatch.setattr(
        upload_module,
        "preflight_release_manifest_publish",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("blocked")),
    )
    monkeypatch.setattr(
        upload_module,
        "promote_staging_to_production_hf",
        lambda *args, **kwargs: promote_calls.append(("hf", args, kwargs)),
    )
    monkeypatch.setattr(
        upload_module,
        "upload_from_hf_staging_to_gcs",
        lambda *args, **kwargs: promote_calls.append(("gcs", args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="blocked"):
        upload_module.promote_datasets(
            version="1.73.0",
            files_with_repo_paths=files_with_repo_paths,
        )

    assert promote_calls == []


def test_upload_datasets_requires_clone_diagnostics_sidecar(tmp_path, monkeypatch):
    files = _prepare_release_files(tmp_path, monkeypatch)
    files["diagnostics"].unlink()
    monkeypatch.setattr(upload_module, "validate_dataset", lambda file_path: None)

    with pytest.raises(FileNotFoundError, match="clone_diagnostics"):
        upload_datasets(require_enhanced_cps=True)


def test_validate_dataset_accepts_clone_diagnostics_sidecar(tmp_path):
    file_path = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    file_path.write_text(json.dumps(VALID_CLONE_DIAGNOSTICS))

    validate_dataset(file_path)


def test_validate_dataset_accepts_multiperiod_clone_diagnostics(tmp_path):
    file_path = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    file_path.write_text(
        json.dumps(
            {
                "periods": {
                    "2024": {
                        key: value
                        for key, value in VALID_CLONE_DIAGNOSTICS.items()
                        if key != "period"
                    }
                }
            }
        )
    )

    validate_dataset(file_path)


def test_validate_dataset_rejects_malformed_clone_diagnostics(tmp_path):
    file_path = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    file_path.write_text("not-json")

    with pytest.raises(DatasetValidationError, match="clone diagnostics JSON"):
        validate_dataset(file_path)


def test_validate_dataset_rejects_incomplete_clone_diagnostics(tmp_path):
    file_path = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    payload = dict(VALID_CLONE_DIAGNOSTICS)
    del payload["clone_household_weight_share_pct"]
    file_path.write_text(json.dumps(payload))

    with pytest.raises(
        DatasetValidationError,
        match="Missing clone diagnostics metric",
    ):
        validate_dataset(file_path)


def test_validate_dataset_rejects_out_of_range_clone_diagnostics(tmp_path):
    file_path = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    payload = dict(VALID_CLONE_DIAGNOSTICS)
    payload["clone_household_weight_share_pct"] = 101.0
    file_path.write_text(json.dumps(payload))

    with pytest.raises(DatasetValidationError, match="between 0 and 100"):
        validate_dataset(file_path)


def test_validate_built_datasets_requires_clone_diagnostics_sidecar(
    tmp_path, monkeypatch
):
    storage_folder = tmp_path / "storage"
    cps_path = storage_folder / "cps_2024.h5"
    enhanced_path = storage_folder / "enhanced_cps_2024.h5"
    small_path = storage_folder / "small_enhanced_cps_2024.h5"

    storage_folder.mkdir(parents=True)
    for path in [cps_path, enhanced_path, small_path]:
        path.write_text("placeholder")

    monkeypatch.setattr(
        upload_module,
        "CPS_2024",
        SimpleNamespace(file_path=cps_path),
    )
    monkeypatch.setattr(
        upload_module,
        "EnhancedCPS_2024",
        SimpleNamespace(file_path=enhanced_path),
    )
    monkeypatch.setattr(upload_module, "STORAGE_FOLDER", storage_folder)
    monkeypatch.setattr(upload_module, "validate_dataset", lambda file_path: None)

    with pytest.raises(FileNotFoundError, match="clone_diagnostics"):
        validate_built_datasets(require_enhanced_cps=True)


def test_validate_built_datasets_can_skip_small_enhanced_cps(tmp_path, monkeypatch):
    storage_folder = tmp_path / "storage"
    cps_path = storage_folder / "cps_2024.h5"
    enhanced_path = storage_folder / "enhanced_cps_2024.h5"
    diagnostics_path = enhanced_path.with_suffix(".clone_diagnostics.json")

    storage_folder.mkdir(parents=True)
    for path in [cps_path, enhanced_path, diagnostics_path]:
        path.write_text("placeholder")

    monkeypatch.setattr(
        upload_module,
        "CPS_2024",
        SimpleNamespace(file_path=cps_path),
    )
    monkeypatch.setattr(
        upload_module,
        "EnhancedCPS_2024",
        SimpleNamespace(file_path=enhanced_path),
    )
    monkeypatch.setattr(upload_module, "STORAGE_FOLDER", storage_folder)
    validated = []
    monkeypatch.setattr(
        upload_module,
        "validate_dataset",
        lambda file_path: validated.append(Path(file_path).name),
    )

    validate_built_datasets(
        require_enhanced_cps=True,
        require_small_enhanced_cps=False,
    )

    assert validated == [
        "cps_2024.h5",
        "enhanced_cps_2024.h5",
        "enhanced_cps_2024.clone_diagnostics.json",
    ]
