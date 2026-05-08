import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from policyengine_us_data.stage_contracts import (
    StageContract,
    contract_from_json,
    contract_to_json,
)
from policyengine_us_data.stage_contracts.calibration_package import (
    CALIBRATION_PACKAGE_CONTRACT_FILENAME,
    build_calibration_package_contract,
    load_calibration_package_payload,
    summarize_calibration_package,
    validate_calibration_package_contract,
    write_calibration_package_contract,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_path = tmp_path / "source_imputed_stratified_extended_cps.h5"
    db_path = tmp_path / "policy_data.db"
    package_path = tmp_path / "calibration_package.pkl"
    dataset_path.write_bytes(b"dataset")
    db_path.write_bytes(b"sqlite")
    return dataset_path, db_path, package_path


def _package() -> dict:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
            ]
        )
    )
    return {
        "X_sparse": matrix,
        "targets_df": pd.DataFrame(
            {
                "value": [100.0, 200.0],
                "domain_variable": ["state", "state"],
                "variable": ["income_tax", "snap"],
                "geo_level": ["state", "state"],
                "geographic_id": ["01", "02"],
            }
        ),
        "target_names": ["state_income_tax_01", "state_snap_02"],
        "metadata": {
            "dataset_sha256": "sha256:dataset",
            "db_sha256": "sha256:db",
            "target_config_path": "policyengine_us_data/calibration/target_config.yaml",
            "target_config_sha256": "sha256:target-config",
            "n_clones": 430,
            "seed": 42,
            "base_n_records": 1,
            "package_scope": "minimal",
            "matrix_builder": "chunked",
            "chunk_size": 25_000,
            "chunk_dir": "/pipeline/artifacts/run-a/matrix_build",
            "git_commit": "abc123",
            "package_version": "1.98.2",
        },
        "initial_weights": np.array([1.0, 1.0, 1.0]),
        "cd_geoid": np.array(["0101", "0102", "0201"]),
        "block_geoid": np.array(["010010001", "010010002", "020010001"]),
    }


def _write_package(path: Path, package: dict | None = None) -> dict:
    package = package or _package()
    with path.open("wb") as handle:
        pickle.dump(package, handle)
    return package


def _parameters() -> dict:
    return {
        "workers": None,
        "n_clones": 430,
        "target_config": "policyengine_us_data/calibration/target_config.yaml",
        "skip_county": True,
        "chunked_matrix": True,
        "chunk_size": 25_000,
        "parallel_matrix": False,
        "num_matrix_workers": None,
    }


def _contract(tmp_path: Path) -> StageContract:
    dataset_path, db_path, package_path = _write_inputs(tmp_path)
    package = _write_package(package_path)
    return build_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=_parameters(),
        run_id="run-a",
        started_at="2026-05-08T12:00:00Z",
        completed_at="2026-05-08T12:02:00Z",
        duration_s=120.0,
    )


def test_calibration_package_contract_records_stage_2_handoff(tmp_path):
    contract = _contract(tmp_path)

    assert contract.stage_id == "2_build_calibration_package"
    assert contract.contract_type == "calibration_package"
    assert contract.run_id == "run-a"
    assert contract.code_sha == "abc123"
    assert contract.package_version == "1.98.2"
    assert {artifact.logical_name for artifact in contract.inputs} == {
        "source_imputed_stratified_extended_cps",
        "policy_data_db",
    }
    assert {artifact.logical_name for artifact in contract.outputs} == {
        "calibration_package"
    }
    assert all(artifact.sha256.startswith("sha256:") for artifact in contract.inputs)
    assert contract.outputs[0].media_type == "application/python-pickle"


def test_calibration_package_contract_normalizes_empty_run_id_to_none(tmp_path):
    dataset_path, db_path, package_path = _write_inputs(tmp_path)
    package = _write_package(package_path)

    contract = build_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=_parameters(),
        run_id="",
        completed_at="2026-05-08T12:02:00Z",
    )

    assert contract.run_id is None


def test_calibration_package_contract_records_matrix_summary(tmp_path):
    contract = _contract(tmp_path)

    summary = contract.metadata["package_summary"]

    assert summary["matrix_shape"] == (2, 3)
    assert summary["matrix_nnz"] == 3
    assert summary["matrix_density"] == 0.5
    assert summary["n_targets"] == 2
    assert summary["target_name_count"] == 2
    assert summary["target_config_sha256"] == "sha256:target-config"
    assert summary["n_clones"] == 430
    assert summary["seed"] == 42
    assert summary["matrix_builder"] == "chunked"
    assert summary["has_initial_weights"] is True
    assert summary["has_cd_geoid"] is True
    assert summary["has_block_geoid"] is True
    assert summary["cd_geoid_length"] == 3
    assert summary["block_geoid_length"] == 3


def test_calibration_package_contract_records_single_substage(tmp_path):
    contract = _contract(tmp_path)

    assert len(contract.substages) == 1
    substage = contract.substages[0]
    assert substage.substage_id == "2a_matrix_build_calibration_target_construction"
    assert substage.status == "completed"
    assert substage.reuse_mode == "handoff"
    assert substage.outputs[0].logical_name == "calibration_package"
    assert substage.metadata == {}


def test_calibration_package_contract_json_round_trip_is_deterministic(tmp_path):
    contract = _contract(tmp_path)

    payload = contract_to_json(contract)
    restored = contract_from_json(payload)

    assert restored == contract
    assert contract_to_json(restored) == payload


def test_calibration_package_summary_omits_bulky_payloads():
    summary = summarize_calibration_package(_package())

    assert "X_sparse" not in summary
    assert "targets_df" not in summary
    assert "target_names" not in summary
    assert "initial_weights" not in summary
    assert "cd_geoid" not in summary
    assert "block_geoid" not in summary


def test_calibration_package_summary_handles_empty_matrix():
    package = _package()
    package["X_sparse"] = sparse.csr_matrix((0, 0))
    package["targets_df"] = package["targets_df"].iloc[0:0]
    package["target_names"] = []

    summary = summarize_calibration_package(package)

    assert summary["matrix_shape"] == (0, 0)
    assert summary["matrix_nnz"] == 0
    assert summary["matrix_density"] == 0.0


def test_calibration_package_contract_rejects_missing_artifact(tmp_path):
    dataset_path, db_path, package_path = _write_inputs(tmp_path)
    package = _package()

    try:
        build_calibration_package_contract(
            package_path=package_path,
            dataset_path=dataset_path,
            db_path=db_path,
            package=package,
            parameters=_parameters(),
            run_id="run-a",
            completed_at="2026-05-08T12:02:00Z",
        )
    except FileNotFoundError as exc:
        assert "calibration package" in str(exc)
    else:
        raise AssertionError("Missing calibration package should fail")


def test_write_and_validate_calibration_package_contract(tmp_path):
    dataset_path, db_path, package_path = _write_inputs(tmp_path)
    package = _write_package(package_path)

    contract = write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )

    contract_path = tmp_path / CALIBRATION_PACKAGE_CONTRACT_FILENAME
    assert contract_path.exists()
    validated = validate_calibration_package_contract(
        package_path=package_path,
        contract_path=contract_path,
        package=package,
        dataset_path=dataset_path,
        db_path=db_path,
    )
    assert validated == contract


def test_validate_calibration_package_contract_fails_on_stale_summary(tmp_path):
    dataset_path, db_path, package_path = _write_inputs(tmp_path)
    package = _write_package(package_path)
    write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )
    changed_package = _package()
    changed_package["X_sparse"] = sparse.csr_matrix(np.ones((3, 3)))

    try:
        validate_calibration_package_contract(
            package_path=package_path,
            package=changed_package,
        )
    except ValueError as exc:
        assert "summary does not match" in str(exc)
    else:
        raise AssertionError("Stale calibration package summary should fail")


def test_validate_calibration_package_contract_fails_on_package_checksum(tmp_path):
    dataset_path, db_path, package_path = _write_inputs(tmp_path)
    package = _write_package(package_path)
    write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )
    package_path.write_bytes(b"not the original pickle")

    try:
        validate_calibration_package_contract(package_path=package_path)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("Stale calibration package checksum should fail")


def test_validate_calibration_package_contract_requires_package_for_summary(tmp_path):
    dataset_path, db_path, package_path = _write_inputs(tmp_path)
    package = _write_package(package_path)
    write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )

    try:
        validate_calibration_package_contract(package_path=package_path)
    except ValueError as exc:
        assert "package is required" in str(exc)
    else:
        raise AssertionError("Package payload should be required for full validation")


def test_load_calibration_package_payload_rejects_non_mapping(tmp_path):
    package_path = tmp_path / "calibration_package.pkl"
    with package_path.open("wb") as handle:
        pickle.dump(["not", "a", "mapping"], handle)

    try:
        load_calibration_package_payload(package_path)
    except ValueError as exc:
        assert "must contain a mapping" in str(exc)
    else:
        raise AssertionError("Non-mapping package payload should fail")
