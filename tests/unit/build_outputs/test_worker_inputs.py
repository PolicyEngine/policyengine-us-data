from pathlib import Path

import pytest

from policyengine_us_data.build_outputs.worker_inputs import (
    WorkerCalibrationInputs,
)


def test_worker_calibration_inputs_round_trip_wire_payload():
    inputs = WorkerCalibrationInputs(
        weights_path=Path("/tmp/calibration_weights.npy"),
        dataset_path=Path("/tmp/source.h5"),
        database_path=Path("/tmp/policy_data.db"),
        geography_path=Path("/tmp/geography_assignment.npz"),
        calibration_package_path=Path("/tmp/calibration_package.pkl"),
        run_config_path=Path("/tmp/unified_run_config.json"),
        n_clones=4,
        seed=123,
    )

    payload = inputs.to_wire_dict()
    normalized = WorkerCalibrationInputs.from_wire_dict(payload)

    assert normalized == inputs
    assert payload == {
        "weights": "/tmp/calibration_weights.npy",
        "dataset": "/tmp/source.h5",
        "database": "/tmp/policy_data.db",
        "geography": "/tmp/geography_assignment.npz",
        "calibration_package": "/tmp/calibration_package.pkl",
        "run_config": "/tmp/unified_run_config.json",
        "n_clones": 4,
        "seed": 123,
    }


def test_worker_calibration_inputs_defaults_legacy_modal_payload_values():
    inputs = WorkerCalibrationInputs.from_wire_dict(
        {
            "weights": "/tmp/calibration_weights.npy",
            "dataset": "/tmp/source.h5",
            "database": "/tmp/policy_data.db",
        }
    )

    assert inputs.n_clones == 430
    assert inputs.seed == 42


def test_worker_calibration_inputs_build_worker_cli_args():
    inputs = WorkerCalibrationInputs(
        weights_path=Path("/tmp/calibration_weights.npy"),
        dataset_path=Path("/tmp/source.h5"),
        database_path=Path("/tmp/policy_data.db"),
        geography_path=Path("/tmp/geography_assignment.npz"),
        calibration_package_path=Path("/tmp/calibration_package.pkl"),
        run_config_path=Path("/tmp/unified_run_config.json"),
        n_clones=4,
        seed=123,
    )

    assert inputs.to_worker_cli_args() == [
        "--weights-path",
        "/tmp/calibration_weights.npy",
        "--dataset-path",
        "/tmp/source.h5",
        "--db-path",
        "/tmp/policy_data.db",
        "--n-clones",
        "4",
        "--seed",
        "123",
        "--geography-path",
        "/tmp/geography_assignment.npz",
        "--calibration-package-path",
        "/tmp/calibration_package.pkl",
        "--run-config-path",
        "/tmp/unified_run_config.json",
    ]


def test_worker_calibration_inputs_build_publishing_input_bundle():
    inputs = WorkerCalibrationInputs(
        weights_path=Path("/tmp/calibration_weights.npy"),
        dataset_path=Path("/tmp/source.h5"),
        database_path=Path("/tmp/policy_data.db"),
        geography_path=Path("/tmp/geography_assignment.npz"),
        calibration_package_path=Path("/tmp/calibration_package.pkl"),
        run_config_path=Path("/tmp/unified_run_config.json"),
        n_clones=4,
        seed=123,
    )

    bundle = inputs.to_publishing_input_bundle(
        run_id="run-123",
        version="1.2.3",
        legacy_blocks_path=Path("/tmp/stacked_blocks.npy"),
    )

    assert bundle.weights_path == inputs.weights_path
    assert bundle.source_dataset_path == inputs.dataset_path
    assert bundle.target_db_path == inputs.database_path
    assert bundle.exact_geography_path == inputs.geography_path
    assert bundle.calibration_package_path == inputs.calibration_package_path
    assert bundle.run_config_path == inputs.run_config_path
    assert bundle.run_id == "run-123"
    assert bundle.version == "1.2.3"
    assert bundle.n_clones == 4
    assert bundle.seed == 123
    assert bundle.legacy_blocks_path == Path("/tmp/stacked_blocks.npy")


def test_worker_calibration_inputs_omit_missing_optional_artifact_paths(tmp_path):
    run_config_path = tmp_path / "unified_run_config.json"
    run_config_path.write_text("{}")

    inputs = WorkerCalibrationInputs.from_artifact_paths(
        weights_path=tmp_path / "calibration_weights.npy",
        dataset_path=tmp_path / "source.h5",
        database_path=tmp_path / "policy_data.db",
        geography_path=tmp_path / "missing_geography_assignment.npz",
        calibration_package_path=tmp_path / "missing_calibration_package.pkl",
        run_config_path=run_config_path,
        n_clones=4,
        seed=123,
    )

    payload = inputs.to_wire_dict()

    assert inputs.geography_path is None
    assert inputs.calibration_package_path is None
    assert inputs.run_config_path == run_config_path
    assert "geography" not in payload
    assert "calibration_package" not in payload
    assert payload["run_config"] == str(run_config_path)


def test_worker_calibration_inputs_reject_missing_required_paths():
    with pytest.raises(KeyError, match="weights"):
        WorkerCalibrationInputs.from_wire_dict(
            {
                "dataset": "/tmp/source.h5",
                "database": "/tmp/policy_data.db",
            }
        )
