"""Fixture helpers for Stage 2 calibration package contract tests."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from policyengine_us_data.stage_contracts import StageContract
from policyengine_us_data.stage_contracts.calibration_package import (
    build_calibration_package_contract,
)

__test__ = False

CALIBRATION_RUN_ID = "run-a"
CALIBRATION_STARTED_AT = "2026-05-08T12:00:00Z"
CALIBRATION_COMPLETED_AT = "2026-05-08T12:02:00Z"
CALIBRATION_DURATION_S = 120.0
TARGET_CONFIG_PATH = "policyengine_us_data/calibration/target_config.yaml"

CALIBRATION_BLOCK_GEOIDS = ("010010001", "010010002", "020010001")
CALIBRATION_CD_GEOIDS = ("0101", "0102", "0201")
CHANGED_CALIBRATION_BLOCK_GEOIDS = ("030010001", "010010002", "020010001")


def contract_input_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write minimal Stage 2 input artifacts and return their paths."""

    dataset_path = tmp_path / "source_imputed_stratified_extended_cps.h5"
    db_path = tmp_path / "policy_data.db"
    package_path = tmp_path / "calibration_package.pkl"
    dataset_path.write_bytes(b"dataset")
    db_path.write_bytes(b"sqlite")
    return dataset_path, db_path, package_path


def calibration_package_payload() -> dict[str, Any]:
    """Return a small calibration package payload."""

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
            "target_config_path": TARGET_CONFIG_PATH,
            "target_config_sha256": "sha256:target-config",
            "n_clones": 3,
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
        "cd_geoid": np.array(CALIBRATION_CD_GEOIDS),
        "block_geoid": np.array(CALIBRATION_BLOCK_GEOIDS),
    }


def calibration_package_payload_without_geography() -> dict[str, Any]:
    """Return a calibration package payload without geography assignment arrays."""

    package = calibration_package_payload()
    package.pop("block_geoid")
    package.pop("cd_geoid")
    return package


def calibration_package_payload_with_block_geoids(
    block_geoids: tuple[str, ...] = CHANGED_CALIBRATION_BLOCK_GEOIDS,
) -> dict[str, Any]:
    """Return a calibration package payload with custom block GEOIDs."""

    package = calibration_package_payload()
    package["block_geoid"] = np.array(block_geoids)
    return package


def calibration_package_payload_with_cd_geoids(
    cd_geoids: tuple[str, ...],
) -> dict[str, Any]:
    """Return a calibration package payload with custom CD GEOIDs."""

    package = calibration_package_payload()
    package["cd_geoid"] = np.array(cd_geoids)
    return package


def empty_matrix_calibration_package_payload() -> dict[str, Any]:
    """Return a calibration package payload with an empty target matrix."""

    package = calibration_package_payload()
    package["X_sparse"] = sparse.csr_matrix((0, 0))
    package["targets_df"] = package["targets_df"].iloc[0:0]
    package["target_names"] = []
    return package


def write_calibration_package_payload(
    path: Path,
    package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write and return a calibration package payload."""

    package = package or calibration_package_payload()
    with path.open("wb") as handle:
        pickle.dump(package, handle)
    return package


def write_non_mapping_calibration_package_payload(path: Path) -> None:
    """Write an invalid calibration package payload for schema tests."""

    with path.open("wb") as handle:
        pickle.dump(["not", "a", "mapping"], handle)


def calibration_package_parameters() -> dict[str, Any]:
    """Return canonical Stage 2 calibration package runtime parameters."""

    return {
        "workers": None,
        "n_clones": 3,
        "target_config": TARGET_CONFIG_PATH,
        "skip_county": True,
        "skip_source_impute": True,
        "skip_takeup_rerandomize": False,
        "chunked_matrix": True,
        "chunk_size": 25_000,
        "parallel_matrix": False,
        "num_matrix_workers": None,
    }


def calibration_package_contract(tmp_path: Path) -> StageContract:
    """Write fixture inputs and return a Stage 2 calibration package contract."""

    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)
    return build_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id=CALIBRATION_RUN_ID,
        started_at=CALIBRATION_STARTED_AT,
        completed_at=CALIBRATION_COMPLETED_AT,
        duration_s=CALIBRATION_DURATION_S,
    )
