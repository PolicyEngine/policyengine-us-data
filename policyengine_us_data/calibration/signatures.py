"""Shared signature helpers for resumable calibration artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from policyengine_us_data.utils.manifest import compute_file_checksum


def hash_string_list(values: Iterable[str]) -> str:
    """Hash an ordered list of strings."""
    digest = hashlib.sha256()
    if values is None:
        values = []
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def hash_numpy_array(values) -> str:
    """Hash an array's shape, dtype, and contents."""
    arr = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash a dataframe's columns plus row contents deterministically."""
    digest = hashlib.sha256()
    normalized = df.copy()
    normalized.columns = [str(col) for col in normalized.columns]
    digest.update(hash_string_list(normalized.columns).encode("utf-8"))
    for dtype in normalized.dtypes:
        digest.update(str(dtype).encode("utf-8"))
        digest.update(b"\0")
    row_hashes = pd.util.hash_pandas_object(
        normalized,
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def hash_sparse_matrix(X_sparse) -> str:
    """Hash sparse matrix structure and values for resume compatibility."""
    X_csr = X_sparse.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(X_csr.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(X_csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(X_csr.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(X_csr.data).tobytes())
    return digest.hexdigest()


def signature_mismatches(
    expected: dict,
    actual: dict,
    *,
    soft_float_keys: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (fatal, soft) mismatches between two signatures."""
    fatal: list[str] = []
    soft: list[str] = []
    soft_float_keys = soft_float_keys or set()
    for key, actual_value in actual.items():
        expected_value = expected.get(key)
        if expected_value is None:
            fatal.append(f"{key} missing from stored signature")
            continue
        if key in soft_float_keys:
            if not np.isclose(expected_value, actual_value):
                soft.append(f"{key} expected {expected_value}, got {actual_value}")
        elif actual_value != expected_value:
            fatal.append(f"{key} expected {expected_value}, got {actual_value}")
    return fatal, soft


def sqlite_checksum(db_uri: str | None) -> str | None:
    """Return a checksum for a sqlite database URI when available."""
    if not db_uri:
        return None
    parsed = urlparse(db_uri)
    if parsed.scheme != "sqlite":
        return None
    db_path = Path(parsed.path)
    if not db_path.exists():
        return None
    return compute_file_checksum(db_path)


def build_checkpoint_signature(
    X_sparse,
    targets: np.ndarray,
    target_names: list,
    lambda_l0: float,
    beta: float,
    lambda_l2: float,
    learning_rate: float,
) -> dict:
    """Build a compact signature to validate calibration checkpoint resume."""
    targets_arr = np.asarray(targets, dtype=np.float64)
    return {
        "n_features": int(X_sparse.shape[1]),
        "n_targets": int(len(targets_arr)),
        "x_sparse_sha256": hash_sparse_matrix(X_sparse),
        "target_names_sha256": hash_string_list(target_names),
        "targets_sha256": hashlib.sha256(targets_arr.tobytes()).hexdigest(),
        "lambda_l0": float(lambda_l0),
        "beta": float(beta),
        "lambda_l2": float(lambda_l2),
        "learning_rate": float(learning_rate),
    }


def checkpoint_signature_mismatches(
    expected: dict,
    actual: dict,
) -> tuple[list[str], list[str]]:
    """Return (fatal, soft) checkpoint compatibility mismatches."""
    return signature_mismatches(
        expected,
        actual,
        soft_float_keys={"lambda_l0", "beta", "lambda_l2", "learning_rate"},
    )


def build_chunk_lineage_signature(
    *,
    dataset_path: str,
    db_uri: str,
    time_period: int,
    geography,
    targets_df: pd.DataFrame,
    target_names: list[str],
    chunk_size: int,
    rerandomize_takeup: bool,
) -> dict:
    """Build a signature for validating chunk cache lineage."""
    target_columns = [
        col
        for col in [
            "target_id",
            "stratum_id",
            "variable",
            "reform_id",
            "value",
            "period",
            "geographic_id",
            "geo_level",
        ]
        if col in targets_df.columns
    ]
    target_frame = targets_df[target_columns].copy()
    return {
        "format_version": 1,
        "dataset_sha256": compute_file_checksum(Path(dataset_path)),
        "db_sha256": sqlite_checksum(db_uri),
        "time_period": int(time_period),
        "n_records": int(geography.n_records),
        "n_clones": int(geography.n_clones),
        "n_columns": int(geography.n_records * geography.n_clones),
        "chunk_size": int(chunk_size),
        "rerandomize_takeup": bool(rerandomize_takeup),
        "target_names_sha256": hash_string_list(target_names),
        "targets_sha256": hash_dataframe(target_frame),
        "state_fips_sha256": hash_numpy_array(geography.state_fips),
        "county_fips_sha256": hash_numpy_array(geography.county_fips),
        "cd_geoid_sha256": hash_numpy_array(geography.cd_geoid),
        "block_geoid_sha256": hash_numpy_array(geography.block_geoid),
    }
