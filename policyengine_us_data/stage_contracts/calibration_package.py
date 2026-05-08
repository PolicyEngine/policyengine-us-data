"""Stage 2 calibration-package contract assembly and validation."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policyengine_us_data.utils.step_manifest import sha256_file

from .artifacts import ArtifactRef
from .contracts import StageContract
from .execution import ExecutionRecord, ReuseSummary
from .fingerprints import canonicalize_for_fingerprint, fingerprint_material
from .io import read_contract, write_contract
from .stages import STAGE_2_BUILD_CALIBRATION_PACKAGE, contract_type_for_stage
from .substages import SubstageRecord

CALIBRATION_PACKAGE_CONTRACT_FILENAME = "calibration_package_contract.json"
CALIBRATION_PACKAGE_CONTRACT_TYPE = contract_type_for_stage(
    STAGE_2_BUILD_CALIBRATION_PACKAGE
)
CALIBRATION_PACKAGE_SUBSTAGE_ID = "2a_matrix_build_calibration_target_construction"


def summarize_calibration_package(package: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a contract-safe summary of a calibration package pickle payload."""

    matrix = _required_package_value(package, "X_sparse")
    targets_df = _required_package_value(package, "targets_df")
    target_names = _required_package_value(package, "target_names")
    metadata = _package_metadata(package)

    try:
        n_targets, n_columns = matrix.shape
    except (AttributeError, ValueError) as exc:
        raise ValueError("X_sparse must expose a two-dimensional shape") from exc
    if not hasattr(matrix, "nnz"):
        raise ValueError("X_sparse must expose nnz")

    n_targets = int(n_targets)
    n_columns = int(n_columns)
    nnz = int(matrix.nnz)
    density = nnz / (n_targets * n_columns) if n_targets * n_columns else 0.0

    summary: dict[str, Any] = {
        "matrix_shape": (n_targets, n_columns),
        "matrix_nnz": nnz,
        "matrix_density": float(density),
        "n_targets": int(len(targets_df)),
        "n_columns": n_columns,
        "target_name_count": int(len(target_names)),
        "dataset_sha256": _optional_metadata_string(metadata, "dataset_sha256"),
        "db_sha256": _optional_metadata_string(metadata, "db_sha256"),
        "target_config_path": _optional_metadata_string(
            metadata,
            "target_config_path",
        ),
        "target_config_sha256": _optional_metadata_string(
            metadata,
            "target_config_sha256",
        ),
        "n_clones": _optional_metadata_int(metadata, "n_clones"),
        "seed": _optional_metadata_int(metadata, "seed"),
        "base_n_records": _optional_metadata_int(metadata, "base_n_records"),
        "package_scope": _optional_metadata_string(metadata, "package_scope"),
        "matrix_builder": _optional_metadata_string(metadata, "matrix_builder"),
        "chunk_size": _optional_metadata_int(metadata, "chunk_size"),
        "chunk_dir": _optional_metadata_string(metadata, "chunk_dir"),
        "has_initial_weights": package.get("initial_weights") is not None,
        "has_cd_geoid": package.get("cd_geoid") is not None,
        "has_block_geoid": package.get("block_geoid") is not None,
        "cd_geoid_length": _optional_len(package.get("cd_geoid")),
        "block_geoid_length": _optional_len(package.get("block_geoid")),
    }
    return summary


def build_calibration_package_contract(
    *,
    package_path: Path,
    dataset_path: Path,
    db_path: Path,
    package: Mapping[str, Any],
    parameters: Mapping[str, Any],
    run_id: str | None,
    completed_at: str,
    started_at: str | None = None,
    duration_s: float | None = None,
    code_sha: str | None = None,
    package_version: str | None = None,
) -> StageContract:
    """Build the Stage 2 handoff contract from a calibration package."""

    package_path = Path(package_path)
    dataset_path = Path(dataset_path)
    db_path = Path(db_path)
    _require_existing_file(package_path, "calibration package")
    _require_existing_file(dataset_path, "source dataset")
    _require_existing_file(db_path, "target database")

    metadata = _package_metadata(package)
    package_summary = summarize_calibration_package(package)
    inputs = (
        _artifact_ref_from_path(
            logical_name="source_imputed_stratified_extended_cps",
            path=dataset_path,
            metadata={
                "artifact_family": "dataset",
                "substage_id": CALIBRATION_PACKAGE_SUBSTAGE_ID,
                "required_for_stage_2": True,
            },
        ),
        _artifact_ref_from_path(
            logical_name="policy_data_db",
            path=db_path,
            metadata={
                "artifact_family": "target_database",
                "substage_id": CALIBRATION_PACKAGE_SUBSTAGE_ID,
                "required_for_stage_2": True,
            },
        ),
    )
    outputs = (
        _artifact_ref_from_path(
            logical_name="calibration_package",
            path=package_path,
            metadata={
                "artifact_family": "calibration_package",
                "substage_id": CALIBRATION_PACKAGE_SUBSTAGE_ID,
            },
        ),
    )
    code_sha = code_sha or _optional_metadata_string(metadata, "git_commit")
    package_version = package_version or _optional_metadata_string(
        metadata,
        "package_version",
    )
    execution = ExecutionRecord(
        status="completed",
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        reuse_decision="not_applicable",
        reuse_summary=ReuseSummary(
            expected_outputs=1,
            valid_reused_outputs=0,
            recomputed_outputs=1,
            invalid_outputs=0,
        ),
    )
    fingerprint = fingerprint_material(
        {
            "stage_id": STAGE_2_BUILD_CALIBRATION_PACKAGE,
            "contract_type": CALIBRATION_PACKAGE_CONTRACT_TYPE,
            "inputs": inputs,
            "outputs": outputs,
            "parameters": parameters,
            "package_summary": package_summary,
        }
    )
    return StageContract(
        contract_type=CALIBRATION_PACKAGE_CONTRACT_TYPE,
        stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
        run_id=run_id,
        created_at=completed_at,
        code_sha=code_sha,
        package_version=package_version,
        inputs=inputs,
        outputs=outputs,
        parameters=parameters,
        fingerprint=fingerprint,
        substages=(
            SubstageRecord(
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                status="completed",
                inputs=inputs,
                outputs=outputs,
                parameters=parameters,
                fingerprint=fingerprint,
                reuse_mode="handoff",
                metadata={"package_summary": package_summary},
            ),
        ),
        execution=execution,
        metadata={
            "artifact_count": len(inputs) + len(outputs),
            "contract_file": CALIBRATION_PACKAGE_CONTRACT_FILENAME,
            "package_summary": package_summary,
        },
    )


def write_calibration_package_contract(
    *,
    package_path: Path,
    dataset_path: Path,
    db_path: Path,
    package: Mapping[str, Any],
    parameters: Mapping[str, Any],
    run_id: str | None,
    completed_at: str,
    started_at: str | None = None,
    duration_s: float | None = None,
    code_sha: str | None = None,
    package_version: str | None = None,
    contract_path: Path | None = None,
) -> StageContract:
    """Write and return the Stage 2 calibration-package contract."""

    package_path = Path(package_path)
    contract = build_calibration_package_contract(
        package_path=package_path,
        dataset_path=Path(dataset_path),
        db_path=Path(db_path),
        package=package,
        parameters=parameters,
        run_id=run_id,
        completed_at=completed_at,
        started_at=started_at,
        duration_s=duration_s,
        code_sha=code_sha,
        package_version=package_version,
    )
    write_contract(
        contract,
        contract_path or package_path.with_name(CALIBRATION_PACKAGE_CONTRACT_FILENAME),
    )
    return contract


def validate_calibration_package_contract(
    *,
    package_path: Path,
    contract_path: Path | None = None,
    package: Mapping[str, Any] | None = None,
    dataset_path: Path | None = None,
    db_path: Path | None = None,
) -> StageContract:
    """Validate that a Stage 2 sidecar describes the calibration package."""

    package_path = Path(package_path)
    contract_path = contract_path or package_path.with_name(
        CALIBRATION_PACKAGE_CONTRACT_FILENAME
    )
    _require_existing_file(package_path, "calibration package")
    _require_existing_file(contract_path, "calibration package contract")

    contract = read_contract(contract_path)
    if contract.stage_id != STAGE_2_BUILD_CALIBRATION_PACKAGE:
        raise ValueError(f"Invalid Stage 2 contract stage_id: {contract.stage_id!r}")
    if contract.contract_type != CALIBRATION_PACKAGE_CONTRACT_TYPE:
        raise ValueError(
            "Invalid Stage 2 contract type: "
            f"{contract.contract_type!r}"
        )
    _assert_artifact_matches_file(
        _single_artifact(contract.outputs, "calibration_package"),
        package_path,
    )
    if dataset_path is not None:
        _assert_artifact_matches_file(
            _single_artifact(
                contract.inputs,
                "source_imputed_stratified_extended_cps",
            ),
            Path(dataset_path),
        )
    if db_path is not None:
        _assert_artifact_matches_file(
            _single_artifact(contract.inputs, "policy_data_db"),
            Path(db_path),
        )
    if package is None:
        return contract

    expected_summary = canonicalize_for_fingerprint(
        summarize_calibration_package(package)
    )
    actual_summary = canonicalize_for_fingerprint(
        contract.metadata.get("package_summary", {})
    )
    if actual_summary != expected_summary:
        raise ValueError("Calibration package contract summary does not match pickle")
    return contract


def load_calibration_package_payload(package_path: Path) -> Mapping[str, Any]:
    """Load a calibration package pickle for sidecar validation."""

    with Path(package_path).open("rb") as handle:
        package = pickle.load(handle)
    if not isinstance(package, Mapping):
        raise ValueError("Calibration package pickle must contain a mapping")
    return package


def _required_package_value(package: Mapping[str, Any], key: str) -> Any:
    if key not in package:
        raise ValueError(f"Calibration package missing required key: {key}")
    return package[key]


def _package_metadata(package: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = package.get("metadata", {})
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("Calibration package metadata must be a mapping")
    return metadata


def _optional_metadata_string(
    metadata: Mapping[str, Any],
    key: str,
) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    return str(value)


def _optional_metadata_int(metadata: Mapping[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Calibration package metadata {key!r} must be an integer")
    return int(value)


def _optional_len(value: Any) -> int | None:
    if value is None:
        return None
    return int(len(value))


def _require_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def _artifact_ref_from_path(
    *,
    logical_name: str,
    path: Path,
    metadata: Mapping[str, Any],
) -> ArtifactRef:
    return ArtifactRef(
        logical_name=logical_name,
        uri=path.resolve().as_uri(),
        sha256=f"sha256:{sha256_file(path)}",
        size_bytes=path.stat().st_size,
        media_type=_media_type_for_path(path),
        metadata=metadata,
    )


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".h5":
        return "application/x-hdf5"
    if suffix == ".db":
        return "application/vnd.sqlite3"
    if suffix == ".json":
        return "application/json"
    if suffix == ".pkl":
        return "application/python-pickle"
    return "application/octet-stream"


def _single_artifact(
    artifacts: tuple[ArtifactRef, ...],
    logical_name: str,
) -> ArtifactRef:
    matches = [
        artifact for artifact in artifacts if artifact.logical_name == logical_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one artifact named {logical_name!r}, found {len(matches)}"
        )
    return matches[0]


def _assert_artifact_matches_file(artifact: ArtifactRef, path: Path) -> None:
    _require_existing_file(path, artifact.logical_name)
    expected_sha = f"sha256:{sha256_file(path)}"
    if artifact.sha256 != expected_sha:
        raise ValueError(
            f"Artifact {artifact.logical_name!r} checksum mismatch: "
            f"{artifact.sha256!r} != {expected_sha!r}"
        )
    if artifact.size_bytes != path.stat().st_size:
        raise ValueError(
            f"Artifact {artifact.logical_name!r} size mismatch: "
            f"{artifact.size_bytes!r} != {path.stat().st_size!r}"
        )
