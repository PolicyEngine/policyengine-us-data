"""Stage 2 calibration-package contract assembly and validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policyengine_us_data.calibration_package.payload import (
    CalibrationPackagePayload,
    CalibrationPackageReader,
)
from policyengine_us_data.calibration_package.specs import (
    CALIBRATION_PACKAGE_CONTRACT_FILENAME,
    CALIBRATION_PACKAGE_SUBSTAGE_ID,
)
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.utils.step_manifest import sha256_file

from .artifacts import ArtifactRef
from .calibration_package_schema import (
    CalibrationPackageParameters,
    CalibrationPackageSummary,
    GeographyAssignmentSummary,
)
from .contracts import StageContract
from .execution import ExecutionRecord, ReuseSummary
from .fingerprints import canonicalize_for_fingerprint, fingerprint_material
from .io import read_contract, write_contract
from .stages import STAGE_2_BUILD_CALIBRATION_PACKAGE, contract_type_for_stage
from .substages import SubstageRecord

CALIBRATION_PACKAGE_CONTRACT_TYPE = contract_type_for_stage(
    STAGE_2_BUILD_CALIBRATION_PACKAGE
)


def summarize_geography_assignment(
    package: CalibrationPackagePayload | Mapping[str, Any],
) -> GeographyAssignmentSummary:
    """Return a contract-safe summary of package-backed geography assignment."""

    return _calibration_package_payload(package, require_core=False).geography_summary()


def summarize_calibration_package(
    package: CalibrationPackagePayload | Mapping[str, Any],
) -> CalibrationPackageSummary:
    """Return a contract-safe summary of a calibration package pickle payload."""

    return _calibration_package_payload(package).summary()


def build_calibration_package_contract(
    *,
    package_path: Path,
    dataset_path: Path,
    db_path: Path,
    package: CalibrationPackagePayload | Mapping[str, Any],
    parameters: CalibrationPackageParameters | Mapping[str, Any],
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

    parameter_schema = _calibration_package_parameters(parameters)
    payload = _calibration_package_payload(package)
    metadata = payload.metadata
    parameter_payload = _parameters_with_package_identity(
        parameter_schema.to_dict(),
        metadata,
    )
    package_summary = payload.summary().to_dict()
    geography_summary = payload.geography_summary().to_dict()
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
            "parameters": parameter_payload,
            "package_summary": package_summary,
            "geography_assignment": geography_summary,
        }
    )
    return StageContract(
        contract_type=CALIBRATION_PACKAGE_CONTRACT_TYPE,
        stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
        run_id=run_id or None,
        created_at=completed_at,
        code_sha=code_sha,
        package_version=package_version,
        inputs=inputs,
        outputs=outputs,
        parameters=parameter_payload,
        fingerprint=fingerprint,
        substages=(
            SubstageRecord(
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                status="completed",
                inputs=inputs,
                outputs=outputs,
                parameters=parameter_payload,
                fingerprint=fingerprint,
                reuse_mode="handoff",
            ),
        ),
        execution=execution,
        metadata={
            "artifact_count": len(inputs) + len(outputs),
            "contract_file": CALIBRATION_PACKAGE_CONTRACT_FILENAME,
            "geography_assignment": geography_summary,
            "package_summary": package_summary,
        },
    )


@pipeline_node(
    PipelineNode(
        id="stage2_calibration_package_contract_writer",
        label="Stage 2 Contract Writer",
        node_type="library",
        description="Write the Stage 2 calibration-package handoff contract next to the package artifact.",
        source_file="policyengine_us_data/stage_contracts/calibration_package.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=["calibration_package.pkl"],
        artifacts_out=[CALIBRATION_PACKAGE_CONTRACT_FILENAME],
        validation_commands=[
            "uv run pytest tests/unit/test_calibration_package_stage_contract.py"
        ],
    )
)
def write_calibration_package_contract(
    *,
    package_path: Path,
    dataset_path: Path,
    db_path: Path,
    package: CalibrationPackagePayload | Mapping[str, Any],
    parameters: CalibrationPackageParameters | Mapping[str, Any],
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


@pipeline_node(
    PipelineNode(
        id="stage2_calibration_package_contract_validator",
        label="Stage 2 Contract Validator",
        node_type="validation",
        description="Validate that the persisted Stage 2 contract describes the calibration package and inputs.",
        source_file="policyengine_us_data/stage_contracts/calibration_package.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=[
            "calibration_package.pkl",
            CALIBRATION_PACKAGE_CONTRACT_FILENAME,
        ],
        validation_commands=[
            "uv run pytest tests/unit/test_calibration_package_stage_contract.py"
        ],
    )
)
def validate_calibration_package_contract(
    *,
    package_path: Path,
    contract_path: Path | None = None,
    package: CalibrationPackagePayload | Mapping[str, Any] | None = None,
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
        raise ValueError(f"Invalid Stage 2 contract type: {contract.contract_type!r}")
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
        raise ValueError("package is required to validate calibration package summary")

    expected_summary = canonicalize_for_fingerprint(
        summarize_calibration_package(package).to_dict()
    )
    actual_summary = canonicalize_for_fingerprint(
        CalibrationPackageSummary.from_dict(
            contract.metadata.get("package_summary", {})
        ).to_dict()
    )
    if actual_summary != expected_summary:
        raise ValueError("Calibration package contract summary does not match pickle")
    expected_geography = canonicalize_for_fingerprint(
        summarize_geography_assignment(package).to_dict()
    )
    actual_geography = canonicalize_for_fingerprint(
        GeographyAssignmentSummary.from_dict(
            contract.metadata.get("geography_assignment", {})
        ).to_dict()
    )
    if actual_geography != expected_geography:
        raise ValueError(
            "Calibration package contract geography assignment does not match pickle"
        )
    return contract


def validate_persisted_calibration_package_contract(
    *,
    package_path: Path,
    contract_path: Path | None = None,
    dataset_path: Path | None = None,
    db_path: Path | None = None,
) -> StageContract:
    """Validate a persisted Stage 2 sidecar against its pickle payload."""

    package = load_calibration_package_payload(package_path)
    return validate_calibration_package_contract(
        package_path=package_path,
        contract_path=contract_path,
        package=package,
        dataset_path=dataset_path,
        db_path=db_path,
    )


def load_calibration_package_payload(package_path: Path) -> CalibrationPackagePayload:
    """Load a typed calibration package payload for sidecar validation."""

    return CalibrationPackageReader(package_path=Path(package_path)).read()


def _calibration_package_payload(
    package: CalibrationPackagePayload | Mapping[str, Any],
    *,
    require_core: bool = True,
) -> CalibrationPackagePayload:
    if isinstance(package, CalibrationPackagePayload):
        return package
    return CalibrationPackagePayload.from_mapping(
        package,
        require_required_keys=require_core,
    )


def _package_metadata(
    package: CalibrationPackagePayload | Mapping[str, Any],
) -> Mapping[str, Any]:
    return _calibration_package_payload(package).metadata


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


def _calibration_package_parameters(
    parameters: CalibrationPackageParameters | Mapping[str, Any],
) -> CalibrationPackageParameters:
    if isinstance(parameters, CalibrationPackageParameters):
        return parameters
    return CalibrationPackageParameters.from_dict(parameters)


def _parameters_with_package_identity(
    parameters: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(parameters)
    metadata_path = _optional_metadata_string(metadata, "target_config_path")
    metadata_sha = _optional_metadata_string(metadata, "target_config_sha256")
    metadata_mode = _optional_metadata_string(metadata, "target_config_mode")

    if metadata_path:
        if payload.get("target_config") is None:
            payload["target_config"] = metadata_path
        if payload["target_config"] != metadata_path:
            raise ValueError(
                "Calibration package contract target_config does not match "
                "package metadata"
            )
    if metadata_sha:
        if payload.get("target_config_sha256") is None:
            payload["target_config_sha256"] = metadata_sha
        if payload["target_config_sha256"] != metadata_sha:
            raise ValueError(
                "Calibration package contract target_config_sha256 does not match "
                "package metadata"
            )
    if metadata_mode:
        if payload.get("target_config_mode") is None:
            payload["target_config_mode"] = metadata_mode
        if payload["target_config_mode"] != metadata_mode:
            raise ValueError(
                "Calibration package contract target_config_mode does not match "
                "package metadata"
            )
    if payload.get("target_config_mode") is None:
        payload["target_config_mode"] = (
            "all_active_targets" if payload.get("target_config") is None else "explicit"
        )
    return CalibrationPackageParameters.from_dict(payload).to_dict()


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
