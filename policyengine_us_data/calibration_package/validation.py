"""Stage 2 calibration-package validation service."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from policyengine_us_data.calibration_package.payload import (
    CalibrationPackagePayload,
    CalibrationPackageReader,
)
from policyengine_us_data.calibration_package.targets import target_facets_from_rows
from policyengine_us_data.calibration_package.specs import (
    CALIBRATION_PACKAGE_SUBSTAGE_ID,
    STAGE2_VALIDATION_FINDINGS_FILENAME,
    STAGE2_VALIDATION_REPORT_FILENAME,
    STAGE2_VALIDATION_SUMMARY_FILENAME,
)
from policyengine_us_data.calibration_package.matrix import (
    ChunkCacheManifest,
)
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    StageContract,
    ValidationFinding,
    ValidationReport,
)
from policyengine_us_data.stage_contracts.calibration_package import (
    validate_persisted_calibration_package_contract,
)
from policyengine_us_data.stage_contracts.calibration_package_schema import (
    MatrixBuildSummary,
)
from policyengine_us_data.stage_contracts.io import read_contract, write_contract
from policyengine_us_data.stage_contracts.stages import (
    STAGE_2_BUILD_CALIBRATION_PACKAGE,
)
from policyengine_us_data.utils.manifest import compute_file_checksum
from policyengine_us_data.utils.step_manifest import sha256_file
from policyengine_us_data.validation_core import (
    ValidationArtifactResolver,
    ValidationCheck,
    ValidationContext,
    ValidationResultWriter,
    ValidationRunner,
    ValidationSuite,
)

__all__ = [
    "CalibrationPackageValidationError",
    "CalibrationPackageValidator",
    "format_validation_report",
]

_CHECK_PACKAGE_LOADABLE = "stage2.package.loadable"
_CHECK_CONTRACT_MATCHES = "stage2.contract.matches_package"
_CHECK_TARGET_CONFIG = "stage2.target_config.identity"
_CHECK_MATRIX = "stage2.matrix.consistency"
_CHECK_TARGET_FRAME = "stage2.target_frame.consistency"
_CHECK_TARGET_METADATA = "stage2.target_metadata.consistency"
_CHECK_GEOGRAPHY = "stage2.geography.consistency"
_CHECK_INITIAL_WEIGHTS = "stage2.initial_weights.consistency"
_CHECK_CHUNK_MANIFEST = "stage2.chunk_manifest.consistency"
_TARGET_FRAME_REQUIRED_COLUMNS = frozenset(
    {"value", "domain_variable", "variable", "geo_level", "geographic_id"}
)
_TARGET_CONFIG_IDENTITY_MODES = frozenset({"default", "explicit", "all_active_targets"})


class CalibrationPackageValidationError(RuntimeError):
    """Raised when Stage 2 calibration-package validation fails."""

    def __init__(self, report: ValidationReport) -> None:
        failing_ids = tuple(
            finding.check_id for finding in report.findings if finding.status == "fail"
        )
        message = "Stage 2 calibration package validation failed"
        if failing_ids:
            message += ": " + ", ".join(failing_ids)
        super().__init__(message)
        self.report = report
        self.failing_ids = failing_ids


@pipeline_node(
    PipelineNode(
        id="stage2_calibration_package_validator",
        label="Stage 2 Calibration Package Validator",
        node_type="validation",
        description="Validate Stage 2 package, target, matrix, geography, chunk, and contract artifacts through the shared validation core.",
        source_file="policyengine_us_data/calibration_package/validation.py",
        status="current",
        stability="moving",
        pathways=["calibration_package", "cross_stage_validation"],
        artifacts_in=[
            "calibration_package.pkl",
            "calibration_package_contract.json",
            "calibration_targets.jsonl",
            "calibration_target_facets.json",
            "geography_assignment_summary.json",
            "matrix_summary.json",
        ],
        artifacts_out=[
            STAGE2_VALIDATION_REPORT_FILENAME,
            STAGE2_VALIDATION_FINDINGS_FILENAME,
            STAGE2_VALIDATION_SUMMARY_FILENAME,
        ],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_validation.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class CalibrationPackageValidator:
    """Validate Stage 2 calibration package artifacts with canonical reports."""

    runner: ValidationRunner = field(default_factory=ValidationRunner)

    def validate(
        self,
        *,
        package_path: str | Path,
        contract_path: str | Path,
        dataset_path: str | Path,
        db_path: str | Path,
        targets_path: str | Path | None = None,
        target_facets_path: str | Path | None = None,
        geography_summary_path: str | Path | None = None,
        matrix_summary_path: str | Path | None = None,
        run_id: str | None = None,
    ) -> ValidationReport:
        """Return a canonical validation report for Stage 2 artifacts."""

        paths = {
            "calibration_package": Path(package_path),
            "calibration_package_contract": Path(contract_path),
            "source_dataset": Path(dataset_path),
            "target_database": Path(db_path),
        }
        optional_paths = {
            "calibration_targets": targets_path,
            "calibration_target_facets": target_facets_path,
            "geography_assignment_summary": geography_summary_path,
            "matrix_summary": matrix_summary_path,
        }
        for logical_name, path in optional_paths.items():
            if path is not None:
                paths[logical_name] = Path(path)

        artifacts = _artifact_refs(paths)
        context = ValidationContext(
            run_id=run_id or "stage2-calibration-package",
            stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
            substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
            resolver=ValidationArtifactResolver(artifacts=artifacts),
            metadata={
                "package_path": str(package_path),
                "contract_path": str(contract_path),
            },
        )
        cache: dict[str, Any] = {}
        return self.runner.run(_validation_suite(cache), context)

    def validate_and_write(
        self,
        *,
        package_path: str | Path,
        contract_path: str | Path,
        dataset_path: str | Path,
        db_path: str | Path,
        reports_dir: str | Path,
        targets_path: str | Path | None = None,
        target_facets_path: str | Path | None = None,
        geography_summary_path: str | Path | None = None,
        matrix_summary_path: str | Path | None = None,
        run_id: str | None = None,
        attach_to_contract: bool = True,
    ) -> ValidationReport:
        """Validate, write report artifacts, and optionally attach the report."""

        report = self.validate(
            package_path=package_path,
            contract_path=contract_path,
            dataset_path=dataset_path,
            db_path=db_path,
            targets_path=targets_path,
            target_facets_path=target_facets_path,
            geography_summary_path=geography_summary_path,
            matrix_summary_path=matrix_summary_path,
            run_id=run_id,
        )
        paths = ValidationResultWriter(
            output_dir=Path(reports_dir),
            report_filename=STAGE2_VALIDATION_REPORT_FILENAME,
            findings_filename=STAGE2_VALIDATION_FINDINGS_FILENAME,
            summary_filename=STAGE2_VALIDATION_SUMMARY_FILENAME,
        ).write(report)
        if attach_to_contract:
            attach_validation_report_to_contract(
                contract_path=Path(contract_path),
                report=report,
                validation_paths=paths,
            )
        return report

    def raise_for_failure(self, report: ValidationReport) -> None:
        """Raise with failing check IDs when ``report`` failed."""

        if report.status == "fail":
            raise CalibrationPackageValidationError(report)


def attach_validation_report_to_contract(
    *,
    contract_path: Path,
    report: ValidationReport,
    validation_paths: Mapping[str, Path],
) -> StageContract:
    """Attach validation output to a Stage 2 contract and rewrite it."""

    contract = read_contract(contract_path)
    validation_artifacts = {
        key: str(path) for key, path in sorted(validation_paths.items())
    }
    metadata = {
        **dict(contract.metadata),
        "validation_artifacts": validation_artifacts,
    }
    substages = tuple(
        replace(substage, validation=report)
        if substage.substage_id == CALIBRATION_PACKAGE_SUBSTAGE_ID
        else substage
        for substage in contract.substages
    )
    updated = replace(
        contract,
        validation=report,
        substages=substages,
        metadata=metadata,
    )
    write_contract(updated, contract_path)
    return updated


def format_validation_report(
    report: ValidationReport,
    *,
    package_path: str | Path | None = None,
) -> str:
    """Return a compact human-readable validation report."""

    lines = ["", "=== Stage 2 Calibration Package Validation ===", ""]
    if package_path is not None:
        lines.append(f"Package: {package_path}")
    lines.append(f"Status: {report.status.upper()}")
    failing = [finding for finding in report.findings if finding.status == "fail"]
    warnings = [finding for finding in report.findings if finding.status == "warn"]
    lines.append(f"Findings: {len(report.findings)}")
    if failing:
        lines.append("")
        lines.append("Failures:")
        for finding in failing:
            lines.append(f"  {finding.check_id}: {finding.message}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for finding in warnings:
            lines.append(f"  {finding.check_id}: {finding.message}")
    return "\n".join(lines)


def _validation_suite(cache: dict[str, Any]) -> ValidationSuite:
    return ValidationSuite(
        suite_id="stage2_calibration_package_validation",
        stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
        substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
        checks=(
            ValidationCheck(
                check_id=_CHECK_PACKAGE_LOADABLE,
                stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                description="Calibration package pickle loads through typed payload reader.",
                required_artifacts=("calibration_package",),
                run=lambda context: _check_package_loadable(context, cache),
            ),
            ValidationCheck(
                check_id=_CHECK_CONTRACT_MATCHES,
                stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                description="Persisted Stage 2 contract matches package and input artifacts.",
                required_artifacts=(
                    "calibration_package",
                    "calibration_package_contract",
                    "source_dataset",
                    "target_database",
                ),
                run=lambda context: _check_contract_matches(context, cache),
            ),
            ValidationCheck(
                check_id=_CHECK_TARGET_CONFIG,
                stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                description="Target config identity is present and checksum-backed.",
                required_artifacts=("calibration_package",),
                run=lambda context: _check_target_config(context, cache),
            ),
            ValidationCheck(
                check_id=_CHECK_MATRIX,
                stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                description="Sparse matrix dimensions match target rows, target names, and summary artifact.",
                required_artifacts=("calibration_package", "matrix_summary"),
                run=lambda context: _check_matrix(context, cache),
            ),
            ValidationCheck(
                check_id=_CHECK_TARGET_FRAME,
                stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                description="Target frame contains required columns and row ordering.",
                required_artifacts=("calibration_package",),
                run=lambda context: _check_target_frame(context, cache),
            ),
            ValidationCheck(
                check_id=_CHECK_TARGET_METADATA,
                stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                description="Target metadata artifacts match package target rows and facets.",
                required_artifacts=(
                    "calibration_package",
                    "calibration_targets",
                    "calibration_target_facets",
                ),
                run=lambda context: _check_target_metadata(context, cache),
            ),
            ValidationCheck(
                check_id=_CHECK_GEOGRAPHY,
                stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                description="Geography assignment arrays and summary artifact are consistent.",
                required_artifacts=(
                    "calibration_package",
                    "geography_assignment_summary",
                ),
                run=lambda context: _check_geography(context, cache),
            ),
            ValidationCheck(
                check_id=_CHECK_INITIAL_WEIGHTS,
                stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                description="Initial weights are present, finite, non-negative, and column-aligned.",
                required_artifacts=("calibration_package",),
                run=lambda context: _check_initial_weights(context, cache),
            ),
            ValidationCheck(
                check_id=_CHECK_CHUNK_MANIFEST,
                stage_id=STAGE_2_BUILD_CALIBRATION_PACKAGE,
                substage_id=CALIBRATION_PACKAGE_SUBSTAGE_ID,
                description="Declared chunk manifest exists, parses, and matches its checksum.",
                required_artifacts=("calibration_package",),
                run=lambda context: _check_chunk_manifest(context, cache),
            ),
        ),
    )


def _check_package_loadable(
    context: ValidationContext,
    cache: dict[str, Any],
) -> ValidationFinding:
    payload = _payload(context, cache)
    summary = payload.summary()
    return _finding(
        _CHECK_PACKAGE_LOADABLE,
        status="pass",
        message="Calibration package payload is loadable.",
        metric="package_target_count",
        value=summary.n_targets,
    )


def _check_contract_matches(
    context: ValidationContext,
    cache: dict[str, Any],
) -> ValidationFinding:
    contract = validate_persisted_calibration_package_contract(
        package_path=_path(context, "calibration_package"),
        contract_path=_path(context, "calibration_package_contract"),
        dataset_path=_path(context, "source_dataset"),
        db_path=_path(context, "target_database"),
    )
    cache["contract"] = contract
    return _finding(
        _CHECK_CONTRACT_MATCHES,
        status="pass",
        message="Stage 2 contract matches the package and input artifacts.",
        value=contract.fingerprint.value,
    )


def _check_target_config(
    context: ValidationContext,
    cache: dict[str, Any],
) -> ValidationFinding:
    payload = _payload(context, cache)
    metadata = payload.metadata
    mode = metadata.get("target_config_mode")
    config_path = metadata.get("target_config_path")
    config_sha = metadata.get("target_config_sha256")
    if mode not in _TARGET_CONFIG_IDENTITY_MODES:
        return _finding(
            _CHECK_TARGET_CONFIG,
            status="fail",
            message=f"Unknown target config mode: {mode!r}",
            metric="target_config_mode",
            value=mode,
        )
    if mode == "all_active_targets":
        if config_path is not None or config_sha is not None:
            return _finding(
                _CHECK_TARGET_CONFIG,
                status="fail",
                message="all_active_targets target config must not include path or checksum.",
                metric="target_config_identity",
                value={"path": config_path, "sha256": config_sha},
            )
        return _finding(
            _CHECK_TARGET_CONFIG,
            status="pass",
            message="All-active-targets package does not require target config identity.",
            metric="target_config_mode",
            value=mode,
        )
    if not config_path or not config_sha:
        return _finding(
            _CHECK_TARGET_CONFIG,
            status="fail",
            message=f"{mode} target config requires path and checksum.",
            metric="target_config_identity",
            value={"path": config_path, "sha256": config_sha},
        )
    resolved_path = _resolve_existing_path(str(config_path))
    if resolved_path is None:
        return _finding(
            _CHECK_TARGET_CONFIG,
            status="fail",
            message=f"Target config path does not exist: {config_path}",
            metric="target_config_path",
            value=str(config_path),
        )
    actual_sha = compute_file_checksum(resolved_path)
    allowed = {actual_sha, f"sha256:{actual_sha}"}
    if str(config_sha) not in allowed:
        return _finding(
            _CHECK_TARGET_CONFIG,
            status="fail",
            message="Target config checksum does not match package metadata.",
            metric="target_config_sha256",
            value=str(config_sha),
            threshold=actual_sha,
            metadata={"path": str(resolved_path)},
        )
    return _finding(
        _CHECK_TARGET_CONFIG,
        status="pass",
        message="Target config identity is checksum-backed.",
        metric="target_config_mode",
        value=mode,
        metadata={"path": str(resolved_path)},
    )


def _check_matrix(
    context: ValidationContext,
    cache: dict[str, Any],
) -> ValidationFinding:
    payload = _payload(context, cache)
    summary = payload.summary()
    if summary.matrix_shape[0] != summary.n_targets:
        return _finding(
            _CHECK_MATRIX,
            status="fail",
            message="Matrix row count does not match target frame length.",
            metric="matrix_shape",
            value=summary.matrix_shape,
            threshold=summary.n_targets,
        )
    if summary.target_name_count != summary.n_targets:
        return _finding(
            _CHECK_MATRIX,
            status="fail",
            message="Target name count does not match target frame length.",
            metric="target_name_count",
            value=summary.target_name_count,
            threshold=summary.n_targets,
        )
    matrix_summary = _matrix_summary(context, cache)
    if matrix_summary is not None:
        expected = {
            "matrix_shape": tuple(summary.matrix_shape),
            "matrix_nnz": summary.matrix_nnz,
            "matrix_density": summary.matrix_density,
            "n_targets": summary.n_targets,
            "n_columns": summary.n_columns,
            "target_name_count": summary.target_name_count,
            "base_n_records": summary.base_n_records,
            "n_clones": summary.n_clones,
            "matrix_builder": summary.matrix_builder,
            "chunk_size": summary.chunk_size,
            "chunk_dir": summary.chunk_dir,
        }
        for key, expected_value in expected.items():
            actual_value = getattr(matrix_summary, key)
            if actual_value != expected_value:
                return _finding(
                    _CHECK_MATRIX,
                    status="fail",
                    message=f"Matrix summary artifact does not match package for {key}.",
                    metric=key,
                    value=actual_value,
                    threshold=expected_value,
                )
    return _finding(
        _CHECK_MATRIX,
        status="pass",
        message="Matrix dimensions and summary are consistent.",
        metric="matrix_shape",
        value=summary.matrix_shape,
    )


def _check_target_frame(
    context: ValidationContext,
    cache: dict[str, Any],
) -> ValidationFinding:
    payload = _payload(context, cache)
    targets_df = payload.targets_df
    missing_columns = sorted(_TARGET_FRAME_REQUIRED_COLUMNS - set(targets_df.columns))
    if missing_columns:
        return _finding(
            _CHECK_TARGET_FRAME,
            status="fail",
            message="Target frame is missing required columns.",
            metric="missing_columns",
            value=missing_columns,
        )
    if len(targets_df) != len(payload.target_names):
        return _finding(
            _CHECK_TARGET_FRAME,
            status="fail",
            message="Target frame row count does not match target_names count.",
            metric="target_row_count",
            value=len(targets_df),
            threshold=len(payload.target_names),
        )
    if bool(targets_df["value"].isna().any()):
        return _finding(
            _CHECK_TARGET_FRAME,
            status="fail",
            message="Target frame contains null target values.",
            metric="null_target_values",
            value=True,
        )
    return _finding(
        _CHECK_TARGET_FRAME,
        status="pass",
        message="Target frame columns and row ordering are valid.",
        metric="target_row_count",
        value=len(targets_df),
    )


def _check_target_metadata(
    context: ValidationContext,
    cache: dict[str, Any],
) -> ValidationFinding:
    payload = _payload(context, cache)
    rows = _target_metadata_rows(context)
    if len(rows) != len(payload.target_names):
        return _finding(
            _CHECK_TARGET_METADATA,
            status="fail",
            message="Target metadata row count does not match package targets.",
            metric="target_metadata_row_count",
            value=len(rows),
            threshold=len(payload.target_names),
        )
    targets_df = payload.targets_df.reset_index(drop=True)
    for index, row in enumerate(rows):
        if row.get("target_index") != index:
            return _target_metadata_mismatch(
                "target_index",
                row.get("target_index"),
                index,
            )
        expected_name = str(payload.target_names[index])
        if row.get("target_name") != expected_name:
            return _target_metadata_mismatch(
                "target_name",
                row.get("target_name"),
                expected_name,
            )
        expected_value = float(targets_df.loc[index, "value"])
        if not np.isclose(float(row.get("target_value")), expected_value):
            return _target_metadata_mismatch(
                "target_value",
                row.get("target_value"),
                expected_value,
            )
        comparisons = {
            "variable": str(targets_df.loc[index, "variable"]),
            "geography_level": _optional_string_value(
                targets_df.loc[index, "geo_level"]
            ),
            "geography_id": _optional_string_value(
                targets_df.loc[index, "geographic_id"]
            ),
            "domain_variable": _optional_string_value(
                targets_df.loc[index, "domain_variable"]
            ),
        }
        for key, expected in comparisons.items():
            if row.get(key) != expected:
                return _target_metadata_mismatch(key, row.get(key), expected)

    facets = _json_artifact(context, "calibration_target_facets")
    if not isinstance(facets, Mapping):
        return _finding(
            _CHECK_TARGET_METADATA,
            status="fail",
            message="Target facets artifact must contain a JSON object.",
            metric="target_facets_type",
            value=type(facets).__name__,
        )
    expected_facets = target_facets_from_rows(rows)
    if dict(facets) != expected_facets:
        return _finding(
            _CHECK_TARGET_METADATA,
            status="fail",
            message="Target facets artifact does not match target metadata rows.",
            metric="target_facets",
            value=dict(facets),
            threshold=expected_facets,
        )
    return _finding(
        _CHECK_TARGET_METADATA,
        status="pass",
        message="Target metadata rows and facets match package target order.",
        metric="target_metadata_row_count",
        value=len(rows),
    )


def _check_geography(
    context: ValidationContext,
    cache: dict[str, Any],
) -> ValidationFinding:
    payload = _payload(context, cache)
    summary = payload.geography_summary()
    if summary.source_kind != "calibration_package" or summary.status != "completed":
        return _finding(
            _CHECK_GEOGRAPHY,
            status="fail",
            message="Calibration package does not include completed geography assignment arrays.",
            metric="geography_status",
            value={
                "source_kind": summary.source_kind,
                "status": summary.status,
            },
        )
    persisted = _json_artifact(context, "geography_assignment_summary")
    if persisted is not None and persisted != summary.to_dict():
        return _finding(
            _CHECK_GEOGRAPHY,
            status="fail",
            message="Geography assignment summary artifact does not match package.",
            metric="canonical_geography_sha256",
            value=persisted.get("canonical_geography_sha256"),
            threshold=summary.canonical_geography_sha256,
        )
    return _finding(
        _CHECK_GEOGRAPHY,
        status="pass",
        message="Geography assignment arrays and summary are consistent.",
        metric="canonical_geography_sha256",
        value=summary.canonical_geography_sha256,
    )


def _check_initial_weights(
    context: ValidationContext,
    cache: dict[str, Any],
) -> ValidationFinding:
    payload = _payload(context, cache)
    if payload.initial_weights is None:
        return _finding(
            _CHECK_INITIAL_WEIGHTS,
            status="fail",
            message="Calibration package is missing initial_weights.",
            metric="has_initial_weights",
            value=False,
        )
    weights = np.asarray(payload.initial_weights)
    n_columns = int(payload.X_sparse.shape[1])
    if len(weights) != n_columns:
        return _finding(
            _CHECK_INITIAL_WEIGHTS,
            status="fail",
            message="Initial weights length does not match matrix columns.",
            metric="initial_weights_length",
            value=len(weights),
            threshold=n_columns,
        )
    if not bool(np.isfinite(weights).all()):
        return _finding(
            _CHECK_INITIAL_WEIGHTS,
            status="fail",
            message="Initial weights must be finite.",
            metric="initial_weights_finite",
            value=False,
        )
    if bool((weights < 0).any()):
        return _finding(
            _CHECK_INITIAL_WEIGHTS,
            status="fail",
            message="Initial weights must be non-negative.",
            metric="initial_weights_non_negative",
            value=False,
        )
    return _finding(
        _CHECK_INITIAL_WEIGHTS,
        status="pass",
        message="Initial weights are finite, non-negative, and column-aligned.",
        metric="initial_weights_length",
        value=len(weights),
    )


def _check_chunk_manifest(
    context: ValidationContext,
    cache: dict[str, Any],
) -> ValidationFinding:
    matrix_summary = _matrix_summary(context, cache)
    if matrix_summary is None:
        return _finding(
            _CHECK_CHUNK_MANIFEST,
            status="pass",
            message="No matrix summary artifact declared a chunk manifest.",
            metric="chunk_manifest_declared",
            value=False,
        )
    if matrix_summary.matrix_builder != "chunked":
        return _finding(
            _CHECK_CHUNK_MANIFEST,
            status="pass",
            message="Non-chunked matrix build does not require a chunk manifest.",
            metric="matrix_builder",
            value=matrix_summary.matrix_builder,
        )
    manifest_path = matrix_summary.chunk_manifest_path
    manifest_sha = matrix_summary.chunk_manifest_sha256
    if manifest_path is None and manifest_sha is None:
        return _finding(
            _CHECK_CHUNK_MANIFEST,
            status="pass",
            message="Chunked matrix summary does not declare a persisted chunk manifest.",
            metric="chunk_manifest_declared",
            value=False,
        )
    if not manifest_path or not manifest_sha:
        return _finding(
            _CHECK_CHUNK_MANIFEST,
            status="fail",
            message="Chunk manifest path and checksum must be declared together.",
            metric="chunk_manifest_identity",
            value={"path": manifest_path, "sha256": manifest_sha},
        )
    path = Path(manifest_path)
    if not path.exists():
        return _finding(
            _CHECK_CHUNK_MANIFEST,
            status="fail",
            message=f"Chunk manifest does not exist: {path}",
            metric="chunk_manifest_path",
            value=str(path),
        )
    actual_sha = compute_file_checksum(path)
    if manifest_sha not in {actual_sha, f"sha256:{actual_sha}"}:
        return _finding(
            _CHECK_CHUNK_MANIFEST,
            status="fail",
            message="Chunk manifest checksum does not match matrix summary.",
            metric="chunk_manifest_sha256",
            value=manifest_sha,
            threshold=actual_sha,
        )
    ChunkCacheManifest.read(path)
    return _finding(
        _CHECK_CHUNK_MANIFEST,
        status="pass",
        message="Chunk manifest exists, parses, and matches its checksum.",
        metric="chunk_manifest_sha256",
        value=manifest_sha,
    )


def _payload(
    context: ValidationContext,
    cache: dict[str, Any],
) -> CalibrationPackagePayload:
    if "payload" not in cache:
        cache["payload"] = CalibrationPackageReader(
            package_path=_path(context, "calibration_package")
        ).read()
    return cache["payload"]


def _matrix_summary(
    context: ValidationContext,
    cache: dict[str, Any],
) -> MatrixBuildSummary | None:
    if "matrix_summary" in cache:
        return cache["matrix_summary"]
    path = _optional_path(context, "matrix_summary")
    if path is None:
        cache["matrix_summary"] = None
        return None
    cache["matrix_summary"] = MatrixBuildSummary.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    return cache["matrix_summary"]


def _json_artifact(
    context: ValidationContext,
    logical_name: str,
) -> Mapping[str, Any] | None:
    path = _optional_path(context, logical_name)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _target_metadata_rows(context: ValidationContext) -> list[dict[str, Any]]:
    path = _path(context, "calibration_targets")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("Target metadata JSONL rows must be objects")
                rows.append(dict(row))
    return rows


def _target_metadata_mismatch(
    key: str,
    actual: Any,
    expected: Any,
) -> ValidationFinding:
    return _finding(
        _CHECK_TARGET_METADATA,
        status="fail",
        message=f"Target metadata artifact does not match package for {key}.",
        metric=key,
        value=actual,
        threshold=expected,
    )


def _finding(
    check_id: str,
    *,
    status: str,
    message: str,
    metric: str | None = None,
    value: Any | None = None,
    threshold: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        check_id=check_id,
        status=status,
        message=message,
        metric=metric,
        value=value,
        threshold=threshold,
        metadata=dict(metadata or {}),
    )


def _artifact_refs(paths: Mapping[str, Path]) -> dict[str, ArtifactRef]:
    refs: dict[str, ArtifactRef] = {}
    for logical_name, path in paths.items():
        if path.exists() and path.is_file():
            refs[logical_name] = ArtifactRef(
                logical_name=logical_name,
                uri=path.resolve().as_uri(),
                sha256=f"sha256:{sha256_file(path)}",
                size_bytes=path.stat().st_size,
                media_type=_media_type_for_path(path),
                metadata={
                    "stage_id": STAGE_2_BUILD_CALIBRATION_PACKAGE,
                    "substage_id": CALIBRATION_PACKAGE_SUBSTAGE_ID,
                },
            )
    return refs


def _path(context: ValidationContext, logical_name: str) -> Path:
    return _artifact_uri_to_path(context.resolver.require(logical_name).uri)


def _optional_path(context: ValidationContext, logical_name: str) -> Path | None:
    artifact = context.resolver.optional(logical_name)
    if artifact is None:
        return None
    return _artifact_uri_to_path(artifact.uri)


def _artifact_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri)
    raise ValueError(f"Unsupported artifact URI scheme: {uri}")


def _resolve_existing_path(path: str) -> Path | None:
    candidate = Path(path)
    candidates = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate]
    repo_candidate = Path(__file__).resolve().parents[2] / candidate
    if repo_candidate not in candidates:
        candidates.append(repo_candidate)
    for item in candidates:
        if item.exists() and item.is_file():
            return item
    return None


def _optional_string_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(np, "isnan"):
        try:
            if bool(np.isnan(value)):
                return None
        except TypeError:
            pass
    return str(value)


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".h5":
        return "application/x-hdf5"
    if suffix == ".db":
        return "application/vnd.sqlite3"
    if suffix == ".json":
        return "application/json"
    if suffix == ".jsonl":
        return "application/x-ndjson"
    if suffix == ".pkl":
        return "application/python-pickle"
    return "application/octet-stream"
