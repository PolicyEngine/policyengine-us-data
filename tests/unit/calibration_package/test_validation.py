import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.fixtures.calibration_package_stage_contract import (
    CALIBRATION_COMPLETED_AT,
    CALIBRATION_RUN_ID,
    CALIBRATION_STARTED_AT,
    calibration_package_parameters,
    calibration_package_payload,
    calibration_package_payload_without_geography,
    contract_input_paths,
    write_calibration_package_payload,
    write_non_mapping_calibration_package_payload,
)

from policyengine_us_data.calibration_package.matrix import (
    ChunkCacheManifest,
    MatrixBuildResult,
    MatrixBuildSpec,
)
from policyengine_us_data.calibration_package.specs import (
    CALIBRATION_REPORTS_DIRNAME,
    CALIBRATION_TARGET_FACETS_FILENAME,
    CALIBRATION_TARGETS_FILENAME,
    GEOGRAPHY_ASSIGNMENT_SUMMARY_FILENAME,
    MATRIX_SUMMARY_FILENAME,
    STAGE2_VALIDATION_FINDINGS_FILENAME,
    STAGE2_VALIDATION_REPORT_FILENAME,
    STAGE2_VALIDATION_SUMMARY_FILENAME,
)
from policyengine_us_data.calibration_package.validation import (
    CalibrationPackageValidationError,
    CalibrationPackageValidator,
)
from policyengine_us_data.calibration_package.targets import target_facets_from_rows
from policyengine_us_data.stage_contracts import ValidationReport
from policyengine_us_data.stage_contracts.calibration_package import (
    summarize_geography_assignment,
    write_calibration_package_contract,
)
from policyengine_us_data.stage_contracts.calibration_package_schema import (
    MatrixBuildSummary,
)
from policyengine_us_data.stage_contracts.io import read_contract
from policyengine_us_data.utils.manifest import compute_file_checksum


def _matrix_build_summary_for_package(package: dict) -> MatrixBuildSummary:
    metadata = package["metadata"]
    return MatrixBuildResult.from_builder_output(
        spec=MatrixBuildSpec(
            matrix_builder=metadata["matrix_builder"],
            base_n_records=metadata["base_n_records"],
            n_clones=metadata["n_clones"],
            chunk_size=metadata["chunk_size"],
            chunk_dir=metadata["chunk_dir"],
        ),
        targets_df=package["targets_df"],
        X_sparse=package["X_sparse"],
        target_names=package["target_names"],
    ).summary()


def _write_artifacts(
    tmp_path: Path,
    *,
    package: dict | None = None,
    matrix_summary_updates: dict | None = None,
) -> SimpleNamespace:
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = package or calibration_package_payload()
    target_config = tmp_path / "target_config.yaml"
    target_config.write_text("include: []\n", encoding="utf-8")
    package["metadata"]["target_config_path"] = str(target_config)
    package["metadata"]["target_config_sha256"] = compute_file_checksum(target_config)
    package["metadata"]["target_config_mode"] = "explicit"
    write_calibration_package_payload(package_path, package)

    target_rows = _target_rows_for_package(package)
    targets_path = tmp_path / CALIBRATION_TARGETS_FILENAME
    targets_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in target_rows),
        encoding="utf-8",
    )
    target_facets_path = tmp_path / CALIBRATION_TARGET_FACETS_FILENAME
    target_facets_path.write_text(
        json.dumps(target_facets_from_rows(target_rows), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    geography_summary = summarize_geography_assignment(package)
    geography_summary_path = tmp_path / GEOGRAPHY_ASSIGNMENT_SUMMARY_FILENAME
    geography_summary_path.write_text(
        json.dumps(geography_summary.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    matrix_summary = _matrix_build_summary_for_package(package).to_dict()
    if matrix_summary_updates:
        matrix_summary.update(matrix_summary_updates)
    matrix_summary_schema = MatrixBuildSummary.from_dict(matrix_summary)
    matrix_summary_path = tmp_path / MATRIX_SUMMARY_FILENAME
    matrix_summary_path.write_text(
        json.dumps(matrix_summary_schema.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parameters = calibration_package_parameters()
    parameters["target_config"] = str(target_config)
    parameters["target_config_sha256"] = compute_file_checksum(target_config)
    write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=parameters,
        run_id=CALIBRATION_RUN_ID,
        started_at=CALIBRATION_STARTED_AT,
        completed_at=CALIBRATION_COMPLETED_AT,
        target_metadata_path=targets_path,
        target_facets_path=target_facets_path,
        target_selection_summary={"target_count": len(package["target_names"])},
        geography_summary_path=geography_summary_path,
        geography_assignment_summary=geography_summary,
        matrix_summary_path=matrix_summary_path,
        matrix_build_summary=matrix_summary_schema,
    )
    return SimpleNamespace(
        package_path=package_path,
        contract_path=tmp_path / "calibration_package_contract.json",
        dataset_path=dataset_path,
        db_path=db_path,
        targets_path=targets_path,
        target_facets_path=target_facets_path,
        geography_summary_path=geography_summary_path,
        matrix_summary_path=matrix_summary_path,
        reports_dir=tmp_path / CALIBRATION_REPORTS_DIRNAME,
    )


def _target_rows_for_package(package: dict) -> list[dict]:
    rows = []
    targets_df = package["targets_df"].reset_index(drop=True)
    for target_index, row in targets_df.iterrows():
        rows.append(
            {
                "constraint_key": "none",
                "domain_variable": row.get("domain_variable"),
                "geography_id": row.get("geographic_id"),
                "geography_level": row.get("geo_level"),
                "included_in_package": True,
                "period": None,
                "source_table": "targets",
                "target_components": [row["variable"]],
                "target_config_mode": package["metadata"]["target_config_mode"],
                "target_config_path": package["metadata"]["target_config_path"],
                "target_config_sha256": package["metadata"]["target_config_sha256"],
                "target_constraints": [],
                "target_expression": None,
                "target_id": target_index,
                "target_index": target_index,
                "target_name": str(package["target_names"][target_index]),
                "target_value": float(row["value"]),
                "variable": row["variable"],
            }
        )
    return rows


def _validate(paths: SimpleNamespace) -> ValidationReport:
    return CalibrationPackageValidator().validate_and_write(
        package_path=paths.package_path,
        contract_path=paths.contract_path,
        dataset_path=paths.dataset_path,
        db_path=paths.db_path,
        reports_dir=paths.reports_dir,
        targets_path=paths.targets_path,
        target_facets_path=paths.target_facets_path,
        geography_summary_path=paths.geography_summary_path,
        matrix_summary_path=paths.matrix_summary_path,
        run_id=CALIBRATION_RUN_ID,
    )


def _failing_ids(report: ValidationReport) -> set[str]:
    return {finding.check_id for finding in report.findings if finding.status == "fail"}


def test_validator_writes_canonical_report_and_attaches_contract(tmp_path):
    paths = _write_artifacts(tmp_path)

    report = _validate(paths)

    assert report.status == "pass"
    assert (paths.reports_dir / STAGE2_VALIDATION_REPORT_FILENAME).exists()
    assert (paths.reports_dir / STAGE2_VALIDATION_FINDINGS_FILENAME).exists()
    assert (paths.reports_dir / STAGE2_VALIDATION_SUMMARY_FILENAME).exists()
    restored = ValidationReport.from_dict(
        json.loads(
            (paths.reports_dir / STAGE2_VALIDATION_REPORT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
    )
    assert restored.status == "pass"
    contract = read_contract(paths.contract_path)
    assert contract.validation == report
    assert contract.substages[0].validation == report
    assert contract.metadata["validation_artifacts"]["report"].endswith(
        STAGE2_VALIDATION_REPORT_FILENAME
    )


def test_validator_reports_target_config_checksum_failure(tmp_path):
    paths = _write_artifacts(tmp_path)
    package = calibration_package_payload()
    package["metadata"]["target_config_path"] = str(tmp_path / "target_config.yaml")
    package["metadata"]["target_config_sha256"] = "sha256:" + "0" * 64
    write_calibration_package_payload(paths.package_path, package)

    report = _validate(paths)

    assert report.status == "fail"
    assert "stage2.target_config.identity" in _failing_ids(report)


def test_validator_reports_unloadable_package_payload(tmp_path):
    paths = _write_artifacts(tmp_path)
    write_non_mapping_calibration_package_payload(paths.package_path)

    report = _validate(paths)

    assert "stage2.package.loadable" in _failing_ids(report)


def test_validator_reports_matrix_summary_mismatch(tmp_path):
    paths = _write_artifacts(tmp_path)
    summary = json.loads(paths.matrix_summary_path.read_text(encoding="utf-8"))
    summary["matrix_nnz"] = 1
    paths.matrix_summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = _validate(paths)

    assert "stage2.matrix.consistency" in _failing_ids(report)


def test_validator_reports_missing_matrix_summary_artifact(tmp_path):
    paths = _write_artifacts(tmp_path)
    paths.matrix_summary_path.unlink()

    report = _validate(paths)

    assert "stage2.matrix.consistency" in _failing_ids(report)


def test_validator_reports_target_frame_missing_required_column(tmp_path):
    package = calibration_package_payload()
    package["targets_df"] = package["targets_df"].drop(columns=["geographic_id"])
    paths = _write_artifacts(tmp_path, package=package)

    report = _validate(paths)

    assert "stage2.target_frame.consistency" in _failing_ids(report)


def test_validator_reports_missing_target_metadata_artifact(tmp_path):
    paths = _write_artifacts(tmp_path)
    paths.targets_path.unlink()

    report = _validate(paths)

    assert "stage2.target_metadata.consistency" in _failing_ids(report)


def test_validator_reports_target_facets_mismatch(tmp_path):
    paths = _write_artifacts(tmp_path)
    facets = json.loads(paths.target_facets_path.read_text(encoding="utf-8"))
    facets["target_count"] = 999
    paths.target_facets_path.write_text(
        json.dumps(facets, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = _validate(paths)

    assert "stage2.target_metadata.consistency" in _failing_ids(report)


def test_validator_reports_missing_geography_summary_artifact(tmp_path):
    paths = _write_artifacts(tmp_path)
    paths.geography_summary_path.unlink()

    report = _validate(paths)

    assert "stage2.geography.consistency" in _failing_ids(report)


def test_validator_reports_missing_geography_arrays(tmp_path):
    paths = _write_artifacts(
        tmp_path,
        package=calibration_package_payload_without_geography(),
    )

    report = _validate(paths)

    assert "stage2.geography.consistency" in _failing_ids(report)


def test_validator_reports_initial_weights_length_mismatch(tmp_path):
    package = calibration_package_payload()
    package["initial_weights"] = np.array([1.0, 1.0])
    paths = _write_artifacts(tmp_path, package=package)

    report = _validate(paths)

    assert "stage2.initial_weights.consistency" in _failing_ids(report)


def test_validator_reports_chunk_manifest_checksum_mismatch(tmp_path):
    manifest_path = tmp_path / "matrix_build" / "chunk_manifest.json"
    ChunkCacheManifest.from_signature({"run_id": CALIBRATION_RUN_ID}).write(
        manifest_path
    )
    paths = _write_artifacts(
        tmp_path,
        matrix_summary_updates={
            "chunk_manifest_path": str(manifest_path),
            "chunk_manifest_sha256": "sha256:" + "0" * 64,
        },
    )

    report = _validate(paths)

    assert "stage2.chunk_manifest.consistency" in _failing_ids(report)


def test_validation_failure_error_includes_finding_ids(tmp_path):
    package = calibration_package_payload()
    package["initial_weights"] = np.array([1.0, 1.0])
    paths = _write_artifacts(tmp_path, package=package)
    validator = CalibrationPackageValidator()
    report = _validate(paths)

    with pytest.raises(
        CalibrationPackageValidationError,
        match="stage2.initial_weights.consistency",
    ):
        validator.raise_for_failure(report)
