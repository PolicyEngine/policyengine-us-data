from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import pytest

from policyengine_us_data.stage_contracts.core import (
    CONTRACT_FINGERPRINT_ALGORITHM,
    CONTRACT_SCHEMA_VERSION,
    ArtifactRef,
    DiagnosticRef,
    ExecutionRecord,
    Fingerprint,
    ReuseSummary,
    StageContract,
    SubstageRecord,
    ValidationFinding,
    ValidationReport,
)
from policyengine_us_data.stage_contracts.io import (
    contract_from_json,
    contract_to_json,
    read_contract,
    write_contract,
)
from policyengine_us_data.stage_contracts.fingerprints import (
    canonicalize_for_fingerprint,
    fingerprint_material,
)


def _fingerprint() -> Fingerprint:
    return Fingerprint(
        value="sha256:contract",
        material={
            "stage_id": "2_build_calibration_package",
            "inputs": {"policy_data_db": "sha256:db"},
        },
    )


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        logical_name="policy_data_db",
        uri="hf://policyengine/policy_data.db",
        sha256="sha256:db",
        size_bytes=128,
        media_type="application/vnd.sqlite3",
        metadata={"source": "stage_1"},
    )


def _execution() -> ExecutionRecord:
    return ExecutionRecord(
        status="completed",
        attempt=1,
        started_at="2026-05-05T10:00:00Z",
        completed_at="2026-05-05T10:00:03Z",
        duration_s=3.0,
        modal_call_id="call-123",
        reuse_decision="computed",
        reuse_summary=ReuseSummary(
            expected_outputs=2,
            valid_reused_outputs=0,
            recomputed_outputs=2,
            invalid_outputs=0,
        ),
        metadata={"worker": "modal"},
    )


def _diagnostic(*, artifact=None) -> DiagnosticRef:
    return DiagnosticRef(
        name="weight_fit_summary",
        kind="json",
        artifact=artifact,
        summary={"max_abs_error": 0.02, "rows": 12},
        severity="warning",
    )


def _validation_report() -> ValidationReport:
    return ValidationReport(
        status="warn",
        findings=(
            ValidationFinding(
                check_id="artifact_exists",
                status="pass",
                message="Output artifact exists.",
                metric="exists",
                value=True,
                threshold=True,
            ),
            ValidationFinding(
                check_id="target_error",
                status="warn",
                message="Target error is within warning range.",
                metric="max_abs_error",
                value=0.02,
                threshold=0.01,
                metadata={"target": "employment_income"},
            ),
            ValidationFinding(
                check_id="optional_check",
                status="fail",
                message="Optional check failed.",
                metadata={"blocking": False},
            ),
        ),
        diagnostics=(_diagnostic(artifact=_artifact()),),
        metadata={"validator": "stage_contracts"},
    )


def _stage_contract(*, parameters=None) -> StageContract:
    return StageContract(
        contract_type="calibration_package",
        stage_id="2_build_calibration_package",
        run_id="run-123",
        created_at="2026-05-05T10:00:03Z",
        code_sha="abc123",
        package_version="1.98.2",
        inputs=(_artifact(),),
        outputs=(
            ArtifactRef(
                logical_name="calibration_package",
                uri="file:///pipeline/calibration_package.pkl",
                sha256="sha256:calibration",
            ),
        ),
        parameters=parameters or {"n_clones": 430, "seed": 42},
        fingerprint=_fingerprint(),
        substages=(
            SubstageRecord(
                substage_id="2a_build_target_matrix",
                status="completed",
                reuse_mode="handoff",
                outputs=(
                    ArtifactRef(
                        logical_name="target_matrix",
                        uri="memory://target_matrix",
                    ),
                ),
            ),
        ),
        execution=_execution(),
        metadata={"target_count": 304},
    )


def _stage_artifact(
    logical_name: str,
    *,
    uri: str | None = None,
    sha256: str | None = None,
    media_type: str | None = None,
    metadata: dict | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        logical_name=logical_name,
        uri=uri or f"hf://policyengine-us-data/stage-artifacts/{logical_name}",
        sha256=sha256 or f"sha256:{logical_name}",
        media_type=media_type,
        metadata=metadata or {},
    )


def _sample_fingerprint(
    *,
    contract_type: str,
    stage_id: str,
    inputs: tuple[ArtifactRef, ...],
    parameters: dict | None = None,
) -> Fingerprint:
    return fingerprint_material(
        {
            "contract_type": contract_type,
            "stage_id": stage_id,
            "inputs": {
                artifact.logical_name: artifact.sha256
                for artifact in inputs
            },
            "parameters": parameters or {},
        }
    )


def make_dataset_build_contract() -> StageContract:
    contract_type = "dataset_build_output"
    stage_id = "1_build_datasets"
    parameters = {"period": 2024, "dataset_release": "fixture"}
    outputs = (
        _stage_artifact("source_imputed_stratified_extended_cps"),
        _stage_artifact(
            "policy_data_db",
            uri="hf://policyengine-us-data/policy_data.db",
            media_type="application/vnd.sqlite3",
        ),
        _stage_artifact("cps_2024"),
        _stage_artifact("enhanced_cps_2024"),
    )
    return StageContract(
        contract_type=contract_type,
        stage_id=stage_id,
        created_at="2026-05-05T01:00:00Z",
        package_version="1.98.2",
        outputs=outputs,
        parameters=parameters,
        fingerprint=_sample_fingerprint(
            contract_type=contract_type,
            stage_id=stage_id,
            inputs=(),
            parameters=parameters,
        ),
        substages=(
            SubstageRecord(
                substage_id="1a_raw_data_download",
                status="completed",
                reuse_mode="checkpointable",
                outputs=(_stage_artifact("raw_census_cps"),),
            ),
            SubstageRecord(
                substage_id="1b_base_dataset_construction",
                status="completed",
                reuse_mode="checkpointable",
                outputs=(_stage_artifact("base_cps_2024"),),
            ),
            SubstageRecord(
                substage_id="1f_source_imputation",
                status="completed",
                reuse_mode="handoff",
                outputs=(_stage_artifact("source_imputed_stratified_extended_cps"),),
            ),
            SubstageRecord(
                substage_id="1g_stage_base_datasets",
                status="completed",
                reuse_mode="handoff",
                outputs=outputs,
            ),
        ),
        execution=ExecutionRecord(status="completed", reuse_decision="computed"),
        metadata={
            "dataset_schema": {
                "households": "placeholder",
                "people": "placeholder",
            },
            "entity_schema": {
                "person": "placeholder",
                "tax_unit": "placeholder",
                "spm_unit": "placeholder",
            },
        },
    )


def make_calibration_package_contract() -> StageContract:
    contract_type = "calibration_package"
    stage_id = "2_build_calibration_package"
    inputs = (
        _stage_artifact("source_imputed_stratified_extended_cps"),
        _stage_artifact("policy_data_db"),
    )
    parameters = {"n_clones": 430, "target_config": "target_config.yaml"}
    return StageContract(
        contract_type=contract_type,
        stage_id=stage_id,
        created_at="2026-05-05T02:00:00Z",
        inputs=inputs,
        outputs=(_stage_artifact("calibration_package"),),
        parameters=parameters,
        fingerprint=_sample_fingerprint(
            contract_type=contract_type,
            stage_id=stage_id,
            inputs=inputs,
            parameters=parameters,
        ),
        substages=(
            SubstageRecord(
                substage_id="2a_build_target_matrix",
                status="completed",
                reuse_mode="handoff",
                inputs=inputs,
                outputs=(_stage_artifact("clone_target_matrix"),),
            ),
            SubstageRecord(
                substage_id="2b_package_calibration_inputs",
                status="completed",
                reuse_mode="handoff",
                inputs=(_stage_artifact("clone_target_matrix"),),
                outputs=(_stage_artifact("calibration_package"),),
            ),
        ),
        execution=ExecutionRecord(status="completed", reuse_decision="computed"),
        metadata={
            "matrix_shape": [184900, 304],
            "nnz": 1245000,
            "target_count": 304,
            "n_clones": 430,
            "base_n_records": 430,
            "matrix_ordering": "clone_major",
            "target_config_checksum": "sha256:target-config",
            "geography_checksum": "sha256:geography",
        },
    )


def make_fitted_weights_contract() -> StageContract:
    contract_type = "fitted_weights"
    stage_id = "3_fit_weights"
    inputs = (_stage_artifact("calibration_package"),)
    parameters = {"solver": "l0", "max_iterations": 1000}
    diagnostics = (
        _diagnostic(
            artifact=_stage_artifact("regional_weight_fit_diagnostics")
        ),
    )
    validation = ValidationReport(
        status="pass",
        findings=(
            ValidationFinding(
                check_id="weights_nonnegative",
                status="pass",
                message="All fitted weights are non-negative.",
            ),
        ),
        diagnostics=diagnostics,
    )
    return StageContract(
        contract_type=contract_type,
        stage_id=stage_id,
        created_at="2026-05-05T03:00:00Z",
        inputs=inputs,
        outputs=(
            _stage_artifact("regional_weights"),
            _stage_artifact("national_weights"),
        ),
        parameters=parameters,
        fingerprint=_sample_fingerprint(
            contract_type=contract_type,
            stage_id=stage_id,
            inputs=inputs,
            parameters=parameters,
        ),
        substages=(
            SubstageRecord(
                substage_id="3a_weight_fitting_regional",
                status="completed",
                reuse_mode="handoff",
                outputs=(_stage_artifact("regional_weights"),),
                validation=validation,
                diagnostics=diagnostics,
            ),
            SubstageRecord(
                substage_id="3b_weight_fitting_national",
                status="completed",
                reuse_mode="handoff",
                outputs=(_stage_artifact("national_weights"),),
            ),
        ),
        execution=ExecutionRecord(status="completed", reuse_decision="computed"),
        validation=validation,
        diagnostics=diagnostics,
        metadata={
            "solver_settings": {"solver": "l0", "max_iterations": 1000},
            "regional_summary": {"areas": 3143, "max_abs_error": 0.01},
            "national_summary": {"records": 430, "max_abs_error": 0.0},
        },
    )


def make_output_build_contract() -> StageContract:
    contract_type = "output_build"
    stage_id = "4_build_outputs"
    inputs = (
        _stage_artifact("regional_weights"),
        _stage_artifact("national_weights"),
        _stage_artifact("source_imputed_stratified_extended_cps"),
        _stage_artifact("geography"),
        _stage_artifact("policy_data_db"),
    )
    parameters = {"h5_format": "policyengine-us-v1", "period": 2024}
    diagnostics = (_diagnostic(artifact=_stage_artifact("diagnostics_manifest")),)
    validation = ValidationReport(
        status="warn",
        findings=(
            ValidationFinding(
                check_id="regional_inventory_complete",
                status="pass",
                message="Expected regional H5 inventory is complete.",
            ),
            ValidationFinding(
                check_id="diagnostics_uploaded",
                status="warn",
                message="Diagnostics are present but not yet promotion-blocking.",
            ),
        ),
        diagnostics=diagnostics,
    )
    return StageContract(
        contract_type=contract_type,
        stage_id=stage_id,
        created_at="2026-05-05T04:00:00Z",
        inputs=inputs,
        outputs=(
            _stage_artifact("regional_h5_inventory"),
            _stage_artifact("national_h5"),
            _stage_artifact("diagnostics_manifest"),
        ),
        parameters=parameters,
        fingerprint=_sample_fingerprint(
            contract_type=contract_type,
            stage_id=stage_id,
            inputs=inputs,
            parameters=parameters,
        ),
        substages=(
            SubstageRecord(
                substage_id="4a_local_area_h5_regional",
                status="completed",
                reuse_mode="handoff",
                outputs=(_stage_artifact("regional_h5_inventory"),),
            ),
            SubstageRecord(
                substage_id="4b_local_area_h5_national",
                status="completed",
                reuse_mode="handoff",
                outputs=(_stage_artifact("national_h5"),),
            ),
            SubstageRecord(
                substage_id="4d_upload_diagnostics",
                status="completed",
                reuse_mode="observed_only",
                outputs=(_stage_artifact("diagnostics_manifest"),),
                diagnostics=diagnostics,
            ),
        ),
        execution=ExecutionRecord(status="completed", reuse_decision="computed"),
        validation=validation,
        diagnostics=diagnostics,
        metadata={
            "h5_inventory": {
                "regional_count": 3143,
                "national_count": 1,
                "missing": 0,
            },
            "validation_summary": {
                "status": "warn",
                "blocking_failures": 0,
            },
        },
    )


def make_release_promotion_contract() -> StageContract:
    contract_type = "release_promotion"
    stage_id = "5_validate_and_promote_release"
    inputs = (_stage_artifact("output_build_release_candidate"),)
    parameters = {
        "hf_repo": "policyengine/policyengine-us-data",
        "gcs_bucket": "policyengine-us-data",
    }
    validation = ValidationReport(
        status="pass",
        findings=(
            ValidationFinding(
                check_id="release_candidate_complete",
                status="pass",
                message="Release candidate contains all required artifacts.",
            ),
        ),
    )
    return StageContract(
        contract_type=contract_type,
        stage_id=stage_id,
        created_at="2026-05-05T05:00:00Z",
        inputs=inputs,
        outputs=(
            _stage_artifact("release_manifest"),
            _stage_artifact("version_manifest"),
        ),
        parameters=parameters,
        fingerprint=_sample_fingerprint(
            contract_type=contract_type,
            stage_id=stage_id,
            inputs=inputs,
            parameters=parameters,
        ),
        substages=(
            SubstageRecord(
                substage_id="5a_validate_outputs",
                status="completed",
                reuse_mode="observed_only",
                validation=validation,
            ),
            SubstageRecord(
                substage_id="5b_promote_huggingface",
                status="completed",
                reuse_mode="handoff",
                outputs=(_stage_artifact("hf_release_refs"),),
            ),
            SubstageRecord(
                substage_id="5c_promote_gcs",
                status="completed",
                reuse_mode="handoff",
                outputs=(_stage_artifact("gcs_release_refs"),),
            ),
            SubstageRecord(
                substage_id="5d_write_version_manifest",
                status="completed",
                reuse_mode="handoff",
                outputs=(_stage_artifact("version_manifest"),),
            ),
        ),
        execution=ExecutionRecord(status="completed", reuse_decision="computed"),
        validation=validation,
        metadata={
            "promotion_summary": {
                "huggingface": "promoted",
                "gcs": "promoted",
                "version_manifest": "written",
            }
        },
    )


def _sample_stage_contracts() -> tuple[StageContract, ...]:
    return (
        make_dataset_build_contract(),
        make_calibration_package_contract(),
        make_fitted_weights_contract(),
        make_output_build_contract(),
        make_release_promotion_contract(),
    )


def test_artifact_ref_dict_round_trip():
    artifact = _artifact()

    restored = ArtifactRef.from_dict(artifact.to_dict())

    assert restored == artifact
    assert restored.schema_version == CONTRACT_SCHEMA_VERSION


def test_reuse_summary_dict_round_trip():
    summary = ReuseSummary(
        expected_outputs=10,
        valid_reused_outputs=7,
        recomputed_outputs=3,
        invalid_outputs=0,
        saved_duration_s=12.5,
    )

    restored = ReuseSummary.from_dict(summary.to_dict())

    assert restored == summary


def test_execution_record_dict_round_trip_with_nested_reuse_summary():
    execution = _execution()

    restored = ExecutionRecord.from_dict(execution.to_dict())

    assert restored == execution
    assert restored.reuse_summary.valid_reused_outputs == 0


def test_substage_record_dict_round_trip_with_nested_values():
    substage = SubstageRecord(
        substage_id="2a_build_target_matrix",
        status="completed",
        inputs=(_artifact(),),
        outputs=(
            ArtifactRef(
                logical_name="calibration_package",
                uri="file:///pipeline/calibration_package.pkl",
                sha256="sha256:calibration",
            ),
        ),
        parameters={"n_clones": 430},
        fingerprint=_fingerprint(),
        reuse_mode="handoff",
        metadata={"matrix_order": "clone_major"},
    )

    restored = SubstageRecord.from_dict(substage.to_dict())

    assert restored == substage
    assert isinstance(restored.inputs, tuple)
    assert isinstance(restored.outputs, tuple)
    assert restored.fingerprint is not None
    assert restored.fingerprint.algorithm == CONTRACT_FINGERPRINT_ALGORITHM


def test_stage_contract_dict_round_trip_with_substages():
    contract = _stage_contract()

    restored = StageContract.from_dict(contract.to_dict())

    assert restored == contract
    assert isinstance(restored.inputs, tuple)
    assert isinstance(restored.outputs, tuple)
    assert isinstance(restored.substages, tuple)
    assert restored.execution.status == "completed"


def test_sample_stage_contract_builders_match_canonical_stage_shapes():
    samples = _sample_stage_contracts()

    assert [(item.stage_id, item.contract_type) for item in samples] == [
        ("1_build_datasets", "dataset_build_output"),
        ("2_build_calibration_package", "calibration_package"),
        ("3_fit_weights", "fitted_weights"),
        ("4_build_outputs", "output_build"),
        ("5_validate_and_promote_release", "release_promotion"),
    ]
    assert [len(item.substages) for item in samples] == [4, 2, 2, 3, 4]
    for contract in samples:
        assert isinstance(contract, StageContract)
        assert all(
            isinstance(substage, SubstageRecord)
            for substage in contract.substages
        )


def test_sample_stage_contracts_include_planned_artifacts_and_metadata():
    dataset_build = make_dataset_build_contract()
    calibration_package = make_calibration_package_contract()
    fitted_weights = make_fitted_weights_contract()
    output_build = make_output_build_contract()
    release_promotion = make_release_promotion_contract()

    assert {artifact.logical_name for artifact in dataset_build.outputs} == {
        "source_imputed_stratified_extended_cps",
        "policy_data_db",
        "cps_2024",
        "enhanced_cps_2024",
    }
    assert calibration_package.metadata["matrix_shape"] == (184900, 304)
    assert calibration_package.metadata["target_config_checksum"]
    assert {artifact.logical_name for artifact in fitted_weights.outputs} == {
        "regional_weights",
        "national_weights",
    }
    assert output_build.metadata["h5_inventory"]["regional_count"] == 3143
    assert output_build.validation.status == "warn"
    assert release_promotion.metadata["promotion_summary"] == {
        "huggingface": "promoted",
        "gcs": "promoted",
        "version_manifest": "written",
    }


def test_sample_stage_contracts_dict_and_json_round_trip():
    for contract in _sample_stage_contracts():
        assert StageContract.from_dict(contract.to_dict()) == contract
        assert contract_from_json(contract_to_json(contract)) == contract


def test_sample_stage_contracts_write_only_to_explicit_paths(tmp_path):
    expected_paths = set()
    for contract in _sample_stage_contracts():
        target = tmp_path / f"{contract.stage_id}.json"
        expected_paths.add(target)

        write_contract(contract, target)

        assert read_contract(target) == contract

    assert set(tmp_path.iterdir()) == expected_paths
    assert not (tmp_path / "contracts").exists()


def test_sample_stage_contract_fingerprints_are_reproducible():
    for contract in _sample_stage_contracts():
        rebuilt = fingerprint_material(contract.fingerprint.material)

        assert rebuilt.value == contract.fingerprint.value
        assert contract.fingerprint.material["stage_id"] == contract.stage_id
        assert (
            contract.fingerprint.material["contract_type"]
            == contract.contract_type
        )


def test_diagnostic_ref_dict_round_trip_with_artifact_and_summary():
    diagnostic = _diagnostic(artifact=_artifact())

    restored = DiagnosticRef.from_dict(diagnostic.to_dict())

    assert restored == diagnostic
    assert restored.artifact == _artifact()
    assert restored.summary["rows"] == 12


def test_diagnostic_ref_dict_round_trip_with_embedded_summary_only():
    diagnostic = DiagnosticRef(
        name="calibration_summary",
        kind="table_summary",
        summary={
            "rows": 3,
            "columns": ("target", "actual", "error"),
        },
        severity="info",
    )

    restored = DiagnosticRef.from_dict(diagnostic.to_dict())

    assert restored == diagnostic
    assert restored.artifact is None
    assert restored.summary["columns"] == ("target", "actual", "error")


def test_validation_finding_dict_round_trip():
    finding = ValidationFinding(
        check_id="max_abs_error",
        status="fail",
        message="Maximum absolute error exceeded the promotion threshold.",
        metric="max_abs_error",
        value={"observed": 0.12},
        threshold={"maximum": 0.05},
        metadata={"blocking": True},
    )

    restored = ValidationFinding.from_dict(finding.to_dict())

    assert restored == finding
    assert restored.value["observed"] == 0.12
    assert restored.threshold["maximum"] == 0.05


def test_validation_report_dict_round_trip_with_mixed_findings():
    report = _validation_report()

    restored = ValidationReport.from_dict(report.to_dict())

    assert restored == report
    assert [finding.status for finding in restored.findings] == [
        "pass",
        "warn",
        "fail",
    ]
    assert restored.diagnostics[0].artifact == _artifact()


def test_substage_record_dict_round_trip_with_validation_and_diagnostics():
    substage = SubstageRecord(
        substage_id="3b_validate_weights",
        status="completed",
        validation=_validation_report(),
        diagnostics=(_diagnostic(),),
    )

    restored = SubstageRecord.from_dict(substage.to_dict())

    assert restored == substage
    assert restored.validation.status == "warn"
    assert restored.diagnostics[0].name == "weight_fit_summary"


def test_stage_contract_dict_round_trip_with_validation_and_diagnostics():
    contract = StageContract(
        contract_type="fitted_weights",
        stage_id="3_fit_weights",
        created_at="2026-05-05T10:00:03Z",
        fingerprint=_fingerprint(),
        execution=_execution(),
        validation=_validation_report(),
        diagnostics=(_diagnostic(artifact=_artifact()),),
    )

    restored = StageContract.from_dict(contract.to_dict())

    assert restored == contract
    assert restored.validation.status == "warn"
    assert restored.diagnostics[0].artifact == _artifact()


def test_invalid_execution_status_raises():
    with pytest.raises(ValueError, match="Invalid execution status"):
        ExecutionRecord(status="done")


def test_invalid_reuse_decision_raises():
    with pytest.raises(ValueError, match="Invalid reuse decision"):
        ExecutionRecord(reuse_decision="maybe")


def test_invalid_substage_status_raises():
    with pytest.raises(ValueError, match="Invalid substage status"):
        SubstageRecord(substage_id="2a_build_target_matrix", status="done")


def test_invalid_substage_reuse_mode_raises():
    with pytest.raises(ValueError, match="Invalid substage reuse mode"):
        SubstageRecord(
            substage_id="2a_build_target_matrix",
            status="completed",
            reuse_mode="cacheable",
        )


def test_invalid_schema_version_raises():
    with pytest.raises(ValueError, match="schema_version"):
        ArtifactRef(
            logical_name="policy_data_db",
            uri="file:///policy_data.db",
            schema_version="0",
        )


def test_invalid_validation_status_raises():
    with pytest.raises(ValueError, match="Invalid validation finding status"):
        ValidationFinding(
            check_id="check",
            status="not_run",
            message="Finding statuses cannot be not_run.",
        )

    with pytest.raises(ValueError, match="Invalid validation report status"):
        ValidationReport(status="unknown")


def test_invalid_diagnostic_severity_raises():
    with pytest.raises(ValueError, match="Invalid diagnostic severity"):
        DiagnosticRef(name="summary", kind="json", severity="debug")


def test_contracts_without_validation_or_diagnostics_remain_valid():
    contract = _stage_contract()
    payload = contract.to_dict()
    payload.pop("validation")
    payload.pop("diagnostics")
    payload["substages"][0].pop("validation")
    payload["substages"][0].pop("diagnostics")

    restored = StageContract.from_dict(payload)

    assert restored.validation is None
    assert restored.diagnostics == ()
    assert restored.substages[0].validation is None
    assert restored.substages[0].diagnostics == ()


def test_from_dict_rejects_none_required_string_fields():
    with pytest.raises(ValueError, match="logical_name"):
        ArtifactRef.from_dict({"logical_name": None, "uri": "file:///artifact"})

    payload = _stage_contract().to_dict()
    payload["stage_id"] = None
    with pytest.raises(ValueError, match="stage_id"):
        StageContract.from_dict(payload)


def test_contract_numeric_fields_reject_non_numeric_constructor_values():
    with pytest.raises(ValueError, match="size_bytes"):
        ArtifactRef(
            logical_name="policy_data_db",
            uri="file:///policy_data.db",
            size_bytes="128",
        )

    with pytest.raises(ValueError, match="expected_outputs"):
        ReuseSummary(expected_outputs=True)

    with pytest.raises(ValueError, match="saved_duration_s"):
        ReuseSummary(saved_duration_s="1.5")

    with pytest.raises(ValueError, match="attempt"):
        ExecutionRecord(attempt="1")

    with pytest.raises(ValueError, match="duration_s"):
        ExecutionRecord(duration_s="3.0")


def test_from_dict_rejects_invalid_numeric_fields():
    with pytest.raises(ValueError, match="size_bytes"):
        ArtifactRef.from_dict(
            {
                "logical_name": "policy_data_db",
                "uri": "file:///policy_data.db",
                "size_bytes": "128",
            }
        )

    with pytest.raises(ValueError, match="expected_outputs"):
        ReuseSummary.from_dict({"expected_outputs": "1"})

    with pytest.raises(ValueError, match="valid_reused_outputs"):
        ReuseSummary.from_dict({"valid_reused_outputs": True})

    with pytest.raises(ValueError, match="attempt"):
        ExecutionRecord.from_dict({"attempt": "1"})

    with pytest.raises(ValueError, match="duration_s"):
        ExecutionRecord.from_dict({"duration_s": "3.0"})


def test_contract_float_fields_reject_non_finite_values():
    with pytest.raises(ValueError, match="saved_duration_s"):
        ReuseSummary(saved_duration_s=float("inf"))

    with pytest.raises(ValueError, match="duration_s"):
        ExecutionRecord.from_dict({"duration_s": float("nan")})


def test_mapping_fields_are_defensively_frozen():
    parameters = {"seed": 42, "nested": {"n_clones": 430}}
    metadata = {"target_count": 304}
    contract = StageContract(
        contract_type="calibration_package",
        stage_id="2_build_calibration_package",
        created_at="2026-05-05T10:00:03Z",
        parameters=parameters,
        fingerprint=_fingerprint(),
        execution=_execution(),
        metadata=metadata,
    )

    parameters["seed"] = 99
    parameters["nested"]["n_clones"] = 1
    metadata["target_count"] = 1

    assert contract.parameters["seed"] == 42
    assert contract.parameters["nested"]["n_clones"] == 430
    assert contract.metadata["target_count"] == 304
    with pytest.raises(TypeError):
        contract.parameters["seed"] = 100
    with pytest.raises(TypeError):
        contract.parameters["nested"]["n_clones"] = 100


def test_contract_to_json_is_deterministic_across_equivalent_contracts():
    first = _stage_contract(parameters={"n_clones": 430, "seed": 42})
    second = _stage_contract(parameters={"seed": 42, "n_clones": 430})

    assert contract_to_json(first) == contract_to_json(second)


def test_contract_from_json_restores_nested_tuple_fields():
    restored = contract_from_json(contract_to_json(_stage_contract()))

    assert isinstance(restored.inputs, tuple)
    assert isinstance(restored.outputs, tuple)
    assert isinstance(restored.substages, tuple)
    assert isinstance(restored.substages[0].outputs, tuple)


def test_write_contract_writes_only_to_explicit_path(tmp_path):
    target = tmp_path / "nested" / "2_build_calibration_package.json"

    write_contract(_stage_contract(), target)

    assert target.exists()
    assert not (tmp_path / "contracts").exists()


def test_read_contract_restores_stage_contract(tmp_path):
    contract = _stage_contract()
    target = tmp_path / "2_build_calibration_package.json"
    write_contract(contract, target)

    restored = read_contract(target)

    assert restored == contract


def test_write_contract_replaces_existing_file_without_temp_files(tmp_path):
    target = tmp_path / "2_build_calibration_package.json"
    write_contract(_stage_contract(parameters={"seed": 1}), target)

    write_contract(_stage_contract(parameters={"seed": 2}), target)

    assert read_contract(target).parameters["seed"] == 2
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_contract_json_ends_with_newline():
    assert contract_to_json(_stage_contract()).endswith("\n")


def test_contract_json_top_level_keys_are_sorted():
    lines = contract_to_json(_stage_contract()).splitlines()

    assert lines[0] == "{"
    assert lines[1].strip().startswith('"code_sha"')


def test_fingerprint_material_is_stable_across_mapping_order():
    first = fingerprint_material(
        {
            "stage_id": "2_build_calibration_package",
            "parameters": {"seed": 42, "n_clones": 430},
        }
    )
    second = fingerprint_material(
        {
            "parameters": {"n_clones": 430, "seed": 42},
            "stage_id": "2_build_calibration_package",
        }
    )

    assert first == second
    assert first.algorithm == CONTRACT_FINGERPRINT_ALGORITHM
    assert first.value.startswith("sha256:")


def test_fingerprint_material_changes_when_semantic_value_changes():
    first = fingerprint_material({"stage_id": "3_fit_weights", "seed": 42})
    second = fingerprint_material({"stage_id": "3_fit_weights", "seed": 43})

    assert first.value != second.value


def test_canonicalize_for_fingerprint_converts_paths_to_strings():
    material = canonicalize_for_fingerprint(
        {"artifact": Path("contracts") / "2_build_calibration_package.json"}
    )

    assert material == {"artifact": "contracts/2_build_calibration_package.json"}


def test_canonicalize_for_fingerprint_uses_to_dict_objects():
    artifact = _artifact()

    material = canonicalize_for_fingerprint({"artifact": artifact})

    assert material["artifact"]["logical_name"] == "policy_data_db"
    assert material["artifact"]["metadata"] == {"source": "stage_1"}


def test_canonicalize_for_fingerprint_rejects_plain_dataclasses():
    @dataclass(frozen=True)
    class PlainRecord:
        value: int

    with pytest.raises(TypeError, match="to_dict"):
        canonicalize_for_fingerprint(PlainRecord(value=1))


def test_canonicalize_for_fingerprint_rejects_unsupported_values():
    with pytest.raises(TypeError, match="Unsupported fingerprint material"):
        canonicalize_for_fingerprint(object())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonicalize_for_fingerprint_rejects_non_finite_floats(value):
    with pytest.raises(ValueError, match="finite"):
        canonicalize_for_fingerprint({"duration_s": value})


def test_stage_contract_package_exports_public_api():
    import policyengine_us_data.stage_contracts as contracts

    assert contracts.ArtifactRef is ArtifactRef
    assert contracts.DiagnosticRef is DiagnosticRef
    assert contracts.Fingerprint is Fingerprint
    assert contracts.ValidationFinding is ValidationFinding
    assert contracts.ValidationReport is ValidationReport
    assert contracts.ReuseSummary is ReuseSummary
    assert contracts.ExecutionRecord is ExecutionRecord
    assert contracts.SubstageRecord is SubstageRecord
    assert contracts.StageContract is StageContract
    assert contracts.contract_to_json is contract_to_json
    assert contracts.contract_from_json is contract_from_json
    assert contracts.write_contract is write_contract
    assert contracts.read_contract is read_contract
    assert contracts.canonicalize_for_fingerprint is canonicalize_for_fingerprint
    assert contracts.fingerprint_material is fingerprint_material
    assert set(contracts.__all__) >= {
        "ArtifactRef",
        "DiagnosticRef",
        "Fingerprint",
        "ValidationFinding",
        "ValidationReport",
        "ReuseSummary",
        "ExecutionRecord",
        "SubstageRecord",
        "StageContract",
        "contract_to_json",
        "contract_from_json",
        "write_contract",
        "read_contract",
        "canonicalize_for_fingerprint",
        "fingerprint_material",
    }


def test_stage_contract_package_import_has_no_heavy_side_effects():
    script = """
import importlib
import json
import sys

importlib.import_module("policyengine_us_data.stage_contracts")
blocked = ["modal", "pandas", "h5py", "torch", "policyengine_us"]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "h5py": False,
        "modal": False,
        "pandas": False,
        "policyengine_us": False,
        "torch": False,
    }
