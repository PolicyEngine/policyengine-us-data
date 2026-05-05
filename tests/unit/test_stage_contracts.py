import pytest

from policyengine_us_data.stage_contracts.core import (
    CONTRACT_FINGERPRINT_ALGORITHM,
    CONTRACT_SCHEMA_VERSION,
    ArtifactRef,
    ExecutionRecord,
    Fingerprint,
    ReuseSummary,
    StageContract,
    SubstageRecord,
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
    contract = StageContract(
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
        parameters={"n_clones": 430, "seed": 42},
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

    restored = StageContract.from_dict(contract.to_dict())

    assert restored == contract
    assert isinstance(restored.inputs, tuple)
    assert isinstance(restored.outputs, tuple)
    assert isinstance(restored.substages, tuple)
    assert restored.execution.status == "completed"


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
