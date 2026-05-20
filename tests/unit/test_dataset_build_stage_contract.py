from pathlib import Path

from policyengine_us_data.stage_contracts import (
    StageContract,
    contract_from_json,
    contract_to_json,
)
from policyengine_us_data.stage_contracts.dataset_build import (
    DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
    build_dataset_build_output_contract,
)
from policyengine_us_data.build_datasets import stage_1_contract_artifact_specs

_ARTIFACT_BYTES = {
    "acs_2022.h5": b"acs",
    "irs_puf_2015.h5": b"irs",
    "cps_2024.h5": b"cps",
    "puf_2024.h5": b"puf",
    "extended_cps_2024.h5": b"extended",
    "enhanced_cps_2024.h5": b"enhanced",
    "small_enhanced_cps_2024.h5": b"small",
    "stratified_extended_cps_2024.h5": b"stratified",
    "source_imputed_stratified_extended_cps_2024.h5": b"source-year",
    "source_imputed_stratified_extended_cps.h5": b"source-alias",
    "policy_data.db": b"sqlite",
    "build_log.txt": b"log",
    "data_build_checkpoint_stats.json": b'{"expected_outputs": 3}',
}


def _write_artifacts(
    artifacts_dir: Path,
    *,
    include_enhanced_cps: bool = True,
    include_stage_5: bool = True,
) -> None:
    artifacts_dir.mkdir(exist_ok=True)
    for filename, payload in _ARTIFACT_BYTES.items():
        if not include_enhanced_cps and filename in {
            "enhanced_cps_2024.h5",
            "small_enhanced_cps_2024.h5",
        }:
            continue
        if not include_stage_5 and filename in {
            "small_enhanced_cps_2024.h5",
            "source_imputed_stratified_extended_cps_2024.h5",
            "source_imputed_stratified_extended_cps.h5",
        }:
            continue
        (artifacts_dir / filename).write_bytes(payload)


def _contract(
    artifacts_dir: Path,
    *,
    run_id: str = "run-a",
    skip_enhanced_cps: bool = False,
    skip_stage_5: bool = False,
) -> StageContract:
    return build_dataset_build_output_contract(
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        code_sha="abc123",
        package_version="1.98.2",
        checkpoint_stats={
            "expected_outputs": 4,
            "valid_reused_outputs": 1,
            "recomputed_outputs": 3,
            "invalid_outputs": 0,
        },
        started_at="2026-05-08T12:00:00Z",
        completed_at="2026-05-08T12:01:00Z",
        duration_s=60.0,
        upload_requested=True,
        stage_only=False,
        skip_enhanced_cps=skip_enhanced_cps,
        skip_stage_5=skip_stage_5,
    )


def test_dataset_build_contract_records_stage_1_handoff_artifacts(tmp_path):
    _write_artifacts(tmp_path)

    contract = _contract(tmp_path)

    assert contract.stage_id == "1_build_datasets"
    assert contract.contract_type == "dataset_build_output"
    assert contract.run_id == "run-a"
    logical_names = {artifact.logical_name for artifact in contract.outputs}
    assert {
        "source_imputed_stratified_extended_cps",
        "source_imputed_stratified_extended_cps_2024",
        "policy_data_db",
        "build_log",
        "data_build_checkpoint_stats",
    } <= logical_names
    assert all(artifact.sha256.startswith("sha256:") for artifact in contract.outputs)
    assert all(artifact.uri.startswith("file://") for artifact in contract.outputs)


def test_dataset_build_contract_outputs_use_shared_artifact_specs(tmp_path):
    _write_artifacts(tmp_path)

    contract = _contract(tmp_path)

    assert {artifact.logical_name for artifact in contract.outputs} == {
        spec.logical_name for spec in stage_1_contract_artifact_specs()
    }


def test_dataset_build_contract_records_substage_shape(tmp_path):
    _write_artifacts(tmp_path)

    contract = _contract(tmp_path)

    assert [record.substage_id for record in contract.substages] == [
        "1a_raw_data_download",
        "1b_base_dataset_construction",
        "1c_extended_cps_puf_clone",
        "1d_enhanced_cps_reweighting",
        "1e_stratified_cps",
        "1f_source_imputation",
        "1g_stage_base_datasets",
    ]
    records = {record.substage_id: record for record in contract.substages}
    assert records["1d_enhanced_cps_reweighting"].status == "completed"
    assert records["1f_source_imputation"].reuse_mode == "handoff"
    assert {
        artifact.logical_name for artifact in records["1g_stage_base_datasets"].outputs
    } >= {"policy_data_db", "build_log", "data_build_checkpoint_stats"}


def test_dataset_build_contract_omits_enhanced_cps_when_skipped(tmp_path):
    _write_artifacts(tmp_path, include_enhanced_cps=False)

    contract = _contract(tmp_path, skip_enhanced_cps=True)

    logical_names = {artifact.logical_name for artifact in contract.outputs}
    assert "enhanced_cps_2024" not in logical_names
    assert "small_enhanced_cps_2024" not in logical_names
    records = {record.substage_id: record for record in contract.substages}
    assert records["1d_enhanced_cps_reweighting"].status == "skipped"
    assert records["1d_enhanced_cps_reweighting"].outputs == ()


def test_dataset_build_contract_omits_phase_5_artifacts_when_skipped(tmp_path):
    _write_artifacts(tmp_path, include_stage_5=False)

    contract = _contract(tmp_path, skip_stage_5=True)

    logical_names = {artifact.logical_name for artifact in contract.outputs}
    assert "enhanced_cps_2024" in logical_names
    assert "small_enhanced_cps_2024" not in logical_names
    assert "source_imputed_stratified_extended_cps_2024" not in logical_names
    assert "source_imputed_stratified_extended_cps" not in logical_names
    assert contract.parameters["skip_stage_5"] is True
    records = {record.substage_id: record for record in contract.substages}
    assert records["1d_enhanced_cps_reweighting"].status == "completed"
    assert records["1f_source_imputation"].status == "skipped"


def test_dataset_build_contract_rejects_missing_required_stage_1_artifact(tmp_path):
    _write_artifacts(tmp_path)
    (tmp_path / "acs_2022.h5").unlink()

    try:
        _contract(tmp_path)
    except FileNotFoundError as exc:
        assert "acs_2022.h5" in str(exc)
    else:
        raise AssertionError("Missing Stage 1 artifact should fail")


def test_dataset_build_contract_execution_records_checkpoint_summary(tmp_path):
    _write_artifacts(tmp_path)

    contract = _contract(tmp_path)

    assert contract.execution.status == "completed"
    assert contract.execution.reuse_decision == "partially_reused"
    assert contract.execution.reuse_summary.expected_outputs == 4
    assert contract.execution.reuse_summary.valid_reused_outputs == 1
    assert contract.execution.reuse_summary.recomputed_outputs == 3
    assert contract.execution.reuse_summary.invalid_outputs == 0


def test_dataset_build_contract_json_round_trip_is_deterministic(tmp_path):
    _write_artifacts(tmp_path)
    contract = _contract(tmp_path)

    payload = contract_to_json(contract)

    assert payload == contract_to_json(contract_from_json(payload))
    assert contract_from_json(payload) == contract


def test_dataset_build_contract_fingerprint_excludes_run_id(tmp_path):
    _write_artifacts(tmp_path)

    first = _contract(tmp_path, run_id="run-a")
    second = _contract(tmp_path, run_id="run-b")

    assert first.fingerprint == second.fingerprint


def test_dataset_build_contract_filename_is_stable():
    assert DATASET_BUILD_OUTPUT_CONTRACT_FILENAME == "dataset_build_output.json"
