from pathlib import Path

from policyengine_us_data.build_datasets import (
    DatasetSubstepResult,
    Stage1ValidationContext,
    Stage1ValidationRunner,
    ValidationTargetCatalog,
    iter_stage_1_validators,
    stage_1_step_specs,
    validators_for_substage,
)


def _substep_result(path: Path) -> DatasetSubstepResult:
    return DatasetSubstepResult(
        substep_id="1b_base_dataset_construction",
        title="Base dataset construction",
        status="completed",
        started_at="2026-05-20T12:00:00Z",
        completed_at="2026-05-20T12:00:01Z",
        duration_s=1.0,
        command_names=("cps.py",),
        artifact_paths=(str(path),),
    )


def test_stage_1_validator_registry_is_wired_from_step_specs():
    validator_ids = {validator.validator_id for validator in iter_stage_1_validators()}

    for spec in stage_1_step_specs():
        assert set(spec.validation_ids) <= validator_ids

    [validator] = validators_for_substage("1c_extended_cps_puf_clone")
    assert validator.validator_id == (
        "stage_1.1c_extended_cps_puf_clone.artifact_contract"
    )


def test_stage_1_validation_runner_validates_substep_artifact(tmp_path):
    artifact = tmp_path / "cps_2024.h5"
    artifact.write_bytes(b"tiny")

    report = Stage1ValidationRunner(run_id="run-a").run_for_substep_result(
        _substep_result(artifact)
    )

    assert report.status == "pass"
    assert report.metadata["substage_id"] == "1b_base_dataset_construction"
    assert report.findings == ()


def test_stage_1_validation_runner_reports_missing_required_logical_artifact():
    context = Stage1ValidationContext(
        run_id="run-a",
        substage_id="1b_base_dataset_construction",
        artifact_refs={},
    )

    report = Stage1ValidationRunner(run_id="run-a").run_for_context(
        context,
        required_artifacts=("cps_2024",),
    )

    assert report.status == "fail"
    [finding] = report.findings
    assert finding.check_id == "stage_1.1b_base_dataset_construction.artifact_contract"
    assert finding.metric == "required_artifact"
    assert finding.value == "cps_2024"


def test_validation_target_catalog_loads_active_targets_deterministically():
    catalog = ValidationTargetCatalog.from_stage_1_specs(
        skip_enhanced_cps=True,
        skip_stage_5=True,
    )

    ids = [target.target_id for target in catalog.targets]
    assert ids == sorted(ids)
    assert "small_enhanced_cps_2024" not in catalog.required_logical_names(
        "1d_enhanced_cps_reweighting"
    )
    assert catalog.required_logical_names("1g_stage_base_datasets") == (
        "build_log",
        "data_build_checkpoint_stats",
        "policy_data_db",
    )


def test_stage_1_validation_runner_rejects_empty_artifacts(tmp_path):
    artifact = tmp_path / "cps_2024.h5"
    artifact.touch()

    report = Stage1ValidationRunner(run_id="run-a").run_for_substep_result(
        _substep_result(artifact)
    )

    assert report.status == "fail"
    assert report.findings[0].metric == "artifact_size_bytes"


def test_stage_1_validation_runner_rejects_missing_declared_artifacts(tmp_path):
    artifact = tmp_path / "cps_2024.h5"

    report = Stage1ValidationRunner(run_id="run-a").run_for_substep_result(
        _substep_result(artifact)
    )

    assert report.status == "fail"
    assert report.findings[0].metric == "artifact_exists"


def test_validators_for_unknown_substage_returns_empty_tuple():
    assert validators_for_substage("unknown") == ()
