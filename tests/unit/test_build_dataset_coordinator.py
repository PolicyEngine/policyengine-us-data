from pathlib import Path

import pytest

from policyengine_us_data.build_datasets import (
    Stage1Coordinator,
    stage_1_substep_id_for_script,
    stage_1_substep_title,
)
from policyengine_us_data.stage_contracts import ValidationFinding, ValidationReport


def test_stage_1_substep_mapping_uses_artifact_specs():
    assert (
        stage_1_substep_id_for_script("policyengine_us_data/datasets/cps/cps.py")
        == "1b_base_dataset_construction"
    )
    assert stage_1_substep_title("1b_base_dataset_construction") == (
        "Base dataset construction"
    )


def test_coordinator_records_completed_substep_and_artifacts(tmp_path):
    coordinator = Stage1Coordinator()
    artifact = tmp_path / "artifact.h5"

    def action():
        artifact.write_text("ok")
        return "done"

    result = coordinator.run_substep(
        "1b_base_dataset_construction",
        "Base dataset construction",
        action,
        command_names=("build-cps",),
        artifact_paths=(artifact,),
    )

    assert result == "done"
    [substep_result] = coordinator.results
    assert substep_result.status == "completed"
    assert substep_result.started_at is not None
    assert substep_result.completed_at is not None
    assert substep_result.duration_s is not None
    assert substep_result.command_names == ("build-cps",)
    assert substep_result.artifact_paths == (str(artifact),)
    assert [event.status for event in coordinator.status_events] == [
        "started",
        "completed",
    ]


def test_coordinator_records_skipped_substep_not_completed():
    coordinator = Stage1Coordinator()

    coordinator.run_substep(
        "1f_source_imputation",
        "Source imputation",
        lambda: None,
        command_names=("source-impute",),
        skip=True,
        skip_reason="--skip-stage-5",
    )

    [result] = coordinator.results
    assert result.status == "skipped"
    assert result.started_at is None
    assert result.metadata["skip_reason"] == "--skip-stage-5"
    assert coordinator.status_events[-1].status == "skipped"


def test_coordinator_records_failure_without_parsing_terminal_text():
    coordinator = Stage1Coordinator()

    def action():
        raise RuntimeError("structured failure")

    with pytest.raises(RuntimeError, match="structured failure"):
        coordinator.run_substep(
            "1c_extended_cps_puf_clone",
            "Extended CPS PUF clone",
            action,
            command_names=("extended-cps",),
        )

    [result] = coordinator.results
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.error_type == "RuntimeError"
    assert result.error.command_name == "extended-cps"
    assert coordinator.error_records == [result.error]


def test_fake_substep_runner_collects_tiny_artifacts(tmp_path: Path):
    coordinator = Stage1Coordinator()
    outputs = [tmp_path / "one.txt", tmp_path / "two.txt"]

    def action():
        for path in outputs:
            path.write_text(path.stem)

    coordinator.run_substep(
        "1g_stage_base_datasets",
        "Stage base datasets",
        action,
        command_names=("fake-stager",),
        artifact_paths=outputs,
    )

    [result] = coordinator.results
    assert result.status == "completed"
    assert result.artifact_paths == tuple(str(path) for path in outputs)


class _PassValidationRunner:
    def run_for_substep_result(self, result):
        return ValidationReport(
            status="pass",
            metadata={"substage_id": result.substep_id, "check_ids": ["check.pass"]},
        )

    def should_stop(self, report):
        return report.status == "fail"


class _FailValidationRunner:
    def run_for_substep_result(self, result):
        return ValidationReport(
            status="fail",
            findings=(
                ValidationFinding(
                    check_id="check.fail",
                    status="fail",
                    message="validator failed",
                ),
            ),
            metadata={"substage_id": result.substep_id, "check_ids": ["check.fail"]},
        )

    def should_stop(self, report):
        return report.status == "fail"


def test_coordinator_attaches_validation_report(tmp_path: Path):
    coordinator = Stage1Coordinator(validation_runner=_PassValidationRunner())
    artifact = tmp_path / "artifact.h5"

    def action():
        artifact.write_text("ok")

    coordinator.run_substep(
        "1b_base_dataset_construction",
        "Base dataset construction",
        action,
        artifact_paths=(artifact,),
    )

    [result] = coordinator.results
    assert result.validation_report["status"] == "pass"
    assert coordinator.status_events[-1].metadata["validation_report"]["status"] == (
        "pass"
    )


def test_coordinator_stops_after_error_level_validation_failure(tmp_path: Path):
    coordinator = Stage1Coordinator(validation_runner=_FailValidationRunner())
    artifact = tmp_path / "artifact.h5"

    def action():
        artifact.write_text("ok")

    with pytest.raises(RuntimeError, match="Stage 1 validation failed"):
        coordinator.run_substep(
            "1b_base_dataset_construction",
            "Base dataset construction",
            action,
            artifact_paths=(artifact,),
        )

    [result] = coordinator.results
    assert result.status == "failed"
    assert result.validation_report["status"] == "fail"
    assert coordinator.error_records == [result.error]
