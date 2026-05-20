from pathlib import Path
import sys

import pytest

from policyengine_us_data.build_datasets import (
    CommandRunner,
    DatasetCommand,
    DatasetCommandError,
    Stage1ErrorRecord,
    Stage1Coordinator,
    stage_1_substep_id_for_script,
    stage_1_substep_title,
)


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


def test_coordinator_aggregates_canonical_substep_results(tmp_path):
    coordinator = Stage1Coordinator()
    first_artifact = tmp_path / "acs.h5"
    second_artifact = tmp_path / "puf.h5"

    coordinator.run_substep(
        "1b_base_dataset_construction",
        "Base dataset construction",
        lambda: first_artifact.write_text("acs"),
        command_names=("build-acs",),
        artifact_paths=(first_artifact,),
        aggregate=True,
    )
    coordinator.run_substep(
        "1b_base_dataset_construction",
        "Base dataset construction",
        lambda: second_artifact.write_text("puf"),
        command_names=("build-puf",),
        artifact_paths=(second_artifact,),
        aggregate=True,
    )

    assert coordinator.results == []
    assert [event.status for event in coordinator.status_events] == ["started"]

    coordinator.finalize_results()

    [result] = coordinator.results
    assert result.status == "completed"
    assert result.command_names == ("build-acs", "build-puf")
    assert result.artifact_paths == (str(first_artifact), str(second_artifact))
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


def test_coordinator_preserves_failed_command_result_details():
    coordinator = Stage1Coordinator()
    command_results = []
    command = DatasetCommand(
        name="failing command",
        argv=(
            sys.executable,
            "-c",
            "import sys; print('structured failure'); sys.exit(7)",
        ),
    )

    def action():
        return CommandRunner(output_tail_lines=5).run(command)

    with pytest.raises(DatasetCommandError):
        coordinator.run_substep(
            "1c_extended_cps_puf_clone",
            "Extended CPS PUF clone",
            action,
            command_names=("failing command",),
            command_results=command_results,
        )

    [result] = coordinator.results
    assert result.status == "failed"
    assert len(result.command_results) == 1
    command_result = result.command_results[0]
    assert command_result.returncode == 7
    assert command_result.combined_output_tail == ("structured failure\n",)
    assert result.error is not None
    assert result.error.substep_id == "1c_extended_cps_puf_clone"
    assert result.error.command_name == "failing command"
    assert result.error.returncode == 7
    assert result.error.metadata["argv"] == list(command.argv)
    assert result.error.metadata["output_tail"] == ["structured failure\n"]

    durable_error = result.error.to_pipeline_error_record(
        run_id="run-123",
        branch="stage-1",
        sha="abc123",
        version="1.0.0",
    )
    assert durable_error.stage_id == "1_build_datasets"
    assert durable_error.substage_id == "1c_extended_cps_puf_clone"
    assert durable_error.error_type == "RuntimeError"
    assert "structured failure" in durable_error.traceback


def test_stage_1_error_record_adapter_redacts_durable_error_text():
    error = Stage1ErrorRecord(
        substep_id="1c_extended_cps_puf_clone",
        command_name="secret command",
        error_type="DatasetCommandError",
        message="failed with API_TOKEN=secret-value",
        returncode=7,
        metadata={
            "argv": ["python", "-m", "secret"],
            "output_tail": ["output contained secret-value\n"],
        },
    )

    durable_error = error.to_pipeline_error_record(
        run_id="run-123",
        branch="stage-1",
        sha="abc123",
        version="1.0.0",
        env={"API_TOKEN": "secret-value"},
    )

    assert durable_error.stage_id == "1_build_datasets"
    assert durable_error.substage_id == "1c_extended_cps_puf_clone"
    assert durable_error.error_type == "DatasetCommandError"
    assert "secret-value" not in durable_error.message
    assert "secret-value" not in durable_error.traceback
    assert "API_TOKEN=<redacted>" in durable_error.message
    assert "<redacted:API_TOKEN>" in durable_error.traceback


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
