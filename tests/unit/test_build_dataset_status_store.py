import json

from policyengine_us_data.build_datasets import (
    DatasetSubstepResult,
    Stage1Coordinator,
    Stage1StatusEvent,
    Stage1StatusRecorder,
    read_stage_1_status_snapshot,
)


def test_stage_1_status_recorder_persists_events_results_and_current(tmp_path):
    commits = []
    recorder = Stage1StatusRecorder(
        tmp_path / "runs" / "run-1",
        commit_callback=lambda: commits.append("commit"),
    )
    event = Stage1StatusEvent(
        substep_id="1c_extended_cps_puf_clone",
        status="started",
        created_at="2026-05-22T12:00:00Z",
        message="Started Extended CPS PUF clone",
    )
    result = DatasetSubstepResult(
        substep_id="1c_extended_cps_puf_clone",
        title="Extended CPS PUF clone",
        status="completed",
        started_at="2026-05-22T12:00:00Z",
        completed_at="2026-05-22T12:05:00Z",
        duration_s=300.0,
        command_names=("extended-cps",),
    )

    recorder.record_event(event)
    recorder.record_result(result)

    snapshot = read_stage_1_status_snapshot(tmp_path / "runs" / "run-1")

    assert commits == ["commit", "commit"]
    assert snapshot.current == {
        "substep_id": "1c_extended_cps_puf_clone",
        "status": "started",
        "created_at": "2026-05-22T12:00:00Z",
        "message": "Started Extended CPS PUF clone",
        "command_name": None,
        "metadata": {},
        "title": "Extended CPS PUF clone",
    }
    assert snapshot.events == (snapshot.current,)
    assert snapshot.results[0]["substep_id"] == "1c_extended_cps_puf_clone"
    assert snapshot.results[0]["status"] == "completed"


def test_stage_1_status_recorder_is_best_effort_by_default(tmp_path):
    def fail_commit():
        raise RuntimeError("volume unavailable")

    recorder = Stage1StatusRecorder(
        tmp_path / "runs" / "run-1",
        commit_callback=fail_commit,
    )

    recorder.record_event(
        Stage1StatusEvent(
            substep_id="1a_raw_data_download",
            status="started",
            created_at="2026-05-22T12:00:00Z",
        )
    )

    snapshot = read_stage_1_status_snapshot(tmp_path / "runs" / "run-1")

    assert snapshot.current["substep_id"] == "1a_raw_data_download"


def test_read_stage_1_status_snapshot_survives_malformed_records(tmp_path):
    run_dir = tmp_path / "runs" / "run-1"
    status_dir = run_dir / "stage_1"
    status_dir.mkdir(parents=True)
    (status_dir / "current_substep.json").write_text("[]\n")
    (status_dir / "status_events.jsonl").write_text(
        json.dumps(
            {
                "substep_id": "1a_raw_data_download",
                "status": "started",
                "created_at": "2026-05-22T12:00:00Z",
                "title": "Raw data download",
            }
        )
        + "\n{not-json\n"
    )

    snapshot = read_stage_1_status_snapshot(run_dir)

    assert snapshot.current["substep_id"] == "1a_raw_data_download"
    assert len(snapshot.events) == 1
    assert [error["error_type"] for error in snapshot.read_errors] == [
        "TypeError",
        "JSONDecodeError",
    ]


def test_stage_1_coordinator_writes_to_status_recorder(tmp_path):
    recorder = Stage1StatusRecorder(tmp_path / "runs" / "run-1")
    coordinator = Stage1Coordinator(status_recorder=recorder)

    coordinator.run_substep(
        "1b_base_dataset_construction",
        "Base dataset construction",
        lambda: "done",
        command_names=("build-cps",),
    )

    snapshot = read_stage_1_status_snapshot(tmp_path / "runs" / "run-1")

    assert [event["status"] for event in snapshot.events] == [
        "started",
        "completed",
    ]
    assert snapshot.current["status"] == "completed"
    assert snapshot.results[0]["substep_id"] == "1b_base_dataset_construction"


def test_stage_1_coordinator_writes_aggregated_status_on_finalize(tmp_path):
    recorder = Stage1StatusRecorder(tmp_path / "runs" / "run-1")
    coordinator = Stage1Coordinator(status_recorder=recorder)

    coordinator.run_substep(
        "1e_stratified_cps",
        "Stratified CPS",
        lambda: "done",
        command_names=("stratified-cps",),
        aggregate=True,
    )
    before_finalize = read_stage_1_status_snapshot(tmp_path / "runs" / "run-1")

    coordinator.finalize_results()
    after_finalize = read_stage_1_status_snapshot(tmp_path / "runs" / "run-1")

    assert [event["status"] for event in before_finalize.events] == ["started"]
    assert [event["status"] for event in after_finalize.events] == [
        "started",
        "completed",
    ]
    assert after_finalize.results[0]["substep_id"] == "1e_stratified_cps"
