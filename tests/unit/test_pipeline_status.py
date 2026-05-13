import json
from unittest.mock import MagicMock, patch

import pytest

from modal_app.step_manifests.errors import (
    PipelineErrorRecord,
    build_pipeline_error_record,
    clear_latest_pipeline_error,
    read_latest_pipeline_error,
    write_pipeline_error_record,
)
from modal_app.step_manifests.specs import BUILD_DATASETS, WEIGHT_FITTING_REGIONAL
from modal_app.step_manifests.status import build_pipeline_status_payload
from modal_app.step_manifests.store import fail_step_manifest
from policyengine_us_data.utils.step_manifest import (
    RunManifest,
    StepManifest,
    read_step_manifest,
    run_manifest_path,
    step_manifest_path,
    write_run_manifest,
    write_step_manifest,
)


def _manifest(step_id: str, *, parent_step_id: str | None = None) -> StepManifest:
    return StepManifest(
        run_id="run-1",
        step_id=step_id,
        parent_step_id=parent_step_id,
        status="running",
        attempt=1,
        started_at="2026-05-12T12:00:00+00:00",
        branch="main",
        sha="abc123",
        version="1.0.0",
        modal_app_name="policyengine-us-data-pipeline-run-1",
        modal_environment="main",
    )


def test_pipeline_error_record_uses_stage_and_substage_ids(tmp_path):
    exc = RuntimeError("failed with secret-value and API_TOKEN=secret-value")
    manifest = _manifest(
        WEIGHT_FITTING_REGIONAL.id,
        parent_step_id=WEIGHT_FITTING_REGIONAL.parent_id,
    )

    record = build_pipeline_error_record(
        exc,
        run_id="run-1",
        manifest=manifest,
        surface="run_pipeline",
        traceback_text="Traceback contains secret-value",
        occurred_at="2026-05-12T12:00:00+00:00",
        env={"API_TOKEN": "secret-value"},
    )
    result = write_pipeline_error_record(record, run_dir=tmp_path / "runs" / "run-1")

    latest = read_latest_pipeline_error(tmp_path / "runs" / "run-1")

    assert result.record.stage_id == "3_fit_weights"
    assert result.record.substage_id == "3a_weight_fitting_regional"
    assert result.record.record_path.endswith("3a_weight_fitting_regional.json")
    assert result.record_ref.role == "error"
    assert latest == result.record
    assert "secret-value" not in result.record.message
    assert "secret-value" not in result.record.traceback
    assert result.record.message == (
        "failed with <redacted:API_TOKEN> and API_TOKEN=<redacted>"
    )


def test_pipeline_error_record_persists_exact_schema_and_bounds_status_payload(
    tmp_path,
):
    long_message = "old message " + ("m" * 3_000) + " newest message"
    long_traceback = "old traceback\n" + ("x" * 30_000) + "\nnewest traceback"
    record = build_pipeline_error_record(
        RuntimeError(long_message),
        run_id="run-1",
        manifest=_manifest(BUILD_DATASETS.id),
        traceback_text=long_traceback,
        occurred_at="2026-05-12T12:00:00+00:00",
    )
    result = write_pipeline_error_record(record, run_dir=tmp_path / "runs" / "run-1")

    persisted = json.loads((tmp_path / result.record.record_path).read_text())
    status_payload = result.record.to_status_dict()

    assert persisted["message"] == long_message
    assert persisted["traceback"] == long_traceback
    assert "traceback_available" not in persisted
    assert "traceback_truncated" not in persisted
    assert "message_truncated" not in persisted
    assert status_payload["message_truncated"] is True
    assert status_payload["message"].endswith("newest message")
    assert "old message" not in status_payload["message"]
    assert status_payload["traceback_truncated"] is True
    assert status_payload["traceback"].endswith("newest traceback")
    assert "old traceback" not in status_payload["traceback"]


def test_pipeline_error_record_infers_canonical_stage_without_manifest_parent():
    manifest = _manifest(WEIGHT_FITTING_REGIONAL.id)

    record = build_pipeline_error_record(
        RuntimeError("fit failed"),
        run_id="run-1",
        manifest=manifest,
        traceback_text="traceback",
    )

    assert record.stage_id == "3_fit_weights"
    assert record.substage_id == "3a_weight_fitting_regional"


def test_pipeline_error_record_rejects_invalid_explicit_stage_pair():
    with pytest.raises(ValueError, match="does not belong"):
        build_pipeline_error_record(
            RuntimeError("bad stage"),
            run_id="run-1",
            stage_id="2_build_calibration_package",
            substage_id="3a_weight_fitting_regional",
            traceback_text="traceback",
        )


def test_pipeline_error_record_rejects_invalid_stored_stage_pair():
    with pytest.raises(ValueError, match="does not belong"):
        PipelineErrorRecord.from_dict(
            {
                "run_id": "run-1",
                "stage_id": "2_build_calibration_package",
                "substage_id": "3a_weight_fitting_regional",
                "surface": "run_pipeline",
                "occurred_at": "2026-05-12T12:00:00+00:00",
                "error_type": "RuntimeError",
                "message": "bad stage",
                "traceback": "traceback",
            }
        )


def test_failed_step_manifest_can_reference_durable_error_record(tmp_path):
    manifest = _manifest(BUILD_DATASETS.id)
    ref = {
        "path": "runs/run-1/errors/error.json",
        "size_bytes": 10,
        "sha256": "abc",
        "role": "error",
        "media_type": "application/json",
    }
    volume = MagicMock()
    runs_dir = tmp_path / "runs"

    with patch("modal_app.step_manifests.state.RUNS_DIR", str(runs_dir)):
        fail_step_manifest(manifest, RuntimeError("boom"), volume, traceback_ref=ref)

    failed = read_step_manifest(
        step_manifest_path(runs_dir / "run-1", BUILD_DATASETS.id)
    )
    assert failed.status == "failed"
    assert failed.error == {
        "type": "RuntimeError",
        "message": "boom",
        "traceback_ref": ref,
    }
    volume.commit.assert_called_once()


def test_failed_step_manifest_redacts_error_message(tmp_path, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "secret-value")
    manifest = _manifest(BUILD_DATASETS.id)
    volume = MagicMock()
    runs_dir = tmp_path / "runs"

    with patch("modal_app.step_manifests.state.RUNS_DIR", str(runs_dir)):
        fail_step_manifest(
            manifest,
            RuntimeError("failed with secret-value and API_TOKEN=secret-value"),
            volume,
        )

    failed = read_step_manifest(
        step_manifest_path(runs_dir / "run-1", BUILD_DATASETS.id)
    )
    assert failed.error["message"] == (
        "failed with <redacted:API_TOKEN> and API_TOKEN=<redacted>"
    )


def test_status_payload_orders_manifests_and_includes_bounded_traceback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("API_TOKEN", "secret-value")
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-1"
    write_run_manifest(
        run_manifest_path(run_dir),
        RunManifest(
            run_id="run-1",
            branch="main",
            sha="abc123",
            version="1.0.0",
            status="failed",
            started_at="2026-05-12T12:00:00+00:00",
            known_step_ids=[
                BUILD_DATASETS.id,
                WEIGHT_FITTING_REGIONAL.id,
                "4a_local_area_h5_regional",
            ],
            run_context={"pipeline_volume_name": "pipeline-artifacts-run-1"},
            modal_app_name="policyengine-us-data-pipeline-run-1",
            modal_environment="main",
            error="RuntimeError: raw secret-value\nTraceback contains secret-value",
        ),
    )
    write_step_manifest(
        step_manifest_path(run_dir, WEIGHT_FITTING_REGIONAL.id),
        _manifest(
            WEIGHT_FITTING_REGIONAL.id,
            parent_step_id=WEIGHT_FITTING_REGIONAL.parent_id,
        ).fail(RuntimeError("fit failed")),
    )
    write_step_manifest(
        step_manifest_path(run_dir, BUILD_DATASETS.id),
        _manifest(BUILD_DATASETS.id).complete(),
    )
    error_record = build_pipeline_error_record(
        RuntimeError("fit failed with secret-value"),
        run_id="run-1",
        manifest=_manifest(
            WEIGHT_FITTING_REGIONAL.id,
            parent_step_id=WEIGHT_FITTING_REGIONAL.parent_id,
        ),
        traceback_text="full traceback with secret-value",
        occurred_at="2026-05-12T12:00:01+00:00",
    )
    write_pipeline_error_record(error_record, run_dir=run_dir, volume_root=tmp_path)

    payload = build_pipeline_status_payload("run-1", runs_dir=runs_dir)

    assert payload["status"] == "failed"
    assert payload["stage_manifests"][0]["stage_id"] == "1_build_datasets"
    assert payload["stage_manifests"][0]["substage_id"] is None
    assert payload["stage_manifests"][1]["stage_id"] == "3_fit_weights"
    assert payload["stage_manifests"][1]["substage_id"] == (
        "3a_weight_fitting_regional"
    )
    assert payload["missing_expected_manifest_ids"] == ["4a_local_area_h5_regional"]
    assert payload["error"]["stage_id"] == "3_fit_weights"
    assert payload["error"]["substage_id"] == "3a_weight_fitting_regional"
    assert payload["error"]["traceback_available"] is True
    assert payload["error"]["traceback"] == ("full traceback with <redacted:API_TOKEN>")
    assert payload["error"]["traceback_truncated"] is False
    assert "secret-value" not in payload["run_manifest"]["error"]
    assert payload["pipeline_volume_name"] == "pipeline-artifacts-run-1"


def test_status_payload_uses_run_manifest_error_as_last_resort(tmp_path, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "secret-value")
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-1"
    write_run_manifest(
        run_manifest_path(run_dir),
        RunManifest(
            run_id="run-1",
            branch="main",
            sha="abc123",
            version="1.0.0",
            status="failed",
            started_at="2026-05-12T12:00:00+00:00",
            known_step_ids=[BUILD_DATASETS.id],
            error="RuntimeError: old failure\nTraceback contains secret-value",
        ),
    )

    payload = build_pipeline_status_payload("run-1", runs_dir=runs_dir)

    assert payload["error"]["source"] == "run_manifest.error"
    assert payload["error"]["error_type"] == "RuntimeError"
    assert payload["error"]["message"] == "old failure"
    assert payload["error"]["traceback_available"] is True
    assert payload["error"]["traceback"].endswith(
        "Traceback contains <redacted:API_TOKEN>"
    )
    assert "secret-value" not in payload["run_manifest"]["error"]


def test_status_payload_bounds_run_manifest_error_message(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-1"
    write_run_manifest(
        run_manifest_path(run_dir),
        RunManifest(
            run_id="run-1",
            branch="main",
            sha="abc123",
            version="1.0.0",
            status="failed",
            started_at="2026-05-12T12:00:00+00:00",
            known_step_ids=[BUILD_DATASETS.id],
            error="RuntimeError: old message " + ("m" * 3_000) + " newest message",
        ),
    )

    payload = build_pipeline_status_payload("run-1", runs_dir=runs_dir)

    assert payload["error"]["message_truncated"] is True
    assert payload["error"]["message"].endswith("newest message")
    assert "old message" not in payload["error"]["message"]
    assert payload["run_manifest"]["error"].endswith("newest message")
    assert "old message" not in payload["run_manifest"]["error"]


def test_status_payload_truncates_oldest_traceback_text(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-1"
    write_run_manifest(
        run_manifest_path(run_dir),
        RunManifest(
            run_id="run-1",
            branch="main",
            sha="abc123",
            version="1.0.0",
            status="failed",
            started_at="2026-05-12T12:00:00+00:00",
            known_step_ids=[BUILD_DATASETS.id],
        ),
    )
    traceback_text = "oldest\n" + ("x" * 30_000) + "\nnewest failure line"
    error_record = build_pipeline_error_record(
        RuntimeError("fit failed"),
        run_id="run-1",
        manifest=_manifest(BUILD_DATASETS.id),
        traceback_text=traceback_text,
        occurred_at="2026-05-12T12:00:01+00:00",
    )
    write_pipeline_error_record(error_record, run_dir=run_dir, volume_root=tmp_path)

    latest = read_latest_pipeline_error(run_dir)
    payload = build_pipeline_status_payload("run-1", runs_dir=runs_dir)

    assert latest.traceback == traceback_text
    assert payload["error"]["traceback_truncated"] is True
    assert payload["error"]["traceback"].startswith(
        "\n[truncated older error text; omitted "
    )
    assert payload["error"]["traceback"].endswith("newest failure line")
    assert "oldest" not in payload["error"]["traceback"]


def test_clear_latest_pipeline_error_is_best_effort(tmp_path):
    run_dir = tmp_path / "runs" / "run-1"
    latest_path = run_dir / "errors" / "latest_error.json"
    latest_path.mkdir(parents=True)

    assert clear_latest_pipeline_error(run_dir) is False
    with pytest.raises(OSError):
        clear_latest_pipeline_error(run_dir, strict=True)


def test_status_payload_reports_missing_run(tmp_path):
    payload = build_pipeline_status_payload("missing-run", runs_dir=tmp_path / "runs")

    assert payload["status"] == "not_found"
    assert payload["stage_manifests"] == []
    assert payload["run_manifest"] is None
