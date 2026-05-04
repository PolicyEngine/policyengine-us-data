import json
from unittest.mock import MagicMock, patch

from modal_app.step_manifests.state import RunMetadata
from modal_app.step_manifests.store import read_run_meta, write_run_meta

from policyengine_us_data.utils.step_manifest import (
    ArtifactReference,
    ReuseMeasurement,
    RunManifest,
    StepManifest,
    completed_validated_outputs,
    evaluate_step_reuse,
    read_step_manifest,
    run_manifest_path,
    step_manifest_path,
    validate_step_outputs,
    write_run_manifest,
    write_step_manifest,
)


def _write(path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_step_manifest_serialization_is_deterministic(tmp_path):
    output = _write(tmp_path / "artifacts" / "out.h5", b"dataset")
    manifest = StepManifest(
        run_id="run-1",
        step_id="04_build_h5_regional",
        scope="regional",
        status="completed",
        attempt=1,
        started_at="2026-04-30T12:00:00+00:00",
        completed_at="2026-04-30T12:00:02+00:00",
        duration_s=2.0,
        branch="main",
        sha="abc123",
        version="1.0.0",
        parameters={"n_clones": 1, "validate": True},
        input_identities={"h5_scope_fingerprint": "abc123fingerprint"},
        outputs=[ArtifactReference.from_path(output, base_dir=tmp_path)],
        reuse_decision="computed",
    )

    first = manifest.to_json()
    second = manifest.to_json()

    assert first == second
    assert json.loads(first)["input_identities"] == {
        "h5_scope_fingerprint": "abc123fingerprint"
    }


def test_evaluate_step_reuse_requires_matching_inputs_parameters_and_outputs(tmp_path):
    output = _write(tmp_path / "out.h5", b"dataset")
    manifest = StepManifest(
        run_id="run-1",
        step_id="01_build_datasets",
        status="completed",
        attempt=1,
        started_at="2026-04-30T12:00:00+00:00",
        completed_at="2026-04-30T12:00:01+00:00",
        parameters={"branch": "main"},
        input_identities={"sha": "abc"},
        outputs=[ArtifactReference.from_path(output, base_dir=tmp_path)],
        reuse_decision="computed",
    )
    manifest_path = tmp_path / "step.json"
    write_step_manifest(manifest_path, manifest)

    decision = evaluate_step_reuse(
        manifest_path,
        expected_input_identities={"sha": "abc"},
        expected_parameters={"branch": "main"},
        output_root=tmp_path,
    )

    assert decision.reusable is True
    assert decision.reason == "prior_success"


def test_evaluate_step_reuse_allows_derived_input_identity_fields(tmp_path):
    output = _write(tmp_path / "out.h5", b"dataset")
    manifest = StepManifest(
        run_id="run-1",
        step_id="04_build_h5_regional",
        status="completed",
        attempt=1,
        started_at="2026-04-30T12:00:00+00:00",
        parameters={"n_clones": 1},
        input_identities={
            "weights": {"sha256": "abc"},
            "h5_scope_fingerprint": "derived-after-run",
        },
        outputs=[ArtifactReference.from_path(output, base_dir=tmp_path)],
        reuse_decision="computed",
    )
    manifest_path = tmp_path / "step.json"
    write_step_manifest(manifest_path, manifest)

    decision = evaluate_step_reuse(
        manifest_path,
        expected_input_identities={"weights": {"sha256": "abc"}},
        expected_parameters={"n_clones": 1},
        output_root=tmp_path,
    )

    assert decision.reusable is True


def test_evaluate_step_reuse_recomputes_when_output_checksum_changes(tmp_path):
    output = _write(tmp_path / "out.h5", b"dataset")
    manifest = StepManifest(
        run_id="run-1",
        step_id="01_build_datasets",
        status="completed",
        attempt=1,
        started_at="2026-04-30T12:00:00+00:00",
        outputs=[ArtifactReference.from_path(output, base_dir=tmp_path)],
        reuse_decision="computed",
    )
    manifest_path = tmp_path / "step.json"
    write_step_manifest(manifest_path, manifest)
    output.write_bytes(b"changed")

    decision = evaluate_step_reuse(manifest_path, output_root=tmp_path)

    assert decision.reusable is False
    assert decision.reason == "checksum_mismatch"
    assert decision.validation.checksum_mismatches == ("out.h5",)


def test_validate_step_outputs_reports_missing_files(tmp_path):
    output = _write(tmp_path / "out.h5", b"dataset")
    artifact = ArtifactReference.from_path(output, base_dir=tmp_path)
    output.unlink()
    manifest = StepManifest(
        run_id="run-1",
        step_id="01_build_datasets",
        status="completed",
        attempt=1,
        started_at="2026-04-30T12:00:00+00:00",
        outputs=[artifact],
        reuse_decision="computed",
    )

    validation = validate_step_outputs(manifest, root=tmp_path)

    assert validation.valid is False
    assert validation.reason == "missing_output"
    assert validation.missing_outputs == ("out.h5",)


def test_partial_h5_reuse_counts_are_manifest_fields(tmp_path):
    output = _write(tmp_path / "staging" / "run-1" / "districts" / "NC-01.h5", b"h5")
    manifest = StepManifest(
        run_id="run-1",
        step_id="04_build_h5_regional",
        scope="regional",
        status="partially_reused",
        attempt=2,
        started_at="2026-04-30T12:00:00+00:00",
        outputs=[ArtifactReference.from_path(output)],
        reuse_decision="partially_reused",
        reuse_reason="prior_success",
        reuse_measurement=ReuseMeasurement(
            expected_outputs=3,
            valid_reused_outputs=1,
            recomputed_outputs=2,
            invalid_outputs=0,
        ),
    )

    data = manifest.to_dict()

    assert data["reuse_measurement"] == {
        "expected_outputs": 3,
        "valid_reused_outputs": 1,
        "recomputed_outputs": 2,
        "invalid_outputs": 0,
    }


def test_completed_validated_outputs_reads_release_candidates_from_steps(tmp_path):
    run_dir = tmp_path / "runs" / "run-1"
    output = _write(tmp_path / "staging" / "run-1" / "states" / "NC.h5", b"h5")
    stale = _write(tmp_path / "staging" / "run-1" / "states" / "SC.h5", b"stale")
    write_run_manifest(
        run_manifest_path(run_dir),
        RunManifest(
            run_id="run-1",
            branch="main",
            sha="abc",
            version="1.0.0",
            status="completed",
            started_at="2026-04-30T12:00:00+00:00",
            known_step_ids=["04_build_h5_regional"],
        ),
    )
    write_step_manifest(
        step_manifest_path(run_dir, "04_build_h5_regional"),
        StepManifest(
            run_id="run-1",
            step_id="04_build_h5_regional",
            status="completed",
            attempt=1,
            started_at="2026-04-30T12:00:00+00:00",
            outputs=[ArtifactReference.from_path(output)],
            reuse_decision="computed",
        ),
    )
    write_step_manifest(
        step_manifest_path(run_dir, "04_build_h5_stale"),
        StepManifest(
            run_id="run-1",
            step_id="04_build_h5_stale",
            status="completed",
            attempt=1,
            started_at="2026-04-30T12:00:00+00:00",
            outputs=[ArtifactReference.from_path(stale)],
            reuse_decision="computed",
        ),
    )
    stale.write_bytes(b"changed")

    outputs = completed_validated_outputs(run_dir)

    assert [artifact.path for artifact in outputs] == [str(output)]
    assert (
        read_step_manifest(step_manifest_path(run_dir, "04_build_h5_regional")).step_id
        == "04_build_h5_regional"
    )


def test_run_state_is_stored_in_run_manifest_not_meta_json(tmp_path):
    meta = RunMetadata(
        run_id="run-1",
        branch="main",
        sha="abc123",
        version="1.0.0",
        start_time="2026-04-30T12:00:00+00:00",
        status="running",
        run_context={"github_run_id": "12345"},
        modal_app_name="policyengine-us-data-pipeline-run-1",
        modal_environment="main",
    )
    volume = MagicMock()
    runs_dir = tmp_path / "runs"

    with patch("modal_app.step_manifests.state.RUNS_DIR", str(runs_dir)):
        write_run_meta(meta, volume)
        roundtripped = read_run_meta("run-1", volume)

    assert (runs_dir / "run-1" / "run_manifest.json").exists()
    assert not (runs_dir / "run-1" / "meta.json").exists()
    assert roundtripped.run_id == meta.run_id
    assert roundtripped.run_context == {"github_run_id": "12345"}
    volume.commit.assert_called_once()
    volume.reload.assert_called_once()
