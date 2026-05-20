import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

from policyengine_us_data.build_datasets import (
    CheckpointStore,
    Stage1ReuseDecision,
    Stage1ReuseManifestRecord,
    stage_1_script_outputs,
)
from policyengine_us_data.stage_contracts import read_contract


def _load_data_build_module():
    fake_modal = ModuleType("modal")

    class _FakeApp:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def local_entrypoint(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    fake_modal.App = _FakeApp
    fake_modal.Secret = SimpleNamespace(from_name=lambda *args, **kwargs: object())
    fake_modal.Volume = SimpleNamespace(from_name=lambda *args, **kwargs: object())

    fake_images = ModuleType("modal_app.images")
    fake_images.cpu_image = object()

    sys.modules["modal"] = fake_modal
    sys.modules["modal_app.images"] = fake_images
    sys.modules.pop("modal_app.data_build", None)
    return importlib.import_module("modal_app.data_build")


def test_checkpoint_stats_are_per_instance():
    data_build = _load_data_build_module()
    first = data_build.CheckpointStats()
    second = data_build.CheckpointStats()

    first.record(
        expected_outputs=3,
        valid_reused_outputs=1,
        recomputed_outputs=2,
        invalid_outputs=2,
    )

    assert first.snapshot() == {
        "expected_outputs": 3,
        "valid_reused_outputs": 1,
        "recomputed_outputs": 2,
        "invalid_outputs": 2,
    }
    assert second.snapshot() == {
        "expected_outputs": 0,
        "valid_reused_outputs": 0,
        "recomputed_outputs": 0,
        "invalid_outputs": 0,
    }


def test_script_outputs_are_generated_from_stage_1_artifact_specs():
    data_build = _load_data_build_module()

    assert data_build.SCRIPT_OUTPUTS == stage_1_script_outputs()


def test_script_identity_material_distinguishes_same_substep_units():
    data_build = _load_data_build_module()

    cps = data_build._script_identity_material(
        script_path=data_build.CPS_BUILD_SCRIPT,
        output_files=["cps_2024.h5"],
        branch="branch",
    )
    puf = data_build._script_identity_material(
        script_path=data_build.PUF_BUILD_SCRIPT,
        output_files=["puf_2024.h5"],
        branch="branch",
    )

    assert cps.substep_id == "1b_base_dataset_construction"
    assert puf.substep_id == "1b_base_dataset_construction"
    assert cps.identity_key != puf.identity_key
    assert data_build.CPS_BUILD_SCRIPT in cps.identity_key
    assert "cps_2024.h5" in cps.identity_key
    assert data_build.PUF_BUILD_SCRIPT in puf.identity_key
    assert "puf_2024.h5" in puf.identity_key


def test_build_datasets_records_stage_base_handoff_substep(tmp_path, monkeypatch):
    data_build = _load_data_build_module()
    calls = []
    command_envs = []
    created_coordinators = []

    class FakeVolume:
        def reload(self):
            pass

        def commit(self):
            pass

    class FakeCoordinator:
        def __init__(self):
            self.status_recorder = None
            created_coordinators.append(self)
            calls.append(("coordinator_created", {}))

        def run_substep(self, substep_id, title, action, **kwargs):
            calls.append((substep_id, kwargs))
            return action()

        def finalize_results(self):
            calls.append(("finalize", {}))
            return ()

    class FakeStager:
        def __init__(self, *, context):
            self.context = context

        def stage_declared_artifacts(self, **kwargs):
            self.context.artifacts_dir.mkdir(parents=True, exist_ok=True)
            path = self.context.artifact_path("policy_data.db")
            path.write_bytes(b"db")
            return (path,)

        def write_checkpoint_stats(self, checkpoint_stats):
            path = self.context.artifact_path("data_build_checkpoint_stats.json")
            path.write_text("{}\n")
            return path

    def fake_write_contract(*, artifacts_dir, **kwargs):
        (artifacts_dir / "dataset_build_output.json").write_text("{}\n")

    def fake_run_script(script_path, *args, **kwargs):
        command_envs.append(dict(kwargs["env"]))
        return script_path

    def fake_run_script_logged(cmd, log_file, env, **kwargs):
        command_envs.append(dict(env))
        return data_build.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("US_DATA_RUN_ID", "outer-run")
    monkeypatch.setenv(data_build.CANDIDATE_VERSION_ENV, "outer-candidate")
    monkeypatch.delenv(data_build.DATA_PACKAGE_VERSION_ENV, raising=False)
    monkeypatch.setattr(data_build, "setup_gcp_credentials", lambda: None)
    monkeypatch.setattr(data_build, "checkpoint_volume", FakeVolume())
    monkeypatch.setattr(data_build, "pipeline_volume", FakeVolume())
    monkeypatch.setattr(data_build, "PIPELINE_MOUNT", str(tmp_path / "pipeline"))
    monkeypatch.setattr(data_build, "VOLUME_MOUNT", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(data_build.os, "chdir", lambda _path: None)
    monkeypatch.setattr(data_build, "get_current_commit", lambda: "abc123456")
    monkeypatch.setattr(data_build, "SCRIPT_OUTPUTS", {})
    monkeypatch.setattr(data_build, "run_script", fake_run_script)
    monkeypatch.setattr(data_build, "run_script_logged", fake_run_script_logged)
    monkeypatch.setattr(data_build, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_build, "Stage1Coordinator", FakeCoordinator)
    monkeypatch.setattr(data_build, "PipelineArtifactStager", FakeStager)
    monkeypatch.setattr(data_build, "write_stage_1_diagnostics", lambda **kwargs: ())
    monkeypatch.setattr(data_build, "write_dataset_build_contract", fake_write_contract)
    monkeypatch.setattr(
        data_build, "validate_and_maybe_upload_datasets", lambda **kwargs: None
    )
    monkeypatch.setattr(data_build, "cleanup_checkpoints", lambda *args, **kwargs: None)

    assert (
        data_build.build_datasets(
            sequential=True,
            skip_tests=True,
            run_id="run-123",
            version="1.2.3",
        )
        == "Data build completed successfully"
    )

    assert [call[0] for call in calls] == [
        "coordinator_created",
        "1a_raw_data_download",
        "finalize",
        "1g_stage_base_datasets",
    ]
    assert created_coordinators[0].status_recorder is not None
    raw_data_kwargs = calls[1][1]
    assert raw_data_kwargs["reuse_decision"]["substep_id"] == "1a_raw_data_download"
    assert raw_data_kwargs["reuse_decision"]["reason"] == "no_previous_identity"
    assert raw_data_kwargs["checkpoint_decisions"] == ()
    stage_base_kwargs = calls[-1][1]
    assert stage_base_kwargs["command_names"] == ("stage_base_datasets",)
    assert any(
        str(path).endswith("dataset_build_output.json")
        for path in stage_base_kwargs["artifact_paths"]
    )
    assert {env["US_DATA_RUN_ID"] for env in command_envs} == {"run-123"}
    assert {env[data_build.CANDIDATE_VERSION_ENV] for env in command_envs} == {"1.2.3"}
    assert {env[data_build.DATA_PACKAGE_VERSION_ENV] for env in command_envs} == {
        "1.2.3"
    }
    assert data_build.os.environ["US_DATA_RUN_ID"] == "outer-run"
    assert data_build.os.environ[data_build.CANDIDATE_VERSION_ENV] == "outer-candidate"
    assert data_build.DATA_PACKAGE_VERSION_ENV not in data_build.os.environ


def test_validate_and_maybe_upload_datasets_validates_before_upload(monkeypatch):
    data_build = _load_data_build_module()
    calls = []

    def fake_run_script(script_path, args=None, env=None, log_file=None):
        calls.append((script_path, args or [], env))
        return script_path

    monkeypatch.setattr(data_build, "run_script", fake_run_script)

    data_build.validate_and_maybe_upload_datasets(
        upload=True,
        skip_enhanced_cps=False,
        env={"TEST_ENV": "1"},
        version="1.73.0",
    )

    assert calls == [
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            ["--validate-only"],
            {"TEST_ENV": "1"},
        ),
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            ["--version=1.73.0"],
            {"TEST_ENV": "1"},
        ),
    ]


def test_validate_and_maybe_upload_datasets_skips_upload_when_disabled(monkeypatch):
    data_build = _load_data_build_module()
    calls = []

    def fake_run_script(script_path, args=None, env=None, log_file=None):
        calls.append((script_path, args or [], env))
        return script_path

    monkeypatch.setattr(data_build, "run_script", fake_run_script)

    data_build.validate_and_maybe_upload_datasets(
        upload=False,
        skip_enhanced_cps=True,
        env={"TEST_ENV": "1"},
    )

    assert calls == [
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            ["--validate-only", "--no-require-enhanced-cps"],
            {"TEST_ENV": "1"},
        ),
    ]


def test_validate_and_maybe_upload_datasets_stages_with_run_id(monkeypatch):
    data_build = _load_data_build_module()
    calls = []

    def fake_run_script(script_path, args=None, env=None, log_file=None):
        calls.append((script_path, args or [], env))
        return script_path

    monkeypatch.setattr(data_build, "run_script", fake_run_script)

    data_build.validate_and_maybe_upload_datasets(
        upload=True,
        skip_enhanced_cps=False,
        env={"TEST_ENV": "1"},
        stage_only=True,
        run_id="abc123",
        version="1.73.0",
    )

    assert calls == [
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            ["--validate-only"],
            {"TEST_ENV": "1"},
        ),
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            ["--stage-only", "--run-id=abc123", "--version=1.73.0"],
            {"TEST_ENV": "1"},
        ),
    ]


def test_validate_and_maybe_upload_datasets_can_skip_small_enhanced_cps(
    monkeypatch,
):
    data_build = _load_data_build_module()
    calls = []

    def fake_run_script(script_path, args=None, env=None, log_file=None):
        calls.append((script_path, args or [], env))
        return script_path

    monkeypatch.setattr(data_build, "run_script", fake_run_script)

    data_build.validate_and_maybe_upload_datasets(
        upload=True,
        skip_enhanced_cps=False,
        require_small_enhanced_cps=False,
        env={"TEST_ENV": "1"},
        stage_only=True,
        run_id="ecps-only",
        version="1.73.0",
    )

    assert calls == [
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            ["--validate-only", "--no-require-small-enhanced-cps"],
            {"TEST_ENV": "1"},
        ),
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            [
                "--no-require-small-enhanced-cps",
                "--stage-only",
                "--run-id=ecps-only",
                "--version=1.73.0",
            ],
            {"TEST_ENV": "1"},
        ),
    ]


def test_run_cps_then_puf_phase_uses_sequential_checkpointed_builds(
    monkeypatch,
):
    data_build = _load_data_build_module()
    calls = []
    volume = object()
    log_file = object()
    env = {"TEST_ENV": "1"}

    def fake_executor(*args, **kwargs):
        raise AssertionError("CPS/PUF phase must not use ThreadPoolExecutor")

    def fake_run_script_with_checkpoint(
        script_path,
        output_files,
        branch,
        volume_arg,
        args=None,
        env=None,
        log_file=None,
        checkpoint_stats=None,
        command_results=None,
    ):
        calls.append(
            (
                script_path,
                output_files,
                branch,
                volume_arg,
                args,
                env,
                log_file,
                checkpoint_stats,
                command_results,
            )
        )
        return script_path

    monkeypatch.setattr(data_build, "ThreadPoolExecutor", fake_executor)
    monkeypatch.setattr(
        data_build,
        "run_script_with_checkpoint",
        fake_run_script_with_checkpoint,
    )

    data_build.run_cps_then_puf_phase(
        "fix-754",
        volume,
        env=env,
        log_file=log_file,
    )

    assert calls == [
        (
            data_build.CPS_BUILD_SCRIPT,
            data_build.SCRIPT_OUTPUTS[data_build.CPS_BUILD_SCRIPT],
            "fix-754",
            volume,
            None,
            env,
            log_file,
            None,
            [],
        ),
        (
            data_build.PUF_BUILD_SCRIPT,
            data_build.SCRIPT_OUTPUTS[data_build.PUF_BUILD_SCRIPT],
            "fix-754",
            volume,
            None,
            env,
            log_file,
            None,
            [],
        ),
    ]


def test_write_dataset_build_contract_writes_stage_1_handoff(tmp_path):
    data_build = _load_data_build_module()
    artifacts = {
        "acs_2022.h5": b"acs",
        "irs_puf_2015.h5": b"irs",
        "cps_2024.h5": b"cps",
        "puf_2024.h5": b"puf",
        "extended_cps_2024.h5": b"extended",
        "stratified_extended_cps_2024.h5": b"stratified",
        "source_imputed_stratified_extended_cps_2024.h5": b"source-year",
        "source_imputed_stratified_extended_cps.h5": b"source-alias",
        "policy_data.db": b"sqlite",
        "build_log.txt": b"log",
        "data_build_checkpoint_stats.json": b"{}",
    }
    for filename, payload in artifacts.items():
        (tmp_path / filename).write_bytes(payload)

    contract = data_build.write_dataset_build_contract(
        artifacts_dir=tmp_path,
        run_id="run-123",
        code_sha="abc123",
        checkpoint_stats={
            "expected_outputs": 2,
            "valid_reused_outputs": 0,
            "recomputed_outputs": 2,
            "invalid_outputs": 0,
        },
        started_at="2026-05-08T12:00:00Z",
        completed_at="2026-05-08T12:00:30Z",
        duration_s=30.0,
        upload_requested=False,
        stage_only=True,
        skip_enhanced_cps=True,
        branch="stage-1",
    )

    contract_path = tmp_path / "dataset_build_output.json"
    assert contract_path.exists()
    assert read_contract(contract_path) == contract
    assert contract.stage_id == "1_build_datasets"
    assert contract.parameters["stage_only"] is True


def test_run_script_with_checkpoint_rejects_blocked_reuse_decision(
    tmp_path,
    monkeypatch,
):
    data_build = _load_data_build_module()

    def fail_run_script(*args, **kwargs):
        raise AssertionError("blocked decisions must not run commands")

    monkeypatch.setattr(data_build, "run_script", fail_run_script)
    decision = Stage1ReuseDecision(
        run_id="run-a",
        rerun_id="attempt-2",
        artifact_namespace="run-a",
        substep_id="1b_base_dataset_construction",
        identity_key=(
            "1b_base_dataset_construction:"
            "script:policyengine_us_data/datasets/cps/cps.py:cps_2024.h5"
        ),
        action="blocked",
        reason="identity_mismatch",
        identity_fingerprint="sha256:abc",
    )

    try:
        data_build.run_script_with_checkpoint(
            "policyengine_us_data/datasets/cps/cps.py",
            "cps_2024.h5",
            "branch",
            object(),
            checkpoint_store=CheckpointStore(
                root=tmp_path,
                branch="branch",
                commit_sha="abc123",
            ),
            reuse_decision=decision,
        )
    except RuntimeError as exc:
        assert "identity_mismatch" in str(exc)
    else:
        raise AssertionError("blocked reuse decision should fail")


def test_command_group_with_checkpoint_recomputes_without_manifest(
    tmp_path,
    monkeypatch,
):
    data_build = _load_data_build_module()
    output_file = "policyengine_us_data/storage/calibration/policy_data.db"
    store = CheckpointStore(
        root=tmp_path / "checkpoints", branch="branch", commit_sha="abc"
    )
    checkpoint = store.checkpoint_path(output_file)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"old-db")
    stats = data_build.CheckpointStats()
    identity_material = data_build._raw_data_identity_material(branch="branch")

    def build_raw_data():
        Path(output_file).parent.mkdir(parents=True)
        Path(output_file).write_bytes(b"new-db")

    monkeypatch.chdir(tmp_path)

    data_build.run_command_group_with_checkpoint(
        substep_id="1a_raw_data_download",
        output_files=(output_file,),
        branch="branch",
        volume=object(),
        action=build_raw_data,
        checkpoint_stats=stats,
        checkpoint_store=store,
        identity_material=identity_material,
    )

    assert (tmp_path / output_file).read_bytes() == b"new-db"
    assert stats.snapshot() == {
        "expected_outputs": 1,
        "valid_reused_outputs": 0,
        "recomputed_outputs": 1,
        "invalid_outputs": 0,
    }
    assert store.load_reuse_manifest().previous_identities() == {
        identity_material.identity_key: identity_material.fingerprint()
    }


def test_command_group_with_checkpoint_restores_manifest_matched_outputs(
    tmp_path,
    monkeypatch,
):
    data_build = _load_data_build_module()
    output_file = "policyengine_us_data/storage/calibration/policy_data.db"
    store = CheckpointStore(
        root=tmp_path / "checkpoints", branch="branch", commit_sha="abc"
    )
    checkpoint = store.checkpoint_path(output_file)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"db")
    stats = data_build.CheckpointStats()
    identity_material = data_build._raw_data_identity_material(branch="branch")
    store.record_reuse_manifest(
        Stage1ReuseManifestRecord(
            substep_id=identity_material.substep_id,
            identity_key=identity_material.identity_key,
            identity_fingerprint=identity_material.fingerprint(),
            identity_material=identity_material.to_dict(),
        )
    )

    def fail_action():
        raise AssertionError("reused command groups must not run commands")

    monkeypatch.chdir(tmp_path)

    data_build.run_command_group_with_checkpoint(
        substep_id="1a_raw_data_download",
        output_files=(output_file,),
        branch="branch",
        volume=object(),
        action=fail_action,
        checkpoint_stats=stats,
        checkpoint_store=store,
        identity_material=identity_material,
    )

    assert (tmp_path / output_file).read_bytes() == b"db"
    assert stats.snapshot() == {
        "expected_outputs": 1,
        "valid_reused_outputs": 1,
        "recomputed_outputs": 0,
        "invalid_outputs": 0,
    }


def test_command_group_with_checkpoint_recomputes_missing_physical_checkpoint(
    tmp_path,
    monkeypatch,
):
    data_build = _load_data_build_module()
    output_file = "policyengine_us_data/storage/calibration/policy_data.db"
    store = CheckpointStore(
        root=tmp_path / "checkpoints", branch="branch", commit_sha="abc"
    )
    stats = data_build.CheckpointStats()
    identity_material = data_build._raw_data_identity_material(branch="branch")
    store.record_reuse_manifest(
        Stage1ReuseManifestRecord(
            substep_id=identity_material.substep_id,
            identity_key=identity_material.identity_key,
            identity_fingerprint=identity_material.fingerprint(),
            identity_material=identity_material.to_dict(),
        )
    )

    def build_raw_data():
        Path(output_file).parent.mkdir(parents=True)
        Path(output_file).write_bytes(b"new-db")

    monkeypatch.chdir(tmp_path)

    data_build.run_command_group_with_checkpoint(
        substep_id="1a_raw_data_download",
        output_files=(output_file,),
        branch="branch",
        volume=object(),
        action=build_raw_data,
        checkpoint_stats=stats,
        checkpoint_store=store,
        identity_material=identity_material,
    )

    assert (tmp_path / output_file).read_bytes() == b"new-db"
    assert stats.snapshot() == {
        "expected_outputs": 1,
        "valid_reused_outputs": 0,
        "recomputed_outputs": 1,
        "invalid_outputs": 1,
    }


def test_stage_1_status_metadata_publishes_reuse_reasoning():
    data_build = _load_data_build_module()

    class Record:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return self.payload

    coordinator = SimpleNamespace(
        results=[
            Record(
                {
                    "substep_id": "1a_raw_data_download",
                    "title": "Raw Data Download",
                    "status": "reused",
                    "reuse_decision": {
                        "action": "reuse",
                        "reason": "identity_match",
                    },
                    "checkpoint_decisions": [
                        {
                            "action": "reuse",
                            "reason": "valid",
                        }
                    ],
                }
            ),
            Record(
                {
                    "substep_id": "1b_base_dataset_construction",
                    "title": "Base Dataset Construction",
                    "status": "completed",
                    "reuse_decision": {
                        "action": "recompute",
                        "reason": "no_previous_identity",
                    },
                    "checkpoint_decisions": [],
                }
            ),
            Record(
                {
                    "substep_id": "1c_extended_cps_puf_clone",
                    "title": "Extended CPS And PUF Clone",
                    "status": "completed",
                    "reuse_decision": {
                        "action": "reuse",
                        "reason": "identity_match",
                    },
                    "checkpoint_decisions": [
                        {
                            "action": "recompute",
                            "reason": "missing",
                        }
                    ],
                }
            ),
            Record(
                {
                    "substep_id": "1d_enhanced_cps_reweighting",
                    "title": "Enhanced CPS Reweighting",
                    "status": "completed",
                    "reuse_decision": {
                        "action": "recompute",
                        "reason": "identity_mismatch",
                    },
                    "checkpoint_decisions": [],
                }
            ),
            Record(
                {
                    "substep_id": "1e_stratified_cps",
                    "title": "Stratified CPS",
                    "status": "completed",
                    "reuse_decision": {
                        "substep_id": "1e_stratified_cps",
                        "decisions": [
                            {
                                "substep_id": "1e_stratified_cps",
                                "identity_key": (
                                    "1e_stratified_cps:enhanced:enhanced_cps_2024.h5"
                                ),
                                "action": "reuse",
                                "reason": "identity_match",
                                "identity_fingerprint": "sha256:enhanced",
                            },
                            {
                                "substep_id": "1e_stratified_cps",
                                "identity_key": (
                                    "1e_stratified_cps:stratified:stratified_cps_2024.h5"
                                ),
                                "action": "recompute",
                                "reason": "no_previous_identity",
                                "identity_fingerprint": "sha256:stratified",
                            },
                        ],
                    },
                    "checkpoint_decisions": [],
                }
            ),
        ],
        status_events=[],
        error_records=[],
    )

    metadata = data_build._stage_1_status_metadata(coordinator)
    reasons = {
        item["substep_id"]: item["outcome_reason"]
        for item in metadata["reuse_reasoning"]
    }

    assert reasons["1a_raw_data_download"] == (
        "Prior semantic identity matched and every expected checkpoint output "
        "was present and non-empty."
    )
    assert reasons["1b_base_dataset_construction"] == (
        "Recomputed because no prior semantic identity manifest record existed "
        "for this Stage 1 execution unit."
    )
    assert reasons["1c_extended_cps_puf_clone"] == (
        "Recomputed because at least one expected checkpoint output was missing "
        "or empty."
    )
    assert reasons["1d_enhanced_cps_reweighting"] == (
        "Recomputed because persisted semantic identity did not match the "
        "current Stage 1 execution-unit identity."
    )
    mixed = next(
        item
        for item in metadata["reuse_reasoning"]
        if item["substep_id"] == "1e_stratified_cps"
    )
    assert mixed["outcome_reason"] == (
        "Recomputed with per-identity semantic reuse reasons: "
        "identity_match=1, no_previous_identity=1."
    )
    assert mixed["identity_decisions"] == [
        {
            "substep_id": "1e_stratified_cps",
            "identity_key": "1e_stratified_cps:enhanced:enhanced_cps_2024.h5",
            "action": "reuse",
            "reason": "identity_match",
            "identity_fingerprint": "sha256:enhanced",
        },
        {
            "substep_id": "1e_stratified_cps",
            "identity_key": "1e_stratified_cps:stratified:stratified_cps_2024.h5",
            "action": "recompute",
            "reason": "no_previous_identity",
            "identity_fingerprint": "sha256:stratified",
        },
    ]


def test_utc_timestamp_renders_zulu_time_for_build_log():
    data_build = _load_data_build_module()
    budapest_summer = timezone(timedelta(hours=2))

    rendered = data_build._utc_timestamp(
        datetime(2026, 7, 1, 12, 30, 45, tzinfo=budapest_summer)
    )

    assert rendered == "2026-07-01T10:30:45Z"
