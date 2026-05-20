import importlib
import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace

from policyengine_us_data.build_datasets import stage_1_script_outputs
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
    )

    contract_path = tmp_path / "dataset_build_output.json"
    assert contract_path.exists()
    assert read_contract(contract_path) == contract
    assert contract.stage_id == "1_build_datasets"
    assert contract.parameters["stage_only"] is True


def test_utc_timestamp_renders_zulu_time_for_build_log():
    data_build = _load_data_build_module()
    budapest_summer = timezone(timedelta(hours=2))

    rendered = data_build._utc_timestamp(
        datetime(2026, 7, 1, 12, 30, 45, tzinfo=budapest_summer)
    )

    assert rendered == "2026-07-01T10:30:45Z"
