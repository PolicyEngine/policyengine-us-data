from pathlib import Path
from types import SimpleNamespace

from tests.support.modal_local_area import load_local_area_module


def test_build_promote_national_publish_script_imports_version_manifest_helpers():
    local_area = load_local_area_module()

    script = local_area._build_promote_national_publish_script(
        version="1.73.0",
        run_id="1.73.0_deadbeef_20260411",
        rel_paths=["national/US.h5"],
    )

    assert "from policyengine_us_data.utils.version_manifest import (" in script
    assert "HFVersionInfo" in script
    assert "build_manifest" in script
    assert "upload_manifest" in script
    assert "preflight_release_manifest_publish" in script
    assert "should_finalize_local_area_release" not in script
    assert script.index(
        "should_finalize, missing_prefixes = preflight_release_manifest_publish("
    ) < script.index("promoted = promote_staging_to_production_hf(")
    assert 'os.environ["US_DATA_RUN_ID"] = run_id' in script
    assert 'os.environ["RUN_ID"]' not in script


def test_build_promote_publish_script_finalizes_complete_release():
    local_area = load_local_area_module()

    script = local_area._build_promote_publish_script(
        version="1.73.0",
        run_id="1.73.0_deadbeef_20260411",
        rel_paths=["states/AL.h5", "districts/AL-01.h5", "cities/NYC.h5"],
    )

    assert "preflight_release_manifest_publish" in script
    assert "should_finalize_local_area_release" not in script
    assert script.index(
        "should_finalize, missing_prefixes = preflight_release_manifest_publish("
    ) < script.index("promoted = promote_staging_to_production_hf(")
    assert "create_tag=should_finalize" in script
    assert "upload_manifest(" in script
    assert 'os.environ["US_DATA_RUN_ID"] = run_id' in script
    assert 'os.environ["RUN_ID"]' not in script


def test_promote_scripts_can_defer_staging_cleanup_for_pipeline_promotion():
    local_area = load_local_area_module()

    regional_script = local_area._build_promote_publish_script(
        version="1.73.0",
        run_id="usdata-gha123-a1-abcdef12",
        rel_paths=["states/AL.h5"],
        cleanup_staging=False,
    )
    national_script = local_area._build_promote_national_publish_script(
        version="1.73.0",
        run_id="usdata-gha123-a1-abcdef12",
        rel_paths=["national/US.h5"],
        cleanup_staging=False,
    )

    assert "cleanup_staging = json.loads('''false''')" in regional_script
    assert "Deferring staged regional cleanup" in regional_script
    assert "cleanup_staging = json.loads('''false''')" in national_script
    assert "Deferring staged national cleanup" in national_script


def test_promote_publish_falls_back_to_package_version_for_new_run_ids(
    monkeypatch, tmp_path
):
    local_area = load_local_area_module()
    run_id = "usdata-gha123-a1-abcdef12"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        '{"files": {"states/NC.h5": {"sha256": "abc"}}}'
    )
    captured = {}

    monkeypatch.setattr(local_area, "VOLUME_MOUNT", str(tmp_path))
    monkeypatch.setattr(local_area, "setup_gcp_credentials", lambda: None)
    monkeypatch.setattr(local_area, "setup_repo", lambda branch: None)
    monkeypatch.setattr(local_area, "get_version", lambda: "1.92.0")
    monkeypatch.setattr(
        local_area,
        "staging_volume",
        SimpleNamespace(reload=lambda: None),
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(local_area.subprocess, "run", fake_run)

    local_area.promote_publish(branch="main", run_id=run_id)

    script = captured["cmd"][-1]
    assert 'version = "1.92.0"' in script
    assert f'version = "{run_id}"' not in script


def test_promote_national_publish_falls_back_to_package_version_for_new_run_ids(
    monkeypatch,
):
    local_area = load_local_area_module()
    run_id = "usdata-gha123-a1-abcdef12"
    captured = {}

    monkeypatch.setattr(local_area, "setup_gcp_credentials", lambda: None)
    monkeypatch.setattr(local_area, "setup_repo", lambda branch: None)
    monkeypatch.setattr(local_area, "get_version", lambda: "1.92.0")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(local_area.subprocess, "run", fake_run)

    local_area.promote_national_publish(branch="main", run_id=run_id)

    script = captured["cmd"][-1]
    assert 'version = "1.92.0"' in script
    assert f'version = "{run_id}"' not in script


def test_build_publishing_input_bundle_preserves_traceability_inputs():
    local_area = load_local_area_module(stub_policyengine=False)

    bundle = local_area._build_publishing_input_bundle(
        weights_path=Path("/tmp/calibration_weights.npy"),
        dataset_path=Path("/tmp/source.h5"),
        db_path=Path("/tmp/policy_data.db"),
        geography_path=Path("/tmp/geography_assignment.npz"),
        calibration_package_path=Path("/tmp/calibration_package.pkl"),
        run_config_path=Path("/tmp/unified_run_config.json"),
        run_id="run-123",
        version="1.2.3",
        n_clones=4,
        seed=42,
        legacy_blocks_path=Path("/tmp/stacked_blocks.npy"),
    )

    assert bundle.weights_path == Path("/tmp/calibration_weights.npy")
    assert bundle.source_dataset_path == Path("/tmp/source.h5")
    assert bundle.target_db_path == Path("/tmp/policy_data.db")
    assert bundle.exact_geography_path == Path("/tmp/geography_assignment.npz")
    assert bundle.calibration_package_path == Path("/tmp/calibration_package.pkl")
    assert bundle.run_config_path == Path("/tmp/unified_run_config.json")
    assert bundle.run_id == "run-123"
    assert bundle.version == "1.2.3"
    assert bundle.n_clones == 4
    assert bundle.seed == 42
    assert bundle.legacy_blocks_path == Path("/tmp/stacked_blocks.npy")


def test_resolve_scope_fingerprint_computes_when_no_pin(monkeypatch):
    local_area = load_local_area_module(stub_policyengine=False)

    seen = {}

    class FakeFingerprintingService:
        def build_traceability(self, *, inputs, scope):
            seen["inputs"] = inputs
            seen["scope"] = scope
            return {"scope": scope, "run_id": inputs.run_id}

        def compute_scope_fingerprint(self, traceability):
            seen["traceability"] = traceability
            return "computed-fingerprint"

    monkeypatch.setattr(
        local_area,
        "FingerprintingService",
        FakeFingerprintingService,
    )

    bundle = local_area._build_publishing_input_bundle(
        weights_path=Path("/tmp/calibration_weights.npy"),
        dataset_path=Path("/tmp/source.h5"),
        db_path=None,
        geography_path=None,
        calibration_package_path=None,
        run_config_path=None,
        run_id="run-123",
        version="1.2.3",
        n_clones=2,
        seed=42,
    )

    fingerprint = local_area._resolve_scope_fingerprint(
        inputs=bundle,
        scope="regional",
    )

    assert fingerprint == "computed-fingerprint"
    assert seen["inputs"] == bundle
    assert seen["scope"] == "regional"
    assert seen["traceability"] == {"scope": "regional", "run_id": "run-123"}


def test_resolve_scope_fingerprint_preserves_matching_pin(monkeypatch, capsys):
    local_area = load_local_area_module(stub_policyengine=False)

    class FakeFingerprintingService:
        def build_traceability(self, *, inputs, scope):
            return scope

        def compute_scope_fingerprint(self, traceability):
            return "pinned-fingerprint"

    monkeypatch.setattr(
        local_area,
        "FingerprintingService",
        FakeFingerprintingService,
    )

    bundle = local_area._build_publishing_input_bundle(
        weights_path=Path("/tmp/calibration_weights.npy"),
        dataset_path=Path("/tmp/source.h5"),
        db_path=None,
        geography_path=None,
        calibration_package_path=None,
        run_config_path=None,
        run_id="run-123",
        version="1.2.3",
        n_clones=2,
        seed=42,
    )

    fingerprint = local_area._resolve_scope_fingerprint(
        inputs=bundle,
        scope="regional",
        expected_fingerprint="pinned-fingerprint",
    )

    captured = capsys.readouterr()
    assert fingerprint == "pinned-fingerprint"
    assert "Using pinned fingerprint from pipeline" in captured.out


def test_resolve_scope_fingerprint_warns_and_preserves_mismatched_pin(
    monkeypatch, capsys
):
    local_area = load_local_area_module(stub_policyengine=False)

    class FakeFingerprintingService:
        def build_traceability(self, *, inputs, scope):
            return scope

        def compute_scope_fingerprint(self, traceability):
            return "computed-fingerprint"

    monkeypatch.setattr(
        local_area,
        "FingerprintingService",
        FakeFingerprintingService,
    )

    bundle = local_area._build_publishing_input_bundle(
        weights_path=Path("/tmp/calibration_weights.npy"),
        dataset_path=Path("/tmp/source.h5"),
        db_path=None,
        geography_path=None,
        calibration_package_path=None,
        run_config_path=None,
        run_id="run-123",
        version="1.2.3",
        n_clones=2,
        seed=42,
    )

    fingerprint = local_area._resolve_scope_fingerprint(
        inputs=bundle,
        scope="national",
        expected_fingerprint="legacy-fingerprint",
    )

    captured = capsys.readouterr()
    assert fingerprint == "legacy-fingerprint"
    assert "Pinned fingerprint differs from current national scope fingerprint" in (
        captured.out
    )
    assert "legacy-fingerprint" in captured.out
    assert "computed-fingerprint" in captured.out


def test_build_worker_bootstrap_invokes_builder_without_changing_inputs(monkeypatch):
    local_area = load_local_area_module(stub_policyengine=False)
    publishing_inputs = object()
    artifacts_dir = Path("/pipeline/artifacts/run-123")
    captured = {}

    class FakeWorkerBootstrapBuilder:
        def build(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                manifest_path=(
                    artifacts_dir
                    / "bootstrap"
                    / kwargs["scope"]
                    / "worker_bootstrap.json"
                )
            )

    monkeypatch.setattr(
        local_area,
        "WorkerBootstrapBuilder",
        FakeWorkerBootstrapBuilder,
    )

    bundle = local_area._build_worker_bootstrap(
        inputs=publishing_inputs,
        scope="regional",
        artifacts_dir=artifacts_dir,
        scope_fingerprint="resolved-fingerprint",
    )

    assert captured == {
        "inputs": publishing_inputs,
        "scope": "regional",
        "artifacts_dir": artifacts_dir,
        "scope_fingerprint": "resolved-fingerprint",
    }
    assert bundle.manifest_path == (
        artifacts_dir / "bootstrap" / "regional" / "worker_bootstrap.json"
    )


def test_build_worker_calibration_inputs_includes_existing_run_config_and_package(
    tmp_path,
):
    local_area = load_local_area_module()
    run_config_path = tmp_path / "unified_run_config.json"
    package_path = tmp_path / "calibration_package.pkl"
    run_config_path.write_text("{}")
    package_path.write_bytes(b"package")

    inputs = local_area._build_worker_calibration_inputs(
        weights_path=tmp_path / "calibration_weights.npy",
        geography_path=tmp_path / "geography_assignment.npz",
        dataset_path=tmp_path / "source.h5",
        db_path=tmp_path / "policy_data.db",
        n_clones=430,
        seed=42,
        run_config_path=run_config_path,
        calibration_package_path=package_path,
    )

    assert inputs.run_config_path == run_config_path
    assert inputs.calibration_package_path == package_path
    assert inputs.n_clones == 430
    assert inputs.seed == 42
    assert inputs.to_wire_dict()["run_config"] == str(run_config_path)


def test_build_worker_calibration_inputs_omits_missing_optional_files(tmp_path):
    local_area = load_local_area_module()

    inputs = local_area._build_worker_calibration_inputs(
        weights_path=tmp_path / "national_calibration_weights.npy",
        geography_path=tmp_path / "national_geography_assignment.npz",
        dataset_path=tmp_path / "source.h5",
        db_path=tmp_path / "policy_data.db",
        n_clones=430,
        seed=42,
        run_config_path=tmp_path / "missing_config.json",
        calibration_package_path=tmp_path / "missing_package.pkl",
    )

    assert inputs.run_config_path is None
    assert inputs.calibration_package_path is None
    assert "run_config" not in inputs.to_wire_dict()
    assert "calibration_package" not in inputs.to_wire_dict()


def test_build_areas_worker_surfaces_successful_worker_stderr(
    monkeypatch,
    capsys,
    tmp_path,
):
    local_area = load_local_area_module()
    monkeypatch.setattr(local_area, "setup_gcp_credentials", lambda: None)
    monkeypatch.setattr(local_area, "setup_repo", lambda branch: None)
    monkeypatch.setattr(local_area, "VOLUME_MOUNT", str(tmp_path / "staging"))
    monkeypatch.setattr(
        local_area,
        "pipeline_volume",
        SimpleNamespace(reload=lambda: None),
    )
    monkeypatch.setattr(
        local_area,
        "staging_volume",
        SimpleNamespace(reload=lambda: None, commit=lambda: None),
    )

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout='{"completed": ["district:NC-01"], "failed": [], "errors": []}',
            stderr="Worker session ready: scope=regional, bootstrap=used\n",
        )

    monkeypatch.setattr(local_area.subprocess, "run", fake_run)

    result = local_area.build_areas_worker(
        branch="main",
        run_id="run-123",
        scope="regional",
        work_items=[{"type": "district", "id": "NC-01"}],
        calibration_inputs={
            "weights": "/tmp/calibration_weights.npy",
            "dataset": "/tmp/source.h5",
            "database": "/tmp/policy_data.db",
        },
        validate=False,
    )

    captured = capsys.readouterr()
    assert result["completed"] == ["district:NC-01"]
    assert "Worker session ready: scope=regional, bootstrap=used" in captured.err
