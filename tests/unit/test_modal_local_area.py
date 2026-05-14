from pathlib import Path
from types import ModuleType, SimpleNamespace

from tests.support.build_outputs.area_catalog import make_geography
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
        run_id="usdata-gha123-a1",
        rel_paths=["states/AL.h5"],
        cleanup_staging=False,
    )
    national_script = local_area._build_promote_national_publish_script(
        version="1.73.0",
        run_id="usdata-gha123-a1",
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
    run_id = "usdata-gha123-a1"
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
    run_id = "usdata-gha123-a1"
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


def test_staging_candidate_version_prefers_pipeline_candidate(monkeypatch):
    local_area = load_local_area_module()

    monkeypatch.setenv("US_DATA_CANDIDATE_VERSION", "1.115.2-patch")

    assert local_area.get_staging_candidate_version("1.115.2") == "1.115.2-patch"


def test_upload_to_staging_uses_candidate_scope(monkeypatch):
    local_area = load_local_area_module()
    captured = {}

    monkeypatch.setattr(local_area, "setup_repo", lambda branch: None)

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(local_area.subprocess, "run", fake_run)

    result = local_area.upload_to_staging(
        branch="main",
        version="1.115.2",
        manifest={"files": {"states/NC.h5": {"sha256": "abc"}}},
        run_id="usdata-gha123-a1",
        candidate_version="1.115.2-patch",
    )

    script = captured["cmd"][-1]
    assert 'version = "1.115.2"' in script
    assert 'staging_candidate_version = "1.115.2-patch"' in script
    assert "candidate_version=staging_candidate_version" in script
    assert result.startswith("Staged candidate 1.115.2-patch")


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


def test_load_area_catalog_geography_uses_mmap_for_weight_shape(
    monkeypatch,
    tmp_path,
):
    local_area = load_local_area_module()
    weights_path = tmp_path / "calibration_weights.npy"
    geography_path = tmp_path / "geography_assignment.npz"
    geography_path.write_text("exists")
    captured = {}

    class FakeWeights:
        ndim = 1
        size = 6
        dtype = local_area.np.dtype("float64")

    def fake_load(path, *, mmap_mode=None):
        captured["load"] = {
            "path": path,
            "mmap_mode": mmap_mode,
        }
        return FakeWeights()

    class FakeLoader:
        def load(self, **kwargs):
            captured["loader"] = kwargs
            return "loaded-geography"

    monkeypatch.setattr(local_area.np, "load", fake_load)
    monkeypatch.setattr(
        local_area,
        "CalibrationGeographyLoader",
        lambda: FakeLoader(),
    )

    result = local_area._load_area_catalog_geography(
        weights_path=weights_path,
        n_clones=3,
        geography_path=geography_path,
    )

    assert result == "loaded-geography"
    assert captured["load"] == {
        "path": weights_path,
        "mmap_mode": "r",
    }
    assert captured["loader"]["n_records"] == 2
    assert captured["loader"]["n_clones"] == 3
    assert captured["loader"]["geography_path"] == geography_path


def test_build_regional_weighted_requests_uses_catalog_geography():
    local_area = load_local_area_module(stub_policyengine=False)
    catalog = local_area.USAreaCatalog(
        state_codes={1: "AL", 36: "NY"},
        nyc_county_fips={"36061"},
        at_large_districts={0, 98},
    )
    geography = make_geography(
        cd_geoids=["101", "102", "3601"],
        county_fips=["01001", "01003", "36061"],
    )

    weighted = local_area._build_regional_weighted_requests(
        geography=geography,
        target_cd_geoids=("101", "102", "3601"),
        catalog=catalog,
    )

    assert [item.key for item in weighted] == [
        "state:AL",
        "state:NY",
        "district:AL-01",
        "district:AL-02",
        "district:NY-01",
        "city:NYC",
    ]
    assert [item.weight for item in weighted] == [2, 1, 1, 1, 1, 11]


def test_load_target_cd_geoids_uses_database_target_adapter(monkeypatch):
    local_area = load_local_area_module()
    captured = {}

    def fake_get_all_cds_from_database(db_uri):
        captured["db_uri"] = db_uri
        return [101, "102"]

    fake_utils = ModuleType("policyengine_us_data.calibration.calibration_utils")
    fake_utils.get_all_cds_from_database = fake_get_all_cds_from_database
    monkeypatch.setitem(
        local_area.sys.modules,
        "policyengine_us_data.calibration.calibration_utils",
        fake_utils,
    )

    result = local_area._load_target_cd_geoids(Path("/tmp/policy_data.db"))

    assert result == ("101", "102")
    assert captured["db_uri"] == "sqlite:////tmp/policy_data.db"


def test_build_weighted_requests_from_work_items_keeps_override_weights():
    local_area = load_local_area_module(stub_policyengine=False)
    catalog = local_area.USAreaCatalog(
        state_codes={1: "AL", 36: "NY"},
        nyc_county_fips={"36061"},
        at_large_districts={0, 98},
    )
    geography = make_geography(
        cd_geoids=["101", "3601"],
        county_fips=["01001", "36061"],
    )

    weighted = local_area._build_weighted_requests_from_work_items(
        work_items=(
            {"type": "district", "id": "AL-01", "weight": 7},
            {"type": "city", "id": "NYC", "weight": 5},
        ),
        geography=geography,
        catalog=catalog,
    )

    assert [item.key for item in weighted] == ["district:AL-01", "city:NYC"]
    assert [item.weight for item in weighted] == [7, 5]


def test_measure_expected_completion_ignores_unexpected_stale_files():
    local_area = load_local_area_module()

    missing, measurement = local_area._measure_expected_completion(
        expected_keys={"state:AL", "district:AL-01"},
        initially_completed={"state:AL", "district:OLD"},
        completed={"state:AL", "district:OLD"},
    )

    assert missing == {"district:AL-01"}
    assert measurement == {
        "expected_outputs": 2,
        "valid_reused_outputs": 1,
        "recomputed_outputs": 0,
        "invalid_outputs": 1,
    }


def test_coordinate_publish_happy_path_with_fake_volumes_and_artifacts(
    monkeypatch,
    tmp_path,
):
    local_area = load_local_area_module()
    run_id = "run-123"
    pipeline_root = tmp_path / "pipeline"
    artifact_dir = pipeline_root / "artifacts" / run_id
    artifact_dir.mkdir(parents=True)
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    for filename in (
        "calibration_weights.npy",
        "source_imputed_stratified_extended_cps.h5",
        "policy_data.db",
        "unified_run_config.json",
    ):
        (artifact_dir / filename).write_text("artifact")

    real_path = Path

    def remapped_path(value=".", *args):
        text = str(value)
        if text.startswith("/pipeline"):
            return real_path(str(pipeline_root) + text[len("/pipeline") :])
        return real_path(value, *args)

    monkeypatch.setattr(local_area, "Path", remapped_path)
    monkeypatch.setattr(local_area, "VOLUME_MOUNT", str(staging_root))
    monkeypatch.setattr(local_area, "setup_gcp_credentials", lambda: None)
    monkeypatch.setattr(local_area, "setup_repo", lambda branch: None)
    monkeypatch.setattr(local_area, "get_version", lambda: "0.0.0")
    monkeypatch.setattr(local_area, "validate_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        local_area, "_load_area_catalog_geography", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        local_area, "_build_publishing_input_bundle", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        local_area, "_resolve_scope_fingerprint", lambda **kwargs: "fingerprint"
    )
    monkeypatch.setattr(
        local_area, "reconcile_run_dir_fingerprint", lambda *args, **kwargs: "fresh"
    )
    monkeypatch.setattr(
        local_area, "_build_worker_bootstrap", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        local_area,
        "pipeline_volume",
        SimpleNamespace(reload=lambda: None, commit=lambda: None),
    )
    monkeypatch.setattr(
        local_area,
        "staging_volume",
        SimpleNamespace(reload=lambda: None, commit=lambda: None),
    )
    requests = (
        local_area.WeightedAreaRequest(
            request=SimpleNamespace(
                area_type="state",
                area_id="NC",
                to_dict=lambda: {"area_type": "state", "area_id": "NC"},
            ),
            weight=1,
        ),
        local_area.WeightedAreaRequest(
            request=SimpleNamespace(
                area_type="district",
                area_id="NC-01",
                to_dict=lambda: {"area_type": "district", "area_id": "NC-01"},
            ),
            weight=1,
        ),
    )
    monkeypatch.setattr(
        local_area,
        "_build_regional_weighted_requests",
        lambda **kwargs: requests,
    )
    monkeypatch.setattr(
        local_area,
        "_load_target_cd_geoids",
        lambda db_path: ("3701",),
    )
    captured = {}

    def fake_run_phase(phase_name, *, weighted_requests, completed, **kwargs):
        captured["phase_name"] = phase_name
        captured["weighted_keys"] = [item.key for item in weighted_requests]
        captured["completed_before"] = set(completed)
        return {"state:NC", "district:NC-01"}, [], [{"variable": "household_count"}]

    monkeypatch.setattr(local_area, "run_phase", fake_run_phase)

    result = local_area.coordinate_publish(
        branch="main",
        num_workers=1,
        skip_upload=True,
        n_clones=1,
        validate=False,
        run_id=run_id,
    )

    assert result["message"] == "Build complete for version 0.0.0. Upload skipped."
    assert result["fingerprint"] == "fingerprint"
    assert result["validation_rows"] == [{"variable": "household_count"}]
    assert result["reuse_measurement"] == {
        "expected_outputs": 2,
        "valid_reused_outputs": 0,
        "recomputed_outputs": 2,
        "invalid_outputs": 0,
    }
    assert captured == {
        "phase_name": "All areas",
        "weighted_keys": ["state:NC", "district:NC-01"],
        "completed_before": set(),
    }


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
    captured_cmd = {}

    def fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        return SimpleNamespace(
            returncode=0,
            stdout='{"completed": ["district:NC-01"], "failed": [], "errors": []}',
            stderr="Worker session ready: scope=regional, bootstrap=used\n",
        )

    monkeypatch.setattr(local_area.subprocess, "run", fake_run)

    result = local_area.build_areas_worker(
        "main",
        "run-123",
        "regional",
        [{"type": "district", "id": "NC-01"}],
        {
            "weights": "/tmp/calibration_weights.npy",
            "dataset": "/tmp/source.h5",
            "database": "/tmp/policy_data.db",
        },
        False,
        "regional-fingerprint",
    )

    captured = capsys.readouterr()
    assert result["completed"] == ["district:NC-01"]
    assert "Worker session ready: scope=regional, bootstrap=used" in captured.err
    assert "--work-items" in captured_cmd["cmd"]
    assert "--requests-json" not in captured_cmd["cmd"]
    assert "--scope-fingerprint" in captured_cmd["cmd"]
    assert "regional-fingerprint" in captured_cmd["cmd"]


def test_build_areas_worker_prefers_typed_request_payloads(
    monkeypatch,
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
    captured_cmd = {}

    def fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        return SimpleNamespace(
            returncode=0,
            stdout='{"completed": ["district:NC-01"], "failed": [], "errors": []}',
            stderr="",
        )

    monkeypatch.setattr(local_area.subprocess, "run", fake_run)

    result = local_area.build_areas_worker(
        branch="main",
        run_id="run-123",
        scope="regional",
        request_payloads=[
            {
                "area_type": "district",
                "area_id": "NC-01",
                "display_name": "NC-01",
                "output_relative_path": "districts/NC-01.h5",
            }
        ],
        calibration_inputs={
            "weights": "/tmp/calibration_weights.npy",
            "dataset": "/tmp/source.h5",
            "database": "/tmp/policy_data.db",
        },
        validate=False,
    )

    assert result["completed"] == ["district:NC-01"]
    assert "--requests-json" in captured_cmd["cmd"]
    assert "--work-items" not in captured_cmd["cmd"]


def test_run_phase_collects_worker_validation_errors(monkeypatch, tmp_path):
    local_area = load_local_area_module()
    request = SimpleNamespace(
        area_type="district",
        area_id="NC-01",
        to_dict=lambda: {"area_type": "district", "area_id": "NC-01"},
    )
    weighted = (local_area.WeightedAreaRequest(request=request, weight=1),)
    run_dir = tmp_path / "run-123"
    run_dir.mkdir()
    captured = {}

    class FakeHandle:
        object_id = "fc-validation-error"

        def get(self):
            return {
                "completed": ["district:NC-01"],
                "failed": [],
                "errors": [
                    {
                        "item": "district:NC-01",
                        "phase": "validation",
                        "error": "validation failed",
                    }
                ],
                "issues": [],
                "validation_rows": [],
            }

    def fake_spawn(**kwargs):
        captured.update(kwargs)
        return FakeHandle()

    monkeypatch.setattr(
        local_area,
        "build_areas_worker",
        SimpleNamespace(spawn=fake_spawn),
    )
    monkeypatch.setattr(
        local_area,
        "staging_volume",
        SimpleNamespace(reload=lambda: None),
    )
    monkeypatch.setattr(
        local_area,
        "get_completed_from_volume",
        lambda path: {"district:NC-01"},
    )

    completed, phase_errors, validation_rows = local_area.run_phase(
        "Typed requests",
        weighted_requests=weighted,
        num_workers=1,
        completed=set(),
        branch="main",
        run_id="run-123",
        calibration_inputs={
            "weights": "/tmp/calibration_weights.npy",
            "dataset": "/tmp/source.h5",
            "database": "/tmp/policy_data.db",
        },
        run_dir=run_dir,
        validate=True,
        scope_fingerprint="regional-fingerprint",
    )

    assert captured["request_payloads"] == [request.to_dict()]
    assert captured["work_items"] is None
    assert captured["scope_fingerprint"] == "regional-fingerprint"
    assert completed == {"district:NC-01"}
    assert validation_rows == []
    assert phase_errors == [
        {
            "item": "district:NC-01",
            "phase": "validation",
            "error": "validation failed",
            "worker": 0,
            "severity": "worker_failure",
        }
    ]


def test_run_phase_partitions_typed_requests_and_aggregates_issues(
    monkeypatch,
    tmp_path,
):
    local_area = load_local_area_module()
    request = SimpleNamespace(
        area_type="district",
        area_id="NC-01",
        to_dict=lambda: {"area_type": "district", "area_id": "NC-01"},
    )
    weighted = (local_area.WeightedAreaRequest(request=request, weight=1),)
    run_dir = tmp_path / "run-123"
    (run_dir / "districts").mkdir(parents=True)
    (run_dir / "districts" / "NC-01.h5").write_text("h5")
    captured = {}

    class FakeHandle:
        object_id = "fc-123"

        def get(self):
            return {
                "completed": ["district:NC-01"],
                "failed": [],
                "errors": [],
                "issues": [
                    {
                        "item": "district:NC-01",
                        "phase": "validation",
                        "error": "validation warning",
                    }
                ],
                "validation_rows": [{"variable": "household_count"}],
            }

    def fake_spawn(**kwargs):
        captured.update(kwargs)
        return FakeHandle()

    monkeypatch.setattr(
        local_area,
        "build_areas_worker",
        SimpleNamespace(spawn=fake_spawn),
    )
    monkeypatch.setattr(
        local_area,
        "staging_volume",
        SimpleNamespace(reload=lambda: None),
    )

    completed, errors, validation_rows = local_area.run_phase(
        "Typed requests",
        weighted_requests=weighted,
        num_workers=1,
        completed=set(),
        branch="main",
        run_id="run-123",
        calibration_inputs={
            "weights": "/tmp/calibration_weights.npy",
            "dataset": "/tmp/source.h5",
            "database": "/tmp/policy_data.db",
        },
        run_dir=run_dir,
        validate=True,
        scope_fingerprint="fingerprint",
    )

    assert captured["request_payloads"] == [request.to_dict()]
    assert captured["work_items"] is None
    assert completed == {"district:NC-01"}
    assert errors == [
        {
            "item": "district:NC-01",
            "phase": "validation",
            "error": "validation warning",
        }
    ]
    assert validation_rows == [{"variable": "household_count"}]


def test_run_phase_records_worker_transport_failure_separately(
    monkeypatch,
    tmp_path,
):
    local_area = load_local_area_module()
    request = SimpleNamespace(
        area_type="district",
        area_id="NC-01",
        to_dict=lambda: {"area_type": "district", "area_id": "NC-01"},
    )
    weighted = (local_area.WeightedAreaRequest(request=request, weight=1),)
    run_dir = tmp_path / "run-123"
    run_dir.mkdir()

    class FakeHandle:
        object_id = "fc-123"

        def get(self):
            raise RuntimeError("modal transport reset")

    monkeypatch.setattr(
        local_area,
        "build_areas_worker",
        SimpleNamespace(spawn=lambda **kwargs: FakeHandle()),
    )
    monkeypatch.setattr(
        local_area,
        "staging_volume",
        SimpleNamespace(reload=lambda: None),
    )

    completed, errors, validation_rows = local_area.run_phase(
        "Typed requests",
        weighted_requests=weighted,
        num_workers=1,
        completed=set(),
        branch="main",
        run_id="run-123",
        calibration_inputs={
            "weights": "/tmp/calibration_weights.npy",
            "dataset": "/tmp/source.h5",
            "database": "/tmp/policy_data.db",
        },
        run_dir=run_dir,
        validate=True,
    )

    assert completed == set()
    assert validation_rows == []
    assert errors[0]["worker"] == 0
    assert errors[0]["error"] == "modal transport reset"
