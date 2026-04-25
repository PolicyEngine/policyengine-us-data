from pathlib import Path

from tests.unit.fixtures.test_modal_local_area import load_local_area_module


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


def test_build_promote_publish_script_finalizes_complete_release():
    local_area = load_local_area_module()

    script = local_area._build_promote_publish_script(
        version="1.73.0",
        run_id="1.73.0_deadbeef_20260411",
        rel_paths=["states/AL.h5", "districts/AL-01.h5", "cities/NYC.h5"],
    )

    assert "should_finalize_local_area_release" in script
    assert "create_tag=should_finalize" in script
    assert "upload_manifest(" in script


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
