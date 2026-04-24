from policyengine_us_data.calibration.local_h5.fingerprinting import (
    FingerprintingService,
)

from tests.integration.local_h5.fixtures import SEED, VERSION, seed_local_h5_artifacts
from tests.unit.fixtures.test_modal_local_area import load_local_area_module


def test_local_area_helpers_match_publish_traceability_contract(tmp_path):
    local_area = load_local_area_module(stub_policyengine=False)
    artifacts = seed_local_h5_artifacts(tmp_path)

    inputs = local_area._build_publishing_input_bundle(
        weights_path=artifacts.weights_path,
        dataset_path=artifacts.dataset_path,
        db_path=artifacts.db_path,
        geography_path=artifacts.geography_path,
        calibration_package_path=artifacts.calibration_package_path,
        run_config_path=artifacts.run_config_path,
        run_id="run-123",
        version=VERSION,
        n_clones=artifacts.n_clones,
        seed=SEED,
    )

    helper_fingerprint = local_area._resolve_scope_fingerprint(
        inputs=inputs,
        scope="regional",
    )
    service = FingerprintingService()
    service_fingerprint = service.compute_scope_fingerprint(
        service.build_traceability(inputs=inputs, scope="regional")
    )

    assert helper_fingerprint == service_fingerprint


def test_local_area_scope_helper_distinguishes_regional_and_national(tmp_path):
    local_area = load_local_area_module(stub_policyengine=False)
    artifacts = seed_local_h5_artifacts(tmp_path)

    inputs = local_area._build_publishing_input_bundle(
        weights_path=artifacts.weights_path,
        dataset_path=artifacts.dataset_path,
        db_path=artifacts.db_path,
        geography_path=artifacts.geography_path,
        calibration_package_path=artifacts.calibration_package_path,
        run_config_path=artifacts.run_config_path,
        run_id="run-123",
        version=VERSION,
        n_clones=artifacts.n_clones,
        seed=SEED,
    )

    regional = local_area._resolve_scope_fingerprint(
        inputs=inputs,
        scope="regional",
    )
    national = local_area._resolve_scope_fingerprint(
        inputs=inputs,
        scope="national",
    )

    assert regional != national
