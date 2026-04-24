from policyengine_us_data.calibration.local_h5.fingerprinting import (
    FingerprintingService,
    PublishingInputBundle,
)

from tests.integration.local_h5.fixtures import SEED, VERSION, seed_local_h5_artifacts


def _fingerprint_for(*, inputs, scope: str = "regional") -> str:
    service = FingerprintingService()
    return service.compute_scope_fingerprint(
        service.build_traceability(inputs=inputs, scope=scope)
    )


def test_saved_geography_bundle_builds_traceability_with_stable_fingerprint(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path)
    inputs = PublishingInputBundle(
        weights_path=artifacts.weights_path,
        source_dataset_path=artifacts.dataset_path,
        target_db_path=artifacts.db_path,
        exact_geography_path=artifacts.geography_path,
        calibration_package_path=None,
        run_config_path=artifacts.run_config_path,
        run_id="run-123",
        version=VERSION,
        n_clones=artifacts.n_clones,
        seed=SEED,
    )

    first = _fingerprint_for(inputs=inputs)
    second = _fingerprint_for(inputs=inputs)

    assert first == second


def test_package_geography_bundle_builds_traceability_with_stable_fingerprint(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path)
    inputs = PublishingInputBundle(
        weights_path=artifacts.weights_path,
        source_dataset_path=artifacts.dataset_path,
        target_db_path=artifacts.db_path,
        exact_geography_path=None,
        calibration_package_path=artifacts.calibration_package_path,
        run_config_path=artifacts.run_config_path,
        run_id="run-123",
        version=VERSION,
        n_clones=artifacts.n_clones,
        seed=SEED,
    )

    first = _fingerprint_for(inputs=inputs)
    second = _fingerprint_for(inputs=inputs)

    assert first == second


def test_saved_and_package_geography_share_the_same_resumability_identity(tmp_path):
    artifacts = seed_local_h5_artifacts(tmp_path)
    saved_inputs = PublishingInputBundle(
        weights_path=artifacts.weights_path,
        source_dataset_path=artifacts.dataset_path,
        target_db_path=artifacts.db_path,
        exact_geography_path=artifacts.geography_path,
        calibration_package_path=None,
        run_config_path=artifacts.run_config_path,
        run_id="run-123",
        version=VERSION,
        n_clones=artifacts.n_clones,
        seed=SEED,
    )
    package_inputs = PublishingInputBundle(
        weights_path=artifacts.weights_path,
        source_dataset_path=artifacts.dataset_path,
        target_db_path=artifacts.db_path,
        exact_geography_path=None,
        calibration_package_path=artifacts.calibration_package_path,
        run_config_path=artifacts.run_config_path,
        run_id="run-123",
        version=VERSION,
        n_clones=artifacts.n_clones,
        seed=SEED,
    )

    saved_fingerprint = _fingerprint_for(inputs=saved_inputs)
    package_fingerprint = _fingerprint_for(inputs=package_inputs)

    assert saved_fingerprint == package_fingerprint
