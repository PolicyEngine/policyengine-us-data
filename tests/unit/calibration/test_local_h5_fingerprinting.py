from tests.unit.calibration.fixtures.test_local_h5_fingerprinting import (
    load_fingerprinting_exports,
    make_publishing_inputs,
)


exports = load_fingerprinting_exports()
FingerprintingService = exports["FingerprintingService"]
PublishingInputBundle = exports["PublishingInputBundle"]


def test_build_traceability_captures_artifact_identity_and_metadata(tmp_path):
    inputs = make_publishing_inputs(PublishingInputBundle, tmp_path=tmp_path)

    service = FingerprintingService()
    traceability = service.build_traceability(inputs=inputs, scope="regional")

    assert traceability.scope == "regional"
    assert traceability.weights.path == inputs.weights_path
    assert traceability.weights.sha256.startswith("sha256:")
    assert traceability.source_dataset.sha256.startswith("sha256:")
    assert traceability.exact_geography is not None
    assert traceability.exact_geography.metadata["source_kind"] == "saved_geography"
    assert traceability.exact_geography.metadata["canonical_sha256"].startswith(
        "sha256:"
    )
    assert traceability.target_db is not None
    assert traceability.model_build["locked_version"] == "1.0.0"
    assert traceability.metadata["n_clones"] == 2
    assert traceability.metadata["seed"] == 42


def test_scope_fingerprint_differs_between_regional_and_national(tmp_path):
    inputs = make_publishing_inputs(PublishingInputBundle, tmp_path=tmp_path)

    service = FingerprintingService()
    regional = service.compute_scope_fingerprint(
        service.build_traceability(inputs=inputs, scope="regional")
    )
    national = service.compute_scope_fingerprint(
        service.build_traceability(inputs=inputs, scope="national")
    )

    assert regional != national


def test_scope_fingerprint_is_stable_for_identical_inputs(tmp_path):
    inputs = make_publishing_inputs(PublishingInputBundle, tmp_path=tmp_path)

    service = FingerprintingService()
    first = service.compute_scope_fingerprint(
        service.build_traceability(inputs=inputs, scope="regional")
    )
    second = service.compute_scope_fingerprint(
        service.build_traceability(inputs=inputs, scope="regional")
    )

    assert first == second


def test_scope_fingerprint_changes_when_relevant_provenance_changes(tmp_path):
    first_inputs = make_publishing_inputs(
        PublishingInputBundle,
        tmp_path=tmp_path / "first",
    )
    second_inputs = make_publishing_inputs(
        PublishingInputBundle,
        tmp_path=tmp_path / "second",
    )
    second_inputs.target_db_path.write_bytes(b"changed-db")

    service = FingerprintingService()
    first = service.compute_scope_fingerprint(
        service.build_traceability(inputs=first_inputs, scope="regional")
    )
    second = service.compute_scope_fingerprint(
        service.build_traceability(inputs=second_inputs, scope="regional")
    )

    assert first != second


def test_traceability_uses_weight_derived_household_count_for_geography(tmp_path):
    inputs = make_publishing_inputs(
        PublishingInputBundle,
        tmp_path=tmp_path,
        n_records=2,
        person_records=5,
        n_clones=2,
    )

    service = FingerprintingService()
    traceability = service.build_traceability(inputs=inputs, scope="regional")

    assert traceability.exact_geography is not None
    assert traceability.exact_geography.metadata["canonical_sha256"].startswith(
        "sha256:"
    )


def test_resumability_material_prefers_canonical_geography_checksum(tmp_path):
    inputs = make_publishing_inputs(PublishingInputBundle, tmp_path=tmp_path)

    service = FingerprintingService()
    traceability = service.build_traceability(inputs=inputs, scope="regional")
    resumability = traceability.resumability_material()

    assert traceability.exact_geography is not None
    assert (
        resumability["exact_geography_sha256"]
        == traceability.exact_geography.metadata["canonical_sha256"]
    )


def test_traceability_handles_missing_optional_artifacts(tmp_path):
    inputs = make_publishing_inputs(PublishingInputBundle, tmp_path=tmp_path)
    standalone_weights_path = tmp_path / "standalone" / "weights.npy"
    standalone_weights_path.parent.mkdir(parents=True, exist_ok=True)
    standalone_weights_path.write_bytes(inputs.weights_path.read_bytes())
    inputs = PublishingInputBundle(
        weights_path=standalone_weights_path,
        source_dataset_path=inputs.source_dataset_path,
        target_db_path=None,
        exact_geography_path=None,
        calibration_package_path=None,
        run_config_path=None,
        run_id=inputs.run_id,
        version=inputs.version,
        n_clones=inputs.n_clones,
        seed=inputs.seed,
        legacy_blocks_path=None,
    )

    service = FingerprintingService()
    traceability = service.build_traceability(inputs=inputs, scope="regional")

    assert traceability.target_db is None
    assert traceability.exact_geography is None
    assert traceability.calibration_package is None
    assert traceability.run_config is None
    assert traceability.code_version == {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }
    assert traceability.model_build == {
        "locked_version": None,
        "git_commit": None,
    }
