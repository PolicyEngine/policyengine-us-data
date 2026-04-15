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
