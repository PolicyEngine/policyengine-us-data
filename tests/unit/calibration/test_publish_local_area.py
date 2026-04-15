import numpy as np

from policyengine_us_data.calibration.publish_local_area import (
    _build_reported_takeup_anchors,
    compute_input_fingerprint,
)


def test_build_reported_takeup_anchors_skips_missing_period():
    data = {
        "person_tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "reported_has_subsidized_marketplace_health_coverage_at_interview": {
            2023: np.array([True, False])
        },
        "has_medicaid_health_coverage_at_interview": {2023: np.array([True, False])},
    }

    assert _build_reported_takeup_anchors(data, 2024) == {}


def test_build_reported_takeup_anchors_uses_present_period():
    data = {
        "person_tax_unit_id": {2024: np.array([1, 1, 2], dtype=np.int64)},
        "tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "reported_has_subsidized_marketplace_health_coverage_at_interview": {
            2024: np.array([True, False, False])
        },
        "has_medicaid_health_coverage_at_interview": {
            2024: np.array([False, True, False])
        },
    }

    anchors = _build_reported_takeup_anchors(data, 2024)

    np.testing.assert_array_equal(
        anchors["takes_up_aca_if_eligible"],
        np.array([True, False]),
    )
    np.testing.assert_array_equal(
        anchors["takes_up_medicaid_if_eligible"],
        np.array([False, True, False]),
    )


def test_build_reported_takeup_anchors_uses_subsidized_marketplace_only():
    data = {
        "person_tax_unit_id": {2024: np.array([1, 1, 2], dtype=np.int64)},
        "tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "has_marketplace_health_coverage_at_interview": {
            2024: np.array([True, False, True])
        },
        "reported_has_subsidized_marketplace_health_coverage_at_interview": {
            2024: np.array([False, False, True])
        },
    }

    anchors = _build_reported_takeup_anchors(data, 2024)

    np.testing.assert_array_equal(
        anchors["takes_up_aca_if_eligible"],
        np.array([False, True]),
    )


def test_compute_input_fingerprint_uses_loader_canonical_geography_identity(
    tmp_path, monkeypatch
):
    weights_path = tmp_path / "weights.npy"
    dataset_path = tmp_path / "dataset.h5"
    geo_one = tmp_path / "geography-one.npz"
    geo_two = tmp_path / "geography-two.npz"

    np.save(weights_path, np.array([1.0, 2.0, 3.0, 4.0]))
    dataset_path.write_bytes(b"dataset")
    geo_one.write_bytes(b"first-raw-geometry")
    geo_two.write_bytes(b"second-raw-geometry")

    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.resolve_source",
        lambda self, **kwargs: object(),
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.compute_canonical_checksum",
        lambda self, **kwargs: "sha256:canonical-geometry",
    )

    first = compute_input_fingerprint(
        weights_path,
        dataset_path,
        n_clones=2,
        geography_path=geo_one,
    )
    second = compute_input_fingerprint(
        weights_path,
        dataset_path,
        n_clones=2,
        geography_path=geo_two,
    )

    assert first == second
