import numpy as np
from types import SimpleNamespace

from policyengine_us_data.calibration.publish_local_area import (
    _build_reported_takeup_anchors,
    compute_input_fingerprint,
    load_calibration_geography,
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
        lambda self, **kwargs: SimpleNamespace(kind="saved_geography"),
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


def test_compute_input_fingerprint_passes_calibration_package_path_to_loader(
    tmp_path, monkeypatch
):
    weights_path = tmp_path / "weights.npy"
    dataset_path = tmp_path / "dataset.h5"
    package_path = tmp_path / "calibration_package.pkl"

    np.save(weights_path, np.array([1.0, 2.0, 3.0, 4.0]))
    dataset_path.write_bytes(b"dataset")
    package_path.write_bytes(b"package")

    seen = {}

    def fake_resolve_source(self, **kwargs):
        seen["resolve"] = kwargs
        return SimpleNamespace(kind="calibration_package")

    def fake_compute_canonical_checksum(self, **kwargs):
        seen["checksum"] = kwargs
        return "sha256:canonical-package"

    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.resolve_source",
        fake_resolve_source,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.compute_canonical_checksum",
        fake_compute_canonical_checksum,
    )

    compute_input_fingerprint(
        weights_path,
        dataset_path,
        n_clones=2,
        calibration_package_path=package_path,
    )

    assert seen["resolve"]["calibration_package_path"] == package_path
    assert seen["checksum"]["calibration_package_path"] == package_path


def test_load_calibration_geography_passes_calibration_package_path_to_loader(
    tmp_path, monkeypatch
):
    weights_path = tmp_path / "weights.npy"
    package_path = tmp_path / "calibration_package.pkl"

    np.save(weights_path, np.array([1.0, 2.0, 3.0, 4.0]))
    package_path.write_bytes(b"package")

    seen = {}

    def fake_resolve_source(self, **kwargs):
        seen["resolve"] = kwargs
        return None

    def fake_load(self, **kwargs):
        seen["load"] = kwargs
        return "geography"

    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.resolve_source",
        fake_resolve_source,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.publish_local_area.CalibrationGeographyLoader.load",
        fake_load,
    )

    result = load_calibration_geography(
        weights_path=weights_path,
        n_records=2,
        n_clones=2,
        calibration_package_path=package_path,
    )

    assert result == "geography"
    assert seen["resolve"]["calibration_package_path"] == package_path
    assert seen["load"]["calibration_package_path"] == package_path
