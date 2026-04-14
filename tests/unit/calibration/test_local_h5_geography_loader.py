import pickle

import numpy as np
import pytest

from tests.unit.calibration.fixtures.test_local_h5_geography_loader import (
    load_geography_loader_exports,
    write_saved_geography,
)


exports = load_geography_loader_exports()
geography_loader_module = exports["module"]
CalibrationGeographyLoader = exports["CalibrationGeographyLoader"]


def test_load_prefers_saved_geography_artifact(tmp_path):
    weights_path = tmp_path / "calibration_weights.npy"
    geography_path = tmp_path / "geography_assignment.npz"
    np.save(weights_path, np.array([1.0, 2.0]))
    write_saved_geography(geography_path, n_records=2, n_clones=2)

    loader = CalibrationGeographyLoader()
    geography = loader.load(weights_path=weights_path, n_records=2, n_clones=2)

    assert geography.n_records == 2
    assert geography.n_clones == 2
    assert tuple(str(item) for item in geography.cd_geoid) == ("101", "102", "101", "102")


def test_load_saved_geography_rejects_size_mismatch(tmp_path):
    weights_path = tmp_path / "calibration_weights.npy"
    geography_path = tmp_path / "geography_assignment.npz"
    np.save(weights_path, np.array([1.0, 2.0]))
    write_saved_geography(geography_path, n_records=2, n_clones=2)

    loader = CalibrationGeographyLoader()

    with pytest.raises(ValueError, match="n_records=2, expected 3"):
        loader.load(weights_path=weights_path, n_records=3, n_clones=2)


def test_load_falls_back_to_legacy_blocks(tmp_path, monkeypatch):
    weights_path = tmp_path / "calibration_weights.npy"
    blocks_path = tmp_path / "stacked_blocks.npy"
    np.save(weights_path, np.array([1.0, 2.0]))
    np.save(blocks_path, np.array(["010010000001", "010010000002", "010010000001", "010010000002"]))

    calls = {}

    def fake_reconstruct_geography_from_blocks(*, block_geoids, n_records, n_clones):
        calls["block_geoids"] = tuple(block_geoids)
        calls["n_records"] = n_records
        calls["n_clones"] = n_clones
        return "reconstructed"

    monkeypatch.setattr(
        geography_loader_module,
        "reconstruct_geography_from_blocks",
        fake_reconstruct_geography_from_blocks,
    )

    loader = CalibrationGeographyLoader()
    geography = loader.load(
        weights_path=weights_path,
        n_records=2,
        n_clones=2,
        blocks_path=blocks_path,
    )

    assert geography == "reconstructed"
    assert calls["n_records"] == 2
    assert calls["n_clones"] == 2


def test_load_from_calibration_package_derives_full_geography(tmp_path):
    weights_path = tmp_path / "calibration_weights.npy"
    package_path = tmp_path / "calibration_package.pkl"
    np.save(weights_path, np.array([1.0, 2.0]))
    with open(package_path, "wb") as handle:
        pickle.dump(
            {
                "block_geoid": np.array(
                    [
                        "010010000001",
                        "010010000002",
                        "010010000001",
                        "010010000002",
                    ]
                ),
                "cd_geoid": np.array(["101", "102", "101", "102"]),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    loader = CalibrationGeographyLoader()
    geography = loader.load(
        weights_path=weights_path,
        n_records=2,
        n_clones=2,
        calibration_package_path=package_path,
    )

    assert geography.n_records == 2
    assert geography.n_clones == 2
    assert tuple(geography.county_fips) == ("01001", "01001", "01001", "01001")
    assert tuple(int(item) for item in geography.state_fips) == (1, 1, 1, 1)


def test_compute_canonical_checksum_is_stable_across_source_formats(tmp_path):
    weights_path = tmp_path / "calibration_weights.npy"
    geography_path = tmp_path / "geography_assignment.npz"
    package_path = tmp_path / "calibration_package.pkl"
    np.save(weights_path, np.array([1.0, 2.0]))
    write_saved_geography(geography_path, n_records=2, n_clones=2)
    with open(package_path, "wb") as handle:
        pickle.dump(
            {
                "block_geoid": np.array(
                    [
                        "010010000001",
                        "010010000002",
                        "010010000001",
                        "010010000002",
                    ]
                ),
                "cd_geoid": np.array(["101", "102", "101", "102"]),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    loader = CalibrationGeographyLoader()
    saved_checksum = loader.compute_canonical_checksum(
        weights_path=weights_path,
        n_records=2,
        n_clones=2,
        geography_path=geography_path,
    )
    package_checksum = loader.compute_canonical_checksum(
        weights_path=weights_path,
        n_records=2,
        n_clones=2,
        calibration_package_path=package_path,
    )

    assert saved_checksum == package_checksum
