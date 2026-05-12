import pytest

from tests.unit.fixtures.geography import write_weights
from tests.unit.fixtures.geography_loader import (
    CalibrationGeographyLoader,
    patch_reconstruct_geography_from_blocks,
    write_legacy_blocks_artifact,
    write_loader_calibration_package,
    write_saved_geography_artifact,
    write_stage_2_loader_calibration_package,
)
from policyengine_us_data.stage_contracts.calibration_package import (
    summarize_geography_assignment,
)


def test_load_prefers_saved_geography_artifact(tmp_path):
    weights_path = tmp_path / "calibration_weights.npy"
    geography_path = tmp_path / "geography_assignment.npz"
    write_weights(weights_path)
    write_saved_geography_artifact(geography_path)

    loader = CalibrationGeographyLoader()
    geography = loader.load(weights_path=weights_path, n_records=2, n_clones=2)

    assert geography.n_records == 2
    assert geography.n_clones == 2
    assert tuple(str(item) for item in geography.cd_geoid) == (
        "101",
        "102",
        "101",
        "102",
    )


def test_load_saved_geography_rejects_size_mismatch(tmp_path):
    weights_path = tmp_path / "calibration_weights.npy"
    geography_path = tmp_path / "geography_assignment.npz"
    write_weights(weights_path)
    write_saved_geography_artifact(geography_path)

    loader = CalibrationGeographyLoader()

    with pytest.raises(ValueError, match="n_records=2, expected 3"):
        loader.load(weights_path=weights_path, n_records=3, n_clones=2)


def test_load_falls_back_to_legacy_blocks(tmp_path, monkeypatch):
    weights_path = tmp_path / "calibration_weights.npy"
    blocks_path = tmp_path / "stacked_blocks.npy"
    write_weights(weights_path)
    write_legacy_blocks_artifact(blocks_path)
    spy = patch_reconstruct_geography_from_blocks(monkeypatch)

    loader = CalibrationGeographyLoader()
    geography = loader.load(
        weights_path=weights_path,
        n_records=2,
        n_clones=2,
        blocks_path=blocks_path,
    )

    assert geography == "reconstructed"
    assert spy.n_records == 2
    assert spy.n_clones == 2


def test_load_from_calibration_package_derives_full_geography(tmp_path):
    weights_path = tmp_path / "calibration_weights.npy"
    package_path = tmp_path / "calibration_package.pkl"
    write_weights(weights_path)
    write_loader_calibration_package(package_path)

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
    write_weights(weights_path)
    write_saved_geography_artifact(geography_path)
    write_loader_calibration_package(package_path)

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


def test_compute_canonical_checksum_matches_stage_2_geography_summary(tmp_path):
    weights_path = tmp_path / "calibration_weights.npy"
    package_path = tmp_path / "calibration_package.pkl"
    write_weights(weights_path)
    package = write_stage_2_loader_calibration_package(package_path)

    loader = CalibrationGeographyLoader()
    loader_checksum = loader.compute_canonical_checksum(
        weights_path=weights_path,
        n_records=2,
        n_clones=2,
        calibration_package_path=package_path,
    )
    summary = summarize_geography_assignment(package)

    assert loader_checksum == summary.canonical_geography_sha256
