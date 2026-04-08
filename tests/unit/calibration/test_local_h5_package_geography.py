import importlib.util
import pickle
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np


def _load_package_geography_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
        / "package_geography.py"
    )
    spec = importlib.util.spec_from_file_location(
        "local_h5_package_geography",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_clone_and_assign(monkeypatch):
    package = types.ModuleType("policyengine_us_data")
    package.__path__ = []
    calibration_package = types.ModuleType("policyengine_us_data.calibration")
    calibration_package.__path__ = []
    clone_module = types.ModuleType("policyengine_us_data.calibration.clone_and_assign")

    @dataclass(frozen=True)
    class FakeGeographyAssignment:
        block_geoid: np.ndarray
        cd_geoid: np.ndarray
        county_fips: np.ndarray
        state_fips: np.ndarray
        n_records: int
        n_clones: int

    def fake_assign_random_geography(*, n_records, n_clones, seed):
        total = n_records * n_clones
        block_geoid = np.asarray(["990010000000001"] * total, dtype=str)
        cd_geoid = np.asarray(["9901"] * total, dtype=str)
        county_fips = np.asarray(["99001"] * total, dtype=str)
        state_fips = np.asarray([99] * total, dtype=np.int64)
        return FakeGeographyAssignment(
            block_geoid=block_geoid,
            cd_geoid=cd_geoid,
            county_fips=county_fips,
            state_fips=state_fips,
            n_records=n_records,
            n_clones=n_clones,
        )

    clone_module.GeographyAssignment = FakeGeographyAssignment
    clone_module.assign_random_geography = fake_assign_random_geography

    monkeypatch.setitem(sys.modules, "policyengine_us_data", package)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration",
        calibration_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.clone_and_assign",
        clone_module,
    )

    return FakeGeographyAssignment


package_geography = _load_package_geography_module()
CalibrationPackageGeographyLoader = package_geography.CalibrationPackageGeographyLoader
require_calibration_package_path = package_geography.require_calibration_package_path


def test_serialize_and_load_serialized_package_geography(monkeypatch):
    FakeGeographyAssignment = _install_fake_clone_and_assign(monkeypatch)
    loader = CalibrationPackageGeographyLoader()
    geography = FakeGeographyAssignment(
        block_geoid=np.asarray(["060010001001001", "060010001001002"], dtype=str),
        cd_geoid=np.asarray(["601", "601"], dtype=str),
        county_fips=np.asarray(["06001", "06001"], dtype=str),
        state_fips=np.asarray([6, 6], dtype=np.int64),
        n_records=2,
        n_clones=1,
    )

    payload = loader.serialize_geography(geography)
    loaded = loader.load_from_package_dict({"geography": payload})

    assert loaded is not None
    assert loaded.source == "serialized_package"
    np.testing.assert_array_equal(loaded.geography.block_geoid, geography.block_geoid)
    np.testing.assert_array_equal(loaded.geography.cd_geoid, geography.cd_geoid)
    np.testing.assert_array_equal(loaded.geography.county_fips, geography.county_fips)
    np.testing.assert_array_equal(loaded.geography.state_fips, geography.state_fips)
    assert loaded.geography.n_records == 2
    assert loaded.geography.n_clones == 1


def test_load_legacy_package_geography_derives_county_and_state(monkeypatch):
    _install_fake_clone_and_assign(monkeypatch)
    loader = CalibrationPackageGeographyLoader()

    loaded = loader.load_from_package_dict(
        {
            "block_geoid": np.asarray(
                [
                    "060010001001001",
                    "060010001001002",
                    "360610001001001",
                    "360610001001002",
                ],
                dtype=str,
            ),
            "cd_geoid": np.asarray(["601", "601", "1208", "1208"], dtype=str),
            "metadata": {"base_n_records": 2, "n_clones": 2},
        }
    )

    assert loaded is not None
    assert loaded.source == "legacy_package"
    np.testing.assert_array_equal(
        loaded.geography.county_fips,
        np.asarray(["06001", "06001", "36061", "36061"], dtype=str),
    )
    np.testing.assert_array_equal(
        loaded.geography.state_fips,
        np.asarray([6, 6, 36, 36], dtype=np.int64),
    )
    assert loaded.geography.n_records == 2
    assert loaded.geography.n_clones == 2
    assert loaded.warnings


def test_resolve_for_weights_falls_back_when_package_geography_length_mismatches(
    monkeypatch, tmp_path
):
    _install_fake_clone_and_assign(monkeypatch)
    loader = CalibrationPackageGeographyLoader()

    package_path = tmp_path / "package.pkl"
    with open(package_path, "wb") as f:
        pickle.dump(
            {
                "geography": {
                    "block_geoid": np.asarray(["060010001001001"] * 4, dtype=str),
                    "cd_geoid": np.asarray(["601"] * 4, dtype=str),
                    "county_fips": np.asarray(["06001"] * 4, dtype=str),
                    "state_fips": np.asarray([6] * 4, dtype=np.int64),
                    "n_records": 2,
                    "n_clones": 2,
                }
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    resolved = loader.resolve_for_weights(
        package_path=package_path,
        weights_length=6,
        n_records=2,
        n_clones=3,
        seed=42,
    )

    assert resolved.source == "generated"
    assert resolved.geography.n_records == 2
    assert resolved.geography.n_clones == 3
    assert any("does not match weights length" in warning for warning in resolved.warnings)


def test_require_calibration_package_path_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.pkl"

    try:
        require_calibration_package_path(missing)
    except FileNotFoundError as error:
        assert "Required calibration package not found" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError for missing package")
