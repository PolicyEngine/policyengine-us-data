import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd


def _module_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[3].joinpath(*parts)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_package_hierarchy(monkeypatch):
    package = types.ModuleType("policyengine_us_data")
    package.__path__ = []
    calibration_package = types.ModuleType("policyengine_us_data.calibration")
    calibration_package.__path__ = []
    local_h5_package = types.ModuleType("policyengine_us_data.calibration.local_h5")
    local_h5_package.__path__ = []

    monkeypatch.setitem(sys.modules, "policyengine_us_data", package)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration",
        calibration_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5",
        local_h5_package,
    )

    clone_module = types.ModuleType("policyengine_us_data.calibration.clone_and_assign")

    @dataclass(frozen=True)
    class FakeGeographyAssignment:
        block_geoid: np.ndarray
        cd_geoid: np.ndarray
        county_fips: np.ndarray
        state_fips: np.ndarray
        n_records: int
        n_clones: int

    clone_module.GeographyAssignment = FakeGeographyAssignment
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.clone_and_assign",
        clone_module,
    )

    _load_module(
        "policyengine_us_data.calibration.local_h5.package_geography",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "package_geography.py",
        ),
    )
    unified_calibration = _load_module(
        "policyengine_us_data.calibration.unified_calibration",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "unified_calibration.py",
        ),
    )
    return FakeGeographyAssignment, unified_calibration


def test_save_and_load_calibration_package_round_trips_serialized_geography(
    monkeypatch, tmp_path
):
    FakeGeographyAssignment, unified_calibration = _install_fake_package_hierarchy(
        monkeypatch
    )

    geography = FakeGeographyAssignment(
        block_geoid=np.asarray(["060010001001001", "360610001001001"], dtype=str),
        cd_geoid=np.asarray(["601", "1208"], dtype=str),
        county_fips=np.asarray(["06001", "36061"], dtype=str),
        state_fips=np.asarray([6, 36], dtype=np.int64),
        n_records=2,
        n_clones=1,
    )
    package_path = tmp_path / "calibration_package.pkl"

    unified_calibration.save_calibration_package(
        path=str(package_path),
        X_sparse=np.zeros((1, 1), dtype=np.float64),
        targets_df=pd.DataFrame({"variable": ["household_count"], "value": [1.0]}),
        target_names=["household_count"],
        metadata={"created_at": "2026-04-10T00:00:00Z"},
        geography=geography,
        initial_weights=np.asarray([1.0], dtype=np.float64),
    )

    loaded = unified_calibration.load_calibration_package(str(package_path))

    assert loaded["geography"] is not None
    np.testing.assert_array_equal(
        loaded["geography"]["block_geoid"],
        geography.block_geoid,
    )
    np.testing.assert_array_equal(
        loaded["geography"]["cd_geoid"],
        geography.cd_geoid,
    )
    np.testing.assert_array_equal(
        loaded["geography"]["county_fips"],
        geography.county_fips,
    )
    np.testing.assert_array_equal(
        loaded["geography"]["state_fips"],
        geography.state_fips,
    )
    assert loaded["geography"]["n_records"] == geography.n_records
    assert loaded["geography"]["n_clones"] == geography.n_clones
    np.testing.assert_array_equal(loaded["cd_geoid"], geography.cd_geoid)
    np.testing.assert_array_equal(loaded["block_geoid"], geography.block_geoid)
