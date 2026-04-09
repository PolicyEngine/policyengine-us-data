import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np
import pytest


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

    contracts = _load_module(
        "policyengine_us_data.calibration.local_h5.contracts",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "contracts.py",
        ),
    )
    weights = _load_module(
        "policyengine_us_data.calibration.local_h5.weights",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "weights.py",
        ),
    )
    selection = _load_module(
        "policyengine_us_data.calibration.local_h5.selection",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "selection.py",
        ),
    )
    return contracts, weights, selection


@dataclass(frozen=True)
class FakeGeography:
    block_geoid: np.ndarray
    cd_geoid: np.ndarray
    county_fips: np.ndarray
    state_fips: np.ndarray
    n_records: int
    n_clones: int


def _sample_geography() -> FakeGeography:
    return FakeGeography(
        block_geoid=np.asarray(
            [
                "060010001001001",
                "360610001001001",
                "060130001001001",
                "360810001001001",
                "060010001001002",
                "360610001001002",
                "060130001001002",
                "360810001001002",
            ],
            dtype=str,
        ),
        cd_geoid=np.asarray(
            ["601", "1208", "605", "1214", "601", "1208", "605", "1214"],
            dtype=str,
        ),
        county_fips=np.asarray(
            ["06001", "36061", "06013", "36081", "06001", "36061", "06013", "36081"],
            dtype=str,
        ),
        state_fips=np.asarray([6, 36, 6, 36, 6, 36, 6, 36], dtype=np.int64),
        n_records=4,
        n_clones=2,
    )


def test_clone_weight_matrix_validates_shape(monkeypatch):
    _, weights_module, _ = _install_fake_package_hierarchy(monkeypatch)
    CloneWeightMatrix = weights_module.CloneWeightMatrix

    matrix = CloneWeightMatrix.from_vector(np.arange(8, dtype=float), n_records=4)

    assert matrix.n_clones == 2
    assert matrix.shape == (2, 4)
    np.testing.assert_array_equal(
        matrix.as_matrix(),
        np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=float),
    )


def test_clone_weight_matrix_rejects_invalid_shapes(monkeypatch):
    _, weights_module, _ = _install_fake_package_hierarchy(monkeypatch)
    CloneWeightMatrix = weights_module.CloneWeightMatrix

    with pytest.raises(ValueError, match="n_records must be positive"):
        CloneWeightMatrix.from_vector(np.arange(4, dtype=float), n_records=0)

    with pytest.raises(ValueError, match="not divisible"):
        CloneWeightMatrix.from_vector(np.arange(7, dtype=float), n_records=4)


def test_area_selector_supports_national_state_district_and_city(monkeypatch):
    contracts, weights_module, selection_module = _install_fake_package_hierarchy(
        monkeypatch
    )
    AreaFilter = contracts.AreaFilter
    CloneWeightMatrix = weights_module.CloneWeightMatrix
    AreaSelector = selection_module.AreaSelector

    weights = CloneWeightMatrix.from_vector(
        np.asarray([1.0, 0.0, 2.0, 0.0, 0.5, 1.5, 0.0, 3.0]),
        n_records=4,
    )
    geography = _sample_geography()
    selector = AreaSelector()

    national = selector.select(weights, geography)
    state = selector.select(
        weights,
        geography,
        filters=(AreaFilter("state_fips", "eq", 6),),
    )
    district = selector.select(
        weights,
        geography,
        filters=(AreaFilter("cd_geoid", "eq", "1208"),),
    )
    city = selector.select(
        weights,
        geography,
        filters=(AreaFilter("county_fips", "in", ("36061", "36081")),),
    )

    np.testing.assert_array_equal(
        national.active_clone_indices,
        np.asarray([0, 0, 1, 1, 1]),
    )
    np.testing.assert_array_equal(
        national.active_household_indices,
        np.asarray([0, 2, 0, 1, 3]),
    )
    np.testing.assert_array_equal(state.active_weights, np.asarray([1.0, 2.0, 0.5]))
    np.testing.assert_array_equal(state.active_state_fips, np.asarray([6, 6, 6]))
    np.testing.assert_array_equal(district.active_weights, np.asarray([1.5]))
    np.testing.assert_array_equal(district.active_cd_geoids, np.asarray(["1208"]))
    np.testing.assert_array_equal(city.active_weights, np.asarray([1.5, 3.0]))
    np.testing.assert_array_equal(
        city.active_county_fips,
        np.asarray(["36061", "36081"]),
    )


def test_area_selector_returns_empty_selection(monkeypatch):
    contracts, weights_module, selection_module = _install_fake_package_hierarchy(
        monkeypatch
    )
    AreaFilter = contracts.AreaFilter
    CloneWeightMatrix = weights_module.CloneWeightMatrix
    AreaSelector = selection_module.AreaSelector

    weights = CloneWeightMatrix.from_vector(
        np.asarray([1.0, 0.0, 2.0, 0.0, 0.5, 1.5, 0.0, 3.0]),
        n_records=4,
    )
    selector = AreaSelector()
    selection = selector.select(
        weights,
        _sample_geography(),
        filters=(AreaFilter("county_fips", "eq", "99999"),),
    )

    assert selection.is_empty
    assert selection.n_household_clones == 0
    assert selection.active_weights.size == 0


def test_area_selector_is_deterministic(monkeypatch):
    contracts, weights_module, selection_module = _install_fake_package_hierarchy(
        monkeypatch
    )
    AreaFilter = contracts.AreaFilter
    CloneWeightMatrix = weights_module.CloneWeightMatrix
    AreaSelector = selection_module.AreaSelector

    weights = CloneWeightMatrix.from_vector(
        np.asarray([1.0, 0.0, 2.0, 0.0, 0.5, 1.5, 0.0, 3.0]),
        n_records=4,
    )
    geography = _sample_geography()
    selector = AreaSelector()
    filters = (AreaFilter("state_fips", "eq", 36),)

    first = selector.select(weights, geography, filters=filters)
    second = selector.select(weights, geography, filters=filters)

    np.testing.assert_array_equal(
        first.active_clone_indices,
        second.active_clone_indices,
    )
    np.testing.assert_array_equal(
        first.active_household_indices, second.active_household_indices
    )
    np.testing.assert_array_equal(first.active_weights, second.active_weights)
