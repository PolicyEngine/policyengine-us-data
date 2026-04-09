import importlib.util
from pathlib import Path
import sys
import types


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

    calibration_utils = types.ModuleType(
        "policyengine_us_data.calibration.calibration_utils"
    )
    calibration_utils.STATE_CODES = {1: "AL", 2: "AK", 36: "NY"}
    calibration_utils.get_all_cds_from_database = lambda _db_uri: ["0101", "0200", "3607"]
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.calibration_utils",
        calibration_utils,
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
    area_catalog = _load_module(
        "policyengine_us_data.calibration.local_h5.area_catalog",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "area_catalog.py",
        ),
    )
    return contracts, area_catalog


def test_us_area_catalog_constructs_weighted_regional_entries(monkeypatch):
    _, area_catalog_module = _install_fake_package_hierarchy(monkeypatch)
    USAreaCatalog = area_catalog_module.USAreaCatalog

    geography = types.SimpleNamespace(
        county_fips=["36061", "01001", "36047"],
        cd_geoid=["3607", "0101", "3607"],
    )
    catalog = USAreaCatalog()

    entries = catalog.resolved_regional_entries(
        "sqlite:////tmp/policy_data.db",
        geography=geography,
    )

    state_entries = [e for e in entries if e.request.area_type == "state"]
    district_entries = [e for e in entries if e.request.area_type == "district"]
    city_entries = [e for e in entries if e.request.area_type == "city"]

    assert [e.request.area_id for e in district_entries] == ["AL-01", "AK-01", "NY-07"]
    assert city_entries[0].request.output_relative_path == "cities/NYC.h5"
    assert city_entries[0].request.validation_geographic_ids == ("3607",)
    assert state_entries[0].request.filters[0].value == ("0101",)
    assert state_entries[1].weight == 1
    assert city_entries[0].weight == 11


def test_us_area_catalog_constructs_national_request(monkeypatch):
    _, area_catalog_module = _install_fake_package_hierarchy(monkeypatch)
    USAreaCatalog = area_catalog_module.USAreaCatalog

    entry = USAreaCatalog().resolved_national_entry()

    assert entry.request.area_type == "national"
    assert entry.request.output_relative_path == "national/US.h5"
    assert entry.request.validation_geo_level == "national"
    assert entry.request.validation_geographic_ids == ("US",)
    assert entry.weight == 1
