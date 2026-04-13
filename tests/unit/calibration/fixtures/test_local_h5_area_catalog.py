"""Fixture helpers for ``test_local_h5_area_catalog.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

__test__ = False


def _ensure_package(name: str, path: Path) -> None:
    """Register a synthetic package so relative imports resolve locally."""

    package = sys.modules.get(name)
    if package is None:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
        return
    package.__path__ = [str(path)]


def _load_module(name: str, path: Path):
    """Load one module from disk under a specific fully-qualified name."""

    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_area_catalog_exports():
    """Load the local H5 area catalog and related request contracts."""

    local_h5_root = (
        Path(__file__).resolve().parents[4]
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
    )
    package_name = "local_h5_area_catalog_fixture"

    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name, None)

    _ensure_package(package_name, local_h5_root)
    requests_module = _load_module(
        f"{package_name}.requests",
        local_h5_root / "requests" / "__init__.py",
    )
    area_catalog_module = _load_module(
        f"{package_name}.area_catalog",
        local_h5_root / "area_catalog.py",
    )
    return {
        "module": area_catalog_module,
        "AreaBuildRequest": requests_module.AreaBuildRequest,
        "AreaFilter": requests_module.AreaFilter,
        "USAreaCatalog": area_catalog_module.USAreaCatalog,
    }


def make_geography(*, cd_geoids, county_fips=None):
    """Build a simple geography-like object for unit tests."""

    return SimpleNamespace(
        cd_geoid=list(cd_geoids),
        county_fips=list(county_fips or []),
    )
