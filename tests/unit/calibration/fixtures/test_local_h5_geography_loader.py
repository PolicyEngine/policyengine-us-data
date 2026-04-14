"""Fixture helpers for ``test_local_h5_geography_loader.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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


def load_geography_loader_exports():
    """Load the local H5 geography loader under a synthetic package name."""

    repo_root = Path(__file__).resolve().parents[4]
    local_h5_root = (
        repo_root
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
    )
    calibration_root = repo_root / "policyengine_us_data" / "calibration"
    storage_root = repo_root / "policyengine_us_data" / "storage"
    package_name = "local_h5_geography_loader_fixture"
    policyengine_package = "policyengine_us_data"
    calibration_package = "policyengine_us_data.calibration"

    for name in list(sys.modules):
        if (
            name == package_name
            or name.startswith(f"{package_name}.")
            or name == policyengine_package
            or name.startswith(f"{policyengine_package}.")
        ):
            sys.modules.pop(name, None)

    _ensure_package(package_name, local_h5_root)
    _ensure_package(policyengine_package, repo_root / "policyengine_us_data")
    _ensure_package(calibration_package, calibration_root)
    _load_module(
        "policyengine_us_data.storage",
        storage_root / "__init__.py",
    )
    _load_module(
        "policyengine_us_data.calibration.clone_and_assign",
        calibration_root / "clone_and_assign.py",
    )
    module = _load_module(
        f"{package_name}.geography_loader",
        local_h5_root / "geography_loader.py",
    )
    return {
        "module": module,
        "CalibrationGeographyLoader": module.CalibrationGeographyLoader,
        "ResolvedGeographySource": module.ResolvedGeographySource,
    }


def write_saved_geography(path: Path, *, n_records: int, n_clones: int) -> None:
    """Write a small saved geography artifact for loader tests."""

    repo_root = Path(__file__).resolve().parents[4]
    calibration_root = repo_root / "policyengine_us_data" / "calibration"
    _ensure_package("policyengine_us_data", repo_root / "policyengine_us_data")
    _ensure_package("policyengine_us_data.calibration", calibration_root)
    clone_and_assign = _load_module(
        "policyengine_us_data.calibration.clone_and_assign",
        calibration_root / "clone_and_assign.py",
    )

    total_rows = n_records * n_clones
    block_geoids = ["010010000001", "010010000002"] * n_clones
    cd_geoids = ["101", "102"] * n_clones
    clone_and_assign.save_geography(
        clone_and_assign.GeographyAssignment(
            block_geoid=block_geoids[:total_rows],
            cd_geoid=cd_geoids[:total_rows],
            county_fips=["01001"] * total_rows,
            state_fips=[1] * total_rows,
            n_records=n_records,
            n_clones=n_clones,
        ),
        path,
    )
