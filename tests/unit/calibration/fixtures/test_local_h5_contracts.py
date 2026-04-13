"""Fixture helpers for ``test_local_h5_contracts.py``.

These helpers load the themed ``local_h5`` subpackages from disk under
a synthetic package so the repository's heavyweight top-level package
initializers never run.
"""

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


def load_contracts_exports():
    """Load the themed local H5 contract packages and return key exports."""

    local_h5_root = (
        Path(__file__).resolve().parents[4]
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
    )
    package_name = "local_h5_fixture"

    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name, None)

    _ensure_package(package_name, local_h5_root)

    requests_module = _load_module(
        f"{package_name}.requests",
        local_h5_root / "requests" / "__init__.py",
    )
    inputs_module = _load_module(
        f"{package_name}.inputs",
        local_h5_root / "inputs" / "__init__.py",
    )
    validation_module = _load_module(
        f"{package_name}.validation",
        local_h5_root / "validation" / "__init__.py",
    )
    results_module = _load_module(
        f"{package_name}.results",
        local_h5_root / "results" / "__init__.py",
    )
    package_module = _load_module(package_name, local_h5_root / "__init__.py")
    return {
        "module": package_module,
        "AreaBuildRequest": requests_module.AreaBuildRequest,
        "AreaBuildResult": results_module.AreaBuildResult,
        "AreaFilter": requests_module.AreaFilter,
        "PublishingInputBundle": inputs_module.PublishingInputBundle,
        "ValidationIssue": validation_module.ValidationIssue,
        "ValidationPolicy": validation_module.ValidationPolicy,
        "ValidationResult": validation_module.ValidationResult,
        "WorkerResult": results_module.WorkerResult,
        "make_national_request": make_national_request,
    }


def make_national_request(area_build_request_cls):
    """Build the canonical national request shape used by current tests."""

    return area_build_request_cls(
        area_type="national",
        area_id="US",
        display_name="US",
        output_relative_path="national/US.h5",
        validation_geo_level="national",
        validation_geographic_ids=("US",),
    )
