"""Fixture helpers for ``test_local_h5_contracts.py``.

These helpers load the refactored ``local_h5`` contract modules from
disk under a synthetic package so the repository's heavyweight
top-level package initializers never run.
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
    """Load the local H5 contracts module and return its public exports."""

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

    module = _load_module(
        f"{package_name}.contracts",
        local_h5_root / "contracts.py",
    )
    return {
        "module": module,
        "AreaBuildRequest": module.AreaBuildRequest,
        "AreaBuildResult": module.AreaBuildResult,
        "AreaFilter": module.AreaFilter,
        "PublishingInputBundle": module.PublishingInputBundle,
        "ValidationIssue": module.ValidationIssue,
        "ValidationPolicy": module.ValidationPolicy,
        "ValidationResult": module.ValidationResult,
        "WorkerResult": module.WorkerResult,
    }
