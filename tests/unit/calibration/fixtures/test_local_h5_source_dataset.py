"""Fixture helpers for ``test_local_h5_source_dataset.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

__test__ = False


def _ensure_package(name: str, path: Path) -> None:
    """Register a synthetic package so local imports resolve from disk."""

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


def load_source_dataset_exports():
    """Load the local H5 source dataset module under a synthetic package name."""

    local_h5_root = (
        Path(__file__).resolve().parents[4]
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
    )
    package_name = "local_h5_source_dataset_fixture"

    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name, None)

    _ensure_package(package_name, local_h5_root)
    source_dataset_module = _load_module(
        f"{package_name}.source_dataset",
        local_h5_root / "source_dataset.py",
    )
    return {
        "module": source_dataset_module,
        "EntityGraph": source_dataset_module.EntityGraph,
        "MicrosimulationVariableProvider": (
            source_dataset_module.MicrosimulationVariableProvider
        ),
        "PolicyEngineDatasetReader": source_dataset_module.PolicyEngineDatasetReader,
        "SourceDatasetSnapshot": source_dataset_module.SourceDatasetSnapshot,
    }


def make_entity_graph_arrays():
    """Return small valid entity graph arrays with two households."""

    return {
        "household_ids": np.array([10, 20], dtype=np.int64),
        "person_household_ids": np.array([10, 10, 20], dtype=np.int64),
        "subentity_ids": {
            "tax_unit": np.array([100, 200], dtype=np.int64),
            "spm_unit": np.array([300, 400], dtype=np.int64),
        },
        "person_subentity_ids": {
            "tax_unit": np.array([100, 100, 200], dtype=np.int64),
            "spm_unit": np.array([300, 300, 400], dtype=np.int64),
        },
    }


class FakeHolder:
    """Small holder test double for lazy-provider tests."""

    def __init__(self, arrays_by_period):
        self.arrays_by_period = dict(arrays_by_period)
        self.known_period_calls = 0
        self.get_array_calls = []

    def get_known_periods(self):
        self.known_period_calls += 1
        return tuple(self.arrays_by_period)

    def get_array(self, period):
        self.get_array_calls.append(period)
        return self.arrays_by_period[period]


class FakeSimulation:
    """Small simulation test double for lazy-provider tests."""

    def __init__(self, holders):
        self.holders = dict(holders)
        self.input_variables = frozenset(holders)
        self.default_calculation_period = 2023
        self.get_holder_calls = []

    def get_holder(self, variable):
        self.get_holder_calls.append(variable)
        if variable not in self.holders:
            raise KeyError(variable)
        return self.holders[variable]

    def calculate(self, variable, map_to=None):
        holder_variable = (
            "person_household_id"
            if variable == "household_id" and map_to == "person"
            else variable
        )
        holder = self.get_holder(holder_variable)
        period = next(iter(holder.arrays_by_period))
        return FakeCalculation(holder.arrays_by_period[period])


class FakeCalculation:
    """Small calculation result object with a ``values`` attribute."""

    def __init__(self, values):
        self.values = np.asarray(values)
