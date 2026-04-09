import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


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

    entity_graph = _load_module(
        "policyengine_us_data.calibration.local_h5.entity_graph",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "entity_graph.py",
        ),
    )
    source_dataset = _load_module(
        "policyengine_us_data.calibration.local_h5.source_dataset",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "source_dataset.py",
        ),
    )
    return entity_graph, source_dataset


def test_entity_graph_extractor_builds_household_relationships(monkeypatch):
    entity_graph_module, _ = _install_fake_package_hierarchy(monkeypatch)
    EntityGraphExtractor = entity_graph_module.EntityGraphExtractor

    extractor = EntityGraphExtractor(("tax_unit", "spm_unit"))
    graph = extractor.extract_from_arrays(
        household_ids=np.asarray([10, 20]),
        person_household_ids=np.asarray([10, 10, 20, 20, 20]),
        entity_id_arrays={
            "tax_unit": np.asarray([100, 200, 300]),
            "spm_unit": np.asarray([900, 901]),
        },
        person_entity_id_arrays={
            "tax_unit": np.asarray([100, 100, 200, 300, 300]),
            "spm_unit": np.asarray([900, 900, 901, 901, 901]),
        },
    )

    assert graph.hh_id_to_index == {10: 0, 20: 1}
    assert graph.hh_to_persons == {0: (0, 1), 1: (2, 3, 4)}
    assert graph.hh_to_entity["tax_unit"] == {0: (0,), 1: (1, 2)}
    assert graph.hh_to_entity["spm_unit"] == {0: (0,), 1: (1,)}


def test_policy_engine_dataset_reader_builds_snapshot_without_eager_holder_access(
    monkeypatch, tmp_path
):
    _, source_dataset_module = _install_fake_package_hierarchy(monkeypatch)
    PolicyEngineDatasetReader = source_dataset_module.PolicyEngineDatasetReader

    class FakeHolder:
        def __init__(self, values):
            self.values = np.asarray(values)

        def get_known_periods(self):
            return (2024,)

        def get_array(self, period):
            assert period == 2024
            return self.values

    class FakeVariableDef:
        def __init__(self, entity_key):
            self.entity = types.SimpleNamespace(key=entity_key)

    class FakeVariables(dict):
        def keys(self):
            return super().keys()

    class FakeMicrosimulation:
        instances = []

        def __init__(self, dataset):
            self.dataset = dataset
            self.default_calculation_period = 2024
            self.input_variables = {"household_id", "tax_unit_id"}
            self.tax_benefit_system = types.SimpleNamespace(
                variables=FakeVariables(
                    {
                        "sample_var": FakeVariableDef("household"),
                    }
                )
            )
            self.get_holder_calls = 0
            FakeMicrosimulation.instances.append(self)

        def calculate(self, variable, map_to=None):
            lookup = {
                ("household_id", "household"): np.asarray([10, 20]),
                ("household_id", "person"): np.asarray([10, 10, 20]),
                ("tax_unit_id", "tax_unit"): np.asarray([100, 200]),
                ("person_tax_unit_id", "person"): np.asarray([100, 100, 200]),
            }
            return types.SimpleNamespace(values=lookup[(variable, map_to)])

        def get_holder(self, variable):
            self.get_holder_calls += 1
            assert variable == "sample_var"
            return FakeHolder([1, 2])

    fake_policyengine_us = types.ModuleType("policyengine_us")
    fake_policyengine_us.Microsimulation = FakeMicrosimulation
    monkeypatch.setitem(sys.modules, "policyengine_us", fake_policyengine_us)

    reader = PolicyEngineDatasetReader(("tax_unit",))
    snapshot = reader.load(tmp_path / "source.h5")

    assert snapshot.dataset_path == tmp_path / "source.h5"
    assert snapshot.time_period == 2024
    assert snapshot.n_households == 2
    assert snapshot.input_variables == frozenset({"household_id", "tax_unit_id"})
    assert snapshot.entity_graph.hh_to_persons == {0: (0, 1), 1: (2,)}

    fake_sim = FakeMicrosimulation.instances[-1]
    assert fake_sim.get_holder_calls == 0

    periods = snapshot.variable_provider.get_known_periods("sample_var")

    assert periods == (2024,)
    assert fake_sim.get_holder_calls == 1
