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
    _load_module(
        "policyengine_us_data.calibration.local_h5.contracts",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "contracts.py",
        ),
    )
    _load_module(
        "policyengine_us_data.calibration.local_h5.weights",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "weights.py",
        ),
    )
    _load_module(
        "policyengine_us_data.calibration.local_h5.selection",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "selection.py",
        ),
    )
    reindexing = _load_module(
        "policyengine_us_data.calibration.local_h5.reindexing",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "reindexing.py",
        ),
    )
    variables = _load_module(
        "policyengine_us_data.calibration.local_h5.variables",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "variables.py",
        ),
    )
    return entity_graph, source_dataset, reindexing, variables


_EnumLike = type("Enum", (), {})


class FakeVariableDef:
    def __init__(self, entity_key, value_type):
        self.entity = types.SimpleNamespace(key=entity_key)
        self.value_type = value_type


class FakeProvider:
    def __init__(self):
        self._definitions = {
            "household_income": FakeVariableDef("household", float),
            "person_age": FakeVariableDef("person", int),
            "tax_unit_amount": FakeVariableDef("tax_unit", float),
            "enum_status": FakeVariableDef("household", _EnumLike),
            "county_fips": FakeVariableDef("household", str),
            "two_period_var": FakeVariableDef("household", float),
            "ignored_output_only": FakeVariableDef("output_only", float),
        }
        self._periods = {
            "household_income": (2024,),
            "person_age": (2024,),
            "tax_unit_amount": (2024,),
            "enum_status": (2024,),
            "county_fips": (2024,),
            "two_period_var": (2023, 2024),
            "ignored_output_only": (2024,),
        }
        self._arrays = {
            ("household_income", 2024): np.asarray([100.0, 200.0]),
            ("person_age", 2024): np.asarray([34, 35, 50, 18, 17]),
            ("tax_unit_amount", 2024): np.asarray([10.0, 20.0, 30.0]),
            ("enum_status", 2024): np.asarray(["A", "B"], dtype=object),
            ("county_fips", 2024): np.asarray(["06001", "36061"], dtype=object),
            ("two_period_var", 2023): np.asarray([1.0, 2.0]),
            ("two_period_var", 2024): np.asarray([3.0, 4.0]),
            ("ignored_output_only", 2024): np.asarray([9.0, 9.0]),
        }

    def list_variables(self):
        return tuple(self._definitions.keys())

    def get_known_periods(self, variable):
        return self._periods[variable]

    def get_array(self, variable, period):
        return self._arrays[(variable, period)]

    def get_variable_definition(self, variable):
        return self._definitions.get(variable)

    def calculate(self, variable, *, map_to=None):
        raise AssertionError("calculate should not be used in VariableCloner tests")


def _make_snapshot(source_dataset_module, entity_graph_module):
    EntityGraph = entity_graph_module.EntityGraph
    SourceDatasetSnapshot = source_dataset_module.SourceDatasetSnapshot

    graph = EntityGraph(
        household_ids=np.asarray([10, 20]),
        person_household_ids=np.asarray([10, 10, 20, 20, 20]),
        hh_id_to_index={10: 0, 20: 1},
        hh_to_persons={0: (0, 1), 1: (2, 3, 4)},
        entity_id_arrays={
            "tax_unit": np.asarray([100, 200, 300]),
        },
        person_entity_id_arrays={
            "tax_unit": np.asarray([100, 100, 200, 300, 300]),
        },
        hh_to_entity={
            "tax_unit": {0: (0,), 1: (1, 2)},
        },
    )
    return SourceDatasetSnapshot(
        dataset_path=Path("/tmp/source.h5"),
        time_period=2024,
        household_ids=np.asarray([10, 20]),
        entity_graph=graph,
        input_variables=frozenset(
            {
                "household_income",
                "person_age",
                "tax_unit_amount",
                "enum_status",
                "county_fips",
                "two_period_var",
                "ignored_output_only",
            }
        ),
        variable_provider=FakeProvider(),
    )


def _make_reindexed(reindexing_module):
    ReindexedEntities = reindexing_module.ReindexedEntities
    return ReindexedEntities(
        household_source_indices=np.asarray([1, 0], dtype=np.int64),
        person_source_indices=np.asarray([2, 3, 4, 0, 1], dtype=np.int64),
        entity_source_indices={
            "tax_unit": np.asarray([1, 2, 0], dtype=np.int64),
        },
        persons_per_clone=np.asarray([3, 2], dtype=np.int64),
        entities_per_clone={
            "tax_unit": np.asarray([2, 1], dtype=np.int64),
        },
        new_household_ids=np.asarray([0, 1], dtype=np.int32),
        new_person_ids=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        new_person_household_ids=np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        new_entity_ids={
            "tax_unit": np.asarray([0, 1, 2], dtype=np.int32),
        },
        new_person_entity_ids={
            "tax_unit": np.asarray([0, 1, 1, 2, 2], dtype=np.int32),
        },
    )


def test_variable_cloner_slices_household_person_and_subentity_arrays(monkeypatch):
    entity_graph, source_dataset, reindexing, variables = _install_fake_package_hierarchy(
        monkeypatch
    )
    VariableCloner = variables.VariableCloner
    VariableExportPolicy = variables.VariableExportPolicy

    snapshot = _make_snapshot(source_dataset, entity_graph)
    reindexed = _make_reindexed(reindexing)
    payload = VariableCloner().clone(snapshot, reindexed, VariableExportPolicy())

    np.testing.assert_array_equal(
        payload.variables["household_income"][2024],
        np.asarray([200.0, 100.0]),
    )
    np.testing.assert_array_equal(
        payload.variables["person_age"][2024],
        np.asarray([50, 18, 17, 34, 35]),
    )
    np.testing.assert_array_equal(
        payload.variables["tax_unit_amount"][2024],
        np.asarray([20.0, 30.0, 10.0]),
    )


def test_variable_cloner_handles_multiple_periods(monkeypatch):
    entity_graph, source_dataset, reindexing, variables = _install_fake_package_hierarchy(
        monkeypatch
    )
    VariableCloner = variables.VariableCloner
    VariableExportPolicy = variables.VariableExportPolicy

    snapshot = _make_snapshot(source_dataset, entity_graph)
    reindexed = _make_reindexed(reindexing)
    payload = VariableCloner().clone(snapshot, reindexed, VariableExportPolicy())

    np.testing.assert_array_equal(
        payload.variables["two_period_var"][2023],
        np.asarray([2.0, 1.0]),
    )
    np.testing.assert_array_equal(
        payload.variables["two_period_var"][2024],
        np.asarray([4.0, 3.0]),
    )


def test_variable_cloner_encodes_enum_and_county_fips_values(monkeypatch):
    entity_graph, source_dataset, reindexing, variables = _install_fake_package_hierarchy(
        monkeypatch
    )
    VariableCloner = variables.VariableCloner
    VariableExportPolicy = variables.VariableExportPolicy

    snapshot = _make_snapshot(source_dataset, entity_graph)
    reindexed = _make_reindexed(reindexing)
    payload = VariableCloner().clone(snapshot, reindexed, VariableExportPolicy())

    assert payload.variables["enum_status"][2024].dtype.kind == "S"
    np.testing.assert_array_equal(
        payload.variables["enum_status"][2024],
        np.asarray([b"B", b"A"]),
    )
    assert payload.variables["county_fips"][2024].dtype == np.int32
    np.testing.assert_array_equal(
        payload.variables["county_fips"][2024],
        np.asarray([36061, 6001], dtype=np.int32),
    )


def test_variable_cloner_respects_excluded_variables(monkeypatch):
    entity_graph, source_dataset, reindexing, variables = _install_fake_package_hierarchy(
        monkeypatch
    )
    VariableCloner = variables.VariableCloner
    VariableExportPolicy = variables.VariableExportPolicy

    snapshot = _make_snapshot(source_dataset, entity_graph)
    reindexed = _make_reindexed(reindexing)
    payload = VariableCloner().clone(
        snapshot,
        reindexed,
        VariableExportPolicy(excluded_variables=frozenset({"household_income"})),
    )

    assert "household_income" not in payload.variables
    assert "person_age" in payload.variables


def test_variable_cloner_respects_required_variables_when_input_variables_disabled(
    monkeypatch,
):
    entity_graph, source_dataset, reindexing, variables = _install_fake_package_hierarchy(
        monkeypatch
    )
    VariableCloner = variables.VariableCloner
    VariableExportPolicy = variables.VariableExportPolicy

    snapshot = _make_snapshot(source_dataset, entity_graph)
    reindexed = _make_reindexed(reindexing)
    payload = VariableCloner().clone(
        snapshot,
        reindexed,
        VariableExportPolicy(
            include_input_variables=False,
            required_variables=frozenset({"person_age"}),
        ),
    )

    assert set(payload.variables) == {"person_age"}


def test_variable_cloner_skips_variables_for_uncloned_entities(monkeypatch):
    entity_graph, source_dataset, reindexing, variables = _install_fake_package_hierarchy(
        monkeypatch
    )
    VariableCloner = variables.VariableCloner
    VariableExportPolicy = variables.VariableExportPolicy

    snapshot = _make_snapshot(source_dataset, entity_graph)
    reindexed = _make_reindexed(reindexing)
    payload = VariableCloner().clone(snapshot, reindexed, VariableExportPolicy())

    assert "ignored_output_only" not in payload.variables
