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
    reindexing = _load_module(
        "policyengine_us_data.calibration.local_h5.reindexing",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "reindexing.py",
        ),
    )
    return contracts, selection, entity_graph, source_dataset, reindexing


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
            "spm_unit": np.asarray([900, 901]),
        },
        person_entity_id_arrays={
            "tax_unit": np.asarray([100, 100, 200, 300, 300]),
            "spm_unit": np.asarray([900, 900, 901, 901, 901]),
        },
        hh_to_entity={
            "tax_unit": {0: (0,), 1: (1, 2)},
            "spm_unit": {0: (0,), 1: (1,)},
        },
    )
    return SourceDatasetSnapshot(
        dataset_path=Path("/tmp/source.h5"),
        time_period=2024,
        household_ids=np.asarray([10, 20]),
        entity_graph=graph,
        input_variables=frozenset({"household_id"}),
        variable_provider=types.SimpleNamespace(),
    )


def _make_selection(selection_module, household_indices):
    CloneSelection = selection_module.CloneSelection
    household_indices = np.asarray(household_indices, dtype=np.int64)
    n = len(household_indices)
    return CloneSelection(
        active_clone_indices=np.arange(n, dtype=np.int64),
        active_household_indices=household_indices,
        active_weights=np.ones(n, dtype=float),
        active_block_geoids=np.asarray([f"block-{i}" for i in range(n)], dtype=str),
        active_cd_geoids=np.asarray([f"cd-{i}" for i in range(n)], dtype=str),
        active_county_fips=np.asarray([f"county-{i}" for i in range(n)], dtype=str),
        active_state_fips=np.asarray([i for i in range(n)], dtype=np.int64),
    )


def test_entity_reindexer_assigns_unique_output_ids(monkeypatch):
    _, selection_module, entity_graph_module, source_dataset_module, reindexing = (
        _install_fake_package_hierarchy(monkeypatch)
    )
    EntityReindexer = reindexing.EntityReindexer

    snapshot = _make_snapshot(source_dataset_module, entity_graph_module)
    selection = _make_selection(selection_module, [0, 1])
    result = EntityReindexer().reindex(snapshot, selection)

    np.testing.assert_array_equal(result.new_household_ids, np.asarray([0, 1]))
    np.testing.assert_array_equal(result.new_person_ids, np.asarray([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(
        result.new_entity_ids["tax_unit"],
        np.asarray([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        result.new_entity_ids["spm_unit"],
        np.asarray([0, 1]),
    )


def test_entity_reindexer_maps_people_and_entities_for_repeated_households(
    monkeypatch,
):
    _, selection_module, entity_graph_module, source_dataset_module, reindexing = (
        _install_fake_package_hierarchy(monkeypatch)
    )
    EntityReindexer = reindexing.EntityReindexer

    snapshot = _make_snapshot(source_dataset_module, entity_graph_module)
    selection = _make_selection(selection_module, [0, 0, 1])
    result = EntityReindexer().reindex(snapshot, selection)

    np.testing.assert_array_equal(
        result.persons_per_clone,
        np.asarray([2, 2, 3]),
    )
    np.testing.assert_array_equal(
        result.new_person_household_ids,
        np.asarray([0, 0, 1, 1, 2, 2, 2]),
    )
    np.testing.assert_array_equal(
        result.new_person_entity_ids["tax_unit"],
        np.asarray([0, 0, 1, 1, 2, 3, 3]),
    )
    np.testing.assert_array_equal(
        result.new_person_entity_ids["spm_unit"],
        np.asarray([0, 0, 1, 1, 2, 2, 2]),
    )


def test_entity_reindexer_handles_multiple_entities_within_one_household(monkeypatch):
    _, selection_module, entity_graph_module, source_dataset_module, reindexing = (
        _install_fake_package_hierarchy(monkeypatch)
    )
    EntityReindexer = reindexing.EntityReindexer

    snapshot = _make_snapshot(source_dataset_module, entity_graph_module)
    selection = _make_selection(selection_module, [1])
    result = EntityReindexer().reindex(snapshot, selection)

    np.testing.assert_array_equal(
        result.entities_per_clone["tax_unit"],
        np.asarray([2]),
    )
    np.testing.assert_array_equal(
        result.entity_source_indices["tax_unit"],
        np.asarray([1, 2]),
    )
    np.testing.assert_array_equal(
        result.new_person_entity_ids["tax_unit"],
        np.asarray([0, 1, 1]),
    )


def test_entity_reindexer_handles_empty_selection(monkeypatch):
    _, selection_module, entity_graph_module, source_dataset_module, reindexing = (
        _install_fake_package_hierarchy(monkeypatch)
    )
    EntityReindexer = reindexing.EntityReindexer

    snapshot = _make_snapshot(source_dataset_module, entity_graph_module)
    selection = _make_selection(selection_module, [])
    result = EntityReindexer().reindex(snapshot, selection)

    assert result.new_household_ids.size == 0
    assert result.new_person_ids.size == 0
    assert result.new_person_household_ids.size == 0
    assert result.entity_source_indices["tax_unit"].size == 0
    assert result.new_person_entity_ids["tax_unit"].size == 0
