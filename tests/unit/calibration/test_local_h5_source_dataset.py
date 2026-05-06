import numpy as np
import pytest

from tests.unit.calibration.fixtures.test_local_h5_source_dataset import (
    load_source_dataset_exports,
    make_entity_graph_arrays,
)


exports = load_source_dataset_exports()
EntityGraph = exports["EntityGraph"]


def _tuple_map(mapping):
    return {
        int(key): tuple(int(value) for value in values)
        for key, values in mapping.items()
    }


def test_entity_graph_builds_household_membership_maps():
    graph = EntityGraph(**make_entity_graph_arrays())

    assert _tuple_map(graph.household_to_person_indices) == {
        0: (0, 1),
        1: (2,),
    }
    assert _tuple_map(graph.household_to_subentity_indices["tax_unit"]) == {
        0: (0,),
        1: (1,),
    }
    assert _tuple_map(graph.household_to_subentity_indices["spm_unit"]) == {
        0: (0,),
        1: (1,),
    }


def test_entity_graph_copies_arrays_and_exposes_read_only_storage():
    arrays = make_entity_graph_arrays()
    graph = EntityGraph(**arrays)

    arrays["household_ids"][0] = 99

    assert graph.household_ids[0] == 10
    with pytest.raises(ValueError, match="read-only"):
        graph.household_ids[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        graph.subentity_ids["tax_unit"][0] = 99


def test_entity_graph_rejects_unknown_person_household_ids():
    arrays = make_entity_graph_arrays()
    arrays["person_household_ids"] = np.array([10, 999, 20], dtype=np.int64)

    with pytest.raises(ValueError, match="person_household_ids"):
        EntityGraph(**arrays)


def test_entity_graph_rejects_unknown_person_subentity_ids():
    arrays = make_entity_graph_arrays()
    arrays["person_subentity_ids"]["tax_unit"] = np.array(
        [100, 999, 200],
        dtype=np.int64,
    )

    with pytest.raises(ValueError, match="person_subentity_ids\\['tax_unit'\\]"):
        EntityGraph(**arrays)


def test_entity_graph_rejects_mismatched_subentity_keys():
    arrays = make_entity_graph_arrays()
    arrays["person_subentity_ids"].pop("spm_unit")

    with pytest.raises(ValueError, match="matching keys"):
        EntityGraph(**arrays)


def test_entity_graph_rejects_duplicate_ids():
    arrays = make_entity_graph_arrays()
    arrays["household_ids"] = np.array([10, 10], dtype=np.int64)

    with pytest.raises(ValueError, match="household_ids must contain unique IDs"):
        EntityGraph(**arrays)


def test_entity_graph_round_trips_through_entity_maps():
    graph = EntityGraph(**make_entity_graph_arrays())

    entity_maps = graph.to_entity_maps(time_period=2023)
    roundtrip = EntityGraph.from_entity_maps(entity_maps)

    assert entity_maps.time_period == 2023
    assert np.array_equal(roundtrip.household_ids, graph.household_ids)
    assert np.array_equal(roundtrip.person_household_ids, graph.person_household_ids)
    assert _tuple_map(roundtrip.household_to_person_indices) == _tuple_map(
        graph.household_to_person_indices
    )
    assert _tuple_map(
        roundtrip.household_to_subentity_indices["tax_unit"]
    ) == _tuple_map(graph.household_to_subentity_indices["tax_unit"])
