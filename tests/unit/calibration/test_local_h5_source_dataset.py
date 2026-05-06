import numpy as np
import pytest

from tests.unit.calibration.fixtures.test_local_h5_source_dataset import (
    FakeHolder,
    FakeSimulation,
    load_source_dataset_exports,
    make_entity_graph_arrays,
)


exports = load_source_dataset_exports()
EntityGraph = exports["EntityGraph"]
MicrosimulationVariableProvider = exports["MicrosimulationVariableProvider"]


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


def test_variable_provider_does_not_load_variables_on_construction():
    simulation = FakeSimulation(
        {
            "household_id": FakeHolder({2023: np.array([1, 2])}),
            "age": FakeHolder({2023: np.array([40, 12, 8])}),
        }
    )

    provider = MicrosimulationVariableProvider(simulation)

    assert provider.input_variables == frozenset({"household_id", "age"})
    assert simulation.get_holder_calls == []
    assert simulation.holders["household_id"].get_array_calls == []
    assert simulation.holders["age"].get_array_calls == []


def test_variable_provider_loads_and_caches_requested_array():
    holder = FakeHolder({2023: np.array([40, 12, 8])})
    simulation = FakeSimulation({"age": holder})
    provider = MicrosimulationVariableProvider(simulation)

    first = provider.get_array("age", 2023)
    second = provider.get_array("age", 2023)

    assert np.array_equal(first, np.array([40, 12, 8]))
    assert first is second
    assert simulation.get_holder_calls == ["age", "age"]
    assert holder.get_array_calls == [2023]
    with pytest.raises(ValueError, match="read-only"):
        first[0] = 99


def test_variable_provider_uses_first_known_period_when_period_is_omitted():
    holder = FakeHolder({2023: np.array([1, 2]), 2024: np.array([3, 4])})
    provider = MicrosimulationVariableProvider(FakeSimulation({"household_id": holder}))

    values = provider.get_array("household_id")

    assert np.array_equal(values, np.array([1, 2]))
    assert holder.known_period_calls == 1
    assert holder.get_array_calls == [2023]


def test_variable_provider_rejects_missing_variables():
    provider = MicrosimulationVariableProvider(FakeSimulation({}))

    with pytest.raises(KeyError, match="missing"):
        provider.get_array("missing", 2023)


def test_variable_provider_rejects_variables_without_periods():
    provider = MicrosimulationVariableProvider(FakeSimulation({"age": FakeHolder({})}))

    with pytest.raises(ValueError, match="no known periods"):
        provider.get_array("age")
