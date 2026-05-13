from types import SimpleNamespace

import numpy as np
import pytest

from policyengine_us_data.build_outputs.simulation_access import (
    calculate_variable_values,
    get_default_calculation_period,
    get_holder,
    get_holder_array,
    get_input_variables,
    get_known_periods,
    get_tax_benefit_variables,
    get_variable_definition,
    get_variable_entity_key,
    get_variable_names,
    get_variable_value_type,
)
from tests.support.build_outputs.source_dataset import FakeHolder, FakeSimulation


def test_simulation_access_reads_variable_inventory_and_metadata():
    simulation = FakeSimulation(
        {"rent": FakeHolder({2023: np.array([1, 2])})},
        variable_entities={"rent": "household"},
        value_types={"rent": float},
    )

    assert get_input_variables(simulation) == frozenset({"rent"})
    assert get_variable_names(simulation) == ("rent",)
    assert (
        get_variable_definition(simulation, "rent")
        is (simulation.tax_benefit_system.variables["rent"])
    )
    assert get_variable_entity_key(simulation, "rent") == "household"
    assert get_variable_value_type(simulation, "rent") is float


def test_simulation_access_variable_names_falls_back_to_input_variables():
    simulation = FakeSimulation(
        {
            "household_id": FakeHolder({2023: np.array([1, 2])}),
            "age": FakeHolder({2023: np.array([40, 12, 8])}),
        }
    )
    simulation.tax_benefit_system = None

    assert get_tax_benefit_variables(simulation) is None
    assert get_variable_names(simulation) == ("age", "household_id")


def test_simulation_access_rejects_missing_variable_metadata():
    simulation = FakeSimulation({})

    with pytest.raises(KeyError, match="missing"):
        get_variable_definition(simulation, "missing")


def test_simulation_access_rejects_variable_metadata_without_entity_key():
    simulation = FakeSimulation({"rent": FakeHolder({2023: np.array([1, 2])})})
    simulation.tax_benefit_system.variables["rent"] = SimpleNamespace(
        entity=SimpleNamespace(key=None),
        value_type=float,
    )

    with pytest.raises(ValueError, match="entity key"):
        get_variable_entity_key(simulation, "rent")


def test_simulation_access_reads_holder_periods_and_arrays():
    holder = FakeHolder({2023: np.array([40, 12, 8])})
    simulation = FakeSimulation({"age": holder})

    assert get_holder(simulation, "age") is holder
    assert get_known_periods(simulation, "age") == (2023,)
    np.testing.assert_array_equal(
        get_holder_array(simulation, "age", 2023),
        np.array([40, 12, 8]),
    )
    assert simulation.get_holder_calls == ["age", "age", "age"]


def test_simulation_access_rejects_missing_holders():
    simulation = FakeSimulation({})

    with pytest.raises(KeyError, match="missing"):
        get_holder(simulation, "missing")


def test_simulation_access_reads_default_period_and_calculation_values():
    simulation = FakeSimulation(
        {"household_id": FakeHolder({2023: np.array([10, 20])})}
    )

    assert get_default_calculation_period(simulation) == 2023
    np.testing.assert_array_equal(
        calculate_variable_values(
            simulation,
            "household_id",
            map_to="household",
        ),
        np.array([10, 20]),
    )
