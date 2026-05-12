"""Fixture helpers for build-output source dataset tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


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

    def __init__(self, holders, *, variable_entities=None, value_types=None):
        self.holders = dict(holders)
        self.input_variables = frozenset(holders)
        self.default_calculation_period = 2023
        self.get_holder_calls = []
        variable_entities = dict(variable_entities or {})
        value_types = dict(value_types or {})
        self.tax_benefit_system = SimpleNamespace(
            variables={
                variable: SimpleNamespace(
                    entity=SimpleNamespace(
                        key=variable_entities.get(
                            variable,
                            _infer_entity_key(variable),
                        )
                    ),
                    value_type=value_types.get(variable, float),
                )
                for variable in holders
            }
        )

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


def _infer_entity_key(variable: str) -> str:
    if variable.startswith("person_") or variable in {"age", "employment_income"}:
        return "person"
    for entity_key in ("tax_unit", "spm_unit", "family", "marital_unit"):
        if variable == f"{entity_key}_id" or variable.startswith(f"{entity_key}_"):
            return entity_key
    return "household"
