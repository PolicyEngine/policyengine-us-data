"""Accessors for PolicyEngine microsimulation-backed build-output seams."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "calculate_variable_values",
    "get_default_calculation_period",
    "get_holder",
    "get_holder_array",
    "get_input_variables",
    "get_known_periods",
    "get_tax_benefit_variables",
    "get_variable_definition",
    "get_variable_entity_key",
    "get_variable_names",
    "get_variable_value_type",
]


def get_input_variables(simulation: Any) -> frozenset[str]:
    """Return input variable names declared by a microsimulation."""

    try:
        input_variables = simulation.input_variables
    except AttributeError as exc:
        raise AttributeError("Simulation has no input_variables attribute") from exc
    return frozenset(str(variable) for variable in input_variables)


def get_tax_benefit_variables(simulation: Any) -> Any | None:
    """Return tax-benefit variable definitions when available."""

    tax_benefit_system = getattr(simulation, "tax_benefit_system", None)
    return getattr(tax_benefit_system, "variables", None)


def get_variable_names(simulation: Any) -> tuple[str, ...]:
    """Return known variable names, falling back to input variables."""

    variables = get_tax_benefit_variables(simulation)
    if variables is None:
        return tuple(sorted(get_input_variables(simulation)))
    return tuple(str(variable) for variable in variables)


def get_variable_definition(simulation: Any, variable: str) -> Any:
    """Return one tax-benefit variable definition."""

    variables = get_tax_benefit_variables(simulation)
    if variables is None or variable not in variables:
        raise KeyError(f"Variable {variable!r} metadata is not available")
    return variables[variable]


def get_variable_entity_key(simulation: Any, variable: str) -> str:
    """Return the entity key for one tax-benefit variable definition."""

    variable_definition = get_variable_definition(simulation, variable)
    entity = getattr(variable_definition, "entity", None)
    entity_key = getattr(entity, "key", None)
    if not entity_key:
        raise ValueError(f"Variable {variable!r} has no entity key")
    return str(entity_key)


def get_variable_value_type(simulation: Any, variable: str) -> object:
    """Return the declared value type for one tax-benefit variable."""

    return getattr(get_variable_definition(simulation, variable), "value_type", None)


def get_holder(simulation: Any, variable: str) -> Any:
    """Return one variable holder from a microsimulation."""

    try:
        holder = simulation.get_holder(variable)
    except Exception as exc:
        raise KeyError(f"Variable {variable!r} is not available") from exc
    if holder is None:
        raise KeyError(f"Variable {variable!r} is not available")
    return holder


def get_known_periods(simulation: Any, variable: str) -> tuple[Any, ...]:
    """Return known holder periods for one variable."""

    return tuple(get_holder(simulation, variable).get_known_periods())


def get_holder_array(simulation: Any, variable: str, period: Any) -> Any:
    """Return the holder array for one variable and period."""

    return get_holder(simulation, variable).get_array(period)


def get_default_calculation_period(simulation: Any) -> int:
    """Return the default calculation period for a microsimulation."""

    try:
        return int(simulation.default_calculation_period)
    except AttributeError as exc:
        raise AttributeError(
            "Simulation has no default_calculation_period attribute"
        ) from exc


def calculate_variable_values(
    simulation: Any,
    variable: str,
    *,
    period: Any | None = None,
    map_to: str | None = None,
) -> np.ndarray:
    """Calculate one variable and return numpy-like values."""

    kwargs: dict[str, Any] = {}
    if period is not None:
        kwargs["period"] = period
    if map_to is not None:
        kwargs["map_to"] = map_to
    return _calculation_values(simulation.calculate(variable, **kwargs))


def _calculation_values(calculation: Any) -> np.ndarray:
    if hasattr(calculation, "__array__"):
        return np.asarray(calculation)
    if hasattr(calculation, "to_numpy"):
        return calculation.to_numpy()
    if hasattr(calculation, "values"):
        return calculation.values
    return np.asarray(calculation)
