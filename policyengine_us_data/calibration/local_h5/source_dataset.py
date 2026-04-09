"""Worker-scoped source dataset loading with lazy variable access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .entity_graph import EntityGraph, EntityGraphExtractor


class VariableArrayProvider(Protocol):
    def list_variables(self) -> tuple[str, ...]: ...

    def get_known_periods(self, variable: str) -> tuple[int | str, ...]: ...

    def get_array(self, variable: str, period: int | str) -> np.ndarray: ...

    def get_variable_definition(self, variable: str) -> Any: ...

    def calculate(self, variable: str, *, map_to: str | None = None) -> Any: ...


class PolicyEngineVariableArrayProvider:
    """Lazy access to source arrays through a single Microsimulation."""

    def __init__(self, simulation: Any):
        self.simulation = simulation
        self._holder_cache: dict[str, Any] = {}

    def list_variables(self) -> tuple[str, ...]:
        return tuple(self.simulation.tax_benefit_system.variables.keys())

    def get_known_periods(self, variable: str) -> tuple[int | str, ...]:
        return tuple(self._get_holder(variable).get_known_periods())

    def get_array(self, variable: str, period: int | str) -> np.ndarray:
        return self._get_holder(variable).get_array(period)

    def get_variable_definition(self, variable: str) -> Any:
        return self.simulation.tax_benefit_system.variables.get(variable)

    def calculate(self, variable: str, *, map_to: str | None = None) -> Any:
        if map_to is None:
            return self.simulation.calculate(variable)
        return self.simulation.calculate(variable, map_to=map_to)

    def _get_holder(self, variable: str) -> Any:
        holder = self._holder_cache.get(variable)
        if holder is None:
            holder = self.simulation.get_holder(variable)
            self._holder_cache[variable] = holder
        return holder


@dataclass(frozen=True)
class SourceDatasetSnapshot:
    dataset_path: Path
    time_period: int
    household_ids: np.ndarray
    entity_graph: EntityGraph
    input_variables: frozenset[str]
    variable_provider: VariableArrayProvider

    @property
    def n_households(self) -> int:
        return int(len(self.household_ids))


class PolicyEngineDatasetReader:
    """Load worker-scoped source dataset structure once."""

    def __init__(self, sub_entities: tuple[str, ...]):
        self.sub_entities = tuple(sub_entities)
        self.entity_graph_extractor = EntityGraphExtractor(self.sub_entities)

    def load(self, dataset_path: str | Path) -> SourceDatasetSnapshot:
        from policyengine_us import Microsimulation

        dataset_path = Path(dataset_path)
        simulation = Microsimulation(dataset=str(dataset_path))
        household_ids = np.asarray(
            simulation.calculate("household_id", map_to="household").values
        )
        entity_graph = self.entity_graph_extractor.extract(simulation, household_ids)
        variable_provider = PolicyEngineVariableArrayProvider(simulation)

        return SourceDatasetSnapshot(
            dataset_path=dataset_path,
            time_period=int(simulation.default_calculation_period),
            household_ids=household_ids,
            entity_graph=entity_graph,
            input_variables=frozenset(simulation.input_variables),
            variable_provider=variable_provider,
        )
