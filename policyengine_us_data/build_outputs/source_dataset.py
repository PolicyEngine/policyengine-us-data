"""Source dataset contracts for local H5 publication.

This module defines the in-memory source dataset structures used by later
local H5 migration slices. It is intentionally pure at this stage: current
workers may still reconstruct source state directly from raw dataset paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike

from policyengine_us_data.pipeline_metadata import pipeline_node

from .simulation_access import (
    calculate_variable_values,
    get_default_calculation_period,
    get_holder_array,
    get_input_variables,
    get_known_periods,
    get_variable_entity_key,
    get_variable_names,
    get_variable_value_type,
)

__all__ = [
    "EntityGraph",
    "MicrosimulationVariableProvider",
    "PolicyEngineDatasetReader",
    "SourceVariableMetadata",
    "SourceDatasetSnapshot",
]


DEFAULT_SUBENTITIES = ("tax_unit", "spm_unit", "family", "marital_unit")


@dataclass(frozen=True)
class SourceVariableMetadata:
    """Entity and value metadata for one source variable."""

    name: str
    entity_key: str
    value_type: object


@dataclass(frozen=True)
class _EntityMapsView:
    time_period: int
    household_ids: np.ndarray
    person_hh_ids: np.ndarray
    hh_to_persons: dict[int, list[int]]
    hh_to_entity: dict[str, dict[int, list[int]]]
    entity_id_arrays: dict[str, np.ndarray]
    person_entity_id_arrays: dict[str, np.ndarray]


@pipeline_node(
    id="local_h5_entity_graph",
    label="EntityGraph",
    node_type="library",
    description=("Explicit source-dataset entity spine for local H5 worker setup."),
    source_file="policyengine_us_data/build_outputs/source_dataset.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_source_dataset.py"
    ],
)
@dataclass(frozen=True)
class EntityGraph:
    """Structural relationships between source dataset entities.

    `EntityGraph` is the canonical in-memory shape for the source entity spine.
    It stores raw ID arrays plus derived household-membership maps so later H5
    builder seams can select households without rebuilding joins.

    Attributes:
        household_ids: Source household IDs, one per household row.
        person_household_ids: Household ID for each source person row.
        subentity_ids: Entity IDs by subentity key, such as `tax_unit`.
        person_subentity_ids: Entity ID for each source person row by subentity.
        household_to_person_indices: Household row index to person row indices.
        household_to_subentity_indices: Subentity key to household row index to
            subentity row indices.
    """

    household_ids: np.ndarray
    person_household_ids: np.ndarray
    subentity_ids: Mapping[str, np.ndarray]
    person_subentity_ids: Mapping[str, np.ndarray]
    household_to_person_indices: Mapping[int, tuple[int, ...]] = field(init=False)
    household_to_subentity_indices: Mapping[
        str,
        Mapping[int, tuple[int, ...]],
    ] = field(init=False)

    def __post_init__(self) -> None:
        household_ids = _readonly_1d_array(self.household_ids, "household_ids")
        person_household_ids = _readonly_1d_array(
            self.person_household_ids,
            "person_household_ids",
        )
        subentity_ids = _readonly_array_mapping(self.subentity_ids, "subentity_ids")
        person_subentity_ids = _readonly_array_mapping(
            self.person_subentity_ids,
            "person_subentity_ids",
        )

        if set(subentity_ids) != set(person_subentity_ids):
            raise ValueError(
                "subentity_ids and person_subentity_ids must have matching keys"
            )
        for entity_key, person_entity_ids in person_subentity_ids.items():
            if person_entity_ids.shape[0] != person_household_ids.shape[0]:
                raise ValueError(
                    f"person_subentity_ids[{entity_key!r}] length "
                    "must equal person_household_ids length"
                )

        household_to_person_indices = _build_household_to_person_indices(
            household_ids,
            person_household_ids,
        )
        household_to_subentity_indices = _build_household_to_subentity_indices(
            household_ids=household_ids,
            person_household_ids=person_household_ids,
            subentity_ids=subentity_ids,
            person_subentity_ids=person_subentity_ids,
        )

        object.__setattr__(self, "household_ids", household_ids)
        object.__setattr__(self, "person_household_ids", person_household_ids)
        object.__setattr__(self, "subentity_ids", subentity_ids)
        object.__setattr__(self, "person_subentity_ids", person_subentity_ids)
        object.__setattr__(
            self,
            "household_to_person_indices",
            _readonly_index_mapping(household_to_person_indices),
        )
        object.__setattr__(
            self,
            "household_to_subentity_indices",
            _readonly_nested_index_mapping(household_to_subentity_indices),
        )

    @classmethod
    def from_simulation(
        cls,
        simulation,
        *,
        subentities: tuple[str, ...] = DEFAULT_SUBENTITIES,
    ) -> "EntityGraph":
        """Build an `EntityGraph` from a `policyengine_us.Microsimulation`.

        Args:
            simulation: Source dataset simulation.
            subentities: Subentity keys to include.

        Returns:
            A validated `EntityGraph`.
        """

        household_ids = calculate_variable_values(
            simulation,
            "household_id",
            map_to="household",
        )
        person_household_ids = calculate_variable_values(
            simulation,
            "household_id",
            map_to="person",
        )
        subentity_ids = {}
        person_subentity_ids = {}
        for entity_key in subentities:
            subentity_ids[entity_key] = calculate_variable_values(
                simulation,
                f"{entity_key}_id",
                map_to=entity_key,
            )
            person_subentity_ids[entity_key] = calculate_variable_values(
                simulation,
                f"person_{entity_key}_id",
                map_to="person",
            )
        return cls(
            household_ids=household_ids,
            person_household_ids=person_household_ids,
            subentity_ids=subentity_ids,
            person_subentity_ids=person_subentity_ids,
        )

    @classmethod
    def from_entity_maps(cls, entity_maps) -> "EntityGraph":
        """Build an `EntityGraph` from existing `entity_clone.EntityMaps`.

        Args:
            entity_maps: Current compatibility map object.

        Returns:
            A validated `EntityGraph`.
        """

        return cls(
            household_ids=entity_maps.household_ids,
            person_household_ids=entity_maps.person_hh_ids,
            subentity_ids=entity_maps.entity_id_arrays,
            person_subentity_ids=entity_maps.person_entity_id_arrays,
        )

    def to_entity_maps(self, time_period: int):
        """Convert to an `entity_clone.EntityMaps`-compatible object.

        Args:
            time_period: Dataset period to store on the compatibility object.

        Returns:
            An object with the same attributes consumed by existing
            `entity_clone` helpers.
        """

        return _EntityMapsView(
            time_period=int(time_period),
            household_ids=self.household_ids,
            person_hh_ids=self.person_household_ids,
            hh_to_persons={
                household_index: list(person_indices)
                for household_index, person_indices in (
                    self.household_to_person_indices or {}
                ).items()
            },
            hh_to_entity={
                entity_key: {
                    household_index: list(entity_indices)
                    for household_index, entity_indices in membership.items()
                }
                for entity_key, membership in (
                    self.household_to_subentity_indices or {}
                ).items()
            },
            entity_id_arrays=dict(self.subentity_ids),
            person_entity_id_arrays=dict(self.person_subentity_ids),
        )


def _readonly_1d_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    normalized = np.array(array, copy=True)
    normalized.setflags(write=False)
    return normalized


def _readonly_array_mapping(
    values: Mapping[str, ArrayLike],
    name: str,
) -> Mapping[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        normalized[key] = _readonly_1d_array(value, f"{name}[{key!r}]")
    return MappingProxyType(normalized)


def _readonly_index_mapping(
    values: Mapping[int, tuple[int, ...]],
) -> Mapping[int, tuple[int, ...]]:
    return MappingProxyType(dict(values))


def _readonly_nested_index_mapping(
    values: Mapping[str, Mapping[int, tuple[int, ...]]],
) -> Mapping[str, Mapping[int, tuple[int, ...]]]:
    return MappingProxyType(
        {
            entity_key: MappingProxyType(dict(membership))
            for entity_key, membership in values.items()
        }
    )


def _build_household_to_person_indices(
    household_ids: np.ndarray,
    person_household_ids: np.ndarray,
) -> dict[int, tuple[int, ...]]:
    household_id_to_index = _unique_id_index(household_ids, "household_ids")
    membership: dict[int, list[int]] = {
        household_index: [] for household_index in range(len(household_ids))
    }
    for person_index, household_id in enumerate(person_household_ids):
        household_index = household_id_to_index.get(_normalized_id(household_id))
        if household_index is None:
            raise ValueError(
                "person_household_ids contains an ID not present in household_ids"
            )
        membership[household_index].append(person_index)
    return {
        household_index: tuple(person_indices)
        for household_index, person_indices in membership.items()
    }


def _build_household_to_subentity_indices(
    *,
    household_ids: np.ndarray,
    person_household_ids: np.ndarray,
    subentity_ids: Mapping[str, np.ndarray],
    person_subentity_ids: Mapping[str, np.ndarray],
) -> dict[str, dict[int, tuple[int, ...]]]:
    household_id_to_index = _unique_id_index(household_ids, "household_ids")
    all_memberships: dict[str, dict[int, tuple[int, ...]]] = {}

    for entity_key, entity_ids in subentity_ids.items():
        entity_id_to_index = _unique_id_index(
            entity_ids,
            f"subentity_ids[{entity_key!r}]",
        )
        membership_sets: dict[int, set[int]] = {
            household_index: set() for household_index in range(len(household_ids))
        }
        for person_index, person_entity_id in enumerate(
            person_subentity_ids[entity_key]
        ):
            household_index = household_id_to_index.get(
                _normalized_id(person_household_ids[person_index])
            )
            if household_index is None:
                raise ValueError(
                    "person_household_ids contains an ID not present in household_ids"
                )
            entity_index = entity_id_to_index.get(_normalized_id(person_entity_id))
            if entity_index is None:
                raise ValueError(
                    f"person_subentity_ids[{entity_key!r}] contains an ID "
                    f"not present in subentity_ids[{entity_key!r}]"
                )
            membership_sets[household_index].add(entity_index)
        all_memberships[entity_key] = {
            household_index: tuple(sorted(entity_indices))
            for household_index, entity_indices in membership_sets.items()
        }
    return all_memberships


def _unique_id_index(values: np.ndarray, name: str) -> dict[object, int]:
    index: dict[object, int] = {}
    for row_index, value in enumerate(values):
        normalized = _normalized_id(value)
        if normalized in index:
            raise ValueError(f"{name} must contain unique IDs")
        index[normalized] = row_index
    return index


def _normalized_id(value) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


@pipeline_node(
    id="local_h5_microsimulation_variable_provider",
    label="MicrosimulationVariableProvider",
    node_type="library",
    description=("Lazy source variable access wrapper for local H5 source snapshots."),
    source_file="policyengine_us_data/build_outputs/source_dataset.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_source_dataset.py"
    ],
)
@dataclass
class MicrosimulationVariableProvider:
    """Lazy holder-backed variable reader for a source microsimulation.

    The provider intentionally reads arrays only when callers request a
    variable/period pair. It caches the normalized array for repeated access
    while keeping construction lightweight.

    Attributes:
        simulation: Source `policyengine_us.Microsimulation` or a compatible
            test double with `input_variables` and `get_holder(...)`.
    """

    simulation: Any
    _array_cache: dict[tuple[str, str], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def input_variables(self) -> frozenset[str]:
        """Return the source simulation input variable inventory."""

        return get_input_variables(self.simulation)

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return variables declared by the source tax-benefit system."""

        return get_variable_names(self.simulation)

    def get_metadata(self, variable: str) -> SourceVariableMetadata:
        """Return entity and value metadata for one source variable."""

        return SourceVariableMetadata(
            name=str(variable),
            entity_key=get_variable_entity_key(self.simulation, variable),
            value_type=get_variable_value_type(self.simulation, variable),
        )

    def known_periods(self, variable: str) -> tuple[Any, ...]:
        """Return periods known to the source holder for `variable`.

        Args:
            variable: Variable name to inspect.

        Returns:
            Tuple of holder periods.
        """

        return get_known_periods(self.simulation, variable)

    def get_array(self, variable: str, period: Any | None = None) -> np.ndarray:
        """Return one source variable array, loading and caching it lazily.

        Args:
            variable: Variable name to load.
            period: Holder period. If omitted, the first known period is used.

        Returns:
            A read-only numpy array copy of the holder values.
        """

        if period is None:
            periods = get_known_periods(self.simulation, variable)
            if not periods:
                raise ValueError(f"Variable {variable!r} has no known periods")
            period = periods[0]

        cache_key = (str(variable), str(period))
        if cache_key not in self._array_cache:
            array = np.array(
                get_holder_array(self.simulation, variable, period),
                copy=True,
            )
            array.setflags(write=False)
            self._array_cache[cache_key] = array
        return self._array_cache[cache_key]

    def get_raw_array(self, variable: str, period: Any) -> Any:
        """Return one holder array without normalizing the backing object."""

        return get_holder_array(self.simulation, variable, period)


@pipeline_node(
    id="local_h5_source_dataset_snapshot",
    label="SourceDatasetSnapshot",
    node_type="library",
    description=("In-memory source H5 dataset contract for local H5 worker setup."),
    source_file="policyengine_us_data/build_outputs/source_dataset.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_source_dataset.py"
    ],
)
@dataclass
class SourceDatasetSnapshot:
    """Explicit in-memory worker view of a source H5 dataset.

    The snapshot groups source dataset structure and lazy variable access. It
    does not serialize artifacts or change current worker execution paths.

    Attributes:
        dataset_path: Source H5 dataset path.
        time_period: Default source calculation period.
        entity_graph: Source entity relationship graph.
        input_variables: Input variable names available from the source.
        variable_provider: Lazy variable reader for the source simulation.
    """

    dataset_path: Path
    time_period: int
    entity_graph: EntityGraph
    input_variables: frozenset[str]
    variable_provider: MicrosimulationVariableProvider

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.time_period = int(self.time_period)
        self.input_variables = frozenset(str(item) for item in self.input_variables)

    @property
    def household_ids(self) -> np.ndarray:
        """Return source household IDs from the entity graph."""

        return self.entity_graph.household_ids

    @property
    def n_households(self) -> int:
        """Return the number of source households."""

        return int(len(self.household_ids))

    @classmethod
    def from_simulation(
        cls,
        dataset_path: Path,
        simulation,
    ) -> "SourceDatasetSnapshot":
        """Build a snapshot from an existing source simulation.

        Args:
            dataset_path: Source H5 dataset path.
            simulation: Source `Microsimulation` or compatible test double.

        Returns:
            A `SourceDatasetSnapshot` using the provided simulation.
        """

        provider = MicrosimulationVariableProvider(simulation)
        return cls(
            dataset_path=Path(dataset_path),
            time_period=get_default_calculation_period(simulation),
            entity_graph=EntityGraph.from_simulation(simulation),
            input_variables=provider.input_variables,
            variable_provider=provider,
        )


@pipeline_node(
    id="local_h5_policyengine_dataset_reader",
    label="PolicyEngineDatasetReader",
    node_type="library",
    description=("PolicyEngine H5 dataset adapter for local H5 source snapshots."),
    source_file="policyengine_us_data/build_outputs/source_dataset.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/integration/build_outputs/h5_worker_runtime/test_worker_script_tiny_fixture.py"
    ],
)
@dataclass(frozen=True)
class PolicyEngineDatasetReader:
    """Read PolicyEngine source H5 files into `SourceDatasetSnapshot` objects."""

    def load(self, dataset_path: Path) -> SourceDatasetSnapshot:
        """Open a source H5 dataset and return its in-memory snapshot.

        Args:
            dataset_path: Source H5 dataset path.

        Returns:
            A `SourceDatasetSnapshot` backed by a PolicyEngine microsimulation.
        """

        from policyengine_us import Microsimulation

        path = Path(dataset_path)
        simulation = Microsimulation(dataset=str(path))
        return SourceDatasetSnapshot.from_simulation(path, simulation)

    def load_with_entity_graph(
        self,
        dataset_path: Path,
        entity_graph: EntityGraph,
    ) -> SourceDatasetSnapshot:
        """Open a source H5 dataset using a prebuilt structural entity graph.

        Args:
            dataset_path: Source H5 dataset path.
            entity_graph: Persisted structural entity graph for the dataset.

        Returns:
            A `SourceDatasetSnapshot` backed by a PolicyEngine microsimulation
            and the supplied entity graph.
        """

        from policyengine_us import Microsimulation

        path = Path(dataset_path)
        simulation = Microsimulation(dataset=str(path))
        provider = MicrosimulationVariableProvider(simulation)
        return SourceDatasetSnapshot(
            dataset_path=path,
            time_period=get_default_calculation_period(simulation),
            entity_graph=entity_graph,
            input_variables=provider.input_variables,
            variable_provider=provider,
        )
