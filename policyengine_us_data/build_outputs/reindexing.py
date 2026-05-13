"""Entity reindexing seam for local H5 publication."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from policyengine_us_data.pipeline_metadata import pipeline_node

from .selection import CloneSelection
from .source_dataset import EntityGraph, SourceDatasetSnapshot

__all__ = ["EntityReindexer", "ReindexedEntities"]


@pipeline_node(
    id="local_h5_entity_reindexer",
    label="EntityReindexer",
    node_type="library",
    description="Reindex selected household, person, and subentity rows for local H5 outputs.",
    source_file="policyengine_us_data/build_outputs/reindexing.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_reindexing.py"],
)
class EntityReindexer:
    """Build sequential entity IDs and relationship arrays after clone selection."""

    def reindex(
        self,
        *,
        source: SourceDatasetSnapshot,
        selection: CloneSelection,
    ) -> "ReindexedEntities":
        """Reindex selected source entities into one output entity graph.

        Args:
            source: Source dataset snapshot with entity graph.
            selection: Selected clone-household rows.

        Returns:
            A `ReindexedEntities` object containing output IDs and source row
            indices for variable cloning.
        """

        _validate_selection_for_source(source=source, selection=selection)

        graph = source.entity_graph
        household_source_indices = selection.source_household_indices
        n_household_clones = selection.n_selected_clones

        household_ids = _create_household_ids(n_household_clones)
        person_reindexing = _build_person_reindexing(
            graph=graph,
            household_source_indices=household_source_indices,
            household_ids=household_ids,
        )
        subentity_reindexing = _build_subentity_reindexing(
            graph=graph,
            household_source_indices=household_source_indices,
            person_source_indices=person_reindexing.source_indices,
            person_household_clone_indices=(
                person_reindexing.person_household_clone_indices
            ),
            n_household_clones=n_household_clones,
        )

        return ReindexedEntities(
            household_ids=household_ids,
            person_ids=person_reindexing.ids,
            person_household_ids=person_reindexing.person_household_ids,
            subentity_ids=subentity_reindexing.ids,
            person_subentity_ids=subentity_reindexing.person_subentity_ids,
            household_source_indices=household_source_indices,
            person_source_indices=person_reindexing.source_indices,
            subentity_source_indices=subentity_reindexing.source_indices,
            persons_per_household_clone=person_reindexing.counts,
            subentities_per_household_clone=subentity_reindexing.counts,
            person_household_clone_indices=(
                person_reindexing.person_household_clone_indices
            ),
            subentity_household_clone_indices=(
                subentity_reindexing.subentity_household_clone_indices
            ),
        )


@dataclass(frozen=True)
class _PersonReindexing:
    counts: np.ndarray
    source_indices: np.ndarray
    ids: np.ndarray
    person_household_ids: np.ndarray
    person_household_clone_indices: np.ndarray


@dataclass(frozen=True)
class _SubentityReindexing:
    counts: dict[str, np.ndarray]
    source_indices: dict[str, np.ndarray]
    ids: dict[str, np.ndarray]
    person_subentity_ids: dict[str, np.ndarray]
    subentity_household_clone_indices: dict[str, np.ndarray]


@pipeline_node(
    id="local_h5_reindexed_entities",
    label="ReindexedEntities",
    node_type="library",
    description="Output entity IDs and source row indices after local H5 clone selection.",
    source_file="policyengine_us_data/build_outputs/reindexing.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_reindexing.py"],
)
@dataclass(frozen=True)
class ReindexedEntities:
    """Entity IDs, relationship arrays, and source indices for one H5 output."""

    household_ids: np.ndarray
    person_ids: np.ndarray
    person_household_ids: np.ndarray
    subentity_ids: Mapping[str, np.ndarray]
    person_subentity_ids: Mapping[str, np.ndarray]
    household_source_indices: np.ndarray
    person_source_indices: np.ndarray
    subentity_source_indices: Mapping[str, np.ndarray]
    persons_per_household_clone: np.ndarray
    subentities_per_household_clone: Mapping[str, np.ndarray]
    person_household_clone_indices: np.ndarray
    subentity_household_clone_indices: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "household_ids",
            _readonly_1d_array(self.household_ids, "household_ids"),
        )
        object.__setattr__(
            self,
            "person_ids",
            _readonly_1d_array(self.person_ids, "person_ids"),
        )
        object.__setattr__(
            self,
            "person_household_ids",
            _readonly_1d_array(self.person_household_ids, "person_household_ids"),
        )
        object.__setattr__(
            self,
            "subentity_ids",
            _readonly_array_mapping(self.subentity_ids, "subentity_ids"),
        )
        object.__setattr__(
            self,
            "person_subentity_ids",
            _readonly_array_mapping(
                self.person_subentity_ids,
                "person_subentity_ids",
            ),
        )
        object.__setattr__(
            self,
            "household_source_indices",
            _readonly_1d_array(
                self.household_source_indices,
                "household_source_indices",
            ),
        )
        object.__setattr__(
            self,
            "person_source_indices",
            _readonly_1d_array(self.person_source_indices, "person_source_indices"),
        )
        object.__setattr__(
            self,
            "subentity_source_indices",
            _readonly_array_mapping(
                self.subentity_source_indices,
                "subentity_source_indices",
            ),
        )
        object.__setattr__(
            self,
            "persons_per_household_clone",
            _readonly_1d_array(
                self.persons_per_household_clone,
                "persons_per_household_clone",
            ),
        )
        object.__setattr__(
            self,
            "subentities_per_household_clone",
            _readonly_array_mapping(
                self.subentities_per_household_clone,
                "subentities_per_household_clone",
            ),
        )
        object.__setattr__(
            self,
            "person_household_clone_indices",
            _readonly_1d_array(
                self.person_household_clone_indices,
                "person_household_clone_indices",
            ),
        )
        object.__setattr__(
            self,
            "subentity_household_clone_indices",
            _readonly_array_mapping(
                self.subentity_household_clone_indices,
                "subentity_household_clone_indices",
            ),
        )
        _validate_reindexed_entities(self)


def _create_household_ids(n_household_clones: int) -> np.ndarray:
    return np.arange(n_household_clones, dtype=np.int32)


def _build_person_reindexing(
    *,
    graph: EntityGraph,
    household_source_indices: np.ndarray,
    household_ids: np.ndarray,
) -> _PersonReindexing:
    person_counts = _create_person_counts(
        graph.household_to_person_indices,
        household_source_indices,
    )
    person_source_indices = _concatenate_membership(
        graph.household_to_person_indices,
        household_source_indices,
    )
    return _PersonReindexing(
        counts=person_counts,
        source_indices=person_source_indices,
        ids=_create_entity_ids(len(person_source_indices)),
        person_household_ids=_create_person_household_ids(
            household_ids=household_ids,
            person_counts=person_counts,
        ),
        person_household_clone_indices=_create_household_clone_indices(
            n_household_clones=len(household_ids),
            member_counts=person_counts,
        ),
    )


def _build_subentity_reindexing(
    *,
    graph: EntityGraph,
    household_source_indices: np.ndarray,
    person_source_indices: np.ndarray,
    person_household_clone_indices: np.ndarray,
    n_household_clones: int,
) -> _SubentityReindexing:
    subentity_counts: dict[str, np.ndarray] = {}
    subentity_source_indices: dict[str, np.ndarray] = {}
    subentity_ids: dict[str, np.ndarray] = {}
    person_subentity_ids: dict[str, np.ndarray] = {}
    subentity_household_clone_indices: dict[str, np.ndarray] = {}

    for entity_key, membership in graph.household_to_subentity_indices.items():
        subentity_counts[entity_key] = _create_subentity_counts(
            membership,
            household_source_indices,
        )
        subentity_source_indices[entity_key] = _concatenate_membership(
            membership,
            household_source_indices,
        )
        subentity_ids[entity_key] = _create_entity_ids(
            len(subentity_source_indices[entity_key]),
        )
        subentity_household_clone_indices[entity_key] = _create_household_clone_indices(
            n_household_clones=n_household_clones,
            member_counts=subentity_counts[entity_key],
        )
        person_subentity_ids[entity_key] = _reindex_person_subentity_ids(
            entity_key=entity_key,
            graph_subentity_ids=graph.subentity_ids[entity_key],
            graph_person_subentity_ids=graph.person_subentity_ids[entity_key],
            person_source_indices=person_source_indices,
            entity_source_indices=subentity_source_indices[entity_key],
            new_entity_ids=subentity_ids[entity_key],
            person_household_clone_indices=person_household_clone_indices,
            entity_household_clone_indices=subentity_household_clone_indices[
                entity_key
            ],
        )

    return _SubentityReindexing(
        counts=subentity_counts,
        source_indices=subentity_source_indices,
        ids=subentity_ids,
        person_subentity_ids=person_subentity_ids,
        subentity_household_clone_indices=subentity_household_clone_indices,
    )


def _create_person_counts(
    membership: Mapping[int, tuple[int, ...]],
    household_source_indices: np.ndarray,
) -> np.ndarray:
    return _count_members_per_household(membership, household_source_indices)


def _create_subentity_counts(
    membership: Mapping[int, tuple[int, ...]],
    household_source_indices: np.ndarray,
) -> np.ndarray:
    return _count_members_per_household(membership, household_source_indices)


def _count_members_per_household(
    membership: Mapping[int, tuple[int, ...]],
    household_indices: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            len(membership.get(int(household_index), ()))
            for household_index in household_indices
        ],
        dtype=np.int64,
    )


def _create_entity_ids(count: int) -> np.ndarray:
    return np.arange(count, dtype=np.int32)


def _create_person_household_ids(
    *,
    household_ids: np.ndarray,
    person_counts: np.ndarray,
) -> np.ndarray:
    return np.repeat(household_ids, person_counts).astype(np.int32)


def _create_household_clone_indices(
    *,
    n_household_clones: int,
    member_counts: np.ndarray,
) -> np.ndarray:
    return np.repeat(
        np.arange(n_household_clones, dtype=np.int64),
        member_counts,
    )


def _concatenate_membership(
    membership: Mapping[int, tuple[int, ...]],
    household_indices: np.ndarray,
) -> np.ndarray:
    parts = [
        np.asarray(membership.get(int(household_index), ()), dtype=np.int64)
        for household_index in household_indices
    ]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


def _validate_selection_for_source(
    *,
    source: SourceDatasetSnapshot,
    selection: CloneSelection,
) -> None:
    if selection.n_source_households != source.n_households:
        raise ValueError(
            "CloneSelection source household count does not match source dataset: "
            f"{selection.n_source_households} != {source.n_households}"
        )

    household_indices = selection.source_household_indices
    if np.any(household_indices < 0) or np.any(
        household_indices >= source.n_households
    ):
        raise IndexError("CloneSelection source household indices are out of bounds")


def _validate_reindexed_entities(reindexed: ReindexedEntities) -> None:
    subentity_keys = set(reindexed.subentity_ids)
    for name, keys in {
        "person_subentity_ids": set(reindexed.person_subentity_ids),
        "subentity_source_indices": set(reindexed.subentity_source_indices),
        "subentities_per_household_clone": set(
            reindexed.subentities_per_household_clone
        ),
        "subentity_household_clone_indices": set(
            reindexed.subentity_household_clone_indices
        ),
    }.items():
        if keys != subentity_keys:
            raise ValueError(f"{name} keys must match subentity_ids keys")

    n_households = len(reindexed.household_ids)
    n_persons = len(reindexed.person_ids)
    if not np.array_equal(reindexed.household_ids, np.arange(n_households)):
        raise ValueError("household_ids must be sequential zero-based IDs")
    if not np.array_equal(reindexed.person_ids, np.arange(n_persons)):
        raise ValueError("person_ids must be sequential zero-based IDs")
    if len(reindexed.household_source_indices) != n_households:
        raise ValueError("household_source_indices length must match household_ids")
    if len(reindexed.person_household_ids) != n_persons:
        raise ValueError("person_household_ids length must match person_ids")
    if len(reindexed.person_source_indices) != n_persons:
        raise ValueError("person_source_indices length must match person_ids")
    if len(reindexed.person_household_clone_indices) != n_persons:
        raise ValueError("person_household_clone_indices length must match person_ids")
    if len(reindexed.persons_per_household_clone) != n_households:
        raise ValueError("persons_per_household_clone length must match household_ids")
    if int(np.sum(reindexed.persons_per_household_clone)) != n_persons:
        raise ValueError("persons_per_household_clone must sum to person count")
    _validate_index_bounds(
        reindexed.person_household_ids,
        upper_bound=n_households,
        name="person_household_ids",
    )
    _validate_index_bounds(
        reindexed.person_household_clone_indices,
        upper_bound=n_households,
        name="person_household_clone_indices",
    )

    for entity_key, entity_ids in reindexed.subentity_ids.items():
        n_entities = len(entity_ids)
        if not np.array_equal(entity_ids, np.arange(n_entities)):
            raise ValueError(f"subentity_ids[{entity_key!r}] must be sequential IDs")

        if len(reindexed.subentity_source_indices[entity_key]) != n_entities:
            raise ValueError(
                f"subentity_source_indices[{entity_key!r}] length must match "
                f"subentity_ids[{entity_key!r}]"
            )
        if len(reindexed.subentity_household_clone_indices[entity_key]) != n_entities:
            raise ValueError(
                f"subentity_household_clone_indices[{entity_key!r}] length must "
                f"match subentity_ids[{entity_key!r}]"
            )
        entity_counts = reindexed.subentities_per_household_clone[entity_key]
        if len(entity_counts) != n_households:
            raise ValueError(
                f"subentities_per_household_clone[{entity_key!r}] length must "
                "match household_ids"
            )
        if int(np.sum(entity_counts)) != n_entities:
            raise ValueError(
                f"subentities_per_household_clone[{entity_key!r}] must sum to "
                f"{entity_key} count"
            )
        if len(reindexed.person_subentity_ids[entity_key]) != n_persons:
            raise ValueError(
                f"person_subentity_ids[{entity_key!r}] length must match person_ids"
            )
        _validate_index_bounds(
            reindexed.person_subentity_ids[entity_key],
            upper_bound=n_entities,
            name=f"person_subentity_ids[{entity_key!r}]",
        )
        _validate_index_bounds(
            reindexed.subentity_household_clone_indices[entity_key],
            upper_bound=n_households,
            name=f"subentity_household_clone_indices[{entity_key!r}]",
        )


def _validate_index_bounds(
    values: np.ndarray,
    *,
    upper_bound: int,
    name: str,
) -> None:
    if len(values) == 0:
        return
    if np.any(values < 0) or np.any(values >= upper_bound):
        raise ValueError(f"{name} contains out-of-bounds IDs")


def _reindex_person_subentity_ids(
    *,
    entity_key: str,
    graph_subentity_ids: np.ndarray,
    graph_person_subentity_ids: np.ndarray,
    person_source_indices: np.ndarray,
    entity_source_indices: np.ndarray,
    new_entity_ids: np.ndarray,
    person_household_clone_indices: np.ndarray,
    entity_household_clone_indices: np.ndarray,
) -> np.ndarray:
    old_entity_ids = graph_subentity_ids[entity_source_indices].astype(np.int64)
    person_old_entity_ids = graph_person_subentity_ids[person_source_indices].astype(
        np.int64
    )

    if len(old_entity_ids) == 0:
        if len(person_old_entity_ids) > 0:
            raise ValueError(
                f"Selected persons reference {entity_key} rows, but no "
                f"{entity_key} rows were selected"
            )
        return np.array([], dtype=np.int32)

    offset = int(old_entity_ids.max()) + 1
    entity_keys = entity_household_clone_indices * offset + old_entity_ids
    sorted_order = np.argsort(entity_keys)
    sorted_keys = entity_keys[sorted_order]
    sorted_new_ids = new_entity_ids[sorted_order]

    person_keys = person_household_clone_indices * offset + person_old_entity_ids
    positions = np.searchsorted(sorted_keys, person_keys)
    if len(positions) > 0:
        valid_positions = positions < len(sorted_keys)
        if not np.all(valid_positions) or not np.array_equal(
            sorted_keys[positions[valid_positions]],
            person_keys[valid_positions],
        ):
            raise ValueError(
                f"Selected person rows could not be mapped to cloned {entity_key} IDs"
            )
    return sorted_new_ids[positions].astype(np.int32)


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
