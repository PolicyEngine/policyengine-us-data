"""Pure entity reindexing for local H5 publishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .selection import CloneSelection
from .source_dataset import SourceDatasetSnapshot


@dataclass(frozen=True)
class ReindexedEntities:
    household_source_indices: np.ndarray
    person_source_indices: np.ndarray
    entity_source_indices: Mapping[str, np.ndarray]
    persons_per_clone: np.ndarray
    entities_per_clone: Mapping[str, np.ndarray]
    new_household_ids: np.ndarray
    new_person_ids: np.ndarray
    new_person_household_ids: np.ndarray
    new_entity_ids: Mapping[str, np.ndarray]
    new_person_entity_ids: Mapping[str, np.ndarray]


class EntityReindexer:
    """Build output IDs and cross-references from a clone selection."""

    def reindex(
        self,
        source: SourceDatasetSnapshot,
        selection: CloneSelection,
    ) -> ReindexedEntities:
        entity_graph = source.entity_graph
        household_source_indices = np.asarray(
            selection.active_household_indices,
            dtype=np.int64,
        )
        n_household_clones = len(household_source_indices)

        persons_per_clone = np.asarray(
            [
                len(entity_graph.hh_to_persons.get(int(household_idx), ()))
                for household_idx in household_source_indices
            ],
            dtype=np.int64,
        )
        person_parts = [
            np.asarray(
                entity_graph.hh_to_persons.get(int(household_idx), ()),
                dtype=np.int64,
            )
            for household_idx in household_source_indices
        ]
        person_source_indices = (
            np.concatenate(person_parts)
            if person_parts
            else np.asarray([], dtype=np.int64)
        )

        entity_source_indices: dict[str, np.ndarray] = {}
        entities_per_clone: dict[str, np.ndarray] = {}
        for entity_key in entity_graph.entity_id_arrays:
            per_clone_counts = np.asarray(
                [
                    len(entity_graph.hh_to_entity[entity_key].get(int(household_idx), ()))
                    for household_idx in household_source_indices
                ],
                dtype=np.int64,
            )
            entities_per_clone[entity_key] = per_clone_counts
            entity_parts = [
                np.asarray(
                    entity_graph.hh_to_entity[entity_key].get(int(household_idx), ()),
                    dtype=np.int64,
                )
                for household_idx in household_source_indices
            ]
            entity_source_indices[entity_key] = (
                np.concatenate(entity_parts)
                if entity_parts
                else np.asarray([], dtype=np.int64)
            )

        n_persons = len(person_source_indices)
        new_household_ids = np.arange(n_household_clones, dtype=np.int32)
        new_person_ids = np.arange(n_persons, dtype=np.int32)
        new_person_household_ids = np.repeat(new_household_ids, persons_per_clone)
        clone_ids_for_persons = np.repeat(
            np.arange(n_household_clones, dtype=np.int64),
            persons_per_clone,
        )

        new_entity_ids: dict[str, np.ndarray] = {}
        new_person_entity_ids: dict[str, np.ndarray] = {}

        for entity_key, source_indices in entity_source_indices.items():
            entity_count = len(source_indices)
            new_entity_ids[entity_key] = np.arange(entity_count, dtype=np.int32)

            if entity_count == 0:
                if n_persons != 0:
                    raise ValueError(
                        f"No source {entity_key} entities for selected persons"
                    )
                new_person_entity_ids[entity_key] = np.asarray([], dtype=np.int32)
                continue

            old_entity_ids = entity_graph.entity_id_arrays[entity_key][
                source_indices
            ].astype(np.int64)
            clone_ids_for_entities = np.repeat(
                np.arange(n_household_clones, dtype=np.int64),
                entities_per_clone[entity_key],
            )

            offset = int(old_entity_ids.max()) + 1 if old_entity_ids.size else 1
            entity_keys = clone_ids_for_entities * offset + old_entity_ids

            sorted_order = np.argsort(entity_keys)
            sorted_keys = entity_keys[sorted_order]
            sorted_new_ids = new_entity_ids[entity_key][sorted_order]

            old_person_entity_ids = entity_graph.person_entity_id_arrays[entity_key][
                person_source_indices
            ].astype(np.int64)
            person_keys = clone_ids_for_persons * offset + old_person_entity_ids

            positions = np.searchsorted(sorted_keys, person_keys)
            if np.any(positions >= len(sorted_keys)):
                raise ValueError(
                    f"Could not map selected persons to new {entity_key} IDs"
                )
            if np.any(sorted_keys[positions] != person_keys):
                raise ValueError(
                    f"Inconsistent selected {entity_key} mappings for persons"
                )
            new_person_entity_ids[entity_key] = sorted_new_ids[positions]

        return ReindexedEntities(
            household_source_indices=household_source_indices,
            person_source_indices=person_source_indices,
            entity_source_indices=entity_source_indices,
            persons_per_clone=persons_per_clone,
            entities_per_clone=entities_per_clone,
            new_household_ids=new_household_ids,
            new_person_ids=new_person_ids,
            new_person_household_ids=new_person_household_ids,
            new_entity_ids=new_entity_ids,
            new_person_entity_ids=new_person_entity_ids,
        )
