"""Source entity-relationship extraction for local H5 publishing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EntityGraph:
    """Static source entity relationships derived from the base dataset."""

    household_ids: np.ndarray
    person_household_ids: np.ndarray
    hh_id_to_index: Mapping[int, int]
    hh_to_persons: Mapping[int, tuple[int, ...]]
    entity_id_arrays: Mapping[str, np.ndarray]
    person_entity_id_arrays: Mapping[str, np.ndarray]
    hh_to_entity: Mapping[str, Mapping[int, tuple[int, ...]]]


class EntityGraphExtractor:
    """Build source entity-relationship maps from source arrays."""

    def __init__(self, sub_entities: Sequence[str]):
        self.sub_entities = tuple(sub_entities)

    def extract(self, simulation: Any, household_ids: np.ndarray) -> EntityGraph:
        person_household_ids = np.asarray(
            simulation.calculate("household_id", map_to="person").values
        )
        entity_id_arrays = {
            entity_key: np.asarray(
                simulation.calculate(f"{entity_key}_id", map_to=entity_key).values
            )
            for entity_key in self.sub_entities
        }
        person_entity_id_arrays = {
            entity_key: np.asarray(
                simulation.calculate(
                    f"person_{entity_key}_id",
                    map_to="person",
                ).values
            )
            for entity_key in self.sub_entities
        }
        return self.extract_from_arrays(
            household_ids=np.asarray(household_ids),
            person_household_ids=person_household_ids,
            entity_id_arrays=entity_id_arrays,
            person_entity_id_arrays=person_entity_id_arrays,
        )

    def extract_from_arrays(
        self,
        *,
        household_ids: np.ndarray,
        person_household_ids: np.ndarray,
        entity_id_arrays: Mapping[str, np.ndarray],
        person_entity_id_arrays: Mapping[str, np.ndarray],
    ) -> EntityGraph:
        household_ids = np.asarray(household_ids)
        person_household_ids = np.asarray(person_household_ids)

        hh_id_to_index = {int(hid): idx for idx, hid in enumerate(household_ids)}

        hh_to_persons_lists: dict[int, list[int]] = defaultdict(list)
        for person_idx, household_id in enumerate(person_household_ids):
            hh_to_persons_lists[hh_id_to_index[int(household_id)]].append(person_idx)
        hh_to_persons = {
            hh_idx: tuple(person_indices)
            for hh_idx, person_indices in hh_to_persons_lists.items()
        }

        hh_to_entity: dict[str, dict[int, tuple[int, ...]]] = {}
        normalized_entity_id_arrays = {
            entity_key: np.asarray(entity_values)
            for entity_key, entity_values in entity_id_arrays.items()
        }
        normalized_person_entity_id_arrays = {
            entity_key: np.asarray(entity_values)
            for entity_key, entity_values in person_entity_id_arrays.items()
        }

        for entity_key in self.sub_entities:
            entity_ids = normalized_entity_id_arrays[entity_key]
            person_entity_ids = normalized_person_entity_id_arrays[entity_key]
            entity_id_to_index = {
                int(entity_id): entity_idx
                for entity_idx, entity_id in enumerate(entity_ids)
            }

            mapping_lists: dict[int, list[int]] = defaultdict(list)
            seen: dict[int, set[int]] = defaultdict(set)
            for person_idx, household_id in enumerate(person_household_ids):
                hh_idx = hh_id_to_index[int(household_id)]
                entity_idx = entity_id_to_index[int(person_entity_ids[person_idx])]
                if entity_idx not in seen[hh_idx]:
                    seen[hh_idx].add(entity_idx)
                    mapping_lists[hh_idx].append(entity_idx)

            hh_to_entity[entity_key] = {
                hh_idx: tuple(sorted(entity_indices))
                for hh_idx, entity_indices in mapping_lists.items()
            }

        return EntityGraph(
            household_ids=household_ids,
            person_household_ids=person_household_ids,
            hh_id_to_index=hh_id_to_index,
            hh_to_persons=hh_to_persons,
            entity_id_arrays=normalized_entity_id_arrays,
            person_entity_id_arrays=normalized_person_entity_id_arrays,
            hh_to_entity=hh_to_entity,
        )
