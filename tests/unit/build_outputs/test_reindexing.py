from pathlib import Path

import numpy as np
import pytest

from policyengine_us_data.build_outputs.reindexing import (
    EntityReindexer,
    ReindexedEntities,
)
from policyengine_us_data.build_outputs.selection import CloneSelection
from policyengine_us_data.build_outputs.source_dataset import (
    EntityGraph,
    SourceDatasetSnapshot,
)
from tests.support.build_outputs.source_dataset import (
    FakeSimulation,
    make_entity_graph_arrays,
)


def _source_snapshot() -> SourceDatasetSnapshot:
    return SourceDatasetSnapshot(
        dataset_path=Path("source.h5"),
        time_period=2024,
        entity_graph=EntityGraph(**make_entity_graph_arrays()),
        input_variables=frozenset(),
        variable_provider=FakeSimulation({}),
    )


def _selection() -> CloneSelection:
    return CloneSelection(
        clone_indices=np.array([0, 1, 1]),
        source_household_indices=np.array([1, 0, 1]),
        weights=np.array([2.0, 3.0, 4.0]),
        block_geoids=np.array(["block-1", "block-2", "block-3"]),
        congressional_district_geoids=np.array(["3701", "3701", "3702"]),
        filters=(),
        n_source_households=2,
        n_total_clones=2,
    )


def _selection_with_source_indices(indices: np.ndarray) -> CloneSelection:
    return CloneSelection(
        clone_indices=np.arange(len(indices)),
        source_household_indices=indices,
        weights=np.ones(len(indices)),
        block_geoids=np.array([f"block-{i}" for i in range(len(indices))]),
        congressional_district_geoids=np.array(["3701"] * len(indices)),
        filters=(),
        n_source_households=2,
        n_total_clones=max(len(indices), 1),
    )


def test_entity_reindexer_builds_sequential_entity_ids_for_repeated_households():
    result = EntityReindexer().reindex(
        source=_source_snapshot(),
        selection=_selection(),
    )

    np.testing.assert_array_equal(result.household_ids, np.array([0, 1, 2]))
    np.testing.assert_array_equal(result.person_ids, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(
        result.person_household_ids,
        np.array([0, 1, 1, 2]),
    )
    np.testing.assert_array_equal(
        result.household_source_indices,
        np.array([1, 0, 1]),
    )
    np.testing.assert_array_equal(
        result.person_source_indices,
        np.array([2, 0, 1, 2]),
    )
    np.testing.assert_array_equal(
        result.persons_per_household_clone,
        np.array([1, 2, 1]),
    )


def test_entity_reindexer_reindexes_person_subentity_links_per_clone():
    result = EntityReindexer().reindex(
        source=_source_snapshot(),
        selection=_selection(),
    )

    np.testing.assert_array_equal(result.subentity_ids["tax_unit"], np.array([0, 1, 2]))
    np.testing.assert_array_equal(
        result.subentity_source_indices["tax_unit"],
        np.array([1, 0, 1]),
    )
    np.testing.assert_array_equal(
        result.person_subentity_ids["tax_unit"],
        np.array([0, 1, 1, 2]),
    )
    np.testing.assert_array_equal(
        result.subentity_household_clone_indices["tax_unit"],
        np.array([0, 1, 2]),
    )


def test_entity_reindexer_outputs_are_read_only():
    result = EntityReindexer().reindex(
        source=_source_snapshot(),
        selection=_selection(),
    )

    with pytest.raises(ValueError, match="read-only"):
        result.person_source_indices[0] = 99
    with pytest.raises(TypeError):
        result.subentity_source_indices["tax_unit"] = np.array([])


def test_entity_reindexer_rejects_source_household_count_mismatch():
    selection = _selection_with_source_indices(np.array([0]))
    object.__setattr__(selection, "n_source_households", 99)

    with pytest.raises(ValueError, match="source household count"):
        EntityReindexer().reindex(source=_source_snapshot(), selection=selection)


def test_entity_reindexer_rejects_out_of_bounds_source_indices():
    selection = _selection_with_source_indices(np.array([2]))

    with pytest.raises(IndexError, match="out of bounds"):
        EntityReindexer().reindex(source=_source_snapshot(), selection=selection)


def test_reindexed_entities_rejects_inconsistent_subentity_keys():
    with pytest.raises(ValueError, match="person_subentity_ids keys"):
        ReindexedEntities(
            household_ids=np.array([0]),
            person_ids=np.array([0]),
            person_household_ids=np.array([0]),
            subentity_ids={"tax_unit": np.array([0])},
            person_subentity_ids={"spm_unit": np.array([0])},
            household_source_indices=np.array([0]),
            person_source_indices=np.array([0]),
            subentity_source_indices={"tax_unit": np.array([0])},
            persons_per_household_clone=np.array([1]),
            subentities_per_household_clone={"tax_unit": np.array([1])},
            person_household_clone_indices=np.array([0]),
            subentity_household_clone_indices={"tax_unit": np.array([0])},
        )


def test_reindexed_entities_rejects_inconsistent_person_counts():
    with pytest.raises(ValueError, match="persons_per_household_clone"):
        ReindexedEntities(
            household_ids=np.array([0]),
            person_ids=np.array([0, 1]),
            person_household_ids=np.array([0, 0]),
            subentity_ids={"tax_unit": np.array([0])},
            person_subentity_ids={"tax_unit": np.array([0, 0])},
            household_source_indices=np.array([0]),
            person_source_indices=np.array([0, 1]),
            subentity_source_indices={"tax_unit": np.array([0])},
            persons_per_household_clone=np.array([1]),
            subentities_per_household_clone={"tax_unit": np.array([1])},
            person_household_clone_indices=np.array([0, 0]),
            subentity_household_clone_indices={"tax_unit": np.array([0])},
        )
