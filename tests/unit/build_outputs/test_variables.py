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
    MicrosimulationVariableProvider,
    SourceDatasetSnapshot,
)
from policyengine_us_data.build_outputs.variables import (
    VariableCloner,
    default_variables_to_save,
)
from tests.support.build_outputs.source_dataset import (
    FakeHolder,
    FakeSimulation,
    make_entity_graph_arrays,
)


def _selection() -> CloneSelection:
    return CloneSelection(
        clone_indices=np.array([0, 1]),
        source_household_indices=np.array([1, 0]),
        weights=np.array([2.0, 3.0]),
        block_geoids=np.array(["block-1", "block-2"]),
        congressional_district_geoids=np.array(["3701", "3702"]),
        filters=(),
        n_source_households=2,
        n_total_clones=2,
    )


def _source_snapshot() -> SourceDatasetSnapshot:
    simulation = FakeSimulation(
        {
            "rent": FakeHolder({2024: np.array([100, 200])}),
            "age": FakeHolder({2024: np.array([10, 20, 30])}),
            "taxable_income": FakeHolder({2024: np.array([1_000, 2_000])}),
            "person_name": FakeHolder({2024: np.array(["a", "b", "c"])}),
            "county_fips": FakeHolder({2024: np.array(["37183", "06037"])}),
            "unsupported": FakeHolder({2024: np.array([1, 2])}),
            "no_period": FakeHolder({}),
        },
        variable_entities={
            "rent": "household",
            "age": "person",
            "taxable_income": "tax_unit",
            "person_name": "person",
            "county_fips": "household",
            "unsupported": "unsupported_entity",
            "no_period": "household",
        },
        value_types={
            "person_name": str,
            "county_fips": str,
        },
    )
    return SourceDatasetSnapshot(
        dataset_path=Path("source.h5"),
        time_period=2024,
        entity_graph=EntityGraph(**make_entity_graph_arrays()),
        input_variables=frozenset(simulation.holders),
        variable_provider=MicrosimulationVariableProvider(simulation),
    )


def test_variable_cloner_clones_by_reindexed_entity_source_indices():
    source = _source_snapshot()
    selection = _selection()
    reindexed = EntityReindexer().reindex(source=source, selection=selection)

    payload = VariableCloner().clone(
        source=source,
        selection=selection,
        reindexed=reindexed,
        variables_to_save={
            "rent",
            "age",
            "taxable_income",
            "person_name",
            "county_fips",
            "unsupported",
            "no_period",
        },
    )

    np.testing.assert_array_equal(payload.data["rent"][2024], np.array([200, 100]))
    np.testing.assert_array_equal(payload.data["age"][2024], np.array([30, 10, 20]))
    np.testing.assert_array_equal(
        payload.data["taxable_income"][2024],
        np.array([2_000, 1_000]),
    )
    np.testing.assert_array_equal(
        payload.data["person_name"][2024],
        np.array([b"c", b"a", b"b"]),
    )
    np.testing.assert_array_equal(
        payload.data["county_fips"][2024],
        np.array([6037, 37183], dtype=np.int32),
    )
    assert "unsupported" not in payload.data
    assert "no_period" not in payload.data
    assert payload.values_saved == 5


def test_variable_cloner_inherits_legacy_skip_and_metadata_error_policy():
    source = _source_snapshot()
    selection = _selection()
    reindexed = EntityReindexer().reindex(source=source, selection=selection)
    source.variable_provider.simulation.tax_benefit_system.variables[
        "rent"
    ].entity.key = ""

    with pytest.raises(ValueError, match="no entity key"):
        VariableCloner().clone(
            source=source,
            selection=selection,
            reindexed=reindexed,
            variables_to_save={
                "rent",
                "unsupported",
                "no_period",
            },
        )


def test_variable_cloner_rejects_reindexed_selection_mismatch():
    source = _source_snapshot()
    selection = _selection()
    reindexed = EntityReindexer().reindex(source=source, selection=selection)
    mismatched = ReindexedEntities(
        household_ids=reindexed.household_ids,
        person_ids=reindexed.person_ids,
        person_household_ids=reindexed.person_household_ids,
        subentity_ids=reindexed.subentity_ids,
        person_subentity_ids=reindexed.person_subentity_ids,
        household_source_indices=np.array([0, 1]),
        person_source_indices=reindexed.person_source_indices,
        subentity_source_indices=reindexed.subentity_source_indices,
        persons_per_household_clone=reindexed.persons_per_household_clone,
        subentities_per_household_clone=reindexed.subentities_per_household_clone,
        person_household_clone_indices=reindexed.person_household_clone_indices,
        subentity_household_clone_indices=reindexed.subentity_household_clone_indices,
    )

    with pytest.raises(ValueError, match="source indices"):
        VariableCloner().clone(
            source=source,
            selection=selection,
            reindexed=mismatched,
            variables_to_save={"rent"},
        )


def test_default_variables_to_save_keeps_current_local_h5_overrides():
    source = _source_snapshot()

    variables = default_variables_to_save(source)

    assert "county" in variables
    assert "congressional_district_geoid" in variables
    assert "block_geoid" in variables
    assert "rent" in variables
