from types import SimpleNamespace

import numpy as np
import pytest

from policyengine_us_data.build_outputs.requests import AreaFilter
from policyengine_us_data.build_outputs.selection import AreaSelector, CloneSelection
from policyengine_us_data.build_outputs.weights import CloneWeightMatrix


def _geography(**overrides):
    values = {
        "block_geoid": np.array(["block-1", "block-2", "block-3", "block-4"]),
        "cd_geoid": np.array(["3701", "3702", "3701", "3702"]),
        "county_fips": np.array(["37183", "37183", "06037", "06037"]),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_area_selector_returns_positive_weight_clone_rows():
    weights = CloneWeightMatrix.from_vector(
        np.array([1.0, 0.0, 2.0, 3.0]),
        n_records=2,
    )

    selection = AreaSelector().select(weights=weights, geography=_geography())

    np.testing.assert_array_equal(selection.clone_indices, np.array([0, 1, 1]))
    np.testing.assert_array_equal(
        selection.source_household_indices,
        np.array([0, 0, 1]),
    )
    np.testing.assert_array_equal(selection.weights, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(
        selection.block_geoids,
        np.array(["block-1", "block-3", "block-4"]),
    )
    assert selection.n_source_households == 2
    assert selection.n_total_clones == 2


def test_area_selector_applies_cd_and_county_filters_as_conjunction():
    weights = CloneWeightMatrix.from_vector(
        np.array([1.0, 1.0, 1.0, 1.0]),
        n_records=2,
    )
    filters = (
        AreaFilter(geography_field="cd_geoid", op="in", value=("3702",)),
        AreaFilter(geography_field="county_fips", op="in", value=("06037",)),
    )

    selection = AreaSelector().select(
        weights=weights,
        geography=_geography(),
        filters=filters,
    )

    np.testing.assert_array_equal(selection.clone_indices, np.array([1]))
    np.testing.assert_array_equal(selection.source_household_indices, np.array([1]))
    np.testing.assert_array_equal(selection.block_geoids, np.array(["block-4"]))


def test_area_selector_preserves_legacy_numeric_cd_geoid_filtering():
    weights = CloneWeightMatrix.from_vector(
        np.array([1.0, 1.0, 1.0, 1.0]),
        n_records=2,
    )
    filters = (AreaFilter(geography_field="cd_geoid", op="in", value=("3702",)),)

    selection = AreaSelector().select(
        weights=weights,
        geography=_geography(cd_geoid=np.array([3701, 3702, 3701, 3702])),
        filters=filters,
    )

    np.testing.assert_array_equal(selection.clone_indices, np.array([0, 1]))
    np.testing.assert_array_equal(selection.source_household_indices, np.array([1, 1]))
    np.testing.assert_array_equal(
        selection.congressional_district_geoids,
        np.array(["3702", "3702"]),
    )


def test_area_selector_rejects_zero_active_clones():
    weights = CloneWeightMatrix.from_vector(
        np.array([1.0, 1.0, 1.0, 1.0]),
        n_records=2,
    )
    filters = (AreaFilter(geography_field="cd_geoid", op="in", value=("9999",)),)

    with pytest.raises(ValueError, match="No active clones"):
        AreaSelector().select(weights=weights, geography=_geography(), filters=filters)


def test_area_selector_rejects_empty_active_block_geoids():
    weights = CloneWeightMatrix.from_vector(
        np.array([1.0, 0.0, 0.0, 0.0]),
        n_records=2,
    )

    with pytest.raises(ValueError, match="empty block GEOIDs"):
        AreaSelector().select(
            weights=weights,
            geography=_geography(block_geoid=np.array(["", "b", "c", "d"])),
        )


def test_area_selector_rejects_misaligned_geography_vectors():
    weights = CloneWeightMatrix.from_vector(
        np.array([1.0, 0.0, 0.0, 0.0]),
        n_records=2,
    )

    with pytest.raises(ValueError, match="length 1"):
        AreaSelector().select(
            weights=weights,
            geography=_geography(cd_geoid=np.array(["3701"])),
        )


def test_clone_selection_is_read_only():
    selection = CloneSelection(
        clone_indices=np.array([0]),
        source_household_indices=np.array([0]),
        weights=np.array([1.0]),
        block_geoids=np.array(["block-1"]),
        congressional_district_geoids=np.array(["3701"]),
        filters=(),
        n_source_households=1,
        n_total_clones=1,
    )

    with pytest.raises(ValueError, match="read-only"):
        selection.clone_indices[0] = 99
