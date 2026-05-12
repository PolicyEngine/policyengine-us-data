import pytest

from tests.unit.fixtures.geography import (
    CHECKSUM_COUNTY_FIPS,
    CHECKSUM_STATE_FIPS,
    checksum_block_geoids,
    checksum_cd_geoids,
)
from policyengine_us_data.utils.geography_checksum import (
    canonical_geography_checksum,
    hash_string_array,
)


def test_hash_string_array_is_independent_of_numpy_string_dtype_width():
    narrow = checksum_block_geoids(dtype="<U9")
    wide = checksum_block_geoids(dtype="<U15")

    assert hash_string_array(narrow) == hash_string_array(wide)


def test_canonical_geography_checksum_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="n_records"):
        canonical_geography_checksum(
            block_geoid=checksum_block_geoids(dtype="<U9")[:1],
            cd_geoid=checksum_cd_geoids(dtype="<U4")[:1],
            n_records=0,
            n_clones=1,
        )
    with pytest.raises(ValueError, match="n_clones"):
        canonical_geography_checksum(
            block_geoid=checksum_block_geoids(dtype="<U9")[:1],
            cd_geoid=checksum_cd_geoids(dtype="<U4")[:1],
            n_records=1,
            n_clones=0,
        )


def test_canonical_geography_checksum_rejects_impossible_row_count():
    with pytest.raises(ValueError, match="n_records \\* n_clones"):
        canonical_geography_checksum(
            block_geoid=checksum_block_geoids(dtype="<U9"),
            cd_geoid=checksum_cd_geoids(dtype="<U4"),
            n_records=1,
            n_clones=3,
        )


def test_canonical_geography_checksum_rejects_mismatched_optional_arrays():
    with pytest.raises(ValueError, match="county_fips"):
        canonical_geography_checksum(
            block_geoid=checksum_block_geoids(dtype="<U9"),
            cd_geoid=checksum_cd_geoids(dtype="<U4"),
            county_fips=CHECKSUM_COUNTY_FIPS,
            n_records=1,
            n_clones=2,
        )
    with pytest.raises(ValueError, match="state_fips"):
        canonical_geography_checksum(
            block_geoid=checksum_block_geoids(dtype="<U9"),
            cd_geoid=checksum_cd_geoids(dtype="<U4"),
            state_fips=CHECKSUM_STATE_FIPS,
            n_records=1,
            n_clones=2,
        )
