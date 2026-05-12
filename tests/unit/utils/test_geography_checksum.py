import numpy as np
import pytest

from policyengine_us_data.utils.geography_checksum import (
    canonical_geography_checksum,
    hash_string_array,
)


def test_hash_string_array_is_independent_of_numpy_string_dtype_width():
    narrow = np.array(["010010001", "010010002"], dtype="<U9")
    wide = np.array(["010010001", "010010002"], dtype="<U15")

    assert hash_string_array(narrow) == hash_string_array(wide)


def test_canonical_geography_checksum_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="n_records"):
        canonical_geography_checksum(
            block_geoid=["010010001"],
            cd_geoid=["0101"],
            n_records=0,
            n_clones=1,
        )
    with pytest.raises(ValueError, match="n_clones"):
        canonical_geography_checksum(
            block_geoid=["010010001"],
            cd_geoid=["0101"],
            n_records=1,
            n_clones=0,
        )


def test_canonical_geography_checksum_rejects_impossible_row_count():
    with pytest.raises(ValueError, match="n_records \\* n_clones"):
        canonical_geography_checksum(
            block_geoid=["010010001", "010010002"],
            cd_geoid=["0101", "0102"],
            n_records=1,
            n_clones=3,
        )


def test_canonical_geography_checksum_rejects_mismatched_optional_arrays():
    with pytest.raises(ValueError, match="county_fips"):
        canonical_geography_checksum(
            block_geoid=["010010001", "010010002"],
            cd_geoid=["0101", "0102"],
            county_fips=["01001"],
            n_records=1,
            n_clones=2,
        )
    with pytest.raises(ValueError, match="state_fips"):
        canonical_geography_checksum(
            block_geoid=["010010001", "010010002"],
            cd_geoid=["0101", "0102"],
            state_fips=[1],
            n_records=1,
            n_clones=2,
        )
