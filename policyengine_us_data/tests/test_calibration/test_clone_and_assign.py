"""Tests for clone_and_assign module.

Uses mock CSV data so tests don't require the real
block_cd_distributions.csv.gz file.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from pathlib import Path

from policyengine_us_data.calibration.clone_and_assign import (
    GeographyAssignment,
    load_global_block_distribution,
    assign_random_geography,
    double_geography_for_puf,
)

# ------------------------------------------------------------------
# Mock data: 3 CDs with known blocks
# ------------------------------------------------------------------

MOCK_BLOCKS = pd.DataFrame(
    {
        "cd_geoid": [
            101,
            101,
            101,
            102,
            102,
            103,
            103,
            103,
            103,
        ],
        "block_geoid": [
            "010010001001001",  # AL CD-01
            "010010001001002",
            "010010001001003",
            "020010001001001",  # AK CD-02
            "020010001001002",
            "360100001001001",  # NY CD-10
            "360100001001002",
            "360100001001003",
            "360100001001004",
        ],
        "probability": [
            0.4,
            0.3,
            0.3,
            0.6,
            0.4,
            0.25,
            0.25,
            0.25,
            0.25,
        ],
    }
)


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """Clear the lru_cache between tests."""
    load_global_block_distribution.cache_clear()
    yield
    load_global_block_distribution.cache_clear()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestLoadGlobalBlockDistribution:
    """Tests for load_global_block_distribution."""

    def _write_mock_csv(self, tmp_path):
        """Write MOCK_BLOCKS as a gzipped CSV into tmp_path."""
        csv_path = tmp_path / "block_cd_distributions.csv.gz"
        MOCK_BLOCKS.to_csv(csv_path, index=False, compression="gzip")
        return tmp_path

    def test_loads_and_normalizes(self, tmp_path):
        """Probabilities sum to 1 globally after loading."""
        storage = self._write_mock_csv(tmp_path)

        with patch(
            "policyengine_us_data.calibration"
            ".clone_and_assign.STORAGE_FOLDER",
            storage,
        ):
            blocks, cds, states, probs = (
                load_global_block_distribution.__wrapped__()
            )

        assert len(blocks) == 9
        assert len(probs) == 9
        np.testing.assert_almost_equal(probs.sum(), 1.0)

    def test_state_fips_extracted(self, tmp_path):
        """State FIPS extracted correctly from block GEOID."""
        storage = self._write_mock_csv(tmp_path)

        with patch(
            "policyengine_us_data.calibration"
            ".clone_and_assign.STORAGE_FOLDER",
            storage,
        ):
            blocks, cds, states, probs = (
                load_global_block_distribution.__wrapped__()
            )

        # First 3 blocks start with "01" -> state 1
        assert states[0] == 1
        assert states[1] == 1
        assert states[2] == 1
        # Next 2 start with "02" -> state 2
        assert states[3] == 2
        assert states[4] == 2
        # Last 4 start with "36" -> state 36
        assert states[5] == 36
        assert states[6] == 36
        assert states[7] == 36
        assert states[8] == 36


class TestAssignRandomGeography:
    """Tests for assign_random_geography."""

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_assign_shape(self, mock_load):
        """Output arrays have length n_records * n_clones."""
        blocks = MOCK_BLOCKS["block_geoid"].values
        cds = MOCK_BLOCKS["cd_geoid"].astype(str).values
        states = np.array([int(b[:2]) for b in blocks])
        probs = MOCK_BLOCKS["probability"].values.astype(np.float64)
        probs = probs / probs.sum()
        mock_load.return_value = (blocks, cds, states, probs)

        result = assign_random_geography(n_records=10, n_clones=3, seed=42)

        assert len(result.block_geoid) == 30
        assert len(result.cd_geoid) == 30
        assert len(result.state_fips) == 30
        assert result.n_records == 10
        assert result.n_clones == 3

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_assign_deterministic(self, mock_load):
        """Same seed produces identical results."""
        blocks = MOCK_BLOCKS["block_geoid"].values
        cds = MOCK_BLOCKS["cd_geoid"].astype(str).values
        states = np.array([int(b[:2]) for b in blocks])
        probs = MOCK_BLOCKS["probability"].values.astype(np.float64)
        probs = probs / probs.sum()
        mock_load.return_value = (blocks, cds, states, probs)

        r1 = assign_random_geography(n_records=10, n_clones=3, seed=99)
        r2 = assign_random_geography(n_records=10, n_clones=3, seed=99)

        np.testing.assert_array_equal(r1.block_geoid, r2.block_geoid)
        np.testing.assert_array_equal(r1.cd_geoid, r2.cd_geoid)
        np.testing.assert_array_equal(r1.state_fips, r2.state_fips)

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_assign_different_seeds(self, mock_load):
        """Different seeds produce different results."""
        blocks = MOCK_BLOCKS["block_geoid"].values
        cds = MOCK_BLOCKS["cd_geoid"].astype(str).values
        states = np.array([int(b[:2]) for b in blocks])
        probs = MOCK_BLOCKS["probability"].values.astype(np.float64)
        probs = probs / probs.sum()
        mock_load.return_value = (blocks, cds, states, probs)

        r1 = assign_random_geography(n_records=100, n_clones=3, seed=1)
        r2 = assign_random_geography(n_records=100, n_clones=3, seed=2)

        # Extremely unlikely to be identical with different seeds
        assert not np.array_equal(r1.block_geoid, r2.block_geoid)

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_state_from_block(self, mock_load):
        """state_fips[i] == int(block_geoid[i][:2]) for all i."""
        blocks = MOCK_BLOCKS["block_geoid"].values
        cds = MOCK_BLOCKS["cd_geoid"].astype(str).values
        states = np.array([int(b[:2]) for b in blocks])
        probs = MOCK_BLOCKS["probability"].values.astype(np.float64)
        probs = probs / probs.sum()
        mock_load.return_value = (blocks, cds, states, probs)

        result = assign_random_geography(n_records=20, n_clones=5, seed=42)

        for i in range(len(result.block_geoid)):
            expected = int(result.block_geoid[i][:2])
            assert result.state_fips[i] == expected, (
                f"Index {i}: state_fips={result.state_fips[i]}"
                f" but block starts with"
                f" {result.block_geoid[i][:2]}"
            )

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_cd_from_block(self, mock_load):
        """cd_geoid matches the block's CD from distribution."""
        blocks = MOCK_BLOCKS["block_geoid"].values
        cds = MOCK_BLOCKS["cd_geoid"].astype(str).values
        states = np.array([int(b[:2]) for b in blocks])
        probs = MOCK_BLOCKS["probability"].values.astype(np.float64)
        probs = probs / probs.sum()
        mock_load.return_value = (blocks, cds, states, probs)

        # Build a lookup from mock data
        block_to_cd = dict(
            zip(
                MOCK_BLOCKS["block_geoid"],
                MOCK_BLOCKS["cd_geoid"].astype(str),
            )
        )

        result = assign_random_geography(n_records=20, n_clones=5, seed=42)

        for i in range(len(result.block_geoid)):
            blk = result.block_geoid[i]
            expected_cd = block_to_cd[blk]
            assert result.cd_geoid[i] == expected_cd, (
                f"Index {i}: cd_geoid={result.cd_geoid[i]}"
                f" but block {blk} belongs to"
                f" CD {expected_cd}"
            )

    @patch(
        "policyengine_us_data.calibration.clone_and_assign"
        ".load_global_block_distribution"
    )
    def test_column_ordering(self, mock_load):
        """Verify clone_idx = i // n_records, etc."""
        blocks = MOCK_BLOCKS["block_geoid"].values
        cds = MOCK_BLOCKS["cd_geoid"].astype(str).values
        states = np.array([int(b[:2]) for b in blocks])
        probs = MOCK_BLOCKS["probability"].values.astype(np.float64)
        probs = probs / probs.sum()
        mock_load.return_value = (blocks, cds, states, probs)

        n_records = 10
        n_clones = 3
        result = assign_random_geography(
            n_records=n_records,
            n_clones=n_clones,
            seed=42,
        )

        n_total = n_records * n_clones
        assert len(result.block_geoid) == n_total

        # Verify the dataclass stores dimensions correctly
        assert result.n_records == n_records
        assert result.n_clones == n_clones

        # Verify indexing convention: i maps to
        # clone_idx = i // n_records
        # record_idx = i % n_records
        for i in range(n_total):
            clone_idx = i // n_records
            record_idx = i % n_records
            assert 0 <= clone_idx < n_clones
            assert 0 <= record_idx < n_records

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError if CSV doesn't exist."""
        fake_storage = tmp_path / "nonexistent_storage"
        fake_storage.mkdir()

        with patch(
            "policyengine_us_data.calibration.clone_and_assign"
            ".STORAGE_FOLDER",
            fake_storage,
        ):
            with pytest.raises(FileNotFoundError):
                load_global_block_distribution.__wrapped__()


class TestDoubleGeographyForPuf:
    """Tests for double_geography_for_puf."""

    def test_doubles_n_records(self):
        """n_records doubles, n_clones stays the same."""
        geo = GeographyAssignment(
            block_geoid=np.array(["010010001001001", "020010001001001"] * 3),
            cd_geoid=np.array(["101", "202"] * 3),
            state_fips=np.array([1, 2] * 3),
            n_records=2,
            n_clones=3,
        )
        result = double_geography_for_puf(geo)
        assert result.n_records == 4
        assert result.n_clones == 3
        assert len(result.block_geoid) == 12  # 4 * 3

    def test_array_length(self):
        """Output arrays have length n_records * 2 * n_clones."""
        geo = GeographyAssignment(
            block_geoid=np.array(["010010001001001"] * 15),
            cd_geoid=np.array(["101"] * 15),
            state_fips=np.array([1] * 15),
            n_records=5,
            n_clones=3,
        )
        result = double_geography_for_puf(geo)
        assert len(result.block_geoid) == 30
        assert len(result.cd_geoid) == 30
        assert len(result.state_fips) == 30

    def test_puf_half_matches_cps_half(self):
        """Each clone's PUF half has same geography as CPS half."""
        geo = GeographyAssignment(
            block_geoid=np.array(
                [
                    "010010001001001",
                    "020010001001001",
                    "360100001001001",
                    "060100001001001",
                    "480100001001001",
                    "120100001001001",
                ]
            ),
            cd_geoid=np.array(["101", "202", "1036", "653", "4831", "1227"]),
            state_fips=np.array([1, 2, 36, 6, 48, 12]),
            n_records=3,
            n_clones=2,
        )
        result = double_geography_for_puf(geo)
        n_new = result.n_records  # 6

        for c in range(result.n_clones):
            start = c * n_new
            mid = start + n_new // 2
            end = start + n_new
            # CPS half
            cps_states = result.state_fips[start:mid]
            # PUF half
            puf_states = result.state_fips[mid:end]
            np.testing.assert_array_equal(cps_states, puf_states)
