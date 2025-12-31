"""Tests for stacked_dataset_builder.py using deterministic test fixture."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.local_area_calibration.stacked_dataset_builder import (
    create_sparse_cd_stacked_dataset,
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "test_fixture_50hh.h5")
TEST_CDS = ["3701", "201"]  # NC-01 and AK at-large
SEED = 42


@pytest.fixture(scope="module")
def fixture_sim():
    return Microsimulation(dataset=FIXTURE_PATH)


@pytest.fixture(scope="module")
def n_households(fixture_sim):
    return fixture_sim.calculate("household_id", map_to="household").shape[0]


@pytest.fixture(scope="module")
def test_weights(n_households):
    """Create deterministic weight vector with known households."""
    np.random.seed(SEED)
    n_cds = len(TEST_CDS)
    w = np.zeros(n_households * n_cds, dtype=float)

    # Give 5 households in each CD a weight
    for cd_idx in range(n_cds):
        hh_indices = np.random.choice(n_households, size=5, replace=False)
        for hh_idx in hh_indices:
            w[cd_idx * n_households + hh_idx] = np.random.uniform(1, 3)

    return w


@pytest.fixture(scope="module")
def stacked_result(test_weights):
    """Run stacked dataset builder and return results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_output.h5")

        create_sparse_cd_stacked_dataset(
            test_weights,
            TEST_CDS,
            cd_subset=TEST_CDS,
            dataset_path=FIXTURE_PATH,
            output_path=output_path,
        )

        sim_after = Microsimulation(dataset=output_path)
        hh_df = pd.DataFrame(
            sim_after.calculate_dataframe(
                [
                    "household_id",
                    "congressional_district_geoid",
                    "county",
                    "household_weight",
                    "state_fips",
                ]
            )
        )

        mapping_path = os.path.join(
            tmpdir, "mappings", "test_output_household_mapping.csv"
        )
        mapping_df = pd.read_csv(mapping_path)

        yield {"hh_df": hh_df, "mapping_df": mapping_df}


class TestStackedDatasetBuilder:
    def test_output_has_correct_cd_count(self, stacked_result):
        """Output should contain households from both CDs."""
        hh_df = stacked_result["hh_df"]
        cds_in_output = hh_df["congressional_district_geoid"].unique()
        assert len(cds_in_output) == len(TEST_CDS)

    def test_output_contains_both_cds(self, stacked_result):
        """Output should contain both NC-01 (3701) and AK-AL (201)."""
        hh_df = stacked_result["hh_df"]
        cds_in_output = set(hh_df["congressional_district_geoid"].unique())
        expected = {3701, 201}
        assert cds_in_output == expected

    def test_state_fips_matches_cd(self, stacked_result):
        """State FIPS should match the CD's state."""
        hh_df = stacked_result["hh_df"]

        for _, row in hh_df.iterrows():
            cd_geoid = row["congressional_district_geoid"]
            state_fips = row["state_fips"]
            expected_state = cd_geoid // 100
            assert state_fips == expected_state

    def test_household_ids_are_unique(self, stacked_result):
        """Each household should have a unique ID."""
        hh_df = stacked_result["hh_df"]
        assert hh_df["household_id"].nunique() == len(hh_df)

    def test_mapping_has_required_columns(self, stacked_result):
        """Mapping CSV should have expected columns."""
        mapping_df = stacked_result["mapping_df"]
        required_cols = [
            "new_household_id",
            "original_household_id",
            "congressional_district",
            "state_fips",
        ]
        for col in required_cols:
            assert col in mapping_df.columns

    def test_mapping_covers_all_output_households(self, stacked_result):
        """Every output household should be in the mapping."""
        hh_df = stacked_result["hh_df"]
        mapping_df = stacked_result["mapping_df"]

        output_hh_ids = set(hh_df["household_id"].values)
        mapped_hh_ids = set(mapping_df["new_household_id"].values)
        assert output_hh_ids == mapped_hh_ids

    def test_weights_are_positive(self, stacked_result):
        """All household weights should be positive."""
        hh_df = stacked_result["hh_df"]
        assert (hh_df["household_weight"] > 0).all()

    def test_counties_match_state(self, stacked_result):
        """County names should end with correct state code."""
        hh_df = stacked_result["hh_df"]

        for _, row in hh_df.iterrows():
            county = row["county"]
            state_fips = row["state_fips"]

            if state_fips == 37:
                assert county.endswith(
                    "_NC"
                ), f"NC county should end with _NC: {county}"
            elif state_fips == 2:
                assert county.endswith(
                    "_AK"
                ), f"AK county should end with _AK: {county}"

    def test_household_count_matches_weights(
        self, stacked_result, test_weights
    ):
        """Number of output households should match non-zero weights."""
        hh_df = stacked_result["hh_df"]
        expected_households = (test_weights > 0).sum()
        assert len(hh_df) == expected_households


@pytest.fixture(scope="module")
def stacked_sim(test_weights):
    """Run stacked dataset builder and return the simulation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_output.h5")

        create_sparse_cd_stacked_dataset(
            test_weights,
            TEST_CDS,
            cd_subset=TEST_CDS,
            dataset_path=FIXTURE_PATH,
            output_path=output_path,
        )

        sim = Microsimulation(dataset=output_path)
        yield sim


@pytest.fixture(scope="module")
def stacked_sim_with_overlap(n_households):
    """Stacked dataset where SAME households appear in BOTH CDs."""
    # Force same households to appear in both CDs - tests reindexing
    w = np.zeros(n_households * len(TEST_CDS), dtype=float)
    overlap_households = [0, 1, 2]  # Same households in both CDs
    for cd_idx in range(len(TEST_CDS)):
        for hh_idx in overlap_households:
            w[cd_idx * n_households + hh_idx] = 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_overlap.h5")
        create_sparse_cd_stacked_dataset(
            w,
            TEST_CDS,
            cd_subset=TEST_CDS,
            dataset_path=FIXTURE_PATH,
            output_path=output_path,
        )
        sim = Microsimulation(dataset=output_path)
        yield {"sim": sim, "n_overlap": len(overlap_households)}


class TestEntityReindexing:
    """Tests for entity ID reindexing to prevent collisions."""

    def test_family_ids_are_unique(self, stacked_sim):
        """Family IDs should be globally unique across all CDs."""
        family_ids = stacked_sim.calculate("family_id", map_to="family").values
        assert len(family_ids) == len(
            set(family_ids)
        ), "Family IDs should be unique"

    def test_tax_unit_ids_are_unique(self, stacked_sim):
        """Tax unit IDs should be globally unique."""
        tax_unit_ids = stacked_sim.calculate(
            "tax_unit_id", map_to="tax_unit"
        ).values
        assert len(tax_unit_ids) == len(
            set(tax_unit_ids)
        ), "Tax unit IDs should be unique"

    def test_spm_unit_ids_are_unique(self, stacked_sim):
        """SPM unit IDs should be globally unique."""
        spm_unit_ids = stacked_sim.calculate(
            "spm_unit_id", map_to="spm_unit"
        ).values
        assert len(spm_unit_ids) == len(
            set(spm_unit_ids)
        ), "SPM unit IDs should be unique"

    def test_person_family_id_matches_family_id(self, stacked_sim):
        """person_family_id should reference valid family_ids."""
        person_family_ids = stacked_sim.calculate(
            "person_family_id", map_to="person"
        ).values
        family_ids = set(
            stacked_sim.calculate("family_id", map_to="family").values
        )
        for pf_id in person_family_ids:
            assert (
                pf_id in family_ids
            ), f"person_family_id {pf_id} not in family_ids"

    def test_family_ids_unique_across_cds(self, stacked_sim_with_overlap):
        """Same household in different CDs should have different family_ids."""
        sim = stacked_sim_with_overlap["sim"]
        n_overlap = stacked_sim_with_overlap["n_overlap"]
        n_cds = len(TEST_CDS)

        family_ids = sim.calculate("family_id", map_to="family").values
        household_ids = sim.calculate("household_id", map_to="household").values

        # Should have n_overlap * n_cds unique families (one per HH-CD pair)
        expected_families = n_overlap * n_cds
        assert len(family_ids) == expected_families, (
            f"Expected {expected_families} families (same HH in {n_cds} CDs), "
            f"got {len(family_ids)}"
        )
        assert len(set(family_ids)) == expected_families, (
            f"Family IDs not unique: {len(set(family_ids))} unique "
            f"out of {len(family_ids)}"
        )
