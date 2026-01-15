import pytest
import numpy as np
from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.local_area_calibration.sparse_matrix_builder import (
    SparseMatrixBuilder,
)


@pytest.fixture
def sim():
    dataset_path = STORAGE_FOLDER / "stratified_extended_cps_2023.h5"
    return Microsimulation(dataset=str(dataset_path))


@pytest.fixture
def builder():
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    db_uri = f"sqlite:///{db_path}"
    cds_to_calibrate = ["101", "601"]  # AL-1, CA-1
    return SparseMatrixBuilder(
        db_uri=db_uri,
        time_period=2023,
        cds_to_calibrate=cds_to_calibrate,
        dataset_path=None,
    )


def test_person_level_aggregation_preserves_totals(sim):
    """Health insurance premiums (person-level) should sum correctly to household."""
    var = "health_insurance_premiums_without_medicare_part_b"
    person_total = sim.calculate(var, 2023, map_to="person").values.sum()
    household_total = sim.calculate(var, 2023, map_to="household").values.sum()
    assert np.isclose(person_total, household_total, rtol=1e-6)


def test_matrix_shape(sim, builder):
    """Matrix should have (n_targets, n_households * n_cds) shape."""
    targets_df, X_sparse, _ = builder.build_matrix(
        sim,
        target_filter={
            "variables": ["health_insurance_premiums_without_medicare_part_b"]
        },
    )
    n_households = len(
        sim.calculate("household_id", map_to="household").values
    )
    n_cds = 2
    assert X_sparse.shape == (1, n_households * n_cds)


def test_combined_snap_and_health_insurance(sim, builder):
    """Matrix should include both SNAP and health insurance targets."""
    targets_df, X_sparse, _ = builder.build_matrix(
        sim,
        target_filter={
            "stratum_group_ids": [4],
            "variables": ["health_insurance_premiums_without_medicare_part_b"],
        },
    )
    variables = targets_df["variable"].unique()
    assert "snap" in variables
    assert "household_count" in variables
    assert "health_insurance_premiums_without_medicare_part_b" in variables
