"""
End-to-end test for SNAP calibration pipeline.

Tests that:
1. Sparse matrix is built correctly for SNAP targets
2. H5 file creation via create_sparse_cd_stacked_dataset works
3. Matrix prediction (X @ w) matches simulation output within tolerance

Uses ~15% aggregate tolerance due to ID reindexing changing random() seeds.
"""

from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from sparse_matrix_builder import SparseMatrixBuilder
from household_tracer import HouseholdTracer
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    create_target_groups,
)
from policyengine_us_data.datasets.cps.local_area_calibration.stacked_dataset_builder import (
    create_sparse_cd_stacked_dataset,
)


def get_test_cds(db_uri):
    """Get a subset of CDs for testing: NC, HI, MT, AK."""
    engine = create_engine(db_uri)
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM strata s
    JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
    WHERE s.stratum_group_id = 1
      AND sc.constraint_variable = 'congressional_district_geoid'
      AND (
        sc.value LIKE '37__'  -- NC (14 CDs)
        OR sc.value LIKE '150_' -- HI (2 CDs)
        OR sc.value LIKE '300_' -- MT (2 CDs)
        OR sc.value = '200' OR sc.value = '201'  -- AK (2 CDs)
      )
    ORDER BY sc.value
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        return [row[0] for row in result]


def test_snap_end_to_end():
    """Test that matrix prediction matches H5 simulation output for SNAP."""
    rng = np.random.default_rng(seed=42)

    db_path = STORAGE_FOLDER / "policy_data.db"
    db_uri = f"sqlite:///{db_path}"
    dataset_uri = STORAGE_FOLDER / "stratified_extended_cps_2023.h5"

    test_cds = get_test_cds(db_uri)
    print(f"Testing with {len(test_cds)} CDs: {test_cds[:5]}...")

    # Build sparse matrix
    sim = Microsimulation(dataset=str(dataset_uri))
    builder = SparseMatrixBuilder(
        db_uri, time_period=2023, cds_to_calibrate=test_cds, dataset_path=str(dataset_uri)
    )

    print("Building SNAP matrix...")
    targets_df, X_sparse, household_id_mapping = builder.build_matrix(
        sim, target_filter={"stratum_group_ids": [4], "variables": ["snap"]}
    )

    target_groups, group_info = create_target_groups(targets_df)
    tracer = HouseholdTracer(targets_df, X_sparse, household_id_mapping, test_cds, sim)
    tracer.print_matrix_structure()

    # Find NC state SNAP row (state_fips=37)
    group_2 = tracer.get_group_rows(2)
    nc_row = group_2[group_2['geographic_id'].astype(str) == '37']
    if nc_row.empty:
        nc_row = group_2.iloc[[0]]
    row_loc = int(nc_row.iloc[0]['row_index'])
    row_info = tracer.get_row_info(row_loc)
    target_geo_id = int(row_info['geographic_id'])
    print(f"Testing state FIPS {target_geo_id}: {row_info['variable']}")

    # Create random weights
    total_size = X_sparse.shape[1]
    w = np.zeros(total_size)
    n_nonzero = 50000
    nonzero_indices = rng.choice(total_size, n_nonzero, replace=False)
    w[nonzero_indices] = 7

    # Create H5 file
    output_dir = "./temp"
    h5_name = "test_snap"
    output_path = f"{output_dir}/{h5_name}.h5"

    print("Creating H5 file...")
    create_sparse_cd_stacked_dataset(
        w, test_cds, dataset_path=str(dataset_uri), output_path=output_path
    )

    # Load and verify
    sim_test = Microsimulation(dataset=output_path)
    hh_test_df = pd.DataFrame(
        sim_test.calculate_dataframe([
            "household_id", "household_weight", "state_fips", "snap"
        ])
    )

    # Compare matrix prediction to simulation
    y_hat = X_sparse @ w
    snap_hat_matrix = y_hat[row_loc]

    state_df = hh_test_df[hh_test_df.state_fips == target_geo_id]
    snap_hat_sim = np.sum(state_df.snap.values * state_df.household_weight.values)

    relative_diff = abs(snap_hat_sim - snap_hat_matrix) / (snap_hat_matrix + 1)
    print(f"\nAggregate comparison:")
    print(f"  Matrix prediction: {snap_hat_matrix:,.0f}")
    print(f"  Simulation output: {snap_hat_sim:,.0f}")
    print(f"  Relative diff: {relative_diff:.1%}")

    assert relative_diff < 0.15, f"Aggregate mismatch too large: {relative_diff:.1%}"
    print("\n✓ End-to-end test PASSED")


if __name__ == "__main__":
    test_snap_end_to_end()
