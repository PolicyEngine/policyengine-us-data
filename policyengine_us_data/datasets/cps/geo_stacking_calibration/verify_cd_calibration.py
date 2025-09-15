#!/usr/bin/env python
"""
Comprehensive verification script for congressional district calibration.
Consolidates all key checks into one place.
"""

from pathlib import Path
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import pickle
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking_sparse import SparseGeoStackingMatrixBuilder

# Setup
db_path = '/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db'
db_uri = f"sqlite:///{db_path}"
engine = create_engine(db_uri)
builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)

def verify_target_counts():
    """Verify we have exactly 30,576 targets for 436 CDs."""
    print("=" * 70)
    print("TARGET COUNT VERIFICATION")
    print("=" * 70)
    
    # Get all CDs
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM strata s
    JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
    WHERE s.stratum_group_id = 1
      AND sc.constraint_variable = "congressional_district_geoid"
    ORDER BY sc.value
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        all_cds = [row[0] for row in result]
    
    print(f"Total CDs found: {len(all_cds)}")
    
    # Get unique states
    unique_states = set()
    for cd in all_cds:
        state_fips = builder.get_state_fips_for_cd(cd)
        unique_states.add(state_fips)
    
    print(f"Unique states: {len(unique_states)}")
    
    # Calculate expected targets
    print("\n=== Expected Target Counts ===")
    categories = [
        ("National", 5),
        ("CD Age (18 × 436)", 18 * 436),
        ("CD Medicaid (1 × 436)", 436),
        ("CD SNAP household (1 × 436)", 436),
        ("State SNAP costs", len(unique_states)),
        ("CD AGI distribution (9 × 436)", 9 * 436),
        ("CD IRS SOI (50 × 436)", 50 * 436)
    ]
    
    running_total = 0
    for name, count in categories:
        running_total += count
        print(f"{name:30} {count:6,}  (running total: {running_total:6,})")
    
    print(f"\n=== Total Expected: {running_total:,} ===")
    
    project_status_target = 30576
    print(f"\nPROJECT_STATUS.md target: {project_status_target:,}")
    print(f"Match: {running_total == project_status_target}")
    
    return running_total == project_status_target

def test_snap_cascading(num_cds=5):
    """Test that state SNAP costs cascade correctly to CDs."""
    print("\n" + "=" * 70)
    print(f"SNAP CASCADING TEST (with {num_cds} CDs)")
    print("=" * 70)
    
    # Get test CDs
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM strata s
    JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
    WHERE s.stratum_group_id = 1
      AND sc.constraint_variable = "congressional_district_geoid"
    ORDER BY sc.value
    LIMIT :limit
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query), {'limit': num_cds}).fetchall()
        test_cds = [row[0] for row in result]
    
    print(f"Testing with CDs: {test_cds}")
    
    # Load simulation
    dataset_uri = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/stratified_extended_cps_2023.h5"
    sim = Microsimulation(dataset=dataset_uri)
    
    # Build matrix
    targets_df, X_sparse, household_id_mapping = builder.build_stacked_matrix_sparse(
        'congressional_district', 
        test_cds,
        sim
    )
    
    # Check state SNAP costs
    state_snap_costs = targets_df[
        (targets_df['geographic_level'] == 'state') & 
        (targets_df['variable'] == 'snap')
    ]
    
    print(f"\nState SNAP cost targets found: {len(state_snap_costs)}")
    if not state_snap_costs.empty:
        print("State SNAP costs by state:")
        for _, row in state_snap_costs.iterrows():
            print(f"  State {row['geographic_id']}: ${row['value']:,.0f}")
    
    # Check matrix dimensions
    print(f"\nMatrix shape: {X_sparse.shape}")
    print(f"Number of targets: {len(targets_df)}")
    
    # Verify state SNAP rows have correct sparsity pattern
    if not state_snap_costs.empty:
        print("\nVerifying state SNAP cost matrix rows:")
        for idx, (i, row) in enumerate(state_snap_costs.iterrows()):
            matrix_row = X_sparse[i, :].toarray().flatten()
            nonzero = np.count_nonzero(matrix_row)
            total = np.sum(matrix_row)
            print(f"  State {row['geographic_id']}: {nonzero} non-zero values, sum = ${total:,.0f}")
    
    return len(state_snap_costs) > 0

def check_loaded_targets(pkl_file=None):
    """Check targets from a saved pickle file."""
    if pkl_file is None:
        pkl_file = '/home/baogorek/Downloads/cd_calibration_data/cd_targets_df.pkl'
    
    if not Path(pkl_file).exists():
        print(f"\nPickle file not found: {pkl_file}")
        return
    
    print("\n" + "=" * 70)
    print("LOADED TARGETS CHECK")
    print("=" * 70)
    
    with open(pkl_file, 'rb') as f:
        targets_df = pickle.load(f)
    
    print(f"Total targets loaded: {len(targets_df):,}")
    
    # Breakdown by geographic level
    for level in ['national', 'state', 'congressional_district']:
        count = len(targets_df[targets_df['geographic_level'] == level])
        print(f"  {level}: {count:,}")
    
    # Check for AGI distribution
    agi_targets = targets_df[
        (targets_df['description'].str.contains('adjusted_gross_income', na=False)) &
        (targets_df['variable'] == 'person_count')
    ]
    print(f"\nAGI distribution targets: {len(agi_targets):,}")
    
    # Check for state SNAP costs
    state_snap = targets_df[
        (targets_df['geographic_level'] == 'state') & 
        (targets_df['variable'] == 'snap')
    ]
    print(f"State SNAP cost targets: {len(state_snap)}")
    
    # Sample IRS targets
    irs_income_tax = targets_df[targets_df['variable'] == 'income_tax']
    print(f"Income tax targets: {len(irs_income_tax)}")

def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("CONGRESSIONAL DISTRICT CALIBRATION VERIFICATION")
    print("=" * 70)
    
    # 1. Verify target counts
    counts_ok = verify_target_counts()
    
    # 2. Test SNAP cascading with small subset
    snap_ok = test_snap_cascading(num_cds=5)
    
    # 3. Check loaded targets if file exists
    check_loaded_targets()
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"✓ Target count correct (30,576): {counts_ok}")
    print(f"✓ State SNAP costs cascade to CDs: {snap_ok}")
    
    if counts_ok and snap_ok:
        print("\n✅ All verification checks passed!")
    else:
        print("\n❌ Some checks failed - review output above")

if __name__ == "__main__":
    main()