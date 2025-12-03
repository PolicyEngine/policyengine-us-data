"""
Test script for SparseMatrixBuilder.
Verifies X_sparse values are correct for state-level SNAP targets.
"""

from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from sparse_matrix_builder import SparseMatrixBuilder, get_calculated_variables
from household_tracer import HouseholdTracer
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import (
    create_target_groups,
)
from policyengine_us_data.datasets.cps.geo_stacking_calibration.household_tracer import HouseholdTracer  

db_path = STORAGE_FOLDER / "policy_data.db"
db_uri = f"sqlite:///{db_path}"
dataset_uri = STORAGE_FOLDER / "stratified_extended_cps_2023.h5"

engine = create_engine(db_uri)
query = """
SELECT DISTINCT sc.value as cd_geoid
FROM strata s
JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE s.stratum_group_id = 1
  AND sc.constraint_variable = 'congressional_district_geoid'
  AND (
    sc.value LIKE '37__'  -- NC (14 CDs: 3701-3714)
    OR sc.value LIKE '150_' -- HI (2 CDs: 1501, 1502)
    OR sc.value LIKE '300_' -- MT (at-large: 3000, 3001)
    OR sc.value = '200' OR sc.value = '201'  -- AK (at-large)
  )
ORDER BY sc.value
"""
with engine.connect() as conn:
    result = conn.execute(text(query)).fetchall()
    test_cds = [row[0] for row in result]

print(f"Testing with {len(test_cds)} CDs: {test_cds}")

sim = Microsimulation(dataset=str(dataset_uri))
builder = SparseMatrixBuilder(db_uri, time_period=2023, cds_to_calibrate=test_cds,
                              dataset_path=str(dataset_uri))

print("\nBuilding matrix with stratum_group_id=4 (SNAP) + variable='snap' (national)...")
targets_df, X_sparse, household_id_mapping = builder.build_matrix(
    sim,
    target_filter={"stratum_group_ids": [4], "variables": ["snap"]}
)

target_groups, group_info = create_target_groups(targets_df)
tracer = HouseholdTracer(targets_df, X_sparse, household_id_mapping, test_cds, sim)

tracer.print_matrix_structure()

print(f"\nMatrix shape: {X_sparse.shape}")
print(f"Non-zero elements: {X_sparse.nnz}")
print(f"Targets found: {len(targets_df)}")
print("\nTargets:")
print(targets_df[['target_id', 'variable', 'value', 'geographic_id']])

n_households = len(sim.calculate("household_id", map_to="household").values)
print(f"\nHouseholds: {n_households}")
print(f"CDs: {len(test_cds)}")
print(f"Expected columns: {n_households * len(test_cds)}")

print("\n" + "="*60)
print("VERIFICATION: Check that X_sparse values match simulation")
print("="*60)

# Group rows by state to minimize sim creation
states_in_test = set()
for _, target in targets_df.iterrows():
    try:
        state_fips = int(target['geographic_id'])
        if state_fips < 100:  # State-level targets only
            states_in_test.add(state_fips)
    except:
        pass

# Create fresh sims for verification (deterministic)
state_sims = {}
for state in states_in_test:
    state_cds = [cd for cd in test_cds if int(cd) // 100 == state]
    if state_cds:
        state_sims[state] = Microsimulation(dataset=str(dataset_uri))
        state_sims[state].set_input("state_fips", 2023,
                                    np.full(n_households, state, dtype=np.int32))

for row_idx, (_, target) in enumerate(targets_df.iterrows()):
    try:
        state_fips = int(target['geographic_id'])
    except:
        continue

    variable = target['variable']
    state_cds = [cd for cd in test_cds if int(cd) // 100 == state_fips]

    if not state_cds or state_fips not in state_sims:
        continue

    state_sim = state_sims[state_fips]
    sim_values = state_sim.calculate(variable, map_to="household").values

    cd = state_cds[0]
    cd_idx = test_cds.index(cd)
    col_start = cd_idx * n_households

    matrix_row = X_sparse[row_idx, col_start:col_start + n_households].toarray().ravel()

    nonzero_sim = np.where(sim_values > 0)[0]
    nonzero_matrix = np.where(matrix_row > 0)[0]

    values_match = np.allclose(sim_values[nonzero_sim], matrix_row[nonzero_sim], rtol=1e-5)

    print(f"\nRow {row_idx}: State {state_fips}, Variable: {variable}")
    print(f"  Sim non-zero count: {len(nonzero_sim)}")
    print(f"  Matrix non-zero count: {len(nonzero_matrix)}")
    print(f"  Values match: {values_match}")

    if not values_match and len(nonzero_sim) > 0:
        mismatches = np.where(~np.isclose(sim_values, matrix_row, rtol=1e-5))[0][:5]
        for idx in mismatches:
            print(f"    Mismatch at hh_idx {idx}: sim={sim_values[idx]:.2f}, matrix={matrix_row[idx]:.2f}")

print("\n" + "="*60)
print("SPARSITY CHECK: Verify zeros in wrong state columns")
print("="*60)

for row_idx, (_, target) in enumerate(targets_df.iterrows()):
    state_fips = int(target['geographic_id'])

    wrong_state_cds = [cd for cd in test_cds if int(cd) // 100 != state_fips]

    all_zero = True
    for cd in wrong_state_cds[:2]:
        cd_idx = test_cds.index(cd)
        col_start = cd_idx * n_households
        matrix_row = X_sparse[row_idx, col_start:col_start + n_households].toarray().ravel()
        if np.any(matrix_row != 0):
            all_zero = False
            print(f"  ERROR: Row {row_idx} (state {state_fips}) has non-zero in CD {cd}")

    if all_zero:
        print(f"Row {row_idx}: State {state_fips} - correctly zero in other states' CDs")

print("\nTest complete!")
