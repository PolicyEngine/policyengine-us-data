#!/usr/bin/env python3
"""
Calibrate household weights for multiple states using L0 sparse optimization.

This version uses sparse matrices throughout the entire pipeline for memory efficiency.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking_sparse import SparseGeoStackingMatrixBuilder
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import create_target_groups

# Setup
db_uri = f"sqlite:///{Path.home()}/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"
builder = SparseGeoStackingMatrixBuilder(db_uri)

# Create simulation 
sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim.build_from_dataset()

print("Testing multi-state stacking with SPARSE matrices: ALL 51 STATES (50 + DC)")
print("=" * 70)

# Build stacked sparse matrix for ALL states and DC
# FIPS codes for all 50 states + DC
states_to_calibrate = [
    '1',   # Alabama
    '2',   # Alaska
    '4',   # Arizona
    '5',   # Arkansas
    '6',   # California
    '8',   # Colorado
    '9',   # Connecticut
    '10',  # Delaware
    '11',  # District of Columbia
    '12',  # Florida
    '13',  # Georgia
    '15',  # Hawaii
    '16',  # Idaho
    '17',  # Illinois
    '18',  # Indiana
    '19',  # Iowa
    '20',  # Kansas
    '21',  # Kentucky
    '22',  # Louisiana
    '23',  # Maine
    '24',  # Maryland
    '25',  # Massachusetts
    '26',  # Michigan
    '27',  # Minnesota
    '28',  # Mississippi
    '29',  # Missouri
    '30',  # Montana
    '31',  # Nebraska
    '32',  # Nevada
    '33',  # New Hampshire
    '34',  # New Jersey
    '35',  # New Mexico
    '36',  # New York
    '37',  # North Carolina
    '38',  # North Dakota
    '39',  # Ohio
    '40',  # Oklahoma
    '41',  # Oregon
    '42',  # Pennsylvania
    '44',  # Rhode Island
    '45',  # South Carolina
    '46',  # South Dakota
    '47',  # Tennessee
    '48',  # Texas
    '49',  # Utah
    '50',  # Vermont
    '51',  # Virginia
    '53',  # Washington
    '54',  # West Virginia
    '55',  # Wisconsin
    '56',  # Wyoming
]

print(f"Total jurisdictions: {len(states_to_calibrate)}")
print("=" * 70)

targets_df, sparse_matrix, household_id_mapping = builder.build_stacked_matrix_sparse(
    'state', 
    states_to_calibrate,
    sim
)

print(f"\nSparse Matrix Statistics:")
print(f"- Shape: {sparse_matrix.shape}")
print(f"- Non-zero elements: {sparse_matrix.nnz:,}")
print(f"- Sparsity: {100 * sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.4f}%")
print(f"- Memory usage: {(sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes) / 1024**2:.2f} MB")

# Compare to dense matrix memory
dense_memory = sparse_matrix.shape[0] * sparse_matrix.shape[1] * 4 / 1024**2  # 4 bytes per float32, in MB
print(f"- Dense matrix would use: {dense_memory:.2f} MB")
print(f"- Memory savings: {100*(1 - (sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes)/(dense_memory * 1024**2)):.2f}%")

# Calibrate using our L0 package
from l0.calibration import SparseCalibrationWeights

# The sparse matrix is already in CSR format
X_sparse = sparse_matrix

model = SparseCalibrationWeights(
    n_features=X_sparse.shape[1],
    beta=0.66,
    gamma=-0.1,
    zeta=1.1,
    init_keep_prob=0.3,
    init_weight_scale=0.5,
)


# Create automatic target groups
target_groups, group_info = create_target_groups(targets_df)

print(f"\nAutomatic target grouping:")
print(f"Total groups: {len(np.unique(target_groups))}")
for info in group_info:
    print(f"  {info}")

model.fit(
    M=X_sparse,
    y=targets_df.value.values,
    target_groups=target_groups,
    lambda_l0=1.5e-7,
    lambda_l2=0,
    lr=0.2,
    epochs=4000,
    loss_type="relative",
    verbose=True,
    verbose_freq=500,
)

w = model.get_weights(deterministic=True).detach().numpy()
n_active = sum(w != 0)
print(f"\nFinal sparsity: {n_active} active weights out of {len(w)} ({100*n_active/len(w):.2f}%)")

# Evaluate group-wise performance
print("\nGroup-wise performance:")
print("-" * 50)

import torch
with torch.no_grad():
    y_pred = model.predict(X_sparse).cpu().numpy()
    y_actual = targets_df.value.values
    rel_errors = np.abs((y_actual - y_pred) / (y_actual + 1))
    
    for group_id in np.unique(target_groups):
        group_mask = target_groups == group_id
        group_errors = rel_errors[group_mask]
        mean_err = np.mean(group_errors)
        max_err = np.max(group_errors)
        
        # Find the group info
        group_label = group_info[group_id]
        print(f"{group_label}:")
        print(f"  Mean error: {mean_err:.2%}, Max error: {max_err:.2%}")

print(f"\nTargets Summary:")
print(f"Total targets: {len(targets_df)}")
print(f"- National targets: {len(targets_df[targets_df['geographic_id'] == 'US'])}")
print(f"- California targets: {len(targets_df[targets_df['geographic_id'] == '6'])}")  
print(f"- North Carolina targets: {len(targets_df[targets_df['geographic_id'] == '37'])}")

print(f"\nTargets by type (stratum_group_id):")
print(f"- National hardcoded: {len(targets_df[targets_df['stratum_group_id'] == 'national_hardcoded'])}")
print(f"- Age (group 2): {len(targets_df[targets_df['stratum_group_id'] == 2])}")
print(f"- AGI distribution (group 3): {len(targets_df[targets_df['stratum_group_id'] == 3])}")
print(f"- SNAP (group 4): {len(targets_df[targets_df['stratum_group_id'] == 4])}")
print(f"- Medicaid (group 5): {len(targets_df[targets_df['stratum_group_id'] == 5])}")
print(f"- EITC (group 6): {len(targets_df[targets_df['stratum_group_id'] == 6])}")
print(f"- AGI total amount: {len(targets_df[targets_df['stratum_group_id'] == 'agi_total_amount'])}")

# Count IRS scalar variables
irs_scalar_count = len([x for x in targets_df['stratum_group_id'].unique() if isinstance(x, str) and x.startswith('irs_scalar_')])
print(f"- IRS scalar variables: {irs_scalar_count} unique variables")

print(f"\nMatrix dimensions: {sparse_matrix.shape}")
print(f"- Rows (targets): {sparse_matrix.shape[0]}")
print(f"- Columns (household copies): {sparse_matrix.shape[1]}")

# Check household naming from mapping
total_households = sum(len(hh_list) for hh_list in household_id_mapping.values())
print(f"\nHousehold copies:")
print(f"- California households: {len(household_id_mapping.get('state6', []))}")
print(f"- North Carolina households: {len(household_id_mapping.get('state37', []))}")
print(f"- Total household copies: {total_households}")

print("\n" + "=" * 70)
print("Sparse matrix calibration test complete!")
print(f"Successfully used sparse matrices throughout the entire pipeline.")
print(f"Memory efficiency gain: ~{100*(1 - sparse_matrix.nnz/(sparse_matrix.shape[0]*sparse_matrix.shape[1])):.1f}% compared to dense")