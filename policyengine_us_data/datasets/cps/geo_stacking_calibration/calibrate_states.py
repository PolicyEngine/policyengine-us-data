#!/usr/bin/env python3
"""
Calibrate household weights for multiple states using L0 sparse optimization.

This script demonstrates geo-stacking calibration for California and North Carolina,
using national and state-level targets with L0-regularized weights.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking import GeoStackingMatrixBuilder
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import create_target_groups

# Setup
db_uri = f"sqlite:///{Path.home()}/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"
builder = GeoStackingMatrixBuilder(db_uri)

# Create simulation 
sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim.build_from_dataset()

print("Testing multi-state stacking: California (6) and North Carolina (37)")
print("=" * 70)

# Build stacked matrix for CA and NC
targets_df, matrix_df = builder.build_stacked_matrix(
    'state', 
    ['6', '37'],  # California and North Carolina FIPS codes
    sim
)

# OK, let's calibrate using our L0 package:

from l0.calibration import SparseCalibrationWeights

# Convert to sparse
X_sparse = sp.csr_matrix(matrix_df)

model = SparseCalibrationWeights(
    n_features=X_sparse.shape[1],  # TODO: why do I need to feed this in when it's part of the data structure?
    beta=0.66,
    gamma=-0.1,
    zeta=1.1,
    init_keep_prob=0.3,
    init_weights=1.0,  # Start all weights at 1.0
    weight_jitter_sd=0.5,  # Add jitter at fit() time to break symmetry
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

print(f"\nMatrix dimensions: {matrix_df.shape}")
print(f"- Rows (targets): {matrix_df.shape[0]}")
print(f"- Columns (household copies): {matrix_df.shape[1]}")

# Check household naming
household_cols = matrix_df.columns.tolist()
ca_households = [col for col in household_cols if '_state6' in col]
nc_households = [col for col in household_cols if '_state37' in col]

print(f"\nHousehold copies:")
print(f"- California households: {len(ca_households)}")
print(f"- North Carolina households: {len(nc_households)}")

# Verify sparsity pattern
print("\nVerifying sparsity pattern:")
print("-" * 40)

# Check a CA age target - should only have non-zero values for CA households
ca_age_targets = targets_df[(targets_df['geographic_id'] == '6') & 
                            (targets_df['variable'].str.contains('age'))]
if not ca_age_targets.empty:
    ca_target_id = ca_age_targets.iloc[0]['stacked_target_id']
    ca_row = matrix_df.loc[ca_target_id]
    ca_nonzero = (ca_row[ca_households] != 0).sum()
    nc_nonzero = (ca_row[nc_households] != 0).sum()
    print(f"CA age target '{ca_target_id}':")
    print(f"  - Non-zero CA households: {ca_nonzero}")
    print(f"  - Non-zero NC households: {nc_nonzero} (should be 0)")

# Check a NC age target - should only have non-zero values for NC households  
nc_age_targets = targets_df[(targets_df['geographic_id'] == '37') & 
                            (targets_df['variable'].str.contains('age'))]
if not nc_age_targets.empty:
    nc_target_id = nc_age_targets.iloc[0]['stacked_target_id']
    nc_row = matrix_df.loc[nc_target_id]
    ca_nonzero = (nc_row[ca_households] != 0).sum()
    nc_nonzero = (nc_row[nc_households] != 0).sum()
    print(f"\nNC age target '{nc_target_id}':")
    print(f"  - Non-zero CA households: {ca_nonzero} (should be 0)")
    print(f"  - Non-zero NC households: {nc_nonzero}")

# Check a national target - should have non-zero values for both
national_targets = targets_df[targets_df['geographic_id'] == 'US']
if not national_targets.empty:
    nat_target_id = national_targets.iloc[0]['stacked_target_id']
    nat_row = matrix_df.loc[nat_target_id]
    ca_nonzero = (nat_row[ca_households] != 0).sum()
    nc_nonzero = (nat_row[nc_households] != 0).sum()
    print(f"\nNational target '{nat_target_id}':")
    print(f"  - Non-zero CA households: {ca_nonzero}")
    print(f"  - Non-zero NC households: {nc_nonzero}")

print("\n" + "=" * 70)
print("Stacking test complete!")
