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
    init_weight_scale=0.5,
)

# Create automatic target groups based on metadata
def create_target_groups(targets_df):
    """
    Automatically create target groups based on metadata.
    
    Grouping rules:
    1. Each national hardcoded target gets its own group (singleton)
       - These are scalar values like "tip_income" or "medical_expenses" 
       - Each one represents a fundamentally different quantity
       - We want each to contribute equally to the loss
       
    2. All demographic targets grouped by (geographic_id, stratum_group_id)
       - All 18 age bins for California form ONE group
       - All 18 age bins for North Carolina form ONE group
       - This prevents age variables from dominating the loss
       
    The result is that each group contributes equally to the total loss,
    regardless of how many individual targets are in the group.
    """
    target_groups = np.zeros(len(targets_df), dtype=int)
    group_id = 0
    group_info = []
    
    print("\n=== Creating Target Groups ===")
    
    # Process national hardcoded targets first - each gets its own group
    national_mask = targets_df['stratum_group_id'] == 'national_hardcoded'
    national_targets = targets_df[national_mask]
    
    if len(national_targets) > 0:
        print(f"\nNational hardcoded targets (each is a singleton group):")
        for idx in national_targets.index:
            target = targets_df.loc[idx]
            var_name = target['variable']
            value = target['value']
            
            target_groups[idx] = group_id
            group_info.append(f"Group {group_id}: National {var_name} (1 target, value={value:,.0f})")
            print(f"  Group {group_id}: {var_name} = {value:,.0f}")
            group_id += 1
    
    # Process demographic targets - grouped by stratum_group_id ONLY (not geography)
    # This ensures all age targets across all states form ONE group
    demographic_mask = ~national_mask
    demographic_df = targets_df[demographic_mask]
    
    if len(demographic_df) > 0:
        print(f"\nDemographic targets (grouped by type across ALL geographies):")
        
        # Get unique stratum_group_ids (NOT grouped by geography)
        unique_stratum_groups = demographic_df['stratum_group_id'].unique()
        
        for stratum_group in unique_stratum_groups:
            # Find ALL targets with this stratum_group_id across ALL geographies
            mask = (targets_df['stratum_group_id'] == stratum_group)
            
            matching_targets = targets_df[mask]
            target_groups[mask] = group_id
            
            # Create descriptive label
            stratum_labels = {
                1: 'Geographic',  # This shouldn't appear in demographic targets
                2: 'Age',
                3: 'Income/AGI',
                4: 'SNAP',
                5: 'Medicaid', 
                6: 'EITC'
            }
            stratum_name = stratum_labels.get(stratum_group, f'Unknown({stratum_group})')
            n_targets = mask.sum()
            
            # Count unique geographies in this group
            unique_geos = matching_targets['geographic_id'].unique()
            n_geos = len(unique_geos)
            
            # Get geographic breakdown
            geo_counts = matching_targets.groupby('geographic_id').size()
            state_names = {'6': 'California', '37': 'North Carolina'}
            geo_breakdown = []
            for geo_id, count in geo_counts.items():
                geo_name = state_names.get(geo_id, f'State {geo_id}')
                geo_breakdown.append(f"{geo_name}: {count}")
            
            group_info.append(f"Group {group_id}: All {stratum_name} targets ({n_targets} total)")
            print(f"  Group {group_id}: {stratum_name} histogram across {n_geos} geographies ({n_targets} total targets)")
            print(f"    Geographic breakdown: {', '.join(geo_breakdown)}")
            
            # Show sample targets from different geographies
            if n_geos > 1 and n_targets > 3:
                for geo_id in unique_geos[:2]:  # Show first two geographies
                    geo_name = state_names.get(geo_id, f'State {geo_id}')
                    geo_targets = matching_targets[matching_targets['geographic_id'] == geo_id]
                    print(f"    {geo_name} samples:")
                    print(f"      - {geo_targets.iloc[0]['description']}")
                    if len(geo_targets) > 1:
                        print(f"      - {geo_targets.iloc[-1]['description']}")
            
            group_id += 1
    
    print(f"\nTotal groups created: {group_id}")
    print("=" * 40)
    
    return target_groups, group_info

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
    lambda_l0=0.0000015,
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
