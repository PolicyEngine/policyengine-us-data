import os
import numpy as np
import pandas as pd
import pickle
from scipy import sparse as sp
from holdout_validation import run_holdout_experiment, simple_holdout

# Load the calibration package
export_dir = os.path.expanduser("~/Downloads/cd_calibration_data")
package_path = os.path.join(export_dir, "calibration_package.pkl")

print(f"Loading calibration package from: {package_path}")
with open(package_path, 'rb') as f:
    data = pickle.load(f)

print(f"Keys in data: {data.keys()}")

X_sparse = data['X_sparse']
targets_df = data['targets_df']
targets = targets_df.value.values
target_groups = data['target_groups']
init_weights = data['initial_weights']
keep_probs = data['keep_probs']

print(f"Loaded {len(targets_df)} targets")
print(f"Target groups shape: {target_groups.shape}")
print(f"Unique groups: {len(np.unique(target_groups))}")

# EXPLORE TARGET GROUPS ----------------------------
unique_groups = np.unique(target_groups)
group_details = []

print(f"\nProcessing {len(unique_groups)} groups...")

for group_id in unique_groups:
    group_mask = target_groups == group_id
    group_targets = targets_df[group_mask].copy()
    
    n_targets = len(group_targets)
    geos = group_targets['geographic_id'].unique()
    variables = group_targets['variable'].unique()
    var_descs = group_targets['variable_desc'].unique()
    
    # Classify the group type
    if len(geos) == 1 and len(variables) == 1:
        if len(var_descs) > 1:
            group_type = f"Single geo/var with {len(var_descs)} bins"
        else:
            group_type = "Single target"
    elif len(geos) > 1 and len(variables) == 1:
        group_type = f"Multi-geo ({len(geos)} geos), single var"
    else:
        group_type = f"Complex: {len(geos)} geos, {len(variables)} vars"
    
    detail = {
        'group_id': group_id,
        'n_targets': n_targets,
        'group_type': group_type,
        'geos': list(geos)[:3],  # First 3 for display
        'n_geos': len(geos),
        'variable': variables[0] if len(variables) == 1 else f"{len(variables)} vars",
        'sample_desc': var_descs[0] if len(var_descs) > 0 else None
    }
    group_details.append(detail)

groups_df = pd.DataFrame(group_details)

if groups_df.empty:
    print("WARNING: groups_df is empty!")
    print(f"group_details has {len(group_details)} items")
    if len(group_details) > 0:
        print(f"First item: {group_details[0]}")
else:
    print(f"\nCreated groups_df with {len(groups_df)} rows")

# Improve the variable column for complex groups
for idx, row in groups_df.iterrows():
    if '2 vars' in str(row['variable']) or 'vars' in str(row['variable']):
        # Get the actual variables for this group
        group_mask = target_groups == row['group_id']
        group_targets = targets_df[group_mask]
        variables = group_targets['variable'].unique()
        # Update with actual variable names
        groups_df.at[idx, 'variable'] = ', '.join(variables[:2])

# Show all groups for selection
print("\nAll target groups (use group_id for selection):")
print(groups_df[['group_id', 'n_targets', 'variable', 'group_type']].to_string())

# CSV export moved to end of file after results

# INTERACTIVE HOLDOUT SELECTION -------------------------------

# EDIT THIS LINE: Choose your group_id values from the table above
N_GROUPS = groups_df.shape[0]

age_ids = [30]
first_5_national_ids = [0, 1, 2, 3, 4]
second_5_national_ids = [5, 6, 7, 8, 9]
third_5_national_ids = [10, 11, 12, 13, 14]
agi_histogram_ids = [31]
agi_value_ids = [33]
eitc_cds_value_ids = [35]
last_15_national_ids = [i for i in range(15, 30)]

union_ids = (
    age_ids + first_5_national_ids + second_5_national_ids + third_5_national_ids + agi_histogram_ids
    + agi_value_ids + eitc_cds_value_ids + last_15_national_ids
)

len(union_ids)

holdout_group_ids = [i for i in range(N_GROUPS) if i not in union_ids]
len(holdout_group_ids)


# Make age the only holdout:
union_ids = [i for i in range(N_GROUPS) if i not in age_ids]
holdout_group_ids = age_ids 

assert len(union_ids) + len(holdout_group_ids) == N_GROUPS

results = simple_holdout(
    X_sparse=X_sparse,
    targets=targets,
    target_groups=target_groups,
    init_weights=init_weights,
    holdout_group_ids=holdout_group_ids,
    targets_df=targets_df,  # Pass targets_df for hierarchical analysis
    check_hierarchical=True,  # Enable hierarchical consistency check
    epochs=2000,
    lambda_l0=0, #8e-7,
    lr=0.3,
    verbose_spacing=100,
    device='cpu',
)

# CREATE RESULTS DATAFRAME
# Build a comprehensive results dataframe
results_data = []

# Add training groups
for group_id, loss in results['train_group_losses'].items():
    # Get group info from original groups_df
    if group_id in groups_df['group_id'].values:
        group_info = groups_df[groups_df['group_id'] == group_id].iloc[0]
        results_data.append({
            'group_id': group_id,
            'set': 'train',
            'loss': loss,
            'n_targets': group_info['n_targets'],
            'variable': group_info['variable'],
            'group_type': group_info['group_type']
        })

# Add holdout groups (now using original IDs directly)
for group_id, loss in results['holdout_group_losses'].items():
    if group_id in groups_df['group_id'].values:
        group_info = groups_df[groups_df['group_id'] == group_id].iloc[0]
        results_data.append({
            'group_id': group_id,
            'set': 'holdout',
            'loss': loss,
            'n_targets': group_info['n_targets'],
            'variable': group_info['variable'],
            'group_type': group_info['group_type']
        })

results_df = pd.DataFrame(results_data)
results_df = results_df.sort_values(['set', 'loss'], ascending=[True, False])
