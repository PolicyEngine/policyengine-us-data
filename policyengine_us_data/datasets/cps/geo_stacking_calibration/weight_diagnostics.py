import os

import numpy as np
import pandas as pd
from scipy import sparse as sp
from policyengine_us import Microsimulation

from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import create_target_groups, download_from_huggingface

# Load the actual microsimulation that was used to create the calibration matrix
# This is our ground truth for household ordering
print("Loading microsimulation...")
sim = Microsimulation(dataset="hf://policyengine/test/extended_cps_2023.h5")
sim.build_from_dataset()

# Get household IDs in their actual order - this is critical!
household_ids = sim.calculate("household_id", map_to="household").values
n_households_total = len(household_ids)
print(f"Total households in simulation: {n_households_total:,}")

# Verify a few household positions match expectations
print(f"Household at position 5: {household_ids[5]} (expected 17)")
print(f"Household at position 586: {household_ids[586]} (expected 1595)")

X_sparse = sp.load_npz(download_from_huggingface('X_sparse.npz'))

w = np.load("/home/baogorek/Downloads/w_array_20250908_185748.npy")
n_active = sum(w != 0)
print(f"\nSparsity: {n_active} active weights out of {len(w)} ({100*n_active/len(w):.2f}%)")

targets_df = pd.read_pickle(download_from_huggingface('targets_df.pkl'))

# Predictions are simply matrix multiplication: X @ w
y_pred = X_sparse @ w
y_actual = targets_df['value'].values

print(np.corrcoef(y_pred, y_actual))

# Calculate errors
abs_errors = np.abs(y_actual - y_pred)
rel_errors = np.abs((y_actual - y_pred) / (y_actual + 1))  # Adding 1 to avoid division by zero

# Add error columns to targets_df for analysis
targets_df['y_pred'] = y_pred
targets_df['abs_error'] = abs_errors
targets_df['rel_error'] = rel_errors

# Overall statistics
print(f"\nOVERALL ERROR STATISTICS:")
print(f"Mean relative error: {np.mean(rel_errors):.2%}")
print(f"Median relative error: {np.median(rel_errors):.2%}")
print(f"Max relative error: {np.max(rel_errors):.2%}")
print(f"95th percentile error: {np.percentile(rel_errors, 95):.2%}")
print(f"99th percentile error: {np.percentile(rel_errors, 99):.2%}")

# Find worst performing targets
print("\n" + "=" * 70)
print("WORST PERFORMING TARGETS (Top 10)")
print("=" * 70)

worst_targets = targets_df.nlargest(10, 'rel_error')
for idx, row in worst_targets.iterrows():
    state_label = f"State {row['geographic_id']}" if row['geographic_id'] != 'US' else "National"
    print(f"\n{state_label} - {row['variable']} (Group {row['stratum_group_id']})")
    print(f"  Description: {row['description']}")
    print(f"  Target: {row['value']:,.0f}, Predicted: {row['y_pred']:,.0f}")
    print(f"  Relative Error: {row['rel_error']:.1%}")

# Analyze errors by state
print("\n" + "=" * 70)
print("ERROR ANALYSIS BY STATE")
print("=" * 70)

state_errors = targets_df.groupby('geographic_id').agg({
    'rel_error': ['mean', 'median', 'max', 'count']
}).round(4)

# Sort by mean relative error
state_errors = state_errors.sort_values(('rel_error', 'mean'), ascending=False)

print("\nTop 10 states with highest mean relative error:")
for state_id in state_errors.head(10).index:
    state_data = state_errors.loc[state_id]
    n_targets = state_data[('rel_error', 'count')]
    mean_err = state_data[('rel_error', 'mean')]
    max_err = state_data[('rel_error', 'max')]
    median_err = state_data[('rel_error', 'median')]
    
    state_label = f"State {state_id:>2}" if state_id != 'US' else "National"
    print(f"{state_label}: Mean={mean_err:.1%}, Median={median_err:.1%}, Max={max_err:.1%} ({n_targets:.0f} targets)")

# Analyze errors by target type (stratum_group_id)
print("\n" + "=" * 70)
print("ERROR ANALYSIS BY TARGET TYPE")
print("=" * 70)

type_errors = targets_df.groupby('stratum_group_id').agg({
    'rel_error': ['mean', 'median', 'max', 'count']
}).round(4)

# Sort by mean relative error
type_errors = type_errors.sort_values(('rel_error', 'mean'), ascending=False)

# Map numeric group IDs to descriptive names
group_name_map = {
    2: 'Age histogram',
    3: 'AGI distribution', 
    4: 'SNAP',
    5: 'Medicaid',
    6: 'EITC'
}

print("\nError by target type (sorted by mean error):")
for type_id in type_errors.head(10).index:
    type_data = type_errors.loc[type_id]
    n_targets = type_data[('rel_error', 'count')]
    mean_err = type_data[('rel_error', 'mean')]
    max_err = type_data[('rel_error', 'max')]
    median_err = type_data[('rel_error', 'median')]
    
    # Use descriptive name if available
    if type_id in group_name_map:
        type_label = group_name_map[type_id]
    else:
        type_label = str(type_id)[:30]  # Truncate long names
    
    print(f"{type_label:30}: Mean={mean_err:.1%}, Median={median_err:.1%}, Max={max_err:.1%} ({n_targets:.0f} targets)")

# Create automatic target groups for comparison with training
target_groups, group_info = create_target_groups(targets_df)

print("\n" + "=" * 70)
print("GROUP-WISE PERFORMANCE (similar to training output)")
print("=" * 70)

# Calculate group-wise errors similar to training output
group_means = []
for group_id in np.unique(target_groups):
    group_mask = target_groups == group_id
    group_errors = rel_errors[group_mask]
    group_means.append(np.mean(group_errors))

print(f"Mean of group means: {np.mean(group_means):.2%}")
print(f"Max group mean: {np.max(group_means):.2%}")

# Analyze active weights by state
print("\n" + "=" * 70)
print("ACTIVE WEIGHTS ANALYSIS BY STATE")
print("=" * 70)

# The weight vector w has one weight per household copy
# States are arranged sequentially in FIPS order
print(f"\nTotal weights: {len(w)}")
print(f"Active weights (non-zero): {n_active}")

# Define states in calibration order (same as calibrate_states_sparse.py)
states_to_calibrate = [
    '1', '2', '4', '5', '6', '8', '9', '10', '11', '12', '13', '15', '16', '17', '18', 
    '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', 
    '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', 
    '48', '49', '50', '51', '53', '54', '55', '56'
]

# Verify weight vector structure
n_states = len(states_to_calibrate)
n_households_per_state = n_households_total  # From sim
expected_weight_length = n_states * n_households_per_state
print(f"\nWeight vector structure:")
print(f"  States: {n_states}")
print(f"  Households per state: {n_households_per_state:,}")
print(f"  Expected weight length: {expected_weight_length:,}")
print(f"  Actual weight length: {len(w):,}")
assert len(w) == expected_weight_length, "Weight vector length mismatch!"

# Map each weight index to its state and household
weight_to_state = {}
weight_to_household = {}
for state_idx, state_fips in enumerate(states_to_calibrate):
    start_idx = state_idx * n_households_per_state
    for hh_idx, hh_id in enumerate(household_ids):
        weight_idx = start_idx + hh_idx
        weight_to_state[weight_idx] = state_fips
        weight_to_household[weight_idx] = (hh_id, state_fips)

# Count active weights per state
active_weights_by_state = {}
for idx, weight_val in enumerate(w):
    if weight_val != 0:  # Active weight
        state = weight_to_state[idx]
        if state not in active_weights_by_state:
            active_weights_by_state[state] = 0
        active_weights_by_state[state] += 1

# Count total weights available per state (same for all states)
total_weights_by_state = {state: n_households_per_state for state in states_to_calibrate}

# Find states with highest and lowest activation rates
sorted_states = sorted(total_weights_by_state.keys(), key=lambda x: int(x))
activation_rates = [(state, active_weights_by_state.get(state, 0) / total_weights_by_state[state]) 
                   for state in total_weights_by_state.keys()]
activation_rates.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 states by activation rate:")
for state, rate in activation_rates[:5]:
    active = active_weights_by_state.get(state, 0)
    total = total_weights_by_state[state]
    # Get the error for this state from our earlier analysis
    state_targets = targets_df[targets_df['geographic_id'] == state]
    if not state_targets.empty:
        mean_error = state_targets['rel_error'].mean()
        print(f"  State {state}: {100*rate:.1f}% active ({active}/{total}), Mean error: {mean_error:.1%}")
    else:
        print(f"  State {state}: {100*rate:.1f}% active ({active}/{total})")

print("\nBottom 5 states by activation rate:")
for state, rate in activation_rates[-5:]:
    active = active_weights_by_state.get(state, 0)
    total = total_weights_by_state[state]
    state_targets = targets_df[targets_df['geographic_id'] == state]
    if not state_targets.empty:
        mean_error = state_targets['rel_error'].mean()
        print(f"  State {state}: {100*rate:.1f}% active ({active}/{total}), Mean error: {mean_error:.1%}")
    else:
        print(f"  State {state}: {100*rate:.1f}% active ({active}/{total})")

# Weight distribution analysis
print("\n" + "=" * 70)
print("WEIGHT DISTRIBUTION ANALYSIS")
print("=" * 70)

# Collect active weights for each state
weights_by_state = {}
for idx, weight_val in enumerate(w):
    if weight_val != 0:  # Active weight
        state = weight_to_state.get(idx, 'unknown')
        if state not in weights_by_state:
            weights_by_state[state] = []
        weights_by_state[state].append(weight_val)

# Get population targets for each state (total population)
state_populations = {}
for state_fips in sorted_states:
    # Sum all age brackets to get total population
    state_age_targets = targets_df[(targets_df['geographic_id'] == state_fips) & 
                                   (targets_df['variable'] == 'person_count') &
                                   (targets_df['description'].str.contains('age', na=False))]
    if not state_age_targets.empty:
        # Get unique age bracket values (they appear multiple times)
        unique_ages = state_age_targets.drop_duplicates(subset=['description'])
        state_populations[state_fips] = unique_ages['value'].sum()

print("\nPopulation Target Achievement for Key States:")
print("-" * 70)

# Focus on key states 
key_states = ['48', '6', '37', '12', '36', '11', '2']  # Texas, CA, NC, FL, NY, DC, Alaska
state_names = {'48': 'Texas', '6': 'California', '37': 'N. Carolina', '12': 'Florida', 
              '36': 'New York', '11': 'DC', '2': 'Alaska'}

print(f"{'State':<15} {'Population':<15} {'Active':<10} {'Sum Weights':<15} {'Achievement':<12}")
print("-" * 70)

for state_fips in key_states:
    if state_fips in weights_by_state and state_fips in state_populations:
        population_target = state_populations[state_fips]
        active_weights = np.array(weights_by_state[state_fips])
        total_weight = np.sum(active_weights)
        achievement_ratio = total_weight / population_target
        n_active = len(active_weights)
        
        state_label = state_names.get(state_fips, f"State {state_fips}")
        
        print(f"{state_label:<15} {population_target:>14,.0f} {n_active:>9} {total_weight:>14,.0f} {achievement_ratio:>11.1%}")

# Demonstrate extracting weights for specific households
print("\n" + "=" * 70)
print("EXAMPLE: EXTRACTING SPECIFIC HOUSEHOLD WEIGHTS")
print("=" * 70)

# Example: Get weight for household 1595 in Texas (state 48)
example_hh_id = 1595
example_state = '48'

# Find household position in the simulation
hh_position = np.where(household_ids == example_hh_id)[0][0]
state_position = states_to_calibrate.index(example_state)
weight_idx = state_position * n_households_per_state + hh_position

print(f"\nHousehold {example_hh_id} in Texas (state {example_state}):")
print(f"  Position in sim: {hh_position}")
print(f"  State position: {state_position}")
print(f"  Weight index: {weight_idx}")
print(f"  Weight value: {w[weight_idx]:.2f}")

# Show a few more examples
print("\nWeights for household 1595 across different states:")
for state in ['6', '11', '37', '48']:  # CA, DC, NC, TX
    state_pos = states_to_calibrate.index(state)
    w_idx = state_pos * n_households_per_state + hh_position
    state_name = {'6': 'California', '11': 'DC', '37': 'N. Carolina', '48': 'Texas'}[state]
    print(f"  {state_name:12}: {w[w_idx]:10.2f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nFor detailed diagnostics, see CALIBRATION_DIAGNOSTICS.md")
print("\nTo create sparse state-stacked dataset, run:")
print("  python create_sparse_state_stacked.py")