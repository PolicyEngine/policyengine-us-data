from pathlib import Path
import os
import tempfile
import urllib.request
import time

import torch
import numpy as np
import pandas as pd
from scipy import sparse as sp
from l0.calibration import SparseCalibrationWeights

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking_sparse import SparseGeoStackingMatrixBuilder
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import create_target_groups


def download_from_huggingface(file_name):
    """Download a file from HuggingFace to a temporary location."""
    base_url = "https://huggingface.co/policyengine/test/resolve/main/"
    url = base_url + file_name
    
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, file_name)
    
    # Check if already downloaded
    if not os.path.exists(local_path):
        print(f"Downloading {file_name} from HuggingFace...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded to {local_path}")
    else:
        print(f"Using cached {local_path}")
    
    return local_path

# Setup - Download database from HuggingFace
db_path = download_from_huggingface("policy_data.db")
db_uri = f"sqlite:///{db_path}"
builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)

print("Loading microsimulation with extended_cps_2023.h5...")
sim = Microsimulation(dataset="hf://policyengine/test/extended_cps_2023.h5")
sim.build_from_dataset()

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

targets_df, X_sparse, household_id_mapping = builder.build_stacked_matrix_sparse(
    'state', 
    states_to_calibrate,
    sim
)

print(f"\nSparse Matrix Statistics:")
print(f"- Shape: {X_sparse.shape}")
print(f"- Non-zero elements: {X_sparse.nnz:,}")
print(f"- Percent non-zero: {100 * X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.4f}%")
print(f"- Memory usage: {(X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes) / 1024**2:.2f} MB")

# Compare to dense matrix memory
dense_memory = X_sparse.shape[0] * X_sparse.shape[1] * 4 / 1024**2  # 4 bytes per float32, in MB
print(f"- Dense matrix would use: {dense_memory:.2f} MB")
print(f"- Memory savings: {100*(1 - (X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes)/(dense_memory * 1024**2)):.2f}%")


# Calibrate using our L0 package ---------------
   
# TRAINING PARAMETERS
EPOCHS_PER_TEMPERATURE = 50  # Number of epochs for each temperature stage
VERBOSE_FREQ = 10  # How often to print training updates

# Initialize weights based on state population sizes
    
state_populations = {}
for state_fips in states_to_calibrate:
    state_age_targets = targets_df[
        (targets_df['geographic_id'] == state_fips) & 
        (targets_df['variable'] == 'person_count') &
        (targets_df['description'].str.contains('age', na=False))
    ]
    if not state_age_targets.empty:
        unique_ages = state_age_targets.drop_duplicates(subset=['description'])
        state_populations[state_fips] = unique_ages['value'].sum()

# Find min population for normalization (DC is smallest)
min_pop = min(state_populations.values())

# Create arrays for both keep probabilities and initial weights
keep_probs = np.zeros(X_sparse.shape[1])
init_weights = np.zeros(X_sparse.shape[1])
cumulative_idx = 0

# Calculate weights for ALL states (not just a subset!)
for state_key, household_list in household_id_mapping.items():
    state_fips = state_key.replace('state', '')
    n_households = len(household_list)
    state_pop = state_populations[state_fips]
    
    # Scale initial keep probability by population
    # Larger states get higher initial keep probability
    pop_ratio = state_pop / min_pop
    # Use sqrt to avoid too extreme differences
    adjusted_keep_prob = min(0.15, 0.02 * np.sqrt(pop_ratio))
    keep_probs[cumulative_idx:cumulative_idx + n_households] = adjusted_keep_prob
    
    # Calculate initial weight based on population and expected sparsity
    # Base weight: population / n_households gives weight if all households were used
    base_weight = state_pop / n_households
    
    # Adjust for expected sparsity: if only keep_prob fraction will be active,
    # those that remain need higher weights
    # But don't fully compensate (use sqrt) to avoid extreme initial values
    sparsity_adjustment = 1.0 / np.sqrt(adjusted_keep_prob)
    
    # Set initial weight with some reasonable bounds
    initial_weight = base_weight * sparsity_adjustment
    initial_weight = np.clip(initial_weight, 100, 100000)  # Reasonable bounds
    
    init_weights[cumulative_idx:cumulative_idx + n_households] = initial_weight
    
    cumulative_idx += n_households

print("State-aware keep probabilities and initial weights calculated.")
print(f"Initial weight range: {init_weights.min():.0f} to {init_weights.max():.0f}")
print(f"Mean initial weight: {init_weights.mean():.0f}")

# Show a few example states for verification (just for display, all states were processed above)
print("\nExample initial weights by state:")
cumulative_idx = 0
states_to_show = ['6', '37', '48', '11', '2']  # CA, NC, TX, DC, AK - just examples
for state_key, household_list in household_id_mapping.items():
    state_fips = state_key.replace('state', '')
    n_households = len(household_list)
    if state_fips in states_to_show:
        state_weights = init_weights[cumulative_idx:cumulative_idx + n_households]
        print(f"  State {state_fips:>2}: pop={state_populations[state_fips]:>10,.0f}, "
              f"weight={state_weights[0]:>7.0f}, keep_prob={keep_probs[cumulative_idx]:.3f}")
    cumulative_idx += n_households

# Create model with per-feature keep probabilities and weights
model = SparseCalibrationWeights(
    n_features=X_sparse.shape[1],
    beta=2/3,  # From paper. We have the option to override it during fitting 
    gamma=-0.1,  # Keep as in paper
    zeta=1.1,    # Keep as in paper
    init_keep_prob=keep_probs,  # Per-household keep probabilities based on state
    init_weights=init_weights,  # Population-based initial weights (ALL states, not just examples!)
    log_weight_jitter_sd=0.05,  # Small jitter to log weights just to break symmetry
)

# Create automatic target groups
target_groups, group_info = create_target_groups(targets_df)

print(f"\nAutomatic target grouping:")
print(f"Total groups: {len(np.unique(target_groups))}")
for info in group_info:
    print(f"  {info}")

start_time = time.perf_counter()

#model.beta = 1.5  # Warm start, if we want
model.fit(
    M=X_sparse,
    y=targets_df.value.values,
    target_groups=target_groups,
    lambda_l0=1.0e-7,  # Note that we can change this as we go, start gentle & go higher
    lambda_l2=0,
    lr=0.2,  # Lower learning rate for warm-up
    epochs=EPOCHS_PER_TEMPERATURE,
    loss_type="relative",
    verbose=True,
    verbose_freq=VERBOSE_FREQ,
)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Total fitting time: {elapsed_time:.4f} seconds.")

# Evaluation
with torch.no_grad():
    y_pred = model.predict(X_sparse).cpu().numpy()
    y_actual = targets_df.value.values
    rel_errors = np.abs((y_actual - y_pred) / (y_actual + 1))
    
    print("\n" + "="*70)
    print("FINAL RESULTS BY GROUP")
    print("="*70)
    
    for group_id in np.unique(target_groups):
        group_mask = target_groups == group_id
        group_errors = rel_errors[group_mask]
        mean_err = np.mean(group_errors)
        max_err = np.max(group_errors)
        
        # Find the group info
        group_label = group_info[group_id]
        print(f"{group_label}:")
        print(f"  Mean error: {mean_err:.2%}, Max error: {max_err:.2%}")
    
    # Get final weights for saving
    weights = model.get_weights(deterministic=True).cpu().numpy()
    active_info = model.get_active_weights()
    print(f"\nFinal sparsity: {active_info['count']} active weights out of {len(weights)} ({100*active_info['count']/len(weights):.2f}%)")
    
    # Save weights if needed
    # np.save("/path/to/save/weights.npy", weights)



# Load weights from Colab notebook
w = np.load("/home/baogorek/Downloads/w2.npy")
n_active = sum(w != 0)
print(f"\nFinal sparsity: {n_active} active weights out of {len(w)} ({100*n_active/len(w):.2f}%)")

# Compute predictions using loaded weights
print("\n" + "=" * 70)
print("COMPUTING PREDICTIONS AND ANALYZING ERRORS")
print("=" * 70)

# Predictions are simply matrix multiplication: X @ w
y_pred = X_sparse @ w
y_actual = targets_df['value'].values

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
# household_id_mapping maps state keys to lists of household indices
print(f"\nTotal weights: {len(w)}")
print(f"Active weights (non-zero): {n_active}")

# Map each weight index to its state
weight_to_state = {}
cumulative_index = 0
for state_key, household_list in household_id_mapping.items():
    # Extract state FIPS from the key (e.g., 'state6' -> '6')
    state_fips = state_key.replace('state', '')
    for i in range(len(household_list)):
        weight_to_state[cumulative_index] = state_fips
        cumulative_index += 1

# Count active weights per state
active_weights_by_state = {}
for idx, weight_val in enumerate(w):
    if weight_val != 0:  # Active weight
        state = weight_to_state.get(idx, 'unknown')
        if state not in active_weights_by_state:
            active_weights_by_state[state] = 0
        active_weights_by_state[state] += 1

# Also count total weights available per state
total_weights_by_state = {}
for state_key, household_list in household_id_mapping.items():
    state_fips = state_key.replace('state', '')
    total_weights_by_state[state_fips] = len(household_list)

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

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nFor detailed diagnostics, see CALIBRATION_DIAGNOSTICS.md")
