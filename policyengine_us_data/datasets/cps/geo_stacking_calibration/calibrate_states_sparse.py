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


if True:
    # Calibrate using our L0 package
    from l0.calibration import SparseCalibrationWeights
    import torch
    
    # The sparse matrix is already in CSR format
    X_sparse = sparse_matrix
    
    # TRAINING PARAMETERS
    EPOCHS_PER_TEMPERATURE = 50  # Number of epochs for each temperature stage
    
    # IMPROVED INITIALIZATION SETTINGS
    model = SparseCalibrationWeights(
        n_features=X_sparse.shape[1],
        beta=0.66,  # Keep as in paper
        gamma=-0.1,  # Keep as in paper
        zeta=1.1,    # Keep as in paper
        init_keep_prob=0.05,  # Start closer to target sparsity (was 0.3)
        init_weight_scale=0.5,  # Initial log weight scale (standard deviation)
        log_weight_jitter_sd=0.01,  # Small jitter to break symmetry
    )
    
    # Optional: State-aware initialization
    # This gives high-population states a better chance of keeping weights active
    if True:  # Set to True to enable state-aware initialization
        print("\nApplying state-aware initialization...")
        
        # Calculate state populations from targets
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
        
        # Adjust initial log_alpha values based on state population
        with torch.no_grad():
            cumulative_idx = 0
            for state_key, household_list in household_id_mapping.items():
                state_fips = state_key.replace('state', '')
                n_households = len(household_list)
                
                if state_fips in state_populations:
                    # Scale initial keep probability by population
                    # Larger states get higher initial keep probability
                    pop_ratio = state_populations[state_fips] / min_pop
                    # Use sqrt to avoid too extreme differences
                    adjusted_keep_prob = min(0.15, 0.02 * np.sqrt(pop_ratio))
                    
                    # Convert to log_alpha with small jitter to break symmetry
                    mu = np.log(adjusted_keep_prob / (1 - adjusted_keep_prob))
                    jitter = np.random.normal(0, 0.01, n_households)
                    model.log_alpha.data[cumulative_idx:cumulative_idx + n_households] = torch.tensor(
                        mu + jitter, dtype=torch.float32
                    )
                
                cumulative_idx += n_households
        
        print("State-aware initialization complete.")
    
    # Create automatic target groups
    target_groups, group_info = create_target_groups(targets_df)
    
    print(f"\nAutomatic target grouping:")
    print(f"Total groups: {len(np.unique(target_groups))}")
    for info in group_info:
        print(f"  {info}")
    
    import time
    
    # OPTION 1: Single-stage training with improved parameters
    if False:  # Set to False to use multi-stage training instead
        print("\nUsing single-stage training with improved parameters...")
        start_time = time.perf_counter()
        
        model.fit(
            M=X_sparse,
            y=targets_df.value.values,
            target_groups=target_groups,
            lambda_l0=1.0e-7,  # Less aggressive sparsity (was 1.5e-7)
            lambda_l2=0,
            lr=0.15,  # Slightly lower learning rate (was 0.2)
            epochs=50,
            loss_type="relative",
            verbose=True,
            verbose_freq=500,
        )
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Fitting the model took {elapsed_time:.4f} seconds.")
    
    # OPTION 2: Multi-stage training with temperature annealing
    else:
        print("\nUsing multi-stage training with temperature annealing...")
        start_time = time.perf_counter()
        
        # Stage 1: Warm start with higher temperature (softer decisions)
        print(f"\nStage 1: Warm-up (beta=1.5, {EPOCHS_PER_TEMPERATURE} epochs)")
        model.beta = 1.5
        model.fit(
            M=X_sparse,
            y=targets_df.value.values,
            target_groups=target_groups,
            lambda_l0=0.5e-7,  # Very gentle sparsity at first
            lambda_l2=0,
            lr=0.1,  # Lower learning rate for warm-up
            epochs=EPOCHS_PER_TEMPERATURE,
            loss_type="relative",
            verbose=True,
            verbose_freq=10,
        )
        
        # Stage 2: Intermediate temperature
        print(f"\nStage 2: Cooling (beta=1.0, {EPOCHS_PER_TEMPERATURE} epochs)")
        model.beta = 1.0
        model.fit(
            M=X_sparse,
            y=targets_df.value.values,
            target_groups=target_groups,
            lambda_l0=0.8e-7,  # Increase sparsity pressure
            lambda_l2=0,
            lr=0.15,
            epochs=EPOCHS_PER_TEMPERATURE,
            loss_type="relative",
            verbose=True,
            verbose_freq=10,
        )
        
        # Stage 3: Final temperature (as in paper)
        print(f"\nStage 3: Final (beta=0.66, {EPOCHS_PER_TEMPERATURE} epochs)")
        model.beta = 0.66
        model.fit(
            M=X_sparse,
            y=targets_df.value.values,
            target_groups=target_groups,
            lambda_l0=1.0e-7,  # Final sparsity level
            lambda_l2=0,
            lr=0.2,  # Can be more aggressive now
            epochs=EPOCHS_PER_TEMPERATURE,
            loss_type="relative",
            verbose=True,
            verbose_freq=10,
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
y_pred = sparse_matrix @ w
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
