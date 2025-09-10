==============================================================
# IMPORTS
# ============================================================================
from pathlib import Path
import os

import torch
import numpy as np
import pandas as pd
from scipy import sparse as sp
from l0.calibration import SparseCalibrationWeights

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking_sparse import SparseGeoStackingMatrixBuilder
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import create_target_groups, download_from_huggingface


# ============================================================================
# STEP 1: DATA LOADING AND MATRIX BUILDING
# ============================================================================
   
db_path = download_from_huggingface("policy_data.db")
db_uri = f"sqlite:///{db_path}"
builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)

sim = Microsimulation(dataset="hf://policyengine/test/extended_cps_2023.h5")
sim.build_from_dataset()

# TODO: where is the cannonical list of geos now? Because you don't want to have this
# list for the 436 congressional districts?
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

targets_df, X_sparse, household_id_mapping = builder.build_stacked_matrix_sparse(
    'state', 
    states_to_calibrate,
    sim
)

# NOTE: I'm not really sure what household_id_mapping gets us, because every state has
# Every household in this "empirical pseudopopulation" approach

targets_df.to_pickle('~/Downloads/targets_df.pkl')

targets = targets_df.value.values

print(f"\nSparse Matrix Statistics:")
print(f"- Shape: {X_sparse.shape}")
print(f"- Non-zero elements: {X_sparse.nnz:,}")
print(f"- Percent non-zero: {100 * X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.4f}%")
print(f"- Memory usage: {(X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes) / 1024**2:.2f} MB")

# Compare to dense matrix memory
dense_memory = X_sparse.shape[0] * X_sparse.shape[1] * 4 / 1024**2  # 4 bytes per float32, in MB
print(f"- Dense matrix would use: {dense_memory:.2f} MB")
print(f"- Memory savings: {100*(1 - (X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes)/(dense_memory * 1024**2)):.2f}%")

# ============================================================================
# STEP 2: MODEL INITIALIZATION
# ============================================================================

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


# Create target groups -------
target_groups, group_info = create_target_groups(targets_df)

print(f"\nAutomatic target grouping:")
print(f"Total groups: {len(np.unique(target_groups))}")
for info in group_info:
    print(f"  {info}")


# Downloads -------
downloads_dir = os.path.expanduser("~/Downloads")

# Save sparse matrix using scipy's native format
sparse_path = os.path.join(downloads_dir, "X_sparse.npz")
sp.save_npz(sparse_path, X_sparse)

# Save targets array separately for direct model.fit() use
targets_array_path = os.path.join(downloads_dir, "targets_array.npy")
np.save(targets_array_path, targets)

target_groups_array_path = os.path.join(downloads_dir, "target_groups_array.npy")
np.save(target_groups_array_path, target_groups)

keep_probs_array_path = os.path.join(downloads_dir, "keep_probs_array.npy")
np.save(keep_probs_array_path, keep_probs)

init_weights_array_path = os.path.join(downloads_dir, "init_weights_array.npy")
np.save(init_weights_array_path, init_weights)


# ============================================================================
# MODEL CREATION - THIS IS THE KEY SECTION FOR KAGGLE
# ============================================================================
# Training parameters
EPOCHS_PER_TEMPERATURE = 100  # Number of epochs for each temperature stage
VERBOSE_FREQ = 10  # How often to print training updates

# Create model with per-feature keep probabilities and weights
model = SparseCalibrationWeights(
    n_features=X_sparse.shape[1],
    beta=2/3,  # From paper. We have the option to override it during fitting 
    gamma=-0.1,  # Keep as in paper
    zeta=1.1,    # Keep as in paper
    init_keep_prob=.999,  #keep_probs,  # Per-household keep probabilities based on state
    init_weights=init_weights,  # Population-based initial weights (ALL states, not just examples!)
    log_weight_jitter_sd=0.05,  # Small jitter to log weights at fit() time to help escape local minima
    log_alpha_jitter_sd=0.01,   # Small jitter to log_alpha at init to break gate symmetry (Louizos et al.)
    # device = "cuda",  # Uncomment for GPU in Kaggle
)

# ============================================================================
# MODEL FITTING - MAIN TRAINING CALL
# ============================================================================

# model.beta = 1.5  # Warm start, if we want
model.fit(
    M=X_sparse,                     # Input: Sparse matrix (CSR format)
    y=targets,      # Input: Target values as numpy array
    target_groups=target_groups,    # Groups for stratified evaluation
    lambda_l0=1.5e-6,  # L0 regularization strength
    lambda_l2=0,       # L2 regularization (0 = disabled)
    lr=0.2,            # Learning rate
    epochs=EPOCHS_PER_TEMPERATURE,
    loss_type="relative",
    verbose=True,
    verbose_freq=VERBOSE_FREQ,
)

# ============================================================================
# STEP 3: EVALUATION (quick) AND WEIGHT EXTRACTION
# ============================================================================

with torch.no_grad():
    y_pred = model.predict(X_sparse).cpu().numpy()
    y_actual = targets
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
    w = model.get_weights(deterministic=True).cpu().numpy()
    active_info = model.get_active_weights()
    print(f"\nFinal sparsity: {active_info['count']} active weights out of {len(w)} ({100*active_info['count']/len(w):.2f}%)")
    
    # Save weights
    weights_path = os.path.expanduser("~/Downloads/calibrated_weights.npy")
    np.save(weights_path, w)
    print(f"\nSaved calibrated weights to: {weights_path}")
