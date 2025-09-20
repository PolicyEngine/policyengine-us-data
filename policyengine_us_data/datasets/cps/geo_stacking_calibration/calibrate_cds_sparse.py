# ============================================================================
# CONFIGURATION
# ============================================================================
import os
# Set before any CUDA operations - helps with memory fragmentation on long runs
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================================================
# IMPORTS
# ============================================================================
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text

import torch
import numpy as np
import pandas as pd
from scipy import sparse as sp
from l0.calibration import SparseCalibrationWeights

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking_sparse import SparseGeoStackingMatrixBuilder
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import create_target_groups, download_from_huggingface





# ============================================================================
# STEP 1: DATA LOADING AND CD LIST RETRIEVAL
# ============================================================================
   
# db_path = download_from_huggingface("policy_data.db")
db_path = '/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db'
db_uri = f"sqlite:///{db_path}"
builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)

# Query all congressional district GEOIDs from database
engine = create_engine(db_uri)
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
    all_cd_geoids = [row[0] for row in result]

print(f"Found {len(all_cd_geoids)} congressional districts in database")

# For testing, use only 10 CDs (can change to all_cd_geoids for full run)
MODE = "Stratified" 
if MODE == "Test":
    # Select 10 diverse CDs from different states
    # Note: CD GEOIDs are 3-4 digits, format is state_fips + district_number
    cds_to_calibrate = [
        '601',   # California CD 1
        '652',   # California CD 52
        '3601',  # New York CD 1
        '3626',  # New York CD 26
        '4801',  # Texas CD 1
        '4838',  # Texas CD 38
        '1201',  # Florida CD 1
        '1228',  # Florida CD 28
        '1701',  # Illinois CD 1
        '1101',  # DC at-large
    ]
    print(f"TEST MODE: Using only {len(cds_to_calibrate)} CDs for testing")
    dataset_uri = "hf://policyengine/test/extended_cps_2023.h5"
elif MODE == "Stratified":
    cds_to_calibrate = all_cd_geoids
    #dataset_uri = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/stratified_extended_cps_2023.h5"
    dataset_uri = "/home/baogorek/devl/stratified_10k.h5"
    print(f"Stratified mode")
else:
    cds_to_calibrate = all_cd_geoids
    dataset_uri = "hf://policyengine/test/extended_cps_2023.h5"
    print(f"FULL MODE (HOPE THERE IS PLENTY RAM!): Using all {len(cds_to_calibrate)} CDs")

sim = Microsimulation(dataset=dataset_uri)

# ============================================================================
# STEP 2: BUILD SPARSE MATRIX
# ============================================================================

print("\nBuilding sparse calibration matrix for congressional districts...")
targets_df, X_sparse, household_id_mapping = builder.build_stacked_matrix_sparse(
    'congressional_district', 
    cds_to_calibrate,
    sim
)

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
# STEP 3: EXPORT FOR GPU PROCESSING
# ============================================================================

# Create export directory
export_dir = os.path.expanduser("~/Downloads/cd_calibration_data")
os.makedirs(export_dir, exist_ok=True)

# Save sparse matrix
sparse_path = os.path.join(export_dir, "cd_matrix_sparse.npz")
sp.save_npz(sparse_path, X_sparse)
print(f"\nExported sparse matrix to: {sparse_path}")

# Create target names array for epoch logging
target_names = []
for _, row in targets_df.iterrows():
    if row['geographic_id'] == 'US':
        name = f"nation/{row['variable']}/{row['description']}"
    elif len(str(row['geographic_id'])) <= 2 or 'state' in row['description'].lower():
        name = f"state{row['geographic_id']}/{row['variable']}/{row['description']}"
    else:
        name = f"CD{row['geographic_id']}/{row['variable']}/{row['description']}"
    target_names.append(name)

# Save target names array (replaces pickled dataframe)
target_names_path = os.path.join(export_dir, "cd_target_names.json")
import json
with open(target_names_path, 'w') as f:
    json.dump(target_names, f)
print(f"Exported target names to: {target_names_path}")

# Save targets array for direct model.fit() use
targets_array_path = os.path.join(export_dir, "cd_targets_array.npy")
np.save(targets_array_path, targets)
print(f"Exported targets array to: {targets_array_path}")

# Save CD list for reference
cd_list_path = os.path.join(export_dir, "cd_list.txt")
with open(cd_list_path, 'w') as f:
    for cd in cds_to_calibrate:
        f.write(f"{cd}\n")
print(f"Exported CD list to: {cd_list_path}")

# ============================================================================
# STEP 4: CALCULATE CD POPULATIONS AND INITIAL WEIGHTS
# ============================================================================

cd_populations = {}
for cd_geoid in cds_to_calibrate:
    cd_age_targets = targets_df[
        (targets_df['geographic_id'] == cd_geoid) & 
        (targets_df['variable'] == 'person_count') &
        (targets_df['description'].str.contains('age', na=False))
    ]
    if not cd_age_targets.empty:
        unique_ages = cd_age_targets.drop_duplicates(subset=['description'])
        cd_populations[cd_geoid] = unique_ages['value'].sum()

if cd_populations:
    min_pop = min(cd_populations.values())
    max_pop = max(cd_populations.values())
    print(f"\nCD population range: {min_pop:,.0f} to {max_pop:,.0f}")
else:
    print("\nWarning: Could not calculate CD populations from targets")
    min_pop = 700000  # Approximate average CD population

# Create arrays for both keep probabilities and initial weights
keep_probs = np.zeros(X_sparse.shape[1])
init_weights = np.zeros(X_sparse.shape[1])
cumulative_idx = 0

# Calculate weights for ALL CDs
for cd_key, household_list in household_id_mapping.items():
    cd_geoid = cd_key.replace('cd', '')
    n_households = len(household_list)
    
    if cd_geoid in cd_populations:
        cd_pop = cd_populations[cd_geoid]
    else:
        cd_pop = min_pop  # Use minimum as default
    
    # Scale initial keep probability by population
    pop_ratio = cd_pop / min_pop
    adjusted_keep_prob = min(0.15, 0.02 * np.sqrt(pop_ratio))
    keep_probs[cumulative_idx:cumulative_idx + n_households] = adjusted_keep_prob
    
    # Calculate initial weight
    base_weight = cd_pop / n_households
    sparsity_adjustment = 1.0 / np.sqrt(adjusted_keep_prob)
    initial_weight = base_weight * sparsity_adjustment
    #initial_weight = np.clip(initial_weight, 0, 100000)   # Not clipping 
    
    init_weights[cumulative_idx:cumulative_idx + n_households] = initial_weight
    cumulative_idx += n_households

print("\nCD-aware keep probabilities and initial weights calculated.")
print(f"Initial weight range: {init_weights.min():.0f} to {init_weights.max():.0f}")
print(f"Mean initial weight: {init_weights.mean():.0f}")

# Save initialization arrays
keep_probs_path = os.path.join(export_dir, "cd_keep_probs.npy")
np.save(keep_probs_path, keep_probs)
print(f"Exported keep probabilities to: {keep_probs_path}")

init_weights_path = os.path.join(export_dir, "cd_init_weights.npy")
np.save(init_weights_path, init_weights)
print(f"Exported initial weights to: {init_weights_path}")

# ============================================================================
# STEP 5: CREATE TARGET GROUPS
# ============================================================================

target_groups, group_info = create_target_groups(targets_df)

print(f"\nAutomatic target grouping:")
print(f"Total groups: {len(np.unique(target_groups))}")
for info in group_info:
    print(f"  {info}")

# Save target groups
target_groups_path = os.path.join(export_dir, "cd_target_groups.npy")
np.save(target_groups_path, target_groups)
print(f"\nExported target groups to: {target_groups_path}")

# ============================================================================
# STEP 6: L0 CALIBRATION WITH EPOCH LOGGING
# ============================================================================

print("\n" + "="*70)
print("RUNNING L0 CALIBRATION WITH EPOCH LOGGING")
print("="*70)

# Create model with per-feature keep probabilities and weights
model = SparseCalibrationWeights(
    n_features=X_sparse.shape[1],
    beta=2/3,
    gamma=-0.1,
    zeta=1.1,
    init_keep_prob=.999, # keep_probs,  # CD-specific keep probabilities
    init_weights=init_weights,  # CD population-based initial weights
    log_weight_jitter_sd=0.05,
    log_alpha_jitter_sd=0.01,
    # device = "cuda",  # Uncomment for GPU
)

# Configuration for epoch logging
ENABLE_EPOCH_LOGGING = True  # Set to False to disable logging
EPOCHS_PER_CHUNK = 2  # Train in chunks of 50 epochs
TOTAL_EPOCHS = 4  # Total epochs to train (set to 3 for quick test)
# For testing, you can use:
# EPOCHS_PER_CHUNK = 1
# TOTAL_EPOCHS = 3

# Initialize CSV files for incremental writing
if ENABLE_EPOCH_LOGGING:
    log_path = os.path.join(export_dir, "cd_calibration_log.csv")
    # Write header
    with open(log_path, 'w') as f:
        f.write('target_name,estimate,target,epoch,error,rel_error,abs_error,rel_abs_error,loss\n')
    print(f"Initialized incremental log at: {log_path}")

# Initialize sparsity tracking CSV with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sparsity_path = os.path.join(export_dir, f"cd_sparsity_history_{timestamp}.csv")
with open(sparsity_path, 'w') as f:
    f.write('epoch,active_weights,total_weights,sparsity_pct\n')
print(f"Initialized sparsity tracking at: {sparsity_path}")

# Train in chunks and capture metrics between chunks
for chunk_start in range(0, TOTAL_EPOCHS, EPOCHS_PER_CHUNK):
    chunk_epochs = min(EPOCHS_PER_CHUNK, TOTAL_EPOCHS - chunk_start)
    current_epoch = chunk_start + chunk_epochs
    
    print(f"\nTraining epochs {chunk_start + 1} to {current_epoch} of {TOTAL_EPOCHS}")
    
    model.fit(
        M=X_sparse,
        y=targets,
        target_groups=target_groups,
        lambda_l0=1.0e-6,
        lambda_l2=0,
        lr=0.2,
        epochs=chunk_epochs,
        loss_type="relative",
        verbose=True,
        verbose_freq=chunk_epochs,  # Print at end of chunk
    )
    
    # Track sparsity after each chunk
    active_info = model.get_active_weights()
    active_count = active_info['count']
    total_count = X_sparse.shape[1]
    sparsity_pct = 100 * (1 - active_count / total_count)
    
    with open(sparsity_path, 'a') as f:
        f.write(f'{current_epoch},{active_count},{total_count},{sparsity_pct:.4f}\n')
    
    if ENABLE_EPOCH_LOGGING:
        # Capture metrics after this chunk
        with torch.no_grad():
            y_pred = model.predict(X_sparse).cpu().numpy()
            
            # Write incrementally to CSV
            with open(log_path, 'a') as f:
                for i in range(len(targets)):
                    # Calculate all metrics
                    estimate = y_pred[i]
                    target = targets[i]
                    error = estimate - target
                    rel_error = error / target if target != 0 else 0
                    abs_error = abs(error)
                    rel_abs_error = abs(rel_error)
                    loss = rel_error ** 2
                    
                    # Write row directly to file
                    f.write(f'"{target_names[i]}",{estimate},{target},{current_epoch},'
                           f'{error},{rel_error},{abs_error},{rel_abs_error},{loss}\n')
        
        # Clear GPU cache after large prediction operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
# Save epoch logging data if enabled
if ENABLE_EPOCH_LOGGING:
    print(f"\nIncremental log complete at: {log_path}")
    print(f"Log contains metrics for {TOTAL_EPOCHS // EPOCHS_PER_CHUNK} logging points")
    
# Final evaluation
with torch.no_grad():
    y_pred = model.predict(X_sparse).cpu().numpy()
    y_actual = targets
    rel_errors = np.abs((y_actual - y_pred) / (y_actual + 1))
    
    print(f"\nAfter {TOTAL_EPOCHS} epochs:")
    print(f"Mean relative error: {np.mean(rel_errors):.2%}")
    print(f"Max relative error: {np.max(rel_errors):.2%}")
    
    # Get sparsity info
    active_info = model.get_active_weights()
    final_sparsity = 100 * (1 - active_info['count'] / X_sparse.shape[1])
    print(f"Active weights: {active_info['count']} out of {X_sparse.shape[1]} ({100*active_info['count']/X_sparse.shape[1]:.2f}%)")
    print(f"Final sparsity: {final_sparsity:.2f}%")
    
    # Save final weights
    w = model.get_weights(deterministic=True).cpu().numpy()
    final_weights_path = os.path.join(export_dir, f"cd_weights_{TOTAL_EPOCHS}epochs.npy")
    np.save(final_weights_path, w)
    print(f"\nSaved final weights ({TOTAL_EPOCHS} epochs) to: {final_weights_path}")
    
print("\n✅ L0 calibration complete! Matrix, targets, and epoch log are ready for analysis.")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("CD CALIBRATION DATA EXPORT COMPLETE")
print("="*70)
print(f"\nAll files exported to: {export_dir}")
print("\nFiles ready for GPU transfer:")
print(f"  1. cd_matrix_sparse.npz - Sparse calibration matrix")
print(f"  2. cd_target_names.json - Target names for epoch logging")
print(f"  3. cd_targets_array.npy - Target values array")
print(f"  4. cd_keep_probs.npy - Initial keep probabilities")
print(f"  5. cd_init_weights.npy - Initial weights")
print(f"  6. cd_target_groups.npy - Target grouping for loss")
print(f"  7. cd_list.txt - List of CD GEOIDs")
if 'w' in locals():
    print(f"  8. cd_weights_{TOTAL_EPOCHS}epochs.npy - Final calibration weights")
if ENABLE_EPOCH_LOGGING:
    print(f"  9. cd_calibration_log.csv - Epoch-by-epoch metrics for dashboard")
print(f"  10. cd_sparsity_history_{timestamp}.csv - Sparsity tracking over epochs")

print("\nTo load on GPU platform:")
print("  import scipy.sparse as sp")
print("  import numpy as np")
print("  import pandas as pd")
print(f"  X = sp.load_npz('{sparse_path}')")
print(f"  targets = np.load('{targets_array_path}')")
print(f"  target_groups = np.load('{target_groups_path}')")
print(f"  keep_probs = np.load('{keep_probs_path}')")
print(f"  init_weights = np.load('{init_weights_path}')")
