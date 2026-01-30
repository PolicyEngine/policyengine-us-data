"""
Fit calibration weights using L0-regularized optimization.
Prototype script for weight calibration using the l0-python package.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

parser = argparse.ArgumentParser(description="Fit calibration weights")
parser.add_argument(
    "--device",
    default="cpu",
    choices=["cpu", "cuda"],
    help="Device for training (cpu or cuda)",
)
parser.add_argument(
    "--epochs", type=int, default=100, help="Total epochs for training"
)
parser.add_argument(
    "--db-path",
    default=None,
    help="Path to policy_data.db (default: STORAGE_FOLDER/calibration/policy_data.db)",
)
parser.add_argument(
    "--dataset-path", default=None, help="Path to stratified CPS h5 file"
)
args = parser.parse_args()

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from sparse_matrix_builder import SparseMatrixBuilder
from calibration_utils import get_all_cds_from_database

try:
    import torch
    from l0.calibration import SparseCalibrationWeights
except ImportError:
    raise ImportError(
        "l0-python is required for weight fitting. "
        "Install with: pip install policyengine-us-data[l0]"
    )

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = args.device
TOTAL_EPOCHS = args.epochs
EPOCHS_PER_CHUNK = 500  # TODO: need a better way to set this. Remember it can blow up the Vercel app

# Hyperparameters
BETA = 0.35
GAMMA = -0.1
ZETA = 1.1
INIT_KEEP_PROB = 0.999
LOG_WEIGHT_JITTER_SD = 0.05
LOG_ALPHA_JITTER_SD = 0.01
LAMBDA_L0 = 1e-8
LAMBDA_L2 = 1e-12
LEARNING_RATE = 0.15

# Data paths
if args.db_path:
    db_path = Path(args.db_path)
else:
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
db_uri = f"sqlite:///{db_path}"

if args.dataset_path:
    dataset_path = Path(args.dataset_path)
else:
    dataset_path = STORAGE_FOLDER / "stratified_extended_cps_2024.h5"

output_dir = STORAGE_FOLDER / "calibration"
output_dir.mkdir(parents=True, exist_ok=True)
time_period = 2024

# Get all CDs from database
cds_to_calibrate = get_all_cds_from_database(db_uri)
print(f"Found {len(cds_to_calibrate)} congressional districts")

# ============================================================================
# STEP 1: BUILD CALIBRATION MATRIX
# ============================================================================
print(f"Loading simulation from {dataset_path}...")
sim = Microsimulation(dataset=str(dataset_path))
n_households = len(sim.calculate("household_id", map_to="household").values)
print(f"Loaded {n_households:,} households")

print("\nBuilding sparse matrix...")
builder = SparseMatrixBuilder(
    db_uri=db_uri,
    time_period=time_period,
    cds_to_calibrate=cds_to_calibrate,
    dataset_path=str(dataset_path),
)

targets_df, X_sparse, household_id_mapping = builder.build_matrix(
    sim,
    target_filter={
        "stratum_group_ids": [4, 5],  # 4=SNAP households, 5=state income tax
        "variables": [
            "health_insurance_premiums_without_medicare_part_b",
            "snap",
            "state_income_tax",  # Census STC state income tax collections
        ],
    },
)

print(f"Matrix shape: {X_sparse.shape}")
print(f"Targets: {len(targets_df)}")

# Filter to achievable targets (rows with non-zero data)
row_sums = np.array(X_sparse.sum(axis=1)).flatten()
achievable_mask = row_sums > 0
n_achievable = achievable_mask.sum()
n_impossible = (~achievable_mask).sum()

print(f"\nAchievable targets: {n_achievable}")
print(f"Impossible targets (filtered out): {n_impossible}")

targets_df = targets_df[achievable_mask].reset_index(drop=True)
X_sparse = X_sparse[achievable_mask, :]

print(f"Filtered matrix shape: {X_sparse.shape}")

# Extract target vector and names
targets = targets_df["value"].values
target_names = [
    f"{row['geographic_id']}/{row['variable']}"
    for _, row in targets_df.iterrows()
]

# ============================================================================
# STEP 2: INITIALIZE WEIGHTS
# ============================================================================
initial_weights = np.ones(X_sparse.shape[1]) * 100
print(f"\nInitial weights shape: {initial_weights.shape}")
print(f"Initial weights sum: {initial_weights.sum():,.0f}")

# ============================================================================
# STEP 3: CREATE MODEL
# ============================================================================
print("\nCreating SparseCalibrationWeights model...")
model = SparseCalibrationWeights(
    n_features=X_sparse.shape[1],
    beta=BETA,
    gamma=GAMMA,
    zeta=ZETA,
    init_keep_prob=INIT_KEEP_PROB,
    init_weights=initial_weights,
    log_weight_jitter_sd=LOG_WEIGHT_JITTER_SD,
    log_alpha_jitter_sd=LOG_ALPHA_JITTER_SD,
    device=DEVICE,
)

# ============================================================================
# STEP 4: TRAIN IN CHUNKS
# ============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
calibration_log = pd.DataFrame()

for chunk_start in range(0, TOTAL_EPOCHS, EPOCHS_PER_CHUNK):
    chunk_epochs = min(EPOCHS_PER_CHUNK, TOTAL_EPOCHS - chunk_start)
    current_epoch = chunk_start + chunk_epochs

    print(f"\nTraining epochs {chunk_start + 1} to {current_epoch}...")

    model.fit(
        M=X_sparse,
        y=targets,
        target_groups=None,
        lambda_l0=LAMBDA_L0,
        lambda_l2=LAMBDA_L2,
        lr=LEARNING_RATE,
        epochs=chunk_epochs,
        loss_type="relative",
        verbose=True,
        verbose_freq=chunk_epochs,
    )

    with torch.no_grad():
        predictions = model.predict(X_sparse).cpu().numpy()

    chunk_df = pd.DataFrame(
        {
            "target_name": target_names,
            "estimate": predictions,
            "target": targets,
        }
    )
    chunk_df["epoch"] = current_epoch
    chunk_df["error"] = chunk_df.estimate - chunk_df.target
    chunk_df["rel_error"] = chunk_df.error / chunk_df.target
    chunk_df["abs_error"] = chunk_df.error.abs()
    chunk_df["rel_abs_error"] = chunk_df.rel_error.abs()
    chunk_df["loss"] = chunk_df.rel_abs_error**2
    calibration_log = pd.concat([calibration_log, chunk_df], ignore_index=True)

# ============================================================================
# STEP 5: EXTRACT AND SAVE WEIGHTS
# ============================================================================
with torch.no_grad():
    w = model.get_weights(deterministic=True).cpu().numpy()

print(f"\nFinal weights shape: {w.shape}")
print(f"Final weights sum: {w.sum():,.0f}")
print(f"Non-zero weights: {(w > 0).sum():,}")

output_path = output_dir / f"calibration_weights_{timestamp}.npy"
np.save(output_path, w)
print(f"\nWeights saved to: {output_path}")
print(f"OUTPUT_PATH:{output_path}")

log_path = output_dir / f"calibration_log_{timestamp}.csv"
calibration_log.to_csv(log_path, index=False)
print(f"Calibration log saved to: {log_path}")
print(f"LOG_PATH:{log_path}")

# ============================================================================
# STEP 6: VERIFY PREDICTIONS
# ============================================================================
print("\n" + "=" * 60)
print("PREDICTION VERIFICATION")
print("=" * 60)

with torch.no_grad():
    predictions = model.predict(X_sparse).cpu().numpy()

for i in range(len(targets)):
    rel_error = (predictions[i] - targets[i]) / targets[i] * 100
    print(
        f"{target_names[i][:50]:50} | "
        f"pred: {predictions[i]:>12,.0f} | "
        f"target: {targets[i]:>12,.0f} | "
        f"err: {rel_error:>6.2f}%"
    )

print("\n" + "=" * 60)
print("FITTING COMPLETED")
print("=" * 60)
