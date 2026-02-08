"""
National L0 calibration for Enhanced CPS.

L0-regularized optimization via l0-python's SparseCalibrationWeights.
Reads active targets from policy_data.db via NationalMatrixBuilder.

Usage:
    python -m policyengine_us_data.calibration.fit_national_weights \\
        --dataset path/to/extended_cps_2024.h5 \\
        --db-path path/to/policy_data.db \\
        --output path/to/enhanced_cps_2024.h5 \\
        --epochs 1000 \\
        --lambda-l0 1e-6
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import scipy.sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# HYPERPARAMETERS (higher L0 than local mode for more sparsification)
# ============================================================================
LAMBDA_L0 = 1e-6
LAMBDA_L2 = 1e-12
LEARNING_RATE = 0.15
DEFAULT_EPOCHS = 1000
BETA = 0.35
GAMMA = -0.1
ZETA = 1.1
INIT_KEEP_PROB = 0.999
LOG_WEIGHT_JITTER_SD = 0.05
LOG_ALPHA_JITTER_SD = 0.01

# Minimum weight floor for zero/negative initial weights
_WEIGHT_FLOOR = 0.01


def parse_args(argv=None):
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="National L0 calibration for Enhanced CPS"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to Extended CPS h5 file",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to policy_data.db",
    )
    parser.add_argument(
        "--geo-level",
        default="all",
        choices=["national", "state", "cd", "all"],
        help="Geographic level filter (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output enhanced_cps h5 file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lambda-l0",
        type=float,
        default=LAMBDA_L0,
        help=f"L0 penalty (default: {LAMBDA_L0})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training (default: cpu)",
    )
    return parser.parse_args(argv)


def initialize_weights(original_weights: np.ndarray) -> np.ndarray:
    """
    Initialize calibration weights from original household weights.

    Zero and negative weights are floored to a small positive value
    to avoid log-domain issues in the L0 optimizer.

    Args:
        original_weights: Array of household weights from the CPS.

    Returns:
        Array of positive initial weights.
    """
    weights = original_weights.copy().astype(np.float64)
    weights[weights <= 0] = _WEIGHT_FLOOR
    return weights


def build_calibration_inputs(
    dataset_class,
    time_period: int,
    db_path: str,
    sim=None,
    geo_level: str = "all",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Build calibration matrix and targets from the database.

    Reads targets from policy_data.db via NationalMatrixBuilder.

    Args:
        dataset_class: The input dataset class (e.g., ExtendedCPS_2024).
        time_period: Tax year for calibration.
        db_path: Path to policy_data.db.
        sim: Optional pre-built Microsimulation instance.
        geo_level: Geographic filter -- ``"national"``,
            ``"state"``, ``"cd"``, or ``"all"`` (default).

    Returns:
        Tuple of (matrix, targets, target_names) where:
        - matrix: shape (n_households, n_targets) float32
        - targets: shape (n_targets,) float64
        - target_names: list of str
    """
    from policyengine_us_data.calibration.national_matrix_builder import (
        NationalMatrixBuilder,
    )

    db_uri = f"sqlite:///{db_path}"
    builder = NationalMatrixBuilder(db_uri=db_uri, time_period=time_period)

    if sim is None:
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=dataset_class)
    sim.default_calculation_period = time_period

    matrix, targets, names = builder.build_matrix(
        sim=sim, dataset_class=dataset_class, geo_level=geo_level
    )
    return (
        matrix.astype(np.float32),
        targets.astype(np.float64),
        names,
    )


def compute_diagnostics(
    targets: np.ndarray,
    estimates: np.ndarray,
    names: list,
    threshold: float = 0.10,
    n_worst: int = 20,
) -> dict:
    """
    Compute calibration diagnostics.

    Args:
        targets: Target values.
        estimates: Predicted values from weighted matrix.
        names: Target names.
        threshold: Fraction for "within X%" check.
        n_worst: Number of worst targets to report.

    Returns:
        Dict with keys:
        - pct_within_10: % of targets within threshold of target
        - worst_targets: list of (name, rel_error) tuples
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_errors = np.where(
            np.abs(targets) > 1e-6,
            (estimates - targets) / targets,
            0.0,
        )

    abs_rel_errors = np.abs(rel_errors)
    within = np.mean(abs_rel_errors <= threshold) * 100.0

    # Sort by absolute relative error descending
    sorted_idx = np.argsort(-abs_rel_errors)
    worst = [(names[i], float(rel_errors[i])) for i in sorted_idx[:n_worst]]

    return {
        "pct_within_10": float(within),
        "worst_targets": worst,
    }


def fit_national_weights(
    matrix: np.ndarray,
    targets: np.ndarray,
    initial_weights: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    lambda_l0: float = LAMBDA_L0,
    lambda_l2: float = LAMBDA_L2,
    learning_rate: float = LEARNING_RATE,
    device: str = "cpu",
) -> np.ndarray:
    """Run L0-regularized calibration to find optimal household
    weights.

    Uses l0-python's ``SparseCalibrationWeights`` which expects::

        M @ w = y_hat

    where ``M`` has shape ``(n_targets, n_features)`` and ``w`` has
    shape ``(n_features,)``.  The input *matrix* is provided in the
    more natural ``(n_households, n_targets)`` layout and is
    transposed internally before being passed to the optimizer.

    Args:
        matrix: Calibration matrix, shape
            (n_households, n_targets).
        targets: Target values, shape (n_targets,).
        initial_weights: Starting household weights
            (n_households,).
        epochs: Number of training epochs.
        lambda_l0: L0 regularization strength.
        lambda_l2: L2 regularization strength.
        learning_rate: Adam learning rate.
        device: Torch device ("cpu" or "cuda").

    Returns:
        Calibrated weight array, shape (n_households,).
    """
    try:
        from l0.calibration import SparseCalibrationWeights
    except ImportError:
        raise ImportError(
            "l0-python is required for L0 calibration. "
            "Install with: pip install l0-python"
        )

    n_households, n_targets = matrix.shape
    logger.info(
        f"Starting L0 calibration: {n_households} households, "
        f"{n_targets} targets, {epochs} epochs"
    )
    logger.info(
        f"Hyperparams: lambda_l0={lambda_l0}, "
        f"lambda_l2={lambda_l2}, lr={learning_rate}"
    )

    # Transpose to (n_targets, n_households) for l0-python:
    #   M @ w = y_hat
    #   (n_targets, n_households) @ (n_households,) = (n_targets,)
    # l0-python expects a scipy sparse matrix, not dense numpy.
    M = scipy.sparse.csr_matrix(matrix.T)

    model = SparseCalibrationWeights(
        n_features=n_households,
        beta=BETA,
        gamma=GAMMA,
        zeta=ZETA,
        init_keep_prob=INIT_KEEP_PROB,
        init_weights=initial_weights,
        log_weight_jitter_sd=LOG_WEIGHT_JITTER_SD,
        log_alpha_jitter_sd=LOG_ALPHA_JITTER_SD,
        device=device,
    )

    model.fit(
        M=M,
        y=targets,
        target_groups=None,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        lr=learning_rate,
        epochs=epochs,
        loss_type="relative",
        verbose=True,
        verbose_freq=max(1, epochs // 10),
    )

    import torch

    with torch.no_grad():
        weights = model.get_weights(deterministic=True).cpu().numpy()

    logger.info(
        f"Calibration complete. "
        f"Non-zero weights: {(weights > 0).sum():,} "
        f"/ {len(weights):,}"
    )
    return weights


def save_weights_to_h5(h5_path: str, weights: np.ndarray, year: int = 2024):
    """
    Save calibrated weights into an existing h5 dataset file.

    Overwrites the household_weight/{year} dataset while preserving
    all other data in the file.

    Args:
        h5_path: Path to the h5 file.
        weights: Calibrated weight array.
        year: Time period key.
    """
    key = f"household_weight/{year}"
    with h5py.File(h5_path, "a") as f:
        if key in f:
            del f[key]
        f.create_dataset(key, data=weights)
    logger.info(f"Saved weights to {h5_path} [{key}]")


def run_validation(weights, matrix, targets, names):
    """Print quick validation of key aggregates."""
    estimates = weights @ matrix

    diag = compute_diagnostics(targets, estimates, names)
    logger.info(f"Targets within 10%%: {diag['pct_within_10']:.1f}%%")
    logger.info("Worst targets:")
    for name, rel_err in diag["worst_targets"][:10]:
        logger.info(f"  {name:60s} {rel_err:+.2%}")

    # Highlight key programs if present
    for keyword in ["population", "income_tax", "snap"]:
        matches = [
            (n, e, t)
            for n, e, t in zip(names, estimates, targets)
            if keyword in n.lower()
        ]
        if matches:
            n, e, t = matches[0]
            logger.info(f"  {n}: est={e:,.0f}, target={t:,.0f}")


def main(argv=None):
    """Entry point for CLI usage."""
    args = parse_args(argv)

    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps.extended_cps import (
        ExtendedCPS_2024,
    )
    from policyengine_us_data.storage import STORAGE_FOLDER

    dataset_path = args.dataset or str(STORAGE_FOLDER / "extended_cps_2024.h5")
    output_path = args.output or str(STORAGE_FOLDER / "enhanced_cps_2024.h5")

    logger.info(f"Loading dataset from {dataset_path}")
    sim = Microsimulation(dataset=dataset_path)
    original_weights = sim.calculate("household_weight").values

    logger.info(
        f"Loaded {len(original_weights):,} households, "
        f"total weight: {original_weights.sum():,.0f}"
    )

    # Build calibration inputs
    matrix, targets, names = build_calibration_inputs(
        dataset_class=ExtendedCPS_2024,
        time_period=2024,
        db_path=args.db_path,
        geo_level=args.geo_level,
    )

    logger.info(
        f"Calibration matrix: {matrix.shape[0]} households x "
        f"{matrix.shape[1]} targets"
    )

    # Initialize and run
    init_weights = initialize_weights(original_weights)
    calibrated_weights = fit_national_weights(
        matrix=matrix,
        targets=targets,
        initial_weights=init_weights,
        epochs=args.epochs,
        lambda_l0=args.lambda_l0,
        device=args.device,
    )

    # Diagnostics
    run_validation(calibrated_weights, matrix, targets, names)

    # Save
    # Copy source to output if different
    import shutil

    if dataset_path != output_path:
        shutil.copy2(dataset_path, output_path)
    save_weights_to_h5(output_path, calibrated_weights, year=2024)
    logger.info(f"Enhanced CPS saved to {output_path}")


if __name__ == "__main__":
    main()
