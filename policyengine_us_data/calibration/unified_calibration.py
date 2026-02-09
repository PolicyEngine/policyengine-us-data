"""
Unified L0 calibration for Enhanced CPS.

Clones the extended CPS, assigns random geography, builds a sparse
calibration matrix against all DB targets, and runs L0-regularized
optimization to produce calibrated weights.

Two presets control output size via L0 regularization:
- local: L0=1e-8, ~3-4M records (for local area dataset)
- national: L0=1e-4, ~50K records (for web app)

Usage:
    python -m policyengine_us_data.calibration.unified_calibration \\
        --dataset path/to/extended_cps_2024.h5 \\
        --db-path path/to/policy_data.db \\
        --output path/to/weights.npy \\
        --preset local \\
        --epochs 100
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# L0 presets
PRESETS = {
    "local": 1e-8,  # ~3-4M records
    "national": 1e-4,  # ~50K records
}

# Shared hyperparameters (matching local area calibration)
BETA = 0.35
GAMMA = -0.1
ZETA = 1.1
INIT_KEEP_PROB = 0.999
LOG_WEIGHT_JITTER_SD = 0.05
LOG_ALPHA_JITTER_SD = 0.01
LAMBDA_L2 = 1e-12
LEARNING_RATE = 0.15
DEFAULT_EPOCHS = 100
DEFAULT_N_CLONES = 130


def parse_args(argv=None):
    """Parse CLI arguments.

    Args:
        argv: Optional list of argument strings. Defaults to
            sys.argv if None.

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Unified L0 calibration for Enhanced CPS"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to extended CPS h5 file",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to policy_data.db",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save calibrated weights (.npy)",
    )
    parser.add_argument(
        "--n-clones",
        type=int,
        default=DEFAULT_N_CLONES,
        help=f"Number of clones (default: {DEFAULT_N_CLONES})",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="L0 preset: local (~3-4M) or national (~50K)",
    )
    parser.add_argument(
        "--lambda-l0",
        type=float,
        default=None,
        help="Custom L0 penalty (overrides preset)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for geography assignment",
    )
    return parser.parse_args(argv)


def run_calibration(
    dataset_path: str,
    db_path: str,
    n_clones: int = DEFAULT_N_CLONES,
    lambda_l0: float = 1e-8,
    epochs: int = DEFAULT_EPOCHS,
    device: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """Run unified calibration pipeline.

    Args:
        dataset_path: Path to extended CPS h5 file.
        db_path: Path to policy_data.db.
        n_clones: Number of dataset clones.
        lambda_l0: L0 regularization strength.
        epochs: Training epochs.
        device: Torch device.
        seed: Random seed.

    Returns:
        Calibrated weight array of shape
        (n_records * n_clones,).
    """
    from policyengine_us import Microsimulation

    from policyengine_us_data.calibration.clone_and_assign import (
        assign_random_geography,
    )
    from policyengine_us_data.calibration.unified_matrix_builder import (
        UnifiedMatrixBuilder,
    )

    # Step 1: Load dataset and get record count
    logger.info("Loading dataset from %s", dataset_path)
    sim = Microsimulation(dataset=dataset_path)
    n_records = len(sim.calculate("household_id", map_to="household").values)
    logger.info("Loaded %d households", n_records)

    # Step 2: Assign random geography
    logger.info(
        "Assigning geography: %d records x %d clones = %d total",
        n_records,
        n_clones,
        n_records * n_clones,
    )
    geography = assign_random_geography(
        n_records=n_records,
        n_clones=n_clones,
        seed=seed,
    )

    # Step 3: Build sparse calibration matrix
    db_uri = f"sqlite:///{db_path}"
    builder = UnifiedMatrixBuilder(db_uri=db_uri, time_period=2024)
    targets_df, X_sparse, target_names = builder.build_matrix(
        dataset_path=dataset_path,
        geography=geography,
    )

    # Step 4: Filter to achievable targets
    row_sums = np.array(X_sparse.sum(axis=1)).flatten()
    achievable = row_sums > 0
    n_achievable = achievable.sum()
    n_impossible = (~achievable).sum()
    logger.info(
        "Achievable: %d, Impossible (removed): %d",
        n_achievable,
        n_impossible,
    )
    X_sparse = X_sparse[achievable, :]
    targets = targets_df[achievable]["value"].values
    target_names = [n for n, a in zip(target_names, achievable) if a]

    # Step 5: Run L0 calibration
    try:
        from l0.calibration import SparseCalibrationWeights
    except ImportError:
        raise ImportError("l0-python required. Install: pip install l0-python")

    import torch

    n_total = X_sparse.shape[1]
    initial_weights = np.ones(n_total) * 100
    logger.info(
        "Starting L0 calibration: %d targets, %d features, "
        "lambda_l0=%.1e, epochs=%d",
        X_sparse.shape[0],
        n_total,
        lambda_l0,
        epochs,
    )

    model = SparseCalibrationWeights(
        n_features=n_total,
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
        M=X_sparse,
        y=targets,
        target_groups=None,
        lambda_l0=lambda_l0,
        lambda_l2=LAMBDA_L2,
        lr=LEARNING_RATE,
        epochs=epochs,
        loss_type="relative",
        verbose=True,
        verbose_freq=max(1, epochs // 10),
    )

    with torch.no_grad():
        weights = model.get_weights(deterministic=True).cpu().numpy()

    n_nonzero = (weights > 0).sum()
    logger.info(
        "Calibration complete. Non-zero: %d / %d " "(%.1f%% sparsity)",
        n_nonzero,
        n_total,
        (1 - n_nonzero / n_total) * 100,
    )
    return weights


def main(argv=None):
    """Entry point for CLI usage.

    Args:
        argv: Optional list of argument strings.
    """
    args = parse_args(argv)

    from policyengine_us_data.storage import STORAGE_FOLDER

    dataset_path = args.dataset or str(STORAGE_FOLDER / "extended_cps_2024.h5")
    db_path = args.db_path or str(
        STORAGE_FOLDER / "calibration" / "policy_data.db"
    )
    output_path = args.output or str(
        STORAGE_FOLDER / "calibration" / "unified_weights.npy"
    )

    # Resolve L0
    if args.lambda_l0 is not None:
        lambda_l0 = args.lambda_l0
    elif args.preset is not None:
        lambda_l0 = PRESETS[args.preset]
    else:
        lambda_l0 = PRESETS["local"]
        logger.info("No preset/lambda specified, using 'local'")

    weights = run_calibration(
        dataset_path=dataset_path,
        db_path=db_path,
        n_clones=args.n_clones,
        lambda_l0=lambda_l0,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )

    np.save(output_path, weights)
    logger.info("Weights saved to %s", output_path)


if __name__ == "__main__":
    main()
