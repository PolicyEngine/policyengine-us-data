"""
L0 sweep: build matrix once, fit at many L0 values, plot results.

Designed to run overnight. Saves intermediate results so it can
resume if interrupted.

Usage:
    python -m policyengine_us_data.calibration.l0_sweep

Output:
    - storage/calibration/l0_sweep_matrix.npz  (sparse matrix)
    - storage/calibration/l0_sweep_targets.npy
    - storage/calibration/l0_sweep_results.csv
    - storage/calibration/l0_sweep_plot.png
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# L0 values to sweep (log-spaced from 1e-10 to 1e-2)
DEFAULT_L0_VALUES = [
    1e-10,
    3e-10,
    1e-9,
    3e-9,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
]

# Hyperparameters (matching unified_calibration.py)
BETA = 0.35
GAMMA = -0.1
ZETA = 1.1
INIT_KEEP_PROB = 0.999
LOG_WEIGHT_JITTER_SD = 0.05
LOG_ALPHA_JITTER_SD = 0.01
LAMBDA_L2 = 1e-12
LEARNING_RATE = 0.15
DEFAULT_EPOCHS = 200
DEFAULT_N_CLONES = 130


def build_and_save_matrix(
    dataset_path: str,
    db_path: str,
    output_dir: Path,
    n_clones: int,
    seed: int,
):
    """Build sparse matrix and save to disk.

    Returns:
        Tuple of (X_sparse, targets, target_names, n_total).
    """
    matrix_path = output_dir / "l0_sweep_matrix.npz"
    targets_path = output_dir / "l0_sweep_targets.npy"
    names_path = output_dir / "l0_sweep_names.txt"

    if matrix_path.exists() and targets_path.exists():
        logger.info("Loading cached matrix from %s", matrix_path)
        X = scipy.sparse.load_npz(str(matrix_path))
        targets = np.load(str(targets_path))
        names = names_path.read_text().strip().split("\n")
        logger.info(
            "Loaded matrix: %d targets x %d columns",
            X.shape[0],
            X.shape[1],
        )
        return X, targets, names, X.shape[1]

    from policyengine_us import Microsimulation

    from policyengine_us_data.calibration.clone_and_assign import (
        assign_random_geography,
    )
    from policyengine_us_data.calibration.unified_matrix_builder import (
        UnifiedMatrixBuilder,
    )

    # Load dataset
    logger.info("Loading dataset from %s", dataset_path)
    sim = Microsimulation(dataset=dataset_path)
    n_records = len(sim.calculate("household_id", map_to="household").values)
    logger.info("Loaded %d households", n_records)

    # Assign geography
    n_total = n_records * n_clones
    logger.info(
        "Assigning geography: %d x %d = %d",
        n_records,
        n_clones,
        n_total,
    )
    geography = assign_random_geography(
        n_records=n_records,
        n_clones=n_clones,
        seed=seed,
    )

    # Build matrix
    db_uri = f"sqlite:///{db_path}"
    builder = UnifiedMatrixBuilder(db_uri=db_uri, time_period=2024)
    targets_df, X_sparse, target_names = builder.build_matrix(
        dataset_path=dataset_path,
        geography=geography,
    )

    # Filter achievable
    row_sums = np.array(X_sparse.sum(axis=1)).flatten()
    achievable = row_sums > 0
    logger.info(
        "Achievable: %d / %d targets",
        achievable.sum(),
        len(achievable),
    )
    X_sparse = X_sparse[achievable, :]
    targets = targets_df[achievable]["value"].values
    target_names = [n for n, a in zip(target_names, achievable) if a]

    # Save
    scipy.sparse.save_npz(str(matrix_path), X_sparse)
    np.save(str(targets_path), targets)
    names_path.write_text("\n".join(target_names))
    logger.info("Saved matrix to %s", output_dir)

    return X_sparse, targets, target_names, n_total


def fit_one_l0(
    X_sparse,
    targets,
    target_names,
    lambda_l0: float,
    epochs: int,
    device: str,
    output_dir: Path,
) -> dict:
    """Fit at one L0 value. Saves full weights array and
    per-target diagnostics.

    Returns:
        Summary dict for the results CSV.
    """
    from l0.calibration import SparseCalibrationWeights

    import torch

    n_total = X_sparse.shape[1]
    initial_weights = np.ones(n_total) * 100

    t0 = time.time()
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
        verbose_freq=max(1, epochs // 5),
    )

    with torch.no_grad():
        weights = (
            model.get_weights(deterministic=True)
            .cpu()
            .numpy()
        )

    n_nonzero = int((weights > 0).sum())
    total_weight = float(weights.sum())
    elapsed = time.time() - t0

    # Save the full weight array.
    l0_tag = f"{lambda_l0:.0e}".replace("+", "")
    weights_path = output_dir / f"weights_l0_{l0_tag}.npy"
    np.save(str(weights_path), weights)
    logger.info("Saved weights to %s", weights_path)

    # Compute per-target diagnostics.
    estimates = weights @ X_sparse.T.toarray()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_errors = np.where(
            np.abs(targets) > 1e-6,
            (estimates - targets) / targets,
            0.0,
        )
    abs_rel_errors = np.abs(rel_errors)

    # Save per-target diagnostics.
    diag_df = pd.DataFrame(
        {
            "target_name": target_names,
            "target_value": targets,
            "estimate": estimates,
            "rel_error": rel_errors,
            "abs_rel_error": abs_rel_errors,
        }
    )
    diag_path = (
        output_dir / f"diagnostics_l0_{l0_tag}.csv"
    )
    diag_df.to_csv(diag_path, index=False)

    # Summary metrics.
    pct_within_5 = float(
        np.mean(abs_rel_errors <= 0.05) * 100
    )
    pct_within_10 = float(
        np.mean(abs_rel_errors <= 0.10) * 100
    )
    pct_within_25 = float(
        np.mean(abs_rel_errors <= 0.25) * 100
    )
    median_rel_error = float(np.median(abs_rel_errors))
    mean_rel_error = float(np.mean(abs_rel_errors))
    max_rel_error = float(np.max(abs_rel_errors))
    # Relative MSE (the actual loss the optimizer minimizes)
    relative_mse = float(
        np.mean(
            np.where(
                np.abs(targets) > 1e-6,
                ((estimates - targets) / targets) ** 2,
                0.0,
            )
        )
    )

    return {
        "lambda_l0": lambda_l0,
        "n_nonzero": n_nonzero,
        "n_total": n_total,
        "sparsity_pct": (1 - n_nonzero / n_total) * 100,
        "total_weight": total_weight,
        "relative_mse": relative_mse,
        "mean_rel_error": mean_rel_error,
        "median_rel_error": median_rel_error,
        "max_rel_error": max_rel_error,
        "pct_within_5": pct_within_5,
        "pct_within_10": pct_within_10,
        "pct_within_25": pct_within_25,
        "elapsed_s": elapsed,
        "weights_path": str(weights_path),
        "diagnostics_path": str(diag_path),
    }


def make_plot(results_df: pd.DataFrame, output_path: Path):
    """Create L0 vs record count plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Left axis: record count
    ax1.semilogx(
        results_df["lambda_l0"],
        results_df["n_nonzero"],
        "b-o",
        linewidth=2,
        markersize=8,
        label="Non-zero records",
    )
    ax1.set_xlabel("L0 regularization (lambda)", fontsize=13)
    ax1.set_ylabel("Non-zero records", color="b", fontsize=13)
    ax1.tick_params(axis="y", labelcolor="b")

    # Reference lines
    ax1.axhline(
        y=4_000_000,
        color="b",
        linestyle="--",
        alpha=0.5,
        label="Target: ~4M (local)",
    )
    ax1.axhline(
        y=50_000,
        color="b",
        linestyle=":",
        alpha=0.5,
        label="Target: ~50K (national)",
    )

    # Right axis: accuracy
    ax2 = ax1.twinx()
    ax2.semilogx(
        results_df["lambda_l0"],
        results_df["pct_within_10"],
        "r-s",
        linewidth=2,
        markersize=8,
        label="% targets within 10%",
    )
    ax2.set_ylabel("% targets within 10%", color="r", fontsize=13)
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(0, 100)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center left",
        fontsize=11,
    )

    ax1.set_title(
        "Unified calibration: L0 vs sparsity and accuracy",
        fontsize=15,
    )
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    logger.info("Plot saved to %s", output_path)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="L0 sweep for unified calibration"
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--n-clones",
        type=int,
        default=DEFAULT_N_CLONES,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from policyengine_us_data.storage import STORAGE_FOLDER

    dataset_path = args.dataset or str(STORAGE_FOLDER / "extended_cps_2024.h5")
    db_path = args.db_path or str(
        STORAGE_FOLDER / "calibration" / "policy_data.db"
    )
    output_dir = Path(args.output_dir or str(STORAGE_FOLDER / "calibration"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build matrix (cached)
    X_sparse, targets, target_names, n_total = build_and_save_matrix(
        dataset_path=dataset_path,
        db_path=db_path,
        output_dir=output_dir,
        n_clones=args.n_clones,
        seed=args.seed,
    )

    # Step 2: Sweep L0 values
    results_path = output_dir / "l0_sweep_results.csv"

    # Resume from existing results if available
    completed = set()
    if results_path.exists():
        existing = pd.read_csv(results_path)
        completed = set(existing["lambda_l0"].values)
        results = existing.to_dict("records")
        logger.info(
            "Resuming: %d L0 values already completed",
            len(completed),
        )
    else:
        results = []

    for l0_val in DEFAULT_L0_VALUES:
        if l0_val in completed:
            logger.info("Skipping L0=%.1e (already done)", l0_val)
            continue

        logger.info(
            "=" * 60 + f"\nFitting L0={l0_val:.1e} "
            f"({len(results)+1}/{len(DEFAULT_L0_VALUES)})"
        )
        result = fit_one_l0(
            X_sparse=X_sparse,
            targets=targets,
            lambda_l0=l0_val,
            epochs=args.epochs,
            device=args.device,
        )
        results.append(result)
        logger.info(
            "L0=%.1e: %d non-zero records (%.1f%% "
            "sparsity), %.1f%% within 10%%",
            l0_val,
            result["n_nonzero"],
            result["sparsity_pct"],
            result["pct_within_10"],
        )

        # Save incrementally
        df = pd.DataFrame(results)
        df.to_csv(results_path, index=False)

    # Step 3: Plot
    df = pd.DataFrame(results).sort_values("lambda_l0")
    plot_path = output_dir / "l0_sweep_plot.png"
    make_plot(df, plot_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 60)
    for _, row in df.iterrows():
        logger.info(
            "L0=%.1e: %8d records, %5.1f%% within 10%%",
            row["lambda_l0"],
            row["n_nonzero"],
            row["pct_within_10"],
        )


if __name__ == "__main__":
    main()
