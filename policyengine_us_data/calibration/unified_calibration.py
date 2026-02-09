"""
Unified L0 calibration pipeline.

New pipeline flow:
    1. Load raw CPS (~55K households)
    2. Clone 10x (v1) / 100x (v2)
    3. Assign random geography (census block -> state, county, CD)
    4. QRF imputation: all vars with state as predictor
       a. ACS -> rent, real_estate_taxes
       b. SIPP -> tip_income, bank/stock/bond_assets
       c. SCF -> net_worth, auto_loan_balance/interest
       d. PUF clone (2x) -> 67 tax variables
    5. PE simulation (via matrix builder)
    6. Build unified sparse calibration matrix
    7. L0-regularized optimization -> calibrated weights

Two presets control output size via L0 regularization:
- local: L0=1e-8, ~3-4M records (for local area dataset)
- national: L0=1e-4, ~50K records (for web app)

Usage:
    python -m policyengine_us_data.calibration.unified_calibration \\
        --dataset path/to/cps_2024.h5 \\
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
DEFAULT_N_CLONES = 10


def parse_args(argv=None):
    """Parse CLI arguments.

    Args:
        argv: Optional list of argument strings. Defaults to
            sys.argv if None.

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Unified L0 calibration pipeline"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to raw CPS h5 file (or extended CPS)",
    )
    parser.add_argument(
        "--puf-dataset",
        default=None,
        help="Path to PUF h5 file for QRF training",
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
    parser.add_argument(
        "--skip-puf",
        action="store_true",
        help="Skip PUF clone + QRF (use raw CPS as-is)",
    )
    parser.add_argument(
        "--skip-source-impute",
        action="store_true",
        help="Skip ACS/SIPP/SCF re-imputation with state",
    )
    return parser.parse_args(argv)


def _build_puf_cloned_dataset(
    dataset_path: str,
    puf_dataset_path: str,
    state_fips: np.ndarray,
    time_period: int = 2024,
    skip_qrf: bool = False,
    skip_source_impute: bool = False,
) -> str:
    """Build a PUF-cloned dataset from raw CPS.

    Loads the CPS dataset, runs source imputations (ACS/SIPP/SCF)
    with state as predictor, then PUF clone + QRF imputation.

    Args:
        dataset_path: Path to raw CPS h5 file.
        puf_dataset_path: Path to PUF h5 file.
        state_fips: State FIPS per household (from geography
            assignment, for the base n_records only).
        time_period: Tax year.
        skip_qrf: Skip QRF imputation (for testing).
        skip_source_impute: Skip ACS/SIPP/SCF imputations.

    Returns:
        Path to the PUF-cloned h5 file.
    """
    from policyengine_us import Microsimulation

    from policyengine_us_data.calibration.puf_impute import (
        puf_clone_dataset,
    )

    logger.info("Building PUF-cloned dataset from %s", dataset_path)

    # Load CPS data
    sim = Microsimulation(dataset=dataset_path)
    data = sim.dataset.load_dataset()

    # Convert to the time_period_arrays format expected by puf_clone
    data_dict = {}
    for var in data:
        values = data[var][...]
        data_dict[var] = {time_period: values}

    # Source imputations (ACS/SIPP/SCF) with state as predictor
    if not skip_source_impute:
        from policyengine_us_data.calibration.source_impute import (
            impute_source_variables,
        )

        data_dict = impute_source_variables(
            data=data_dict,
            state_fips=state_fips,
            time_period=time_period,
            dataset_path=dataset_path,
        )

    # Determine PUF dataset
    puf_dataset = puf_dataset_path if not skip_qrf else None

    # PUF clone + QRF impute
    new_data = puf_clone_dataset(
        data=data_dict,
        state_fips=state_fips,
        time_period=time_period,
        puf_dataset=puf_dataset,
        skip_qrf=skip_qrf,
        dataset_path=dataset_path,
    )

    # Save expanded dataset
    output_path = str(
        Path(dataset_path).parent / f"puf_cloned_{Path(dataset_path).stem}.h5"
    )

    import h5py

    with h5py.File(output_path, "w") as f:
        for var, time_dict in new_data.items():
            for tp, values in time_dict.items():
                key = f"{var}/{tp}"
                f.create_dataset(key, data=values)

    logger.info("PUF-cloned dataset saved to %s", output_path)
    return output_path


def run_calibration(
    dataset_path: str,
    db_path: str,
    n_clones: int = DEFAULT_N_CLONES,
    lambda_l0: float = 1e-8,
    epochs: int = DEFAULT_EPOCHS,
    device: str = "cpu",
    seed: int = 42,
    puf_dataset_path: str = None,
    skip_puf: bool = False,
    skip_source_impute: bool = False,
) -> np.ndarray:
    """Run unified calibration pipeline.

    New pipeline:
        1. Load raw CPS -> get n_records
        2. Clone n_clones x, assign geography
        3. Source imputations (ACS/SIPP/SCF) with state
        4. PUF clone (2x) + QRF impute with state
        5. Build sparse calibration matrix
        6. L0 calibration

    Args:
        dataset_path: Path to raw CPS h5 file.
        db_path: Path to policy_data.db.
        n_clones: Number of dataset clones.
        lambda_l0: L0 regularization strength.
        epochs: Training epochs.
        device: Torch device.
        seed: Random seed.
        puf_dataset_path: Path to PUF h5 for QRF training.
        skip_puf: Skip PUF clone step.
        skip_source_impute: Skip ACS/SIPP/SCF imputations.

    Returns:
        Calibrated weight array of shape
        (n_records * n_clones,).
    """
    from policyengine_us import Microsimulation

    from policyengine_us_data.calibration.clone_and_assign import (
        assign_random_geography,
        double_geography_for_puf,
    )
    from policyengine_us_data.calibration.unified_matrix_builder import (
        UnifiedMatrixBuilder,
    )

    # Step 1: Load raw CPS and get record count
    logger.info("Loading dataset from %s", dataset_path)
    sim = Microsimulation(dataset=dataset_path)
    n_records = len(sim.calculate("household_id", map_to="household").values)
    logger.info("Loaded %d households", n_records)
    del sim

    # Step 2: Clone and assign geography
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

    # Step 3: PUF clone (2x) + QRF imputation
    if not skip_puf:
        # Get state_fips for the base records (first clone)
        base_states = geography.state_fips[:n_records]

        puf_cloned_path = _build_puf_cloned_dataset(
            dataset_path=dataset_path,
            puf_dataset_path=puf_dataset_path or "",
            state_fips=base_states,
            time_period=2024,
            skip_qrf=puf_dataset_path is None,
            skip_source_impute=skip_source_impute,
        )

        # Double geography to match PUF-cloned records
        geography = double_geography_for_puf(geography)
        dataset_for_matrix = puf_cloned_path
        n_records_for_matrix = n_records * 2

        logger.info(
            "After PUF clone: %d records x %d clones = %d total",
            n_records_for_matrix,
            n_clones,
            n_records_for_matrix * n_clones,
        )
    else:
        # Even without PUF, run source imputations if requested
        if not skip_source_impute:
            from policyengine_us import Microsimulation

            from policyengine_us_data.calibration.source_impute import (
                impute_source_variables,
            )

            sim = Microsimulation(dataset=dataset_path)
            raw_data = sim.dataset.load_dataset()
            data_dict = {}
            for var in raw_data:
                data_dict[var] = {2024: raw_data[var][...]}
            del sim

            base_states = geography.state_fips[:n_records]
            data_dict = impute_source_variables(
                data=data_dict,
                state_fips=base_states,
                time_period=2024,
                dataset_path=dataset_path,
            )

            # Save updated dataset
            import h5py

            source_path = str(
                Path(dataset_path).parent
                / f"source_imputed_{Path(dataset_path).stem}.h5"
            )
            with h5py.File(source_path, "w") as f:
                for var, time_dict in data_dict.items():
                    for tp, values in time_dict.items():
                        f.create_dataset(f"{var}/{tp}", data=values)
            dataset_for_matrix = source_path
            logger.info(
                "Source-imputed dataset saved to %s",
                source_path,
            )
        else:
            dataset_for_matrix = dataset_path
        n_records_for_matrix = n_records

    # Step 5: Build sparse calibration matrix
    db_uri = f"sqlite:///{db_path}"
    builder = UnifiedMatrixBuilder(db_uri=db_uri, time_period=2024)
    targets_df, X_sparse, target_names = builder.build_matrix(
        dataset_path=dataset_for_matrix,
        geography=geography,
    )

    # Step 5: Filter to achievable targets
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

    # Step 6: Run L0 calibration
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

    dataset_path = args.dataset or str(STORAGE_FOLDER / "cps_2024_full.h5")
    puf_dataset_path = args.puf_dataset or str(STORAGE_FOLDER / "puf_2024.h5")
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
        puf_dataset_path=puf_dataset_path,
        skip_puf=args.skip_puf,
        skip_source_impute=args.skip_source_impute,
    )

    np.save(output_path, weights)
    logger.info("Weights saved to %s", output_path)


if __name__ == "__main__":
    main()
