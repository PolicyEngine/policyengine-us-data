"""
Unified L0 calibration pipeline.

Pipeline flow:
    1. Load CPS dataset -> get n_records
    2. Clone Nx, assign random geography (census block)
    3. Re-randomize simple takeup variables per block
    4. Build sparse calibration matrix (clone-by-clone)
    5. L0-regularized optimization -> calibrated weights
    6. Save weights, diagnostics, run config

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
import builtins
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

PRESETS = {
    "local": 1e-8,
    "national": 1e-4,
}

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

SIMPLE_TAKEUP_VARS = [
    {
        "variable": "takes_up_snap_if_eligible",
        "entity": "spm_unit",
        "rate_key": "snap",
    },
    {
        "variable": "takes_up_aca_if_eligible",
        "entity": "tax_unit",
        "rate_key": "aca",
    },
    {
        "variable": "takes_up_dc_ptc",
        "entity": "tax_unit",
        "rate_key": "dc_ptc",
    },
    {
        "variable": "takes_up_head_start_if_eligible",
        "entity": "person",
        "rate_key": "head_start",
    },
    {
        "variable": "takes_up_early_head_start_if_eligible",
        "entity": "person",
        "rate_key": "early_head_start",
    },
    {
        "variable": "takes_up_ssi_if_eligible",
        "entity": "person",
        "rate_key": "ssi",
    },
    {
        "variable": "would_file_taxes_voluntarily",
        "entity": "tax_unit",
        "rate_key": "voluntary_filing",
    },
    {
        "variable": "takes_up_medicaid_if_eligible",
        "entity": "person",
        "rate_key": "medicaid",
    },
]


def rerandomize_takeup(
    sim,
    clone_block_geoids: np.ndarray,
    clone_state_fips: np.ndarray,
    time_period: int,
) -> None:
    """Re-randomize simple takeup variables per census block.

    Groups entities by their household's block GEOID and draws
    new takeup booleans using seeded_rng(var_name, salt=block).
    Overrides the simulation's stored inputs.

    Args:
        sim: Microsimulation instance (already has state_fips).
        clone_block_geoids: Block GEOIDs per household.
        clone_state_fips: State FIPS per household.
        time_period: Tax year.
    """
    from policyengine_us_data.parameters import (
        load_take_up_rate,
    )
    from policyengine_us_data.utils.randomness import (
        seeded_rng,
    )

    n_households = len(clone_block_geoids)
    hh_ids = sim.calculate("household_id", map_to="household").values
    hh_to_block = dict(zip(hh_ids, clone_block_geoids))
    hh_to_state = dict(zip(hh_ids, clone_state_fips))

    for spec in SIMPLE_TAKEUP_VARS:
        var_name = spec["variable"]
        entity_level = spec["entity"]
        rate_key = spec["rate_key"]

        rate_or_dict = load_take_up_rate(rate_key, time_period)

        is_state_specific = isinstance(rate_or_dict, dict)

        entity_ids = sim.calculate(
            f"{entity_level}_id", map_to=entity_level
        ).values
        entity_hh_ids = sim.calculate(
            "household_id", map_to=entity_level
        ).values
        n_entities = len(entity_ids)

        draws = np.zeros(n_entities, dtype=np.float64)
        rates = np.zeros(n_entities, dtype=np.float64)

        entity_blocks = np.array(
            [hh_to_block.get(hid, "0") for hid in entity_hh_ids]
        )

        unique_blocks = np.unique(entity_blocks)
        for block in unique_blocks:
            mask = entity_blocks == block
            n_in_block = mask.sum()
            rng = seeded_rng(var_name, salt=str(block))
            draws[mask] = rng.random(n_in_block)

            if is_state_specific:
                block_hh_ids = entity_hh_ids[mask]
                for i, hid in enumerate(block_hh_ids):
                    state = int(hh_to_state.get(hid, 0))
                    state_str = str(state)
                    r = rate_or_dict.get(
                        state_str,
                        rate_or_dict.get(state, 0.8),
                    )
                    idx = np.where(mask)[0][i]
                    rates[idx] = r
            else:
                rates[mask] = rate_or_dict

        new_values = draws < rates
        sim.set_input(var_name, time_period, new_values)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified L0 calibration pipeline"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to CPS h5 file",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to policy_data.db",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save weights (.npy)",
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
        help="L0 preset: local or national",
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
        "--domain-variables",
        type=str,
        default=None,
        help=(
            "Comma-separated domain variables for " "target_overview filtering"
        ),
    )
    parser.add_argument(
        "--hierarchical-domains",
        type=str,
        default=None,
        help=(
            "Comma-separated domains for hierarchical "
            "uprating + CD reconciliation"
        ),
    )
    parser.add_argument(
        "--skip-takeup-rerandomize",
        action="store_true",
        help="Skip takeup re-randomization",
    )
    return parser.parse_args(argv)


def fit_l0_weights(
    X_sparse,
    targets: np.ndarray,
    lambda_l0: float,
    epochs: int = DEFAULT_EPOCHS,
    device: str = "cpu",
    verbose_freq: int = None,
) -> np.ndarray:
    """Fit L0-regularized calibration weights.

    Args:
        X_sparse: Sparse matrix (targets x records).
        targets: Target values array.
        lambda_l0: L0 regularization strength.
        epochs: Training epochs.
        device: Torch device.
        verbose_freq: Print frequency. Defaults to 10%.

    Returns:
        Weight array of shape (n_records,).
    """
    import time

    try:
        from l0.calibration import SparseCalibrationWeights
    except ImportError:
        raise ImportError(
            "l0-python required. " "Install: pip install l0-python"
        )

    import torch

    n_total = X_sparse.shape[1]
    initial_weights = np.ones(n_total) * 100

    logger.info(
        "L0 calibration: %d targets, %d features, "
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

    if verbose_freq is None:
        verbose_freq = max(1, epochs // 10)

    _builtin_print = builtins.print

    def _flushed_print(*args, **kwargs):
        _builtin_print(*args, **kwargs)
        sys.stdout.flush()

    builtins.print = _flushed_print

    t0 = time.time()
    try:
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
            verbose_freq=verbose_freq,
        )
    finally:
        builtins.print = _builtin_print

    elapsed = time.time() - t0
    logger.info(
        "L0 done in %.1f min (%.1f sec/epoch)",
        elapsed / 60,
        elapsed / epochs,
    )

    with torch.no_grad():
        weights = model.get_weights(deterministic=True).cpu().numpy()

    n_nz = (weights > 0).sum()
    logger.info(
        "Non-zero: %d / %d (%.1f%% sparsity)",
        n_nz,
        n_total,
        (1 - n_nz / n_total) * 100,
    )
    return weights


def compute_diagnostics(
    weights: np.ndarray,
    X_sparse,
    targets_df,
    target_names: list,
) -> "pd.DataFrame":
    import pandas as pd

    estimates = X_sparse.dot(weights)
    true_values = targets_df["value"].values
    row_sums = np.array(X_sparse.sum(axis=1)).flatten()

    rel_errors = np.where(
        np.abs(true_values) > 0,
        (estimates - true_values) / np.abs(true_values),
        0.0,
    )
    return pd.DataFrame(
        {
            "target": target_names,
            "true_value": true_values,
            "estimate": estimates,
            "rel_error": rel_errors,
            "abs_rel_error": np.abs(rel_errors),
            "achievable": row_sums > 0,
        }
    )


def run_calibration(
    dataset_path: str,
    db_path: str,
    n_clones: int = DEFAULT_N_CLONES,
    lambda_l0: float = 1e-8,
    epochs: int = DEFAULT_EPOCHS,
    device: str = "cpu",
    seed: int = 42,
    domain_variables: list = None,
    hierarchical_domains: list = None,
    skip_takeup_rerandomize: bool = False,
):
    """Run unified calibration pipeline.

    Args:
        dataset_path: Path to CPS h5 file.
        db_path: Path to policy_data.db.
        n_clones: Number of dataset clones.
        lambda_l0: L0 regularization strength.
        epochs: Training epochs.
        device: Torch device.
        seed: Random seed.
        domain_variables: Filter targets by domain variable.
        hierarchical_domains: Domains for hierarchical
            uprating + CD reconciliation.
        skip_takeup_rerandomize: Skip takeup step.

    Returns:
        (weights, targets_df, X_sparse, target_names)
    """
    import time

    from policyengine_us import Microsimulation

    from policyengine_us_data.calibration.clone_and_assign import (
        assign_random_geography,
    )
    from policyengine_us_data.calibration.unified_matrix_builder import (
        UnifiedMatrixBuilder,
    )

    t0 = time.time()

    # Step 1: Load dataset
    logger.info("Loading dataset from %s", dataset_path)
    sim = Microsimulation(dataset=dataset_path)
    n_records = len(sim.calculate("household_id", map_to="household").values)
    logger.info("Loaded %d households", n_records)

    # Step 2: Clone and assign geography
    logger.info(
        "Assigning geography: %d x %d = %d total",
        n_records,
        n_clones,
        n_records * n_clones,
    )
    geography = assign_random_geography(
        n_records=n_records,
        n_clones=n_clones,
        seed=seed,
    )

    # Step 3: Build sim_modifier for takeup rerandomization
    sim_modifier = None
    if not skip_takeup_rerandomize:
        time_period = 2024

        def sim_modifier(s, clone_idx):
            col_start = clone_idx * n_records
            col_end = col_start + n_records
            blocks = geography.block_geoid[col_start:col_end]
            states = geography.state_fips[col_start:col_end]
            rerandomize_takeup(s, blocks, states, time_period)

    # Step 4: Build target filter
    target_filter = {}
    if domain_variables:
        target_filter["domain_variables"] = domain_variables

    # Step 5: Build sparse calibration matrix
    t_matrix = time.time()
    db_uri = f"sqlite:///{db_path}"
    builder = UnifiedMatrixBuilder(
        db_uri=db_uri,
        time_period=2024,
        dataset_path=dataset_path,
    )
    targets_df, X_sparse, target_names = builder.build_matrix(
        geography=geography,
        sim=sim,
        target_filter=target_filter,
        hierarchical_domains=hierarchical_domains,
        sim_modifier=sim_modifier,
    )

    builder.print_uprating_summary(targets_df)
    logger.info(
        "Matrix built in %.1f min",
        (time.time() - t_matrix) / 60,
    )
    logger.info(
        "Matrix shape: %s, nnz: %d",
        X_sparse.shape,
        X_sparse.nnz,
    )

    # Step 6: L0 calibration
    targets = targets_df["value"].values

    row_sums = np.array(X_sparse.sum(axis=1)).flatten()
    achievable = row_sums > 0
    logger.info(
        "Achievable: %d / %d targets",
        achievable.sum(),
        len(achievable),
    )

    weights = fit_l0_weights(
        X_sparse=X_sparse,
        targets=targets,
        lambda_l0=lambda_l0,
        epochs=epochs,
        device=device,
    )

    logger.info(
        "Total pipeline: %.1f min",
        (time.time() - t0) / 60,
    )
    return weights, targets_df, X_sparse, target_names


def main(argv=None):
    import json
    import time

    import pandas as pd

    try:
        if not sys.stderr.isatty():
            sys.stderr.reconfigure(line_buffering=True)
        if not sys.stdout.isatty():
            sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    args = parse_args(argv)

    from policyengine_us_data.storage import STORAGE_FOLDER

    dataset_path = args.dataset or str(
        STORAGE_FOLDER / "stratified_extended_cps_2024.h5"
    )
    db_path = args.db_path or str(
        STORAGE_FOLDER / "calibration" / "policy_data.db"
    )
    output_path = args.output or str(
        STORAGE_FOLDER / "calibration" / "unified_weights.npy"
    )

    if args.lambda_l0 is not None:
        lambda_l0 = args.lambda_l0
    elif args.preset is not None:
        lambda_l0 = PRESETS[args.preset]
    else:
        lambda_l0 = PRESETS["local"]

    domain_variables = None
    if args.domain_variables:
        domain_variables = [
            x.strip() for x in args.domain_variables.split(",")
        ]

    hierarchical_domains = None
    if args.hierarchical_domains:
        hierarchical_domains = [
            x.strip() for x in args.hierarchical_domains.split(",")
        ]

    t_start = time.time()

    weights, targets_df, X_sparse, target_names = run_calibration(
        dataset_path=dataset_path,
        db_path=db_path,
        n_clones=args.n_clones,
        lambda_l0=lambda_l0,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        domain_variables=domain_variables,
        hierarchical_domains=hierarchical_domains,
        skip_takeup_rerandomize=(args.skip_takeup_rerandomize),
    )

    # Save weights
    np.save(output_path, weights)
    logger.info("Weights saved to %s", output_path)
    print(f"OUTPUT_PATH:{output_path}")

    # Save diagnostics
    output_dir = Path(output_path).parent
    diag_df = compute_diagnostics(weights, X_sparse, targets_df, target_names)
    diag_path = output_dir / "unified_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)

    ach = diag_df[diag_df.achievable]
    err_pct = ach.abs_rel_error * 100
    logger.info(
        "Diagnostics: %d targets, "
        "mean=%.1f%%, median=%.1f%%, "
        "<10%%=%.1f%%, <25%%=%.1f%%",
        len(ach),
        err_pct.mean(),
        err_pct.median(),
        (err_pct < 10).mean() * 100,
        (err_pct < 25).mean() * 100,
    )

    # Save run config
    t_end = time.time()
    run_config = {
        "dataset": dataset_path,
        "db_path": db_path,
        "n_clones": args.n_clones,
        "lambda_l0": lambda_l0,
        "epochs": args.epochs,
        "device": args.device,
        "seed": args.seed,
        "domain_variables": domain_variables,
        "hierarchical_domains": hierarchical_domains,
        "n_targets": len(targets_df),
        "n_records": X_sparse.shape[1],
        "weight_sum": float(weights.sum()),
        "weight_nonzero": int((weights > 0).sum()),
        "mean_error_pct": float(err_pct.mean()),
        "elapsed_seconds": round(t_end - t_start, 1),
    }
    config_path = output_dir / "unified_run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    logger.info("Config saved to %s", config_path)
    print(f"LOG_PATH:{diag_path}")


if __name__ == "__main__":
    main()
