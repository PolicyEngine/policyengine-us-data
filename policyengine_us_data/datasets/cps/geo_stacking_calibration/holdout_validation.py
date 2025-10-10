import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp
from typing import Tuple, List, Dict, Optional


def create_holdout_split(
    X_sparse: sp.csr_matrix,
    targets: np.ndarray,
    target_groups: np.ndarray,
    holdout_group_indices: List[int],
) -> Tuple[Dict, Dict]:
    """
    Split data into training and holdout sets based on target group indices.

    Args:
        X_sparse: Sparse calibration matrix (n_targets x n_features)
        targets: Target values array
        target_groups: Group assignment for each target
        holdout_group_indices: List of group indices to put in holdout set

    Returns:
        train_data: Dict with X, targets, target_groups for training
        holdout_data: Dict with X, targets, target_groups for holdout
    """
    holdout_group_set = set(holdout_group_indices)

    # Create masks
    holdout_mask = np.isin(target_groups, list(holdout_group_set))
    train_mask = ~holdout_mask

    # Split data
    train_data = {
        "X": X_sparse[train_mask, :],
        "targets": targets[train_mask],
        "target_groups": target_groups[train_mask],
        "original_groups": target_groups[train_mask],  # Keep original IDs
    }

    holdout_data = {
        "X": X_sparse[holdout_mask, :],
        "targets": targets[holdout_mask],
        "target_groups": target_groups[holdout_mask],
        "original_groups": target_groups[holdout_mask],  # Keep original IDs
    }

    # Renumber groups to be consecutive for model training
    train_data["target_groups"] = renumber_groups(train_data["target_groups"])
    # For holdout, also renumber for consistency in model evaluation
    # But keep original_groups for reporting
    holdout_data["target_groups"] = renumber_groups(
        holdout_data["target_groups"]
    )

    return train_data, holdout_data


def renumber_groups(groups: np.ndarray) -> np.ndarray:
    """Renumber groups to be consecutive starting from 0."""
    unique_groups = np.unique(groups)
    mapping = {old: new for new, old in enumerate(unique_groups)}
    return np.array([mapping[g] for g in groups])


def calculate_group_losses(
    model,
    X_sparse: sp.csr_matrix,
    targets: np.ndarray,
    target_groups: np.ndarray,
    loss_type: str = "relative",
    original_groups: np.ndarray = None,
) -> Dict[str, float]:
    """
    Calculate mean loss per group and overall mean group loss.

    Args:
        model: Trained SparseCalibrationWeights model
        X_sparse: Sparse calibration matrix
        targets: Target values
        target_groups: Group assignments (possibly renumbered)
        loss_type: Type of loss ("relative" or "absolute")
        original_groups: Original group IDs (optional, for reporting)

    Returns:
        Dict with per-group losses and mean group loss
    """
    with torch.no_grad():
        predictions = model.predict(X_sparse).cpu().numpy()

    # Calculate per-target losses
    if loss_type == "relative":
        # For reporting, use absolute relative error to match L0's verbose output
        # L0 reports |relative_error|, not squared
        losses = np.abs((predictions - targets) / (targets + 1))
    else:
        # For absolute, also use non-squared for consistency
        losses = np.abs(predictions - targets)

    # Use original groups if provided, otherwise use renumbered groups
    groups_for_reporting = (
        original_groups if original_groups is not None else target_groups
    )

    # Calculate mean loss per group
    unique_groups = np.unique(groups_for_reporting)
    group_losses = {}

    for group_id in unique_groups:
        group_mask = groups_for_reporting == group_id
        group_losses[int(group_id)] = np.mean(losses[group_mask])

    # Mean across groups (not weighted by group size)
    mean_group_mare = np.mean(list(group_losses.values()))

    return {
        "per_group": group_losses,
        "mean_group_mare": mean_group_mare,
        "n_groups": len(unique_groups),
    }


def run_holdout_experiment(
    X_sparse: sp.csr_matrix,
    targets: np.ndarray,
    target_groups: np.ndarray,
    holdout_group_indices: List[int],
    model_params: Dict,
    training_params: Dict,
) -> Dict:
    """
    Run a single holdout experiment with specified groups.

    Args:
        X_sparse: Full sparse calibration matrix
        targets: Full target values
        target_groups: Full group assignments
        holdout_group_indices: Groups to hold out
        model_params: Parameters for SparseCalibrationWeights
        training_params: Parameters for model.fit()

    Returns:
        Dict with training and holdout results
    """
    from l0.calibration import SparseCalibrationWeights

    # Split data
    train_data, holdout_data = create_holdout_split(
        X_sparse, targets, target_groups, holdout_group_indices
    )

    print(
        f"Training samples: {len(train_data['targets'])}, "
        f"Holdout samples: {len(holdout_data['targets'])}"
    )
    print(
        f"Training groups: {len(np.unique(train_data['target_groups']))}, "
        f"Holdout groups: {len(np.unique(holdout_data['target_groups']))}"
    )

    # Create and train model
    model = SparseCalibrationWeights(
        n_features=X_sparse.shape[1], **model_params
    )

    model.fit(
        M=train_data["X"],
        y=train_data["targets"],
        target_groups=train_data["target_groups"],
        **training_params,
    )

    # Calculate losses with original group IDs
    train_losses = calculate_group_losses(
        model,
        train_data["X"],
        train_data["targets"],
        train_data["target_groups"],
        training_params.get("loss_type", "relative"),
        original_groups=train_data["original_groups"],
    )

    holdout_losses = calculate_group_losses(
        model,
        holdout_data["X"],
        holdout_data["targets"],
        holdout_data["target_groups"],
        training_params.get("loss_type", "relative"),
        original_groups=holdout_data["original_groups"],
    )

    # Get sparsity info
    active_info = model.get_active_weights()

    # Get the actual weight values
    with torch.no_grad():
        weights = model.get_weights(deterministic=True).cpu().numpy()

    results = {
        "train_mean_group_mare": train_losses["mean_group_mare"],
        "holdout_mean_group_mare": holdout_losses["mean_group_mare"],
        "train_group_losses": train_losses["per_group"],
        "holdout_group_losses": holdout_losses["per_group"],
        "n_train_groups": train_losses["n_groups"],
        "n_holdout_groups": holdout_losses["n_groups"],
        "active_weights": active_info["count"],
        "total_weights": X_sparse.shape[1],
        "sparsity_pct": 100 * (1 - active_info["count"] / X_sparse.shape[1]),
        "weights": weights,  # Store the weight vector
        "model": model,  # Optionally store the entire model object
    }

    return results


def compute_aggregate_losses(
    X_sparse: sp.csr_matrix,
    weights: np.ndarray,
    targets_df: pd.DataFrame,
    target_groups: np.ndarray,
    training_group_ids: List[int],
    holdout_group_ids: List[int],
) -> Dict:
    """
    Compute aggregate losses showing how well CD/state predictions aggregate to higher levels.
    Returns losses organized by group_id with 'state' and 'national' sub-keys.

    Args:
        X_sparse: Calibration matrix
        weights: Calibrated weights
        targets_df: DataFrame with geographic info and group assignments
        target_groups: Group assignments array
        training_group_ids: Groups used in training
        holdout_group_ids: Groups held out

    Returns:
        Dict with train_aggregate_losses and holdout_aggregate_losses
    """

    # Calculate predictions
    predictions = X_sparse @ weights
    targets_df = targets_df.copy()
    targets_df["prediction"] = predictions
    targets_df["group_id"] = target_groups

    # Identify which groups are training vs holdout
    train_aggregate_losses = {}
    holdout_aggregate_losses = {}

    # Process each unique group
    for group_id in np.unique(target_groups):
        group_mask = target_groups == group_id
        group_targets = targets_df[group_mask].copy()

        if len(group_targets) == 0:
            continue

        # Determine if this is a training or holdout group
        is_training = group_id in training_group_ids
        is_holdout = group_id in holdout_group_ids

        if not (is_training or is_holdout):
            continue  # Skip unknown groups

        # Get the primary geographic level of this group
        geo_ids = group_targets["geographic_id"].unique()

        # Determine the geographic level
        if "US" in geo_ids and len(geo_ids) == 1:
            # National-only group - no aggregation possible, skip
            continue
        elif all(len(str(g)) > 2 for g in geo_ids if g != "US"):
            # CD-level group - can aggregate to state and national
            primary_level = "cd"
        elif all(len(str(g)) <= 2 for g in geo_ids if g != "US"):
            # State-level group - can aggregate to national only
            primary_level = "state"
        else:
            # Mixed or unclear - skip
            continue

        aggregate_losses = {}

        # For CD-level groups, compute state and national aggregation
        if primary_level == "cd":
            # Extract state from CD codes
            group_targets["state"] = group_targets["geographic_id"].apply(
                lambda x: (
                    x[:2]
                    if len(str(x)) == 4
                    else str(x)[:-2] if len(str(x)) == 3 else str(x)[:2]
                )
            )

            # Get the variable(s) for this group
            variables = group_targets["variable"].unique()

            state_losses = []
            for variable in variables:
                var_targets = group_targets[
                    group_targets["variable"] == variable
                ]

                # Aggregate by state
                state_aggs = var_targets.groupby("state").agg(
                    {"value": "sum", "prediction": "sum"}
                )

                # Compute relative error for each state
                for state_id, row in state_aggs.iterrows():
                    if row["value"] != 0:
                        rel_error = abs(
                            (row["prediction"] - row["value"]) / row["value"]
                        )
                        state_losses.append(rel_error)

            # Mean across all states
            if state_losses:
                aggregate_losses["state"] = np.mean(state_losses)

            # National aggregation
            total_actual = group_targets["value"].sum()
            total_pred = group_targets["prediction"].sum()
            if total_actual != 0:
                aggregate_losses["national"] = abs(
                    (total_pred - total_actual) / total_actual
                )

        # For state-level groups, compute national aggregation only
        elif primary_level == "state":
            total_actual = group_targets["value"].sum()
            total_pred = group_targets["prediction"].sum()
            if total_actual != 0:
                aggregate_losses["national"] = abs(
                    (total_pred - total_actual) / total_actual
                )

        # Store in appropriate dict
        if aggregate_losses:
            if is_training:
                train_aggregate_losses[group_id] = aggregate_losses
            else:
                holdout_aggregate_losses[group_id] = aggregate_losses

    return {
        "train_aggregate_losses": train_aggregate_losses,
        "holdout_aggregate_losses": holdout_aggregate_losses,
    }


def simple_holdout(
    X_sparse,
    targets,
    target_groups,
    init_weights,
    holdout_group_ids,
    targets_df=None,  # Optional: needed for hierarchical checks
    check_hierarchical=False,  # Optional: enable hierarchical analysis
    epochs=10,
    lambda_l0=8e-7,
    lr=0.2,
    verbose_spacing=5,
    device="cuda",  # Add device parameter
):
    """
    Simple holdout validation for notebooks - no DataFrame dependencies.

    Args:
        X_sparse: Sparse matrix from cd_matrix_sparse.npz
        targets: Target values from cd_targets_array.npy
        target_groups: Group assignments from cd_target_groups.npy
        init_weights: Initial weights from cd_init_weights.npy
        holdout_group_ids: List of group IDs to hold out (e.g. [10, 25, 47])
        targets_df: Optional DataFrame with geographic info for hierarchical checks
        check_hierarchical: If True and targets_df provided, analyze hierarchical consistency
        epochs: Training epochs
        lambda_l0: L0 regularization parameter
        lr: Learning rate
        verbose_spacing: How often to print progress
        device: 'cuda' for GPU, 'cpu' for CPU

    Returns:
        Dictionary with train/holdout losses, summary stats, and optionally hierarchical analysis
    """

    # Model parameters (matching calibrate_cds_sparse.py)
    model_params = {
        "beta": 2 / 3,
        "gamma": -0.1,
        "zeta": 1.1,
        "init_keep_prob": 0.999,
        "init_weights": init_weights,
        "log_weight_jitter_sd": 0.05,
        "log_alpha_jitter_sd": 0.01,
        "device": device,  # Pass device to model
    }

    training_params = {
        "lambda_l0": lambda_l0,
        "lambda_l2": 0,
        "lr": lr,
        "epochs": epochs,
        "loss_type": "relative",
        "verbose": True,
        "verbose_freq": verbose_spacing,
    }

    # Use the existing run_holdout_experiment function
    results = run_holdout_experiment(
        X_sparse=X_sparse,
        targets=targets,
        target_groups=target_groups,
        holdout_group_indices=holdout_group_ids,
        model_params=model_params,
        training_params=training_params,
    )

    # Add hierarchical consistency check if requested
    if check_hierarchical and targets_df is not None:
        # Get training group IDs (all groups not in holdout)
        all_group_ids = set(np.unique(target_groups))
        training_group_ids = list(all_group_ids - set(holdout_group_ids))

        # Compute aggregate losses
        aggregate_results = compute_aggregate_losses(
            X_sparse=X_sparse,
            weights=results["weights"],
            targets_df=targets_df,
            target_groups=target_groups,
            training_group_ids=training_group_ids,
            holdout_group_ids=holdout_group_ids,
        )

        # Add to results
        results["train_aggregate_losses"] = aggregate_results[
            "train_aggregate_losses"
        ]
        results["holdout_aggregate_losses"] = aggregate_results[
            "holdout_aggregate_losses"
        ]

        # Print summary if available
        if (
            aggregate_results["train_aggregate_losses"]
            or aggregate_results["holdout_aggregate_losses"]
        ):
            print("\n" + "=" * 60)
            print("HIERARCHICAL AGGREGATION PERFORMANCE")
            print("=" * 60)

            # Show training group aggregates
            if aggregate_results["train_aggregate_losses"]:
                print("\nTraining groups (CD→State/National aggregation):")
                for group_id, losses in list(
                    aggregate_results["train_aggregate_losses"].items()
                )[:5]:
                    print(f"  Group {group_id}:", end="")
                    if "state" in losses:
                        print(f" State={losses['state']:.2%}", end="")
                    if "national" in losses:
                        print(f" National={losses['national']:.2%}", end="")
                    print()

            # Show holdout group aggregates
            if aggregate_results["holdout_aggregate_losses"]:
                print("\nHoldout groups (CD→State/National aggregation):")
                for group_id, losses in list(
                    aggregate_results["holdout_aggregate_losses"].items()
                )[:5]:
                    print(f"  Group {group_id}:", end="")
                    if "state" in losses:
                        print(f" State={losses['state']:.2%}", end="")
                    if "national" in losses:
                        print(f" National={losses['national']:.2%}", end="")
                    print()
                print(
                    "  → Good performance here shows hierarchical generalization!"
                )

    return results
