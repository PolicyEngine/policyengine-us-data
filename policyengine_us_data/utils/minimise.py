from policyengine_us_data.utils.loss import build_loss_matrix
from policyengine_core.data import Dataset
from policyengine_us import Microsimulation
import numpy as np
import pandas as pd
import h5py
from policyengine_us_data.storage import STORAGE_FOLDER
from typing import Optional, Callable
from policyengine_us_data.datasets.cps.enhanced_cps import reweight

bad_targets = [
    "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Head of Household",
    "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Head of Household",
    "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
    "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
    "nation/irs/count/count/AGI in 10k-15k/taxable/Head of Household",
    "nation/irs/count/count/AGI in 15k-20k/taxable/Head of Household",
    "nation/irs/count/count/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
    "nation/irs/count/count/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
]


def create_calibration_log_file(file_path, epoch=0):
    dataset = Dataset.from_file(file_path)
    sim = Microsimulation(dataset=dataset)

    loss_matrix, targets = build_loss_matrix(dataset, 2024)

    bad_mask = loss_matrix.columns.isin(bad_targets)
    keep_mask_bool = ~bad_mask
    keep_idx = np.where(keep_mask_bool)[0]
    loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
    targets_clean = targets[keep_idx]

    assert loss_matrix_clean.shape[1] == targets_clean.size

    estimates = (
        sim.calculate("household_weight", 2024).values @ loss_matrix_clean
    )
    target_names = loss_matrix_clean.columns

    # Calculate and print some key metrics
    errors = estimates - targets_clean
    rel_errors = errors / targets_clean

    df = pd.DataFrame(
        {
            "target_name": target_names,
            "estimate": estimates,
            "target": targets_clean,
        }
    )
    df["epoch"] = epoch
    df["error"] = df["estimate"] - df["target"]
    df["rel_error"] = df["error"] / df["target"]
    df["abs_error"] = df["error"].abs()
    df["rel_abs_error"] = (
        df["abs_error"] / df["target"].abs()
        if df["target"].abs().sum() > 0
        else np.nan
    )
    df["loss"] = (df["rel_error"] ** 2).mean()

    df.to_csv(
        str(file_path).replace(".h5", "_calibration_log.csv"), index=False
    )


def losses_for_candidates(
    base_weights: np.ndarray,
    idxs: np.ndarray,
    est_mat: np.ndarray,
    targets: np.ndarray,
    norm: np.ndarray,
    chunk_size: Optional[int] = 25_000,
) -> np.ndarray:
    """
    Return the loss value *for each* candidate deletion in `idxs`
    in one matrix multiplication.

    Parameters
    ----------
    base_weights : (n,) original weight vector
    idxs         : (k,) candidate row indices to zero-out
    est_mat      : (n, m) estimate matrix
    targets      : (m,) calibration targets
    norm         : (m,) normalisation factors
    chunk_size   : max number of candidates to process at once

    Returns
    -------
    losses       : (k,) loss if row i were removed (and weights rescaled)
    """
    W = base_weights
    total = W.sum()
    k = len(idxs)
    losses = np.empty(k, dtype=float)

    # Work through the candidate list in blocks
    for start in range(0, k, chunk_size):
        stop = min(start + chunk_size, k)
        part = idxs[start:stop]  # (p,) where p ≤ chunk_size
        p = len(part)

        # Build the delta matrix only for this chunk
        delta = np.zeros((p, len(W)))
        delta[np.arange(p), part] = -W[part]

        keep_total = total + delta.sum(axis=1)  # (p,)
        delta *= (total / keep_total)[:, None]

        # Matrix–matrix multiply → one matrix multiplication per chunk
        ests = (W + delta) @ est_mat  # (p, m)
        rel_err = ((ests - targets) + 1) / (targets + 1)
        losses[start:stop] = ((rel_err * norm) ** 2).mean(axis=1)

    return losses


def get_loss_from_mask(
    weights, inclusion_mask, estimate_matrix, targets, normalisation_factor
):
    """
    Calculate the loss based on the inclusion mask and the estimate matrix.
    """
    # Step 1: Apply mask and rescale weights
    masked_weights = weights.copy()
    original_weight_total = masked_weights.sum()
    if (~inclusion_mask).sum() > 0:
        masked_weights[~inclusion_mask] = 0
    masked_weight_total = masked_weights.sum()
    masked_weights[inclusion_mask] *= (
        original_weight_total / masked_weight_total
    )

    # Step 2: Re-calibrate the masked weights to hit targets
    # Only calibrate the included households
    included_weights = masked_weights[inclusion_mask]
    included_estimate_matrix = estimate_matrix.iloc[
        inclusion_mask
    ]  # Keep as DataFrame

    # Call reweight function to calibrate the selected households
    calibrated_weights_included = reweight(
        included_weights,
        included_estimate_matrix,
        targets,
        epochs=250,
    )

    # Put calibrated weights back into full array
    calibrated_weights = np.zeros_like(masked_weights)
    calibrated_weights[inclusion_mask] = calibrated_weights_included

    # Calculate estimates and loss from calibrated weights
    estimates = calibrated_weights @ estimate_matrix
    rel_error = ((estimates - targets) + 1) / (targets + 1)
    loss = ((rel_error * normalisation_factor) ** 2).mean()

    return loss


def candidate_loss_contribution(
    weights: np.ndarray,
    estimate_matrix: np.ndarray,
    targets: np.ndarray,
    normalisation_factor: np.ndarray,
    loss_rel_change_max: float,
    count_iterations: int = 5,
    view_fraction_per_iteration: float = 0.3,
    fraction_remove_per_iteration: float = 0.1,
) -> np.ndarray:
    """
    Minimization approach based on candidate loss contribution.

    This function iteratively removes households that contribute least to the loss,
    maintaining the calibration quality within the specified tolerance.

    Parameters
    ----------
    weights : (n,) household weights
    estimate_matrix : (n, m) matrix mapping weights to estimates
    targets : (m,) calibration targets
    normalisation_factor : (m,) normalisation factors for different targets
    loss_rel_change_max : maximum allowed relative change in loss
    count_iterations : number of iterations to perform
    view_fraction_per_iteration : fraction of households to evaluate each iteration
    fraction_remove_per_iteration : fraction of households to remove each iteration

    Returns
    -------
    inclusion_mask : (n,) boolean mask of households to keep
    """
    from tqdm import tqdm

    full_mask = np.ones_like(weights, dtype=bool)

    for i in range(count_iterations):
        inclusion_mask = full_mask.copy()
        baseline_loss = get_loss_from_mask(
            weights,
            inclusion_mask,
            estimate_matrix,
            targets,
            normalisation_factor,
        )

        # Sample only households that are currently included
        indices = np.random.choice(
            np.where(full_mask)[0],
            size=int(full_mask.sum() * view_fraction_per_iteration),
            replace=False,
        )
        # 2. compute losses for the batch in one shot
        candidate_losses = losses_for_candidates(
            weights, indices, estimate_matrix, targets, normalisation_factor
        )
        # 3. convert to relative change vs. baseline
        household_loss_rel_changes = (
            candidate_losses - baseline_loss
        ) / baseline_loss

        inclusion_mask = full_mask.copy()
        household_loss_rel_changes = np.array(household_loss_rel_changes)
        # Sort by the relative change in loss
        sorted_indices = np.argsort(household_loss_rel_changes)

        # Remove the worst households
        num_to_remove = int(len(weights) * fraction_remove_per_iteration)
        worst_indices = indices[sorted_indices[:num_to_remove]]
        inclusion_mask[worst_indices] = False

        # Calculate the new loss
        new_loss = get_loss_from_mask(
            weights,
            inclusion_mask,
            estimate_matrix,
            targets,
            normalisation_factor,
        )
        rel_change = (new_loss - baseline_loss) / baseline_loss

        if rel_change > loss_rel_change_max:
            print(
                f"Iteration {i + 1}: Loss changed from {baseline_loss} to {new_loss}, "
                f"which is too high ({rel_change:.2%}). Stopping."
            )
            break

        print(
            f"Iteration {i + 1}: Loss changed from {baseline_loss} to {new_loss}"
        )
        print(
            f"Removed {num_to_remove} households with worst relative loss changes."
        )

        # Update the full mask
        full_mask &= inclusion_mask

    return full_mask


def random_sampling_minimization(
    weights,
    estimate_matrix,
    targets,
    normalisation_factor,
    random=True,
    target_fractions=[0.5, 0.6, 0.7, 0.8, 0.9],
):
    """A simple random sampling approach"""
    n = len(weights)

    household_weights_normalized = weights / weights.sum()

    final_mask = None
    lowest_loss = float("inf")
    for fraction in target_fractions:
        target_size = int(n * fraction)
        # Random sampling with multiple attempts
        best_mask = None
        best_loss = float("inf")

        for _ in range(3):  # Try 3 random samples
            mask = np.zeros(n, dtype=bool)
            mask[
                np.random.choice(
                    n,
                    target_size,
                    p=household_weights_normalized if random else None,
                    replace=False,
                )
            ] = True

            loss = get_loss_from_mask(
                weights, mask, estimate_matrix, targets, normalisation_factor
            )

            if loss < best_loss:
                best_loss = loss
                best_mask = mask

        if lowest_loss > best_loss:
            lowest_loss = best_loss
            final_mask = best_mask

    return final_mask


def minimize_dataset(
    dataset,
    output_path: str,
    minimization_function: Callable = candidate_loss_contribution,
    loss_matrix: Optional[pd.DataFrame] = None,
    targets: Optional[np.ndarray] = None,
    **kwargs,
) -> None:
    """
    Main function to minimize a dataset using a specified minimization approach.

    Parameters
    ----------
    dataset : path to the dataset file or Dataset object
    output_path : path where the minimized dataset will be saved
    loss_rel_change_max : maximum allowed relative change in loss
    minimization_function : function that implements the minimization logic
    **kwargs : additional arguments to pass to the minimization function
    """
    dataset = str(dataset)
    create_calibration_log_file(dataset)

    dataset = Dataset.from_file(dataset)
    if loss_matrix is None or targets is None:
        loss_matrix, targets = build_loss_matrix(dataset, 2024)

        bad_mask = loss_matrix.columns.isin(bad_targets)
        keep_mask_bool = ~bad_mask
        keep_idx = np.where(keep_mask_bool)[0]
        loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
        targets_clean = targets[keep_idx]
        assert loss_matrix_clean.shape[1] == targets_clean.size

    sim = Microsimulation(dataset=dataset)

    weights = sim.calculate("household_weight", 2024).values
    is_national = loss_matrix_clean.columns.str.startswith("nation/")
    nation_normalisation_factor = is_national * (1 / is_national.sum())
    state_normalisation_factor = ~is_national * (1 / (~is_national).sum())
    normalisation_factor = np.where(
        is_national, nation_normalisation_factor, state_normalisation_factor
    )

    # Call the minimization function
    inclusion_mask = minimization_function(
        weights=weights,
        estimate_matrix=loss_matrix_clean,
        targets=targets_clean,
        normalisation_factor=normalisation_factor,
        **kwargs,  # Allows for passing either loss_rel_change_max OR target_fractions, depending on normalisation_factor choice.
    )

    # Extract household IDs for remaining households
    household_ids = sim.calculate("household_id", 2024).values
    remaining_households = household_ids[inclusion_mask]

    # Create a smaller dataset with only the remaining households
    df = sim.to_input_dataframe()
    smaller_df = df[df["household_id__2024"].isin(remaining_households)]

    weight_rel_change = (
        smaller_df["household_weight__2024"].sum()
        / df["household_weight__2024"].sum()
    )
    print(f"Weight relative change: {weight_rel_change:.2%}")

    # Create new simulation with smaller dataset
    sim = Microsimulation(dataset=smaller_df)

    # Rescale weights to maintain total
    initial_weights = (
        sim.calculate("household_weight", 2024).values / weight_rel_change
    )

    # Re-calibrate the final selected households to hit targets
    print("Re-calibrating final selected households...")

    # Build loss matrix for the smaller dataset
    smaller_loss_matrix, smaller_targets = build_loss_matrix(sim.dataset, 2024)

    # Apply same filtering as before
    bad_mask = smaller_loss_matrix.columns.isin(bad_targets)
    keep_mask_bool = ~bad_mask
    keep_idx = np.where(keep_mask_bool)[0]
    smaller_loss_matrix_clean = smaller_loss_matrix.iloc[:, keep_idx]
    smaller_targets_clean = smaller_targets[keep_idx]

    calibrated_weights = reweight(
        initial_weights,
        smaller_loss_matrix_clean,  # Now matches the smaller dataset size
        smaller_targets_clean,
        epochs=250,  # Reduced epochs for faster processing
    )
    sim.set_input("household_weight", 2024, calibrated_weights)
    print("Final calibration completed successfully")
    # Prepare data for saving
    data = {}
    for variable in sim.input_variables:
        data[variable] = {2024: sim.calculate(variable, 2024).values}
        if data[variable][2024].dtype == "object":
            data[variable][2024] = data[variable][2024].astype("S")

    # Save to HDF5 file
    with h5py.File(output_path, "w") as f:
        for variable, values in data.items():
            for year, value in values.items():
                f.create_dataset(f"{variable}/{year}", data=value)

    print(f"Saved minimised dataset to {output_path}")
    create_calibration_log_file(output_path, epoch=500)


if __name__ == "__main__":
    # Example usage
    files = [
        STORAGE_FOLDER / "enhanced_cps_2024.h5",
    ]

    for file in files:
        output_path = file.with_name(file.stem + "_minimised.h5")
        minimize_dataset(
            file,
            output_path,
        )
