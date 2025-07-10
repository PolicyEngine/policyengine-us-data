from policyengine_us_data.utils.loss import build_loss_matrix
from policyengine_core.data import Dataset
from policyengine_us import Microsimulation
import numpy as np
import pandas as pd
import h5py
from policyengine_us_data.storage import STORAGE_FOLDER


def create_calibration_log_file(file_path):
    dataset = Dataset.from_file(file_path)

    loss_matrix = build_loss_matrix(dataset, 2024)

    sim = Microsimulation(dataset=dataset)

    estimates = sim.calculate("household_weight", 2024).values @ loss_matrix[0]
    target_names = loss_matrix[0].columns
    target_values = loss_matrix[1]

    df = pd.DataFrame(
        {
            "target_name": target_names,
            "estimate": estimates,
            "target": target_values,
        }
    )
    df["epoch"] = 0
    df["error"] = df["estimate"] - df["target"]
    df["rel_error"] = df["error"] / df["target"]
    df["abs_error"] = df["error"].abs()
    df["rel_abs_error"] = df["abs_error"] / df["target"].abs()
    df["loss"] = (df["rel_error"] ** 2).mean()

    df.to_csv(file_path.replace(".h5", "_calibration_log.csv"), index=False)


def minimise_dataset(
    dataset, output_path: str, loss_rel_change_max: float
) -> None:
    create_calibration_log_file(dataset)

    loss_matrix = build_loss_matrix(dataset, 2024)

    sim = Microsimulation(dataset=dataset)

    weights = sim.calculate("household_weight", 2024).values
    estimate_matrix, targets = loss_matrix
    is_national = estimate_matrix.columns.str.startswith("nation/")
    nation_normalisation_factor = is_national * (1 / is_national.sum())
    state_normalisation_factor = ~is_national * (1 / (~is_national).sum())
    normalisation_factor = np.where(
        is_national, nation_normalisation_factor, state_normalisation_factor
    )
    weights @ estimate_matrix

    def get_loss_from_mask(
        inclusion_mask, estimate_matrix, targets, normalisation_factor
    ):
        """
        Calculate the loss based on the inclusion mask and the estimate matrix.
        """
        masked_weights = weights.copy()
        original_weight_total = masked_weights.sum()
        if (~inclusion_mask).sum() > 0:
            masked_weights[~inclusion_mask] = 0
        masked_weight_total = masked_weights.sum()
        masked_weights[inclusion_mask] *= (
            original_weight_total / masked_weight_total
        )
        estimates = masked_weights @ estimate_matrix
        rel_error = ((estimates - targets) + 1) / (targets + 1)
        loss = ((rel_error * normalisation_factor) ** 2).mean()

        return loss

    COUNT_ITERATIONS = 5
    VIEW_FRACTION_PER_ITERATION = 0.3
    FRACTION_REMOVE_PER_ITERATION = 0.1
    from tqdm import tqdm

    full_mask = np.ones_like(weights, dtype=bool)
    for i in range(COUNT_ITERATIONS):
        inclusion_mask = full_mask.copy()
        baseline_loss = get_loss_from_mask(
            inclusion_mask, estimate_matrix, targets, normalisation_factor
        )
        household_loss_rel_changes = []
        indices = np.random.choice(
            np.arange(len(weights)),
            size=int(len(weights) * VIEW_FRACTION_PER_ITERATION),
            replace=False,
        )
        for household_index in tqdm(indices):
            # Skip if this household is already excluded
            if not inclusion_mask[household_index]:
                household_loss_rel_changes.append(np.inf)
                continue
            # Calculate loss if this household is removed
            inclusion_mask = inclusion_mask.copy()
            inclusion_mask[household_index] = False
            loss = get_loss_from_mask(
                inclusion_mask, estimate_matrix, targets, normalisation_factor
            )
            rel_change = (loss - baseline_loss) / baseline_loss
            household_loss_rel_changes.append(rel_change)
        inclusion_mask = full_mask.copy()
        household_loss_rel_changes = np.array(household_loss_rel_changes)
        # Sort by the relative change in loss
        sorted_indices = np.argsort(household_loss_rel_changes)
        # Remove the worst households
        num_to_remove = int(len(weights) * FRACTION_REMOVE_PER_ITERATION)
        worst_indices = sorted_indices[:num_to_remove]
        inclusion_mask[worst_indices] = False
        # Calculate the new loss
        new_loss = get_loss_from_mask(
            inclusion_mask, estimate_matrix, targets, normalisation_factor
        )
        rel_change = (new_loss - baseline_loss) / baseline_loss
        if rel_change > loss_rel_change_max:
            print(
                f"Iteration {i + 1}: Loss changed from {baseline_loss} to {new_loss}, which is too high ({rel_change:.2%}). Stopping."
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

    household_ids = sim.calculate("household_id", 2024).values
    remaining_households = household_ids[full_mask]

    # At this point we have a mask of households to keep

    # I'm saving to a csv for ease of debugging, but we need to save to a .h5 file

    df = sim.to_input_dataframe()
    smaller_df = df[df["household_id__2024"].isin(remaining_households)]

    weight_rel_change = (
        smaller_df["household_weight__2024"].sum()
        / df["household_weight__2024"].sum()
    )
    print(f"Weight relative change: {weight_rel_change:.2%}")

    sim = Microsimulation(dataset=smaller_df)

    sim.set_input(
        "household_weight",
        2024,
        sim.calculate("household_weight", 2024).values / weight_rel_change,
    )

    data = {}

    for variable in sim.input_variables:
        data[variable] = {2024: sim.calculate(variable, 2024).values}
        if data[variable][2024].dtype == "object":
            data[variable][2024] = data[variable][2024].astype("S")

    with h5py.File(output_path, "w") as f:
        for variable, values in data.items():
            for year, value in values.items():
                f.create_dataset(f"{variable}/{year}", data=value)
    print(f"Saved minimised dataset to {output_path}")

    create_calibration_log_file(output_path)


if __name__ == "__main__":
    # Example usage
    files = [
        STORAGE_FOLDER / "enhanced_cps_2024.h5",
    ]

    for file in files:
        output_path = file.with_name(file.stem + "_minimised.h5")
        minimise_dataset(file, output_path, loss_rel_change_max=10)
