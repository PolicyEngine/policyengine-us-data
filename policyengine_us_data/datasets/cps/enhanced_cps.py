from policyengine_core.data import Dataset
import pandas as pd
from policyengine_us_data.utils import (
    build_loss_matrix,
    HardConcrete,
)
import numpy as np
from typing import Type
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.extended_cps import (
    ExtendedCPS_2024,
    CPS_2019,
    CPS_2024,
)
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

try:
    import torch
except ImportError:
    torch = None


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


def reweight(
    original_weights,
    loss_matrix,
    targets_array,
    dropout_rate=0.05,
    epochs=500,
    log_path="calibration_log.csv",
    l0_lambda=1e-5,
    init_mean=0.999,
    temperature=0.5,
    sparse=False,
):
    if loss_matrix.shape[1] == 0:
        raise ValueError("loss_matrix has no columns after filtering")

    # Store column names before converting to tensor
    target_names = np.array(loss_matrix.columns)
    is_national = loss_matrix.columns.str.startswith("nation/")

    # Keep numpy versions for final diagnostics
    loss_matrix_numpy = loss_matrix.values
    targets_array_numpy = np.array(targets_array)

    # Convert to tensors for training
    loss_matrix_tensor = torch.tensor(loss_matrix_numpy, dtype=torch.float32)
    targets_array_tensor = torch.tensor(
        targets_array_numpy, dtype=torch.float32
    )

    # Compute normalization factors
    nation_normalisation_factor = is_national * (1 / is_national.sum())
    state_normalisation_factor = ~is_national * (1 / (~is_national).sum())
    normalisation_factor = np.where(
        is_national, nation_normalisation_factor, state_normalisation_factor
    )
    normalisation_factor_tensor = torch.tensor(
        normalisation_factor, dtype=torch.float32
    )
    inv_mean_normalisation = 1 / np.mean(normalisation_factor)

    # Initialize weights
    weights = torch.tensor(
        np.log(original_weights), requires_grad=True, dtype=torch.float32
    )

    def loss(weights):
        if torch.isnan(weights).any():
            raise ValueError("Weights contain NaNs")
        if torch.isnan(loss_matrix_tensor).any():
            raise ValueError("Loss matrix contains NaNs")

        estimate = weights @ loss_matrix_tensor

        if torch.isnan(estimate).any():
            raise ValueError("Estimate contains NaNs")

        rel_error = (
            ((estimate - targets_array_tensor) + 1)
            / (targets_array_tensor + 1)
        ) ** 2
        rel_error_normalized = (
            inv_mean_normalisation * rel_error * normalisation_factor_tensor
        )

        if torch.isnan(rel_error_normalized).any():
            raise ValueError("Relative error contains NaNs")

        return rel_error_normalized.mean()

    def dropout_weights(weights, p):
        if p == 0:
            return weights
        mask = torch.rand_like(weights) < p
        mean = weights[~mask].mean()
        masked_weights = weights.clone()
        masked_weights[mask] = mean
        return masked_weights

    def compute_diagnostics(final_weights, label=""):
        """Helper function to compute and log diagnostics"""
        estimate = final_weights @ loss_matrix_numpy
        rel_error = (
            ((estimate - targets_array_numpy) + 1) / (targets_array_numpy + 1)
        ) ** 2
        within_10_percent_mask = np.abs(estimate - targets_array_numpy) <= (
            0.10 * np.abs(targets_array_numpy)
        )
        percent_within_10 = np.mean(within_10_percent_mask) * 100

        logger.info(
            f"\n\n---{label} Solutions: reweighting quick diagnostics----\n"
        )
        logger.info(
            f"{np.sum(final_weights == 0)} are zero, {np.sum(final_weights != 0)} weights are nonzero"
        )
        logger.info(
            f"rel_error: min: {np.min(rel_error):.2f}\n"
            f"max: {np.max(rel_error):.2f}\n"
            f"mean: {np.mean(rel_error):.2f}\n"
            f"median: {np.median(rel_error):.2f}\n"
            f"Within 10% of target: {percent_within_10:.2f}%"
        )
        logger.info("Relative error over 100% for:")
        for i in np.where(rel_error > 1)[0]:
            logger.info(f"target_name: {target_names[i]}")
            logger.info(f"target_value: {targets_array_numpy[i]}")
            logger.info(f"estimate_value: {estimate[i]}")
            logger.info(f"has rel_error: {rel_error[i]:.2f}\n")
        logger.info("---End of reweighting quick diagnostics------")

    if not sparse:
        # Dense training
        optimizer = torch.optim.Adam([weights], lr=3e-1)
        from tqdm import trange

        start_loss = None
        iterator = trange(epochs)
        performance = pd.DataFrame()

        for i in iterator:
            optimizer.zero_grad()
            weights_ = dropout_weights(weights, dropout_rate)
            l = loss(torch.exp(weights_))

            if (log_path is not None) and (i % 10 == 0):
                with torch.no_grad():
                    estimates = (
                        torch.exp(weights) @ loss_matrix_tensor
                    ).numpy()
                df = pd.DataFrame(
                    {
                        "target_name": target_names,
                        "estimate": estimates,
                        "target": targets_array_numpy,
                    }
                )
                df["epoch"] = i
                df["error"] = df.estimate - df.target
                df["rel_error"] = df.error / df.target
                df["abs_error"] = df.error.abs()
                df["rel_abs_error"] = df.rel_error.abs()
                df["loss"] = df.rel_abs_error**2
                performance = pd.concat([performance, df], ignore_index=True)

            if (log_path is not None) and (i % 1000 == 0):
                performance.to_csv(log_path, index=False)

            if start_loss is None:
                start_loss = l.item()
            loss_rel_change = (l.item() - start_loss) / start_loss

            l.backward()
            iterator.set_postfix(
                {"loss": l.item(), "loss_rel_change": loss_rel_change}
            )
            optimizer.step()

        if log_path is not None:
            performance.to_csv(log_path, index=False)

        final_weights_dense = torch.exp(weights).detach().numpy()
        compute_diagnostics(final_weights_dense, "Dense")
        return final_weights_dense

    else:
        # Sparse training
        weights = torch.tensor(
            np.log(original_weights), requires_grad=True, dtype=torch.float32
        )
        gates = HardConcrete(
            len(original_weights), init_mean=init_mean, temperature=temperature
        )

        optimizer = torch.optim.Adam(
            [weights] + list(gates.parameters()), lr=3e-1
        )
        from tqdm import trange

        start_loss = None
        iterator = trange(epochs)
        performance = pd.DataFrame()

        for i in iterator:
            optimizer.zero_grad()
            weights_ = dropout_weights(weights, dropout_rate)
            masked = torch.exp(weights_) * gates()
            l_main = loss(masked)
            l = l_main + l0_lambda * gates.get_penalty()

            if (log_path is not None) and (i % 10 == 0):
                gates.eval()
                with torch.no_grad():
                    estimates = (
                        (torch.exp(weights) * gates()) @ loss_matrix_tensor
                    ).numpy()
                gates.train()

                df = pd.DataFrame(
                    {
                        "target_name": target_names,
                        "estimate": estimates,
                        "target": targets_array_numpy,
                    }
                )
                df["epoch"] = i
                df["error"] = df.estimate - df.target
                df["rel_error"] = df.error / df.target
                df["abs_error"] = df.error.abs()
                df["rel_abs_error"] = df.rel_error.abs()
                df["loss"] = df.rel_abs_error**2
                performance = pd.concat([performance, df], ignore_index=True)

            if (log_path is not None) and (i % 1000 == 0):
                performance.to_csv(log_path, index=False)

            if start_loss is None:
                start_loss = l.item()
            loss_rel_change = (l.item() - start_loss) / start_loss

            l.backward()
            iterator.set_postfix(
                {"loss": l.item(), "loss_rel_change": loss_rel_change}
            )
            optimizer.step()

        if log_path is not None:
            performance.to_csv(log_path, index=False)

        gates.eval()
        final_weights_sparse = (torch.exp(weights) * gates()).detach().numpy()
        compute_diagnostics(final_weights_sparse, "Sparse")

        return final_weights_sparse


def train_previous_year_income_model():
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2019)

    sim.subsample(10_000)

    VARIABLES = [
        "previous_year_income_available",
        "employment_income",
        "self_employment_income",
        "age",
        "is_male",
        "spm_unit_state_fips",
        "dividend_income",
        "interest_income",
        "social_security",
        "capital_gains",
        "is_disabled",
        "is_blind",
        "is_married",
        "tax_unit_children",
        "pension_income",
    ]

    OUTPUTS = [
        "employment_income_last_year",
        "self_employment_income_last_year",
    ]

    df = sim.calculate_dataframe(VARIABLES + OUTPUTS, 2019, map_to="person")
    df_train = df[df.previous_year_income_available]

    from policyengine_us_data.utils import QRF

    income_last_year = QRF()
    X = df_train[VARIABLES[1:]]
    y = df_train[OUTPUTS]

    income_last_year.fit(X, y)

    return income_last_year


class EnhancedCPS(Dataset):
    data_format = Dataset.TIME_PERIOD_ARRAYS
    input_dataset: Type[Dataset]
    start_year: int
    end_year: int

    def generate(self):
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        data["household_weight"] = {}
        original_weights = sim.calculate("household_weight")
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )

        # Run the optimization procedure to get (close to) minimum loss weights
        for year in range(self.start_year, self.end_year + 1):
            loss_matrix, targets_array = build_loss_matrix(
                self.input_dataset, year
            )

            bad_mask = loss_matrix.columns.isin(bad_targets)
            keep_mask_bool = ~bad_mask
            keep_idx = np.where(keep_mask_bool)[0]
            loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
            targets_array_clean = targets_array[keep_idx]
            assert loss_matrix_clean.shape[1] == targets_array_clean.size

            optimised_weights = reweight(
                original_weights,
                loss_matrix_clean,
                targets_array_clean,
                log_path="calibration_log.csv",
                epochs=150,
            )
            data["household_weight"][year] = optimised_weights

        self.save_dataset(data)


class ReweightedCPS_2024(Dataset):
    data_format = Dataset.ARRAYS
    file_path = STORAGE_FOLDER / "reweighted_cps_2024.h5"
    name = "reweighted_cps_2024"
    label = "Reweighted CPS 2024"
    input_dataset = CPS_2024
    time_period = 2024

    def generate(self):
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        original_weights = sim.calculate("household_weight")
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )
        for year in [2024]:
            loss_matrix, targets_array = build_loss_matrix(
                self.input_dataset, year
            )
            optimised_weights = reweight(
                original_weights, loss_matrix, targets_array
            )
            data["household_weight"] = optimised_weights

        self.save_dataset(data)


class MinimizedEnhancedCPS_2024(EnhancedCPS):
    input_dataset = ExtendedCPS_2024
    start_year = 2024
    end_year = 2024
    name = "minimized_enhanced_cps_2024"
    label = "Minimized Enhanced CPS 2024"
    file_path = STORAGE_FOLDER / "minimized_enhanced_cps_2024.h5"
    url = (
        "hf://policyengine/policyengine-us-data/minimized_enhanced_cps_2024.h5"
    )

    def generate(self):
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        data["household_weight"] = {}
        original_weights = sim.calculate("household_weight")
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )

        # Run the optimization procedure to get (close to) minimum loss weights
        for year in range(self.start_year, self.end_year + 1):
            loss_matrix, targets_array = build_loss_matrix(
                self.input_dataset, year
            )

            bad_mask = loss_matrix.columns.isin(bad_targets)
            keep_mask_bool = ~bad_mask
            keep_idx = np.where(keep_mask_bool)[0]

            # Check if filtering would remove all columns
            if len(keep_idx) == 0:
                print(
                    "WARNING: bad_targets filtering would remove all columns, using all columns instead"
                )
                keep_idx = np.arange(loss_matrix.shape[1])
                targets_array_clean = targets_array
                loss_matrix_clean = loss_matrix
            else:
                loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
                targets_array_clean = targets_array[keep_idx]
            assert loss_matrix_clean.shape[1] == targets_array_clean.size

        from policyengine_us_data.utils.minimize import (
            candidate_loss_contribution,
            minimize_dataset,
        )

        minimize_dataset(
            self.input_dataset,
            self.file_path,
            minimization_function=candidate_loss_contribution,
            loss_matrix=loss_matrix_clean,
            targets=targets_array_clean,
            loss_rel_change_max=[0.1],  # maximum relative change in loss
            count_iterations=6,
            view_fraction_per_iteration=0.4,
            fraction_remove_per_iteration=0.1,
        )


class SparseEnhancedCPS_2024(EnhancedCPS):
    input_dataset = ExtendedCPS_2024
    start_year = 2024
    end_year = 2024
    name = "sparse_enhanced_cps_2024"
    label = "Sparse Enhanced CPS 2024"
    file_path = STORAGE_FOLDER / "sparse_enhanced_cps_2024.h5"
    url = "hf://policyengine/policyengine-us-data/sparse_enhanced_cps_2024.h5"

    def generate(self):
        from policyengine_us import Microsimulation
        from policyengine_us_data.utils.minimize import (
            create_calibration_log_file,
        )

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        data["household_weight"] = {}
        original_weights = sim.calculate("household_weight")
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )

        # Run the optimization procedure to get (close to) minimum loss weights
        for year in range(self.start_year, self.end_year + 1):
            loss_matrix, targets_array = build_loss_matrix(
                self.input_dataset, year
            )

            bad_mask = loss_matrix.columns.isin(bad_targets)
            keep_mask_bool = ~bad_mask
            keep_idx = np.where(keep_mask_bool)[0]
            loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
            targets_array_clean = targets_array[keep_idx]
            assert loss_matrix_clean.shape[1] == targets_array_clean.size

            optimised_weights = reweight(
                original_weights,
                loss_matrix_clean,
                targets_array_clean,
                log_path="calibration_log.csv",
                epochs=150,
                sparse=True,
            )
            data["household_weight"][year] = optimised_weights
            # Also save as sparse weights for small_enhanced_cps.py
            if "household_sparse_weight" not in data:
                data["household_sparse_weight"] = {}
            data["household_sparse_weight"][year] = optimised_weights

        self.save_dataset(data)

        create_calibration_log_file(self.file_path)


class EnhancedCPS_2024(EnhancedCPS):
    input_dataset = ExtendedCPS_2024
    start_year = 2024
    end_year = 2024
    name = "enhanced_cps_2024"
    label = "Enhanced CPS 2024"
    file_path = STORAGE_FOLDER / "enhanced_cps_2024.h5"
    url = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"


if __name__ == "__main__":
    EnhancedCPS_2024().generate()
    # MinimizedEnhancedCPS_2024().generate()
    SparseEnhancedCPS_2024().generate()
