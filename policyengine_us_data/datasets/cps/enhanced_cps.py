from policyengine_core.data import Dataset
import pandas as pd
from policyengine_us_data.utils import (
    build_loss_matrix,
    HardConcrete,
    print_reweighting_diagnostics,
    set_seeds,
)
import numpy as np
from tqdm import trange
from typing import Type
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.extended_cps import (
    ExtendedCPS_2024,
    CPS_2019,
    CPS_2024,
)
from pathlib import Path
import logging


try:
    import torch
except ImportError:
    torch = None


def reweight(
    original_weights,
    loss_matrix,
    targets_array,
    dropout_rate=0.05,
    log_path="calibration_log.csv",
    epochs=500,
    l0_lambda=5.000015625e-07,  # L0 penalty to induce sparsity
    init_mean=0.999,  # initial proportion with non-zero weights
    temperature=0.25,
    seed=1456,
):
    set_seeds(seed)
    target_names = np.array(loss_matrix.columns)
    is_national = loss_matrix.columns.str.startswith("nation/")
    loss_matrix = torch.tensor(loss_matrix.values, dtype=torch.float32)
    nation_normalisation_factor = is_national * (1 / is_national.sum())
    state_normalisation_factor = ~is_national * (1 / (~is_national).sum())
    normalisation_factor = np.where(
        is_national, nation_normalisation_factor, state_normalisation_factor
    )
    normalisation_factor = torch.tensor(
        normalisation_factor, dtype=torch.float32
    )
    targets_array = torch.tensor(targets_array, dtype=torch.float32)
    weights = torch.tensor(
        np.log(original_weights), requires_grad=True, dtype=torch.float32
    )

    inv_mean_normalisation = 1 / np.mean(normalisation_factor.numpy())

    def loss(weights):
        # Check for Nans in either the weights or the loss matrix
        if torch.isnan(weights).any():
            raise ValueError("Weights contain NaNs")
        if torch.isnan(loss_matrix).any():
            raise ValueError("Loss matrix contains NaNs")
        estimate = weights @ loss_matrix
        if torch.isnan(estimate).any():
            raise ValueError("Estimate contains NaNs")
        rel_error = (
            ((estimate - targets_array) + 1) / (targets_array + 1)
        ) ** 2
        rel_error_normalized = (
            inv_mean_normalisation * rel_error * normalisation_factor
        )
        if torch.isnan(rel_error_normalized).any():
            raise ValueError("Relative error contains NaNs")
        return rel_error_normalized.mean()

    def dropout_weights(weights, p):
        if p == 0:
            return weights
        # Replace p% of the weights with the mean value of the rest of them
        mask = torch.rand_like(weights) < p
        mean = weights[~mask].mean()
        masked_weights = weights.clone()
        masked_weights[mask] = mean
        return masked_weights

    # Original (Dense) path ---
    optimizer = torch.optim.Adam([weights], lr=3e-1)
    start_loss = None

    iterator = trange(epochs)
    performance = pd.DataFrame()
    for i in iterator:
        optimizer.zero_grad()
        weights_ = dropout_weights(weights, dropout_rate)
        l = loss(torch.exp(weights_))
        if (log_path is not None) and (i % 10 == 0):
            estimates = torch.exp(weights) @ loss_matrix
            estimates = estimates.detach().numpy()
            df = pd.DataFrame(
                {
                    "target_name": target_names,
                    "estimate": estimates,
                    "target": targets_array.detach().numpy(),
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

    optimised_weights = final_weights_dense
    print_reweighting_diagnostics(
        final_weights_dense,
        loss_matrix,
        targets_array,
        "Dense Solutions",
    )

    # New (Sparse) path depending on temperature, init_mean, l0_lambda -----
    logging.info(
        f"Sparse optimization using seed {seed}, temp {temperature} "
        + f"init_mean {init_mean}, l0_lambda {l0_lambda}"
    )
    set_seeds(seed)
    p = Path(log_path)
    log_path_sparse = p.with_name(f"{p.stem}_sparse{p.suffix}")

    weights = torch.tensor(
        np.log(original_weights), requires_grad=True, dtype=torch.float32
    )
    gates = HardConcrete(
        len(original_weights), init_mean=init_mean, temperature=temperature
    )
    # NOTE: Results are pretty sensitve to learning rates
    # optimizer breaks down somewhere near .005, does better at above .1
    optimizer = torch.optim.Adam([weights] + list(gates.parameters()), lr=0.2)
    start_loss = None

    iterator = trange(epochs * 2)  # lower learning rate, harder optimization
    performance = pd.DataFrame()
    for i in iterator:
        optimizer.zero_grad()
        masked = torch.exp(weights) * gates()
        l_main = loss(masked)
        l = l_main + l0_lambda * gates.get_penalty()
        if (log_path_sparse is not None) and (i % 10 == 0):
            gates.eval()
            estimates = (torch.exp(weights) * gates()) @ loss_matrix
            gates.train()
            estimates = estimates.detach().numpy()
            df = pd.DataFrame(
                {
                    "target_name": target_names,
                    "estimate": estimates,
                    "target": targets_array.detach().numpy(),
                }
            )
            df["epoch"] = i
            df["error"] = df.estimate - df.target
            df["rel_error"] = df.error / df.target
            df["abs_error"] = df.error.abs()
            df["rel_abs_error"] = df.rel_error.abs()
            df["loss"] = df.rel_abs_error**2
            performance = pd.concat([performance, df], ignore_index=True)

        if (log_path_sparse is not None) and (i % 1000 == 0):
            performance.to_csv(log_path_sparse, index=False)
        if start_loss is None:
            start_loss = l.item()
        loss_rel_change = (l.item() - start_loss) / start_loss
        l.backward()
        iterator.set_postfix(
            {"loss": l.item(), "loss_rel_change": loss_rel_change}
        )
        optimizer.step()
        if log_path_sparse is not None:
            performance.to_csv(log_path_sparse, index=False)

    gates.eval()
    final_weights_sparse = (torch.exp(weights) * gates()).detach().numpy()

    print_reweighting_diagnostics(
        final_weights_sparse,
        loss_matrix,
        targets_array,
        "L0 Sparse Solution",
    )

    # Log household count for CI monitoring
    nonzero_count = np.sum(final_weights_sparse > 0.01)
    logging.info(
        f"HOUSEHOLD_COUNT_CHECK: {nonzero_count} non-zero households (target: 20k-25k)"
    )

    return final_weights_dense, final_weights_sparse


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
        data["household_sparse_weight"] = {}
        original_weights = sim.calculate("household_weight")
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )

        bad_targets = [
            "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Head of Household",
            "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Head of Household",
            "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
            "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
            "nation/irs/count/count/AGI in 10k-15k/taxable/Head of Household",
            "nation/irs/count/count/AGI in 15k-20k/taxable/Head of Household",
            "nation/irs/count/count/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
            "nation/irs/count/count/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
            "state/RI/adjusted_gross_income/amount/-inf_1",
            "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Head of Household",
            "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Head of Household",
            "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
            "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
            "nation/irs/count/count/AGI in 10k-15k/taxable/Head of Household",
            "nation/irs/count/count/AGI in 15k-20k/taxable/Head of Household",
            "nation/irs/count/count/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
            "nation/irs/count/count/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
            "state/RI/adjusted_gross_income/amount/-inf_1",
            "nation/irs/exempt interest/count/AGI in -inf-inf/taxable/All",
        ]

        # Run the optimization procedure to get (close to) minimum loss weights
        for year in range(self.start_year, self.end_year + 1):
            loss_matrix, targets_array = build_loss_matrix(
                self.input_dataset, year
            )
            zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
            bad_mask = loss_matrix.columns.isin(bad_targets)
            keep_mask_bool = ~(zero_mask | bad_mask)
            keep_idx = np.where(keep_mask_bool)[0]
            loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
            targets_array_clean = targets_array[keep_idx]
            assert loss_matrix_clean.shape[1] == targets_array_clean.size

            optimised_weights_dense, optimised_weights_sparse = reweight(
                original_weights,
                loss_matrix_clean,
                targets_array_clean,
                log_path="calibration_log.csv",
                epochs=200,
                seed=1456,
            )
            data["household_weight"][year] = optimised_weights_dense
            data["household_sparse_weight"][year] = optimised_weights_sparse

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
