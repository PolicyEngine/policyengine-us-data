from policyengine_core.data import Dataset
import pandas as pd
from policyengine_us_data.utils import (
    pe_to_soi,
    get_soi,
    build_loss_matrix,
    fmt,
)
import numpy as np
from typing import Type
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.extended_cps import (
    ExtendedCPS_2024,
    CPS_2019,
    CPS_2024,
)
import torch
import os


def reweight(
    original_weights,
    loss_matrix,
    targets_array,
):
    target_names = np.array(loss_matrix.columns)
    loss_matrix = torch.tensor(loss_matrix.values, dtype=torch.float32)
    targets_array = torch.tensor(targets_array, dtype=torch.float32)
    weights = torch.tensor(
        np.log(original_weights), requires_grad=True, dtype=torch.float32
    )

    # TODO: replace this with a call to the python reweight.py package.
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
        if torch.isnan(rel_error).any():
            raise ValueError("Relative error contains NaNs")
        worst_name = target_names[torch.argmax(rel_error)]
        worst_val = rel_error[torch.argmax(rel_error)].item()
        return rel_error.mean(), worst_name, worst_val

    optimizer = torch.optim.Adam([weights], lr=1e-2)
    from tqdm import trange

    iterator = (
        trange(10_000) if not os.environ.get("TEST_LITE") else trange(100)
    )
    for i in iterator:
        optimizer.zero_grad()
        l, worst_name, worst_val = loss(torch.exp(weights))
        l.backward()
        iterator.set_postfix(
            {"loss": l.item(), "worst": worst_name, "val": worst_val}
        )
        optimizer.step()

    return torch.exp(weights).detach().numpy()


def train_previous_year_income_model():
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2019)

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

    from survey_enhance import Imputation

    income_last_year = Imputation()
    X = df_train[VARIABLES[1:]]
    y = df_train[OUTPUTS]

    income_last_year.train(X, y)

    return income_last_year


class EnhancedCPS(Dataset):
    data_format = Dataset.TIME_PERIOD_ARRAYS
    input_dataset: Type[Dataset]
    start_year: int
    end_year: int
    url = "release://policyengine/policyengine-us-data/release/enhanced_cps_2024.h5"

    def generate(self):
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.input_dataset)
        data = sim.dataset.load_dataset()
        data["household_weight"] = {}
        original_weights = sim.calculate("household_weight")
        original_weights = original_weights.values + np.random.normal(
            1, 0.1, len(original_weights)
        )
        for year in range(self.start_year, self.end_year + 1):
            loss_matrix, targets_array = build_loss_matrix(
                self.input_dataset, year
            )
            optimised_weights = reweight(
                original_weights, loss_matrix, targets_array
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


class EnhancedCPS_2024(EnhancedCPS):
    input_dataset = ExtendedCPS_2024
    start_year = 2024
    end_year = 2024
    name = "enhanced_cps_2024"
    label = "Enhanced CPS 2024"
    file_path = STORAGE_FOLDER / "enhanced_cps_2024.h5"


if __name__ == "__main__":
    EnhancedCPS_2024().generate()
