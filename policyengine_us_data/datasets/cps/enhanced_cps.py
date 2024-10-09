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

try:
    import torch
except ImportError:
    torch = None


def reweight(
    original_weights,
    loss_matrix,
    targets_array,
    dropout_rate=0.05,
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
        return rel_error.mean()

    def dropout_weights(weights, p):
        if p == 0:
            return weights
        # Replace p% of the weights with the mean value of the rest of them
        mask = torch.rand_like(weights) < p
        mean = weights[~mask].mean()
        masked_weights = weights.clone()
        masked_weights[mask] = mean
        return masked_weights

    optimizer = torch.optim.Adam([weights], lr=1e-1)
    from tqdm import trange

    start_loss = None

    iterator = trange(5_000)
    for i in iterator:
        optimizer.zero_grad()
        weights_ = dropout_weights(weights, dropout_rate)
        l = loss(torch.exp(weights_))
        if start_loss is None:
            start_loss = l.item()
        loss_rel_change = (l.item() - start_loss) / start_loss
        l.backward()
        iterator.set_postfix(
            {"loss": l.item(), "loss_rel_change": loss_rel_change}
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
    url = "release://policyengine/policyengine-us-data/1.11.0/enhanced_cps_2024.h5"


if __name__ == "__main__":
    EnhancedCPS_2024().generate()
