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
from policyengine_us_data.data_storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps import ExtendedCPS_2024
import torch


def reweight(
    original_weights,
    loss_matrix,
    targets_array,
):
    loss_matrix = torch.tensor(loss_matrix.values, dtype=torch.float32)
    targets_array = torch.tensor(targets_array, dtype=torch.float32)

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

    weights = torch.tensor(
        np.log(original_weights), requires_grad=True, dtype=torch.float32
    )
    optimizer = torch.optim.Adam([weights], lr=1e-2)
    from tqdm import trange

    iterator = trange(5_000)
    for i in iterator:
        optimizer.zero_grad()
        l = loss(torch.exp(weights))
        l.backward()
        iterator.set_postfix({"loss": l.item()})
        optimizer.step()

    return torch.exp(weights).detach().numpy()


class EnhancedCPS(Dataset):
    data_format = Dataset.FLAT_FILE
    input_dataset: Type[Dataset]
    start_year: int
    end_year: int

    def generate(self):
        df = self.input_dataset(require=True).load()
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.input_dataset)
        original_weights = sim.calculate("household_weight")
        original_weights = original_weights.values + np.random.normal(
            10, 1, len(original_weights)
        )
        for year in range(self.start_year, self.end_year + 1):
            print(f"Enhancing CPS for {year}")
            loss_matrix, targets_array = build_loss_matrix(
                self.input_dataset, year
            )
            optimised_weights = reweight(
                original_weights, loss_matrix, targets_array
            )
            df[f"household_weight__{year}"] = sim.map_result(
                optimised_weights, "household", "person"
            )

        self.save_dataset(df)


class EnhancedCPS_2024(EnhancedCPS):
    input_dataset = ExtendedCPS_2024
    start_year = 2024
    end_year = 2024
    name = "enhanced_cps_2024"
    label = "Enhanced CPS 2024"
    file_path = STORAGE_FOLDER / "enhanced_cps_2024.csv"


if __name__ == "__main__":
    EnhancedCPS_2024().generate()
