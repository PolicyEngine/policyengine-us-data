import torch
from policyengine_us import Microsimulation
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import os
import argparse

import pandas as pd
import numpy as np
from pathlib import Path

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data import DATASETS, CPS_2022

FOLDER = Path(__file__).parent

if False:  # Interactive use
    %cd /mnt/c/devl/policyengine-us-data/policyengine_us_data/datasets/cps/local_areas/districts
    FOLDER = Path('.')


# TODO: why can't I just call on the string dataset name in my functions?
from policyengine_us_data.datasets.cps.cps import CPS_2022


def create_district_target_matrix(
    dataset: str = "CPS_2022",
    time_period: int = 2025,
    reform=None,
):
    ages = pd.read_csv(FOLDER / "targets" / "age.csv")

    ages_count_matrix = ages.iloc[:, 4:]
    age_ranges = list(ages_count_matrix.columns)

    sim = Microsimulation(dataset=CPS_2022)  # TODO: took back up to argument, and why isn't a string working?
    sim.default_calculation_period = time_period

    matrix = pd.DataFrame()
    y = pd.DataFrame()

    age = sim.calculate("age").values
    for age_range in age_ranges:
        if age_range != "85+":
            lower_age, upper_age = age_range.split("-")
            in_age_band = (age >= int(lower_age)) & (age < int(upper_age))
        else:
            in_age_band = age >= 85 

        # Mapping people ages to household ages, how exactly
        matrix[f"age/{age_range}"] = sim.map_result(
            in_age_band, "person", "household"
        )

        y[f"age/{age_range}"] = ages[age_range]

    # TODO: What type of Masks will I need?
    #district_mask = create_district_mask(
    #    household_districts= ?? sim.calculate("??").values,
    #    codes=ages.iloc[:, 1]
    #)

    return matrix, y #, district_mask


# TODO bring this into loss
def calibrate(
    epochs: int = 128,
    excluded_training_targets=[],
    log_csv="training_log.csv",
    overwrite_efrs=True,
):
    matrix_, y_ = create_district_target_matrix(
        CPS_2022, 2025
    )

    sim = Microsimulation(dataset = CPS_2022)

    COUNT_DISTRICTS = 435 

    original_weights = np.log(
        sim.calculate("household_weight", 2025).values / COUNT_DISTRICTS
    )
    weights = torch.tensor(
        np.ones((COUNT_DISTRICTS, len(original_weights)))
        * original_weights,
        dtype=torch.float32,
        requires_grad=True,
    )

    metrics = torch.tensor(matrix_.values, dtype=torch.float32)
    y = torch.tensor(y_.values, dtype=torch.float32)
    # r = torch.tensor(country_mask, dtype=torch.float32)

    def loss(w): #, validation: bool = False):
        pred = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        mse = torch.mean((pred / (1 + y) - 1) ** 2)
        return mse

    optimizer = torch.optim.Adam([weights], lr=0.15)

    desc = range(32) if os.environ.get("DATA_LITE") else range(epochs)
    #final_weights = (torch.exp(weights) * r).detach().numpy()
    final_weights = (torch.exp(weights)).detach().numpy()

    for epoch in desc:
        optimizer.zero_grad()
        #weights_ = torch.exp(dropout_weights(weights, 0.05)) * r
        #l = loss(weights_)
        l = loss(weights)
        l.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Loss: {l.item()}, Epoch: {epoch}")
        if epoch % 10 == 0:
            #final_weights = (torch.exp(weights) * r).detach().numpy()
            final_weights = torch.exp(weights).detach().numpy()  # what's with the exp?

            with h5py.File(
                STORAGE_FOLDER / "congressional_district_weights.h5", "w"
            ) as f:
                f.create_dataset("2025", data=final_weights)

            #if overwrite_ecps:
            #    with h5py.File(
            #        STORAGE_FOLDER / "enhanced_ecps_2022_23.h5", "r+"
            #    ) as f:
            #        if "household_weight/2025" in f:
            #            del f["household_weight/2025"]
            #        f.create_dataset(
            #            "household_weight/2025", data=final_weights.sum(axis=0)
            #        )

    return final_weights


if __name__ == "__main__":
    calibrate()
