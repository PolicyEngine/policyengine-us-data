import torch
from policyengine_us import Microsimulation
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import os
import argparse

# Fill in missing constituencies with average column values
import pandas as pd
import numpy as np

#from policyengine_us_data.datasets.frs.local_areas.constituencies.loss import (
#    create_constituency_target_matrix,
#    create_national_target_matrix,
#)
#from policyengine_uk_data.datasets.frs.local_areas.constituencies.boundary_changes.mapping_matrix import (
#    mapping_matrix,
#)
from pathlib import Path
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets import CPS_2022

FOLDER = Path(__file__).parent




def calibrate(
    epochs: int = 128,
    excluded_training_targets=[],
    log_csv="training_log.csv",
    overwrite_efrs=True,
):
    matrix_, y_, country_mask = create_district_target_matrix(
        CPS_2022, 2025
    )

    sim = Microsimulation(CPS_2022)

    COUNT_CONSTITUENCIES = 650

    # Weights - 650 x 100180
    original_weights = np.log(
        sim.calculate("household_weight", 2025).values / COUNT_CONSTITUENCIES
    )
    weights = torch.tensor(
        np.ones((COUNT_CONSTITUENCIES, len(original_weights)))
        * original_weights,
        dtype=torch.float32,
        requires_grad=True,
    )
    validation_targets_c = matrix_.columns.isin(excluded_training_targets)
    validation_targets_n = m_national_.columns.isin(excluded_training_targets)
    if len(excluded_training_targets) > 0:
        dropout_targets = True
    else:
        dropout_targets = False

    metrics = torch.tensor(matrix_.values, dtype=torch.float32)
    y = torch.tensor(y_.values, dtype=torch.float32)
    matrix_national = torch.tensor(m_national_.values, dtype=torch.float32)
    y_national = torch.tensor(y_national_.values, dtype=torch.float32)
    r = torch.tensor(country_mask, dtype=torch.float32)

    def loss(w, validation: bool = False):
        pred_c = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        if dropout_targets:
            if validation:
                mask = validation_targets_c
            else:
                mask = ~validation_targets_c
            pred_c = pred_c[:, mask]
            mse_c = torch.mean((pred_c / (1 + y[:, mask]) - 1) ** 2)
        else:
            mse_c = torch.mean((pred_c / (1 + y) - 1) ** 2)

        pred_n = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
        if dropout_targets:
            if validation:
                mask = validation_targets_n
            else:
                mask = ~validation_targets_n
            pred_n = pred_n[mask]
            mse_n = torch.mean((pred_n / (1 + y_national[mask]) - 1) ** 2)
        else:
            mse_n = torch.mean((pred_n / (1 + y_national) - 1) ** 2)

        return mse_c + mse_n

    def pct_close(w, t=0.1, constituency=True, national=True):
        # Return the percentage of metrics that are within t% of the target
        numerator = 0
        denominator = 0
        pred_c = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        e_c = torch.sum(torch.abs((pred_c / (1 + y) - 1)) < t).item()
        c_c = pred_c.shape[0] * pred_c.shape[1]

        if constituency:
            numerator += e_c
            denominator += c_c

        pred_n = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
        e_n = torch.sum(torch.abs((pred_n / (1 + y_national) - 1)) < t).item()
        c_n = pred_n.shape[0]

        if national:
            numerator += e_n
            denominator += c_n

        return numerator / denominator

    def dropout_weights(weights, p):
        if p == 0:
            return weights
        # Replace p% of the weights with the mean value of the rest of them
        mask = torch.rand_like(weights) < p
        mean = weights[~mask].mean()
        masked_weights = weights.clone()
        masked_weights[mask] = mean
        return masked_weights

    optimizer = torch.optim.Adam([weights], lr=0.15)

    desc = range(32) if os.environ.get("DATA_LITE") else range(epochs)
    final_weights = (torch.exp(weights) * r).detach().numpy()
    performance = pd.DataFrame()

    for epoch in desc:
        optimizer.zero_grad()
        weights_ = torch.exp(dropout_weights(weights, 0.05)) * r
        l = loss(weights_)
        l.backward()
        optimizer.step()
        c_close = pct_close(weights_, constituency=True, national=False)
        n_close = pct_close(weights_, constituency=False, national=True)
        if epoch % 1 == 0:
            if dropout_targets:
                validation_loss = loss(weights_, validation=True)
                print(
                    f"Training loss: {l.item():,.3f}, Validation loss: {validation_loss.item():,.3f}, Epoch: {epoch}, "
                    f"Constituency<10%: {c_close:.1%}, National<10%: {n_close:.1%}"
                )
            else:
                print(
                    f"Loss: {l.item()}, Epoch: {epoch}, Constituency<10%: {c_close:.1%}, National<10%: {n_close:.1%}"
                )
        if epoch % 10 == 0:
            final_weights = (torch.exp(weights) * r).detach().numpy()

            performance_step = get_performance(
                final_weights,
                matrix_,
                y_,
                m_national_,
                y_national_,
                excluded_training_targets,
            )
            performance_step["epoch"] = epoch
            performance = pd.concat(
                [performance, performance_step], ignore_index=True
            )

            if log_csv:
                performance.to_csv(log_csv, index=False)

            with h5py.File(
                STORAGE_FOLDER / "parliamentary_constituency_weights.h5", "w"
            ) as f:
                f.create_dataset("2025", data=final_weights)

            if overwrite_efrs:
                with h5py.File(
                    STORAGE_FOLDER / "enhanced_frs_2022_23.h5", "r+"
                ) as f:
                    if "household_weight/2025" in f:
                        del f["household_weight/2025"]
                    f.create_dataset(
                        "household_weight/2025", data=final_weights.sum(axis=0)
                    )

    return final_weights


def get_performance(weights, m_c, y_c, m_n, y_n, excluded_targets):
    constituency_target_matrix, constituency_actuals = m_c, y_c
    national_target_matrix, national_actuals = m_n, y_n
    constituencies = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")
    constituency_wide = weights @ constituency_target_matrix
    constituency_wide.index = constituencies.code.values
    constituency_wide["name"] = constituencies.name.values

    constituency_results = pd.melt(
        constituency_wide.reset_index(),
        id_vars=["index", "name"],
        var_name="variable",
        value_name="value",
    )

    constituency_actuals.index = constituencies.code.values
    constituency_actuals["name"] = constituencies.name.values
    constituency_actuals_long = pd.melt(
        constituency_actuals.reset_index(),
        id_vars=["index", "name"],
        var_name="variable",
        value_name="value",
    )

    constituency_target_validation = pd.merge(
        constituency_results,
        constituency_actuals_long,
        on=["index", "variable"],
        suffixes=("_target", "_actual"),
    )
    constituency_target_validation.drop("name_actual", axis=1, inplace=True)
    constituency_target_validation.columns = [
        "index",
        "name",
        "metric",
        "estimate",
        "target",
    ]

    constituency_target_validation["error"] = (
        constituency_target_validation["estimate"]
        - constituency_target_validation["target"]
    )
    constituency_target_validation["abs_error"] = (
        constituency_target_validation["error"].abs()
    )
    constituency_target_validation["rel_abs_error"] = (
        constituency_target_validation["abs_error"]
        / constituency_target_validation["target"]
    )

    national_performance = weights.sum(axis=0) @ national_target_matrix
    national_target_validation = pd.DataFrame(
        {
            "metric": national_performance.index,
            "estimate": national_performance.values,
        }
    )
    national_target_validation["target"] = national_actuals.values

    national_target_validation["error"] = (
        national_target_validation["estimate"]
        - national_target_validation["target"]
    )
    national_target_validation["abs_error"] = national_target_validation[
        "error"
    ].abs()
    national_target_validation["rel_abs_error"] = (
        national_target_validation["abs_error"]
        / national_target_validation["target"]
    )

    df = pd.concat(
        [
            constituency_target_validation,
            national_target_validation.assign(name="UK", index=0),
        ]
    ).reset_index(drop=True)

    df["validation"] = df.metric.isin(excluded_targets)

    return df


if __name__ == "__main__":
    calibrate()
