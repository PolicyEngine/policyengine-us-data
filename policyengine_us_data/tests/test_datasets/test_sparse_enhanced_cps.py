import pytest

import numpy as np

from policyengine_us_data.utils import (
    build_loss_matrix,
    print_reweighting_diagnostics,
)


def test_sparse_ecps():
    from policyengine_core.data import Dataset
    from policyengine_us_data.storage import STORAGE_FOLDER
    from policyengine_us import Microsimulation

    # NOTE: replace with "small_enhanced_cps_2024.h5 to see the difference!
    sim = Microsimulation(
        dataset=Dataset.from_file(
            STORAGE_FOLDER / f"sparse_enhanced_cps_2024.h5",
        )
    )

    data = sim.dataset.load_dataset()
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

    year = 2024
    loss_matrix, targets_array = build_loss_matrix(sim.dataset, year)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = loss_matrix.columns.isin(bad_targets)
    keep_mask_bool = ~(zero_mask | bad_mask)
    keep_idx = np.where(keep_mask_bool)[0]
    loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
    targets_array_clean = targets_array[keep_idx]
    assert loss_matrix_clean.shape[1] == targets_array_clean.size

    optimised_weights = data["household_weight"]["2024"]
    percent_within_10 = print_reweighting_diagnostics(
        optimised_weights,
        loss_matrix_clean,
        targets_array_clean,
        "Sparse Solutions",
    )

    assert percent_within_10 > 70.0


if __name__ == "main":
    test_sparse_ecps()
