import pytest
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from policyengine_core.data import Dataset
from policyengine_core.reforms import Reform
from policyengine_us import Microsimulation
from policyengine_us_data.utils import (
    build_loss_matrix,
    print_reweighting_diagnostics,
)
from policyengine_us_data.storage import STORAGE_FOLDER, CALIBRATION_FOLDER


@pytest.fixture(scope="session")
def data():
    return Dataset.from_file(STORAGE_FOLDER / "sparse_enhanced_cps_2024.h5")


@pytest.fixture(scope="session")
def sim(data):
    return Microsimulation(dataset=data)


@pytest.mark.filterwarnings("ignore:DataFrame is highly fragmented")
@pytest.mark.filterwarnings("ignore:The distutils package is deprecated")
@pytest.mark.filterwarnings(
    "ignore:Series.__getitem__ treating keys as positions is deprecated"
)
@pytest.mark.filterwarnings(
    "ignore:Setting an item of incompatible dtype is deprecated"
)
@pytest.mark.filterwarnings(
    "ignore:Boolean Series key will be reindexed to match DataFrame index."
)
def test_sparse_ecps(sim):
    data = sim.dataset.load_dataset()
    optimised_weights = data["household_weight"]["2024"]

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

    loss_matrix, targets_array = build_loss_matrix(sim.dataset, 2024)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = loss_matrix.columns.isin(bad_targets)
    keep_mask_bool = ~(zero_mask | bad_mask)
    keep_idx = np.where(keep_mask_bool)[0]
    loss_matrix_clean = loss_matrix.iloc[:, keep_idx]
    targets_array_clean = targets_array[keep_idx]
    assert loss_matrix_clean.shape[1] == targets_array_clean.size

    percent_within_10 = print_reweighting_diagnostics(
        optimised_weights,
        loss_matrix_clean,
        targets_array_clean,
        "Sparse Solutions",
    )
    assert percent_within_10 > 60.0


def test_sparse_ecps_has_mortgage_interest(sim):
    assert sim.calculate("deductible_mortgage_interest").sum() > 1


def test_sparse_ecps_has_tips(sim):
    # Ensure we impute at least $40 billion in tip income.
    # We currently target $38 billion * 1.4 = $53.2 billion.
    TIP_INCOME_MINIMUM = 40e9
    assert sim.calculate("tip_income").sum() > TIP_INCOME_MINIMUM


def test_sparse_ecps_replicates_jct_tax_expenditures():
    calibration_log = pd.read_csv(
        "calibration_log_sparse.csv",
    )

    jct_rows = calibration_log[
        (calibration_log["target_name"].str.contains("jct/"))
        & (calibration_log["epoch"] == calibration_log["epoch"].max())
    ]

    assert (
        jct_rows.rel_abs_error.max() < 0.5
    ), "JCT tax expenditure targets not met (see the calibration log for details). Max relative error: {:.2%}".format(
        jct_rows.rel_abs_error.max()
    )


def deprecated_test_sparse_ecps_replicates_jct_tax_expenditures_full(sim):

    # JCT tax expenditure targets
    EXPENDITURE_TARGETS = {
        "salt_deduction": 21.247e9,
        "medical_expense_deduction": 11.4e9,
        "charitable_deduction": 65.301e9,
        "interest_deduction": 24.8e9,
    }

    baseline = sim
    income_tax_b = baseline.calculate(
        "income_tax", period=2024, map_to="household"
    )

    for deduction, target in EXPENDITURE_TARGETS.items():
        # Create reform that neutralizes the deduction
        class RepealDeduction(Reform):
            def apply(self):
                self.neutralize_variable(deduction)

        # Run reform simulation
        reformed = Microsimulation(reform=RepealDeduction, dataset=sim.dataset)
        income_tax_r = reformed.calculate(
            "income_tax", period=2024, map_to="household"
        )

        # Calculate tax expenditure
        tax_expenditure = (income_tax_r - income_tax_b).sum()
        pct_error = abs((tax_expenditure - target) / target)
        TOLERANCE = 0.4

        logging.info(
            f"{deduction} tax expenditure {tax_expenditure/1e9:.1f}bn "
            f"differs from target {target/1e9:.1f}bn by {pct_error:.2%}"
        )
        assert pct_error < TOLERANCE, deduction


def test_sparse_ssn_card_type_none_target(sim):

    TARGET_COUNT = 13e6
    TOLERANCE = 0.2  # Allow 20% error

    # Calculate the number of individuals with ssn_card_type == "NONE"
    ssn_type_none_mask = sim.calculate("ssn_card_type") == "NONE"
    count = ssn_type_none_mask.sum()

    pct_error = abs((count - TARGET_COUNT) / TARGET_COUNT)

    logging.info(
        f'SSN card type "NONE" count: {count:.0f}, '
        f"target: {TARGET_COUNT:.0f}, error: {pct_error:.2%}"
    )
    assert pct_error < TOLERANCE


def test_sparse_aca_calibration(sim):

    TARGETS_PATH = Path(
        "policyengine_us_data/storage/calibration_targets/aca_spending_and_enrollment_2024.csv"
    )
    targets = pd.read_csv(TARGETS_PATH)
    # Monthly to yearly
    targets["spending"] = targets["spending"] * 12
    # Adjust to match national target
    targets["spending"] = targets["spending"] * (
        98e9 / targets["spending"].sum()
    )

    state_code_hh = sim.calculate("state_code", map_to="household").values
    aca_ptc = sim.calculate("aca_ptc", map_to="household", period=2025)

    TOLERANCE = 1.0
    failed = False
    for _, row in targets.iterrows():
        state = row["state"]
        target_spending = row["spending"]
        simulated = aca_ptc[state_code_hh == state].sum()

        pct_error = abs(simulated - target_spending) / target_spending
        logging.info(
            f"{state}: simulated ${simulated/1e9:.2f} bn  "
            f"target ${target_spending/1e9:.2f} bn  "
            f"error {pct_error:.2%}"
        )

        if pct_error > TOLERANCE:
            failed = True

    assert (
        not failed
    ), f"One or more states exceeded tolerance of {TOLERANCE:.0%}."


def test_sparse_medicaid_calibration(sim):

    TARGETS_PATH = Path(
        "policyengine_us_data/storage/calibration_targets/medicaid_enrollment_2024.csv"
    )
    targets = pd.read_csv(TARGETS_PATH)

    state_code_hh = sim.calculate("state_code", map_to="household").values
    medicaid_enrolled = sim.calculate(
        "medicaid_enrolled", map_to="household", period=2025
    )

    TOLERANCE = 1.0
    failed = False
    for _, row in targets.iterrows():
        state = row["state"]
        target_enrollment = row["enrollment"]
        simulated = medicaid_enrolled[state_code_hh == state].sum()

        pct_error = abs(simulated - target_enrollment) / target_enrollment
        logging.info(
            f"{state}: simulated ${simulated/1e9:.2f} bn  "
            f"target ${target_enrollment/1e9:.2f} bn  "
            f"error {pct_error:.2%}"
        )

        if pct_error > TOLERANCE:
            failed = True

    assert (
        not failed
    ), f"One or more states exceeded tolerance of {TOLERANCE:.0%}."
