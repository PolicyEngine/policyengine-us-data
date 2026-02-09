import pytest
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from policyengine_core.data import Dataset
from policyengine_us import Microsimulation
from policyengine_us_data.storage import (
    STORAGE_FOLDER,
    CALIBRATION_FOLDER,
)


@pytest.fixture(scope="session")
def data():
    return Dataset.from_file(STORAGE_FOLDER / "enhanced_cps_2024.h5")


@pytest.fixture(scope="session")
def sim(data):
    return Microsimulation(dataset=data)


def test_sparse_ecps_has_mortgage_interest(sim):
    assert sim.calculate("deductible_mortgage_interest").sum() > 1


def test_sparse_ecps_has_tips(sim):
    # Ensure we impute at least $40 billion in tip income.
    TIP_INCOME_MINIMUM = 40e9
    assert sim.calculate("tip_income").sum() > TIP_INCOME_MINIMUM


def test_sparse_ssn_card_type_none_target(sim):

    TARGET_COUNT = 13e6
    TOLERANCE = 0.2  # Allow 20% error

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
        "policyengine_us_data/storage/calibration_targets/"
        "aca_spending_and_enrollment_2024.csv"
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
        "policyengine_us_data/storage/calibration_targets/"
        "medicaid_enrollment_2024.csv"
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
