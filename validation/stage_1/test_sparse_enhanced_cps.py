"""Integration tests for Sparse Enhanced CPS dataset (requires enhanced_cps_2024.h5)."""

import pytest
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from policyengine_core.data import Dataset
from policyengine_core.reforms import Reform
from policyengine_us import Microsimulation
from policyengine_us_data.utils import ABSOLUTE_ERROR_SCALE_TARGETS
from policyengine_us_data.storage import STORAGE_FOLDER


def _period_array(period_values, period):
    return period_values.get(period, period_values[str(period)])


def _require_identification_fields(data):
    required_fields = ("has_tin", "has_itin", "has_valid_ssn", "taxpayer_id_type")
    missing = [field for field in required_fields if field not in data]
    if missing:
        pytest.skip(
            "enhanced_cps_2024.h5 fixture predates raw identification fields: "
            + ", ".join(missing)
        )


@pytest.fixture(scope="session")
def data():
    return Dataset.from_file(STORAGE_FOLDER / "enhanced_cps_2024.h5")


@pytest.fixture(scope="session")
def sim(data):
    return Microsimulation(dataset=data)


@pytest.fixture(scope="module")
def sparse_sim():
    path = STORAGE_FOLDER / "sparse_enhanced_cps_2024.h5"
    if not path.exists():
        pytest.skip("sparse_enhanced_cps_2024.h5 not found")
    return Microsimulation(dataset=Dataset.from_file(path))


# ── Sparse dataset sanity checks ──────────────────────────────


def test_sparse_household_count(sparse_sim):
    total_hh = sparse_sim.calculate("household_weight").values.sum()
    assert 100e6 < total_hh < 200e6, (
        f"Sparse total households = {total_hh:.2e}, expected 100M-200M."
    )


def test_sparse_poverty_rate_reasonable(sparse_sim):
    in_poverty = sparse_sim.calculate("person_in_poverty", map_to="person")
    rate = in_poverty.mean()
    assert 0.05 < rate < 0.35, f"Sparse poverty rate = {rate:.1%}, expected 5-35%."


# ── Reweighting and calibration checks ────────────────────────


def test_sparse_ecps():
    calibration_log = pd.read_csv("calibration_log.csv")
    final_epoch = calibration_log["epoch"].max()
    final_rows = calibration_log[calibration_log["epoch"] == final_epoch].copy()

    assert not final_rows.empty, "No final-epoch calibration diagnostics found."

    tolerance = 0.10 * final_rows["target"].abs()
    for target_name, scale in ABSOLUTE_ERROR_SCALE_TARGETS.items():
        tolerance.loc[final_rows["target_name"] == target_name] = 0.10 * scale

    percent_within_10 = (final_rows["abs_error"] <= tolerance).mean() * 100
    assert percent_within_10 > 60.0


def test_sparse_ecps_employment_income_positive(sim):
    """Direct check that employment income is in the trillions.

    Unlike test_sparse_ecps which filters out zero targets via zero_mask,
    this test would have caught the bug where employment_income_before_lsr
    was dropped, zeroing out all employment income.
    """
    total = sim.calculate("employment_income").sum()
    assert total > 5e12, (
        f"employment_income sum is {total:.2e}, expected > 5T. "
        "Likely missing employment_income_before_lsr in dataset."
    )


def test_sparse_ecps_has_mortgage_interest(sim):
    assert sim.calculate("deductible_mortgage_interest").sum() > 1


def test_sparse_ecps_has_tips(sim):
    # Ensure we impute at least $40 billion in tip income.
    # We currently target $38 billion * 1.4 = $53.2 billion.
    TIP_INCOME_MINIMUM = 40e9
    assert sim.calculate("tip_income").sum() > TIP_INCOME_MINIMUM


def test_sparse_ecps_replicates_jct_tax_expenditures():
    from validation.stage_1.jct_calibration import (
        assert_no_unexpected_high_error_jct_diagnostics,
    )

    calibration_log = pd.read_csv(
        "calibration_log.csv",
    )

    assert_no_unexpected_high_error_jct_diagnostics(calibration_log)


def deprecated_test_sparse_ecps_replicates_jct_tax_expenditures_full(sim):
    # JCT tax expenditure targets
    EXPENDITURE_TARGETS = {
        "salt_deduction": 21.247e9,
        "medical_expense_deduction": 11.4e9,
        "charitable_deduction": 65.301e9,
        "interest_deduction": 24.8e9,
    }

    baseline = sim
    income_tax_b = baseline.calculate("income_tax", period=2024, map_to="household")

    for deduction, target in EXPENDITURE_TARGETS.items():
        # Create reform that neutralizes the deduction
        class RepealDeduction(Reform):
            def apply(self):
                self.neutralize_variable(deduction)

        # Run reform simulation
        reformed = Microsimulation(reform=RepealDeduction, dataset=sim.dataset)
        income_tax_r = reformed.calculate("income_tax", period=2024, map_to="household")

        # Calculate tax expenditure
        tax_expenditure = (income_tax_r - income_tax_b).sum()
        pct_error = abs((tax_expenditure - target) / target)
        TOLERANCE = 0.4

        logging.info(
            f"{deduction} tax expenditure {tax_expenditure / 1e9:.1f}bn "
            f"differs from target {target / 1e9:.1f}bn by {pct_error:.2%}"
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


def test_sparse_has_tin_matches_identification_inputs(sim):
    data = sim.dataset.load_dataset()
    _require_identification_fields(data)
    has_tin = _period_array(data["has_tin"], 2024)
    has_itin = _period_array(data["has_itin"], 2024)
    has_valid_ssn = _period_array(data["has_valid_ssn"], 2024)
    ssn_card_type = _period_array(data["ssn_card_type"], 2024).astype(str)
    taxpayer_id_type = _period_array(data["taxpayer_id_type"], 2024).astype(str)

    np.testing.assert_array_equal(has_itin, has_tin)
    np.testing.assert_array_equal(has_valid_ssn, taxpayer_id_type == "VALID_SSN")
    np.testing.assert_array_equal(has_tin, taxpayer_id_type != "NONE")
    assert np.all(has_tin[has_valid_ssn])
    np.testing.assert_array_equal(has_valid_ssn[ssn_card_type == "NONE"], False)
    np.testing.assert_array_equal(
        taxpayer_id_type,
        np.where(
            has_valid_ssn,
            "VALID_SSN",
            np.where(has_tin, "OTHER_TIN", "NONE"),
        ),
    )


def test_sparse_aca_calibration(sim):
    TARGETS_PATH = Path(
        "policyengine_us_data/storage/calibration_targets/aca_spending_and_enrollment_2024.csv"
    )
    targets = pd.read_csv(TARGETS_PATH)
    # Monthly to yearly
    targets["spending"] = targets["spending"] * 12
    # Adjust to match national target
    targets["spending"] = targets["spending"] * (98e9 / targets["spending"].sum())

    state_code_hh = sim.calculate("state_code", map_to="household").values
    aca_ptc = sim.calculate("aca_ptc", map_to="household", period=2025)

    # See test_aca_calibration in test_enhanced_cps.py for the full
    # CMS-vs-IRS concept mismatch rationale; tracked in issue #805.
    TOLERANCE = 10.0
    failed = False
    for _, row in targets.iterrows():
        state = row["state"]
        target_spending = row["spending"]
        simulated = aca_ptc[state_code_hh == state].sum()

        pct_error = abs(simulated - target_spending) / target_spending
        logging.info(
            f"{state}: simulated ${simulated / 1e9:.2f} bn  "
            f"target ${target_spending / 1e9:.2f} bn  "
            f"error {pct_error:.2%}"
        )

        if pct_error > TOLERANCE:
            failed = True

    assert not failed, f"One or more states exceeded tolerance of {TOLERANCE:.0%}."


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
            f"{state}: simulated ${simulated / 1e9:.2f} bn  "
            f"target ${target_enrollment / 1e9:.2f} bn  "
            f"error {pct_error:.2%}"
        )

        if pct_error > TOLERANCE:
            failed = True

    assert not failed, f"One or more states exceeded tolerance of {TOLERANCE:.0%}."
