"""Integration tests for Enhanced CPS dataset (requires enhanced_cps_2024.h5)."""

import numpy as np
import pytest


def _period_array(period_values, period):
    return period_values.get(period, period_values[str(period)])


@pytest.fixture(scope="module")
def ecps_sim():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    return Microsimulation(dataset=EnhancedCPS_2024)


# ── Sanity checks ─────────────────────────────────────────────


def test_ecps_employment_income_positive(ecps_sim):
    """Employment income must be in the trillions, not zero."""
    total = ecps_sim.calculate("employment_income").sum()
    assert total > 5e12, (
        f"employment_income sum is {total:.2e}, expected > 5T. "
        "Likely missing employment_income_before_lsr in dataset."
    )


def test_ecps_self_employment_income_positive(ecps_sim):
    total = ecps_sim.calculate("self_employment_income").sum()
    assert total > 50e9, f"self_employment_income sum is {total:.2e}, expected > 50B."


def test_ecps_household_count(ecps_sim):
    total_hh = ecps_sim.calculate("household_weight").values.sum()
    assert 100e6 < total_hh < 200e6, (
        f"Total households = {total_hh:.2e}, expected 100M-200M."
    )


def test_ecps_person_count(ecps_sim):
    total_people = ecps_sim.calculate("household_weight", map_to="person").values.sum()
    assert 250e6 < total_people < 400e6, (
        f"Total people = {total_people:.2e}, expected 250M-400M."
    )


def test_ecps_poverty_rate_reasonable(ecps_sim):
    in_poverty = ecps_sim.calculate("person_in_poverty", map_to="person")
    rate = in_poverty.mean()
    assert 0.05 < rate < 0.30, (
        f"Poverty rate = {rate:.1%}, expected 5-30%. "
        "If ~40%, income variables are likely zero."
    )


def test_ecps_income_tax_positive(ecps_sim):
    total = ecps_sim.calculate("income_tax").sum()
    assert total > 1e12, f"income_tax sum is {total:.2e}, expected > 1T."


def test_ecps_mean_employment_income_reasonable(ecps_sim):
    income = ecps_sim.calculate("employment_income", map_to="person")
    mean = income.mean()
    assert 15_000 < mean < 80_000, (
        f"Mean employment income = ${mean:,.0f}, expected $15k-$80k."
    )


def test_ecps_file_size():
    from policyengine_us_data.storage import STORAGE_FOLDER

    path = STORAGE_FOLDER / "enhanced_cps_2024.h5"
    if not path.exists():
        pytest.skip("enhanced_cps_2024.h5 not found")
    size_mb = path.stat().st_size / (1024 * 1024)
    assert size_mb > 95, f"enhanced_cps_2024.h5 is only {size_mb:.1f}MB, expected >95MB"


# ── Feature checks ────────────────────────────────────────────


def test_ecps_employment_income_direct():
    """Direct check that employment income from the actual dataset is > 5T.

    This tests the ACTUAL H5 dataset, not calibration_log.csv, and would
    have caught the bug where employment_income_before_lsr was dropped.
    """
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)
    total = sim.calculate("employment_income").sum()
    assert total > 5e12, (
        f"employment_income sum is {total:.2e}, expected > 5T. "
        "Likely missing employment_income_before_lsr in dataset."
    )


def test_ecps_has_mortgage_interest():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    assert sim.calculate("deductible_mortgage_interest").sum() > 1


def test_ecps_has_tips():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)
    # Ensure we impute at least $40 billion in tip income.
    # We currently target $38 billion * 1.4 = $53.2 billion.
    TIP_INCOME_MINIMUM = 40e9
    assert sim.calculate("tip_income").sum() > TIP_INCOME_MINIMUM


def test_ecps_replicates_jct_tax_expenditures():
    import pandas as pd

    calibration_log = pd.read_csv(
        "calibration_log.csv",
    )

    jct_rows = calibration_log[
        (calibration_log["target_name"].str.contains("jct/"))
        & (calibration_log["epoch"] == calibration_log["epoch"].max())
    ]

    assert jct_rows.rel_abs_error.max() < 0.5, (
        "JCT tax expenditure targets not met (see the calibration log for details). Max relative error: {:.2%}".format(
            jct_rows.rel_abs_error.max()
        )
    )


def deprecated_test_ecps_replicates_jct_tax_expenditures_full():
    from policyengine_us import Microsimulation
    from policyengine_core.reforms import Reform
    from policyengine_us_data.datasets import EnhancedCPS_2024

    # JCT tax expenditure targets
    EXPENDITURE_TARGETS = {
        "salt_deduction": 21.247e9,
        "medical_expense_deduction": 11.4e9,
        "charitable_deduction": 65.301e9,
        "interest_deduction": 24.8e9,
    }

    baseline = Microsimulation(dataset=EnhancedCPS_2024)
    income_tax_b = baseline.calculate("income_tax", period=2024, map_to="household")

    for deduction, target in EXPENDITURE_TARGETS.items():
        # Create reform that neutralizes the deduction
        class RepealDeduction(Reform):
            def apply(self):
                self.neutralize_variable(deduction)

        # Run reform simulation
        reformed = Microsimulation(reform=RepealDeduction, dataset=EnhancedCPS_2024)
        income_tax_r = reformed.calculate("income_tax", period=2024, map_to="household")

        # Calculate tax expenditure
        tax_expenditure = (income_tax_r - income_tax_b).sum()
        pct_error = abs((tax_expenditure - target) / target)
        TOLERANCE = 0.4

        print(
            f"{deduction} tax expenditure {tax_expenditure / 1e9:.1f}bn differs from target {target / 1e9:.1f}bn by {pct_error:.2%}"
        )
        assert pct_error < TOLERANCE, deduction


def test_ssn_card_type_none_target():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    TARGET_COUNT = 13e6
    TOLERANCE = 0.2  # Allow ±20% error

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    # Calculate the number of individuals with ssn_card_type == "NONE"
    ssn_type_none_mask = sim.calculate("ssn_card_type") == "NONE"
    count = ssn_type_none_mask.sum()

    pct_error = abs((count - TARGET_COUNT) / TARGET_COUNT)

    print(
        f'SSN card type "NONE" count: {count:.0f}, target: {TARGET_COUNT:.0f}, error: {pct_error:.2%}'
    )
    assert pct_error < TOLERANCE


def test_undocumented_matches_ssn_none():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation
    import numpy as np

    TARGET_COUNT = 13e6
    TOLERANCE = 0.2  # ±20 %

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    ssn_type_none_mask = sim.calculate("ssn_card_type") == "NONE"
    undocumented_mask = sim.calculate("immigration_status") == "UNDOCUMENTED"

    # 1. Per-person equivalence
    mismatches = np.where(ssn_type_none_mask != undocumented_mask)[0]
    assert mismatches.size == 0, (
        f"{mismatches.size} mismatches between 'NONE' SSN and 'UNDOCUMENTED' status"
    )

    # 2. Optional aggregate sanity-check
    count = undocumented_mask.sum()
    pct_error = abs((count - TARGET_COUNT) / TARGET_COUNT)
    print(
        f'Immigrant class "UNDOCUMENTED" count: {count:.0f}, target: {TARGET_COUNT:.0f}, error: {pct_error:.2%}'
    )
    assert pct_error < TOLERANCE


def test_has_tin_matches_identification_inputs(ecps_sim):
    data = ecps_sim.dataset.load_dataset()
    has_tin = _period_array(data["has_tin"], 2024)
    has_itin = _period_array(data["has_itin"], 2024)
    ssn_card_type = _period_array(data["ssn_card_type"], 2024).astype(str)

    # has_itin is still an alias for has_tin
    np.testing.assert_array_equal(has_itin, has_tin)
    # Everyone with an SSN card has a TIN
    assert has_tin[ssn_card_type != "NONE"].all()
    # Some code-0 (NONE) people have TINs via ITIN
    none_mask = ssn_card_type == "NONE"
    assert none_mask.any(), "Expected some ssn_card_type == NONE"
    assert has_tin[none_mask].any(), "Expected some ITIN holders among code-0"


def test_aca_calibration():

    import pandas as pd
    from pathlib import Path
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024

    TARGETS_PATH = Path(
        "policyengine_us_data/storage/calibration_targets/aca_spending_and_enrollment_2024.csv"
    )
    targets = pd.read_csv(TARGETS_PATH)
    # Monthly to yearly
    targets["spending"] = targets["spending"] * 12
    # Adjust to match national target
    targets["spending"] = targets["spending"] * (98e9 / targets["spending"].sum())

    sim = Microsimulation(dataset=EnhancedCPS_2024)
    state_code_hh = sim.calculate("state_code", map_to="household").values
    aca_ptc = sim.calculate("aca_ptc", map_to="household", period=2025)

    TOLERANCE = 0.70
    failed = False
    for _, row in targets.iterrows():
        state = row["state"]
        target_spending = row["spending"]
        simulated = aca_ptc[state_code_hh == state].sum()

        pct_error = abs(simulated - target_spending) / target_spending
        print(
            f"{state}: simulated ${simulated / 1e9:.2f} bn  "
            f"target ${target_spending / 1e9:.2f} bn  "
            f"error {pct_error:.2%}"
        )

        if pct_error > TOLERANCE:
            failed = True

    assert not failed, f"One or more states exceeded tolerance of {TOLERANCE:.0%}."


def test_aca_2025_takeup_override_helper():
    from policyengine_us_data.datasets.cps.enhanced_cps import (
        create_aca_2025_takeup_override,
    )

    result = create_aca_2025_takeup_override(
        base_takeup=np.array([True, False, False], dtype=bool),
        person_enrolled_if_takeup=np.array([True, True, True, True], dtype=bool),
        person_weights=np.array([2.0, 1.0, 3.0, 4.0], dtype=np.float64),
        person_tax_unit_ids=np.array([10, 10, 11, 12], dtype=np.int64),
        tax_unit_ids=np.array([10, 11, 12], dtype=np.int64),
        target_people=6.0,
    )

    assert np.all(np.array([True, False, False]) <= result)
    assert result.dtype == bool
    assert result.sum() == 2


def test_immigration_status_diversity():
    """Test that immigration statuses show appropriate diversity (not all citizens)."""
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    # Get immigration status for all persons (weighted MicroSeries)
    immigration_status = sim.calculate("immigration_status", 2024)

    # Weighted counts by status
    weighted_counts = immigration_status.weights.groupby(immigration_status).sum()
    total_weighted = weighted_counts.sum()

    for status, wt in weighted_counts.items():
        pct = 100 * wt / total_weighted
        print(f"  {status}: {wt:,.0f} ({pct:.1f}%)")

    citizen_pct = 100 * weighted_counts.get("CITIZEN", 0) / total_weighted

    # Fail if more than 99% are citizens (indicating the default is being used)
    assert citizen_pct < 99, (
        f"Too many citizens ({citizen_pct:.1f}%) - likely using default value. "
        "Immigration status not being read from data."
    )

    # Also check that we have a reasonable percentage of citizens (should be 85-90%)
    assert 80 < citizen_pct < 95, (
        f"Citizen percentage ({citizen_pct:.1f}%) outside expected range (80-95%)"
    )

    # Check that we have some non-citizens
    non_citizen_pct = 100 - citizen_pct
    assert non_citizen_pct > 5, (
        f"Too few non-citizens ({non_citizen_pct:.1f}%) - expected at least 5%"
    )

    print(f"Immigration status diversity test passed: {citizen_pct:.1f}% citizens")


def test_medicaid_calibration():

    import pandas as pd
    from pathlib import Path
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024

    TARGETS_PATH = Path(
        "policyengine_us_data/storage/calibration_targets/medicaid_enrollment_2024.csv"
    )
    targets = pd.read_csv(TARGETS_PATH)

    sim = Microsimulation(dataset=EnhancedCPS_2024)
    state_code_hh = sim.calculate("state_code", map_to="household").values
    medicaid_enrolled = sim.calculate(
        "medicaid_enrolled", map_to="household", period=2025
    )

    TOLERANCE = 0.45
    failed = False
    for _, row in targets.iterrows():
        state = row["state"]
        target_enrollment = row["enrollment"]
        simulated = medicaid_enrolled[state_code_hh == state].sum()

        pct_error = abs(simulated - target_enrollment) / target_enrollment
        print(
            f"{state}: simulated ${simulated / 1e9:.2f} bn  "
            f"target ${target_enrollment / 1e9:.2f} bn  "
            f"error {pct_error:.2%}"
        )

        if pct_error > TOLERANCE:
            failed = True

    assert not failed, f"One or more states exceeded tolerance of {TOLERANCE:.0%}."
