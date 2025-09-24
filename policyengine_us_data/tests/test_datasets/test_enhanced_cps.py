import pytest


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

    assert (
        jct_rows.rel_abs_error.max() < 0.5
    ), "JCT tax expenditure targets not met (see the calibration log for details). Max relative error: {:.2%}".format(
        jct_rows.rel_abs_error.max()
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
    income_tax_b = baseline.calculate(
        "income_tax", period=2024, map_to="household"
    )

    for deduction, target in EXPENDITURE_TARGETS.items():
        # Create reform that neutralizes the deduction
        class RepealDeduction(Reform):
            def apply(self):
                self.neutralize_variable(deduction)

        # Run reform simulation
        reformed = Microsimulation(
            reform=RepealDeduction, dataset=EnhancedCPS_2024
        )
        income_tax_r = reformed.calculate(
            "income_tax", period=2024, map_to="household"
        )

        # Calculate tax expenditure
        tax_expenditure = (income_tax_r - income_tax_b).sum()
        pct_error = abs((tax_expenditure - target) / target)
        TOLERANCE = 0.4

        print(
            f"{deduction} tax expenditure {tax_expenditure/1e9:.1f}bn differs from target {target/1e9:.1f}bn by {pct_error:.2%}"
        )
        assert pct_error < TOLERANCE, deduction


def test_ssn_card_type_none_target():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation
    import numpy as np

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
    assert (
        mismatches.size == 0
    ), f"{mismatches.size} mismatches between 'NONE' SSN and 'UNDOCUMENTED' status"

    # 2. Optional aggregate sanity-check
    count = undocumented_mask.sum()
    pct_error = abs((count - TARGET_COUNT) / TARGET_COUNT)
    print(
        f'Immigrant class "UNDOCUMENTED" count: {count:.0f}, target: {TARGET_COUNT:.0f}, error: {pct_error:.2%}'
    )
    assert pct_error < TOLERANCE


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
    targets["spending"] = targets["spending"] * (
        98e9 / targets["spending"].sum()
    )

    sim = Microsimulation(dataset=EnhancedCPS_2024)
    state_code_hh = sim.calculate("state_code", map_to="household").values
    aca_ptc = sim.calculate("aca_ptc", map_to="household", period=2025)

    TOLERANCE = 0.45
    failed = False
    for _, row in targets.iterrows():
        state = row["state"]
        target_spending = row["spending"]
        simulated = aca_ptc[state_code_hh == state].sum()

        pct_error = abs(simulated - target_spending) / target_spending
        print(
            f"{state}: simulated ${simulated/1e9:.2f} bn  "
            f"target ${target_spending/1e9:.2f} bn  "
            f"error {pct_error:.2%}"
        )

        if pct_error > TOLERANCE:
            failed = True

    assert (
        not failed
    ), f"One or more states exceeded tolerance of {TOLERANCE:.0%}."


def test_immigration_status_diversity():
    """Test that immigration statuses show appropriate diversity (not all citizens)."""
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation
    import numpy as np

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    # Get immigration status for all persons (already weighted MicroSeries)
    immigration_status = sim.calculate("immigration_status", 2024)

    # Count different statuses
    unique_statuses, counts = np.unique(immigration_status, return_counts=True)

    # Calculate percentages using the weights directly
    total_population = len(immigration_status)
    status_percentages = {}

    for status, count in zip(unique_statuses, counts):
        pct = 100 * count / total_population
        status_percentages[status] = pct
        print(f"  {status}: {count:,} ({pct:.1f}%)")

    # Test that not everyone is a citizen (would indicate default value being used)
    citizen_pct = status_percentages.get("CITIZEN", 0)

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
            f"{state}: simulated ${simulated/1e9:.2f} bn  "
            f"target ${target_enrollment/1e9:.2f} bn  "
            f"error {pct_error:.2%}"
        )

        if pct_error > TOLERANCE:
            failed = True

    assert (
        not failed
    ), f"One or more states exceeded tolerance of {TOLERANCE:.0%}."
