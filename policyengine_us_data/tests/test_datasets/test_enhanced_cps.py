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
    # Ensure we impute at least $45 billion in tip income.
    # We currently target $38 billion * 1.4 = $53.2 billion.
    TIP_INCOME_MINIMUM = 45e9
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
        jct_rows.rel_abs_error.max() < 0.4
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


def test_ctc_reform_child_recipient_difference():
    """
    Test CTC reform impact for validation purposes only.
    Note: This is no longer actively targeted in loss matrix calibration
    due to uncertainty around assumptions from hearing comments.
    """
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation
    from policyengine_core.reforms import Reform

    TARGET_COUNT = 2e6
    TOLERANCE = 4.5  # Allow +/-450% error

    # Define the CTC reform
    ctc_reform = Reform.from_dict(
        {
            "gov.contrib.reconciliation.ctc.in_effect": {
                "2025-01-01.2100-12-31": True
            }
        },
        country_id="us",
    )

    # Create baseline and reform simulations
    baseline_sim = Microsimulation(dataset=EnhancedCPS_2024)
    reform_sim = Microsimulation(dataset=EnhancedCPS_2024, reform=ctc_reform)

    # Calculate baseline CTC recipients (children with ctc_individual_maximum > 0 and ctc_value > 0)
    baseline_is_child = baseline_sim.calculate("is_child")
    baseline_ctc_individual_maximum = baseline_sim.calculate(
        "ctc_individual_maximum"
    )
    baseline_ctc_value = baseline_sim.calculate("ctc_value", map_to="person")
    baseline_child_ctc_recipients = (
        baseline_is_child
        * (baseline_ctc_individual_maximum > 0)
        * (baseline_ctc_value > 0)
    ).sum()

    # Calculate reform CTC recipients (children with ctc_individual_maximum > 0 and ctc_value > 0)
    reform_is_child = reform_sim.calculate("is_child")
    reform_ctc_individual_maximum = reform_sim.calculate(
        "ctc_individual_maximum"
    )
    reform_ctc_value = reform_sim.calculate("ctc_value", map_to="person")
    reform_child_ctc_recipients = (
        reform_is_child
        * (reform_ctc_individual_maximum > 0)
        * (reform_ctc_value > 0)
    ).sum()

    # Calculate the difference (baseline - reform child CTC recipients)
    ctc_recipient_difference = (
        baseline_child_ctc_recipients - reform_child_ctc_recipients
    )

    pct_error = abs((ctc_recipient_difference - TARGET_COUNT) / TARGET_COUNT)

    print(
        f"CTC reform child recipient difference: {ctc_recipient_difference:.0f}, target: {TARGET_COUNT:.0f}, error: {pct_error:.2%}"
    )
    print(
        "Note: CTC targeting removed from calibration - this is validation only"
    )
    assert pct_error < TOLERANCE


def test_aca_calibration():

    import pandas as pd
    from pathlib import Path
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024

    TARGETS_PATH = Path(
        "policyengine_us_data/storage/aca_spending_and_enrollment_2024.csv"
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
