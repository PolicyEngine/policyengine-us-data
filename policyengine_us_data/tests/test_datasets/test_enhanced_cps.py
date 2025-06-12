import pytest


@pytest.mark.parametrize("year", [2024])
def test_policyengine_cps_generates(year: int):
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024

    dataset_by_year = {
        2024: EnhancedCPS_2024,
    }

    dataset_by_year[year](require=True)


@pytest.mark.parametrize("year", [2024])
def test_policyengine_cps_loads(year: int):
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024

    dataset_by_year = {
        2024: EnhancedCPS_2024,
    }

    dataset = dataset_by_year[year]

    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=dataset)

    assert not sim.calculate("household_net_income").isna().any()


def test_ecps_has_mortgage_interest():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    assert sim.calculate("deductible_mortgage_interest").sum() > 1


def test_ecps_has_tips():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)
    # Ensure we impute at least $50 billion in tip income.
    # We currently target $38 billion * 1.4 = $53.2 billion.
    TIP_INCOME_MINIMUM = 50e9
    assert sim.calculate("tip_income").sum() > TIP_INCOME_MINIMUM


def test_ecps_replicates_jct_tax_expenditures():
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
        TOLERANCE = 0.3

        print(
            f"{deduction} tax expenditure {tax_expenditure/1e9:.1f}bn differs from target {target/1e9:.1f}bn by {pct_error:.2%}"
        )
        assert pct_error < TOLERANCE, deduction


def test_ssn_card_type_none_target():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation
    import numpy as np

    TARGET_COUNT = 11e6
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


def test_aca_calibration():

    import pandas as pd
    from pathlib import Path
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024

    TARGETS_PATH = Path(
        "policyengine_us_data/storage/aca_spending_and_enrollment_2024.csv"
    )
    targets = pd.read_csv(TARGETS_PATH)
    targets["spending"] = (
        targets["spending"].astype(str).str.replace("_", "").astype(int)
    )

    sim = Microsimulation(dataset=EnhancedCPS_2024)
    state_code_hh = sim.calculate("state_code", map_to="household").values
    aca_ptc = sim.calculate("aca_ptc", map_to="household", period=2025).values

    TOLERANCE = 0.20
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

        assert (
            pct_error < TOLERANCE
        ), f"{state} spending off by {pct_error:.1%}"
