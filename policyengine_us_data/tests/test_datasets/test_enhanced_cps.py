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

    assert sim.calculate("tip_income").sum() > 50e9


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
        TOLERANCE = 0.2

        print(
            f"{deduction} tax expenditure {tax_expenditure/1e9:.1f}bn differs from target {target/1e9:.1f}bn by {pct_error:.2%}"
        )
        assert pct_error < TOLERANCE, deduction
