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
    assert sim.calculate("interest_expense").sum() > 1


def test_ecps_replicates_jct_salt_te():
    from policyengine_us import Microsimulation
    from policyengine_core.reforms import Reform
    from policyengine_core.data import Dataset
    from policyengine_us_data.datasets import EnhancedCPS_2024

    reform = Reform.from_dict(
        {
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.JOINT": {
                "2024-01-01.2100-12-31": 0
            },
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.SINGLE": {
                "2024-01-01.2100-12-31": 0
            },
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.SEPARATE": {
                "2024-01-01.2100-12-31": 0
            },
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.SURVIVING_SPOUSE": {
                "2024-01-01.2100-12-31": 0
            },
            "gov.irs.deductions.itemized.salt_and_real_estate.cap.HEAD_OF_HOUSEHOLD": {
                "2024-01-01.2100-12-31": 0
            },
        },
        country_id="us",
    )

    baseline = Microsimulation(dataset=EnhancedCPS_2024)
    reformed = Microsimulation(reform=reform, dataset=EnhancedCPS_2024)

    income_tax_b = baseline.calculate(
        "income_tax", period=2024, map_to="household"
    )
    income_tax_r = reformed.calculate(
        "income_tax", period=2024, map_to="household"
    )
    tax_change = income_tax_r - income_tax_b
    federal_tax_expenditure = tax_change.sum() / 1e9

    assert abs(federal_tax_expenditure - 20e9) < 5e9
