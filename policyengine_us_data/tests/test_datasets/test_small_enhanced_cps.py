import pytest
import numpy as np


@pytest.mark.parametrize("year", [2024])
def test_small_ecps_loads(year: int):
    from policyengine_core.data import Dataset
    from policyengine_us_data.storage import STORAGE_FOLDER
    from policyengine_us import Microsimulation

    sim = Microsimulation(
        dataset=Dataset.from_file(
            STORAGE_FOLDER / f"small_enhanced_cps_{year}.h5",
        )
    )

    # Basic load check
    assert not sim.calculate("household_net_income", 2025).isna().any()

    # Employment income should be positive (not zero from missing vars)
    emp_income = sim.calculate("employment_income", 2025).sum()
    assert emp_income > 0, (
        f"Small ECPS employment_income sum is {emp_income}, expected > 0."
    )

    # Should have a reasonable number of households
    hh_count = len(sim.calculate("household_net_income", 2025))
    assert hh_count > 100, (
        f"Small ECPS has only {hh_count} households, expected > 100."
    )
