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

    assert not sim.calculate("household_net_income", 2025).isna().any()
