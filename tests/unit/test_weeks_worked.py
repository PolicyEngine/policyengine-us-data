"""
Tests for weeks_worked extraction from CPS ASEC.

The Census CPS ASEC exposes WKSWORK directly, which we now carry through as
the model input for future-year SPM work-expense calculations.
"""

import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.census_cps import PERSON_COLUMNS
from policyengine_us_data.datasets.cps.cps import add_personal_income_variables


def _person_frame(**columns):
    n_persons = len(next(iter(columns.values())))
    data = {column: np.zeros(n_persons, dtype=int) for column in PERSON_COLUMNS}
    data.update(columns)
    return pd.DataFrame(data)


class TestWeeksWorked:
    """Test suite for weeks_worked variable extraction."""

    def test_census_cps_loads_wkswork(self):
        assert "WKSWORK" in PERSON_COLUMNS

    def test_cps_maps_weeks_worked_from_wkswork(self):
        person = _person_frame(
            A_AGE=np.full(6, 35),
            WKSWORK=np.array([-4, 0, 1, 26, 52, 60]),
        )

        cps = {}
        add_personal_income_variables(cps, person, 2024)

        np.testing.assert_array_equal(
            cps["weeks_worked"],
            np.array([0, 0, 1, 26, 52, 52]),
        )
