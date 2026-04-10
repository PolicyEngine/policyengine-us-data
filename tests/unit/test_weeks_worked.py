"""
Tests for weeks_worked extraction from CPS ASEC.

The Census CPS ASEC exposes WKSWORK directly, which we now carry through as
the model input for future-year SPM work-expense calculations.
"""

import numpy as np

from policyengine_us_data.datasets.cps.census_cps import PERSON_COLUMNS
from policyengine_us_data.datasets.cps.cps import derive_weeks_worked


class TestWeeksWorked:
    """Test suite for weeks_worked variable extraction."""

    def test_census_cps_loads_wkswork(self):
        assert "WKSWORK" in PERSON_COLUMNS

    def test_cps_derives_weeks_worked_from_wkswork(self):
        np.testing.assert_array_equal(
            derive_weeks_worked(np.array([-4, 0, 1, 26, 52, 60])),
            np.array([0, 0, 1, 26, 52, 52]),
        )
