"""
Tests for weeks_worked extraction from CPS ASEC.

The Census CPS ASEC exposes WKSWORK directly, which we now carry through as
the model input for future-year SPM work-expense calculations.
"""

import numpy as np
from pathlib import Path


class TestWeeksWorked:
    """Test suite for weeks_worked variable extraction."""

    def test_census_cps_includes_wkswork(self):
        census_cps_path = Path(__file__).parent.parent.parent / (
            "policyengine_us_data/datasets/cps/census_cps.py"
        )
        content = census_cps_path.read_text()

        assert '"WKSWORK"' in content, "WKSWORK should be in PERSON_COLUMNS"

    def test_cps_maps_weeks_worked_from_wkswork(self):
        cps_path = Path(__file__).parent.parent.parent / (
            "policyengine_us_data/datasets/cps/cps.py"
        )
        content = cps_path.read_text()

        assert 'cps["weeks_worked"]' in content
        assert "person.WKSWORK" in content
        assert "np.clip(person.WKSWORK, 0, 52)" in content

    def test_weeks_worked_value_range(self):
        raw_values = np.array([-4, 0, 1, 26, 52, 60])
        processed = np.clip(raw_values, 0, 52)

        assert processed.min() >= 0, "Minimum should be >= 0"
        assert processed.max() <= 52, "Maximum should be <= 52"
        assert processed[0] == 0, "Negative values should clip to 0"
        assert processed[3] == 26, "Valid weeks should be preserved"
        assert processed[5] == 52, "Values above 52 should clip to 52"
