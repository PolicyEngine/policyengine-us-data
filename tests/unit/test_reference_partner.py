"""
Tests for reference-person partner extraction from CPS ASEC.

The public CPS ASEC relationship-to-reference-person variable PERRP identifies
unmarried partners of the household head/reference person. We carry that
through so the SPM childcare cap can distinguish the reference person's partner
from unrelated adults in the same SPM unit.
"""

import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.census_cps import PERSON_COLUMNS
from policyengine_us_data.datasets.cps.cps import (
    PERRP_UNMARRIED_PARTNER_OF_HOUSEHOLD_HEAD_CODES,
    add_personal_variables,
)


def _person_frame(**columns):
    n_persons = len(next(iter(columns.values())))
    data = {column: np.zeros(n_persons, dtype=int) for column in PERSON_COLUMNS}
    data.update(columns)
    return pd.DataFrame(data)


class TestReferencePartner:
    """Test suite for CPS relationship-to-reference-person extraction."""

    def test_census_cps_loads_perrp(self):
        assert "PERRP" in PERSON_COLUMNS

    def test_unmarried_partner_perrp_code_table_matches_census_labels(self):
        assert PERRP_UNMARRIED_PARTNER_OF_HOUSEHOLD_HEAD_CODES == {
            43: "Opposite Sex Unmarried Partner with Relatives",
            44: "Opposite Sex Unmarried Partner without Relatives",
            46: "Same Sex Unmarried Partner with Relatives",
            47: "Same Sex Unmarried Partner without Relatives",
        }

    def test_cps_maps_unmarried_partner_from_perrp(self):
        person = _person_frame(
            PH_SEQ=np.arange(7) + 1,
            A_LINENO=np.ones(7),
            A_AGE=np.full(7, 35),
            PERRP=np.array([40, 43, 44, 45, 46, 47, 48]),
        )

        cps = {}
        add_personal_variables(cps, person)

        np.testing.assert_array_equal(
            cps["is_unmarried_partner_of_household_head"],
            np.array([False, True, True, False, True, True, False]),
        )
