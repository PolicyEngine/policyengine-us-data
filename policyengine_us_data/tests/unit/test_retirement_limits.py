"""Tests for retirement contribution limits utility."""

import pytest
from policyengine_us_data.utils.retirement_limits import (
    get_retirement_limits,
)

# Expected values sourced from IRS announcements and policyengine-us
# parameter tree.
EXPECTED = {
    2020: {
        "401k": 19_500,
        "401k_catch_up": 6_500,
        "ira": 6_000,
        "ira_catch_up": 1_000,
    },
    2023: {
        "401k": 22_500,
        "401k_catch_up": 7_500,
        "ira": 6_500,
        "ira_catch_up": 1_000,
    },
    2025: {
        "401k": 23_500,
        "401k_catch_up": 7_500,
        "ira": 7_000,
        "ira_catch_up": 1_000,
    },
}


@pytest.mark.parametrize("year", EXPECTED.keys())
def test_retirement_limits(year):
    limits = get_retirement_limits(year)
    assert limits == EXPECTED[year]
