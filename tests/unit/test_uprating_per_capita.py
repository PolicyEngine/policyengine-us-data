"""Tests for the ``is_per_capita_parameter`` classifier in
``policyengine_us_data.utils.uprating``.

Context
-------
``create_policyengine_uprating_factors_table`` converts each uprating
parameter into a per-capita growth factor:

    growth(year) = parameter(year) / parameter(START_YEAR)
    per_capita_growth(year) = growth / population_growth

That divide is only correct when the parameter is a *total* dollar
aggregate. When the parameter is already a per-capita index (CPI, the
SSA COLA, ``*_per_capita`` spending, etc.) the divide double-adjusts
and introduces a small compounding downward drift.

The classifier below decides whether to skip the divide based on the
parameter *path* rather than the variable name, so the fix is robust
to new uprated variables being added to policyengine-us.
"""

import pytest

from policyengine_us_data.utils import uprating as uprating_module


@pytest.mark.parametrize(
    "path",
    [
        "gov.bls.cpi.cpi_u",
        "gov.bls.cpi.cpi_u_cd",
        "gov.ssa.uprating",
        "calibration.gov.hhs.cms.moop_per_capita",
        "calibration.gov.bea.wages_per_worker",
        "calibration.gov.hhs.cms.cost_per_recipient",
        "gov.something.index.value",
    ],
)
def test_per_capita_parameter_paths_skip_population_divisor(path):
    assert uprating_module.is_per_capita_parameter(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "calibration.gov.irs.soi.employment_income",
        "calibration.gov.irs.soi.farm_income",
        "calibration.gov.irs.soi.long_term_capital_gains",
        "calibration.gov.irs.soi.partnership_s_corp_income",
        "calibration.gov.irs.soi.social_security",
        "calibration.gov.census.populations.total",
    ],
)
def test_total_dollar_aggregates_keep_population_divisor(path):
    assert uprating_module.is_per_capita_parameter(path) is False


def test_per_capita_markers_include_cpi_and_ssa_uprating():
    """Source-level sanity: the two most common double-adjusted
    parameters (BLS CPI and the SSA uprating series) are in the
    marker tuple."""
    markers = uprating_module.PER_CAPITA_PARAMETER_PATH_MARKERS
    assert any("cpi" in m for m in markers)
    assert any("ssa.uprating" in m for m in markers)
    assert any("per_capita" in m for m in markers)


def test_classifier_is_substring_match_not_equality():
    """Path-substring matching means adding a new
    ``calibration.gov.hhs.cms.moop_per_capita.adjusted`` does not
    require listing every child path — the ``per_capita`` substring
    catches it."""
    assert uprating_module.is_per_capita_parameter(
        "calibration.gov.hhs.cms.moop_per_capita.adjusted"
    )
    assert uprating_module.is_per_capita_parameter(
        "calibration.gov.bls.cpi.series.cpi_u_seasonal"
    )
