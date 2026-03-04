"""Sanity checks for built datasets.

These tests catch catastrophic data issues like missing income
variables, wrong population counts, or corrupted files. They run
after every data build and would have caught the enhanced CPS
overwrite bug (PR #569) where employment_income_before_lsr was
dropped, zeroing out all employment income.
"""

import pytest
import numpy as np


@pytest.fixture(scope="module")
def ecps_sim():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    return Microsimulation(dataset=EnhancedCPS_2024)


@pytest.fixture(scope="module")
def cps_sim():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    return Microsimulation(dataset=CPS_2024)


# ── Enhanced CPS sanity checks ──────────────────────────────────


def test_ecps_employment_income_positive(ecps_sim):
    """Employment income must be in the trillions, not zero."""
    total = ecps_sim.calculate("employment_income").sum()
    assert total > 5e12, (
        f"employment_income sum is {total:.2e}, expected > 5T. "
        "Likely missing employment_income_before_lsr in dataset."
    )


def test_ecps_self_employment_income_positive(ecps_sim):
    total = ecps_sim.calculate("self_employment_income").sum()
    assert (
        total > 50e9
    ), f"self_employment_income sum is {total:.2e}, expected > 50B."


def test_ecps_household_count(ecps_sim):
    """Household count should be roughly 130-160M."""
    weights = ecps_sim.calculate("household_weight")
    total_hh = weights.sum()
    assert (
        100e6 < total_hh < 200e6
    ), f"Total households = {total_hh:.2e}, expected 100M-200M."


def test_ecps_person_count(ecps_sim):
    """Weighted person count should be roughly 330M."""
    weights = ecps_sim.calculate("household_weight", map_to="person")
    total_people = weights.sum()
    assert (
        250e6 < total_people < 400e6
    ), f"Total people = {total_people:.2e}, expected 250M-400M."


def test_ecps_poverty_rate_reasonable(ecps_sim):
    """SPM poverty rate should be 8-25%, not 40%+."""
    in_poverty = ecps_sim.calculate("person_in_poverty", map_to="person")
    rate = in_poverty.mean()
    assert 0.05 < rate < 0.25, (
        f"Poverty rate = {rate:.1%}, expected 5-25%. "
        "If ~40%, income variables are likely zero."
    )


def test_ecps_income_tax_positive(ecps_sim):
    """Federal income tax revenue should be in the trillions."""
    total = ecps_sim.calculate("income_tax").sum()
    assert total > 1e12, f"income_tax sum is {total:.2e}, expected > 1T."


def test_ecps_mean_employment_income_reasonable(ecps_sim):
    """Mean employment income per person should be $20k-$60k."""
    income = ecps_sim.calculate("employment_income", map_to="person")
    mean = income.mean()
    assert 15_000 < mean < 80_000, (
        f"Mean employment income = ${mean:,.0f}, " "expected $15k-$80k."
    )


# ── CPS sanity checks ───────────────────────────────────────────


def test_cps_employment_income_positive(cps_sim):
    total = cps_sim.calculate("employment_income").sum()
    assert total > 5e12, (
        f"CPS employment_income sum is {total:.2e}, " "expected > 5T."
    )


def test_cps_household_count(cps_sim):
    weights = cps_sim.calculate("household_weight")
    total_hh = weights.sum()
    assert 100e6 < total_hh < 200e6, f"CPS total households = {total_hh:.2e}."
