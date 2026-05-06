"""Validate the built Stage 1 ACS artifact."""

import h5py
import numpy as np
import pytest

from policyengine_us_data.datasets.acs.acs import ACS_2022


@pytest.fixture(scope="module")
def acs_path():
    if not ACS_2022.file_path.exists():
        pytest.skip("acs_2022.h5 not built locally")
    return ACS_2022.file_path


@pytest.fixture(scope="module")
def acs_file(acs_path):
    with h5py.File(acs_path, "r") as h5:
        yield h5


def test_acs_has_expected_stage_1_variables(acs_file):
    expected = {
        "age",
        "employment_income",
        "household_id",
        "household_weight",
        "is_household_head",
        "person_household_id",
        "person_id",
        "rent",
        "state_fips",
        "tenure_type",
    }

    assert expected <= set(acs_file.keys())


def test_acs_entity_lengths_are_consistent(acs_file):
    person_count = len(acs_file["person_id"])
    household_count = len(acs_file["household_id"])

    assert person_count > household_count > 0
    for variable in (
        "age",
        "employment_income",
        "is_household_head",
        "person_household_id",
        "person_id",
        "rent",
    ):
        assert len(acs_file[variable]) == person_count

    for variable in ("household_id", "household_weight", "state_fips"):
        assert len(acs_file[variable]) == household_count


def test_acs_expected_value_ranges(acs_file):
    age = acs_file["age"][...]
    household_weight = acs_file["household_weight"][...]
    state_fips = acs_file["state_fips"][...]
    person_household_id = acs_file["person_household_id"][...].astype(int)
    employment_income = acs_file["employment_income"][...].astype(float)

    assert age.min() >= 0
    assert age.max() <= 100
    assert 100e6 < household_weight.sum() < 200e6
    assert state_fips.min() >= 1
    assert state_fips.max() <= 56

    weighted_employment_income = (
        employment_income * household_weight[person_household_id]
    ).sum()
    assert weighted_employment_income > 5e12


def test_acs_tenure_values_match_expected_categories(acs_file):
    tenure_values = set(np.unique(acs_file["tenure_type"][...].astype(str)))

    assert tenure_values <= {
        "OWNED_WITH_MORTGAGE",
        "OWNED_OUTRIGHT",
        "RENTED",
        "NONE",
    }
    assert {"OWNED_WITH_MORTGAGE", "RENTED"} <= tenure_values
