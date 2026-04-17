"""Regression tests for the ACS state_fips entity mismatch (N8).

``policyengine_us_data.datasets.acs.acs.ACS.add_household_variables``
used to write

    acs["state_fips"] = acs["household_state_fips"] = household.ST.astype(int)

That assigns a household-length array to ``state_fips``, which is a
*person*-entity variable in policyengine-us. Depending on the data
path this either silently mismatches lengths with other person
arrays or trips set_input broadcast downstream.

Fix: broadcast the household-level ``ST`` through the
``person["household_id"] -> household_id`` mapping before writing
``state_fips`` to the h5 file.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ACS_SOURCE = REPO_ROOT / "policyengine_us_data" / "datasets" / "acs" / "acs.py"


def test_add_household_variables_accepts_person_frame():
    """Source-level: the static method must take ``person`` as a second
    argument so it can broadcast the household-length ST array to
    the person axis."""
    tree = ast.parse(ACS_SOURCE.read_text())
    # Locate ACS class
    cls = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ACS"
        ),
        None,
    )
    assert cls is not None
    # Locate add_household_variables static method
    method = next(
        (
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "add_household_variables"
        ),
        None,
    )
    assert method is not None
    arg_names = [a.arg for a in method.args.args]
    assert arg_names == ["acs", "person", "household"], (
        f"add_household_variables must take (acs, person, household); got {arg_names}"
    )


def test_broadcasting_state_fips_to_person_axis_matches_expected_length():
    """Simulate the fix pattern on a 2-household, 5-person toy frame
    and verify that the person-level state_fips vector has length 5
    (matches person) rather than length 2 (matches household)."""
    household = pd.DataFrame({"household_id": [0, 1], "ST": [6, 36]})
    person = pd.DataFrame(
        {
            "household_id": [0, 0, 1, 1, 1],
        }
    )

    household_state_fips = household["ST"].astype(int)
    state_fips_by_household_id = pd.Series(
        household_state_fips.values, index=household["household_id"].values
    )
    state_fips = state_fips_by_household_id.loc[person["household_id"].values].values

    assert len(state_fips) == len(person) == 5
    # California for the first two persons, New York for the last three.
    np.testing.assert_array_equal(state_fips, np.array([6, 6, 36, 36, 36]))


def test_acs_source_no_longer_double_assigns_state_fips_to_household_length():
    """Pin the fix: the module must not do

        acs["state_fips"] = acs["household_state_fips"] = household.ST.astype(int)

    which was the original (entity-mismatched) assignment."""
    src = ACS_SOURCE.read_text()
    assert 'acs["state_fips"] = acs["household_state_fips"]' not in src, (
        "acs.py still contains the chained household-length assignment"
    )
