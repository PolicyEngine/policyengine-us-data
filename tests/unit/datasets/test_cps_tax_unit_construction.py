"""Integration coverage for the CPS dataset's use of the microunit engine.

The tax-unit construction *engine* itself is tested in the microunit package
(PolicyEngine/microunit), which is the canonical home for those rules. These
tests exercise this repository's own wiring into that engine: that
``CensusCPS._create_tax_unit_table`` calls ``microunit.construct_tax_units``
with the dataset's time period and construction mode, writes the constructed
``TAX_ID`` back onto the person table, preserves the original Census identifiers
under ``CENSUS_TAX_ID``, and returns the per-unit table.
"""

import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.census_cps import CensusCPS_2024


def _person_fixture(**overrides):
    n = max((len(value) for value in overrides.values()), default=1)
    defaults = {
        "PH_SEQ": np.ones(n, dtype=int),
        "A_LINENO": np.arange(1, n + 1, dtype=int),
        "TAX_ID": np.arange(1, n + 1, dtype=int),
        "A_AGE": np.zeros(n, dtype=int),
        "A_MARITL": np.full(n, 7, dtype=int),
        "A_SPOUSE": np.zeros(n, dtype=int),
        "PEPAR1": np.full(n, -1, dtype=int),
        "PEPAR2": np.full(n, -1, dtype=int),
        "A_EXPRRP": np.full(n, 14, dtype=int),
        "WSAL_VAL": np.zeros(n, dtype=float),
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


def test_create_tax_unit_table_wires_microunit_and_writes_back_tax_id():
    person = _person_fixture(
        A_AGE=[40, 38, 8],
        A_MARITL=[1, 1, 7],
        A_SPOUSE=[2, 1, 0],
        A_EXPRRP=[1, 4, 5],
        PEPAR1=[-1, -1, 1],
        PEPAR2=[-1, -1, 2],
        TAX_ID=[10, 10, 10],
        WSAL_VAL=[60_000, 20_000, 0],
    )

    tax_unit_df = CensusCPS_2024()._create_tax_unit_table(person)

    # The married couple plus their child collapse into a single constructed unit.
    assert person["TAX_ID"].nunique() == 1
    assert tax_unit_df.columns.tolist() == ["TAX_ID"]
    assert tax_unit_df["TAX_ID"].tolist() == [1]
    # The original Census identifier is preserved for downstream comparison.
    assert person["CENSUS_TAX_ID"].tolist() == [10, 10, 10]


def test_create_tax_unit_table_splits_unrelated_adults():
    person = _person_fixture(
        A_AGE=[45, 22],
        A_EXPRRP=[1, 5],
        PEPAR1=[-1, 1],
        TAX_ID=[7, 7],
        WSAL_VAL=[70_000, 10_000],
    )

    tax_unit_df = CensusCPS_2024()._create_tax_unit_table(person)

    # A high-income adult child cannot be claimed and forms an independent unit.
    assert person["TAX_ID"].tolist() == [1, 2]
    assert sorted(tax_unit_df["TAX_ID"].tolist()) == [1, 2]


def test_create_tax_unit_table_respects_dataset_year():
    # 2024 dependent gross income limit is $5,050: $5,000 of income keeps the
    # under-19 child claimable, exercising the year passed through to microunit.
    person = _person_fixture(
        A_AGE=[45, 17],
        A_EXPRRP=[1, 5],
        PEPAR1=[-1, 1],
        TAX_ID=[3, 3],
        WSAL_VAL=[70_000, 5_000],
    )

    tax_unit_df = CensusCPS_2024()._create_tax_unit_table(person)

    assert person["TAX_ID"].tolist() == [1, 1]
    assert tax_unit_df["TAX_ID"].tolist() == [1]
