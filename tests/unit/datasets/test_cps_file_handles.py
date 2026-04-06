from types import SimpleNamespace

import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.cps import add_previous_year_income


class _FakeStore:
    def __init__(self, person: pd.DataFrame):
        self.person = person
        self.closed = False

    def close(self):
        self.closed = True


class _FakeDataset:
    store: _FakeStore | None = None

    def __init__(self, require: bool = False):
        assert require is True

    def load(self):
        assert self.store is not None
        return self.store


def test_add_previous_year_income_closes_raw_cps_handles():
    current_person = pd.DataFrame(
        {
            "PERIDNUM": [10, 20],
            "I_ERNVAL": [0, 0],
            "I_SEVAL": [0, 0],
        }
    )
    previous_person = pd.DataFrame(
        {
            "PERIDNUM": [10, 20],
            "WSAL_VAL": [1_000, 2_000],
            "SEMP_VAL": [100, 200],
            "I_ERNVAL": [0, 0],
            "I_SEVAL": [0, 0],
        }
    )

    current_store = _FakeStore(current_person)
    previous_store = _FakeStore(previous_person)

    current_dataset = type("CurrentDataset", (_FakeDataset,), {"store": current_store})
    previous_dataset = type(
        "PreviousDataset", (_FakeDataset,), {"store": previous_store}
    )

    holder = SimpleNamespace(
        raw_cps=current_dataset,
        previous_year_raw_cps=previous_dataset,
    )
    cps = {}

    add_previous_year_income(holder, cps)

    np.testing.assert_array_equal(cps["employment_income_last_year"], [1000, 2000])
    np.testing.assert_array_equal(cps["self_employment_income_last_year"], [100, 200])
    np.testing.assert_array_equal(cps["previous_year_income_available"], [True, True])
    assert current_store.closed is True
    assert previous_store.closed is True
