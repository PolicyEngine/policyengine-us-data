from types import SimpleNamespace

import numpy as np
import pandas as pd

import policyengine_us_data.datasets.cps.cps as cps_module
from policyengine_us_data.datasets.cps.cps import (
    add_rent,
    add_auto_loan_interest_and_net_worth,
    add_previous_year_income,
)


class _FakeStore:
    def __init__(self, person: pd.DataFrame):
        self.person = person
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

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


def test_add_previous_year_income_opens_hdfstores_read_only(tmp_path, monkeypatch):
    current_path = tmp_path / "current.h5"
    previous_path = tmp_path / "previous.h5"

    with pd.HDFStore(current_path, mode="w") as store:
        store["person"] = pd.DataFrame(
            {
                "PERIDNUM": [10, 20],
                "I_ERNVAL": [0, 0],
                "I_SEVAL": [0, 0],
            }
        )

    with pd.HDFStore(previous_path, mode="w") as store:
        store["person"] = pd.DataFrame(
            {
                "PERIDNUM": [10, 20],
                "WSAL_VAL": [1_000, 2_000],
                "SEMP_VAL": [100, 200],
                "I_ERNVAL": [0, 0],
                "I_SEVAL": [0, 0],
            }
        )

    real_hdfstore = pd.HDFStore
    opened_modes = []

    def recording_hdfstore(path, mode="a", *args, **kwargs):
        opened_modes.append(mode)
        return real_hdfstore(path, mode=mode, *args, **kwargs)

    monkeypatch.setattr(cps_module.pd, "HDFStore", recording_hdfstore)

    class CurrentDataset:
        file_path = current_path

        def __init__(self, require: bool = False):
            assert require is True

    class PreviousDataset:
        file_path = previous_path

        def __init__(self, require: bool = False):
            assert require is True

    holder = SimpleNamespace(
        raw_cps=CurrentDataset,
        previous_year_raw_cps=PreviousDataset,
    )

    cps = {}
    add_previous_year_income(holder, cps)

    assert opened_modes == ["r", "r"]
    np.testing.assert_array_equal(cps["employment_income_last_year"], [1000, 2000])


def test_add_auto_loan_interest_and_net_worth_uses_outer_receiver_data(monkeypatch):
    raw_person = pd.DataFrame(
        {
            "A_SEX": [1, 2],
            "A_MARITL": [1, 3],
        }
    )
    raw_store = _FakeStore(raw_person)

    class FakeRawCPS:
        def __call__(self, require: bool = False):
            assert require is True
            return self

        def load(self):
            return raw_store

    class FakeDataset:
        def __init__(self):
            self.raw_cps = FakeRawCPS()
            self.saved_dataset = None

        def save_dataset(self, data):
            self.saved_dataset = data

        def load_dataset(self):
            return {
                "person_household_id": np.array([10, 20]),
                "age": np.array([35, 40]),
                "is_female": np.array([False, True]),
                "cps_race": np.array([1, 2]),
                "own_children_in_household": np.array([0, 1]),
                "employment_income": np.array([40_000.0, 25_000.0]),
                "taxable_interest_income": np.array([100.0, 0.0]),
                "tax_exempt_interest_income": np.array([0.0, 0.0]),
                "qualified_dividend_income": np.array([0.0, 0.0]),
                "non_qualified_dividend_income": np.array([0.0, 0.0]),
                "tax_exempt_private_pension_income": np.array([0.0, 0.0]),
                "taxable_private_pension_income": np.array([0.0, 0.0]),
                "social_security_retirement": np.array([0.0, 0.0]),
            }

    class FakeSCF:
        def load_dataset(self):
            return {
                "age": np.array([30, 50]),
                "is_female": np.array([False, True]),
                "cps_race": np.array([1, 2]),
                "is_married": np.array([True, False]),
                "own_children_in_household": np.array([0, 1]),
                "employment_income": np.array([35_000.0, 20_000.0]),
                "interest_dividend_income": np.array([100.0, 50.0]),
                "social_security_pension_income": np.array([0.0, 0.0]),
                "networth": np.array([10_000.0, 5_000.0]),
                "auto_loan_balance": np.array([2_000.0, 1_000.0]),
                "auto_loan_interest": np.array([200.0, 100.0]),
                "wgt": np.array([1.0, 1.0]),
            }

    class FakeQRF:
        def fit(
            self,
            X_train,
            predictors,
            imputed_variables,
            weight_col,
            tune_hyperparameters,
        ):
            assert predictors[0] == "age"
            assert weight_col == "wgt"
            self.imputed_variables = imputed_variables
            return self

        def predict(self, X_test):
            assert X_test["is_married"].tolist() == [True, False]
            return pd.DataFrame(
                {
                    "networth": [10_000.0, 5_000.0],
                    "auto_loan_balance": [2_000.0, 1_000.0],
                    "auto_loan_interest": [200.0, 100.0],
                }
            )

    import policyengine_us_data.datasets.scf.scf as scf_module
    import microimpute.models.qrf as qrf_module

    monkeypatch.setattr(scf_module, "SCF_2022", FakeSCF)
    monkeypatch.setattr(qrf_module, "QRF", FakeQRF)

    dataset = FakeDataset()
    add_auto_loan_interest_and_net_worth(dataset, {})

    assert raw_store.closed is True
    np.testing.assert_array_equal(
        dataset.saved_dataset["net_worth"], [10_000.0, 5_000.0]
    )
    np.testing.assert_array_equal(
        dataset.saved_dataset["auto_loan_interest"], [200.0, 100.0]
    )


def test_add_rent_replaces_existing_hdf_using_read_only_hdfstore(tmp_path, monkeypatch):
    existing_path = tmp_path / "existing_cps.h5"
    with pd.HDFStore(existing_path, mode="w") as store:
        store["stale_var"] = pd.Series([1, 2, 3])

    real_hdfstore = pd.HDFStore
    opened_modes = []

    def recording_hdfstore(path, mode="a", *args, **kwargs):
        opened_modes.append(mode)
        return real_hdfstore(path, mode=mode, *args, **kwargs)

    def fail_h5py_file(*args, **kwargs):
        raise AssertionError("add_rent should not reopen the existing H5 with h5py")

    class FakeQRF:
        def fit(self, X_train, predictors, imputed_variables):
            return self

        def predict(self, X_test):
            return pd.DataFrame(
                {
                    "rent": np.full(len(X_test), 1_000.0),
                    "real_estate_taxes": np.full(len(X_test), 250.0),
                }
            )

    class FakeMicrosimulation:
        def __init__(self, dataset):
            self.dataset = dataset

        def calculate_dataframe(self, columns):
            if "rent" in columns:
                df = pd.DataFrame(
                    {
                        "is_household_head": np.ones(10_000, dtype=bool),
                        "age": np.full(10_000, 40),
                        "is_male": np.zeros(10_000, dtype=bool),
                        "tenure_type": ["RENTED"] * 10_000,
                        "employment_income": np.full(10_000, 30_000.0),
                        "self_employment_income": np.zeros(10_000),
                        "social_security": np.zeros(10_000),
                        "pension_income": np.zeros(10_000),
                        "state_code_str": ["CA"] * 10_000,
                        "household_size": np.ones(10_000),
                        "rent": np.full(10_000, 1_200.0),
                        "real_estate_taxes": np.zeros(10_000),
                    }
                )
            else:
                df = pd.DataFrame(
                    {
                        "is_household_head": [True],
                        "age": [40],
                        "is_male": [False],
                        "tenure_type": ["RENTED"],
                        "employment_income": [30_000.0],
                        "self_employment_income": [0.0],
                        "social_security": [0.0],
                        "pension_income": [0.0],
                        "state_code_str": ["CA"],
                        "household_size": [1],
                    }
                )
            return df[columns]

    class FakeDataset:
        def __init__(self):
            self.file_path = existing_path
            self.saved = []

        def save_dataset(self, data):
            self.saved.append(data)

    monkeypatch.setattr(cps_module.pd, "HDFStore", recording_hdfstore)
    monkeypatch.setattr(cps_module.h5py, "File", fail_h5py_file)
    monkeypatch.setattr(cps_module, "QRF", FakeQRF)

    import policyengine_us
    import policyengine_us_data.datasets.acs.acs as acs_module

    monkeypatch.setattr(policyengine_us, "Microsimulation", FakeMicrosimulation)
    monkeypatch.setattr(acs_module, "ACS_2022", object())

    dataset = FakeDataset()
    cps = {
        "age": np.array([40], dtype=np.int32),
        "spm_unit_capped_housing_subsidy_reported": np.array([0.0]),
    }
    person = pd.DataFrame({"dummy": [1]})
    household = pd.DataFrame({"H_TENURE": [2]})

    add_rent(dataset, cps, person, household)

    assert opened_modes == ["r"]
    assert not existing_path.exists()
    np.testing.assert_array_equal(cps["rent"], np.array([1_000.0]))
    np.testing.assert_array_equal(cps["real_estate_taxes"], np.array([250.0]))
