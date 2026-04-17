import h5py
import numpy as np
import pandas as pd


def test_add_tips_derives_tipped_status_from_raw_cps(monkeypatch):
    import policyengine_us_data.datasets.sipp as sipp_module
    from policyengine_us_data.datasets.cps.cps import add_tips

    class FakeRawData:
        def __init__(self):
            self.person = pd.DataFrame(
                {
                    "A_MARITL": [1, 3],
                    "PEIOOCC": [4040, 9999],
                }
            )

        def __getitem__(self, key):
            if key == "person":
                return self.person
            raise KeyError(key)

        def close(self):
            pass

    class FakeRawCPS:
        def __call__(self, require=True):
            return self

        def load(self):
            return FakeRawData()

    class FakeDataset:
        def __init__(self):
            self.raw_cps = FakeRawCPS()
            self.saved_dataset = None
            self.base_dataset = {
                "person_id": [1, 2],
                "person_household_id": [10, 20],
                "employment_income": [25_000.0, 30_000.0],
                "taxable_interest_income": [100.0, 0.0],
                "tax_exempt_interest_income": [25.0, 0.0],
                "qualified_dividend_income": [40.0, 0.0],
                "non_qualified_dividend_income": [10.0, 0.0],
                "rental_income": [0.0, 0.0],
                "age": [30, 45],
                "household_weight": [1.0, 1.0],
                "is_female": [False, True],
                "is_household_head": [True, True],
                "tenure_type": [b"OWNED_WITH_MORTGAGE", b"RENTED"],
            }

        def save_dataset(self, data):
            if self.saved_dataset is None:
                self.saved_dataset = {}
            if hasattr(data, "items"):
                for key, value in data.items():
                    self.saved_dataset[key] = (
                        value.values if hasattr(value, "values") else value
                    )

        def load_dataset(self):
            return self.base_dataset

    class FakeTipModel:
        def predict(self, X_test, mean_quantile):
            assert X_test["is_tipped_occupation"].tolist() == [True, False]
            return pd.DataFrame({"tip_income": [100.0, 0.0]})

    class FakeAssetModel:
        def predict(self, X_test, mean_quantile):
            assert X_test["interest_income"].tolist() == [125.0, 0.0]
            assert X_test["dividend_income"].tolist() == [50.0, 0.0]
            return pd.DataFrame(
                {
                    "bank_account_assets": [0.0, 0.0],
                    "stock_assets": [0.0, 0.0],
                    "bond_assets": [0.0, 0.0],
                }
            )

    class FakeVehicleModel:
        def predict(self, X_test, mean_quantile):
            assert X_test["household_id"].tolist() == [10, 20]
            return pd.DataFrame(
                {
                    "household_vehicles_owned": [2.0, 1.0],
                    "household_vehicles_value": [18_000.0, 7_500.0],
                }
            )

    monkeypatch.setattr(sipp_module, "get_tip_model", lambda: FakeTipModel())
    monkeypatch.setattr(sipp_module, "get_asset_model", lambda: FakeAssetModel())
    monkeypatch.setattr(sipp_module, "get_vehicle_model", lambda: FakeVehicleModel())

    dataset = FakeDataset()
    add_tips(
        dataset,
        {
            "person_spm_unit_id": [101, 202],
            "spm_unit_id": [101, 202],
        },
    )

    assert dataset.saved_dataset["tip_income"].tolist() == [100.0, 0.0]
    assert dataset.saved_dataset["bank_account_assets"].tolist() == [0.0, 0.0]
    assert dataset.saved_dataset["stock_assets"].tolist() == [0.0, 0.0]
    assert dataset.saved_dataset["bond_assets"].tolist() == [0.0, 0.0]
    assert dataset.saved_dataset["household_vehicles_owned"].tolist() == [2, 1]
    assert dataset.saved_dataset["household_vehicles_value"].tolist() == [
        18_000.0,
        7_500.0,
    ]


def test_add_rent_requests_person_level_frames(monkeypatch, tmp_path):
    import policyengine_us
    import policyengine_us_data.datasets.acs.acs as acs_module
    from policyengine_us_data.datasets.cps.cps import add_rent

    fake_acs_dataset = object()
    monkeypatch.setattr(acs_module, "ACS_2022", fake_acs_dataset)

    class FakeDataset:
        def __init__(self):
            self.file_path = tmp_path / "cps_2024.h5"
            self.saved_datasets = []

        def save_dataset(self, data):
            self.saved_datasets.append(data.copy())

    class FakeMicrosimulation:
        calls = []

        def __init__(self, dataset):
            self.dataset = dataset

        def calculate_dataframe(
            self, columns, period=None, map_to=None, use_weights=True
        ):
            FakeMicrosimulation.calls.append((self.dataset, tuple(columns), map_to))
            if self.dataset is fake_acs_dataset:
                rows = 10_050
                return pd.DataFrame(
                    {
                        "is_household_head": [True] * rows,
                        "age": np.full(rows, 45, dtype=np.int32),
                        "is_male": np.ones(rows, dtype=bool),
                        "tenure_type": np.array(["RENTED"] * rows),
                        "employment_income": np.full(rows, 50_000, dtype=np.int32),
                        "self_employment_income": np.zeros(rows, dtype=np.int32),
                        "social_security": np.zeros(rows, dtype=np.int32),
                        "pension_income": np.zeros(rows, dtype=np.int32),
                        "state_code_str": np.array(["CA"] * rows),
                        "household_size": np.full(rows, 2, dtype=np.int32),
                        "rent": np.full(rows, 1_500, dtype=np.int32),
                        "real_estate_taxes": np.zeros(rows, dtype=np.int32),
                    }
                )[list(columns)]

            return pd.DataFrame(
                {
                    "is_household_head": [True, False, True],
                    "age": [40, 12, 70],
                    "is_male": [True, False, False],
                    "tenure_type": ["RENTED", "NONE", "OWNED_WITH_MORTGAGE"],
                    "employment_income": [60_000, 0, 10_000],
                    "self_employment_income": [0, 0, 0],
                    "social_security": [0, 0, 8_000],
                    "pension_income": [0, 0, 2_000],
                    "state_code_str": ["CA", "CA", "NY"],
                    "household_size": [2, 2, 1],
                }
            )[list(columns)]

    class FakeQRFModel:
        def predict(self, X_test):
            assert len(X_test) == 2
            return pd.DataFrame(
                {
                    "rent": [1_200.0, 0.0],
                    "real_estate_taxes": [0.0, 4_000.0],
                }
            )

    class FakeQRF:
        def fit(self, X_train, predictors, imputed_variables):
            assert len(X_train) == 10_000
            assert predictors[-1] == "household_size"
            assert imputed_variables == ["rent", "real_estate_taxes"]
            return FakeQRFModel()

    monkeypatch.setattr(policyengine_us, "Microsimulation", FakeMicrosimulation)
    monkeypatch.setattr("policyengine_us_data.datasets.cps.cps.QRF", FakeQRF)

    dataset = FakeDataset()
    with h5py.File(dataset.file_path, "w") as stale:
        stale.create_dataset("stale_var", data=np.array([1], dtype=np.int8))

    cps = {
        "age": np.array([40, 12, 70], dtype=np.int32),
        "spm_unit_capped_housing_subsidy_reported": np.zeros(3, dtype=np.float32),
    }
    person = pd.DataFrame({"P_SEQ": [1, 2, 1]})
    household = pd.DataFrame({"H_TENURE": [2, 1]})

    add_rent(dataset, cps, person, household)

    assert [call[2] for call in FakeMicrosimulation.calls] == ["person", "person"]
    np.testing.assert_array_equal(cps["rent"], np.array([1200, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        cps["real_estate_taxes"],
        np.array([0, 0, 4000], dtype=np.int32),
    )
    assert not dataset.file_path.exists()
