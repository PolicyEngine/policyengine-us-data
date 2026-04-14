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
                "interest_income": [0.0, 0.0],
                "dividend_income": [0.0, 0.0],
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
                    "household_vehicles_debt": [6_000.0, 500.0],
                }
            )

    class FakeNonliquidAssetModel:
        def predict(self, X_test, mean_quantile):
            assert X_test["household_id"].tolist() == [10, 20]
            return pd.DataFrame(
                {
                    "household_other_real_estate_value": [25_000.0, 0.0],
                    "household_other_real_estate_debt": [5_000.0, 0.0],
                    "household_rental_property_value": [40_000.0, 0.0],
                    "household_rental_property_debt": [10_000.0, 0.0],
                    "household_business_assets_value": [8_000.0, 2_000.0],
                    "household_business_assets_debt": [1_000.0, 500.0],
                }
            )

    monkeypatch.setattr(sipp_module, "get_tip_model", lambda: FakeTipModel())
    monkeypatch.setattr(sipp_module, "get_asset_model", lambda: FakeAssetModel())
    monkeypatch.setattr(sipp_module, "get_vehicle_model", lambda: FakeVehicleModel())
    monkeypatch.setattr(
        sipp_module,
        "get_nonliquid_asset_model",
        lambda: FakeNonliquidAssetModel(),
    )

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
    assert dataset.saved_dataset["household_vehicles_debt"].tolist() == [
        6_000.0,
        500.0,
    ]
    assert dataset.saved_dataset["household_vehicles_equity"].tolist() == [
        12_000.0,
        7_000.0,
    ]
    assert dataset.saved_dataset["household_other_real_estate_value"].tolist() == [
        25_000.0,
        0.0,
    ]
    assert dataset.saved_dataset["household_other_real_estate_debt"].tolist() == [
        5_000.0,
        0.0,
    ]
    assert dataset.saved_dataset["household_other_real_estate_equity"].tolist() == [
        20_000.0,
        0.0,
    ]
    assert dataset.saved_dataset["household_rental_property_equity"].tolist() == [
        30_000.0,
        0.0,
    ]
    assert dataset.saved_dataset["household_business_assets_equity"].tolist() == [
        7_000.0,
        1_500.0,
    ]
