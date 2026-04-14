import pandas as pd


def test_add_tips_derives_tipped_status_from_raw_cps(monkeypatch):
    import policyengine_us
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

        def save_dataset(self, data):
            self.saved_dataset = data

    class FakeMicrosimulation:
        def __init__(self, dataset):
            self.dataset = dataset

        def calculate_dataframe(self, columns, year):
            base = pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "household_id": [10, 20],
                    "employment_income": [25_000, 30_000],
                    "interest_income": [0.0, 0.0],
                    "dividend_income": [0.0, 0.0],
                    "rental_income": [0.0, 0.0],
                    "age": [30, 45],
                    "household_weight": [1.0, 1.0],
                    "is_female": [False, True],
                }
            )
            return base[columns]

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

    monkeypatch.setattr(policyengine_us, "Microsimulation", FakeMicrosimulation)
    monkeypatch.setattr(sipp_module, "get_tip_model", lambda: FakeTipModel())
    monkeypatch.setattr(sipp_module, "get_asset_model", lambda: FakeAssetModel())

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
