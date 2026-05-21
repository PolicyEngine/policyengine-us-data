import h5py
import numpy as np
import pandas as pd


def test_drop_persisted_dataset_variables_removes_stale_h5_keys(tmp_path):
    from policyengine_us_data.datasets.cps.cps import (
        _drop_persisted_dataset_variables,
    )

    file_path = tmp_path / "cps_2024.h5"
    with h5py.File(file_path, "w") as h5_file:
        h5_file.create_dataset("snap_reported", data=np.array([1_200.0]))
        h5_file.create_dataset("ssi_reported", data=np.array([600.0]))
        h5_file.create_dataset("social_security_retirement", data=np.array([8_000.0]))

    _drop_persisted_dataset_variables(
        file_path,
        ("snap_reported", "ssi_reported"),
    )

    with h5py.File(file_path, "r") as h5_file:
        assert "snap_reported" not in h5_file
        assert "ssi_reported" not in h5_file
        assert h5_file["social_security_retirement"][:].tolist() == [8_000.0]


def test_add_takeup_removes_temporary_source_anchors_from_saved_h5(
    monkeypatch,
    tmp_path,
):
    import policyengine_us
    import policyengine_us_data.datasets.cps.cps as cps_module
    import policyengine_us_data.db.etl_pregnancy as etl_pregnancy
    from policyengine_us_data.datasets.cps.cps import add_takeup

    class FakeResult:
        def __init__(self, values):
            self.values = np.asarray(values)

    class FakeMicrosimulation:
        def __init__(self, dataset):
            self.dataset = dataset

        def calculate(self, variable_name):
            values_by_variable = {
                "eitc_child_count": [0],
                "eitc": [0],
                "state_code_str": ["CA"],
                "wic_category_str": ["NONE", "NONE"],
                "receives_wic": [False, False],
                "hud_income_level": ["VERY_LOW"],
                "spm_unit_tenure_type": ["RENTER"],
                "is_eligible_for_housing_assistance": [True],
                "tax_unit_child_dependents": [0],
                "age_head": [40],
            }
            return FakeResult(values_by_variable[variable_name])

    class FakeDataset:
        def __init__(self, file_path):
            self.file_path = file_path
            self.time_period = 2024
            self.data = {
                "person_id": np.array([1, 2], dtype=np.int32),
                "tax_unit_id": np.array([10], dtype=np.int32),
                "spm_unit_id": np.array([100], dtype=np.int32),
                "household_id": np.array([1_000], dtype=np.int32),
                "person_tax_unit_id": np.array([10, 10], dtype=np.int32),
                "person_household_id": np.array([1_000, 1_000], dtype=np.int32),
                "snap_reported": np.array([1_200.0], dtype=np.float32),
                "ssi_reported": np.array([600.0, 0.0], dtype=np.float32),
                "receives_housing_assistance": np.array([True]),
                "reported_has_subsidized_marketplace_health_coverage_at_interview": np.array(
                    [False, False]
                ),
                "has_medicaid_health_coverage_at_interview": np.array([False, False]),
                "employment_income": np.array([20_000.0, 0.0], dtype=np.float32),
                "age": np.array([40, 66], dtype=np.int32),
                "is_female": np.array([True, False]),
                "is_disabled": np.array([False, False]),
            }

        def load_dataset(self):
            return {name: values.copy() for name, values in self.data.items()}

        def save_dataset(self, data):
            with h5py.File(self.file_path, "a") as h5_file:
                for name, values in data.items():
                    if name in h5_file:
                        del h5_file[name]
                    h5_file.create_dataset(name, data=values)

    voluntary_filing_rates = {
        children: {
            wage: {"under_65": 0.0, "age_65_plus": 0.0}
            for wage in ("zero", "low", "medium", "high")
        }
        for children in ("with_children", "no_children")
    }
    rates = {
        "eitc": {0: 0.0},
        "dc_ptc": 0.0,
        "snap": 1.0,
        "aca": 0.0,
        "medicaid": {"CA": 0.0},
        "head_start": 0.0,
        "early_head_start": 0.0,
        "ssi": 1.0,
        "housing_assistance": 0.0,
        "voluntary_filing": voluntary_filing_rates,
        "tanf": 0.0,
        "wic_takeup": {"NONE": 0.0},
        "wic_nutritional_risk": {"NONE": 0.0},
    }

    monkeypatch.setattr(policyengine_us, "Microsimulation", FakeMicrosimulation)
    monkeypatch.setattr(cps_module, "load_take_up_rate", lambda name, year: rates[name])
    monkeypatch.setattr(
        etl_pregnancy,
        "get_state_pregnancy_rates",
        lambda cdc_year, acs_year: {"CA": 0.0},
    )

    file_path = tmp_path / "cps_2024.h5"
    with h5py.File(file_path, "w") as h5_file:
        h5_file.create_dataset("snap_reported", data=np.array([1_200.0]))
        h5_file.create_dataset("ssi_reported", data=np.array([600.0, 0.0]))

    add_takeup(FakeDataset(file_path))

    with h5py.File(file_path, "r") as h5_file:
        assert "snap_reported" not in h5_file
        assert "ssi_reported" not in h5_file
        assert "takes_up_snap_if_eligible" in h5_file
        assert "takes_up_ssi_if_eligible" in h5_file
        assert h5_file["takes_up_housing_assistance_if_eligible"][:].tolist() == [True]


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

    class FakeSsiDisabilityModel:
        pass

    def fake_predict_ssi_disability_criteria(model, receiver_df):
        assert isinstance(model, FakeSsiDisabilityModel)
        assert receiver_df["employment_income"].tolist() == [25_000.0, 30_000.0]
        return np.array([True, False])

    monkeypatch.setattr(sipp_module, "get_tip_model", lambda: FakeTipModel())
    monkeypatch.setattr(sipp_module, "get_asset_model", lambda: FakeAssetModel())
    monkeypatch.setattr(sipp_module, "get_vehicle_model", lambda: FakeVehicleModel())
    monkeypatch.setattr(
        sipp_module,
        "get_ssi_disability_model",
        lambda: FakeSsiDisabilityModel(),
    )
    monkeypatch.setattr(
        sipp_module,
        "predict_ssi_disability_criteria",
        fake_predict_ssi_disability_criteria,
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
    assert dataset.saved_dataset["meets_ssi_disability_criteria"].tolist() == [
        True,
        False,
    ]


def test_add_rent_requests_person_level_frames(monkeypatch, tmp_path):
    import policyengine_us
    import policyengine_us_data.datasets.acs.acs as acs_module
    from policyengine_us_data.datasets.cps.cps import add_rent

    fake_acs_path = tmp_path / "acs_2022.h5"
    with h5py.File(fake_acs_path, "w") as fake_acs_h5:
        fake_acs_h5.create_dataset(
            "is_household_head",
            data=np.ones(10_050, dtype=bool),
        )

    class FakeACSDataset:
        file_path = fake_acs_path

    fake_acs_dataset = FakeACSDataset()
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
        "is_household_head": np.array([True, False, True], dtype=bool),
        "spm_unit_id": np.array([1, 2, 3], dtype=np.int32),
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


def test_add_spm_variables_keeps_spm_output_aggregates_out_of_dataset():
    from policyengine_us_data.datasets.cps.cps import add_spm_variables

    cps = {}
    spm_unit = pd.DataFrame(
        {
            "SPM_TOTVAL": [50_000],
            "SPM_RESOURCES": [45_000],
            "SPM_SNAPSUB": [1_200],
            "SPM_CAPHOUSESUB": [3_000],
            "SPM_ENGVAL": [500],
            "SPM_SCHLUNCH": [800],
            "SPM_WICVAL": [200],
            "SPM_BBSUBVAL": [360],
            "SPM_FICA": [3_825],
            "SPM_FEDTAX": [2_000],
            "SPM_STTAX": [1_000],
            "SPM_CAPWKCCXPNS": [4_000],
            "SPM_CHILDCAREXPNS": [4_500],
            "SPM_TENMORTSTATUS": [3],
        }
    )

    add_spm_variables(None, cps, spm_unit)

    assert "spm_unit_total_income_reported" not in cps
    assert "spm_unit_net_income_reported" not in cps
    assert cps["snap_reported"].tolist() == [1_200]
    assert "spm_unit_capped_housing_subsidy" not in cps
    assert "housing_assistance" not in cps
    assert cps["receives_housing_assistance"].tolist() == [True]
    assert cps["spm_unit_energy_subsidy"].tolist() == [500]
    assert cps["spm_unit_tenure_type"].tolist() == [b"RENTER"]
    for variable in (
        "free_school_meals_reported",
        "reduced_price_school_meals_reported",
        "spm_unit_wic_reported",
        "spm_unit_broadband_subsidy_reported",
        "spm_unit_payroll_tax_reported",
        "spm_unit_federal_tax_reported",
        "spm_unit_state_tax_reported",
    ):
        assert variable not in cps
