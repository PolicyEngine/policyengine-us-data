import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "policyengine_us_data"


def load_soi_module():
    for name in [
        "policyengine_us_data.utils.soi",
        "policyengine_us_data.utils.uprating",
        "policyengine_us_data.utils",
        "policyengine_us_data.storage",
        "policyengine_us_data",
    ]:
        sys.modules.pop(name, None)

    package = types.ModuleType("policyengine_us_data")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["policyengine_us_data"] = package

    utils_package = types.ModuleType("policyengine_us_data.utils")
    utils_package.__path__ = [str(PACKAGE_ROOT / "utils")]
    sys.modules["policyengine_us_data.utils"] = utils_package

    storage_spec = importlib.util.spec_from_file_location(
        "policyengine_us_data.storage",
        PACKAGE_ROOT / "storage" / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT / "storage")],
    )
    storage_module = importlib.util.module_from_spec(storage_spec)
    assert storage_spec.loader is not None
    sys.modules["policyengine_us_data.storage"] = storage_module
    storage_spec.loader.exec_module(storage_module)

    uprating_spec = importlib.util.spec_from_file_location(
        "policyengine_us_data.utils.uprating",
        PACKAGE_ROOT / "utils" / "uprating.py",
    )
    uprating_module = importlib.util.module_from_spec(uprating_spec)
    assert uprating_spec.loader is not None
    sys.modules["policyengine_us_data.utils.uprating"] = uprating_module
    uprating_spec.loader.exec_module(uprating_module)

    soi_spec = importlib.util.spec_from_file_location(
        "policyengine_us_data.utils.soi",
        PACKAGE_ROOT / "utils" / "soi.py",
    )
    soi_module = importlib.util.module_from_spec(soi_spec)
    assert soi_spec.loader is not None
    sys.modules["policyengine_us_data.utils.soi"] = soi_module
    soi_spec.loader.exec_module(soi_module)
    return soi_module


def test_get_soi_includes_mortgage_interest_deduction_targets():
    soi_module = load_soi_module()
    soi = soi_module.get_soi(2024)
    mortgage_interest = soi[soi.Variable == "mortgage_interest_deductions"]

    assert not mortgage_interest.empty
    assert mortgage_interest["Value"].gt(0).all()


def test_pe_to_soi_combines_sstb_and_non_sstb_schedule_c(monkeypatch):
    soi_module = load_soi_module()
    n = 2

    class FakeMicrosimulation:
        def __init__(self, dataset):
            self.dataset = dataset
            self.default_calculation_period = None

        def calculate(self, variable, map_to=None):
            values = {
                "self_employment_income": np.array([100.0, -10.0]),
                "sstb_self_employment_income": np.array([50.0, -25.0]),
                "filing_status": np.array(["SINGLE", "SINGLE"]),
                "tax_unit_weight": np.ones(n),
                "household_id": np.arange(1, n + 1),
            }
            return values.get(variable, np.zeros(n))

    fake_policyengine_us = types.ModuleType("policyengine_us")
    fake_policyengine_us.Microsimulation = FakeMicrosimulation
    monkeypatch.setitem(sys.modules, "policyengine_us", fake_policyengine_us)

    soi = soi_module.pe_to_soi(object(), 2024)

    np.testing.assert_array_equal(
        soi["business_net_profits"].to_numpy(), np.array([150.0, 0.0])
    )
    np.testing.assert_array_equal(
        soi["business_net_losses"].to_numpy(), np.array([0.0, 35.0])
    )


def test_get_soi_uses_best_available_year_per_variable(monkeypatch):
    soi_module = load_soi_module()
    fake_soi = pd.DataFrame(
        [
            {
                "Year": 2021,
                "Variable": "mortgage_interest_deductions",
                "Value": 100.0,
            },
            {
                "Year": 2023,
                "Variable": "mortgage_interest_deductions",
                "Value": 110.0,
            },
            {
                "Year": 2023,
                "Variable": "taxable_interest_income",
                "Value": 200.0,
            },
            {
                "Year": 2025,
                "Variable": "taxable_interest_income",
                "Value": 300.0,
            },
        ]
    )
    for column, default in {
        "SOI table": "Table 1.4",
        "XLSX column": "A",
        "XLSX row": 9,
        "Filing status": "All",
        "AGI lower bound": float("-inf"),
        "AGI upper bound": float("inf"),
        "Count": False,
        "Taxable only": False,
        "Full population": True,
    }.items():
        fake_soi[column] = default

    uprating = pd.DataFrame(
        {
            2021: [1.0, 1.0],
            2023: [1.5, 3.0],
            2024: [2.0, 4.0],
        },
        index=["interest_deduction", "taxable_interest_income"],
    )

    monkeypatch.setattr(soi_module, "load_tracked_soi_targets", lambda: fake_soi.copy())
    monkeypatch.setattr(
        soi_module,
        "create_policyengine_uprating_factors_table",
        lambda: uprating,
    )

    soi = soi_module.get_soi(2024)

    assert set(soi["Variable"]) == {
        "mortgage_interest_deductions",
        "taxable_interest_income",
    }
    mortgage_value = soi.loc[
        soi["Variable"] == "mortgage_interest_deductions", "Value"
    ].iat[0]
    taxable_interest_value = soi.loc[
        soi["Variable"] == "taxable_interest_income", "Value"
    ].iat[0]

    assert np.isclose(mortgage_value, 146.6666666667)
    assert np.isclose(taxable_interest_value, 266.6666666667)


def test_get_tracked_soi_row_selects_requested_best_year(monkeypatch):
    soi_module = load_soi_module()
    fake_soi = pd.DataFrame(
        [
            {
                "Year": 2021,
                "Variable": "business_net_profits",
                "SOI table": "Table 1.4",
                "XLSX column": "AG",
                "XLSX row": 9,
                "Filing status": "All",
                "AGI lower bound": float("-inf"),
                "AGI upper bound": float("inf"),
                "Count": False,
                "Taxable only": False,
                "Full population": True,
                "Value": 10.0,
            },
            {
                "Year": 2023,
                "Variable": "business_net_profits",
                "SOI table": "Table 1.4",
                "XLSX column": "AG",
                "XLSX row": 9,
                "Filing status": "All",
                "AGI lower bound": float("-inf"),
                "AGI upper bound": float("inf"),
                "Count": False,
                "Taxable only": False,
                "Full population": True,
                "Value": 30.0,
            },
        ]
    )
    monkeypatch.setattr(soi_module, "load_tracked_soi_targets", lambda: fake_soi.copy())

    row_2022 = soi_module.get_tracked_soi_row("business_net_profits", 2022, count=False)
    row_2024 = soi_module.get_tracked_soi_row("business_net_profits", 2024, count=False)

    assert row_2022["Year"] == 2021
    assert row_2024["Year"] == 2023


def test_get_national_soi_aggregate_rows_filters_to_all_returns(monkeypatch):
    soi_module = load_soi_module()
    fake_soi = pd.DataFrame(
        [
            {
                "Year": 2023,
                "Variable": "business_net_profits",
                "SOI table": "Table 1.4",
                "XLSX column": "AG",
                "XLSX row": 9,
                "Filing status": "All",
                "AGI lower bound": float("-inf"),
                "AGI upper bound": float("inf"),
                "Count": False,
                "Taxable only": False,
                "Full population": True,
                "Value": 30.0,
            },
            {
                "Year": 2023,
                "Variable": "business_net_profits",
                "SOI table": "Table 1.4",
                "XLSX column": "AG",
                "XLSX row": 29,
                "Filing status": "All",
                "AGI lower bound": float("-inf"),
                "AGI upper bound": float("inf"),
                "Count": False,
                "Taxable only": True,
                "Full population": False,
                "Value": 25.0,
            },
        ]
    )
    monkeypatch.setattr(soi_module, "load_tracked_soi_targets", lambda: fake_soi.copy())

    result = soi_module.get_national_soi_aggregate_rows(2024)

    assert len(result) == 1
    assert result.iloc[0]["Value"] == 30.0
