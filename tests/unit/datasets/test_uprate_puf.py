import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "policyengine_us_data"


def load_uprate_puf_module(storage_root: Path):
    for name in [
        "policyengine_us_data.datasets.puf.uprate_puf",
        "policyengine_us_data.datasets.puf",
        "policyengine_us_data.datasets",
        "policyengine_us_data.storage",
        "policyengine_us_data",
    ]:
        sys.modules.pop(name, None)

    package = types.ModuleType("policyengine_us_data")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["policyengine_us_data"] = package

    datasets_package = types.ModuleType("policyengine_us_data.datasets")
    datasets_package.__path__ = [str(PACKAGE_ROOT / "datasets")]
    sys.modules["policyengine_us_data.datasets"] = datasets_package

    puf_package = types.ModuleType("policyengine_us_data.datasets.puf")
    puf_package.__path__ = [str(PACKAGE_ROOT / "datasets" / "puf")]
    sys.modules["policyengine_us_data.datasets.puf"] = puf_package

    storage_spec = importlib.util.spec_from_file_location(
        "policyengine_us_data.storage",
        PACKAGE_ROOT / "storage" / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT / "storage")],
    )
    storage_module = importlib.util.module_from_spec(storage_spec)
    assert storage_spec.loader is not None
    sys.modules["policyengine_us_data.storage"] = storage_module
    storage_spec.loader.exec_module(storage_module)
    storage_module.STORAGE_FOLDER = storage_root
    storage_module.CALIBRATION_FOLDER = storage_root / "calibration_targets"

    uprate_spec = importlib.util.spec_from_file_location(
        "policyengine_us_data.datasets.puf.uprate_puf",
        PACKAGE_ROOT / "datasets" / "puf" / "uprate_puf.py",
    )
    uprate_module = importlib.util.module_from_spec(uprate_spec)
    assert uprate_spec.loader is not None
    sys.modules["policyengine_us_data.datasets.puf.uprate_puf"] = uprate_module
    uprate_spec.loader.exec_module(uprate_module)
    return uprate_module


def write_soi_targets(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Year": 2021,
                "Variable": "employment_income",
                "Filing status": "All",
                "AGI lower bound": float("-inf"),
                "AGI upper bound": float("inf"),
                "Count": False,
                "Taxable only": False,
                "Full population": True,
                "Value": 200.0,
            },
            {
                "Year": 2021,
                "Variable": "count",
                "Filing status": "All",
                "AGI lower bound": float("-inf"),
                "AGI upper bound": float("inf"),
                "Count": True,
                "Taxable only": False,
                "Full population": True,
                "Value": 100.0,
            },
        ]
    ).to_csv(path, index=False)


def test_get_soi_aggregate_falls_back_to_calibration_targets(tmp_path: Path):
    write_soi_targets(tmp_path / "calibration_targets" / "soi_targets.csv")
    module = load_uprate_puf_module(tmp_path)

    assert module.get_soi_aggregate("employment_income", 2021, False) == 200.0
    assert module.get_soi_aggregate("count", 2021, True) == 100.0


def test_get_soi_aggregate_raises_clear_error_when_missing(tmp_path: Path):
    module = load_uprate_puf_module(tmp_path)

    with pytest.raises(FileNotFoundError, match="soi_targets.csv"):
        module.load_soi_aggregates()
