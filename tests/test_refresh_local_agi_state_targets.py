import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "policyengine_us_data"
MODULE_PATH = (
    PACKAGE_ROOT
    / "storage"
    / "calibration_targets"
    / "refresh_local_agi_state_targets.py"
)
TRACKED_TARGET_PATH = PACKAGE_ROOT / "storage" / "calibration_targets" / "agi_state.csv"


def load_module():
    refresh_spec = importlib.util.spec_from_file_location(
        "refresh_local_agi_state_targets",
        MODULE_PATH,
    )
    refresh_module = importlib.util.module_from_spec(refresh_spec)
    assert refresh_spec.loader is not None
    refresh_spec.loader.exec_module(refresh_module)
    return refresh_module


def make_raw_state_soi():
    return pd.DataFrame(
        [
            {"STATE": "AL", "AGI_STUB": 0, "N1": 999, "A00100": 999},
            {"STATE": "AL", "AGI_STUB": 1, "N1": 10, "A00100": 5},
            {"STATE": "AL", "AGI_STUB": 9, "N1": 20, "A00100": 7},
            {"STATE": "AL", "AGI_STUB": 10, "N1": 30, "A00100": 11},
            {"STATE": "DC", "AGI_STUB": 1, "N1": 4, "A00100": 2},
            {"STATE": "DC", "AGI_STUB": 9, "N1": 6, "A00100": 3},
            {"STATE": "DC", "AGI_STUB": 10, "N1": 8, "A00100": 5},
            {"STATE": "PR", "AGI_STUB": 1, "N1": 100, "A00100": 100},
        ]
    )


def test_build_local_agi_state_targets_uses_local_loss_format():
    module = load_module()

    refreshed = module.build_local_agi_state_targets(make_raw_state_soi())

    assert list(refreshed.columns) == [
        "GEO_ID",
        "GEO_NAME",
        "AGI_LOWER_BOUND",
        "AGI_UPPER_BOUND",
        "VALUE",
        "IS_COUNT",
        "VARIABLE",
    ]
    assert set(refreshed["GEO_NAME"]) == {"AL", "DC"}
    assert "PR" not in set(refreshed["GEO_NAME"])
    assert (refreshed["GEO_NAME"].str.startswith("state_")).sum() == 0

    dc_rows = refreshed[refreshed["GEO_NAME"] == "DC"]
    assert set(dc_rows["GEO_ID"]) == {"0400000US11"}

    upper_mid_count = refreshed[
        (refreshed["GEO_NAME"] == "AL")
        & (refreshed["VARIABLE"] == "adjusted_gross_income/count")
        & (refreshed["AGI_LOWER_BOUND"] == 500_000)
        & (refreshed["AGI_UPPER_BOUND"] == 1_000_000)
    ]
    top_count = refreshed[
        (refreshed["GEO_NAME"] == "AL")
        & (refreshed["VARIABLE"] == "adjusted_gross_income/count")
        & (refreshed["AGI_LOWER_BOUND"] == 1_000_000)
        & np.isposinf(refreshed["AGI_UPPER_BOUND"])
    ]
    upper_mid_amount = refreshed[
        (refreshed["GEO_NAME"] == "AL")
        & (refreshed["VARIABLE"] == "adjusted_gross_income/amount")
        & (refreshed["AGI_LOWER_BOUND"] == 500_000)
        & (refreshed["AGI_UPPER_BOUND"] == 1_000_000)
    ]
    top_amount = refreshed[
        (refreshed["GEO_NAME"] == "AL")
        & (refreshed["VARIABLE"] == "adjusted_gross_income/amount")
        & (refreshed["AGI_LOWER_BOUND"] == 1_000_000)
        & np.isposinf(refreshed["AGI_UPPER_BOUND"])
    ]
    assert upper_mid_count["VALUE"].iat[0] == 20
    assert top_count["VALUE"].iat[0] == 30
    assert upper_mid_amount["VALUE"].iat[0] == 7_000
    assert top_amount["VALUE"].iat[0] == 11_000


def test_refresh_local_agi_state_targets_writes_expected_csv(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module,
        "_load_state_soi_raw",
        lambda tax_year=module.LOCAL_STATE_SOI_TAX_YEAR: make_raw_state_soi(),
    )

    output_path = tmp_path / "agi_state.csv"
    written_path = module.refresh_local_agi_state_targets(output_path)

    written = pd.read_csv(written_path)

    assert written_path == output_path
    assert written["GEO_ID"].isna().sum() == 0
    assert set(written["VARIABLE"]) == {
        "adjusted_gross_income/count",
        "adjusted_gross_income/amount",
    }


def test_tracked_agi_state_targets_have_complete_geo_ids():
    tracked = pd.read_csv(TRACKED_TARGET_PATH)

    assert tracked["GEO_ID"].isna().sum() == 0
    assert tracked["GEO_NAME"].nunique() == 51
    assert set(tracked["VARIABLE"]) == {
        "adjusted_gross_income/count",
        "adjusted_gross_income/amount",
    }
    assert set(tracked.groupby(["GEO_NAME", "VARIABLE"]).size()) == {10}
    assert (
        (tracked["AGI_LOWER_BOUND"] == 500_000)
        & (tracked["AGI_UPPER_BOUND"] == 1_000_000)
    ).any()
    assert (
        (tracked["AGI_LOWER_BOUND"] == 1_000_000)
        & np.isposinf(tracked["AGI_UPPER_BOUND"])
    ).any()
