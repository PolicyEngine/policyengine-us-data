import pandas as pd
import pytest

from policyengine_us_data.db import etl_state_income_tax as stc_module


def test_extract_state_income_tax_data_parses_census_t40(monkeypatch):
    mapping = {
        "02": "AK",
        "06": "CA",
        "33": "NH",
        "47": "TN",
        "53": "WA",
    }
    monkeypatch.setattr(stc_module, "STATE_FIPS_TO_ABBREV", mapping)
    monkeypatch.setattr(
        stc_module,
        "STATE_ABBREV_TO_FIPS",
        {abbrev: fips for fips, abbrev in mapping.items()},
    )
    monkeypatch.setattr(stc_module, "is_cached", lambda _: False)

    saved = {}

    def fake_save_json(filename, data):
        saved["filename"] = filename
        saved["data"] = data

    monkeypatch.setattr(stc_module, "save_json", fake_save_json)

    t40_row = {
        "ITEM": "T40",
        "AK": "X",
        "CA": "96379294",
        "NH": "149485",
        "TN": "2926",
        "WA": "846835",
    }
    monkeypatch.setattr(
        stc_module.pd,
        "read_csv",
        lambda url, dtype=str: pd.DataFrame(
            [
                {"ITEM": "T00"},
                t40_row,
            ]
        ),
    )

    df = stc_module.extract_state_income_tax_data(2023)
    actual = dict(zip(df["state_abbrev"], df["income_tax_collections"]))

    assert actual == {
        "AK": 0,
        "CA": 96_379_294_000,
        "NH": 149_485_000,
        "TN": 2_926_000,
        "WA": 846_835_000,
    }
    assert saved["filename"] == "census_stc_t40_individual_income_tax_2023.json"
    assert saved["data"] == df.to_dict(orient="records")


def test_extract_state_income_tax_data_rejects_unsupported_year():
    with pytest.raises(ValueError, match="Only years"):
        stc_module.extract_state_income_tax_data(2022)
