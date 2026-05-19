import io
import zipfile

import pytest

from policyengine_us_data.utils import bea_regional


def _sainc4_csv(state_code, state_fips, values_by_line):
    lines = [
        (
            "GeoFIPS,GeoName,Region,TableName,LineCode,"
            "IndustryClassification,Description,Unit,2023,2024"
        )
    ]
    for line_code, value in values_by_line.items():
        lines.append(
            f'"{state_fips:02d}000","{state_code}",0,SAINC4,{line_code},'
            f'"...","line {line_code}","Millions of dollars",0,{value}'
        )
    return "\n".join(lines)


def _zip_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for name, content in files.items():
            zip_file.writestr(name, content)
    return buffer.getvalue()


def test_bea_state_wages_residence_adjust_and_scale(monkeypatch):
    monkeypatch.setattr(bea_regional, "STATE_CODES", {6: "CA", 36: "NY"})
    zip_data = _zip_bytes(
        {
            "SAINC4_CA_1929_2025.csv": _sainc4_csv(
                "CA",
                6,
                {
                    36: 10,
                    42: 30,
                    50: 100,
                    60: 20,
                },
            ),
            "SAINC4_NY_1929_2025.csv": _sainc4_csv(
                "NY",
                36,
                {
                    36: 20,
                    42: -10,
                    50: 200,
                    60: 30,
                },
            ),
            "SAINC4_US_1929_2025.csv": _sainc4_csv(
                "US",
                0,
                {
                    36: 30,
                    42: 0,
                    50: 300,
                    60: 50,
                },
            ),
        }
    )

    wages, data_year = bea_regional.extract_bea_state_wages_and_salaries(
        2024,
        zip_bytes=zip_data,
    )
    targets = bea_regional.scale_state_wages_to_national_total(
        wages,
        national_total=1_000,
    )

    assert data_year == 2024
    assert targets["state_code"].tolist() == ["CA", "NY"]
    assert targets.loc[0, "wages_and_salaries"] == pytest.approx(
        100_000_000 + 30_000_000 * 100 / 130
    )
    assert targets.loc[1, "wages_and_salaries"] == pytest.approx(
        200_000_000 - 10_000_000 * 200 / 250
    )
    assert targets["employment_income_before_lsr"].sum() == pytest.approx(1_000)


def test_bea_state_wages_rejects_missing_state(monkeypatch):
    monkeypatch.setattr(bea_regional, "STATE_CODES", {6: "CA", 36: "NY"})
    zip_data = _zip_bytes(
        {
            "SAINC4_CA_1929_2025.csv": _sainc4_csv(
                "CA",
                6,
                {
                    36: 10,
                    42: 0,
                    50: 100,
                    60: 20,
                },
            ),
        }
    )

    with pytest.raises(ValueError, match="missing states"):
        bea_regional.extract_bea_state_wages_and_salaries(
            2024,
            zip_bytes=zip_data,
        )
