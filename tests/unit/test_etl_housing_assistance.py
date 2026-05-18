import io

import pandas as pd
import pytest
from sqlalchemy import text
from sqlmodel import Session

from policyengine_us_data.db import etl_housing_assistance
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    create_database,
)


def _workbook_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


def _add_geo_strata(session: Session):
    national = Stratum(notes="United States")
    session.add(national)
    session.flush()

    for state_fips, name in [(1, "Alabama"), (6, "California")]:
        state = Stratum(
            parent_stratum_id=national.stratum_id,
            notes=name,
        )
        state.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value=str(state_fips),
            )
        ]
        session.add(state)
    session.commit()


def test_transform_hud_picture_state_data_filters_summary_states():
    raw = pd.DataFrame(
        {
            "program_label": [
                "Summary of All HUD Programs",
                "Public Housing",
                "Summary of All HUD Programs",
                "Summary of All HUD Programs",
            ],
            "State": ["AL", "AL", "CA", "PR"],
            "number_reported": [81_143, 25_762, 469_502, 96_625],
        }
    )

    result = etl_housing_assistance.transform_hud_picture_state_data(
        _workbook_bytes(raw)
    )

    assert result["ucgid_str"].tolist() == ["0400000US01", "0400000US06"]
    assert result["assisted_households"].tolist() == [81_143, 469_502]


def test_load_housing_assistance_data_creates_count_targets(tmp_path, monkeypatch):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    db_uri = f"sqlite:///{calibration_dir / 'policy_data.db'}"
    engine = create_database(db_uri)

    with Session(engine) as session:
        _add_geo_strata(session)

    monkeypatch.setattr(etl_housing_assistance, "STORAGE_FOLDER", tmp_path)
    state_df = pd.DataFrame(
        {
            "ucgid_str": ["0400000US01", "0400000US06"],
            "assisted_households": [81_143, 469_502],
        }
    )

    etl_housing_assistance.load_housing_assistance_data(state_df, 2024)

    with Session(engine) as session:
        rows = session.execute(
            text(
                """
                SELECT variable, value, geo_level, geographic_id, domain_variable
                FROM target_overview
                WHERE domain_variable = 'housing_assistance'
                ORDER BY geo_level, geographic_id
                """
            )
        ).fetchall()

    assert len(rows) == 3
    national = [row for row in rows if row.geo_level == "national"][0]
    assert national.variable == "household_count"
    assert national.value == pytest.approx(550_645)

    states = {
        int(row.geographic_id): row.value for row in rows if row.geo_level == "state"
    }
    assert states == {1: 81_143, 6: 469_502}
