import pandas as pd
from sqlmodel import Session, create_engine, select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)
from policyengine_us_data.db import etl_bea_state_wages


def _make_stratum(session, *, notes, constraints=None):
    stratum = Stratum(notes=notes)
    stratum.constraints_rel = constraints or []
    session.add(stratum)
    session.commit()
    session.refresh(stratum)
    return stratum


def test_load_bea_state_wage_targets_upserts_state_rows(tmp_path, monkeypatch):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    db_path = calibration_dir / "policy_data.db"
    create_database(f"sqlite:///{db_path}")
    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        _make_stratum(session, notes="United States")
        _make_stratum(
            session,
            notes="California",
            constraints=[
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value="6",
                )
            ],
        )

    monkeypatch.setattr(etl_bea_state_wages, "STORAGE_FOLDER", tmp_path)

    loaded = etl_bea_state_wages.load_bea_state_wage_targets(
        pd.DataFrame(
            [
                {
                    "state_fips": 6,
                    "state_code": "CA",
                    "wages_and_salaries": 1_767_800_000_000.0,
                    "scale_factor": 1.0005,
                    "employment_income_before_lsr": 1_768_683_900_000.0,
                }
            ]
        ),
        target_year=2024,
        source_year=2024,
    )

    with Session(engine) as session:
        target = session.exec(
            select(Target).where(
                Target.variable == "employment_income_before_lsr",
                Target.period == 2024,
            )
        ).one()
        stratum = session.get(Stratum, target.stratum_id)

    assert loaded == 1
    assert target.value == 1_768_683_900_000.0
    assert target.source == "BEA Regional SAINC4"
    assert "residence basis" in target.notes
    assert stratum.notes == "California"
