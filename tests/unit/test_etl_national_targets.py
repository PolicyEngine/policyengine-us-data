import pandas as pd
from sqlmodel import Session

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)
from policyengine_us_data.db.etl_national_targets import (
    load_national_targets,
)


def _make_stratum(session, parent_id=None, notes=None, constraints=None):
    stratum = Stratum(parent_stratum_id=parent_id, notes=notes)
    stratum.constraints_rel = constraints or []
    session.add(stratum)
    session.commit()
    session.refresh(stratum)
    return stratum


def test_load_national_targets_deactivates_stale_baseline_rows(tmp_path, monkeypatch):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    db_uri = f"sqlite:///{calibration_dir / 'policy_data.db'}"
    engine = create_database(db_uri)

    with Session(engine) as session:
        national = _make_stratum(session, notes="United States")
        filer = _make_stratum(
            session,
            parent_id=national.stratum_id,
            notes="United States - Tax Filers",
            constraints=[
                StratumConstraint(
                    constraint_variable="tax_unit_is_filer",
                    operation="==",
                    value="1",
                )
            ],
        )
        itemizer = _make_stratum(
            session,
            parent_id=national.stratum_id,
            notes="United States - Itemizing Tax Filers",
            constraints=[
                StratumConstraint(
                    constraint_variable="tax_unit_is_filer",
                    operation="==",
                    value="1",
                ),
                StratumConstraint(
                    constraint_variable="tax_unit_itemizes",
                    operation="==",
                    value="1",
                ),
            ],
        )

        session.add(
            Target(
                stratum_id=filer.stratum_id,
                variable="qualified_business_income_deduction",
                period=2024,
                value=63.1e9,
                active=True,
                reform_id=0,
            )
        )
        session.add(
            Target(
                stratum_id=itemizer.stratum_id,
                variable="salt_deduction",
                period=2024,
                value=21.247e9,
                active=True,
                reform_id=0,
            )
        )
        session.commit()

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_national_targets.STORAGE_FOLDER",
        tmp_path,
    )

    tax_expenditure_df = pd.DataFrame(
        [
            {
                "reform_id": 1,
                "variable": "salt_deduction",
                "value": 21.247e9,
                "source": "Joint Committee on Taxation",
                "notes": "SALT deduction tax expenditure",
                "year": 2024,
            },
            {
                "reform_id": 5,
                "variable": "qualified_business_income_deduction",
                "value": 63.1e9,
                "source": "Joint Committee on Taxation",
                "notes": "QBI deduction tax expenditure",
                "year": 2024,
            },
        ]
    )

    load_national_targets(
        direct_targets_df=pd.DataFrame(),
        tax_filer_df=pd.DataFrame(),
        tax_expenditure_df=tax_expenditure_df,
        conditional_targets=[],
    )
    load_national_targets(
        direct_targets_df=pd.DataFrame(),
        tax_filer_df=pd.DataFrame(),
        tax_expenditure_df=tax_expenditure_df,
        conditional_targets=[],
    )

    with Session(engine) as session:
        stale_rows = session.query(Target).filter(Target.reform_id == 0).all()
        assert stale_rows
        assert all(not target.active for target in stale_rows)

        reform_rows = session.query(Target).filter(Target.reform_id > 0).all()
        assert len(reform_rows) == 2
        assert all(target.active for target in reform_rows)
        assert {(target.variable, target.reform_id) for target in reform_rows} == {
            ("salt_deduction", 1),
            ("qualified_business_income_deduction", 5),
        }
        assert all(
            "Modeled as repeal-based income tax expenditure target"
            in (target.notes or "")
            for target in reform_rows
        )
