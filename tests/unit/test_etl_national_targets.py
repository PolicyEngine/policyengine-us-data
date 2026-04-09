import pandas as pd
from sqlmodel import Session, select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)
from policyengine_us_data.db.etl_national_targets import (
    _get_aca_national_targets,
    _get_medicaid_national_targets,
    _retime_target_row,
    _select_best_available_target_row,
    load_national_targets,
)


def _make_stratum(session, parent_id=None, notes=None, constraints=None):
    stratum = Stratum(parent_stratum_id=parent_id, notes=notes)
    stratum.constraints_rel = constraints or []
    session.add(stratum)
    session.commit()
    session.refresh(stratum)
    return stratum


def test_retime_target_row_carries_forward_2024_values_to_2025():
    target = {
        "variable": "alimony_income",
        "value": 13e9,
        "source": "Survey-reported",
        "notes": "Alimony received",
        "year": 2024,
    }

    retimed = _retime_target_row(target, requested_year=2025)

    assert retimed["year"] == 2025
    assert retimed["value"] == target["value"]
    assert "Source year 2024 carried forward to 2025" in retimed["notes"]


def test_select_best_available_target_row_uses_latest_liheap_source_for_2025():
    liheap_targets = [
        {
            "constraint_variable": "spm_unit_energy_subsidy_reported",
            "target_variable": "household_count",
            "household_count": 5_939_605,
            "source": "2023 source",
            "notes": "LIHEAP total households served by state programs",
            "year": 2023,
        },
        {
            "constraint_variable": "spm_unit_energy_subsidy_reported",
            "target_variable": "household_count",
            "household_count": 5_876_646,
            "source": "2024 source",
            "notes": "LIHEAP total households served by state programs",
            "year": 2024,
        },
    ]

    selected = _select_best_available_target_row(liheap_targets, requested_year=2025)

    assert selected["year"] == 2025
    assert selected["household_count"] == 5_876_646
    assert "Source year 2024 carried forward to 2025" in selected["notes"]


def test_national_health_targets_use_2025_data_when_available():
    _, aca_enrollment, aca_data_year = _get_aca_national_targets(2025)
    _, medicaid_enrollment, medicaid_data_year = _get_medicaid_national_targets(2025)

    assert aca_data_year == 2025
    assert medicaid_data_year == 2025
    assert aca_enrollment > 0
    assert medicaid_enrollment > 0


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
        stale_rows = session.exec(select(Target).where(Target.reform_id == 0)).all()
        assert stale_rows
        assert all(not target.active for target in stale_rows)

        reform_rows = session.exec(select(Target).where(Target.reform_id > 0)).all()
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


def test_load_national_targets_supports_liheap_household_counts(tmp_path, monkeypatch):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    db_uri = f"sqlite:///{calibration_dir / 'policy_data.db'}"
    engine = create_database(db_uri)

    with Session(engine) as session:
        national = _make_stratum(session, notes="United States")
        assert national is not None

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_national_targets.STORAGE_FOLDER",
        tmp_path,
    )

    conditional_targets = [
        {
            "constraint_variable": "spm_unit_energy_subsidy_reported",
            "target_variable": "household_count",
            "household_count": 5_876_646,
            "source": "https://example.com/liheap-2024.pdf",
            "notes": "LIHEAP total households served by state programs",
            "year": 2024,
        }
    ]

    load_national_targets(
        direct_targets_df=pd.DataFrame(),
        tax_filer_df=pd.DataFrame(),
        tax_expenditure_df=pd.DataFrame(),
        conditional_targets=conditional_targets,
    )

    with Session(engine) as session:
        liheap_stratum = session.exec(
            select(Stratum).where(
                Stratum.notes == "National LIHEAP Recipient Households"
            )
        ).first()
        assert liheap_stratum is not None

        constraints = {
            (
                constraint.constraint_variable,
                constraint.operation,
                constraint.value,
            )
            for constraint in liheap_stratum.constraints_rel
        }
        assert ("spm_unit_energy_subsidy_reported", ">", "0") in constraints

        liheap_target = session.exec(
            select(Target).where(
                Target.stratum_id == liheap_stratum.stratum_id,
                Target.variable == "household_count",
                Target.period == 2024,
            )
        ).first()
        assert liheap_target is not None
        assert liheap_target.value == 5_876_646
