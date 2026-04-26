import pandas as pd
from sqlalchemy import text
from sqlmodel import Session, select

from policyengine_us_data.calibration.unified_matrix_builder import (
    UnifiedMatrixBuilder,
)
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)
from policyengine_us_data.db.etl_irs_soi import (
    GEOGRAPHY_FILE_TARGET_SPECS,
    get_geography_soi_year,
    get_national_geography_soi_agi_targets,
    get_national_geography_soi_target,
    _get_geography_file_aggregate_target_spec,
    _skip_coarse_state_agi_person_count_target,
    _get_or_create_national_domain_stratum,
    _upsert_target,
    load_national_eitc_agi_child_targets,
    load_national_geography_ctc_agi_targets,
    load_national_geography_ctc_targets,
    load_national_workbook_soi_targets,
)


def _create_test_engine(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'policy_data.db'}"
    engine = create_database(db_uri)
    return db_uri, engine


def _create_national_filer_stratum(session: Session) -> Stratum:
    stratum = Stratum(notes="National filers")
    stratum.constraints_rel = [
        StratumConstraint(
            constraint_variable="tax_unit_is_filer",
            operation="==",
            value="1",
        )
    ]
    session.add(stratum)
    session.commit()
    session.refresh(stratum)
    return stratum


def test_upsert_target_preserves_reform_rows(tmp_path):
    _, engine = _create_test_engine(tmp_path)

    with Session(engine) as session:
        stratum = _create_national_filer_stratum(session)
        session.add(
            Target(
                stratum_id=stratum.stratum_id,
                variable="aca_ptc",
                period=2023,
                reform_id=1,
                value=999.0,
                active=True,
                source="IRS SOI",
            )
        )
        session.commit()

        _upsert_target(
            session,
            stratum_id=stratum.stratum_id,
            variable="aca_ptc",
            period=2023,
            value=123.0,
            source="IRS SOI",
        )
        session.commit()

        targets = session.exec(
            select(Target)
            .where(
                Target.stratum_id == stratum.stratum_id,
                Target.variable == "aca_ptc",
                Target.period == 2023,
            )
            .order_by(Target.reform_id)
        ).all()

    assert [(target.reform_id, target.value) for target in targets] == [
        (0, 123.0),
        (1, 999.0),
    ]


def test_workbook_overlay_wins_best_period_selection(monkeypatch, tmp_path):
    db_uri, engine = _create_test_engine(tmp_path)

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_irs_soi.WORKBOOK_NATIONAL_DOMAIN_TARGETS",
        {"taxable_interest_income": "taxable_interest_income"},
    )

    def fake_get_tracked_soi_row(variable, requested_year, **kwargs):
        count = kwargs["count"]
        rows = {
            ("adjusted_gross_income", False): {
                "Year": 2023,
                "Value": 1_000_000.0,
                "SOI table": "Table 1.1",
            },
            ("taxable_interest_income", True): {
                "Year": 2023,
                "Value": 50.0,
                "SOI table": "Table 1.4",
            },
            ("taxable_interest_income", False): {
                "Year": 2023,
                "Value": 2_000.0,
                "SOI table": "Table 1.4",
            },
        }
        return pd.Series(rows[(variable, count)])

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_irs_soi.get_tracked_soi_row",
        fake_get_tracked_soi_row,
    )

    with Session(engine) as session:
        national_filer_stratum = _create_national_filer_stratum(session)
        domain_stratum = _get_or_create_national_domain_stratum(
            session,
            national_filer_stratum.stratum_id,
            "taxable_interest_income",
        )
        session.commit()
        session.refresh(domain_stratum)
        domain_stratum_id = domain_stratum.stratum_id

        session.add_all(
            [
                Target(
                    stratum_id=domain_stratum_id,
                    variable="tax_unit_count",
                    period=2022,
                    reform_id=0,
                    value=10.0,
                    active=True,
                    source="IRS SOI",
                ),
                Target(
                    stratum_id=domain_stratum_id,
                    variable="taxable_interest_income",
                    period=2022,
                    reform_id=0,
                    value=1_000.0,
                    active=True,
                    source="IRS SOI",
                ),
            ]
        )
        session.commit()

        load_national_workbook_soi_targets(
            session,
            national_filer_stratum.stratum_id,
            2024,
        )
        session.commit()

    builder = UnifiedMatrixBuilder(db_uri=db_uri, time_period=2024)

    amount_rows = builder._query_targets(
        {
            "stratum_ids": [domain_stratum_id],
            "variables": ["taxable_interest_income"],
        }
    )
    count_rows = builder._query_targets(
        {
            "stratum_ids": [domain_stratum_id],
            "variables": ["tax_unit_count"],
        }
    )

    assert len(amount_rows) == 1
    assert int(amount_rows.iloc[0]["period"]) == 2023
    assert float(amount_rows.iloc[0]["value"]) == 2_000.0

    assert len(count_rows) == 1
    assert int(count_rows.iloc[0]["period"]) == 2023
    assert float(count_rows.iloc[0]["value"]) == 50.0


def test_skip_coarse_state_agi_person_count_target_only_for_state_stub_9():
    assert _skip_coarse_state_agi_person_count_target("state", 9) is True
    assert _skip_coarse_state_agi_person_count_target("state", 8) is False
    assert _skip_coarse_state_agi_person_count_target("district", 9) is False
    assert _skip_coarse_state_agi_person_count_target("national", 9) is False


def test_get_geography_soi_year_uses_standard_lag_and_latest_release():
    assert get_geography_soi_year(2024) == 2022
    assert get_geography_soi_year(2023) == 2021
    assert get_geography_soi_year(2026) == 2022


def test_geography_file_aggregate_target_spec_reuses_shared_registry():
    spec = _get_geography_file_aggregate_target_spec("non_refundable_ctc")

    assert spec == {
        "code": "07225",
        "name": "non_refundable_ctc",
        "breakdown": None,
    }
    assert spec in GEOGRAPHY_FILE_TARGET_SPECS


def test_get_national_geography_soi_target_reads_amount_and_count(monkeypatch):
    fake_raw = pd.DataFrame(
        [
            {
                "STATE": "US",
                "agi_stub": 0,
                "N11070": 17.0,
                "A11070": 33.0,
                "N07225": 37.0,
                "A07225": 81.0,
            }
        ]
    )

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_irs_soi.extract_soi_data",
        lambda year: fake_raw,
    )

    refundable_target = get_national_geography_soi_target("refundable_ctc", 2024)
    non_refundable_target = get_national_geography_soi_target(
        "non_refundable_ctc",
        2024,
    )

    assert refundable_target["source_year"] == 2022
    assert refundable_target["count"] == 17.0
    assert refundable_target["amount"] == 33_000.0

    assert non_refundable_target["source_year"] == 2022
    assert non_refundable_target["count"] == 37.0
    assert non_refundable_target["amount"] == 81_000.0


def test_get_national_geography_soi_agi_targets_aggregates_state_rows(monkeypatch):
    fake_raw = pd.DataFrame(
        [
            {
                "STATE": "US",
                "CONG_DISTRICT": 0,
                "agi_stub": 0,
                "N11070": 99.0,
                "A11070": 999.0,
            },
            {
                "STATE": "CA",
                "CONG_DISTRICT": 0,
                "agi_stub": 1,
                "N11070": 10.0,
                "A11070": 20.0,
            },
            {
                "STATE": "NY",
                "CONG_DISTRICT": 0,
                "agi_stub": 1,
                "N11070": 3.0,
                "A11070": 7.0,
            },
            {
                "STATE": "CA",
                "CONG_DISTRICT": 0,
                "agi_stub": 2,
                "N11070": 8.0,
                "A11070": 11.0,
            },
            {
                "STATE": "CA",
                "CONG_DISTRICT": 12,
                "agi_stub": 2,
                "N11070": 100.0,
                "A11070": 100.0,
            },
        ]
    )

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_irs_soi.extract_soi_data",
        lambda year: fake_raw,
    )

    targets = get_national_geography_soi_agi_targets("refundable_ctc", 2024)

    assert targets == [
        {
            "variable": "refundable_ctc",
            "source_year": 2022,
            "agi_stub": 1,
            "agi_lower_bound": float("-inf"),
            "agi_upper_bound": 1.0,
            "count": 13.0,
            "amount": 27_000.0,
        },
        {
            "variable": "refundable_ctc",
            "source_year": 2022,
            "agi_stub": 2,
            "agi_lower_bound": 1.0,
            "agi_upper_bound": 10_000.0,
            "count": 8.0,
            "amount": 11_000.0,
        },
    ]


def test_load_national_geography_ctc_targets_uses_geography_year_for_ctc_periods(
    monkeypatch, tmp_path
):
    _, engine = _create_test_engine(tmp_path)

    geography_targets = {
        "refundable_ctc": {
            "source_year": 2022,
            "count": 17.0,
            "amount": 33_000.0,
        },
        "non_refundable_ctc": {
            "source_year": 2022,
            "count": 37.0,
            "amount": 81_000.0,
        },
    }
    monkeypatch.setattr(
        "policyengine_us_data.db.etl_irs_soi._get_national_geography_soi_target_from_year",
        lambda variable, geography_year: geography_targets[variable],
    )

    with Session(engine) as session:
        national_filer_stratum = _create_national_filer_stratum(session)

        load_national_geography_ctc_targets(
            session,
            national_filer_stratum.stratum_id,
            2022,
        )
        session.commit()

        for variable, expected in geography_targets.items():
            stratum = session.exec(
                select(Stratum).where(
                    Stratum.parent_stratum_id == national_filer_stratum.stratum_id,
                    Stratum.notes == f"National filers with {variable} > 0",
                )
            ).first()
            assert stratum is not None

            targets = session.exec(
                select(Target)
                .where(
                    Target.stratum_id == stratum.stratum_id,
                    Target.period == 2022,
                )
                .order_by(Target.variable)
            ).all()
            assert {target.variable: target.value for target in targets} == {
                "tax_unit_count": expected["count"],
                variable: expected["amount"],
            }


def test_load_national_geography_ctc_agi_targets_creates_agi_domain_strata(
    monkeypatch, tmp_path
):
    db_uri, engine = _create_test_engine(tmp_path)

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_irs_soi._get_national_geography_soi_agi_targets_from_year",
        lambda variable, geography_year: [
            {
                "variable": variable,
                "source_year": geography_year,
                "agi_stub": 7,
                "agi_lower_bound": 100_000.0,
                "agi_upper_bound": 200_000.0,
                "count": 12.0,
                "amount": 34_000.0,
            }
        ],
    )

    with Session(engine) as session:
        national_filer_stratum = _create_national_filer_stratum(session)
        load_national_geography_ctc_agi_targets(
            session,
            national_filer_stratum.stratum_id,
            2022,
        )
        session.commit()

    builder = UnifiedMatrixBuilder(db_uri=db_uri, time_period=2024)
    rows = builder._query_targets(
        {
            "geo_level": "national",
            "variables": ["tax_unit_count", "refundable_ctc", "non_refundable_ctc"],
            "domain_variables": ["adjusted_gross_income,refundable_ctc"],
        }
    )

    assert set(rows["variable"]) == {"tax_unit_count", "refundable_ctc"}
    assert set(rows["period"].astype(int)) == {2022}
    assert set(rows["value"].astype(float)) == {12.0, 34_000.0}

    with engine.connect() as conn:
        overview_rows = conn.execute(
            text(
                """
                SELECT domain_variable, geographic_id
                FROM target_overview
                WHERE geo_level = 'national'
                  AND period = 2022
                  AND variable IN ('tax_unit_count', 'refundable_ctc')
                  AND domain_variable LIKE '%refundable_ctc%'
                  AND domain_variable LIKE '%adjusted_gross_income%'
                """
            )
        ).fetchall()

    assert overview_rows
    assert all(row.geographic_id == "US" for row in overview_rows)


def test_load_national_eitc_agi_child_targets_creates_structured_db_rows(
    monkeypatch, tmp_path
):
    db_uri, engine = _create_test_engine(tmp_path)
    calibration_dir = tmp_path / "calibration_targets"
    calibration_dir.mkdir()
    (calibration_dir / "eitc_by_agi_and_children.csv").write_text(
        "# IRS SOI Table 2.5 sample\n"
        "count_children,agi_lower,agi_upper,returns,amount\n"
        "0,1,1000,10,1000\n"
        "3,1,1000,20,2000\n"
        "3,1000,2000,0,0\n"
    )
    monkeypatch.setattr(
        "policyengine_us_data.db.etl_irs_soi.CALIBRATION_FOLDER",
        calibration_dir,
    )

    with Session(engine) as session:
        national_filer_stratum = _create_national_filer_stratum(session)
        load_national_eitc_agi_child_targets(
            session,
            national_filer_stratum.stratum_id,
            source_year=2022,
        )
        session.commit()

    builder = UnifiedMatrixBuilder(db_uri=db_uri, time_period=2024)
    rows = builder._query_targets(
        {
            "variables": ["tax_unit_count", "eitc"],
            "domain_variables": ["adjusted_gross_income,eitc,eitc_child_count"],
        }
    )

    assert len(rows) == 4
    assert set(rows["period"].astype(int)) == {2022}
    assert set(rows["variable"]) == {"tax_unit_count", "eitc"}
    assert set(rows["value"].astype(float)) == {10.0, 20.0, 1000.0, 2000.0}

    with engine.connect() as conn:
        constraints = conn.execute(
            text(
                """
                SELECT tv.value, sc.constraint_variable, sc.operation, sc.value
                FROM target_overview tv
                JOIN stratum_constraints sc ON tv.stratum_id = sc.stratum_id
                WHERE tv.variable = 'eitc'
                  AND tv.domain_variable = 'adjusted_gross_income,eitc,eitc_child_count'
                ORDER BY tv.value, sc.constraint_variable, sc.operation
                """
            )
        ).fetchall()

    constraints_by_target = {}
    for target_value, variable, operation, constraint_value in constraints:
        constraints_by_target.setdefault(float(target_value), set()).add(
            (variable, operation, constraint_value)
        )

    assert ("eitc_child_count", "==", "0") in constraints_by_target[1000.0]
    assert ("eitc_child_count", ">", "2") in constraints_by_target[2000.0]
    assert ("adjusted_gross_income", ">=", "1.0") in constraints_by_target[2000.0]
    assert ("adjusted_gross_income", "<", "1000.0") in constraints_by_target[2000.0]
