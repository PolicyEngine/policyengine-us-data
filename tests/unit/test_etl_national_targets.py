import inspect

import pandas as pd
from sqlalchemy import text
from sqlmodel import Session, select

from policyengine_us_data.db import etl_national_targets
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)
from policyengine_us_data.db.etl_national_targets import (
    MEDICARE_PART_B_AGE_TARGET_YEAR,
    extract_national_targets,
    load_medicare_part_b_age_targets,
    load_national_targets,
    load_state_acs_rent_targets,
)


def test_national_targets_do_not_extract_treasury_eitc():
    source = inspect.getsource(etl_national_targets.extract_national_targets)

    assert "tax_expenditures.eitc" not in source


def test_transform_national_targets_ignores_treasury_eitc_compat_key():
    raw_targets = {
        "direct_sum_targets": [],
        "tax_filer_targets": [],
        "tax_expenditure_targets": [],
        "conditional_count_targets": [],
        "cbo_targets": [],
        "irs_soi_targets": [],
        "treasury_targets": [
            {
                "variable": "eitc",
                "value": 67.33e9,
                "source": "Treasury/JCT Tax Expenditures",
                "notes": "EITC tax expenditure",
                "year": 2024,
            }
        ],
    }

    _, tax_filer_df, _, _ = etl_national_targets.transform_national_targets(raw_targets)

    assert tax_filer_df.empty


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
            "constraint_variable": "spm_unit_energy_subsidy",
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
        assert ("spm_unit_energy_subsidy", ">", "0") in constraints

        liheap_target = session.exec(
            select(Target).where(
                Target.stratum_id == liheap_stratum.stratum_id,
                Target.variable == "household_count",
                Target.period == 2024,
            )
        ).first()
        assert liheap_target is not None
        assert liheap_target.value == 5_876_646


def test_load_state_acs_rent_targets_creates_state_rows(tmp_path, monkeypatch):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    db_uri = f"sqlite:///{calibration_dir / 'policy_data.db'}"
    engine = create_database(db_uri)

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

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_national_targets.STORAGE_FOLDER",
        tmp_path,
    )

    targets = pd.DataFrame(
        [
            {
                "state_code": "CA",
                "state_fips": "06",
                "annual_contract_rent": 143_291_068_800,
                "real_estate_taxes": 52_872_735_400,
            }
        ]
    )
    load_state_acs_rent_targets(targets, year=2024)

    with Session(engine) as session:
        target = session.exec(
            select(Target).where(
                Target.variable == "rent",
                Target.period == 2024,
            )
        ).first()
        assert target is not None
        assert target.value == 143_291_068_800
        assert target.source == "PolicyEngine"
        assert "Census ACS 2024 1-year table B25060" in target.notes


def test_load_medicare_part_b_age_targets_creates_age_domain_rows(
    tmp_path, monkeypatch
):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    db_uri = f"sqlite:///{calibration_dir / 'policy_data.db'}"
    engine = create_database(db_uri)

    with Session(engine) as session:
        _make_stratum(session, notes="United States")

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_national_targets.STORAGE_FOLDER",
        tmp_path,
    )

    targets = pd.DataFrame(
        [
            {
                "age_10_year_lower_bound": 70,
                "medicare_part_b_premiums": 54_002_252_445.0,
            },
            {
                "age_10_year_lower_bound": 80,
                "medicare_part_b_premiums": 24_692_726_700.0,
            },
        ]
    )
    load_medicare_part_b_age_targets(targets)

    with Session(engine) as session:
        rows = session.exec(
            select(Target).where(Target.variable == "medicare_part_b_premium")
        ).all()
        assert len(rows) == 2
        assert {row.period for row in rows} == {MEDICARE_PART_B_AGE_TARGET_YEAR}

    with engine.connect() as conn:
        overview = conn.execute(
            text(
                """
                SELECT variable, domain_variable, value
                FROM target_overview
                WHERE variable = 'medicare_part_b_premium'
                ORDER BY value
                """
            )
        ).fetchall()

    assert {row.domain_variable for row in overview} == {"age"}
    assert {float(row.value) for row in overview} == {
        54_002_252_445.0,
        24_692_726_700.0,
    }


def test_extract_national_targets_drops_survey_spm_targets():
    targets = extract_national_targets(year=2024)
    direct_sum_variables = {
        target["variable"] for target in targets["direct_sum_targets"]
    }
    removed_targets = {
        "alimony_income",
        "alimony_expense",
        "child_support_expense",
        "child_support_received",
        "employer_sponsored_insurance_premiums",
        "health_insurance_premiums_without_medicare_part_b",
        "other_medical_expenses",
        "over_the_counter_health_expenses",
        "spm_unit_spm_threshold",
        "spm_unit_capped_housing_subsidy",
        "spm_unit_capped_work_childcare_expenses",
    }

    assert removed_targets.isdisjoint(direct_sum_variables)
    assert {
        "rent",
        "real_estate_taxes",
        "childcare_expenses",
        "medicare_part_b_premium",
    } <= direct_sum_variables

    direct_sum_targets = {
        target["variable"]: target for target in targets["direct_sum_targets"]
    }
    assert direct_sum_targets["rent"]["value"] == 764_925_694_800
    assert direct_sum_targets["real_estate_taxes"]["value"] == 370_014_207_400
    assert direct_sum_targets["childcare_expenses"]["value"] == 63_092e6


def test_extract_national_targets_includes_wic_targets():
    targets = extract_national_targets(year=2024)
    direct_sum_targets = {
        target["variable"]: target for target in targets["direct_sum_targets"]
    }
    wic_count_targets = [
        target
        for target in targets["conditional_count_targets"]
        if target["constraint_variable"] == "wic"
    ]

    assert direct_sum_targets["wic"]["value"] == 4_911_500_000
    assert direct_sum_targets["wic"]["source"] == (
        etl_national_targets.WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE
    )
    assert len(wic_count_targets) == 1
    assert wic_count_targets[0]["person_count"] == 6_704_000


def test_load_national_targets_uses_medicaid_enrolled_for_enrollment_counts(
    tmp_path, monkeypatch
):
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
            "constraint_variable": "medicaid_enrolled",
            "person_count": 72_429_055,
            "source": "CMS/HHS administrative data",
            "notes": "Medicaid enrollment count",
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
        medicaid_stratum = session.exec(
            select(Stratum).where(Stratum.notes == "National Medicaid Enrollment")
        ).first()
        assert medicaid_stratum is not None

        constraints = {
            (
                constraint.constraint_variable,
                constraint.operation,
                constraint.value,
            )
            for constraint in medicaid_stratum.constraints_rel
        }
        assert ("medicaid_enrolled", ">", "0") in constraints

        medicaid_target = session.exec(
            select(Target).where(
                Target.stratum_id == medicaid_stratum.stratum_id,
                Target.variable == "person_count",
                Target.period == 2024,
            )
        ).first()
        assert medicaid_target is not None
        assert medicaid_target.value == 72_429_055


def test_load_national_targets_supports_medicare_enrollment_counts(
    tmp_path, monkeypatch
):
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
            "constraint_variable": "medicare_enrolled",
            "person_count": 68_030_000,
            "source": "CMS 2024 Medicare Trustees Report Table V.B3",
            "notes": "Total Medicare enrollment count",
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
        medicare_stratum = session.exec(
            select(Stratum).where(
                Stratum.notes == "National medicare_enrolled Recipients"
            )
        ).first()
        assert medicare_stratum is not None

        constraints = {
            (
                constraint.constraint_variable,
                constraint.operation,
                constraint.value,
            )
            for constraint in medicare_stratum.constraints_rel
        }
        assert ("medicare_enrolled", ">", "0") in constraints

        medicare_target = session.exec(
            select(Target).where(
                Target.stratum_id == medicare_stratum.stratum_id,
                Target.variable == "person_count",
                Target.period == 2024,
            )
        ).first()
        assert medicare_target is not None
        assert medicare_target.value == 68_030_000


def test_load_national_targets_supports_wic_targets(tmp_path, monkeypatch):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    db_uri = f"sqlite:///{calibration_dir / 'policy_data.db'}"
    engine = create_database(db_uri)

    with Session(engine) as session:
        national = _make_stratum(session, notes="United States")
        national_id = national.stratum_id

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_national_targets.STORAGE_FOLDER",
        tmp_path,
    )

    direct_targets_df = pd.DataFrame(
        [
            {
                "variable": "wic",
                "value": 4_911_500_000,
                "source": etl_national_targets.WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE,
                "notes": "FY 2024 WIC food costs from FNS annual summary",
                "year": 2024,
            }
        ]
    )
    conditional_targets = [
        {
            "constraint_variable": "wic",
            "person_count": 6_704_000,
            "source": etl_national_targets.WIC_NATIONAL_ANNUAL_SUMMARY_SOURCE,
            "notes": "FY 2024 WIC average monthly participation",
            "year": 2024,
        }
    ]

    load_national_targets(
        direct_targets_df=direct_targets_df,
        tax_filer_df=pd.DataFrame(),
        tax_expenditure_df=pd.DataFrame(),
        conditional_targets=conditional_targets,
    )

    with Session(engine) as session:
        wic_total_target = session.exec(
            select(Target).where(
                Target.stratum_id == national_id,
                Target.variable == "wic",
                Target.period == 2024,
            )
        ).first()
        assert wic_total_target is not None
        assert wic_total_target.value == 4_911_500_000

        wic_stratum = session.exec(
            select(Stratum).where(Stratum.notes == "National WIC Recipients")
        ).first()
        assert wic_stratum is not None

        constraints = {
            (
                constraint.constraint_variable,
                constraint.operation,
                constraint.value,
            )
            for constraint in wic_stratum.constraints_rel
        }
        assert ("wic", ">", "0") in constraints

        wic_count_target = session.exec(
            select(Target).where(
                Target.stratum_id == wic_stratum.stratum_id,
                Target.variable == "person_count",
                Target.period == 2024,
            )
        ).first()
        assert wic_count_target is not None
        assert wic_count_target.value == 6_704_000


def test_loads_gross_wage_and_filer_tax_wage_targets(tmp_path, monkeypatch):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    db_uri = f"sqlite:///{calibration_dir / 'policy_data.db'}"
    engine = create_database(db_uri)

    with Session(engine) as session:
        _make_stratum(session, notes="United States")

    monkeypatch.setattr(
        "policyengine_us_data.db.etl_national_targets.STORAGE_FOLDER",
        tmp_path,
    )

    load_national_targets(
        direct_targets_df=pd.DataFrame(
            [
                {
                    "variable": "employment_income_before_lsr",
                    "value": 12_387_929_000_000,
                    "source": "BEA NIPA Table 2.1",
                    "notes": "Gross all-worker wages",
                    "year": 2024,
                },
                {
                    "variable": (
                        etl_national_targets.NIPA_PERSONAL_INTEREST_INCOME_VARIABLE
                    ),
                    "value": 1_926_644_000_000,
                    "source": "BEA NIPA Table 2.1",
                    "notes": "Personal interest income",
                    "year": 2024,
                },
                {
                    "variable": etl_national_targets.NIPA_PROPRIETORS_INCOME_VARIABLE,
                    "value": 2_023_080_000_000,
                    "source": "BEA NIPA Table 2.1",
                    "notes": "Proprietors' income",
                    "year": 2024,
                },
                {
                    "variable": "dividend_income",
                    "value": 2_218_700_000_000,
                    "source": "BEA NIPA Table 2.1",
                    "notes": "Personal dividend income",
                    "year": 2024,
                },
            ]
        ),
        tax_filer_df=pd.DataFrame(
            [
                {
                    "variable": "irs_employment_income",
                    "value": 10_832_700_000_000,
                    "source": "CBO Revenue Projections",
                    "notes": "AGI-by-source wages",
                    "year": 2024,
                }
            ]
        ),
        tax_expenditure_df=pd.DataFrame(),
        conditional_targets=[],
    )

    with Session(engine) as session:
        gross_wage_target = session.exec(
            select(Target).where(Target.variable == "employment_income_before_lsr")
        ).one()
        tax_wage_target = session.exec(
            select(Target).where(Target.variable == "irs_employment_income")
        ).one()
        interest_target = session.exec(
            select(Target).where(
                Target.variable
                == etl_national_targets.NIPA_PERSONAL_INTEREST_INCOME_VARIABLE
            )
        ).one()
        proprietors_target = session.exec(
            select(Target).where(
                Target.variable == etl_national_targets.NIPA_PROPRIETORS_INCOME_VARIABLE
            )
        ).one()
        dividend_target = session.exec(
            select(Target).where(Target.variable == "dividend_income")
        ).one()
        filer_constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == tax_wage_target.stratum_id
            )
        ).all()

    assert gross_wage_target.value == 12_387_929_000_000
    assert tax_wage_target.value == 10_832_700_000_000
    assert interest_target.value == 1_926_644_000_000
    assert proprietors_target.value == 2_023_080_000_000
    assert dividend_target.value == 2_218_700_000_000
    assert gross_wage_target.stratum_id != tax_wage_target.stratum_id
    assert [
        (
            constraint.constraint_variable,
            constraint.operation,
            constraint.value,
        )
        for constraint in filer_constraints
    ] == [("tax_unit_is_filer", "==", "1")]


def test_extracts_income_targets_from_primary_concepts(monkeypatch):
    class FakeIncomeBySource:
        _children = {
            "employment_income": 10_832_700_000_000,
            "self_employment_income": 1_916_000_000_000,
            "taxable_pension_income": 1_522_500_000_000,
            "taxable_social_security": 577_200_000_000,
            "qualified_dividend_income": 354_300_000_000,
            "net_capital_gain": 1_290_900_000_000,
            "taxable_interest_and_ordinary_dividends": 309_700_000_000,
        }

    class FakeCBO:
        income_by_source = FakeIncomeBySource()
        _children = {
            "income_tax": 0,
            "snap": 0,
            "social_security": 0,
            "ssi": 0,
            "unemployment_compensation": 0,
        }

    class FakeSOI:
        _children = {"long_term_capital_gains": 0}

    class FakeGov:
        cbo = FakeCBO()
        irs = type("FakeIRS", (), {"soi": FakeSOI()})()

    class FakeCalibration:
        gov = FakeGov()

    class FakeParameters:
        def __call__(self, year):
            return self

        calibration = FakeCalibration()

    class FakeTaxBenefitSystem:
        parameters = FakeParameters()

    monkeypatch.setattr(
        "policyengine_us.CountryTaxBenefitSystem",
        FakeTaxBenefitSystem,
    )

    raw_targets = etl_national_targets.extract_national_targets(year=2024)

    gross_wage_targets = [
        target
        for target in raw_targets["direct_sum_targets"]
        if target["variable"] == "employment_income_before_lsr"
    ]
    proprietors_targets = [
        target
        for target in raw_targets["direct_sum_targets"]
        if target["variable"] == etl_national_targets.NIPA_PROPRIETORS_INCOME_VARIABLE
    ]
    interest_targets = [
        target
        for target in raw_targets["direct_sum_targets"]
        if target["variable"]
        == etl_national_targets.NIPA_PERSONAL_INTEREST_INCOME_VARIABLE
    ]
    dividend_targets = [
        target
        for target in raw_targets["direct_sum_targets"]
        if target["variable"] == "dividend_income"
    ]
    tax_wage_targets = [
        target
        for target in raw_targets["tax_filer_targets"]
        if target["variable"] == "irs_employment_income"
    ]
    cbo_self_employment_targets = [
        target
        for target in raw_targets["tax_filer_targets"]
        if target["variable"] == "self_employment_income"
    ]

    assert gross_wage_targets == [
        {
            "variable": "employment_income_before_lsr",
            "value": etl_national_targets.BEA_NIPA_WAGES_AND_SALARIES_2024,
            "source": "BEA NIPA Table 2.1",
            "notes": (
                "Gross wages and salaries for all workers, including "
                "nonfilers; FRED/BEA series A034RC1A027NBEA"
            ),
            "year": 2024,
        }
    ]
    assert tax_wage_targets == [
        {
            "variable": "irs_employment_income",
            "value": 10_832_700_000_000,
            "source": "CBO Revenue Projections",
            "notes": (
                "CBO detailed AGI-by-source employment income; restricted "
                "to tax filers because this is an AGI tax-return concept"
            ),
            "year": 2024,
        }
    ]
    assert proprietors_targets == [
        {
            "variable": etl_national_targets.NIPA_PROPRIETORS_INCOME_VARIABLE,
            "value": etl_national_targets.BEA_NIPA_PROPRIETORS_INCOME_2024,
            "source": "BEA NIPA Table 2.1",
            "notes": (
                "Proprietors' income with IVA and CCAdj for all persons, "
                "including nonfilers; FRED/BEA series A041RC1A027NBEA. "
                "Mapped to the PolicyEngine-US NIPA proprietors' income "
                "aggregate."
            ),
            "year": 2024,
        }
    ]
    assert interest_targets == [
        {
            "variable": etl_national_targets.NIPA_PERSONAL_INTEREST_INCOME_VARIABLE,
            "value": etl_national_targets.BEA_NIPA_PERSONAL_INTEREST_INCOME_2024,
            "source": "BEA NIPA Table 2.1",
            "notes": (
                "Personal interest income for all persons, including "
                "nonfilers; FRED/BEA series A064RC1A027NBEA. NIPA also "
                "includes imputed interest, so this is a macro benchmark "
                "rather than a pure tax concept."
            ),
            "year": 2024,
        }
    ]
    assert dividend_targets == [
        {
            "variable": "dividend_income",
            "value": (etl_national_targets.BEA_NIPA_PERSONAL_DIVIDEND_INCOME_2024),
            "source": "BEA NIPA Table 2.1",
            "notes": (
                "Personal dividend income for all persons, including "
                "nonfilers; FRED/BEA series B703RC1A027NBEA. NIPA "
                "includes dividends received through pension funds and "
                "private trusts, so this is a macro benchmark rather than "
                "a pure tax concept."
            ),
            "year": 2024,
        }
    ]
    assert cbo_self_employment_targets == [
        {
            "variable": "self_employment_income",
            "value": 1_916_000_000_000,
            "source": "CBO Revenue Projections",
            "notes": (
                "CBO detailed AGI-by-source self-employment income; "
                "restricted to tax filers because this is an AGI tax-return "
                "concept"
            ),
            "year": 2024,
        }
    ]
