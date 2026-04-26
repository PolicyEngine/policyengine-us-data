import json
import sqlite3

from policyengine_us_data.utils.national_target_parity import (
    Constraint,
    NationalTargetIndex,
    TargetRecord,
    build_national_target_parity_manifest,
    classify_national_target,
    extract_target_names_from_json,
    load_national_target_records,
)


def _record(
    target_id,
    *,
    variable,
    period=2024,
    domain_variable=None,
    reform_id=0,
    constraints=(),
):
    return TargetRecord(
        target_id=target_id,
        stratum_id=target_id + 100,
        variable=variable,
        reform_id=reform_id,
        value=target_id * 100.0,
        period=period,
        source="test",
        notes=None,
        geo_level="national",
        geographic_id="US",
        domain_variable=domain_variable,
        constraints=tuple(constraints),
    )


def test_classify_eitc_agi_child_target_matches_structured_db_row():
    index = NationalTargetIndex(
        [
            _record(
                9763,
                variable="eitc",
                period=2022,
                domain_variable="adjusted_gross_income,eitc,eitc_child_count",
                constraints=[
                    Constraint("tax_unit_is_filer", "==", "1"),
                    Constraint("eitc", ">", "0"),
                    Constraint("eitc_child_count", ">", "2"),
                    Constraint("adjusted_gross_income", ">=", "1.0"),
                    Constraint("adjusted_gross_income", "<", "1000.0"),
                ],
            )
        ]
    )

    row = classify_national_target(
        "nation/irs/eitc/amount/c3_1_1k",
        index,
        period=2024,
    )

    assert row["status"] == "matched"
    assert row["target_id"] == 9763
    assert row["reason"] == "structured_eitc_agi_child_target"


def test_classify_known_legacy_target_gets_named_reason():
    row = classify_national_target(
        (
            "nation/irs/business net profits/total/AGI in "
            "20k-25k/taxable/Married Filing Jointly/Surviving Spouse"
        ),
        NationalTargetIndex([]),
        period=2024,
    )

    assert row == {
        "target_name": (
            "nation/irs/business net profits/total/AGI in "
            "20k-25k/taxable/Married Filing Jointly/Surviving Spouse"
        ),
        "scope": "national",
        "status": "legacy_only",
        "reason": "legacy_soi_taxable_agi_filing_status_detail_not_in_target_db",
    }


def test_classify_soi_taxable_agi_filing_status_target_matches_structured_row():
    index = NationalTargetIndex(
        [
            _record(
                1201,
                variable="adjusted_gross_income",
                period=2023,
                domain_variable=(
                    "adjusted_gross_income,filing_status,income_tax_before_credits"
                ),
                constraints=[
                    Constraint("tax_unit_is_filer", "==", "1"),
                    Constraint("income_tax_before_credits", ">", "0"),
                    Constraint("adjusted_gross_income", ">=", "20000.0"),
                    Constraint("adjusted_gross_income", "<", "25000.0"),
                    Constraint("filing_status", "in", "JOINT|SURVIVING_SPOUSE"),
                ],
            )
        ]
    )

    row = classify_national_target(
        (
            "nation/irs/adjusted gross income/total/AGI in "
            "20k-25k/taxable/Married Filing Jointly/Surviving Spouse"
        ),
        index,
        period=2024,
    )

    assert row["status"] == "matched"
    assert row["target_id"] == 1201
    assert row["reason"] == "structured_soi_taxable_agi_filing_status_target"


def test_classify_soi_taxable_count_target_matches_all_filers_row():
    index = NationalTargetIndex(
        [
            _record(
                1202,
                variable="tax_unit_count",
                period=2023,
                domain_variable="adjusted_gross_income,income_tax_before_credits",
                constraints=[
                    Constraint("tax_unit_is_filer", "==", "1"),
                    Constraint("income_tax_before_credits", ">", "0"),
                    Constraint("adjusted_gross_income", ">=", "50000.0"),
                    Constraint("adjusted_gross_income", "<", "75000.0"),
                ],
            )
        ]
    )

    row = classify_national_target(
        "nation/irs/count/count/AGI in 50k-75k/taxable/All",
        index,
        period=2024,
    )

    assert row["status"] == "matched"
    assert row["target_id"] == 1202
    assert row["reason"] == "structured_soi_taxable_agi_filing_status_target"


def test_lossy_soi_taxable_agi_label_gets_explicit_legacy_reason():
    row = classify_national_target(
        "nation/irs/count/count/AGI in 2m-2m/taxable/All",
        NationalTargetIndex([]),
        period=2024,
    )

    assert row == {
        "target_name": "nation/irs/count/count/AGI in 2m-2m/taxable/All",
        "scope": "national",
        "status": "legacy_only",
        "reason": "legacy_soi_taxable_agi_label_has_lossy_bucket_encoding",
    }


def test_zero_eitc_agi_child_target_is_classified_as_intentionally_omitted():
    row = classify_national_target(
        "nation/irs/eitc/returns/c0_50k_inf",
        NationalTargetIndex([]),
        period=2024,
        target_value=0.0,
    )

    assert row == {
        "target_name": "nation/irs/eitc/returns/c0_50k_inf",
        "scope": "national",
        "status": "legacy_only",
        "reason": "zero_eitc_agi_child_target_omitted_from_target_db",
    }


def test_manifest_summarizes_matches_and_explicit_legacy_reasons(tmp_path):
    db_path = tmp_path / "policy_data.db"
    _create_minimal_target_db(db_path)

    manifest = build_national_target_parity_manifest(
        [
            "nation/cbo/snap",
            "nation/jct/interest_deduction_expenditure",
            "nation/census/population_by_age/80",
            "state/census/age/CA/0-4",
        ],
        db_path=db_path,
        period=2024,
    )

    assert manifest["summary"]["total"] == 3
    assert manifest["summary"]["statuses"] == {
        "legacy_only": 1,
        "matched": 2,
    }
    assert (
        manifest["summary"]["reasons"][
            "legacy_single_year_age_targets_replaced_by_db_age_bins"
        ]
        == 1
    )
    assert manifest["targets"][0]["target_id"] == 1
    assert manifest["targets"][1]["target_id"] == 2


def test_extract_target_names_from_diagnostic_json(tmp_path):
    path = tmp_path / "diagnostic.json"
    path.write_text(
        json.dumps(
            {
                "targets": [
                    {"target_name": "nation/cbo/snap"},
                    {"target_name": "state/census/age/CA/0-4"},
                ]
            }
        )
    )

    assert extract_target_names_from_json(path) == [
        "nation/cbo/snap",
        "state/census/age/CA/0-4",
    ]


def _create_minimal_target_db(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE targets (
                target_id INTEGER PRIMARY KEY,
                stratum_id INTEGER,
                variable TEXT,
                reform_id INTEGER,
                value REAL,
                period INTEGER,
                active INTEGER,
                source TEXT,
                notes TEXT
            );
            CREATE TABLE target_overview (
                target_id INTEGER,
                stratum_id INTEGER,
                variable TEXT,
                reform_id INTEGER,
                value REAL,
                period INTEGER,
                active INTEGER,
                geo_level TEXT,
                geographic_id TEXT,
                domain_variable TEXT
            );
            CREATE TABLE stratum_constraints (
                stratum_id INTEGER,
                constraint_variable TEXT,
                operation TEXT,
                value TEXT
            );
            INSERT INTO targets VALUES
                (1, 101, 'snap', 0, 1.0, 2024, 1, 'test', NULL),
                (2, 102, 'deductible_mortgage_interest', 4, 2.0, 2024, 1, 'test', NULL);
            INSERT INTO target_overview VALUES
                (1, 101, 'snap', 0, 1.0, 2024, 1, 'national', 'US', NULL),
                (2, 102, 'deductible_mortgage_interest', 4, 2.0, 2024, 1, 'national', 'US', NULL);
            """
        )

    records = load_national_target_records(path)
    assert len(records) == 2
