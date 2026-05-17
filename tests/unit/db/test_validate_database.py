import sqlite3

import pytest

from policyengine_us_data.db.validate_database import validate_database


TAX_EXPENDITURE_TARGETS = [
    "salt_deduction",
    "charitable_deduction",
    "deductible_mortgage_interest",
    "medical_expense_deduction",
    "qualified_business_income_deduction",
]


def _write_policy_data_db(db_path, target_variables):
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE strata (
                stratum_id INTEGER PRIMARY KEY,
                parent_stratum_id INTEGER
            );
            CREATE TABLE stratum_constraints (
                stratum_id INTEGER,
                constraint_variable TEXT
            );
            CREATE TABLE targets (
                stratum_id INTEGER,
                variable TEXT,
                active INTEGER,
                reform_id INTEGER
            );
        """
        )
        conn.execute(
            "INSERT INTO strata (stratum_id, parent_stratum_id) VALUES (?, ?)",
            (1, None),
        )
        conn.execute(
            "INSERT INTO stratum_constraints (stratum_id, constraint_variable) "
            "VALUES (?, ?)",
            (1, "total_self_employment_income"),
        )

        for reform_id, variable in enumerate(TAX_EXPENDITURE_TARGETS, start=1):
            conn.execute(
                "INSERT INTO targets (stratum_id, variable, active, reform_id) "
                "VALUES (?, ?, ?, ?)",
                (1, variable, 1, reform_id),
            )

        for variable in target_variables:
            conn.execute(
                "INSERT INTO targets (stratum_id, variable, active, reform_id) "
                "VALUES (?, ?, ?, ?)",
                (1, variable, 1, 0),
            )

        conn.commit()
    finally:
        conn.close()


def test_validate_database_accepts_total_self_employment_income(tmp_path):
    db_path = tmp_path / "policy_data.db"
    _write_policy_data_db(
        db_path,
        [
            "total_self_employment_income",
            "taxable_interest_income+tax_exempt_interest_income",
        ],
    )

    validate_database(db_path)


@pytest.mark.parametrize(
    "variable",
    [
        "taxable_interest_income+",
        "taxable_interest_income++tax_exempt_interest_income",
    ],
)
def test_validate_database_rejects_empty_additive_terms(tmp_path, variable):
    db_path = tmp_path / "policy_data.db"
    _write_policy_data_db(db_path, [variable])

    with pytest.raises(ValueError, match="not a policyengine-us variable"):
        validate_database(db_path)
