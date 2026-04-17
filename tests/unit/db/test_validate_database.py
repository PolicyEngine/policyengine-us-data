import sqlite3

from policyengine_us_data.db.validate_database import validate_database


def test_validate_database_accepts_total_self_employment_income(tmp_path):
    db_path = tmp_path / "policy_data.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
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
        """)
        conn.execute(
            "INSERT INTO strata (stratum_id, parent_stratum_id) VALUES (?, ?)",
            (1, None),
        )
        conn.execute(
            "INSERT INTO stratum_constraints (stratum_id, constraint_variable) "
            "VALUES (?, ?)",
            (1, "total_self_employment_income"),
        )

        for reform_id, variable in enumerate(
            [
                "salt_deduction",
                "charitable_deduction",
                "deductible_mortgage_interest",
                "medical_expense_deduction",
                "qualified_business_income_deduction",
            ],
            start=1,
        ):
            conn.execute(
                "INSERT INTO targets (stratum_id, variable, active, reform_id) "
                "VALUES (?, ?, ?, ?)",
                (1, variable, 1, reform_id),
            )

        conn.execute(
            "INSERT INTO targets (stratum_id, variable, active, reform_id) "
            "VALUES (?, ?, ?, ?)",
            (1, "total_self_employment_income", 1, 0),
        )
        conn.commit()
    finally:
        conn.close()

    validate_database(db_path)
