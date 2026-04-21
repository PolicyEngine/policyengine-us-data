"""
End-to-end test for the calibration database build pipeline.

Runs every ETL script in the same order as ``make database`` and
validates the resulting SQLite database has the expected structure and
content.  This catches API mismatches, missing imports, and data-loading
errors that unit tests on individual tables would miss.
"""

import sqlite3
import subprocess
import sys

import pytest

from policyengine_us_data.storage import STORAGE_FOLDER

# Directory and file for the calibration database.
DB_DIR = STORAGE_FOLDER / "calibration"
DB_PATH = DB_DIR / "policy_data.db"

# Scripts run in the same order as `make database` in the Makefile.
# create_database_tables.py and validate_database.py do not use etl_argparser.
PIPELINE_SCRIPTS = [
    ("db/create_database_tables.py", []),
    ("db/create_initial_strata.py", ["--year", "2024"]),
    ("db/etl_national_targets.py", ["--year", "2024"]),
    ("db/etl_age.py", ["--year", "2024"]),
    ("db/etl_medicaid.py", ["--year", "2024"]),
    ("db/etl_snap.py", ["--year", "2024"]),
    ("db/etl_tanf.py", ["--year", "2024"]),
    ("db/etl_state_income_tax.py", ["--year", "2024"]),
    ("db/etl_irs_soi.py", ["--year", "2024"]),
    ("db/etl_aca_agi_state_targets.py", ["--year", "2024"]),
    ("db/etl_aca_marketplace.py", ["--year", "2024"]),
    ("db/etl_pregnancy.py", ["--year", "2024"]),
    ("db/validate_database.py", []),
]

PKG_ROOT = STORAGE_FOLDER.parent


def _run_script(
    relative_path: str,
    extra_args: list,
) -> subprocess.CompletedProcess:
    """Run a script from the package root and return the result."""
    script = PKG_ROOT / relative_path
    assert script.exists(), f"Script not found: {script}"
    return subprocess.run(
        [sys.executable, str(script)] + extra_args,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def built_db():
    """Build the calibration database from scratch once per module.

    Removes any existing DB first so the test validates a clean build.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    errors = []
    for script, args in PIPELINE_SCRIPTS:
        result = _run_script(script, args)
        if result.returncode != 0:
            errors.append(
                f"{script} failed (rc={result.returncode}):\n"
                f"  stderr (last 500 chars): "
                f"{result.stderr[-500:]}"
            )

    if errors:
        pytest.fail(f"{len(errors)} ETL script(s) failed:\n" + "\n\n".join(errors))

    assert DB_PATH.exists(), "policy_data.db was not created"
    return DB_PATH


def test_all_etl_scripts_succeed(built_db):
    """The fixture itself asserts all scripts pass; this makes the
    assertion visible as a named test."""
    assert built_db.exists()


def test_expected_tables_exist(built_db):
    """Core tables must be present."""
    conn = sqlite3.connect(str(built_db))
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    conn.close()

    for expected in ["strata", "stratum_constraints", "targets"]:
        assert expected in tables, f"Missing table: {expected}"


def test_national_targets_loaded(built_db):
    """National targets should include well-known variables."""
    conn = sqlite3.connect(str(built_db))
    # The national stratum has no constraints in stratum_constraints.
    rows = conn.execute("""
        SELECT DISTINCT t.variable
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        LEFT JOIN stratum_constraints sc
            ON s.stratum_id = sc.stratum_id
        WHERE sc.stratum_id IS NULL
        """).fetchall()
    conn.close()

    variables = {r[0] for r in rows}
    for expected in ["snap", "social_security", "ssi"]:
        assert expected in variables, (
            f"National target '{expected}' missing. Found: {sorted(variables)}"
        )


def test_jct_mortgage_tax_expenditure_uses_mortgage_specific_variable(built_db):
    """The mortgage JCT target should point at a mortgage-specific variable."""
    conn = sqlite3.connect(str(built_db))
    rows = conn.execute("""
        SELECT DISTINCT t.variable, t.source, t.notes
        FROM targets t
        WHERE t.variable = 'deductible_mortgage_interest'
        """).fetchall()
    conn.close()

    assert rows == [
        (
            "deductible_mortgage_interest",
            "PolicyEngine",
            "Mortgage interest deduction tax expenditure | Modeled as repeal-based income tax expenditure target | Source: Joint Committee on Taxation",
        )
    ]


def test_jct_tax_expenditure_targets_have_distinct_reform_ids(built_db):
    """Each JCT tax expenditure target should have its own reform id."""
    conn = sqlite3.connect(str(built_db))
    rows = conn.execute("""
        SELECT t.variable, t.reform_id
        FROM targets t
        WHERE t.notes LIKE '%Modeled as repeal-based income tax expenditure target%'
          AND t.notes LIKE '%Source: Joint Committee on Taxation%'
        ORDER BY t.variable
        """).fetchall()
    conn.close()

    expected = [
        ("charitable_deduction", 3),
        ("deductible_mortgage_interest", 4),
        ("medical_expense_deduction", 2),
        ("qualified_business_income_deduction", 5),
        ("salt_deduction", 1),
    ]

    assert rows == expected


def test_state_income_tax_targets(built_db):
    """State income tax targets should match the official FY2023 Census T40 row."""
    conn = sqlite3.connect(str(built_db))
    rows = conn.execute("""
        SELECT sc.value, t.value
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE t.variable = 'state_income_tax'
          AND sc.constraint_variable = 'state_fips'
        """).fetchall()
    conn.close()

    state_totals = {r[0]: r[1] for r in rows}

    n = len(state_totals)
    assert n >= 42, f"Expected >= 42 state income tax targets, got {n}"

    # Values come from Census STC FY2023 Table 1 / item T40
    # (Individual Income Taxes), reported in thousands of dollars.
    ca_val = state_totals.get("06") or state_totals.get("6")
    assert ca_val is not None, "California (FIPS 06) target missing"
    assert ca_val == 96_379_294_000

    wa_val = state_totals.get("53")
    assert wa_val == 846_835_000

    nh_val = state_totals.get("33")
    assert nh_val == 149_485_000

    tn_val = state_totals.get("47")
    assert tn_val == 2_926_000


def test_state_aca_and_agi_targets_loaded(built_db):
    """Legacy ACA spending/enrollment and AGI state targets should be present
    (loaded by etl_aca_agi_state_targets.py)."""
    conn = sqlite3.connect(str(built_db))
    aca_spending = conn.execute(
        """
        SELECT COUNT(*)
        FROM target_overview
        WHERE variable = 'aca_ptc'
          AND geo_level = 'state'
        """
    ).fetchone()[0]
    aca_enrollment = conn.execute(
        """
        SELECT COUNT(*)
        FROM target_overview
        WHERE variable = 'person_count'
          AND geo_level = 'state'
          AND domain_variable LIKE '%aca_ptc%'
        """
    ).fetchone()[0]
    agi_amount = conn.execute(
        """
        SELECT COUNT(*)
        FROM target_overview
        WHERE variable = 'adjusted_gross_income'
          AND geo_level = 'state'
          AND domain_variable LIKE '%adjusted_gross_income%'
        """
    ).fetchone()[0]
    agi_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM target_overview
        WHERE variable = 'tax_unit_count'
          AND geo_level = 'state'
          AND domain_variable LIKE '%adjusted_gross_income%'
        """
    ).fetchone()[0]
    conn.close()

    assert aca_spending > 0, "Missing ACA spending targets by state"
    assert aca_enrollment > 0, "Missing ACA enrollment targets by state"
    assert agi_amount > 0, "Missing state AGI amount targets"
    assert agi_count > 0, "Missing state AGI count targets"


def test_state_marketplace_targets_loaded(built_db):
    """ACA marketplace APTC and bronze state targets should be present, with
    canonical alphabetical domain_variable strings that ``target_config.yaml``
    rules can match."""
    conn = sqlite3.connect(str(built_db))
    aptc_targets = conn.execute(
        """
        SELECT COUNT(*)
        FROM target_overview
        WHERE variable = 'tax_unit_count'
          AND geo_level = 'state'
          AND domain_variable = 'used_aca_ptc'
        """
    ).fetchone()[0]
    # Regression for the bronze domain_variable ordering bug: must match the
    # alphabetical form in target_config.yaml:68, not an insertion-ordered
    # alternative.
    bronze_targets = conn.execute(
        """
        SELECT COUNT(*)
        FROM target_overview
        WHERE variable = 'tax_unit_count'
          AND geo_level = 'state'
          AND domain_variable
              = 'selected_marketplace_plan_benchmark_ratio,used_aca_ptc'
        """
    ).fetchone()[0]
    conn.close()

    # HC.gov had 32 states in 2024; allow a cushion for data updates.
    assert aptc_targets >= 27, (
        f"Missing state marketplace APTC targets (got {aptc_targets})"
    )
    assert bronze_targets >= 27, (
        "Missing state marketplace bronze-selection targets with canonical "
        f"domain_variable (got {bronze_targets})"
    )


def test_tanf_targets(built_db):
    """TANF recipient-family and spending targets should load from ACF files."""
    conn = sqlite3.connect(str(built_db))
    rows = conn.execute("""
        SELECT
            COALESCE(state_sc.value, 'US') AS geography,
            t.variable,
            t.value
        FROM targets t
        JOIN strata s
            ON t.stratum_id = s.stratum_id
        JOIN stratum_constraints tanf_sc
            ON s.stratum_id = tanf_sc.stratum_id
        LEFT JOIN stratum_constraints state_sc
            ON s.stratum_id = state_sc.stratum_id
           AND state_sc.constraint_variable = 'state_fips'
        WHERE tanf_sc.constraint_variable = 'tanf'
          AND tanf_sc.operation = '>'
          AND tanf_sc.value = '0'
          AND t.variable IN ('spm_unit_count', 'tanf')
    """).fetchall()
    conn.close()

    tanf_targets = {(geo, variable): value for geo, variable, value in rows}

    assert tanf_targets[("US", "spm_unit_count")] == pytest.approx(841_208.666667)
    assert tanf_targets[("US", "tanf")] == pytest.approx(7_788_317_474.55)
    assert tanf_targets[("6", "spm_unit_count")] == pytest.approx(290_247.75)
    assert tanf_targets[("6", "tanf")] == pytest.approx(3_742_540_224.36)
    assert tanf_targets[("11", "spm_unit_count")] == pytest.approx(5_056.25)
    assert tanf_targets[("11", "tanf")] == pytest.approx(45_666_113.50)


def test_congressional_district_strata(built_db):
    """Should have strata for >= 435 congressional districts."""
    conn = sqlite3.connect(str(built_db))
    n_cds = conn.execute("""
        SELECT COUNT(DISTINCT sc.value)
        FROM stratum_constraints sc
        WHERE sc.constraint_variable = 'congressional_district_geoid'
        """).fetchone()[0]
    conn.close()

    assert n_cds >= 435, f"Expected >= 435 CD strata, got {n_cds}"


def test_all_target_variables_exist_in_policyengine(built_db):
    """Every target variable must be a valid policyengine-us variable."""
    from policyengine_us.system import system

    conn = sqlite3.connect(str(built_db))
    variables = {r[0] for r in conn.execute("SELECT DISTINCT variable FROM targets")}
    conn.close()

    missing = [v for v in variables if v not in system.variables]
    assert not missing, f"Target variables not in policyengine-us: {missing}"


def test_total_target_count(built_db):
    """Sanity check: should have a healthy number of targets."""
    conn = sqlite3.connect(str(built_db))
    count = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    conn.close()

    # With national + age + medicaid + SNAP + state income tax + IRS SOI,
    # we expect thousands of targets.
    assert count > 1000, f"Expected > 1000 total targets, got {count}"
