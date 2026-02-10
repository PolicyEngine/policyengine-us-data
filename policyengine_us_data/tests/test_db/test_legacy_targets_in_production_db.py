"""Tests that the production policy_data.db contains all legacy
calibration targets from loss.py.

These targets were loaded by running ``load_all_targets()`` from
``etl_all_targets.py`` against the production database. The tests
verify minimum target counts by category to guard against regressions.
"""

import sqlite3

import pytest

from policyengine_us_data.storage import STORAGE_FOLDER

DB_PATH = STORAGE_FOLDER / "calibration" / "policy_data.db"


@pytest.fixture
def cursor():
    """Provide a read-only cursor to the production DB."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    yield cur
    conn.close()


def _count_targets_by_notes(cursor, pattern, period=2024):
    """Count targets whose notes contain the given pattern."""
    cursor.execute(
        "SELECT COUNT(*) FROM targets " "WHERE notes LIKE ? AND period = ?",
        (f"%{pattern}%", period),
    )
    return cursor.fetchone()[0]


def _count_targets_by_stratum_group(cursor, group_id, period=2024):
    """Count targets whose stratum has the given group_id."""
    cursor.execute(
        "SELECT COUNT(*) FROM targets t "
        "JOIN strata s ON t.stratum_id = s.stratum_id "
        "WHERE s.stratum_group_id = ? AND t.period = ?",
        (group_id, period),
    )
    return cursor.fetchone()[0]


class TestLegacyTargetsPresent:
    """Verify all legacy loss.py target categories are present."""

    def test_census_single_year_age(self, cursor):
        """86 single-year age bins (ages 0-85)."""
        count = _count_targets_by_notes(cursor, "Census age bin")
        assert count == 86, f"Expected 86 age targets, got {count}"

    def test_eitc_by_child_count(self, cursor):
        """4 child buckets x 2 metrics (returns + spending) + 1
        national EITC.
        """
        count = _count_targets_by_notes(cursor, "EITC")
        # 4 child counts x 2 = 8, plus 1 national = 9
        assert count >= 8, f"Expected >= 8 EITC targets, got {count}"

    def test_soi_filer_counts(self, cursor):
        """7 AGI bands for SOI filer counts."""
        count = _count_targets_by_notes(cursor, "SOI filer count")
        assert count == 7, f"Expected 7 filer count targets, got {count}"

    def test_healthcare_spending_by_age(self, cursor):
        """9 age bands x 4 expense types = 36."""
        count = _count_targets_by_notes(cursor, "Healthcare")
        assert count == 36, f"Expected 36 healthcare targets, got {count}"

    def test_spm_threshold_deciles(self, cursor):
        """10 deciles x 2 (AGI + count) = 20."""
        count = _count_targets_by_notes(cursor, "SPM threshold")
        assert count == 20, f"Expected 20 SPM threshold targets, got {count}"

    def test_negative_market_income(self, cursor):
        """2 targets: total + count."""
        count = _count_targets_by_notes(cursor, "Negative household market")
        assert (
            count == 2
        ), f"Expected 2 negative market income targets, got {count}"

    def test_tax_expenditures(self, cursor):
        """5 deductions with reform_id=1."""
        cursor.execute(
            "SELECT COUNT(*) FROM targets "
            "WHERE reform_id = 1 AND period = 2024",
        )
        count = cursor.fetchone()[0]
        assert (
            count >= 5
        ), f"Expected >= 5 tax expenditure targets, got {count}"

    def test_state_population(self, cursor):
        """51 state totals + 51 under-5 = 102 minimum."""
        count = _count_targets_by_notes(cursor, "State")
        assert count >= 100, f"Expected >= 100 state targets, got {count}"

    def test_state_real_estate_taxes(self, cursor):
        """51 states."""
        count = _count_targets_by_notes(cursor, "State real estate")
        assert count == 51, f"Expected 51 real estate tax targets, got {count}"

    def test_state_aca(self, cursor):
        """51 states x 2 (spending + enrollment) = 102."""
        spending = _count_targets_by_notes(cursor, "ACA spending")
        enrollment = _count_targets_by_notes(cursor, "ACA enrollment")
        assert (
            spending >= 50
        ), f"Expected >= 50 ACA spending targets, got {spending}"
        assert (
            enrollment >= 50
        ), f"Expected >= 50 ACA enrollment targets, got {enrollment}"

    def test_state_medicaid_enrollment(self, cursor):
        """51 states."""
        count = _count_targets_by_notes(cursor, "State Medicaid")
        assert count == 51, f"Expected 51 Medicaid targets, got {count}"

    def test_state_10yr_age(self, cursor):
        """50 states x 18 ranges = 900."""
        count = _count_targets_by_notes(cursor, "State 10yr age")
        assert count == 900, f"Expected 900 state age targets, got {count}"

    def test_state_agi(self, cursor):
        """918 state AGI targets."""
        count = _count_targets_by_notes(cursor, "State AGI")
        assert count == 918, f"Expected 918 state AGI targets, got {count}"

    def test_soi_filing_status(self, cursor):
        """SOI filing-status x AGI bin targets."""
        count = _count_targets_by_notes(cursor, "SOI filing-status")
        assert (
            count >= 280
        ), f"Expected >= 280 SOI filing-status targets, got {count}"

    def test_net_worth(self, cursor):
        """National net worth target."""
        cursor.execute(
            "SELECT value FROM targets "
            "WHERE variable = 'net_worth' AND period = 2024 "
            "AND stratum_id = 1",
        )
        row = cursor.fetchone()
        assert row is not None, "Net worth target not found"
        assert row[0] == 160e12

    def test_total_2024_targets_minimum(self, cursor):
        """Overall count: the old loss.py had ~350+ national targets.
        With state/CD targets, we expect well over 9000 total.
        """
        cursor.execute("SELECT COUNT(*) FROM targets WHERE period = 2024")
        count = cursor.fetchone()[0]
        assert (
            count >= 12000
        ), f"Expected >= 12000 total 2024 targets, got {count}"

    def test_income_tax_constraint_uses_valid_variable(self, cursor):
        """Verify that SOI filing-status strata use 'income_tax'
        (not the invalid 'total_income_tax') as a constraint
        variable.
        """
        cursor.execute(
            "SELECT COUNT(*) FROM stratum_constraints "
            "WHERE constraint_variable = 'total_income_tax'"
        )
        count = cursor.fetchone()[0]
        assert count == 0, (
            f"Found {count} constraints using invalid "
            f"'total_income_tax' -- should be 'income_tax'"
        )

        cursor.execute(
            "SELECT COUNT(*) FROM stratum_constraints "
            "WHERE constraint_variable = 'income_tax'"
        )
        count = cursor.fetchone()[0]
        assert count > 0, "No constraints using 'income_tax' found"
