"""
Tests for best-period selection and uprating in SparseMatrixBuilder.
"""

import unittest
import tempfile
import os
import pandas as pd
from sqlalchemy import create_engine, text

from policyengine_us_data.datasets.cps.local_area_calibration.sparse_matrix_builder import (
    SparseMatrixBuilder,
)


class TestPeriodSelectionAndUprating(unittest.TestCase):
    """Test best-period SQL CTE and uprating logic."""

    @classmethod
    def setUpClass(cls):
        cls.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.db_path = cls.temp_db.name
        cls.temp_db.close()

        cls.db_uri = f"sqlite:///{cls.db_path}"
        engine = create_engine(cls.db_uri)

        with engine.connect() as conn:
            conn.execute(
                text("CREATE TABLE strata (" "stratum_id INTEGER PRIMARY KEY)")
            )
            conn.execute(
                text(
                    "CREATE TABLE stratum_constraints ("
                    "constraint_id INTEGER PRIMARY KEY, "
                    "stratum_id INTEGER, "
                    "constraint_variable TEXT, "
                    "operation TEXT, "
                    "value TEXT)"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE targets ("
                    "target_id INTEGER PRIMARY KEY, "
                    "stratum_id INTEGER, "
                    "variable TEXT, "
                    "value REAL, "
                    "period INTEGER)"
                )
            )

            # Create the target_overview view (matches production schema)
            conn.execute(text("""
                CREATE VIEW target_overview AS
                SELECT
                    t.target_id,
                    t.stratum_id,
                    t.variable,
                    t.value,
                    t.period,
                    1 as active,
                    CASE
                        WHEN MAX(CASE
                            WHEN sc.constraint_variable
                                = 'congressional_district_geoid'
                                THEN 1
                            WHEN sc.constraint_variable = 'ucgid_str'
                                AND length(sc.value) = 13 THEN 1
                            ELSE 0 END) = 1 THEN 'district'
                        WHEN MAX(CASE
                            WHEN sc.constraint_variable = 'state_fips'
                                THEN 1
                            WHEN sc.constraint_variable = 'ucgid_str'
                                AND length(sc.value) = 11 THEN 1
                            ELSE 0 END) = 1 THEN 'state'
                        ELSE 'national'
                    END AS geo_level,
                    COALESCE(
                        MAX(CASE
                            WHEN sc.constraint_variable
                                = 'congressional_district_geoid'
                            THEN sc.value END),
                        MAX(CASE
                            WHEN sc.constraint_variable = 'state_fips'
                            THEN sc.value END),
                        MAX(CASE
                            WHEN sc.constraint_variable = 'ucgid_str'
                            THEN sc.value END),
                        'US'
                    ) AS geographic_id,
                    GROUP_CONCAT(DISTINCT CASE
                        WHEN sc.constraint_variable NOT IN (
                            'state_fips',
                            'congressional_district_geoid',
                            'tax_unit_is_filer', 'ucgid_str'
                        ) THEN sc.constraint_variable
                    END) AS domain_variable
                FROM targets t
                LEFT JOIN stratum_constraints sc
                    ON t.stratum_id = sc.stratum_id
                GROUP BY t.target_id, t.stratum_id, t.variable,
                         t.value, t.period
                """))
            conn.commit()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.db_path)

    def setUp(self):
        engine = create_engine(self.db_uri)
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM targets"))
            conn.execute(text("DELETE FROM stratum_constraints"))
            conn.execute(text("DELETE FROM strata"))
            conn.commit()

    def _insert_test_data(self, strata, constraints, targets):
        engine = create_engine(self.db_uri)
        with engine.connect() as conn:
            for stratum_id, group_id in strata:
                conn.execute(
                    text("INSERT INTO strata VALUES (:sid)"),
                    {"sid": stratum_id},
                )
            for i, (stratum_id, var, op, val) in enumerate(constraints):
                conn.execute(
                    text(
                        "INSERT INTO stratum_constraints "
                        "VALUES (:cid, :sid, :var, :op, :val)"
                    ),
                    {
                        "cid": i + 1,
                        "sid": stratum_id,
                        "var": var,
                        "op": op,
                        "val": val,
                    },
                )
            for i, (
                stratum_id,
                variable,
                value,
                period,
            ) in enumerate(targets):
                conn.execute(
                    text(
                        "INSERT INTO targets "
                        "VALUES (:tid, :sid, :var, :val, :period)"
                    ),
                    {
                        "tid": i + 1,
                        "sid": stratum_id,
                        "var": variable,
                        "val": value,
                        "period": period,
                    },
                )
            conn.commit()

    def _make_builder(self, time_period=2024):
        return SparseMatrixBuilder(
            db_uri=self.db_uri,
            time_period=time_period,
            cds_to_calibrate=["601"],
        )

    # ---- Period selection tests ----

    def test_best_period_prefers_past(self):
        """Targets at 2022 and 2026 -> picks 2022 for time_period=2024."""
        self._insert_test_data(
            strata=[(1, 1)],
            constraints=[
                (1, "congressional_district_geoid", "=", "601"),
            ],
            targets=[
                (1, "snap", 1000, 2022),
                (1, "snap", 2000, 2026),
            ],
        )
        builder = self._make_builder(time_period=2024)
        df = builder._query_targets({"stratum_ids": [1]})
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["period"], 2022)
        self.assertEqual(df.iloc[0]["value"], 1000)

    def test_best_period_uses_future_when_no_past(self):
        """Target only at 2026 -> picks 2026 for time_period=2024."""
        self._insert_test_data(
            strata=[(1, 1)],
            constraints=[
                (1, "congressional_district_geoid", "=", "601"),
            ],
            targets=[
                (1, "snap", 5000, 2026),
            ],
        )
        builder = self._make_builder(time_period=2024)
        df = builder._query_targets({"stratum_ids": [1]})
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["period"], 2026)

    def test_best_period_exact_match(self):
        """Targets at 2022, 2024, 2026 -> picks 2024 exactly."""
        self._insert_test_data(
            strata=[(1, 1)],
            constraints=[
                (1, "congressional_district_geoid", "=", "601"),
            ],
            targets=[
                (1, "snap", 1000, 2022),
                (1, "snap", 1500, 2024),
                (1, "snap", 2000, 2026),
            ],
        )
        builder = self._make_builder(time_period=2024)
        df = builder._query_targets({"stratum_ids": [1]})
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["period"], 2024)
        self.assertEqual(df.iloc[0]["value"], 1500)

    def test_independent_per_stratum_and_variable(self):
        """Different strata/variables select independently."""
        self._insert_test_data(
            strata=[(1, 1), (2, 1)],
            constraints=[
                (1, "congressional_district_geoid", "=", "601"),
                (2, "congressional_district_geoid", "=", "601"),
            ],
            targets=[
                (1, "snap", 1000, 2024),
                (1, "snap", 800, 2022),
                (2, "person_count", 500, 2022),
                (2, "person_count", 600, 2026),
            ],
        )
        builder = self._make_builder(time_period=2024)
        df = builder._query_targets({"stratum_ids": [1, 2]})
        self.assertEqual(len(df), 2)
        snap_row = df[df["variable"] == "snap"].iloc[0]
        self.assertEqual(snap_row["period"], 2024)
        count_row = df[df["variable"] == "person_count"].iloc[0]
        self.assertEqual(count_row["period"], 2022)

    # ---- Uprating info tests ----

    def test_cpi_uprating_for_dollar_vars(self):
        builder = self._make_builder(time_period=2024)
        factors = {(2022, "cpi"): 1.06, (2022, "pop"): 1.01}
        factor, type_ = builder._get_uprating_info("snap", 2022, factors)
        self.assertAlmostEqual(factor, 1.06)
        self.assertEqual(type_, "cpi")

    def test_pop_uprating_for_count_vars(self):
        builder = self._make_builder(time_period=2024)
        factors = {(2022, "cpi"): 1.06, (2022, "pop"): 1.01}
        factor, type_ = builder._get_uprating_info(
            "person_count", 2022, factors
        )
        self.assertAlmostEqual(factor, 1.01)
        self.assertEqual(type_, "pop")

    def test_no_uprating_for_current_period(self):
        builder = self._make_builder(time_period=2024)
        factors = {(2024, "cpi"): 1.0, (2024, "pop"): 1.0}
        factor, type_ = builder._get_uprating_info("snap", 2024, factors)
        self.assertAlmostEqual(factor, 1.0)
        self.assertEqual(type_, "none")

    def test_pop_uprating_households_variable(self):
        builder = self._make_builder(time_period=2024)
        factors = {(2022, "cpi"): 1.06, (2022, "pop"): 1.02}
        factor, type_ = builder._get_uprating_info("households", 2022, factors)
        self.assertAlmostEqual(factor, 1.02)
        self.assertEqual(type_, "pop")

    def test_pop_uprating_tax_units_variable(self):
        builder = self._make_builder(time_period=2024)
        factors = {(2022, "cpi"): 1.06, (2022, "pop"): 1.02}
        factor, type_ = builder._get_uprating_info("tax_units", 2022, factors)
        self.assertAlmostEqual(factor, 1.02)
        self.assertEqual(type_, "pop")

    def test_missing_factor_defaults_to_1(self):
        builder = self._make_builder(time_period=2024)
        factors = {}
        factor, type_ = builder._get_uprating_info("snap", 2020, factors)
        self.assertAlmostEqual(factor, 1.0)
        self.assertEqual(type_, "cpi")
