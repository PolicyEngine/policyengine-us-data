"""
Tests for hierarchical uprating and CD reconciliation.
"""

import unittest
import tempfile
import os
import pandas as pd
from sqlalchemy import create_engine, text

from policyengine_us_data.datasets.cps.local_area_calibration.sparse_matrix_builder import (
    SparseMatrixBuilder,
)


def _create_test_db(db_path):
    """Create test DB with target_overview view and sample data."""
    db_uri = f"sqlite:///{db_path}"
    engine = create_engine(db_uri)

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
                "period INTEGER, "
                "active INTEGER DEFAULT 1)"
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
                t.active,
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
                     t.value, t.period, t.active
            """))
        conn.commit()

    return db_uri, engine


def _insert_aca_ptc_data(engine):
    """Insert ACA PTC test data at national/state/district levels.

    State 6 (CA): 3 CDs (601, 602, 603)
    State 37 (NC): 2 CDs (3701, 3702)

    All IRS data at period=2022.
    One CMS national person_count at period=2024.
    """
    with engine.connect() as conn:
        # Strata: national(1), state CA(2), state NC(3),
        # CDs: 601(4), 602(5), 603(6), 3701(7), 3702(8)
        # CMS national(9)
        strata = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for sid in strata:
            conn.execute(
                text("INSERT INTO strata VALUES (:sid)"),
                {"sid": sid},
            )

        # Constraints
        constraints = [
            # National: aca_ptc > 0
            (1, 1, "aca_ptc", ">", "0"),
            # State CA: aca_ptc > 0, state_fips=6
            (2, 2, "aca_ptc", ">", "0"),
            (3, 2, "state_fips", "=", "6"),
            # State NC: aca_ptc > 0, state_fips=37
            (4, 3, "aca_ptc", ">", "0"),
            (5, 3, "state_fips", "=", "37"),
            # CD 601
            (6, 4, "aca_ptc", ">", "0"),
            (7, 4, "congressional_district_geoid", "=", "601"),
            # CD 602
            (8, 5, "aca_ptc", ">", "0"),
            (9, 5, "congressional_district_geoid", "=", "602"),
            # CD 603
            (10, 6, "aca_ptc", ">", "0"),
            (11, 6, "congressional_district_geoid", "=", "603"),
            # CD 3701
            (12, 7, "aca_ptc", ">", "0"),
            (13, 7, "congressional_district_geoid", "=", "3701"),
            # CD 3702
            (14, 8, "aca_ptc", ">", "0"),
            (15, 8, "congressional_district_geoid", "=", "3702"),
            # CMS national: aca_ptc > 0
            (16, 9, "aca_ptc", ">", "0"),
        ]
        for cid, sid, var, op, val in constraints:
            conn.execute(
                text(
                    "INSERT INTO stratum_constraints "
                    "VALUES (:cid, :sid, :var, :op, :val)"
                ),
                {
                    "cid": cid,
                    "sid": sid,
                    "var": var,
                    "op": op,
                    "val": val,
                },
            )

        # Targets
        targets = [
            # National aca_ptc 2022
            (1, 1, "aca_ptc", 10000.0, 2022),
            # National tax_unit_count 2022
            (2, 1, "tax_unit_count", 500.0, 2022),
            # State CA aca_ptc 2022: 6000
            (3, 2, "aca_ptc", 6000.0, 2022),
            # State CA tax_unit_count 2022: 300
            (4, 2, "tax_unit_count", 300.0, 2022),
            # State NC aca_ptc 2022: 4000
            (5, 3, "aca_ptc", 4000.0, 2022),
            # State NC tax_unit_count 2022: 200
            (6, 3, "tax_unit_count", 200.0, 2022),
            # CD 601 aca_ptc 2022: 2000
            (7, 4, "aca_ptc", 2000.0, 2022),
            # CD 602 aca_ptc 2022: 2500
            (8, 5, "aca_ptc", 2500.0, 2022),
            # CD 603 aca_ptc 2022: 1500
            (9, 6, "aca_ptc", 1500.0, 2022),
            # CD 601 tax_unit_count 2022: 100
            (10, 4, "tax_unit_count", 100.0, 2022),
            # CD 602 tax_unit_count 2022: 120
            (11, 5, "tax_unit_count", 120.0, 2022),
            # CD 603 tax_unit_count 2022: 80
            (12, 6, "tax_unit_count", 80.0, 2022),
            # CD 3701 aca_ptc 2022: 2200
            (13, 7, "aca_ptc", 2200.0, 2022),
            # CD 3702 aca_ptc 2022: 1800
            (14, 8, "aca_ptc", 1800.0, 2022),
            # CD 3701 tax_unit_count 2022: 110
            (15, 7, "tax_unit_count", 110.0, 2022),
            # CD 3702 tax_unit_count 2022: 90
            (16, 8, "tax_unit_count", 90.0, 2022),
            # CMS national person_count 2024
            (17, 9, "person_count", 19743689.0, 2024),
        ]
        for tid, sid, var, val, period in targets:
            conn.execute(
                text(
                    "INSERT INTO targets "
                    "VALUES (:tid, :sid, :var, :val, :period, 1)"
                ),
                {
                    "tid": tid,
                    "sid": sid,
                    "var": var,
                    "val": val,
                    "period": period,
                },
            )
        conn.commit()


class TestQueryTargetsOverview(unittest.TestCase):
    """Test _query_targets_overview path."""

    @classmethod
    def setUpClass(cls):
        cls.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.db_path = cls.temp_db.name
        cls.temp_db.close()
        cls.db_uri, cls.engine = _create_test_db(cls.db_path)
        _insert_aca_ptc_data(cls.engine)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.db_path)

    def _make_builder(self, time_period=2024):
        return SparseMatrixBuilder(
            db_uri=self.db_uri,
            time_period=time_period,
            cds_to_calibrate=["601", "602", "603", "3701", "3702"],
        )

    def test_domain_variables_filter(self):
        builder = self._make_builder()
        df = builder._query_targets({"domain_variables": ["aca_ptc"]})
        self.assertGreater(len(df), 0)
        self.assertIn("geo_level", df.columns)
        self.assertIn("geographic_id", df.columns)
        self.assertIn("domain_variable", df.columns)

    def test_all_geo_levels_returned(self):
        builder = self._make_builder()
        df = builder._query_targets({"domain_variables": ["aca_ptc"]})
        geo_levels = set(df["geo_level"].unique())
        self.assertEqual(geo_levels, {"national", "state", "district"})

    def test_best_period_selection(self):
        """All aca_ptc targets at 2022, CMS at 2024."""
        builder = self._make_builder(time_period=2024)
        df = builder._query_targets({"domain_variables": ["aca_ptc"]})
        aca_rows = df[df["variable"] == "aca_ptc"]
        self.assertTrue((aca_rows["period"] == 2022).all())

        cms_rows = df[df["variable"] == "person_count"]
        self.assertEqual(len(cms_rows), 1)
        self.assertEqual(cms_rows.iloc[0]["period"], 2024)

    def test_geographic_id_populated(self):
        builder = self._make_builder()
        df = builder._query_targets({"domain_variables": ["aca_ptc"]})
        national = df[df["geo_level"] == "national"]
        self.assertTrue((national["geographic_id"] == "US").all())

        state_ca = df[
            (df["geo_level"] == "state") & (df["geographic_id"] == "6")
        ]
        self.assertGreater(len(state_ca), 0)

        district_601 = df[
            (df["geo_level"] == "district") & (df["geographic_id"] == "601")
        ]
        self.assertGreater(len(district_601), 0)


class TestHierarchicalUprating(unittest.TestCase):
    """Test _apply_hierarchical_uprating logic."""

    @classmethod
    def setUpClass(cls):
        cls.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.db_path = cls.temp_db.name
        cls.temp_db.close()
        cls.db_uri, cls.engine = _create_test_db(cls.db_path)
        _insert_aca_ptc_data(cls.engine)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.db_path)

    def _make_builder(self, time_period=2024):
        return SparseMatrixBuilder(
            db_uri=self.db_uri,
            time_period=time_period,
            cds_to_calibrate=["601", "602", "603", "3701", "3702"],
        )

    def _get_targets_with_uprating(self, cpi_factor=1.1, pop_factor=1.02):
        builder = self._make_builder(time_period=2024)
        df = builder._query_targets({"domain_variables": ["aca_ptc"]})
        factors = {
            (2022, "cpi"): cpi_factor,
            (2022, "pop"): pop_factor,
        }
        df["original_value"] = df["value"].copy()
        df["uprating_factor"] = df.apply(
            lambda row: builder._get_uprating_info(
                row["variable"], row["period"], factors
            )[0],
            axis=1,
        )
        df["value"] = df["original_value"] * df["uprating_factor"]
        return builder, df, factors

    def test_cd_sums_match_uprated_state_totals(self):
        """After reconciliation, CD sums must equal uprated state."""
        builder, df, factors = self._get_targets_with_uprating(
            cpi_factor=1.1, pop_factor=1.02
        )
        # Before hierarchical: store expected state totals
        state_ca_aca = 6000.0 * 1.1
        state_nc_aca = 4000.0 * 1.1
        state_ca_tu = 300.0 * 1.02
        state_nc_tu = 200.0 * 1.02

        result = builder._apply_hierarchical_uprating(df, ["aca_ptc"], factors)

        # Only district rows should remain for aca_ptc domain
        # (plus CMS person_count at national)
        for var, state_fips, expected_total in [
            ("aca_ptc", 6, state_ca_aca),
            ("aca_ptc", 37, state_nc_aca),
            ("tax_unit_count", 6, state_ca_tu),
            ("tax_unit_count", 37, state_nc_tu),
        ]:
            cd_rows = result[
                (result["variable"] == var)
                & (result["geo_level"] == "district")
                & (
                    result["geographic_id"].apply(
                        lambda g, s=state_fips: (
                            int(g) // 100 == s if g.isdigit() else False
                        )
                    )
                )
            ]
            cd_sum = cd_rows["value"].sum()
            self.assertAlmostEqual(
                cd_sum,
                expected_total,
                places=2,
                msg=f"CD sum for {var} state {state_fips}",
            )

    def test_national_and_state_rows_dropped(self):
        """IRS national and state rows (period!=2024) are dropped."""
        builder, df, factors = self._get_targets_with_uprating()
        result = builder._apply_hierarchical_uprating(df, ["aca_ptc"], factors)

        irs_national = result[
            (result["geo_level"] == "national") & (result["period"] != 2024)
        ]
        self.assertEqual(len(irs_national), 0)

        state_rows = result[result["geo_level"] == "state"]
        self.assertEqual(len(state_rows), 0)

    def test_cms_person_count_preserved(self):
        """CMS national person_count (period=2024) is NOT dropped."""
        builder, df, factors = self._get_targets_with_uprating()
        result = builder._apply_hierarchical_uprating(df, ["aca_ptc"], factors)

        cms = result[
            (result["variable"] == "person_count") & (result["period"] == 2024)
        ]
        self.assertEqual(len(cms), 1)
        self.assertAlmostEqual(cms.iloc[0]["value"], 19743689.0, places=0)

    def test_reconciliation_factor_column(self):
        """Diagnostic reconciliation_factor column is populated."""
        builder, df, factors = self._get_targets_with_uprating(cpi_factor=1.1)
        result = builder._apply_hierarchical_uprating(df, ["aca_ptc"], factors)

        cd_aca = result[
            (result["variable"] == "aca_ptc")
            & (result["geo_level"] == "district")
        ]
        self.assertTrue(cd_aca["reconciliation_factor"].notna().all())

    def test_uniform_reconciliation_with_uniform_uprating(self):
        """With uniform national factors, recon factor == CPI factor."""
        cpi_factor = 1.15
        builder, df, factors = self._get_targets_with_uprating(
            cpi_factor=cpi_factor
        )
        result = builder._apply_hierarchical_uprating(df, ["aca_ptc"], factors)

        cd_aca = result[
            (result["variable"] == "aca_ptc")
            & (result["geo_level"] == "district")
        ]
        for _, row in cd_aca.iterrows():
            self.assertAlmostEqual(
                row["reconciliation_factor"],
                cpi_factor,
                places=6,
                msg=(
                    f"CD {row['geographic_id']} recon factor "
                    f"should equal CPI factor"
                ),
            )

    def test_no_data_loss_for_non_hierarchical_rows(self):
        """Rows not in hierarchical_domains are untouched."""
        builder, df, factors = self._get_targets_with_uprating()

        # Add a non-hierarchical row
        extra = pd.DataFrame(
            [
                {
                    "target_id": 999,
                    "stratum_id": 999,
                    "variable": "snap",
                    "value": 5000.0,
                    "period": 2022,
                    "geo_level": "national",
                    "geographic_id": "US",
                    "domain_variable": "snap",
                    "original_value": 5000.0,
                    "uprating_factor": 1.1,
                }
            ]
        )
        df_with_snap = pd.concat([df, extra], ignore_index=True)

        result = builder._apply_hierarchical_uprating(
            df_with_snap, ["aca_ptc"], factors
        )

        snap_rows = result[result["domain_variable"] == "snap"]
        self.assertEqual(len(snap_rows), 1)
        self.assertEqual(snap_rows.iloc[0]["value"], 5000.0)


class TestGetStateUpratingFactors(unittest.TestCase):
    """Test _get_state_uprating_factors."""

    @classmethod
    def setUpClass(cls):
        cls.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.db_path = cls.temp_db.name
        cls.temp_db.close()
        cls.db_uri, cls.engine = _create_test_db(cls.db_path)
        _insert_aca_ptc_data(cls.engine)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.db_path)

    def test_returns_uniform_factors(self):
        builder = SparseMatrixBuilder(
            db_uri=self.db_uri,
            time_period=2024,
            cds_to_calibrate=["601"],
        )
        df = builder._query_targets({"domain_variables": ["aca_ptc"]})
        national_factors = {
            (2022, "cpi"): 1.08,
            (2022, "pop"): 1.015,
        }
        df["original_value"] = df["value"].copy()

        result = builder._get_state_uprating_factors(
            "aca_ptc", df, national_factors
        )

        # Should have factors for state 6 (CA) and 37 (NC)
        self.assertIn(6, result)
        self.assertIn(37, result)

        # aca_ptc is dollar -> CPI factor
        self.assertAlmostEqual(result[6]["aca_ptc"], 1.08)
        self.assertAlmostEqual(result[37]["aca_ptc"], 1.08)

        # tax_unit_count is count -> pop factor
        self.assertAlmostEqual(result[6]["tax_unit_count"], 1.015)
        self.assertAlmostEqual(result[37]["tax_unit_count"], 1.015)


if __name__ == "__main__":
    unittest.main()
