"""Tests for UnifiedMatrixBuilder.

Ports uprating/hierarchical tests from test_hierarchical_uprating.py.
Uses in-memory SQLite DBs, self-contained.
"""

import unittest
import tempfile
import os

import pandas as pd
from sqlalchemy import create_engine, text

from policyengine_us_data.calibration.unified_matrix_builder import (
    UnifiedMatrixBuilder,
)
from policyengine_us_data.db.create_database_tables import (
    TARGET_OVERVIEW_VIEW,
)


def _create_test_db(db_path):
    db_uri = f"sqlite:///{db_path}"
    engine = create_engine(db_uri)

    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE strata ("
                "stratum_id INTEGER PRIMARY KEY, "
                "definition_hash VARCHAR(64), "
                "parent_stratum_id INTEGER, "
                "notes VARCHAR)"
            )
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
        conn.execute(text(TARGET_OVERVIEW_VIEW))
        conn.commit()

    return db_uri, engine


def _insert_aca_ptc_data(engine):
    with engine.connect() as conn:
        strata = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for sid in strata:
            conn.execute(
                text(
                    "INSERT INTO strata "
                    "(stratum_id, parent_stratum_id) "
                    "VALUES (:sid, :parent)"
                ),
                {
                    "sid": sid,
                    "parent": None if sid == 1 else 1,
                },
            )

        constraints = [
            (1, 1, "aca_ptc", ">", "0"),
            (2, 2, "aca_ptc", ">", "0"),
            (3, 2, "state_fips", "=", "6"),
            (4, 3, "aca_ptc", ">", "0"),
            (5, 3, "state_fips", "=", "37"),
            (6, 4, "aca_ptc", ">", "0"),
            (7, 4, "congressional_district_geoid", "=", "601"),
            (8, 5, "aca_ptc", ">", "0"),
            (9, 5, "congressional_district_geoid", "=", "602"),
            (10, 6, "aca_ptc", ">", "0"),
            (11, 6, "congressional_district_geoid", "=", "603"),
            (12, 7, "aca_ptc", ">", "0"),
            (13, 7, "congressional_district_geoid", "=", "3701"),
            (14, 8, "aca_ptc", ">", "0"),
            (15, 8, "congressional_district_geoid", "=", "3702"),
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

        targets = [
            (1, 1, "aca_ptc", 10000.0, 2022),
            (2, 1, "tax_unit_count", 500.0, 2022),
            (3, 2, "aca_ptc", 6000.0, 2022),
            (4, 2, "tax_unit_count", 300.0, 2022),
            (5, 3, "aca_ptc", 4000.0, 2022),
            (6, 3, "tax_unit_count", 200.0, 2022),
            (7, 4, "aca_ptc", 2000.0, 2022),
            (8, 5, "aca_ptc", 2500.0, 2022),
            (9, 6, "aca_ptc", 1500.0, 2022),
            (10, 4, "tax_unit_count", 100.0, 2022),
            (11, 5, "tax_unit_count", 120.0, 2022),
            (12, 6, "tax_unit_count", 80.0, 2022),
            (13, 7, "aca_ptc", 2200.0, 2022),
            (14, 8, "aca_ptc", 1800.0, 2022),
            (15, 7, "tax_unit_count", 110.0, 2022),
            (16, 8, "tax_unit_count", 90.0, 2022),
            (17, 9, "person_count", 19743689.0, 2024),
        ]
        for tid, sid, var, val, period in targets:
            conn.execute(
                text(
                    "INSERT INTO targets "
                    "VALUES (:tid, :sid, :var, :val, "
                    ":period, 1)"
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


class TestQueryTargets(unittest.TestCase):
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
        return UnifiedMatrixBuilder(
            db_uri=self.db_uri,
            time_period=time_period,
        )

    def test_domain_variables_filter(self):
        b = self._make_builder()
        df = b._query_targets({"domain_variables": ["aca_ptc"]})
        self.assertGreater(len(df), 0)
        self.assertIn("geo_level", df.columns)
        self.assertIn("geographic_id", df.columns)
        self.assertIn("domain_variable", df.columns)

    def test_all_geo_levels_returned(self):
        b = self._make_builder()
        df = b._query_targets({"domain_variables": ["aca_ptc"]})
        geo_levels = set(df["geo_level"].unique())
        self.assertEqual(geo_levels, {"national", "state", "district"})

    def test_best_period_selection(self):
        b = self._make_builder(time_period=2024)
        df = b._query_targets({"domain_variables": ["aca_ptc"]})
        aca = df[df["variable"] == "aca_ptc"]
        self.assertTrue((aca["period"] == 2022).all())
        cms = df[df["variable"] == "person_count"]
        self.assertEqual(len(cms), 1)
        self.assertEqual(cms.iloc[0]["period"], 2024)

    def test_geographic_id_populated(self):
        b = self._make_builder()
        df = b._query_targets({"domain_variables": ["aca_ptc"]})
        national = df[df["geo_level"] == "national"]
        self.assertTrue((national["geographic_id"] == "US").all())
        state_ca = df[
            (df["geo_level"] == "state") & (df["geographic_id"] == "6")
        ]
        self.assertGreater(len(state_ca), 0)


class TestHierarchicalUprating(unittest.TestCase):
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
        return UnifiedMatrixBuilder(
            db_uri=self.db_uri,
            time_period=time_period,
        )

    def _get_targets_with_uprating(self, cpi_factor=1.1, pop_factor=1.02):
        b = self._make_builder(time_period=2024)
        df = b._query_targets({"domain_variables": ["aca_ptc"]})
        factors = {
            (2022, "cpi"): cpi_factor,
            (2022, "pop"): pop_factor,
        }
        df["original_value"] = df["value"].copy()
        df["uprating_factor"] = df.apply(
            lambda row: b._get_uprating_info(
                row["variable"], row["period"], factors
            )[0],
            axis=1,
        )
        df["value"] = df["original_value"] * df["uprating_factor"]
        return b, df, factors

    def test_cd_sums_match_uprated_state(self):
        b, df, factors = self._get_targets_with_uprating(
            cpi_factor=1.1, pop_factor=1.02
        )
        result = b._apply_hierarchical_uprating(df, ["aca_ptc"], factors)
        csv_factors = b._load_aca_ptc_factors()

        for var, sf, orig in [
            ("aca_ptc", 6, 6000.0),
            ("aca_ptc", 37, 4000.0),
            ("tax_unit_count", 6, 300.0),
            ("tax_unit_count", 37, 200.0),
        ]:
            expected = orig * csv_factors[sf][var]
            cd_rows = result[
                (result["variable"] == var)
                & (result["geo_level"] == "district")
                & (
                    result["geographic_id"].apply(
                        lambda g, s=sf: (
                            int(g) // 100 == s if g.isdigit() else False
                        )
                    )
                )
            ]
            self.assertAlmostEqual(
                cd_rows["value"].sum(),
                expected,
                places=2,
                msg=f"{var} state {sf}",
            )

    def test_national_and_state_rows_dropped(self):
        b, df, factors = self._get_targets_with_uprating()
        result = b._apply_hierarchical_uprating(df, ["aca_ptc"], factors)
        irs_national = result[
            (result["geo_level"] == "national") & (result["period"] != 2024)
        ]
        self.assertEqual(len(irs_national), 0)
        state_rows = result[result["geo_level"] == "state"]
        self.assertEqual(len(state_rows), 0)

    def test_cms_person_count_preserved(self):
        b, df, factors = self._get_targets_with_uprating()
        result = b._apply_hierarchical_uprating(df, ["aca_ptc"], factors)
        cms = result[
            (result["variable"] == "person_count") & (result["period"] == 2024)
        ]
        self.assertEqual(len(cms), 1)
        self.assertAlmostEqual(cms.iloc[0]["value"], 19743689.0, places=0)

    def test_hif_is_one_when_cds_sum_to_state(self):
        b, df, factors = self._get_targets_with_uprating(cpi_factor=1.15)
        result = b._apply_hierarchical_uprating(df, ["aca_ptc"], factors)
        cd_aca = result[
            (result["variable"] == "aca_ptc")
            & (result["geo_level"] == "district")
        ]
        for _, row in cd_aca.iterrows():
            self.assertAlmostEqual(row["hif"], 1.0, places=6)

    def test_non_hierarchical_rows_untouched(self):
        b, df, factors = self._get_targets_with_uprating()
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
        df2 = pd.concat([df, extra], ignore_index=True)
        result = b._apply_hierarchical_uprating(df2, ["aca_ptc"], factors)
        snap = result[result["domain_variable"] == "snap"]
        self.assertEqual(len(snap), 1)
        self.assertEqual(snap.iloc[0]["value"], 5000.0)


class TestGetStateUpratingFactors(unittest.TestCase):
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

    def test_aca_ptc_uses_csv_factors(self):
        b = UnifiedMatrixBuilder(db_uri=self.db_uri, time_period=2024)
        df = b._query_targets({"domain_variables": ["aca_ptc"]})
        nf = {(2022, "cpi"): 1.08, (2022, "pop"): 1.015}
        df["original_value"] = df["value"].copy()

        result = b._get_state_uprating_factors("aca_ptc", df, nf)
        self.assertIn(6, result)
        self.assertIn(37, result)

    def test_non_aca_uses_national_factors(self):
        b = UnifiedMatrixBuilder(db_uri=self.db_uri, time_period=2024)
        df = pd.DataFrame(
            [
                {
                    "domain_variable": "snap",
                    "geo_level": "state",
                    "geographic_id": "6",
                    "variable": "snap",
                    "period": 2022,
                    "value": 1000.0,
                    "original_value": 1000.0,
                },
                {
                    "domain_variable": "snap",
                    "geo_level": "state",
                    "geographic_id": "6",
                    "variable": "household_count",
                    "period": 2022,
                    "value": 500.0,
                    "original_value": 500.0,
                },
            ]
        )
        nf = {(2022, "cpi"): 1.08, (2022, "pop"): 1.015}
        result = b._get_state_uprating_factors("snap", df, nf)
        self.assertIn(6, result)
        self.assertAlmostEqual(result[6]["snap"], 1.08)
        self.assertAlmostEqual(result[6]["household_count"], 1.015)


class TestCountTargetDetection(unittest.TestCase):
    def test_endswith_count(self):
        count_vars = [
            "person_count",
            "tax_unit_count",
            "household_count",
        ]
        value_vars = ["snap", "aca_ptc", "income_tax"]
        for v in count_vars:
            self.assertTrue(
                v.endswith("_count"),
                f"{v} should be detected as count",
            )
        for v in value_vars:
            self.assertFalse(
                v.endswith("_count"),
                f"{v} should not be a count target",
            )


if __name__ == "__main__":
    unittest.main()
