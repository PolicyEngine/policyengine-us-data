"""
Tests for concept ID building, constraint extraction, and deduplication.

These tests verify that:
1. Concept IDs are built consistently from variable + non-geo constraints
2. Constraints are correctly extracted from DataFrame rows
3. Deduplication correctly identifies and removes duplicates via the builder
"""

import unittest
import tempfile
import os
import pandas as pd
from sqlalchemy import create_engine, text

from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    build_concept_id,
    extract_constraints_from_row,
)
from policyengine_us_data.datasets.cps.local_area_calibration.sparse_matrix_builder import (
    SparseMatrixBuilder,
)


class TestBuildConceptId(unittest.TestCase):
    """Test concept ID building from variable + constraints."""

    def test_variable_only(self):
        """Test concept ID with no constraints."""
        result = build_concept_id("snap", [])
        self.assertEqual(result, "snap")

    def test_single_constraint(self):
        """Test concept ID with single constraint."""
        result = build_concept_id("snap", ["snap>0"])
        self.assertEqual(result, "snap_snap_gt_0")

    def test_multiple_constraints_sorted(self):
        """Test that constraints are sorted for consistency."""
        # Order shouldn't matter - result should be the same
        result1 = build_concept_id("person_count", ["age>=5", "age<18"])
        result2 = build_concept_id("person_count", ["age<18", "age>=5"])
        self.assertEqual(result1, result2)
        self.assertEqual(result1, "person_count_age_lt_18_age_gte_5")

    def test_operator_normalization(self):
        """Test that operators are normalized correctly."""
        self.assertIn("_gte_", build_concept_id("x", ["a>=1"]))
        self.assertIn("_lte_", build_concept_id("x", ["a<=1"]))
        self.assertIn("_gt_", build_concept_id("x", ["a>1"]))
        self.assertIn("_lt_", build_concept_id("x", ["a<1"]))
        self.assertIn("_eq_", build_concept_id("x", ["a==1"]))
        self.assertIn("_eq_", build_concept_id("x", ["a=1"]))

    def test_spaces_removed(self):
        """Test that spaces are removed from constraints."""
        result = build_concept_id("x", ["age >= 5"])
        self.assertNotIn(" ", result)


class TestExtractConstraints(unittest.TestCase):
    """Test constraint extraction from DataFrame rows."""

    def test_no_constraint_info(self):
        """Test row without constraint_info column."""
        row = pd.Series({"variable": "snap", "value": 1000})
        result = extract_constraints_from_row(row)
        self.assertEqual(result, [])

    def test_null_constraint_info(self):
        """Test row with null constraint_info."""
        row = pd.Series(
            {"variable": "snap", "constraint_info": None, "value": 1000}
        )
        result = extract_constraints_from_row(row)
        self.assertEqual(result, [])

    def test_single_constraint(self):
        """Test row with single constraint."""
        row = pd.Series(
            {"variable": "snap", "constraint_info": "snap>0", "value": 1000}
        )
        result = extract_constraints_from_row(row)
        self.assertEqual(result, ["snap>0"])

    def test_multiple_constraints(self):
        """Test row with pipe-separated constraints."""
        row = pd.Series(
            {
                "variable": "person_count",
                "constraint_info": "age>=5|age<18",
                "value": 1000,
            }
        )
        result = extract_constraints_from_row(row)
        self.assertEqual(result, ["age>=5", "age<18"])

    def test_exclude_geo_constraints(self):
        """Test that geographic constraints are excluded by default."""
        row = pd.Series(
            {
                "variable": "person_count",
                "constraint_info": "age>=5|state_fips=6|age<18",
                "value": 1000,
            }
        )
        result = extract_constraints_from_row(row, exclude_geo=True)
        self.assertEqual(result, ["age>=5", "age<18"])
        self.assertNotIn("state_fips=6", result)

    def test_include_geo_constraints(self):
        """Test that geographic constraints can be included."""
        row = pd.Series(
            {
                "variable": "person_count",
                "constraint_info": "age>=5|state_fips=6",
                "value": 1000,
            }
        )
        result = extract_constraints_from_row(row, exclude_geo=False)
        self.assertIn("state_fips=6", result)

    def test_exclude_cd_geoid(self):
        """Test that CD geoid constraints are excluded."""
        row = pd.Series(
            {
                "variable": "snap",
                "constraint_info": "snap>0|congressional_district_geoid=601",
                "value": 1000,
            }
        )
        result = extract_constraints_from_row(row, exclude_geo=True)
        self.assertEqual(result, ["snap>0"])

    def test_exclude_filer_constraint(self):
        """Test that tax_unit_is_filer constraint is excluded."""
        row = pd.Series(
            {
                "variable": "income_tax",
                "constraint_info": "tax_unit_is_filer=True|income>0",
                "value": 1000,
            }
        )
        result = extract_constraints_from_row(row, exclude_geo=True)
        self.assertEqual(result, ["income>0"])


class TestBuilderDeduplication(unittest.TestCase):
    """Test deduplication logic through SparseMatrixBuilder."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary database with test data."""
        cls.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.db_path = cls.temp_db.name
        cls.temp_db.close()

        cls.db_uri = f"sqlite:///{cls.db_path}"
        engine = create_engine(cls.db_uri)

        # Create schema
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE stratum_groups (
                    stratum_group_id INTEGER PRIMARY KEY,
                    name TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE strata (
                    stratum_id INTEGER PRIMARY KEY,
                    stratum_group_id INTEGER
                )
            """))
            conn.execute(text("""
                CREATE TABLE stratum_constraints (
                    constraint_id INTEGER PRIMARY KEY,
                    stratum_id INTEGER,
                    constraint_variable TEXT,
                    operation TEXT,
                    value TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE targets (
                    target_id INTEGER PRIMARY KEY,
                    stratum_id INTEGER,
                    variable TEXT,
                    value REAL,
                    period INTEGER
                )
            """))
            conn.commit()

    @classmethod
    def tearDownClass(cls):
        """Remove temporary database."""
        os.unlink(cls.db_path)

    def setUp(self):
        """Clear tables before each test."""
        engine = create_engine(self.db_uri)
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM targets"))
            conn.execute(text("DELETE FROM stratum_constraints"))
            conn.execute(text("DELETE FROM strata"))
            conn.execute(text("DELETE FROM stratum_groups"))
            conn.commit()

    def _insert_test_data(self, strata, constraints, targets):
        """Helper to insert test data into database."""
        engine = create_engine(self.db_uri)
        with engine.connect() as conn:
            # Insert stratum groups
            conn.execute(
                text("INSERT OR IGNORE INTO stratum_groups VALUES (1, 'test')")
            )

            # Insert strata
            for stratum_id, group_id in strata:
                conn.execute(
                    text("INSERT INTO strata VALUES (:sid, :gid)"),
                    {"sid": stratum_id, "gid": group_id},
                )

            # Insert constraints
            for i, (stratum_id, var, op, val) in enumerate(constraints):
                conn.execute(
                    text("""
                        INSERT INTO stratum_constraints
                        VALUES (:cid, :sid, :var, :op, :val)
                    """),
                    {
                        "cid": i + 1,
                        "sid": stratum_id,
                        "var": var,
                        "op": op,
                        "val": val,
                    },
                )

            # Insert targets
            for i, (stratum_id, variable, value, period) in enumerate(targets):
                conn.execute(
                    text("""
                        INSERT INTO targets
                        VALUES (:tid, :sid, :var, :val, :period)
                    """),
                    {
                        "tid": i + 1,
                        "sid": stratum_id,
                        "var": variable,
                        "val": value,
                        "period": period,
                    },
                )

            conn.commit()

    def test_no_duplicates_preserved(self):
        """Test that targets with different concepts are all preserved."""
        # Two different variables for the same CD - should NOT deduplicate
        self._insert_test_data(
            strata=[(1, 1), (2, 1)],
            constraints=[
                (1, "congressional_district_geoid", "=", "601"),
                (2, "congressional_district_geoid", "=", "601"),
            ],
            targets=[
                (1, "snap", 1000, 2023),
                (2, "medicaid", 2000, 2023),
            ],
        )

        builder = SparseMatrixBuilder(
            db_uri=self.db_uri,
            time_period=2023,
            cds_to_calibrate=["601"],
        )

        # Call _deduplicate_targets directly with prepared DataFrame
        targets_df = builder._query_targets({"stratum_group_ids": [1]})
        targets_df["geographic_id"] = targets_df["stratum_id"].apply(
            builder._get_geographic_id
        )
        targets_df["constraint_info"] = targets_df["stratum_id"].apply(
            builder._get_constraint_info
        )

        result = builder._deduplicate_targets(targets_df)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(builder.dedup_warnings), 0)

    def test_duplicate_same_geo_deduplicated(self):
        """Test that same concept at same geography is deduplicated."""
        # Same variable, same CD, different periods - should deduplicate
        self._insert_test_data(
            strata=[(1, 1), (2, 1)],
            constraints=[
                (1, "congressional_district_geoid", "=", "601"),
                (2, "congressional_district_geoid", "=", "601"),
            ],
            targets=[
                (1, "snap", 1000, 2023),
                (2, "snap", 1100, 2022),  # Same concept, same geo
            ],
        )

        builder = SparseMatrixBuilder(
            db_uri=self.db_uri,
            time_period=2023,
            cds_to_calibrate=["601"],
        )

        targets_df = builder._query_targets({"stratum_group_ids": [1]})
        targets_df["geographic_id"] = targets_df["stratum_id"].apply(
            builder._get_geographic_id
        )
        targets_df["constraint_info"] = targets_df["stratum_id"].apply(
            builder._get_constraint_info
        )

        result = builder._deduplicate_targets(targets_df)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(builder.dedup_warnings), 1)

    def test_same_concept_different_geos_preserved(self):
        """Test that same concept at different geos is NOT deduplicated."""
        # Same variable, different CDs - should NOT deduplicate
        self._insert_test_data(
            strata=[(1, 1), (2, 1)],
            constraints=[
                (1, "congressional_district_geoid", "=", "601"),
                (2, "congressional_district_geoid", "=", "602"),
            ],
            targets=[
                (1, "snap", 1000, 2023),
                (2, "snap", 1100, 2023),
            ],
        )

        builder = SparseMatrixBuilder(
            db_uri=self.db_uri,
            time_period=2023,
            cds_to_calibrate=["601", "602"],
        )

        targets_df = builder._query_targets({"stratum_group_ids": [1]})
        targets_df["geographic_id"] = targets_df["stratum_id"].apply(
            builder._get_geographic_id
        )
        targets_df["constraint_info"] = targets_df["stratum_id"].apply(
            builder._get_constraint_info
        )

        result = builder._deduplicate_targets(targets_df)

        self.assertEqual(len(result), 2)  # Both kept
        self.assertEqual(len(builder.dedup_warnings), 0)

    def test_different_constraints_different_concepts(self):
        """Test that different constraints create different concepts."""
        # Same variable but different age constraints - different concepts
        self._insert_test_data(
            strata=[(1, 1), (2, 1)],
            constraints=[
                (1, "congressional_district_geoid", "=", "601"),
                (1, "age", ">=", "5"),
                (1, "age", "<", "18"),
                (2, "congressional_district_geoid", "=", "601"),
                (2, "age", ">=", "18"),
                (2, "age", "<", "65"),
            ],
            targets=[
                (1, "person_count", 1000, 2023),
                (2, "person_count", 2000, 2023),
            ],
        )

        builder = SparseMatrixBuilder(
            db_uri=self.db_uri,
            time_period=2023,
            cds_to_calibrate=["601"],
        )

        targets_df = builder._query_targets({"stratum_group_ids": [1]})
        targets_df["geographic_id"] = targets_df["stratum_id"].apply(
            builder._get_geographic_id
        )
        targets_df["constraint_info"] = targets_df["stratum_id"].apply(
            builder._get_constraint_info
        )

        result = builder._deduplicate_targets(targets_df)

        self.assertEqual(len(result), 2)  # Different concepts
        self.assertEqual(len(builder.dedup_warnings), 0)

    def test_hierarchical_fallback_keeps_most_specific(self):
        """Test hierarchical fallback mode keeps CD over state over national."""
        # Same concept at CD, state, and national levels
        self._insert_test_data(
            strata=[(1, 1), (2, 1), (3, 1)],
            constraints=[
                (1, "congressional_district_geoid", "=", "601"),
                (2, "state_fips", "=", "6"),
                # stratum 3 has no geo constraint = national
            ],
            targets=[
                (1, "snap", 1200000, 2023),  # CD level
                (2, "snap", 15000000, 2023),  # State level
                (3, "snap", 110000000000, 2023),  # National level
            ],
        )

        builder = SparseMatrixBuilder(
            db_uri=self.db_uri,
            time_period=2023,
            cds_to_calibrate=["601"],
        )

        targets_df = builder._query_targets({"stratum_group_ids": [1]})
        targets_df["geographic_id"] = targets_df["stratum_id"].apply(
            builder._get_geographic_id
        )
        targets_df["constraint_info"] = targets_df["stratum_id"].apply(
            builder._get_constraint_info
        )

        result = builder._deduplicate_targets(
            targets_df, mode="hierarchical_fallback"
        )

        self.assertEqual(len(result), 1)
        # CD level should be kept (geo_priority=1)
        self.assertEqual(result.iloc[0]["geographic_id"], "601")
        self.assertEqual(result.iloc[0]["value"], 1200000)
