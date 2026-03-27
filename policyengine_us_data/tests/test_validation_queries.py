"""
Tests that validation SQL queries are compatible with the current DB schema.

Uses create_database() for the real schema, then exercises the query functions
from validate_staging.py to catch column/view mismatches before production.
"""

import os
import tempfile
import unittest

from sqlmodel import Session

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    create_database,
)


def _make_stratum(
    session: Session,
    constraints: list,
    parent_id: int = None,
    notes: str = None,
) -> int:
    constraint_objs = [
        StratumConstraint(constraint_variable=var, operation=op, value=val)
        for var, op, val in constraints
    ]
    stratum = Stratum(
        definition_hash="placeholder",
        parent_stratum_id=parent_id,
        notes=notes,
        constraints_rel=constraint_objs,
    )
    session.add(stratum)
    session.commit()
    session.refresh(stratum)
    return stratum.stratum_id


def _add_target(
    session: Session,
    stratum_id: int,
    variable: str,
    period: int,
    value: float,
    active: bool = True,
) -> Target:
    target = Target(
        stratum_id=stratum_id,
        variable=variable,
        period=period,
        value=value,
        active=active,
    )
    session.add(target)
    session.commit()
    session.refresh(target)
    return target


class TestValidationQueries(unittest.TestCase):
    """Verify validate_staging SQL queries work against current schema."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.db_path = cls._tmp.name
        cls._tmp.close()

        cls.db_uri = f"sqlite:///{cls.db_path}"
        cls.engine = create_database(cls.db_uri)

        with Session(cls.engine) as session:
            cls.national_id = _make_stratum(session, [], notes="national")

            cls.state_ca_id = _make_stratum(
                session,
                [("state_fips", "==", "6")],
                notes="CA",
            )

            cls.cd_601_id = _make_stratum(
                session,
                [
                    ("state_fips", "==", "6"),
                    ("congressional_district_geoid", "==", "601"),
                ],
                notes="CA-01",
            )

            cls.snap_domain_id = _make_stratum(
                session,
                [("snap", ">", "0"), ("state_fips", "==", "6")],
                parent_id=cls.state_ca_id,
                notes="CA SNAP",
            )

            _add_target(session, cls.national_id, "person_count", 2024, 330_000_000)
            _add_target(session, cls.state_ca_id, "person_count", 2024, 39_000_000)
            _add_target(session, cls.cd_601_id, "person_count", 2024, 750_000)
            _add_target(session, cls.snap_domain_id, "snap", 2024, 5_000_000)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        os.unlink(cls.db_path)

    def test_query_all_active_targets(self):
        from policyengine_us_data.calibration.validate_staging import (
            _query_all_active_targets,
        )

        df = _query_all_active_targets(self.engine, 2024)
        self.assertGreater(len(df), 0)
        expected_cols = {
            "target_id",
            "stratum_id",
            "variable",
            "value",
            "period",
            "geo_level",
            "geographic_id",
            "domain_variable",
        }
        self.assertTrue(
            expected_cols.issubset(set(df.columns)),
            f"Missing columns: {expected_cols - set(df.columns)}",
        )

    def test_batch_stratum_constraints(self):
        from policyengine_us_data.calibration.validate_staging import (
            _batch_stratum_constraints,
        )

        stratum_ids = [self.national_id, self.state_ca_id, self.cd_601_id]
        result = _batch_stratum_constraints(self.engine, stratum_ids)
        self.assertIsInstance(result, dict)
        for sid in stratum_ids:
            self.assertIn(sid, result)
            self.assertIsInstance(result[sid], list)

    def test_resolve_district_ids(self):
        from policyengine_us_data.calibration.validate_staging import (
            _resolve_district_ids,
        )

        result = _resolve_district_ids(self.engine, None)
        self.assertIsInstance(result, list)
        self.assertIn("601", result)

    def test_unified_matrix_builder_query_targets(self):
        from policyengine_us_data.calibration.unified_matrix_builder import (
            UnifiedMatrixBuilder,
        )

        builder = UnifiedMatrixBuilder(
            db_uri=self.db_uri,
            time_period=2024,
        )
        df = builder._query_targets({"variables": ["person_count"]})
        self.assertGreater(len(df), 0)
        self.assertIn("variable", df.columns)
        self.assertTrue((df["variable"] == "person_count").all())
        builder.engine.dispose()
