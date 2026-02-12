"""
Tests for SQL views (stratum_domain, target_overview) and
constraint-based lookup functions (get_geographic_strata,
get_all_cds_from_database, get_cd_index_mapping).

Uses create_database() to get the full production schema
including both views — no duplicated SQL.
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
from policyengine_us_data.utils.db import get_geographic_strata
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_all_cds_from_database,
    get_cd_index_mapping,
)


def _make_stratum(
    session: Session,
    constraints: list,
    parent_id: int = None,
    notes: str = None,
) -> int:
    """Insert a stratum with its constraints via the ORM.

    Args:
        session: Active SQLModel session.
        constraints: List of (variable, operation, value) tuples.
        parent_id: Optional parent stratum ID.
        notes: Optional notes.

    Returns:
        The stratum_id of the persisted Stratum.
    """
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
    """Insert a target row."""
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


class TestSchemaViewsAndLookups(unittest.TestCase):
    """Shared database with standard test data for all sub-tests."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.db_path = cls._tmp.name
        cls._tmp.close()

        cls.db_uri = f"sqlite:///{cls.db_path}"
        cls.engine = create_database(cls.db_uri)

        with Session(cls.engine) as session:
            # 1) National stratum (no constraints)
            cls.national_id = _make_stratum(session, [], notes="national")

            # 2-3) State strata
            cls.state_ca_id = _make_stratum(
                session,
                [("state_fips", "==", "6")],
                notes="CA",
            )
            cls.state_nc_id = _make_stratum(
                session,
                [("state_fips", "==", "37")],
                notes="NC",
            )

            # 4-6) District strata
            cls.cd_601_id = _make_stratum(
                session,
                [
                    ("state_fips", "==", "6"),
                    (
                        "congressional_district_geoid",
                        "==",
                        "601",
                    ),
                ],
                notes="CA-01",
            )
            cls.cd_602_id = _make_stratum(
                session,
                [
                    ("state_fips", "==", "6"),
                    (
                        "congressional_district_geoid",
                        "==",
                        "602",
                    ),
                ],
                notes="CA-02",
            )
            cls.cd_3701_id = _make_stratum(
                session,
                [
                    ("state_fips", "==", "37"),
                    (
                        "congressional_district_geoid",
                        "==",
                        "3701",
                    ),
                ],
                notes="NC-01",
            )

            # 7) Domain stratum: snap > 0, state_fips = 6
            cls.snap_domain_id = _make_stratum(
                session,
                [("snap", ">", "0"), ("state_fips", "==", "6")],
                parent_id=cls.state_ca_id,
                notes="CA SNAP",
            )

            # 8) Multi-constraint domain: age >= 18, age < 25,
            #    state_fips = 6
            cls.age_domain_id = _make_stratum(
                session,
                [
                    ("age", ">=", "18"),
                    ("age", "<", "25"),
                    ("state_fips", "==", "6"),
                ],
                parent_id=cls.state_ca_id,
                notes="CA young-adult",
            )

            # -- Targets --
            _add_target(
                session,
                cls.national_id,
                "person_count",
                2024,
                330_000_000,
            )
            _add_target(
                session,
                cls.state_ca_id,
                "person_count",
                2024,
                39_000_000,
            )
            _add_target(
                session,
                cls.state_nc_id,
                "person_count",
                2024,
                10_000_000,
            )
            _add_target(
                session,
                cls.cd_601_id,
                "person_count",
                2024,
                750_000,
            )
            _add_target(
                session,
                cls.snap_domain_id,
                "snap",
                2024,
                5_000_000,
                active=True,
            )
            _add_target(
                session,
                cls.snap_domain_id,
                "household_count",
                2024,
                100_000,
                active=False,
            )

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        os.unlink(cls.db_path)

    # ----------------------------------------------------------------
    # stratum_domain view
    # ----------------------------------------------------------------

    def _query_stratum_domain(self):
        from sqlalchemy import text

        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM stratum_domain")
            ).fetchall()
        return rows

    def test_geographic_stratum_excluded(self):
        """National/state/district strata produce no rows."""
        rows = self._query_stratum_domain()
        geo_ids = {
            self.national_id,
            self.state_ca_id,
            self.state_nc_id,
            self.cd_601_id,
            self.cd_602_id,
            self.cd_3701_id,
        }
        domain_stratum_ids = {r[0] for r in rows}
        self.assertTrue(
            domain_stratum_ids.isdisjoint(geo_ids),
            "Geographic strata should not appear in " "stratum_domain",
        )

    def test_single_domain_variable(self):
        """SNAP stratum returns domain_variable = 'snap'."""
        rows = self._query_stratum_domain()
        snap_rows = [r for r in rows if r[0] == self.snap_domain_id]
        domain_vars = {r[1] for r in snap_rows}
        self.assertIn("snap", domain_vars)

    def test_multi_constraint_domain(self):
        """Age stratum with age >= 18 and age < 25 returns
        domain_variable = 'age'."""
        rows = self._query_stratum_domain()
        age_rows = [r for r in rows if r[0] == self.age_domain_id]
        domain_vars = {r[1] for r in age_rows}
        self.assertIn("age", domain_vars)
        # Even though there are two age constraints,
        # DISTINCT means one 'age' row
        self.assertEqual(len(domain_vars), 1)

    def test_geographic_constraints_filtered(self):
        """state_fips, congressional_district_geoid,
        tax_unit_is_filer, ucgid_str are excluded."""
        rows = self._query_stratum_domain()
        all_domain_vars = {r[1] for r in rows}
        excluded = {
            "state_fips",
            "congressional_district_geoid",
            "tax_unit_is_filer",
            "ucgid_str",
        }
        self.assertTrue(
            all_domain_vars.isdisjoint(excluded),
            f"Found excluded vars: " f"{all_domain_vars & excluded}",
        )

    # ----------------------------------------------------------------
    # target_overview view
    # ----------------------------------------------------------------

    def _query_target_overview(self):
        from sqlalchemy import text

        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM target_overview")
            ).fetchall()
        return rows

    def _overview_columns(self):
        from sqlalchemy import text

        with self.engine.connect() as conn:
            cursor = conn.execute(
                text("SELECT * FROM target_overview LIMIT 0")
            )
            return [desc[0] for desc in cursor.cursor.description]

    def test_national_geo_level(self):
        """Target on national stratum -> geo_level='national',
        geographic_id='US'."""
        rows = self._query_target_overview()
        cols = self._overview_columns()
        sid_idx = cols.index("stratum_id")
        geo_level_idx = cols.index("geo_level")
        geo_id_idx = cols.index("geographic_id")

        nat_rows = [r for r in rows if r[sid_idx] == self.national_id]
        self.assertGreater(len(nat_rows), 0)
        for r in nat_rows:
            self.assertEqual(r[geo_level_idx], "national")
            self.assertEqual(r[geo_id_idx], "US")

    def test_state_geo_level(self):
        """Target on state stratum -> geo_level='state',
        geographic_id='6'."""
        rows = self._query_target_overview()
        cols = self._overview_columns()
        sid_idx = cols.index("stratum_id")
        geo_level_idx = cols.index("geo_level")
        geo_id_idx = cols.index("geographic_id")

        ca_rows = [r for r in rows if r[sid_idx] == self.state_ca_id]
        self.assertGreater(len(ca_rows), 0)
        for r in ca_rows:
            self.assertEqual(r[geo_level_idx], "state")
            self.assertEqual(r[geo_id_idx], "6")

    def test_district_geo_level(self):
        """Target on district stratum -> geo_level='district',
        geographic_id='601'."""
        rows = self._query_target_overview()
        cols = self._overview_columns()
        sid_idx = cols.index("stratum_id")
        geo_level_idx = cols.index("geo_level")
        geo_id_idx = cols.index("geographic_id")

        cd_rows = [r for r in rows if r[sid_idx] == self.cd_601_id]
        self.assertGreater(len(cd_rows), 0)
        for r in cd_rows:
            self.assertEqual(r[geo_level_idx], "district")
            self.assertEqual(r[geo_id_idx], "601")

    def test_domain_variable_populated(self):
        """Target on SNAP stratum -> domain_variable='snap'."""
        rows = self._query_target_overview()
        cols = self._overview_columns()
        sid_idx = cols.index("stratum_id")
        dv_idx = cols.index("domain_variable")

        snap_rows = [r for r in rows if r[sid_idx] == self.snap_domain_id]
        self.assertGreater(len(snap_rows), 0)
        for r in snap_rows:
            self.assertEqual(r[dv_idx], "snap")

    def test_active_flag_passthrough(self):
        """Active/inactive targets correctly reflected."""
        rows = self._query_target_overview()
        cols = self._overview_columns()
        sid_idx = cols.index("stratum_id")
        var_idx = cols.index("variable")
        active_idx = cols.index("active")

        snap_rows = [r for r in rows if r[sid_idx] == self.snap_domain_id]
        for r in snap_rows:
            if r[var_idx] == "snap":
                self.assertTrue(bool(r[active_idx]))
            elif r[var_idx] == "household_count":
                self.assertFalse(bool(r[active_idx]))

    # ----------------------------------------------------------------
    # get_geographic_strata()
    # ----------------------------------------------------------------

    def test_returns_national(self):
        """result['national'] is the national stratum ID."""
        with Session(self.engine) as session:
            result = get_geographic_strata(session)
        self.assertEqual(result["national"], self.national_id)

    def test_returns_states(self):
        """result['state'] maps {6: <id>, 37: <id>}."""
        with Session(self.engine) as session:
            result = get_geographic_strata(session)
        self.assertEqual(result["state"][6], self.state_ca_id)
        self.assertEqual(result["state"][37], self.state_nc_id)

    def test_returns_districts(self):
        """result['district'] maps CD geoids to stratum IDs."""
        with Session(self.engine) as session:
            result = get_geographic_strata(session)
        self.assertEqual(result["district"][601], self.cd_601_id)
        self.assertEqual(result["district"][602], self.cd_602_id)
        self.assertEqual(result["district"][3701], self.cd_3701_id)

    def test_excludes_domain_strata(self):
        """SNAP and age domain strata do NOT appear in any
        geographic category."""
        with Session(self.engine) as session:
            result = get_geographic_strata(session)

        all_ids = set()
        if result["national"] is not None:
            all_ids.add(result["national"])
        all_ids.update(result["state"].values())
        all_ids.update(result["district"].values())

        self.assertNotIn(self.snap_domain_id, all_ids)
        self.assertNotIn(self.age_domain_id, all_ids)

    # ----------------------------------------------------------------
    # get_all_cds_from_database()
    # ----------------------------------------------------------------

    def test_returns_all_cd_geoids(self):
        """Returns sorted list of CD GEOIDs."""
        cds = get_all_cds_from_database(self.db_uri)
        self.assertEqual(sorted(cds), ["3701", "601", "602"])

    def test_excludes_non_cd_constraints(self):
        """State/SNAP/age constraints not included."""
        cds = get_all_cds_from_database(self.db_uri)
        for val in cds:
            # Should only be CD geoids, not state FIPS or other
            self.assertTrue(
                int(val) >= 100,
                f"Unexpected value {val} in CD list",
            )

    # ----------------------------------------------------------------
    # get_cd_index_mapping()
    # ----------------------------------------------------------------

    def test_index_mapping_roundtrip(self):
        """cd_to_index and index_to_cd are inverses."""
        cd_to_idx, idx_to_cd, _ = get_cd_index_mapping(self.db_uri)
        for cd, idx in cd_to_idx.items():
            self.assertEqual(idx_to_cd[idx], cd)
        for idx, cd in idx_to_cd.items():
            self.assertEqual(cd_to_idx[cd], idx)

    def test_ordered_list_matches_sorted_cds(self):
        """cds_ordered matches the sorted CD list."""
        _, _, cds_ordered = get_cd_index_mapping(self.db_uri)
        expected = get_all_cds_from_database(self.db_uri)
        self.assertEqual(cds_ordered, expected)
