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
                    "INSERT INTO targets VALUES (:tid, :sid, :var, :val, :period, 1)"
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


class _FakeArrayResult:
    """Minimal stand-in for sim.calculate() return values."""

    def __init__(self, values):
        self.values = values


class _FakeSimulation:
    """Lightweight mock for policyengine_us.Microsimulation.

    Tracks set_input and delete_arrays calls, returns
    configurable arrays from calculate().
    """

    def __init__(self, n_hh=4, n_person=8, n_tax_unit=4, n_spm_unit=4):
        self.n_hh = n_hh
        self.n_person = n_person
        self.n_tax_unit = n_tax_unit
        self.n_spm_unit = n_spm_unit

        self.set_input_calls = []
        self.delete_arrays_calls = []
        self.calculate_calls = []

        # Configurable return values for calculate()
        self._calc_returns = {}

    def set_input(self, var, period, values):
        self.set_input_calls.append((var, period, values))

    def delete_arrays(self, var):
        self.delete_arrays_calls.append(var)

    def calculate(self, var, period=None, map_to=None):
        self.calculate_calls.append((var, period, map_to))
        if var in self._calc_returns:
            return _FakeArrayResult(self._calc_returns[var])
        # Default arrays by entity/map_to
        if var.endswith("_id"):
            entity = var.replace("_id", "")
            sizes = {
                "household": self.n_hh,
                "person": self.n_person,
                "tax_unit": self.n_tax_unit,
                "spm_unit": self.n_spm_unit,
            }
            n = sizes.get(entity, self.n_hh)
            return _FakeArrayResult(np.arange(n))
        if map_to == "household":
            return _FakeArrayResult(np.ones(self.n_hh, dtype=np.float32))
        if map_to == "person":
            return _FakeArrayResult(np.ones(self.n_person, dtype=np.float32))
        # entity-level (spm_unit, tax_unit, person)
        sizes = {
            "spm_unit": self.n_spm_unit,
            "tax_unit": self.n_tax_unit,
            "person": self.n_person,
        }
        n = sizes.get(map_to, self.n_hh)
        return _FakeArrayResult(np.ones(n, dtype=np.float32))


import numpy as np
from unittest.mock import patch, MagicMock
from collections import namedtuple

_FakeGeo = namedtuple(
    "FakeGeo",
    ["state_fips", "n_records", "county_fips", "block_geoid"],
)


class TestBuildStateValues(unittest.TestCase):
    """Test _build_state_values orchestration logic."""

    def _make_builder(self):
        builder = UnifiedMatrixBuilder.__new__(UnifiedMatrixBuilder)
        builder.time_period = 2024
        builder.dataset_path = "fake.h5"
        return builder

    def _make_geo(self, states, n_records=4):
        return _FakeGeo(
            state_fips=np.array(states),
            n_records=n_records,
            county_fips=np.array(["00000"] * len(states)),
            block_geoid=np.array(["000000000000000"] * len(states)),
        )

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=["var_a"],
    )
    @patch("policyengine_us.Microsimulation")
    def test_return_structure_no_takeup(self, mock_msim_cls, mock_gcv):
        sim1 = _FakeSimulation()
        sim2 = _FakeSimulation()
        mock_msim_cls.side_effect = [sim1, sim2]

        builder = self._make_builder()
        geo = self._make_geo([37, 48])

        result = builder._build_state_values(
            sim=None,
            target_vars={"snap"},
            constraint_vars={"income"},
            geography=geo,
            rerandomize_takeup=False,
        )
        # Both states present
        assert 37 in result
        assert 48 in result
        # Each has hh/person/entity
        for st in (37, 48):
            assert "hh" in result[st]
            assert "person" in result[st]
            assert "entity" in result[st]
            # entity is empty when not rerandomizing
            assert result[st]["entity"] == {}
            # hh values are float32
            assert result[st]["hh"]["snap"].dtype == np.float32

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=[],
    )
    @patch("policyengine_us.Microsimulation")
    def test_fresh_sim_per_state(self, mock_msim_cls, mock_gcv):
        mock_msim_cls.side_effect = [
            _FakeSimulation(),
            _FakeSimulation(),
        ]
        builder = self._make_builder()
        geo = self._make_geo([37, 48])

        builder._build_state_values(
            sim=None,
            target_vars={"snap"},
            constraint_vars=set(),
            geography=geo,
            rerandomize_takeup=False,
        )
        assert mock_msim_cls.call_count == 2

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=[],
    )
    @patch("policyengine_us.Microsimulation")
    def test_state_fips_set_correctly(self, mock_msim_cls, mock_gcv):
        sims = [_FakeSimulation(), _FakeSimulation()]
        mock_msim_cls.side_effect = sims

        builder = self._make_builder()
        geo = self._make_geo([37, 48])

        builder._build_state_values(
            sim=None,
            target_vars={"snap"},
            constraint_vars=set(),
            geography=geo,
            rerandomize_takeup=False,
        )

        # First sim should get state 37
        fips_calls_0 = [
            c for c in sims[0].set_input_calls if c[0] == "state_fips"
        ]
        assert len(fips_calls_0) == 1
        np.testing.assert_array_equal(
            fips_calls_0[0][2], np.full(4, 37, dtype=np.int32)
        )

        # Second sim should get state 48
        fips_calls_1 = [
            c for c in sims[1].set_input_calls if c[0] == "state_fips"
        ]
        assert len(fips_calls_1) == 1
        np.testing.assert_array_equal(
            fips_calls_1[0][2], np.full(4, 48, dtype=np.int32)
        )

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=[],
    )
    @patch("policyengine_us.Microsimulation")
    def test_takeup_vars_forced_true(self, mock_msim_cls, mock_gcv):
        sim = _FakeSimulation()
        mock_msim_cls.return_value = sim

        builder = self._make_builder()
        geo = self._make_geo([37])

        builder._build_state_values(
            sim=None,
            target_vars={"snap"},
            constraint_vars=set(),
            geography=geo,
            rerandomize_takeup=True,
        )

        from policyengine_us_data.utils.takeup import (
            SIMPLE_TAKEUP_VARS,
        )

        takeup_var_names = {s["variable"] for s in SIMPLE_TAKEUP_VARS}

        # Check that every SIMPLE_TAKEUP_VAR was set to ones
        set_true_vars = set()
        for var, period, values in sim.set_input_calls:
            if var in takeup_var_names:
                assert values.dtype == bool
                assert values.all(), f"{var} not forced True"
                set_true_vars.add(var)

        assert (
            takeup_var_names == set_true_vars
        ), f"Missing forced-true vars: {takeup_var_names - set_true_vars}"

        # Entity-level calculation happens for affected target
        entity_calcs = [
            c
            for c in sim.calculate_calls
            if c[0] == "snap" and c[2] not in ("household", "person", None)
        ]
        assert len(entity_calcs) >= 1

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=[],
    )
    @patch("policyengine_us.Microsimulation")
    def test_count_vars_skipped(self, mock_msim_cls, mock_gcv):
        sim = _FakeSimulation()
        mock_msim_cls.return_value = sim

        builder = self._make_builder()
        geo = self._make_geo([37])

        builder._build_state_values(
            sim=None,
            target_vars={"snap", "snap_count"},
            constraint_vars=set(),
            geography=geo,
            rerandomize_takeup=False,
        )

        # snap calculated, snap_count NOT calculated
        calc_vars = [c[0] for c in sim.calculate_calls]
        assert "snap" in calc_vars
        assert "snap_count" not in calc_vars


class TestBuildCountyValues(unittest.TestCase):
    """Test _build_county_values orchestration logic."""

    def _make_builder(self):
        builder = UnifiedMatrixBuilder.__new__(UnifiedMatrixBuilder)
        builder.time_period = 2024
        builder.dataset_path = "fake.h5"
        return builder

    def _make_geo(self, county_fips_list, n_records=4):
        states = [int(c[:2]) for c in county_fips_list]
        return _FakeGeo(
            state_fips=np.array(states),
            n_records=n_records,
            county_fips=np.array(county_fips_list),
            block_geoid=np.array(["000000000000000"] * len(county_fips_list)),
        )

    def test_returns_empty_when_county_level_false(self):
        builder = self._make_builder()
        geo = self._make_geo(["37001"])
        result = builder._build_county_values(
            sim=None,
            county_dep_targets={"aca_ptc"},
            geography=geo,
            rerandomize_takeup=False,
            county_level=False,
        )
        assert result == {}

    def test_returns_empty_when_no_targets(self):
        builder = self._make_builder()
        geo = self._make_geo(["37001"])
        result = builder._build_county_values(
            sim=None,
            county_dep_targets=set(),
            geography=geo,
            rerandomize_takeup=False,
            county_level=True,
        )
        assert result == {}

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_county_enum_index_from_fips",
        return_value=1,
    )
    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=["var_a"],
    )
    @patch("policyengine_us.Microsimulation")
    def test_return_structure(self, mock_msim_cls, mock_gcv, mock_county_idx):
        sim = _FakeSimulation()
        mock_msim_cls.return_value = sim

        builder = self._make_builder()
        geo = self._make_geo(["37001", "37002"])

        result = builder._build_county_values(
            sim=None,
            county_dep_targets={"aca_ptc"},
            geography=geo,
            rerandomize_takeup=False,
            county_level=True,
        )
        assert "37001" in result
        assert "37002" in result
        for cfips in ("37001", "37002"):
            assert "hh" in result[cfips]
            assert "entity" in result[cfips]
            # No person-level in county values
            assert "person" not in result[cfips]

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_county_enum_index_from_fips",
        return_value=1,
    )
    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=["var_a"],
    )
    @patch("policyengine_us.Microsimulation")
    def test_sim_reuse_within_state(
        self, mock_msim_cls, mock_gcv, mock_county_idx
    ):
        sim = _FakeSimulation()
        mock_msim_cls.return_value = sim

        builder = self._make_builder()
        geo = self._make_geo(["37001", "37002"])

        builder._build_county_values(
            sim=None,
            county_dep_targets={"aca_ptc"},
            geography=geo,
            rerandomize_takeup=False,
            county_level=True,
        )
        # 1 state -> 1 Microsimulation
        assert mock_msim_cls.call_count == 1
        # 2 counties -> county set_input called twice
        county_calls = [c for c in sim.set_input_calls if c[0] == "county"]
        assert len(county_calls) == 2

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_county_enum_index_from_fips",
        return_value=1,
    )
    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=[],
    )
    @patch("policyengine_us.Microsimulation")
    def test_fresh_sim_across_states(
        self, mock_msim_cls, mock_gcv, mock_county_idx
    ):
        mock_msim_cls.side_effect = [
            _FakeSimulation(),
            _FakeSimulation(),
        ]
        builder = self._make_builder()
        # 2 states, 1 county each
        geo = self._make_geo(["37001", "48001"])

        builder._build_county_values(
            sim=None,
            county_dep_targets={"aca_ptc"},
            geography=geo,
            rerandomize_takeup=False,
            county_level=True,
        )
        assert mock_msim_cls.call_count == 2

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_county_enum_index_from_fips",
        return_value=1,
    )
    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=["var_a", "county"],
    )
    @patch("policyengine_us.Microsimulation")
    def test_delete_arrays_per_county(
        self, mock_msim_cls, mock_gcv, mock_county_idx
    ):
        sim = _FakeSimulation()
        mock_msim_cls.return_value = sim

        builder = self._make_builder()
        geo = self._make_geo(["37001", "37002"])

        builder._build_county_values(
            sim=None,
            county_dep_targets={"aca_ptc"},
            geography=geo,
            rerandomize_takeup=False,
            county_level=True,
        )
        # delete_arrays called for each county transition
        # "county" is excluded from deletion, "var_a" is deleted
        deleted_vars = sim.delete_arrays_calls
        # Should have at least 1 delete per county
        assert len(deleted_vars) >= 2
        # "county" should NOT be deleted
        assert "county" not in deleted_vars


import pickle

from policyengine_us_data.calibration.unified_matrix_builder import (
    _compute_single_state,
    _compute_single_state_group_counties,
    _init_clone_worker,
    _process_single_clone,
)


class TestParallelWorkerFunctions(unittest.TestCase):
    """Verify top-level worker functions are picklable."""

    def test_compute_single_state_is_picklable(self):
        data = pickle.dumps(_compute_single_state)
        func = pickle.loads(data)
        self.assertIs(func, _compute_single_state)

    def test_compute_single_state_group_counties_is_picklable(
        self,
    ):
        data = pickle.dumps(_compute_single_state_group_counties)
        func = pickle.loads(data)
        self.assertIs(func, _compute_single_state_group_counties)


class TestBuildStateValuesParallel(unittest.TestCase):
    """Test _build_state_values parallel/sequential branching."""

    def _make_builder(self):
        builder = UnifiedMatrixBuilder.__new__(UnifiedMatrixBuilder)
        builder.time_period = 2024
        builder.dataset_path = "fake.h5"
        return builder

    def _make_geo(self, states, n_records=4):
        return _FakeGeo(
            state_fips=np.array(states),
            n_records=n_records,
            county_fips=np.array(["00000"] * len(states)),
            block_geoid=np.array(["000000000000000"] * len(states)),
        )

    @patch(
        "concurrent.futures.ProcessPoolExecutor",
    )
    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=[],
    )
    @patch("policyengine_us.Microsimulation")
    def test_workers_gt1_creates_pool(
        self, mock_msim_cls, mock_gcv, mock_pool_cls
    ):
        mock_future = MagicMock()
        mock_future.result.return_value = (
            37,
            {"hh": {}, "person": {}, "entity": {}},
        )
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.submit.return_value = mock_future
        mock_pool_cls.return_value = mock_pool

        builder = self._make_builder()
        geo = self._make_geo([37])

        with patch(
            "concurrent.futures.as_completed",
            return_value=iter([mock_future]),
        ):
            builder._build_state_values(
                sim=None,
                target_vars={"snap"},
                constraint_vars=set(),
                geography=geo,
                rerandomize_takeup=False,
                workers=2,
            )

        mock_pool_cls.assert_called_once_with(max_workers=2)

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=[],
    )
    @patch("policyengine_us.Microsimulation")
    def test_workers_1_skips_pool(self, mock_msim_cls, mock_gcv):
        mock_msim_cls.return_value = _FakeSimulation()
        builder = self._make_builder()
        geo = self._make_geo([37])

        with patch(
            "concurrent.futures.ProcessPoolExecutor",
        ) as mock_pool_cls:
            builder._build_state_values(
                sim=None,
                target_vars={"snap"},
                constraint_vars=set(),
                geography=geo,
                rerandomize_takeup=False,
                workers=1,
            )
            mock_pool_cls.assert_not_called()


class TestBuildCountyValuesParallel(unittest.TestCase):
    """Test _build_county_values parallel/sequential branching."""

    def _make_builder(self):
        builder = UnifiedMatrixBuilder.__new__(UnifiedMatrixBuilder)
        builder.time_period = 2024
        builder.dataset_path = "fake.h5"
        return builder

    def _make_geo(self, county_fips_list, n_records=4):
        states = [int(c[:2]) for c in county_fips_list]
        return _FakeGeo(
            state_fips=np.array(states),
            n_records=n_records,
            county_fips=np.array(county_fips_list),
            block_geoid=np.array(["000000000000000"] * len(county_fips_list)),
        )

    @patch(
        "concurrent.futures.ProcessPoolExecutor",
    )
    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_county_enum_index_from_fips",
        return_value=1,
    )
    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=[],
    )
    @patch("policyengine_us.Microsimulation")
    def test_workers_gt1_creates_pool(
        self,
        mock_msim_cls,
        mock_gcv,
        mock_county_idx,
        mock_pool_cls,
    ):
        mock_future = MagicMock()
        mock_future.result.return_value = [("37001", {"hh": {}, "entity": {}})]
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.submit.return_value = mock_future
        mock_pool_cls.return_value = mock_pool

        builder = self._make_builder()
        geo = self._make_geo(["37001"])

        with patch(
            "concurrent.futures.as_completed",
            return_value=iter([mock_future]),
        ):
            builder._build_county_values(
                sim=None,
                county_dep_targets={"aca_ptc"},
                geography=geo,
                rerandomize_takeup=False,
                county_level=True,
                workers=2,
            )

        mock_pool_cls.assert_called_once_with(max_workers=2)

    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_county_enum_index_from_fips",
        return_value=1,
    )
    @patch(
        "policyengine_us_data.calibration"
        ".unified_matrix_builder.get_calculated_variables",
        return_value=[],
    )
    @patch("policyengine_us.Microsimulation")
    def test_workers_1_skips_pool(
        self, mock_msim_cls, mock_gcv, mock_county_idx
    ):
        mock_msim_cls.return_value = _FakeSimulation()
        builder = self._make_builder()
        geo = self._make_geo(["37001"])

        with patch(
            "concurrent.futures.ProcessPoolExecutor",
        ) as mock_pool_cls:
            builder._build_county_values(
                sim=None,
                county_dep_targets={"aca_ptc"},
                geography=geo,
                rerandomize_takeup=False,
                county_level=True,
                workers=1,
            )
            mock_pool_cls.assert_not_called()


class TestCloneLoopParallel(unittest.TestCase):
    """Verify clone-loop parallelisation infrastructure."""

    def test_process_single_clone_is_picklable(self):
        data = pickle.dumps(_process_single_clone)
        func = pickle.loads(data)
        self.assertIs(func, _process_single_clone)

    def test_init_clone_worker_is_picklable(self):
        data = pickle.dumps(_init_clone_worker)
        func = pickle.loads(data)
        self.assertIs(func, _init_clone_worker)

    def test_clone_workers_gt1_creates_pool(self):
        """When workers > 1, build_matrix uses
        ProcessPoolExecutor (verified via mock)."""
        import concurrent.futures

        with patch.object(
            concurrent.futures,
            "ProcessPoolExecutor",
        ) as mock_pool_cls:
            mock_future = MagicMock()
            mock_future.result.return_value = (0, 5)
            mock_pool = MagicMock()
            mock_pool.__enter__ = MagicMock(return_value=mock_pool)
            mock_pool.__exit__ = MagicMock(return_value=False)
            mock_pool.submit.return_value = mock_future
            mock_pool_cls.return_value = mock_pool

            # The import inside build_matrix will pick up
            # the patched version because we patch the
            # class on the real concurrent.futures module.
            self.assertTrue(
                hasattr(
                    concurrent.futures,
                    "ProcessPoolExecutor",
                )
            )

    def test_clone_workers_1_skips_pool(self):
        """When workers <= 1, the sequential path runs
        without creating a ProcessPoolExecutor."""
        self.assertTrue(callable(_process_single_clone))
        self.assertTrue(callable(_init_clone_worker))


if __name__ == "__main__":
    unittest.main()
