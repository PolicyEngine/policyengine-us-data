"""
Tests for UnifiedMatrixBuilder.

Uses a mock in-memory SQLite database with representative targets
and mocked Microsimulation to avoid heavy dependencies.
"""

import numpy as np
import pytest
from collections import namedtuple
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    Source,
    SourceType,
)
from policyengine_us_data.calibration.unified_matrix_builder import (
    UnifiedMatrixBuilder,
    COUNT_VARIABLES,
    _GEO_STATE_VARS,
    _GEO_CD_VARS,
    _GEO_VARS,
)

# -------------------------------------------------------------------
# Helper: lightweight GeographyAssignment stand-in
# -------------------------------------------------------------------


class MockGeography:
    """Minimal geography assignment for testing.

    Attributes:
        n_records: Number of base records (households).
        n_clones: Number of clones.
        block_geoid: Array of length n_records * n_clones with
            15-char census block GEOID strings.
        state_fips: Array of length n_records * n_clones with
            state FIPS for each column.
        cd_geoid: Array of length n_records * n_clones with CD
            GEOID strings for each column.
    """

    def __init__(
        self, n_records, n_clones, state_fips, cd_geoid, block_geoid=None
    ):
        self.n_records = n_records
        self.n_clones = n_clones
        self.state_fips = np.asarray(state_fips)
        self.cd_geoid = np.asarray(cd_geoid, dtype=object)
        if block_geoid is not None:
            self.block_geoid = np.asarray(block_geoid, dtype=object)
        else:
            # Generate synthetic block GEOIDs from state_fips
            self.block_geoid = np.array(
                [f"{int(s):02d}0010001001001" for s in self.state_fips],
                dtype=object,
            )


# -------------------------------------------------------------------
# Helper: build a mock Microsimulation
# -------------------------------------------------------------------


def _make_mock_sim(n_households=4, n_persons=8):
    """Create a mock Microsimulation with controllable data.

    Household layout:
        hh0: persons 0,1  (income=1000, snap=100)
        hh1: persons 2,3  (income=2000, snap=200)
        hh2: persons 4,5  (income=3000, snap=0)
        hh3: persons 6,7  (income=4000, snap=0)
    """
    sim = MagicMock()

    person_hh_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    person_tu_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    person_spm_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    hh_ids = np.arange(n_households)

    hh_income = np.array([1000, 2000, 3000, 4000], dtype=float)
    hh_snap = np.array([100, 200, 0, 0], dtype=float)
    # Person-level ages: hh0 has adult+child, rest all adults
    person_age = np.array([35, 10, 40, 45, 50, 55, 60, 65], dtype=float)

    def calculate_side_effect(var, period=None, map_to=None):
        result = MagicMock()
        if map_to == "person":
            mapping = {
                "household_id": person_hh_ids,
                "tax_unit_id": person_tu_ids,
                "spm_unit_id": person_spm_ids,
                "person_id": np.arange(n_persons),
                "age": person_age,
                "income": hh_income[person_hh_ids],
                "snap": hh_snap[person_hh_ids],
                "person_count": np.ones(n_persons, dtype=float),
            }
            result.values = mapping.get(var, np.zeros(n_persons, dtype=float))
        elif map_to == "household":
            mapping = {
                "household_id": hh_ids,
                "income": hh_income,
                "snap": hh_snap,
                "person_count": np.array([2, 2, 2, 2], dtype=float),
                "household_count": np.ones(n_households, dtype=float),
            }
            result.values = mapping.get(
                var, np.zeros(n_households, dtype=float)
            )
        else:
            result.values = np.zeros(n_households, dtype=float)
        return result

    sim.calculate = calculate_side_effect

    def map_result_side_effect(values, from_entity, to_entity, how=None):
        if from_entity == "person" and to_entity == "household":
            result = np.zeros(n_households, dtype=float)
            for i in range(n_persons):
                result[person_hh_ids[i]] += float(values[i])
            return result
        return values

    sim.map_result = map_result_side_effect

    # Mock set_input and delete_arrays as no-ops
    sim.set_input = MagicMock()
    sim.delete_arrays = MagicMock()

    return sim


# -------------------------------------------------------------------
# Fixtures: in-memory SQLite databases
# -------------------------------------------------------------------


def _seed_db(engine):
    """Populate test DB with targets at national, state, and CD levels.

    Creates:
    - National stratum -> income target (value=1e9)
    - CA stratum (state_fips=6) -> snap target (value=5e8)
    - NY stratum (state_fips=36) -> snap target (value=3e8)
    - CD 0601 stratum (CA CD) -> income target (value=1e8)
    - National -> household_count target (value=1e6)
    """
    with Session(engine) as session:
        source = Source(name="Test", type=SourceType.HARDCODED)
        session.add(source)
        session.flush()

        # --- National stratum (no constraints) ---
        us_stratum = Stratum(
            stratum_group_id=0,
            notes="United States",
        )
        us_stratum.constraints_rel = []
        session.add(us_stratum)
        session.flush()

        # Target 1: national income sum
        session.add(
            Target(
                stratum_id=us_stratum.stratum_id,
                variable="income",
                period=2024,
                value=1e9,
                source_id=source.source_id,
                active=True,
            )
        )

        # Target 2: national household_count
        session.add(
            Target(
                stratum_id=us_stratum.stratum_id,
                variable="household_count",
                period=2024,
                value=1e6,
                source_id=source.source_id,
                active=True,
            )
        )

        # --- California stratum ---
        ca_stratum = Stratum(
            parent_stratum_id=us_stratum.stratum_id,
            stratum_group_id=1,
            notes="California",
        )
        ca_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="6",
            )
        ]
        session.add(ca_stratum)
        session.flush()

        # Target 3: CA snap
        session.add(
            Target(
                stratum_id=ca_stratum.stratum_id,
                variable="snap",
                period=2024,
                value=5e8,
                source_id=source.source_id,
                active=True,
            )
        )

        # --- New York stratum ---
        ny_stratum = Stratum(
            parent_stratum_id=us_stratum.stratum_id,
            stratum_group_id=1,
            notes="New York",
        )
        ny_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="36",
            )
        ]
        session.add(ny_stratum)
        session.flush()

        # Target 4: NY snap
        session.add(
            Target(
                stratum_id=ny_stratum.stratum_id,
                variable="snap",
                period=2024,
                value=3e8,
                source_id=source.source_id,
                active=True,
            )
        )

        # --- CD 0601 stratum (under CA) ---
        cd_stratum = Stratum(
            parent_stratum_id=ca_stratum.stratum_id,
            stratum_group_id=2,
            notes="CA-01",
        )
        cd_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="congressional_district_geoid",
                operation="==",
                value="0601",
            )
        ]
        session.add(cd_stratum)
        session.flush()

        # Target 5: CD 0601 income
        session.add(
            Target(
                stratum_id=cd_stratum.stratum_id,
                variable="income",
                period=2024,
                value=1e8,
                source_id=source.source_id,
                active=True,
            )
        )

        session.commit()

    return engine


def _seed_constrained_db(engine):
    """Populate test DB with a constrained target.

    Creates a state-level target with an age constraint:
    - CA stratum with age >= 18 -> income target
    """
    with Session(engine) as session:
        source = Source(name="Test", type=SourceType.HARDCODED)
        session.add(source)
        session.flush()

        # National parent (no constraints)
        us_stratum = Stratum(
            stratum_group_id=0,
            notes="United States",
        )
        us_stratum.constraints_rel = []
        session.add(us_stratum)
        session.flush()

        # CA stratum
        ca_stratum = Stratum(
            parent_stratum_id=us_stratum.stratum_id,
            stratum_group_id=1,
            notes="California",
        )
        ca_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="state_fips",
                operation="==",
                value="6",
            )
        ]
        session.add(ca_stratum)
        session.flush()

        # CA adults stratum (child of CA)
        ca_adults = Stratum(
            parent_stratum_id=ca_stratum.stratum_id,
            stratum_group_id=3,
            notes="CA adults",
        )
        ca_adults.constraints_rel = [
            StratumConstraint(
                constraint_variable="age",
                operation=">=",
                value="18",
            )
        ]
        session.add(ca_adults)
        session.flush()

        # Target: CA adult income
        session.add(
            Target(
                stratum_id=ca_adults.stratum_id,
                variable="income",
                period=2024,
                value=7e8,
                source_id=source.source_id,
                active=True,
            )
        )

        session.commit()

    return engine


@pytest.fixture
def mock_db():
    """In-memory SQLite DB with representative targets."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return _seed_db(engine)


@pytest.fixture
def constrained_db():
    """In-memory SQLite DB with a constrained (age) target."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return _seed_constrained_db(engine)


@pytest.fixture
def mock_sim():
    """Standard mock Microsimulation with 4 households / 8 persons."""
    return _make_mock_sim()


def _builder_with_engine(engine, time_period=2024):
    """Create a UnifiedMatrixBuilder and inject the test engine."""
    builder = UnifiedMatrixBuilder(
        db_uri="sqlite://",
        time_period=time_period,
    )
    builder.engine = engine
    return builder


# -------------------------------------------------------------------
# Geography setup helper
# -------------------------------------------------------------------


def _make_geography(n_records, n_clones, assignments):
    """Build a MockGeography from a list of (state_fips, cd_geoid)
    tuples, one per clone.

    Column ordering: clone_idx * n_records + record_idx

    Args:
        n_records: Number of base records.
        n_clones: Number of clones.
        assignments: List of (state_fips, cd_geoid) of length
            n_clones.  Each assignment applies to ALL records
            cloned into that clone slot.
    """
    assert len(assignments) == n_clones
    n_total = n_records * n_clones
    state_fips = np.zeros(n_total, dtype=int)
    cd_geoid = np.empty(n_total, dtype=object)

    for clone_idx, (sfips, cd) in enumerate(assignments):
        start = clone_idx * n_records
        end = start + n_records
        state_fips[start:end] = sfips
        cd_geoid[start:end] = cd

    return MockGeography(n_records, n_clones, state_fips, cd_geoid)


# -------------------------------------------------------------------
# Tests: module constants
# -------------------------------------------------------------------


class TestModuleConstants:
    def test_count_variables_not_empty(self):
        assert len(COUNT_VARIABLES) > 0

    def test_household_count_in_count_variables(self):
        assert "household_count" in COUNT_VARIABLES

    def test_geo_vars_include_state_and_cd(self):
        assert "state_fips" in _GEO_VARS
        assert "congressional_district_geoid" in _GEO_VARS


# -------------------------------------------------------------------
# Tests: _classify_constraint_geo
# -------------------------------------------------------------------


class TestClassifyConstraintGeo:
    def test_national_when_no_geo_constraints(self):
        builder = UnifiedMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "age",
                "operation": ">=",
                "value": "18",
            }
        ]
        level, geo_id = builder._classify_constraint_geo(constraints)
        assert level == "national"
        assert geo_id is None

    def test_state_level(self):
        builder = UnifiedMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "state_fips",
                "operation": "==",
                "value": "6",
            }
        ]
        level, geo_id = builder._classify_constraint_geo(constraints)
        assert level == "state"
        assert geo_id == "6"

    def test_cd_level(self):
        builder = UnifiedMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "state_fips",
                "operation": "==",
                "value": "6",
            },
            {
                "variable": "congressional_district_geoid",
                "operation": "==",
                "value": "0601",
            },
        ]
        level, geo_id = builder._classify_constraint_geo(constraints)
        assert level == "cd"
        assert geo_id == "0601"

    def test_empty_constraints_returns_national(self):
        builder = UnifiedMatrixBuilder(db_uri="sqlite://", time_period=2024)
        level, geo_id = builder._classify_constraint_geo([])
        assert level == "national"
        assert geo_id is None


# -------------------------------------------------------------------
# Tests: _make_target_name
# -------------------------------------------------------------------


class TestMakeTargetName:
    def test_national_unconstrained(self):
        name = UnifiedMatrixBuilder._make_target_name(
            "income", [], reform_id=0
        )
        assert "national" in name
        assert "income" in name

    def test_state_constraint_in_name(self):
        constraints = [
            {
                "variable": "state_fips",
                "operation": "==",
                "value": "6",
            }
        ]
        name = UnifiedMatrixBuilder._make_target_name(
            "snap", constraints, reform_id=0
        )
        assert "state_6" in name
        assert "snap" in name

    def test_cd_constraint_in_name(self):
        constraints = [
            {
                "variable": "congressional_district_geoid",
                "operation": "==",
                "value": "0601",
            }
        ]
        name = UnifiedMatrixBuilder._make_target_name(
            "income", constraints, reform_id=0
        )
        assert "cd_0601" in name

    def test_reform_appends_expenditure(self):
        name = UnifiedMatrixBuilder._make_target_name(
            "salt_deduction", [], reform_id=1
        )
        assert "expenditure" in name


# -------------------------------------------------------------------
# Tests: build_matrix
# -------------------------------------------------------------------


class TestMatrixShape:
    """Verify the output matrix has the correct shape."""

    @patch("policyengine_us.Microsimulation")
    def test_matrix_shape(self, MockMicrosim, mock_db, mock_sim):
        """Matrix shape is (n_targets, n_records * n_clones)."""
        MockMicrosim.return_value = mock_sim

        n_records = 4
        n_clones = 3
        geography = _make_geography(
            n_records,
            n_clones,
            [
                (6, "0601"),  # clone 0 -> CA
                (36, "3601"),  # clone 1 -> NY
                (6, "0602"),  # clone 2 -> CA
            ],
        )

        builder = _builder_with_engine(mock_db)
        targets_df, X, names = builder.build_matrix("dummy_path.h5", geography)

        # 5 targets in mock_db
        assert X.shape == (5, n_records * n_clones)
        assert len(names) == 5
        assert len(targets_df) == 5


class TestStateTargetFillsOnlyStateColumns:
    """State-level target should only have nonzero values in columns
    assigned to that state."""

    @patch("policyengine_us.Microsimulation")
    def test_state_target_fills_only_state_columns(
        self, MockMicrosim, mock_db, mock_sim
    ):
        MockMicrosim.return_value = mock_sim

        n_records = 4
        n_clones = 3
        geography = _make_geography(
            n_records,
            n_clones,
            [
                (6, "0601"),  # clone 0: CA  cols 0-3
                (36, "3601"),  # clone 1: NY  cols 4-7
                (6, "0602"),  # clone 2: CA  cols 8-11
            ],
        )

        builder = _builder_with_engine(mock_db)
        targets_df, X, names = builder.build_matrix("dummy_path.h5", geography)
        X_dense = X.toarray()

        # Find the CA snap target (state_fips=6, variable=snap)
        ca_snap_idx = None
        for i, name in enumerate(names):
            if "state_6" in name and "snap" in name:
                ca_snap_idx = i
                break
        assert (
            ca_snap_idx is not None
        ), f"Could not find CA snap target in names: {names}"

        ca_row = X_dense[ca_snap_idx]
        # Cols 0-3 (clone 0, CA) and 8-11 (clone 2, CA) can be nonzero
        # Cols 4-7 (clone 1, NY) must be zero
        ny_cols = np.arange(4, 8)
        np.testing.assert_array_equal(
            ca_row[ny_cols],
            np.zeros(len(ny_cols)),
            err_msg="CA snap target has nonzero values in NY columns",
        )

        # At least some CA columns should be nonzero (snap=[100,200,0,0])
        ca_cols = np.concatenate([np.arange(0, 4), np.arange(8, 12)])
        assert np.any(
            ca_row[ca_cols] != 0
        ), "CA snap target has all zeros in CA columns"


class TestCdTargetFillsOnlyCdColumns:
    """CD-level target should only fill columns assigned to that CD."""

    @patch("policyengine_us.Microsimulation")
    def test_cd_target_fills_only_cd_columns(
        self, MockMicrosim, mock_db, mock_sim
    ):
        MockMicrosim.return_value = mock_sim

        n_records = 4
        n_clones = 3
        geography = _make_geography(
            n_records,
            n_clones,
            [
                (6, "0601"),  # clone 0: CA CD-01  cols 0-3
                (36, "3601"),  # clone 1: NY CD-01  cols 4-7
                (6, "0602"),  # clone 2: CA CD-02  cols 8-11
            ],
        )

        builder = _builder_with_engine(mock_db)
        targets_df, X, names = builder.build_matrix("dummy_path.h5", geography)
        X_dense = X.toarray()

        # Find the CD 0601 income target
        cd_income_idx = None
        for i, name in enumerate(names):
            if "cd_0601" in name and "income" in name:
                cd_income_idx = i
                break
        assert (
            cd_income_idx is not None
        ), f"Could not find CD 0601 income target in names: {names}"

        cd_row = X_dense[cd_income_idx]
        # Only cols 0-3 (clone 0, CD 0601) should be nonzero
        # Cols 4-7 (NY) and 8-11 (CA CD 0602) must be zero
        non_cd_cols = np.arange(4, 12)
        np.testing.assert_array_equal(
            cd_row[non_cd_cols],
            np.zeros(len(non_cd_cols)),
            err_msg=("CD 0601 target has nonzero in non-CD-0601 columns"),
        )

        cd_cols = np.arange(0, 4)
        assert np.any(
            cd_row[cd_cols] != 0
        ), "CD 0601 income target is all zeros in its own columns"


class TestNationalTargetFillsAllColumns:
    """National target fills columns across all states."""

    @patch("policyengine_us.Microsimulation")
    def test_national_target_fills_all_columns(
        self, MockMicrosim, mock_db, mock_sim
    ):
        MockMicrosim.return_value = mock_sim

        n_records = 4
        n_clones = 3
        geography = _make_geography(
            n_records,
            n_clones,
            [
                (6, "0601"),
                (36, "3601"),
                (6, "0602"),
            ],
        )

        builder = _builder_with_engine(mock_db)
        targets_df, X, names = builder.build_matrix("dummy_path.h5", geography)
        X_dense = X.toarray()

        # Find national income target (no state_ or cd_ prefix)
        national_income_idx = None
        for i, name in enumerate(names):
            if "national" in name and "income" in name and "cd_" not in name:
                national_income_idx = i
                break
        assert (
            national_income_idx is not None
        ), f"Could not find national income target in: {names}"

        nat_row = X_dense[national_income_idx]
        # hh_income = [1000, 2000, 3000, 4000] -- all nonzero
        # Should have nonzero values in every clone's columns
        for clone_idx in range(n_clones):
            start = clone_idx * n_records
            end = start + n_records
            clone_slice = nat_row[start:end]
            assert np.any(clone_slice != 0), (
                f"National income target is all zeros in clone "
                f"{clone_idx} (cols {start}-{end-1})"
            )


class TestColumnValuesUseCorrectRecord:
    """Column i should use values from record i % n_records."""

    @patch("policyengine_us.Microsimulation")
    def test_column_values_use_correct_record(
        self, MockMicrosim, mock_db, mock_sim
    ):
        MockMicrosim.return_value = mock_sim

        n_records = 4
        n_clones = 3
        geography = _make_geography(
            n_records,
            n_clones,
            [
                (6, "0601"),
                (36, "3601"),
                (6, "0602"),
            ],
        )

        builder = _builder_with_engine(mock_db)
        targets_df, X, names = builder.build_matrix("dummy_path.h5", geography)
        X_dense = X.toarray()

        # Find national income target
        national_income_idx = None
        for i, name in enumerate(names):
            if "national" in name and "income" in name and "cd_" not in name:
                national_income_idx = i
                break
        assert national_income_idx is not None

        nat_row = X_dense[national_income_idx]
        # hh_income = [1000, 2000, 3000, 4000]
        expected_values = np.array([1000, 2000, 3000, 4000], dtype=np.float32)

        # Each clone should replicate the same base record values
        for clone_idx in range(n_clones):
            start = clone_idx * n_records
            end = start + n_records
            np.testing.assert_array_almost_equal(
                nat_row[start:end],
                expected_values,
                err_msg=(f"Clone {clone_idx} has wrong record mapping"),
            )


class TestConstraintMaskApplied:
    """Non-geographic constraints filter which records contribute."""

    @patch("policyengine_us.Microsimulation")
    def test_constraint_mask_applied(
        self,
        MockMicrosim,
        constrained_db,
        mock_sim,
    ):
        MockMicrosim.return_value = mock_sim

        n_records = 4
        n_clones = 2
        geography = _make_geography(
            n_records,
            n_clones,
            [
                (6, "0601"),  # clone 0: CA
                (36, "3601"),  # clone 1: NY
            ],
        )

        builder = _builder_with_engine(constrained_db)
        targets_df, X, names = builder.build_matrix("dummy_path.h5", geography)
        X_dense = X.toarray()

        # The only target is CA adults income (age >= 18)
        assert X_dense.shape[0] == 1
        row = X_dense[0]

        # NY columns (clone 1, cols 4-7) must be zero because
        # this is a state=6 target
        ny_cols = np.arange(4, 8)
        np.testing.assert_array_equal(row[ny_cols], np.zeros(len(ny_cols)))

        # CA columns (clone 0, cols 0-3):
        # The age constraint (age >= 18) is evaluated per person:
        # person_age = [35,10, 40,45, 50,55, 60,65]
        # hh0 has one adult (person 0, age 35) -> mask True
        #   (any person satisfies -> household passes)
        # hh1: both adults -> True
        # hh2: both adults -> True
        # hh3: both adults -> True
        # So all CA households pass the constraint, and income
        # values are [1000, 2000, 3000, 4000].
        ca_cols = np.arange(0, 4)
        expected = np.array([1000, 2000, 3000, 4000], dtype=np.float32)
        np.testing.assert_array_almost_equal(row[ca_cols], expected)


class TestCountVariableHandling:
    """Count variables should produce 1.0 per qualifying household."""

    @patch("policyengine_us.Microsimulation")
    def test_household_count_is_one_per_household(
        self, MockMicrosim, mock_db, mock_sim
    ):
        MockMicrosim.return_value = mock_sim

        n_records = 4
        n_clones = 2
        geography = _make_geography(
            n_records,
            n_clones,
            [
                (6, "0601"),
                (36, "3601"),
            ],
        )

        builder = _builder_with_engine(mock_db)
        targets_df, X, names = builder.build_matrix("dummy_path.h5", geography)
        X_dense = X.toarray()

        # Find household_count target
        hh_count_idx = None
        for i, name in enumerate(names):
            if "household_count" in name:
                hh_count_idx = i
                break
        assert hh_count_idx is not None

        row = X_dense[hh_count_idx]
        # household_count is a count variable with no constraints
        # -> mask is all True -> values = mask.astype(float32) = 1.0
        # for all columns
        expected = np.ones(n_records * n_clones, dtype=np.float32)
        np.testing.assert_array_almost_equal(row, expected)


class TestQueryActiveTargets:
    """Test that _query_active_targets returns correct data."""

    def test_returns_all_active(self, mock_db):
        builder = _builder_with_engine(mock_db)
        df = builder._query_active_targets()
        assert len(df) == 5

    def test_filters_zero_values(self):
        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            source = Source(name="Test", type=SourceType.HARDCODED)
            session.add(source)
            session.flush()

            stratum = Stratum(stratum_group_id=0, notes="US")
            stratum.constraints_rel = []
            session.add(stratum)
            session.flush()

            session.add(
                Target(
                    stratum_id=stratum.stratum_id,
                    variable="income",
                    period=2024,
                    value=0.0,
                    source_id=source.source_id,
                    active=True,
                )
            )
            session.commit()

        builder = _builder_with_engine(engine)
        df = builder._query_active_targets()
        # Zero-value target filtered out
        assert len(df) == 0


class TestGetAllConstraints:
    """Test constraint chain walking."""

    def test_no_constraints_for_national(self, mock_db):
        builder = _builder_with_engine(mock_db)
        targets_df = builder._query_active_targets()
        national_row = targets_df[
            (targets_df["variable"] == "income") & (targets_df["value"] > 5e8)
        ].iloc[0]
        constraints = builder._get_all_constraints(national_row["stratum_id"])
        assert constraints == []

    def test_state_constraint_present(self, mock_db):
        builder = _builder_with_engine(mock_db)
        targets_df = builder._query_active_targets()
        ca_row = targets_df[
            (targets_df["variable"] == "snap") & (targets_df["value"] == 5e8)
        ].iloc[0]
        constraints = builder._get_all_constraints(ca_row["stratum_id"])
        var_names = {c["variable"] for c in constraints}
        assert "state_fips" in var_names

    def test_cd_walks_to_parent_state(self, mock_db):
        builder = _builder_with_engine(mock_db)
        targets_df = builder._query_active_targets()
        cd_row = targets_df[targets_df["value"] == 1e8].iloc[0]
        constraints = builder._get_all_constraints(cd_row["stratum_id"])
        var_names = {c["variable"] for c in constraints}
        assert "congressional_district_geoid" in var_names
        assert "state_fips" in var_names


class TestExtendedCPSHasNoCalculatedVars:
    """The extended CPS h5 should contain only input variables.

    The unified calibration pipeline assigns new geography and
    then invokes PE to compute all derived variables from
    scratch.  If the h5 includes variables with PE formulas,
    those stored values could conflict with what PE would
    compute fresh.  This test ensures the h5 only stores
    true survey inputs.
    """

    # Variables that have PE formulas but are stored in the
    # h5 as survey-reported or imputed input values.  These
    # are acceptable because PE's set_input mechanism means
    # the stored value takes precedence over the formula.
    # Each entry should have a comment explaining why it's
    # allowed.
    _ALLOWED_FORMULA_VARS = {
        # CPS/PUF-reported values with PE fallback formulas
        "employment_income",
        "self_employment_income",
        "weekly_hours_worked",
        # PUF-imputed tax credits (PE has formulas but we
        # trust the imputed values from the tax model)
        "american_opportunity_credit",
        "foreign_tax_credit",
        "savers_credit",
        "energy_efficient_home_improvement_credit",
        "cdcc_relevant_expenses",
        "taxable_unemployment_compensation",
        # Derived from other h5 inputs, not geography
        "rent",
        "person_id",
        "employment_income_last_year",
        "immigration_status",
    }

    @pytest.mark.xfail(
        reason="in_nyc should be removed from extended CPS h5",
        strict=True,
    )
    def test_no_formula_vars_in_h5(self):
        """H5 should not contain PE formula variables.

        Any variable with a PE formula that's stored in
        the h5 risks providing stale values (especially
        after geography reassignment).  Only explicitly
        allowed exceptions are permitted.

        Currently xfail because in_nyc is in the h5 and
        needs to be removed from the dataset build.
        """
        import h5py
        from pathlib import Path

        h5_path = Path("policyengine_us_data/storage/extended_cps_2024.h5")
        if not h5_path.exists():
            pytest.skip("extended_cps_2024.h5 not available")

        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=str(h5_path))

        with h5py.File(h5_path, "r") as f:
            h5_vars = set(f.keys())

        unexpected = set()
        for var_name in h5_vars:
            if var_name not in sim.tax_benefit_system.variables:
                continue
            var = sim.tax_benefit_system.variables[var_name]
            has_formula = hasattr(var, "formulas") and len(var.formulas) > 0
            if has_formula and var_name not in self._ALLOWED_FORMULA_VARS:
                unexpected.add(var_name)

        assert unexpected == set(), (
            f"Extended CPS h5 contains {len(unexpected)} "
            f"variable(s) with PE formulas that are not in "
            f"the allowlist. Either remove them from the "
            f"h5 or add to _ALLOWED_FORMULA_VARS with a "
            f"justification: {sorted(unexpected)}"
        )
