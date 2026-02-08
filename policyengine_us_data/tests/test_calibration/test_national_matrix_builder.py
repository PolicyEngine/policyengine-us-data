"""
Tests for NationalMatrixBuilder.

Uses a mock in-memory SQLite database with representative targets
and a mock Microsimulation to avoid heavy dependencies.
"""

import numpy as np
import pytest
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
from policyengine_us_data.calibration.national_matrix_builder import (
    NationalMatrixBuilder,
    COUNT_VARIABLES,
    PERSON_LEVEL_VARIABLES,
    SPM_UNIT_VARIABLES,
)


# -------------------------------------------------------------------
# Helper: build a mock Microsimulation with controllable data
# -------------------------------------------------------------------


def _make_mock_sim(n_households=5, n_persons=10):
    """
    Create a mock Microsimulation with controllable data.

    Household layout:
        hh0: persons 0,1  (tax_unit 0, spm_unit 0)
        hh1: persons 2,3  (tax_unit 1, spm_unit 1)
        hh2: persons 4,5  (tax_unit 2, spm_unit 2)
        hh3: persons 6,7  (tax_unit 3, spm_unit 3)
        hh4: persons 8,9  (tax_unit 4, spm_unit 4)
    """
    sim = MagicMock()

    person_hh_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    person_tu_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    person_spm_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    hh_ids = np.arange(n_households)

    # Person-level data
    person_medicaid = np.array(
        [100, 0, 200, 0, 0, 0, 300, 0, 0, 0], dtype=float
    )
    person_state_fips = np.array([6, 6, 6, 6, 36, 36, 36, 36, 6, 6], dtype=int)
    # Tax-unit-level data (5 tax units)
    tu_is_filer = np.array([1, 1, 0, 1, 1], dtype=float)
    tu_agi = np.array([75000, 120000, 30000, 60000, 200000], dtype=float)
    tu_income_tax_positive = np.array(
        [8000, 15000, 0, 5000, 40000], dtype=float
    )
    # Household-level data
    hh_medicaid = np.array([100, 200, 0, 300, 0], dtype=float)
    hh_net_worth = np.array(
        [500000, 1000000, 200000, 50000, 3000000], dtype=float
    )
    hh_snap = np.array([0, 5000, 0, 3000, 0], dtype=float)
    hh_person_count = np.array([2, 2, 2, 2, 2], dtype=float)

    # Map tax_unit vars to person level
    person_is_filer = tu_is_filer[person_tu_ids]
    person_agi = tu_agi[person_tu_ids]
    person_income_tax_positive = tu_income_tax_positive[person_tu_ids]

    def calculate_side_effect(var, period=None, map_to=None):
        """Mock sim.calculate() returning appropriate arrays."""
        result = MagicMock()

        if map_to == "person":
            mapping = {
                "household_id": person_hh_ids,
                "tax_unit_id": person_tu_ids,
                "spm_unit_id": person_spm_ids,
                "person_id": np.arange(n_persons),
                "medicaid": person_medicaid,
                "state_fips": person_state_fips,
                "tax_unit_is_filer": person_is_filer,
                "adjusted_gross_income": person_agi,
                "income_tax_positive": person_income_tax_positive,
                "person_count": np.ones(n_persons, dtype=float),
                "snap": hh_snap[person_hh_ids].astype(float),
                "net_worth": hh_net_worth[person_hh_ids].astype(float),
            }
            result.values = mapping.get(var, np.zeros(n_persons, dtype=float))
        elif map_to == "household":
            mapping = {
                "household_id": hh_ids,
                "medicaid": hh_medicaid,
                "net_worth": hh_net_worth,
                "snap": hh_snap,
                "income_tax_positive": np.array(
                    [8000, 15000, 0, 5000, 40000], dtype=float
                ),
                "person_count": hh_person_count,
                "adjusted_gross_income": np.array(
                    [75000, 120000, 30000, 60000, 200000],
                    dtype=float,
                ),
                "tax_unit_is_filer": tu_is_filer,
                "state_fips": np.array([6, 6, 36, 36, 6], dtype=int),
            }
            result.values = mapping.get(
                var, np.zeros(n_households, dtype=float)
            )
        else:
            # Default: tax_unit level
            mapping = {
                "tax_unit_is_filer": tu_is_filer,
                "adjusted_gross_income": tu_agi,
                "income_tax_positive": tu_income_tax_positive,
            }
            result.values = mapping.get(var, np.zeros(5, dtype=float))

        return result

    sim.calculate = calculate_side_effect

    def map_result_side_effect(values, from_entity, to_entity, how=None):
        """Mock sim.map_result() for person->household sum."""
        if from_entity == "person" and to_entity == "household":
            result = np.zeros(n_households, dtype=float)
            for i in range(n_persons):
                result[person_hh_ids[i]] += float(values[i])
            return result
        elif from_entity == "tax_unit" and to_entity == "household":
            return np.array(values, dtype=float)[:n_households]
        return values

    sim.map_result = map_result_side_effect

    return sim


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


def _seed_db(engine, include_inactive=False, include_geo=True):
    """Populate an in-memory SQLite DB with test targets.

    Returns the engine for convenience.
    """
    with Session(engine) as session:
        source = Source(
            name="Test",
            type=SourceType.HARDCODED,
        )
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

        # Target 1: national medicaid sum
        session.add(
            Target(
                stratum_id=us_stratum.stratum_id,
                variable="medicaid",
                period=2024,
                value=871.7e9,
                source_id=source.source_id,
                active=True,
            )
        )

        # Target 2: national net_worth sum
        session.add(
            Target(
                stratum_id=us_stratum.stratum_id,
                variable="net_worth",
                period=2024,
                value=160e12,
                source_id=source.source_id,
                active=True,
            )
        )

        # --- National filer stratum ---
        filer_stratum = Stratum(
            parent_stratum_id=us_stratum.stratum_id,
            stratum_group_id=2,
            notes="United States - Tax Filers",
        )
        filer_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="tax_unit_is_filer",
                operation="==",
                value="1",
            )
        ]
        session.add(filer_stratum)
        session.flush()

        # Target 3: income_tax_positive on filer stratum
        session.add(
            Target(
                stratum_id=filer_stratum.stratum_id,
                variable="income_tax_positive",
                period=2024,
                value=2.5e12,
                source_id=source.source_id,
                active=True,
            )
        )

        # --- AGI band stratum ---
        agi_stratum = Stratum(
            parent_stratum_id=filer_stratum.stratum_id,
            stratum_group_id=3,
            notes="National filers, AGI 50k-100k",
        )
        agi_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="adjusted_gross_income",
                operation=">=",
                value="50000",
            ),
            StratumConstraint(
                constraint_variable="adjusted_gross_income",
                operation="<",
                value="100000",
            ),
        ]
        session.add(agi_stratum)
        session.flush()

        # Target 4: person_count in AGI band
        session.add(
            Target(
                stratum_id=agi_stratum.stratum_id,
                variable="person_count",
                period=2024,
                value=32_801_908,
                source_id=source.source_id,
                active=True,
            )
        )

        # --- Medicaid enrollment stratum ---
        medicaid_stratum = Stratum(
            parent_stratum_id=us_stratum.stratum_id,
            stratum_group_id=5,
            notes="National Medicaid Enrollment",
        )
        medicaid_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="medicaid",
                operation=">",
                value="0",
            )
        ]
        session.add(medicaid_stratum)
        session.flush()

        # Target 5: person_count with medicaid > 0
        session.add(
            Target(
                stratum_id=medicaid_stratum.stratum_id,
                variable="person_count",
                period=2024,
                value=72_429_055,
                source_id=source.source_id,
                active=True,
            )
        )

        if include_geo:
            # --- State geographic stratum (California) ---
            ca_stratum = Stratum(
                parent_stratum_id=us_stratum.stratum_id,
                stratum_group_id=1,
                notes="State FIPS 6 - California",
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

            # Target 6: snap in CA
            session.add(
                Target(
                    stratum_id=ca_stratum.stratum_id,
                    variable="snap",
                    period=2024,
                    value=10e9,
                    source_id=source.source_id,
                    active=True,
                )
            )

        if include_inactive:
            # Inactive target that should be excluded
            session.add(
                Target(
                    stratum_id=us_stratum.stratum_id,
                    variable="ssi",
                    period=2024,
                    value=60e9,
                    source_id=source.source_id,
                    active=False,
                )
            )

        session.commit()

    return engine


@pytest.fixture
def mock_db():
    """Create an in-memory SQLite DB with representative targets."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return _seed_db(engine)


@pytest.fixture
def mock_db_with_inactive():
    """DB that includes both active and inactive targets."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return _seed_db(engine, include_inactive=True)


@pytest.fixture
def empty_db():
    """An empty DB with tables but no rows."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def mock_sim():
    """Standard mock Microsimulation with 5 households / 10 persons."""
    return _make_mock_sim()


def _builder_with_engine(engine, time_period=2024):
    """Create a NationalMatrixBuilder and inject the test engine."""
    builder = NationalMatrixBuilder(
        db_uri="sqlite://",
        time_period=time_period,
    )
    builder.engine = engine
    return builder


# -------------------------------------------------------------------
# Module-level constants tests
# -------------------------------------------------------------------


class TestModuleConstants:
    """Verify that module-level constant sets are defined."""

    def test_count_variables_not_empty(self):
        assert len(COUNT_VARIABLES) > 0

    def test_person_count_is_count_variable(self):
        assert "person_count" in COUNT_VARIABLES

    def test_person_count_is_person_level(self):
        assert "person_count" in PERSON_LEVEL_VARIABLES

    def test_spm_unit_variables_defined(self):
        assert isinstance(SPM_UNIT_VARIABLES, set)


# -------------------------------------------------------------------
# Database query tests
# -------------------------------------------------------------------


class TestQueryAllTargets:
    """Test _query_active_targets reads the right rows."""

    def test_returns_all_active_targets(self, mock_db):
        builder = _builder_with_engine(mock_db)
        df = builder._query_active_targets()

        # 6 active targets were inserted (no inactive)
        assert len(df) == 6

    def test_excludes_inactive_targets(self, mock_db_with_inactive):
        builder = _builder_with_engine(mock_db_with_inactive)
        df = builder._query_active_targets()

        # Still 6 active targets; the inactive "ssi" should be excluded
        assert len(df) == 6
        assert "ssi" not in df["variable"].values

    def test_required_columns_present(self, mock_db):
        builder = _builder_with_engine(mock_db)
        df = builder._query_active_targets()

        for col in [
            "target_id",
            "stratum_id",
            "variable",
            "value",
            "period",
        ]:
            assert col in df.columns

    def test_empty_db_returns_empty(self, empty_db):
        builder = _builder_with_engine(empty_db)
        df = builder._query_active_targets()
        assert len(df) == 0


class TestGetConstraints:
    """Test _get_all_constraints retrieves constraint rows."""

    def test_no_constraints_for_national(self, mock_db):
        builder = _builder_with_engine(mock_db)
        targets_df = builder._query_active_targets()

        # National medicaid has stratum with no constraints
        national_row = targets_df[targets_df["variable"] == "medicaid"].iloc[0]
        constraints = builder._get_all_constraints(national_row["stratum_id"])
        assert constraints == []

    def test_single_constraint_filer(self, mock_db):
        builder = _builder_with_engine(mock_db)
        targets_df = builder._query_active_targets()

        filer_row = targets_df[
            targets_df["variable"] == "income_tax_positive"
        ].iloc[0]
        constraints = builder._get_all_constraints(filer_row["stratum_id"])
        assert len(constraints) == 1
        assert constraints[0]["variable"] == "tax_unit_is_filer"
        assert constraints[0]["operation"] == "=="
        assert constraints[0]["value"] == "1"

    def test_multiple_constraints_agi_band(self, mock_db):
        builder = _builder_with_engine(mock_db)
        targets_df = builder._query_active_targets()

        # person_count target with AGI band constraints
        agi_row = targets_df[
            (targets_df["variable"] == "person_count")
            & (targets_df["value"] == 32_801_908)
        ].iloc[0]
        constraints = builder._get_all_constraints(agi_row["stratum_id"])
        assert len(constraints) == 3

        var_names = {c["variable"] for c in constraints}
        assert "tax_unit_is_filer" in var_names
        assert "adjusted_gross_income" in var_names

    def test_nonexistent_stratum_returns_empty(self, mock_db):
        builder = _builder_with_engine(mock_db)
        constraints = builder._get_all_constraints(99999)
        assert constraints == []


# -------------------------------------------------------------------
# Constraint evaluation tests
# -------------------------------------------------------------------


class TestEvaluateConstraints:
    """Test _evaluate_constraints mask computation."""

    def test_empty_constraints_returns_all_true(self, mock_sim):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        mask = builder._evaluate_constraints_entity_aware(
            mock_sim, [], n_households=5
        )
        assert mask.shape == (5,)
        assert np.all(mask)

    def test_filer_constraint_mask(self, mock_sim):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "tax_unit_is_filer",
                "operation": "==",
                "value": "1",
            }
        ]
        mask = builder._evaluate_constraints_entity_aware(
            mock_sim, constraints, n_households=5
        )
        # tu_is_filer = [1, 1, 0, 1, 1]
        # hh2 should be False (non-filer)
        assert mask.dtype == bool
        assert mask[0] is np.True_
        assert mask[1] is np.True_
        assert mask[2] is np.False_
        assert mask[3] is np.True_
        assert mask[4] is np.True_

    def test_compound_constraints_agi_band(self, mock_sim):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "tax_unit_is_filer",
                "operation": "==",
                "value": "1",
            },
            {
                "variable": "adjusted_gross_income",
                "operation": ">=",
                "value": "50000",
            },
            {
                "variable": "adjusted_gross_income",
                "operation": "<",
                "value": "100000",
            },
        ]
        mask = builder._evaluate_constraints_entity_aware(
            mock_sim, constraints, n_households=5
        )
        # AGI: [75k, 120k, 30k, 60k, 200k]
        # Filer: [1, 1, 0, 1, 1]
        # In band AND filer: hh0(75k), hh3(60k)
        assert mask[0] is np.True_
        assert mask[1] is np.False_  # AGI too high
        assert mask[2] is np.False_  # not filer
        assert mask[3] is np.True_
        assert mask[4] is np.False_  # AGI too high

    def test_geographic_constraint(self, mock_sim):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "state_fips",
                "operation": "==",
                "value": "6",
            }
        ]
        mask = builder._evaluate_constraints_entity_aware(
            mock_sim, constraints, n_households=5
        )
        # person_state_fips = [6,6, 6,6, 36,36, 36,36, 6,6]
        # hh0(CA), hh1(CA), hh2(NY), hh3(NY), hh4(CA)
        assert mask[0] is np.True_
        assert mask[1] is np.True_
        assert mask[2] is np.False_
        assert mask[3] is np.False_
        assert mask[4] is np.True_


# -------------------------------------------------------------------
# Entity relationship cache tests
# -------------------------------------------------------------------


class TestEntityRelationshipCache:
    """Test that entity relationship is built once and cached."""

    def test_cache_is_populated_on_first_call(self, mock_sim):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        assert builder._entity_rel_cache is None

        df = builder._build_entity_relationship(mock_sim)
        assert builder._entity_rel_cache is not None
        assert len(df) == 10  # n_persons

    def test_cache_is_reused(self, mock_sim):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)

        df1 = builder._build_entity_relationship(mock_sim)
        df2 = builder._build_entity_relationship(mock_sim)

        # Should be the exact same object (cached)
        assert df1 is df2

    def test_entity_rel_has_required_columns(self, mock_sim):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        df = builder._build_entity_relationship(mock_sim)
        for col in [
            "person_id",
            "household_id",
            "tax_unit_id",
            "spm_unit_id",
        ]:
            assert col in df.columns


# -------------------------------------------------------------------
# Target column computation tests
# -------------------------------------------------------------------


class TestComputeTargetColumn:
    """Test _compute_target_column for different variable types."""

    def test_sum_variable_no_constraints(self, mock_sim):
        """Unconstrained sum variable returns raw household values."""
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        col = builder._compute_target_column(
            mock_sim, "medicaid", [], n_households=5
        )
        expected = np.array([100, 200, 0, 300, 0], dtype=float)
        np.testing.assert_array_almost_equal(col, expected)

    def test_sum_variable_with_constraint(self, mock_sim):
        """Constrained sum variable zeros out non-matching."""
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "state_fips",
                "operation": "==",
                "value": "6",
            }
        ]
        col = builder._compute_target_column(
            mock_sim, "snap", constraints, n_households=5
        )
        # hh_snap = [0, 5000, 0, 3000, 0]
        # CA mask = [T, T, F, F, T]
        expected = np.array([0, 5000, 0, 0, 0], dtype=float)
        np.testing.assert_array_almost_equal(col, expected)

    def test_person_count_no_constraints(self, mock_sim):
        """person_count with no constraints counts all persons."""
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        col = builder._compute_target_column(
            mock_sim, "person_count", [], n_households=5
        )
        # Each household has 2 persons
        expected = np.array([2, 2, 2, 2, 2], dtype=float)
        np.testing.assert_array_almost_equal(col, expected)

    def test_person_count_with_constraints(self, mock_sim):
        """person_count with medicaid > 0 counts qualifying persons
        per household."""
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "medicaid",
                "operation": ">",
                "value": "0",
            }
        ]
        col = builder._compute_target_column(
            mock_sim, "person_count", constraints, n_households=5
        )
        # person_medicaid = [100,0, 200,0, 0,0, 300,0, 0,0]
        # hh0: 1 person, hh1: 1 person, hh2: 0, hh3: 1, hh4: 0
        expected = np.array([1, 1, 0, 1, 0], dtype=float)
        np.testing.assert_array_almost_equal(col, expected)

    def test_tax_unit_count_returns_mask(self, mock_sim):
        """tax_unit_count returns the household-level mask as float."""
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "tax_unit_is_filer",
                "operation": "==",
                "value": "1",
            }
        ]
        col = builder._compute_target_column(
            mock_sim,
            "tax_unit_count",
            constraints,
            n_households=5,
        )
        # Filer mask: [T, T, F, T, T]
        expected = np.array([1, 1, 0, 1, 1], dtype=float)
        np.testing.assert_array_almost_equal(col, expected)

    def test_column_dtype_is_float64(self, mock_sim):
        """All columns should be float64."""
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        col = builder._compute_target_column(
            mock_sim, "medicaid", [], n_households=5
        )
        assert col.dtype == np.float64


# -------------------------------------------------------------------
# Target name generation tests
# -------------------------------------------------------------------


class TestMakeTargetName:
    """Test _make_target_name label generation."""

    def test_national_unconstrained(self):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        name = builder._make_target_name("medicaid", [], "United States")
        assert "national" in name
        assert "medicaid" in name

    def test_geographic_constraint_in_name(self):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "state_fips",
                "operation": "==",
                "value": "6",
            }
        ]
        name = builder._make_target_name("snap", constraints, "California")
        assert "state_6" in name
        assert "snap" in name
        # Should NOT have "national" prefix
        assert "national" not in name

    def test_non_geo_constraints_in_brackets(self):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "tax_unit_is_filer",
                "operation": "==",
                "value": "1",
            }
        ]
        name = builder._make_target_name(
            "income_tax_positive",
            constraints,
            "Filers",
        )
        assert "national" in name
        assert "income_tax_positive" in name
        assert "[" in name
        assert "tax_unit_is_filer" in name

    def test_mixed_geo_and_non_geo(self):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "state_fips",
                "operation": "==",
                "value": "6",
            },
            {
                "variable": "tax_unit_is_filer",
                "operation": "==",
                "value": "1",
            },
        ]
        name = builder._make_target_name("eitc", constraints, "CA filers")
        assert "state_6" in name
        assert "eitc" in name
        assert "tax_unit_is_filer" in name

    def test_congressional_district_geo(self):
        builder = NationalMatrixBuilder(db_uri="sqlite://", time_period=2024)
        constraints = [
            {
                "variable": "congressional_district_geoid",
                "operation": "==",
                "value": "0601",
            },
        ]
        name = builder._make_target_name("snap", constraints, "CD 0601")
        assert "cd_0601" in name


# -------------------------------------------------------------------
# Full build_matrix integration tests
# -------------------------------------------------------------------


class TestBuildMatrix:
    """Test the main build_matrix method end-to-end."""

    def test_matrix_shape(self, mock_db, mock_sim):
        """Matrix shape is (n_households, n_active_targets)."""
        builder = _builder_with_engine(mock_db)
        matrix, targets, names = builder.build_matrix(mock_sim)

        assert matrix.shape == (5, 6)
        assert targets.shape == (6,)
        assert len(names) == 6

    def test_target_values_match_db(self, mock_db, mock_sim):
        """Target values array matches what was inserted."""
        builder = _builder_with_engine(mock_db)
        _, targets, names = builder.build_matrix(mock_sim)

        expected_values = {
            871.7e9,
            160e12,
            2.5e12,
            32_801_908,
            72_429_055,
            10e9,
        }
        actual_values = set(targets)
        assert actual_values == expected_values

    def test_all_names_non_empty(self, mock_db, mock_sim):
        """Every target name is a non-empty string."""
        builder = _builder_with_engine(mock_db)
        _, _, names = builder.build_matrix(mock_sim)

        for name in names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_no_nan_in_matrix(self, mock_db, mock_sim):
        """Matrix and targets contain no NaN values."""
        builder = _builder_with_engine(mock_db)
        matrix, targets, _ = builder.build_matrix(mock_sim)

        assert not np.any(np.isnan(matrix))
        assert not np.any(np.isnan(targets))

    def test_matrix_dtype_is_float64(self, mock_db, mock_sim):
        """Matrix dtype should be float64."""
        builder = _builder_with_engine(mock_db)
        matrix, targets, _ = builder.build_matrix(mock_sim)

        assert matrix.dtype == np.float64
        assert targets.dtype == np.float64

    def test_unconstrained_medicaid_column(self, mock_db, mock_sim):
        """National medicaid column = raw household medicaid."""
        builder = _builder_with_engine(mock_db)
        matrix, targets, names = builder.build_matrix(mock_sim)

        # Find medicaid column (not the person_count one)
        idx = next(
            i
            for i, n in enumerate(names)
            if "medicaid" in n and "person_count" not in n
        )
        expected = np.array([100, 200, 0, 300, 0], dtype=float)
        np.testing.assert_array_almost_equal(matrix[:, idx], expected)
        assert targets[idx] == pytest.approx(871.7e9)

    def test_filer_masked_income_tax(self, mock_db, mock_sim):
        """income_tax_positive column zeros out non-filer hh."""
        builder = _builder_with_engine(mock_db)
        matrix, _, names = builder.build_matrix(mock_sim)

        idx = next(
            i for i, n in enumerate(names) if "income_tax_positive" in n
        )
        col = matrix[:, idx]
        # hh2 is non-filer -> 0
        assert col[2] == 0
        # filer households have positive values
        assert col[0] > 0
        assert col[1] > 0
        assert col[3] > 0
        assert col[4] > 0

    def test_agi_band_person_count(self, mock_db, mock_sim):
        """person_count in AGI 50k-100k band."""
        builder = _builder_with_engine(mock_db)
        matrix, targets, names = builder.build_matrix(mock_sim)

        # Find the person_count target with value 32_801_908
        idx = next(
            i
            for i, n in enumerate(names)
            if "person_count" in n and targets[i] == pytest.approx(32_801_908)
        )
        col = matrix[:, idx]
        # hh0 (AGI 75k, filer) -> in band, 2 persons qualifying
        # hh3 (AGI 60k, filer) -> in band, 2 persons qualifying
        assert col[0] > 0
        assert col[3] > 0
        # hh1 (AGI 120k), hh2 (not filer), hh4 (AGI 200k) -> 0
        assert col[1] == 0
        assert col[2] == 0
        assert col[4] == 0

    def test_medicaid_enrollment_count(self, mock_db, mock_sim):
        """person_count with medicaid > 0 counts correctly."""
        builder = _builder_with_engine(mock_db)
        matrix, targets, names = builder.build_matrix(mock_sim)

        idx = next(
            i
            for i, n in enumerate(names)
            if "person_count" in n and targets[i] == pytest.approx(72_429_055)
        )
        col = matrix[:, idx]
        # person_medicaid = [100,0, 200,0, 0,0, 300,0, 0,0]
        # hh0:1, hh1:1, hh2:0, hh3:1, hh4:0
        assert col[0] == pytest.approx(1.0)
        assert col[1] == pytest.approx(1.0)
        assert col[2] == pytest.approx(0.0)
        assert col[3] == pytest.approx(1.0)
        assert col[4] == pytest.approx(0.0)

    def test_geographic_snap_target(self, mock_db, mock_sim):
        """snap in CA only includes CA households."""
        builder = _builder_with_engine(mock_db)
        matrix, _, names = builder.build_matrix(mock_sim)

        idx = next(i for i, n in enumerate(names) if "snap" in n)
        col = matrix[:, idx]
        # CA mask = [T, T, F, F, T]
        # hh_snap = [0, 5000, 0, 3000, 0]
        assert col[0] == 0  # CA, no snap
        assert col[1] == 5000  # CA, has snap
        assert col[2] == 0  # not CA
        assert col[3] == 0  # not CA (snap zeroed)
        assert col[4] == 0  # CA, no snap

    def test_constraint_cache_populated(self, mock_db, mock_sim):
        """After build_matrix, constraint cache should be non-empty."""
        builder = _builder_with_engine(mock_db)
        # Call build_matrix to populate internal state
        builder.build_matrix(mock_sim)

        # The entity relationship cache should be set
        assert builder._entity_rel_cache is not None

    def test_empty_db_raises_value_error(self, empty_db, mock_sim):
        """Empty database raises ValueError."""
        builder = _builder_with_engine(empty_db)
        with pytest.raises(ValueError, match="No active targets"):
            builder.build_matrix(mock_sim)


# -------------------------------------------------------------------
# Inactive target filtering test
# -------------------------------------------------------------------


class TestInactiveTargetFiltering:
    """Verify that active=False targets are excluded."""

    def test_inactive_excluded_from_matrix(
        self, mock_db_with_inactive, mock_sim
    ):
        builder = _builder_with_engine(mock_db_with_inactive)
        matrix, targets, names = builder.build_matrix(mock_sim)

        # Should still have 6 active targets, not 7
        assert matrix.shape[1] == 6
        assert len(targets) == 6

        # "ssi" should not appear in any name
        for name in names:
            assert "ssi" not in name.lower()


# -------------------------------------------------------------------
# Edge case tests
# -------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_target_db(self, mock_sim):
        """DB with only one active target produces 1-column matrix."""
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
                    variable="medicaid",
                    period=2024,
                    value=500e9,
                    source_id=source.source_id,
                    active=True,
                )
            )
            session.commit()

        builder = _builder_with_engine(engine)
        matrix, targets, names = builder.build_matrix(mock_sim)

        assert matrix.shape == (5, 1)
        assert len(targets) == 1
        assert targets[0] == pytest.approx(500e9)

    def test_all_households_masked_out(self, mock_sim):
        """A constraint that no household satisfies produces
        a zero column."""
        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            source = Source(name="Test", type=SourceType.HARDCODED)
            session.add(source)
            session.flush()

            stratum = Stratum(stratum_group_id=0, notes="Impossible")
            stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value="99",  # no household has this
                )
            ]
            session.add(stratum)
            session.flush()

            session.add(
                Target(
                    stratum_id=stratum.stratum_id,
                    variable="snap",
                    period=2024,
                    value=1e9,
                    source_id=source.source_id,
                    active=True,
                )
            )
            session.commit()

        builder = _builder_with_engine(engine)
        matrix, _, _ = builder.build_matrix(mock_sim)

        # All values in the column should be 0
        np.testing.assert_array_equal(matrix[:, 0], np.zeros(5))

    def test_large_target_value_preserved(self, mock_sim):
        """Very large target values are not corrupted."""
        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)

        big_value = 1.5e15

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
                    variable="net_worth",
                    period=2024,
                    value=big_value,
                    source_id=source.source_id,
                    active=True,
                )
            )
            session.commit()

        builder = _builder_with_engine(engine)
        _, targets, _ = builder.build_matrix(mock_sim)
        assert targets[0] == pytest.approx(big_value)

    def test_zero_target_value_filtered_out(self, mock_sim):
        """A target with value=0 is filtered out (not useful for
        calibration), raising ValueError when it's the only target."""
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
                    variable="medicaid",
                    period=2024,
                    value=0.0,
                    source_id=source.source_id,
                    active=True,
                )
            )
            session.commit()

        builder = _builder_with_engine(engine)
        with pytest.raises(ValueError, match="zero or null"):
            builder.build_matrix(mock_sim)

    def test_multiple_targets_same_stratum(self, mock_sim):
        """Multiple targets sharing one stratum produce separate
        columns with the same mask."""
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
                    variable="medicaid",
                    period=2024,
                    value=100e9,
                    source_id=source.source_id,
                    active=True,
                )
            )
            session.add(
                Target(
                    stratum_id=stratum.stratum_id,
                    variable="snap",
                    period=2024,
                    value=50e9,
                    source_id=source.source_id,
                    active=True,
                )
            )
            session.commit()

        builder = _builder_with_engine(engine)
        matrix, targets, names = builder.build_matrix(mock_sim)

        assert matrix.shape == (5, 2)
        assert len(targets) == 2


# -------------------------------------------------------------------
# apply_op utility tests (used by constraint evaluation)
# -------------------------------------------------------------------


class TestApplyOp:
    """Test the apply_op utility used for constraint evaluation."""

    def test_equality(self):
        from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
            apply_op,
        )

        vals = np.array([1, 2, 3, 4, 5])
        result = apply_op(vals, "==", "3")
        expected = np.array([False, False, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_greater_than(self):
        from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
            apply_op,
        )

        vals = np.array([1, 2, 3, 4, 5])
        result = apply_op(vals, ">", "3")
        expected = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_less_than(self):
        from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
            apply_op,
        )

        vals = np.array([1, 2, 3, 4, 5])
        result = apply_op(vals, "<", "3")
        expected = np.array([True, True, False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_gte(self):
        from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
            apply_op,
        )

        vals = np.array([1, 2, 3, 4, 5])
        result = apply_op(vals, ">=", "3")
        expected = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_lte(self):
        from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
            apply_op,
        )

        vals = np.array([1, 2, 3, 4, 5])
        result = apply_op(vals, "<=", "3")
        expected = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_not_equal(self):
        from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
            apply_op,
        )

        vals = np.array([1, 2, 3, 4, 5])
        result = apply_op(vals, "!=", "3")
        expected = np.array([True, True, False, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_float_value_parsing(self):
        from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
            apply_op,
        )

        vals = np.array([1.5, 2.5, 3.5, 4.5])
        result = apply_op(vals, ">", "2.5")
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(result, expected)
