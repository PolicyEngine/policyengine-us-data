"""
Tests for count target handling in SparseMatrixBuilder.

These tests verify that count targets (e.g., person_count, tax_unit_count)
are correctly handled by counting entities that satisfy constraints, rather
than summing values.
"""

import pytest
import numpy as np
from dataclasses import dataclass

from policyengine_us_data.datasets.cps.local_area_calibration.sparse_matrix_builder import (
    SparseMatrixBuilder,
)


@dataclass
class MockEntity:
    """Mock entity with a key attribute."""

    key: str


@dataclass
class MockVariable:
    """Mock variable with entity information."""

    entity: MockEntity

    @classmethod
    def create(cls, entity_key: str) -> "MockVariable":
        return cls(entity=MockEntity(key=entity_key))


class MockTaxBenefitSystem:
    """Mock tax benefit system with variable definitions."""

    def __init__(self):
        self.variables = {
            "person_count": MockVariable.create("person"),
            "tax_unit_count": MockVariable.create("tax_unit"),
            "household_count": MockVariable.create("household"),
            "spm_unit_count": MockVariable.create("spm_unit"),
            "snap": MockVariable.create("spm_unit"),
        }


@dataclass
class MockCalculationResult:
    """Mock result from simulation.calculate()."""

    values: np.ndarray


class MockSimulation:
    """Mock simulation for testing count target calculations."""

    def __init__(self, entity_data: dict, variable_values: dict):
        """
        Args:
            entity_data: Dict with person_id, household_id, tax_unit_id,
                spm_unit_id arrays (all at person level)
            variable_values: Dict mapping variable names to their values
                at the appropriate entity level
        """
        self.entity_data = entity_data
        self.variable_values = variable_values
        self.tax_benefit_system = MockTaxBenefitSystem()

    def calculate(self, variable: str, map_to: str = None):
        """Return mock calculation result."""
        if variable in self.entity_data:
            # Entity ID variables
            if map_to == "person":
                values = np.array(self.entity_data[variable])
            elif map_to == "household":
                # Return unique household IDs
                values = np.array(
                    sorted(set(self.entity_data["household_id"]))
                )
            else:
                values = np.array(self.entity_data[variable])
        elif variable in self.variable_values:
            # Regular variables - return at requested level
            val_data = self.variable_values[variable]
            if map_to == "person":
                values = np.array(val_data["person"])
            elif map_to == "household":
                values = np.array(val_data["household"])
            else:
                values = np.array(val_data.get("default", []))
        else:
            values = np.array([])

        return MockCalculationResult(values=values)


@pytest.fixture
def basic_entity_data():
    """
    Create mock entity relationships with known household compositions.

    Household 1 (id=100): 3 people (ages 5, 12, 40) -> 2 aged 5-17
    Household 2 (id=200): 2 people (ages 3, 25) -> 0 aged 5-17
    Household 3 (id=300): 4 people (ages 6, 8, 10, 45) -> 3 aged 5-17
    """
    return {
        "person_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "household_id": [100, 100, 100, 200, 200, 300, 300, 300, 300],
        "tax_unit_id": [10, 10, 10, 20, 20, 30, 30, 30, 30],
        "spm_unit_id": [
            1000,
            1000,
            1000,
            2000,
            2000,
            3000,
            3000,
            3000,
            3000,
        ],
    }


@pytest.fixture
def basic_variable_values():
    """Variable values for basic household composition tests."""
    return {
        "age": {
            "person": [5, 12, 40, 3, 25, 6, 8, 10, 45],
            "household": [40, 25, 45],  # Not used for age constraints
        },
        "person_count": {
            "person": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "household": [3, 2, 4],  # Sum per household
        },
        "snap": {
            "person": [100, 100, 100, 0, 0, 200, 200, 200, 200],
            "household": [300, 0, 800],
        },
    }


@pytest.fixture
def basic_sim(basic_entity_data, basic_variable_values):
    """Mock simulation with basic household compositions."""
    return MockSimulation(basic_entity_data, basic_variable_values)


@pytest.fixture
def builder():
    """Create a minimal SparseMatrixBuilder (won't use DB for unit tests)."""
    return SparseMatrixBuilder(
        db_uri="sqlite:///:memory:",
        time_period=2023,
        cds_to_calibrate=["101"],
    )


# Tests for basic count target calculation
class TestCountTargetCalculation:
    """Test _calculate_target_values_entity_aware for count targets."""

    def test_person_count_with_age_constraints(self, builder, basic_sim):
        """Test person_count correctly counts persons in age range per HH."""
        # Constraints: age >= 5 AND age < 18
        constraints = [
            {"variable": "age", "operation": ">=", "value": 5},
            {"variable": "age", "operation": "<", "value": 18},
        ]

        geo_mask = np.array([True, True, True])  # All households included
        n_households = 3

        result = builder._calculate_target_values_entity_aware(
            basic_sim,
            "person_count",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: HH1 has 2 people (ages 5, 12), HH2 has 0, HH3 has 3 (6,8,10)
        expected = np.array([2, 0, 3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_person_count_no_constraints(self, builder, basic_sim):
        """Test person_count without constraints returns all persons per HH."""
        constraints = []
        geo_mask = np.array([True, True, True])
        n_households = 3

        result = builder._calculate_target_values_entity_aware(
            basic_sim,
            "person_count",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: HH1 has 3 people, HH2 has 2, HH3 has 4
        expected = np.array([3, 2, 4], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_person_count_with_geo_mask(self, builder, basic_sim):
        """Test person_count respects geographic mask."""
        constraints = [
            {"variable": "age", "operation": ">=", "value": 5},
            {"variable": "age", "operation": "<", "value": 18},
        ]

        # Only include households 1 and 3
        geo_mask = np.array([True, False, True])
        n_households = 3

        result = builder._calculate_target_values_entity_aware(
            basic_sim,
            "person_count",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: HH1=2, HH2=0 (masked out), HH3=3
        expected = np.array([2, 0, 3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_value_target_uses_sum(self, builder, basic_sim):
        """Test that non-count targets sum values (existing behavior)."""
        # SNAP is a value target, not a count target
        constraints = []
        geo_mask = np.array([True, True, True])
        n_households = 3

        result = builder._calculate_target_values_entity_aware(
            basic_sim,
            "snap",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: Sum of snap values per household
        expected = np.array([300, 0, 800], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_household_count_no_constraints(self, builder, basic_sim):
        """Test household_count returns 1 for each qualifying household."""
        constraints = []
        geo_mask = np.array([True, True, True])
        n_households = 3

        result = builder._calculate_target_values_entity_aware(
            basic_sim,
            "household_count",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: 1 for each household in geo_mask
        expected = np.array([1, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_household_count_with_geo_mask(self, builder, basic_sim):
        """Test household_count respects geographic mask."""
        constraints = []
        geo_mask = np.array([True, False, True])
        n_households = 3

        result = builder._calculate_target_values_entity_aware(
            basic_sim,
            "household_count",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: 1 for HH1, 0 for HH2 (masked), 1 for HH3
        expected = np.array([1, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


# Fixtures for complex entity relationship tests
@pytest.fixture
def complex_entity_data():
    """
    Create entity data with multiple tax units per household.

    Household 1 (id=100): 4 people in 2 tax units
      Tax unit 10: person 1 (age 30, filer), person 2 (age 28)
      Tax unit 11: person 3 (age 65, filer), person 4 (age 62)
    Household 2 (id=200): 2 people in 1 tax unit
      Tax unit 20: person 5 (age 45, filer), person 6 (age 16)
    """
    return {
        "person_id": [1, 2, 3, 4, 5, 6],
        "household_id": [100, 100, 100, 100, 200, 200],
        "tax_unit_id": [10, 10, 11, 11, 20, 20],
        "spm_unit_id": [1000, 1000, 1000, 1000, 2000, 2000],
    }


@pytest.fixture
def complex_variable_values():
    """Variable values for complex entity relationship tests."""
    return {
        "age": {
            "person": [30, 28, 65, 62, 45, 16],
            "household": [65, 45],
        },
        "is_tax_unit_head": {
            "person": [True, False, True, False, True, False],
            "household": [2, 1],  # count of heads per HH
        },
        "tax_unit_count": {
            "person": [1, 1, 1, 1, 1, 1],
            "household": [2, 1],
        },
        "person_count": {
            "person": [1, 1, 1, 1, 1, 1],
            "household": [4, 2],
        },
    }


@pytest.fixture
def complex_sim(complex_entity_data, complex_variable_values):
    """Mock simulation with complex entity relationships."""
    return MockSimulation(complex_entity_data, complex_variable_values)


# Tests for complex entity relationships
class TestCountTargetWithRealEntities:
    """Test count targets with more complex entity relationships."""

    def test_tax_unit_count_no_constraints(self, builder, complex_sim):
        """Test tax_unit_count counts all tax units per household."""
        constraints = []
        geo_mask = np.array([True, True])
        n_households = 2

        result = builder._calculate_target_values_entity_aware(
            complex_sim,
            "tax_unit_count",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: HH1 has 2 tax units, HH2 has 1
        expected = np.array([2, 1], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_tax_unit_count_with_age_constraint(self, builder, complex_sim):
        """Test tax_unit_count with age constraint on members."""
        # Count tax units that have at least one person aged >= 65
        constraints = [
            {"variable": "age", "operation": ">=", "value": 65},
        ]
        geo_mask = np.array([True, True])
        n_households = 2

        result = builder._calculate_target_values_entity_aware(
            complex_sim,
            "tax_unit_count",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: HH1 has 1 tax unit (TU 11) with person >=65, HH2 has 0
        expected = np.array([1, 0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_person_count_seniors(self, builder, complex_sim):
        """Test person_count for seniors (age >= 65)."""
        constraints = [
            {"variable": "age", "operation": ">=", "value": 65},
        ]
        geo_mask = np.array([True, True])
        n_households = 2

        result = builder._calculate_target_values_entity_aware(
            complex_sim,
            "person_count",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: HH1 has 1 senior (age 65), HH2 has 0
        expected = np.array([1, 0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_person_count_children(self, builder, complex_sim):
        """Test person_count for children (age < 18)."""
        constraints = [
            {"variable": "age", "operation": "<", "value": 18},
        ]
        geo_mask = np.array([True, True])
        n_households = 2

        result = builder._calculate_target_values_entity_aware(
            complex_sim,
            "person_count",
            constraints,
            geo_mask,
            n_households,
        )

        # Expected: HH1 has 0 children, HH2 has 1 (age 16)
        expected = np.array([0, 1], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
