"""
Unit tests for constraint validation logic.

Tests cover:
- Valid range constraints
- Empty range detection
- Equality operations must be alone
- Conflicting lower/upper bounds
- Multiple variables (each validated independently)
"""

import pytest

from policyengine_us_data.utils.constraint_validation import (
    Constraint,
    ConstraintValidationError,
    ensure_consistent_constraint_set,
)


class TestValidRanges:
    """Tests for valid range constraint combinations."""

    def test_valid_range_ge_lt(self):
        """age >= 25 AND age < 30 should pass."""
        constraints = [
            Constraint(variable="age", operation=">=", value="25"),
            Constraint(variable="age", operation="<", value="30"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_valid_range_gt_le(self):
        """age > 20 AND age <= 65 should pass."""
        constraints = [
            Constraint(variable="age", operation=">", value="20"),
            Constraint(variable="age", operation="<=", value="65"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_valid_range_gt_lt(self):
        """age > 0 AND age < 100 should pass."""
        constraints = [
            Constraint(variable="age", operation=">", value="0"),
            Constraint(variable="age", operation="<", value="100"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_valid_range_ge_le(self):
        """age >= 0 AND age <= 85 should pass."""
        constraints = [
            Constraint(variable="age", operation=">=", value="0"),
            Constraint(variable="age", operation="<=", value="85"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception


class TestEmptyRanges:
    """Tests for empty range detection."""

    def test_empty_range_lower_greater_than_upper(self):
        """age >= 50 AND age < 30 should fail (empty range)."""
        constraints = [
            Constraint(variable="age", operation=">=", value="50"),
            Constraint(variable="age", operation="<", value="30"),
        ]
        with pytest.raises(ConstraintValidationError, match="empty range"):
            ensure_consistent_constraint_set(constraints)

    def test_empty_range_equal_bounds_not_inclusive(self):
        """age > 30 AND age < 30 should fail (empty range)."""
        constraints = [
            Constraint(variable="age", operation=">", value="30"),
            Constraint(variable="age", operation="<", value="30"),
        ]
        with pytest.raises(ConstraintValidationError, match="empty range"):
            ensure_consistent_constraint_set(constraints)

    def test_empty_range_equal_bounds_one_inclusive(self):
        """age >= 30 AND age < 30 should fail (empty range)."""
        constraints = [
            Constraint(variable="age", operation=">=", value="30"),
            Constraint(variable="age", operation="<", value="30"),
        ]
        with pytest.raises(ConstraintValidationError, match="empty range"):
            ensure_consistent_constraint_set(constraints)

    def test_valid_point_range_both_inclusive(self):
        """age >= 30 AND age <= 30 should pass (valid point)."""
        constraints = [
            Constraint(variable="age", operation=">=", value="30"),
            Constraint(variable="age", operation="<=", value="30"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception


class TestEqualityOperations:
    """Tests for equality operation rules."""

    def test_equality_alone_is_valid(self):
        """state_fips == '06' alone should pass."""
        constraints = [
            Constraint(variable="state_fips", operation="==", value="06"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_not_equal_alone_is_valid(self):
        """state_fips != '72' alone should pass."""
        constraints = [
            Constraint(variable="state_fips", operation="!=", value="72"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_equality_with_range_fails(self):
        """state_fips == '06' AND state_fips > '05' should fail."""
        constraints = [
            Constraint(variable="state_fips", operation="==", value="06"),
            Constraint(variable="state_fips", operation=">", value="05"),
        ]
        with pytest.raises(ConstraintValidationError, match="cannot combine"):
            ensure_consistent_constraint_set(constraints)

    def test_not_equal_with_range_fails(self):
        """state_fips != '06' AND state_fips < '10' should fail."""
        constraints = [
            Constraint(variable="state_fips", operation="!=", value="06"),
            Constraint(variable="state_fips", operation="<", value="10"),
        ]
        with pytest.raises(ConstraintValidationError, match="cannot combine"):
            ensure_consistent_constraint_set(constraints)


class TestConflictingBounds:
    """Tests for conflicting lower/upper bound detection."""

    def test_conflicting_lower_bounds(self):
        """age > 20 AND age >= 25 should fail."""
        constraints = [
            Constraint(variable="age", operation=">", value="20"),
            Constraint(variable="age", operation=">=", value="25"),
        ]
        with pytest.raises(
            ConstraintValidationError, match="conflicting lower bounds"
        ):
            ensure_consistent_constraint_set(constraints)

    def test_conflicting_upper_bounds(self):
        """age < 50 AND age <= 45 should fail."""
        constraints = [
            Constraint(variable="age", operation="<", value="50"),
            Constraint(variable="age", operation="<=", value="45"),
        ]
        with pytest.raises(
            ConstraintValidationError, match="conflicting upper bounds"
        ):
            ensure_consistent_constraint_set(constraints)


class TestMultipleVariables:
    """Tests for constraints on multiple variables."""

    def test_multiple_variables_independent(self):
        """age >= 25 AND state_fips == '06' should pass."""
        constraints = [
            Constraint(variable="age", operation=">=", value="25"),
            Constraint(variable="state_fips", operation="==", value="06"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_multiple_variables_both_ranges(self):
        """age >= 25 AND age < 65 AND income > 0 AND income < 50000 should pass."""
        constraints = [
            Constraint(variable="age", operation=">=", value="25"),
            Constraint(variable="age", operation="<", value="65"),
            Constraint(variable="income", operation=">", value="0"),
            Constraint(variable="income", operation="<", value="50000"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_one_variable_invalid_other_valid(self):
        """Invalid on one variable should fail even if other is valid."""
        constraints = [
            Constraint(variable="age", operation=">=", value="50"),
            Constraint(variable="age", operation="<", value="30"),  # Invalid
            Constraint(variable="state_fips", operation="==", value="06"),
        ]
        with pytest.raises(ConstraintValidationError, match="empty range"):
            ensure_consistent_constraint_set(constraints)


class TestNonNumericValues:
    """Tests for non-numeric constraint values."""

    def test_string_equality_valid(self):
        """medicaid_enrolled == 'True' should pass."""
        constraints = [
            Constraint(
                variable="medicaid_enrolled", operation="==", value="True"
            ),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_string_values_skip_range_check(self):
        """Non-numeric values should skip range validation."""
        constraints = [
            Constraint(variable="ssn_card_type", operation="==", value="NONE"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_constraint_list(self):
        """Empty constraint list should pass."""
        constraints = []
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_single_lower_bound(self):
        """Single lower bound should pass."""
        constraints = [
            Constraint(variable="snap", operation=">", value="0"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_single_upper_bound(self):
        """Single upper bound should pass."""
        constraints = [
            Constraint(variable="income", operation="<", value="100000"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception

    def test_infinity_bounds(self):
        """AGI >= -inf AND AGI < 1 should pass."""
        constraints = [
            Constraint(variable="agi", operation=">=", value="-inf"),
            Constraint(variable="agi", operation="<", value="1"),
        ]
        ensure_consistent_constraint_set(constraints)  # No exception
