"""
Constraint validation for stratum definitions.

This module provides validation for constraint sets BEFORE they are
inserted into the database. This prevents logically inconsistent
constraints from ever being stored.

Validation Rules:
1. Operation Compatibility (per constraint_variable):
   - `==` and `!=` must be alone (cannot combine with other operations)
   - `>` and `>=` cannot coexist (conflicting lower bounds)
   - `<` and `<=` cannot coexist (conflicting upper bounds)
   - `>` or `>=` can combine with `<` or `<=` to form valid ranges

2. Value Checks (if operations are compatible):
   - No empty ranges: lower bound must be < upper bound
   - For equal bounds, both must be inclusive to be valid
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Constraint:
    """A constraint to validate (before creating StratumConstraint)."""

    variable: str
    operation: str
    value: str


class ConstraintValidationError(Exception):
    """Raised when constraint set is logically inconsistent."""

    pass


# Operation compatibility groups
EQUALITY_OPS = {"==", "!="}
LOWER_BOUND_OPS = {">", ">="}
UPPER_BOUND_OPS = {"<", "<="}
RANGE_OPS = LOWER_BOUND_OPS | UPPER_BOUND_OPS


def ensure_consistent_constraint_set(constraints: List[Constraint]) -> None:
    """
    Validate that a set of constraints is logically consistent.

    Call this BEFORE inserting constraints into the database.

    Args:
        constraints: List of Constraint objects to validate.

    Raises:
        ConstraintValidationError: If constraints are logically inconsistent.

    Example:
        >>> constraints = [
        ...     Constraint(variable="age", operation=">=", value="25"),
        ...     Constraint(variable="age", operation="<", value="30"),
        ... ]
        >>> ensure_consistent_constraint_set(constraints)  # No exception
    """
    # Group constraints by variable
    by_variable: dict = {}
    for c in constraints:
        by_variable.setdefault(c.variable, []).append(c)

    for var_name, var_constraints in by_variable.items():
        _validate_variable_constraints(var_name, var_constraints)


def _validate_variable_constraints(
    var_name: str, constraints: List[Constraint]
) -> None:
    """Validate all constraints on a single variable."""
    operations = {c.operation for c in constraints}

    # Rule 1: Check operation compatibility
    _check_operation_compatibility(var_name, operations)

    # Rule 2: If range operations, check for empty range
    if operations & RANGE_OPS:
        _check_range_validity(var_name, constraints)


def _check_operation_compatibility(var_name: str, operations: set) -> None:
    """Check that operations on a variable are compatible."""
    has_equality = bool(operations & EQUALITY_OPS)
    has_range = bool(operations & RANGE_OPS)

    # Equality ops must be alone
    if has_equality:
        if len(operations) > 1:
            raise ConstraintValidationError(
                f"{var_name}: '==' or '!=' cannot combine with other "
                f"operations, found: {operations}"
            )

    # Cannot have both > and >= (conflicting lower bounds)
    if ">" in operations and ">=" in operations:
        raise ConstraintValidationError(
            f"{var_name}: cannot have both '>' and '>=' "
            "(conflicting lower bounds)"
        )

    # Cannot have both < and <= (conflicting upper bounds)
    if "<" in operations and "<=" in operations:
        raise ConstraintValidationError(
            f"{var_name}: cannot have both '<' and '<=' "
            "(conflicting upper bounds)"
        )


def _check_range_validity(
    var_name: str, constraints: List[Constraint]
) -> None:
    """Check that range constraints don't create an empty range."""
    lower_bound = float("-inf")
    upper_bound = float("inf")
    lower_inclusive = False
    upper_inclusive = False

    for c in constraints:
        try:
            val = float(c.value)
        except ValueError:
            # Non-numeric value - skip range check
            continue

        if c.operation == ">":
            if val > lower_bound or (
                val == lower_bound and not lower_inclusive
            ):
                lower_bound = val
                lower_inclusive = False
        elif c.operation == ">=":
            if val > lower_bound or (val == lower_bound and lower_inclusive):
                lower_bound = val
                lower_inclusive = True
        elif c.operation == "<":
            if val < upper_bound or (
                val == upper_bound and not upper_inclusive
            ):
                upper_bound = val
                upper_inclusive = False
        elif c.operation == "<=":
            if val < upper_bound or (val == upper_bound and upper_inclusive):
                upper_bound = val
                upper_inclusive = True

    # Check for empty range
    if lower_bound > upper_bound:
        raise ConstraintValidationError(
            f"{var_name}: empty range - lower bound {lower_bound} > "
            f"upper bound {upper_bound}"
        )
    if lower_bound == upper_bound and not (
        lower_inclusive and upper_inclusive
    ):
        raise ConstraintValidationError(
            f"{var_name}: empty range - bounds equal at {lower_bound} "
            "but not both inclusive"
        )
