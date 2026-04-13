"""Validation-related local H5 contracts.

This package owns typed validation policy and validation result objects.
Future validation helpers and adapters should live alongside these
contracts.
"""

from .contracts import (
    ValidationIssue,
    ValidationPolicy,
    ValidationResult,
    ValidationStatus,
)

__all__ = [
    "ValidationIssue",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationStatus",
]
