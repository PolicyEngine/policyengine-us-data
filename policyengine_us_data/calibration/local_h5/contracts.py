"""Compatibility barrel for local H5 contract exports.

The typed contracts now live in themed subpackages so future helpers can
be placed next to the contract family they support. This module stays as
the stable import surface during the migration.
"""

from __future__ import annotations

from .inputs import PublishingInputBundle
from .requests import (
    AreaBuildRequest,
    AreaFilter,
    AreaType,
    FilterOp,
)
from .results import (
    AreaBuildResult,
    BuildStatus,
    WorkerResult,
)
from .validation import (
    ValidationIssue,
    ValidationPolicy,
    ValidationResult,
    ValidationStatus,
)

__all__ = [
    "AreaBuildRequest",
    "AreaBuildResult",
    "AreaFilter",
    "AreaType",
    "BuildStatus",
    "FilterOp",
    "PublishingInputBundle",
    "ValidationIssue",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationStatus",
    "WorkerResult",
]
