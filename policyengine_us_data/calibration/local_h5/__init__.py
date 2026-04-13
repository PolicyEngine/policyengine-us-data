"""Internal contracts and helpers for incremental local H5 migration.

This package intentionally exposes a small compatibility surface while
the implementation is being subdivided into themed packages.
"""

from .contracts import (
    AreaBuildRequest,
    AreaBuildResult,
    AreaFilter,
    PublishingInputBundle,
    ValidationIssue,
    ValidationPolicy,
    ValidationResult,
    WorkerResult,
)
from .partitioning import partition_weighted_work_items, work_item_key

__all__ = [
    "AreaBuildRequest",
    "AreaBuildResult",
    "AreaFilter",
    "PublishingInputBundle",
    "ValidationIssue",
    "ValidationPolicy",
    "ValidationResult",
    "WorkerResult",
    "partition_weighted_work_items",
    "work_item_key",
]
