"""Typed cleanup result models for release promotion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from policyengine_us_data.release_promotion.results._coercion import (
    bool_value,
    nonnegative_int,
)
from policyengine_us_data.stage_contracts._coercion import (
    schema_version,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION

CLEANUP_STATUS_SKIPPED = "skipped"
CLEANUP_STATUS_COMPLETED = "completed"
CLEANUP_STATUS_FAILED = "failed"
CLEANUP_STATUSES = frozenset(
    {
        CLEANUP_STATUS_SKIPPED,
        CLEANUP_STATUS_COMPLETED,
        CLEANUP_STATUS_FAILED,
    }
)


def cleanup_status(value: Any) -> str:
    """Return a known cleanup status value."""

    if value not in CLEANUP_STATUSES:
        allowed = ", ".join(sorted(CLEANUP_STATUSES))
        raise ValueError(f"cleanup status must be one of: {allowed}")
    return str(value)


@dataclass(frozen=True, kw_only=True)
class CleanupPromotionResult:
    """Result for post-certification staging cleanup."""

    cleaned_count: int
    attempted: bool = True
    status: str = CLEANUP_STATUS_COMPLETED
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "cleaned_count",
            nonnegative_int(self.cleaned_count, "cleaned_count"),
        )
        object.__setattr__(self, "attempted", bool_value(self.attempted, "attempted"))
        object.__setattr__(self, "status", cleanup_status(self.status))
        if not self.attempted and self.status != CLEANUP_STATUS_SKIPPED:
            raise ValueError("cleanup status must be skipped when attempted is false")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "cleaned_count": self.cleaned_count,
            "attempted": self.attempted,
            "status": self.status,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CleanupPromotionResult":
        """Restore a cleanup result from a mapping."""

        return cls(
            cleaned_count=nonnegative_int(
                data.get("cleaned_count"),
                "cleaned_count",
            ),
            attempted=bool_value(data.get("attempted", True), "attempted"),
            status=cleanup_status(
                data.get("status", CLEANUP_STATUS_COMPLETED),
            ),
            schema_version=schema_version(data),
        )
