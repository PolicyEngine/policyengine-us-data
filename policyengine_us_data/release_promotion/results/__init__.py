"""Typed Stage 5 promotion result models."""

from .cleanup import (
    CLEANUP_STATUS_COMPLETED,
    CLEANUP_STATUS_FAILED,
    CLEANUP_STATUS_SKIPPED,
    CLEANUP_STATUSES,
    CleanupPromotionResult,
)
from .destinations import (
    GcsPromotionResult,
    HuggingFacePromotionResult,
)
from .full import FullPromotionResult, parse_full_promotion_result_json
from .manifests import (
    CompletionMarkerPromotionResult,
    ReleaseManifestPromotionResult,
    VersionManifestPromotionResult,
)

__all__ = [
    "CLEANUP_STATUS_COMPLETED",
    "CLEANUP_STATUS_FAILED",
    "CLEANUP_STATUS_SKIPPED",
    "CLEANUP_STATUSES",
    "CleanupPromotionResult",
    "CompletionMarkerPromotionResult",
    "FullPromotionResult",
    "GcsPromotionResult",
    "HuggingFacePromotionResult",
    "parse_full_promotion_result_json",
    "ReleaseManifestPromotionResult",
    "VersionManifestPromotionResult",
]
