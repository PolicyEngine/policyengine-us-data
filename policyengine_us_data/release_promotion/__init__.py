"""Typed Stage 5 release promotion boundaries.

This package starts with release-candidate identity and candidate-bundle
schemas. Promotion side effects still live in the existing transaction engine
until later Stage 5 migration slices move them behind typed services.
"""

from .artifacts import (
    BASE_RELEASE_ARTIFACT_PATHS,
    ReleaseArtifactSpec,
    dedupe_normalized_release_paths,
    infer_artifact_identity,
    infer_release_artifact_spec,
    logical_name_for_release_path,
    normalize_release_path,
    strip_staging_prefix,
)
from .candidate import (
    ReleaseCandidateInputBundle,
    build_legacy_release_candidate_bundle,
    build_release_candidate_bundle_from_stage4_contract,
    read_stage4_release_candidate_bundle,
)
from .context import ReleasePromotionContext
from .validation import build_release_candidate_shape_report

__all__ = [
    "BASE_RELEASE_ARTIFACT_PATHS",
    "ReleaseArtifactSpec",
    "ReleaseCandidateInputBundle",
    "ReleasePromotionContext",
    "build_legacy_release_candidate_bundle",
    "build_release_candidate_bundle_from_stage4_contract",
    "build_release_candidate_shape_report",
    "dedupe_normalized_release_paths",
    "infer_artifact_identity",
    "infer_release_artifact_spec",
    "logical_name_for_release_path",
    "normalize_release_path",
    "read_stage4_release_candidate_bundle",
    "strip_staging_prefix",
]
