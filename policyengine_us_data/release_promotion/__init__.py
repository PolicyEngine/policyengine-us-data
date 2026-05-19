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
from .contract import (
    RELEASE_PROMOTION_CONTRACT_FILENAME,
    RELEASE_PROMOTION_CONTRACT_TYPE,
    ReleasePromotionContractBuilder,
    build_release_promotion_contract,
    release_promotion_contract_path,
    release_promotion_contract_repo_path,
    write_release_promotion_contract,
)
from .published_index import (
    PUBLISHED_ARTIFACT_INDEX_FILENAME,
    PUBLISHED_ARTIFACT_INDEX_MEDIA_TYPE,
    PublishedArtifactIndexRow,
    build_published_artifact_index,
    published_artifact_index_artifact_ref,
    published_artifact_index_from_jsonl,
    published_artifact_index_path,
    published_artifact_index_repo_path,
    published_artifact_index_to_jsonl,
    read_published_artifact_index,
    write_published_artifact_index,
)
from .results import (
    CleanupPromotionResult,
    CompletionMarkerPromotionResult,
    FullPromotionResult,
    GcsPromotionResult,
    HuggingFacePromotionResult,
    ReleaseManifestPromotionResult,
    VersionManifestPromotionResult,
)
from .validation import build_release_candidate_shape_report
from .validation import (
    DEFAULT_REQUIRED_RELEASE_ARTIFACT_FAMILIES,
    RELEASE_VALIDATION_SUBSTAGE_ID,
    ReleaseCandidateValidationDependencies,
    ReleaseCandidateValidator,
    default_release_candidate_validation_dependencies,
)

__all__ = [
    "BASE_RELEASE_ARTIFACT_PATHS",
    "DEFAULT_REQUIRED_RELEASE_ARTIFACT_FAMILIES",
    "RELEASE_VALIDATION_SUBSTAGE_ID",
    "RELEASE_PROMOTION_CONTRACT_FILENAME",
    "RELEASE_PROMOTION_CONTRACT_TYPE",
    "PUBLISHED_ARTIFACT_INDEX_FILENAME",
    "PUBLISHED_ARTIFACT_INDEX_MEDIA_TYPE",
    "CleanupPromotionResult",
    "CompletionMarkerPromotionResult",
    "FullPromotionResult",
    "GcsPromotionResult",
    "HuggingFacePromotionResult",
    "PublishedArtifactIndexRow",
    "ReleaseArtifactSpec",
    "ReleaseCandidateInputBundle",
    "ReleasePromotionContractBuilder",
    "ReleasePromotionContext",
    "ReleaseCandidateValidationDependencies",
    "ReleaseCandidateValidator",
    "ReleaseManifestPromotionResult",
    "VersionManifestPromotionResult",
    "build_legacy_release_candidate_bundle",
    "build_published_artifact_index",
    "build_release_promotion_contract",
    "build_release_candidate_bundle_from_stage4_contract",
    "build_release_candidate_shape_report",
    "default_release_candidate_validation_dependencies",
    "dedupe_normalized_release_paths",
    "infer_artifact_identity",
    "infer_release_artifact_spec",
    "logical_name_for_release_path",
    "normalize_release_path",
    "published_artifact_index_artifact_ref",
    "published_artifact_index_from_jsonl",
    "published_artifact_index_path",
    "published_artifact_index_repo_path",
    "published_artifact_index_to_jsonl",
    "release_promotion_contract_path",
    "release_promotion_contract_repo_path",
    "read_published_artifact_index",
    "read_stage4_release_candidate_bundle",
    "strip_staging_prefix",
    "write_published_artifact_index",
    "write_release_promotion_contract",
]
