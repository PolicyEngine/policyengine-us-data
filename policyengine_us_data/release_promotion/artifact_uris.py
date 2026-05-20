"""URI parsing helpers for Stage 5 release candidate artifacts."""

from __future__ import annotations

from urllib.parse import ParseResult, urlparse

from .artifacts import (
    BASE_RELEASE_ARTIFACT_PATHS,
    normalize_release_path,
    strip_staging_prefix,
)
from .context import ReleasePromotionContext


def release_path_from_artifact_uri(
    uri: str,
    *,
    context: ReleasePromotionContext,
) -> str | None:
    """Return a release-relative path from a staged artifact URI."""

    parsed = urlparse(uri)
    if parsed.scheme or parsed.netloc:
        validate_uri_repo(parsed, context)
    raw_path = parsed.path.lstrip("/") if parsed.scheme else uri
    candidate_paths = [raw_path]
    if parsed.netloc:
        candidate_paths.append(f"{parsed.netloc}/{raw_path}")
    for candidate in candidate_paths:
        if context.hf_staging_prefix and context.hf_staging_prefix in candidate:
            return strip_staging_prefix(
                candidate[candidate.index(context.hf_staging_prefix) :],
                context.hf_staging_prefix,
            )
        if parsed.scheme or parsed.netloc:
            if contains_release_artifact_path(candidate):
                raise ValueError(
                    "external artifact URI must point under the expected staging prefix"
                )
            continue
        if candidate.startswith("staging/"):
            return strip_staging_prefix(candidate, context.hf_staging_prefix)
        for prefix in ("states/", "districts/", "cities/", "national/"):
            if prefix in candidate:
                return normalize_release_path(candidate[candidate.index(prefix) :])
        for path in BASE_RELEASE_ARTIFACT_PATHS:
            if candidate == path or candidate.endswith(f"/{path}"):
                return normalize_release_path(
                    candidate[candidate.rindex(path) :],
                )
    return None


def contains_release_artifact_path(candidate: str) -> bool:
    """Return whether a path-shaped string names a release artifact."""

    if any(
        prefix in candidate
        for prefix in ("states/", "districts/", "cities/", "national/")
    ):
        return True
    return any(
        candidate.endswith(f"/{path}") or candidate == path
        for path in BASE_RELEASE_ARTIFACT_PATHS
    )


def validate_uri_repo(
    parsed_uri: ParseResult,
    context: ReleasePromotionContext,
) -> None:
    """Require external artifact URIs to reference the expected HF repository."""

    if not parsed_uri.scheme or not parsed_uri.netloc:
        return
    path_parts = parsed_uri.path.strip("/").split("/")
    if not path_parts or not path_parts[0]:
        return
    repo_name = f"{parsed_uri.netloc}/{path_parts[0]}"
    if repo_name != context.hf_repo_name:
        raise ValueError("external artifact URI repo must match context.hf_repo_name")
