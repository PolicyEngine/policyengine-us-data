"""Full release promotion orchestration.

This module keeps the transaction-like release flow separate from the lower
level upload/download primitives in ``data_upload.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from policyengine_us_data.utils.release_completion import release_completion_marker_path


ManifestFile = tuple[Path, str]
ReleaseManifest = dict[str, Any]


@dataclass(frozen=True)
class FullReleasePromotionConfig:
    """Inputs for promoting one run-scoped staged release."""

    rel_paths: Sequence[str]
    version: str
    run_id: str
    files_with_paths: Sequence[tuple[Path | str, str]] | None = None
    extra_cleanup_paths: Sequence[str] = ()
    gcs_bucket_name: str = "policyengine-us-data"
    hf_repo_name: str = "policyengine/policyengine-us-data"
    hf_repo_type: str = "model"
    cleanup_staging: bool = True


@dataclass(frozen=True)
class FullReleasePromotionDependencies:
    """Side-effecting operations used by the promotion orchestration."""

    dedupe_preserving_order: Callable[[Sequence[str]], list[str]]
    download_staged_artifacts_for_manifest: Callable[..., list[ManifestFile]]
    get_matching_finalized_release_manifest: Callable[..., ReleaseManifest | None]
    list_missing_staged_artifacts: Callable[..., list[str]]
    preflight_release_manifest_publish: Callable[..., tuple[bool, list[str]]]
    promote_staging_to_production_hf: Callable[..., int]
    upload_from_hf_staging_to_gcs: Callable[..., int]
    publish_release_manifest_to_hf: Callable[..., ReleaseManifest]
    upload_final_version_manifest: Callable[..., None]
    upload_release_completion_marker: Callable[..., ReleaseManifest]
    release_completion_marker_exists: Callable[..., bool]
    cleanup_staging_hf: Callable[..., int]


def promote_full_release(
    config: FullReleasePromotionConfig,
    deps: FullReleasePromotionDependencies,
) -> dict[str, Any]:
    """Promote all artifacts for one staged run.

    The order is deliberately transaction-like:
    validate staged HF inputs, copy every HF artifact in one commit, upload
    every GCS artifact, publish release_manifest.json and TRACE TRO, update
    version_manifest.json, write the release completion marker, tag that final
    marker commit, and only then clean staged inputs.
    """
    rel_paths = _validated_release_paths(config, deps)
    manifest_files = _manifest_files_for_release(config, rel_paths, deps)

    finalized_manifest = deps.get_matching_finalized_release_manifest(
        files_with_paths=list(manifest_files),
        version=config.version,
        hf_repo_name=config.hf_repo_name,
        hf_repo_type=config.hf_repo_type,
        model_package_name="policyengine-us",
    )
    if finalized_manifest is not None:
        return _finish_already_finalized_release(
            config=config,
            rel_paths=rel_paths,
            finalized_manifest=finalized_manifest,
            deps=deps,
        )

    _assert_staging_complete(config, rel_paths, deps)
    _assert_release_can_finalize(config, rel_paths, manifest_files, deps)

    promoted_hf = deps.promote_staging_to_production_hf(
        rel_paths,
        version=config.version,
        hf_repo_name=config.hf_repo_name,
        hf_repo_type=config.hf_repo_type,
        run_id=config.run_id,
        allow_noop=True,
    )
    uploaded_gcs = deps.upload_from_hf_staging_to_gcs(
        rel_paths,
        version=config.version,
        gcs_bucket_name=config.gcs_bucket_name,
        hf_repo_name=config.hf_repo_name,
        hf_repo_type=config.hf_repo_type,
        run_id=config.run_id,
    )
    release_manifest = deps.publish_release_manifest_to_hf(
        list(manifest_files),
        version=config.version,
        hf_repo_name=config.hf_repo_name,
        hf_repo_type=config.hf_repo_type,
        create_tag=False,
    )
    _upload_version_manifest(config, release_manifest, deps)
    completion_marker = _upload_release_completion_marker(
        config=config,
        release_manifest=release_manifest,
        rel_paths=rel_paths,
        promoted_hf=promoted_hf,
        uploaded_gcs=uploaded_gcs,
        create_tag=True,
        deps=deps,
    )

    cleaned = _cleanup_staging_after_release(
        config=config,
        rel_paths=rel_paths,
        deps=deps,
        warning="Release %s was finalized, but staging cleanup failed.",
    )

    return {
        "run_id": config.run_id,
        "version": config.version,
        "artifact_count": len(rel_paths),
        "hf_promoted": promoted_hf,
        "gcs_uploaded": uploaded_gcs,
        "release_manifest_artifacts": len(release_manifest["artifacts"]),
        "release_completion_marker": completion_marker.get("marker_path"),
        "staging_cleaned": cleaned,
    }


def _validated_release_paths(
    config: FullReleasePromotionConfig,
    deps: FullReleasePromotionDependencies,
) -> list[str]:
    if not config.run_id:
        raise ValueError("run_id is required for full release promotion.")
    if not config.version:
        raise ValueError("version is required for full release promotion.")

    rel_paths = deps.dedupe_preserving_order(config.rel_paths)
    if not rel_paths:
        raise ValueError("No release artifact paths were provided.")
    return rel_paths


def _manifest_files_for_release(
    config: FullReleasePromotionConfig,
    rel_paths: Sequence[str],
    deps: FullReleasePromotionDependencies,
) -> list[ManifestFile]:
    if config.files_with_paths is None:
        return list(
            deps.download_staged_artifacts_for_manifest(
                rel_paths,
                hf_repo_name=config.hf_repo_name,
                hf_repo_type=config.hf_repo_type,
                run_id=config.run_id,
            )
        )

    manifest_files = [
        (Path(path), repo_path) for path, repo_path in config.files_with_paths
    ]
    _assert_supplied_manifest_files_cover_release(rel_paths, manifest_files)
    return manifest_files


def _assert_supplied_manifest_files_cover_release(
    rel_paths: Sequence[str],
    manifest_files: Sequence[ManifestFile],
) -> None:
    manifest_paths = {repo_path for _, repo_path in manifest_files}
    missing_manifest_paths = [
        rel_path for rel_path in rel_paths if rel_path not in manifest_paths
    ]
    if missing_manifest_paths:
        raise ValueError(
            "Missing local files for release manifest: "
            + ", ".join(sorted(missing_manifest_paths))
        )

    missing_local_files = [
        str(path) for path, _ in manifest_files if not Path(path).exists()
    ]
    if missing_local_files:
        raise FileNotFoundError(
            "Missing local release manifest files: "
            + ", ".join(sorted(missing_local_files))
        )


def _finish_already_finalized_release(
    *,
    config: FullReleasePromotionConfig,
    rel_paths: Sequence[str],
    finalized_manifest: ReleaseManifest,
    deps: FullReleasePromotionDependencies,
) -> dict[str, Any]:
    completion_marker_path = _assert_finalized_release_has_completion_marker(
        config=config,
        deps=deps,
    )
    cleaned = _cleanup_staging_after_release(
        config=config,
        rel_paths=rel_paths,
        deps=deps,
        warning="Release %s was already finalized, but staging cleanup failed.",
    )
    return {
        "run_id": config.run_id,
        "version": config.version,
        "artifact_count": len(rel_paths),
        "hf_promoted": 0,
        "gcs_uploaded": 0,
        "release_manifest_artifacts": len(finalized_manifest["artifacts"]),
        "release_completion_marker": completion_marker_path,
        "staging_cleaned": cleaned,
        "already_finalized": True,
    }


def _assert_staging_complete(
    config: FullReleasePromotionConfig,
    rel_paths: Sequence[str],
    deps: FullReleasePromotionDependencies,
) -> None:
    missing = deps.list_missing_staged_artifacts(
        rel_paths,
        hf_repo_name=config.hf_repo_name,
        hf_repo_type=config.hf_repo_type,
        run_id=config.run_id,
    )
    if missing:
        raise FileNotFoundError(
            "Missing staged release artifacts: " + ", ".join(sorted(missing))
        )


def _assert_release_can_finalize(
    config: FullReleasePromotionConfig,
    rel_paths: Sequence[str],
    manifest_files: Sequence[ManifestFile],
    deps: FullReleasePromotionDependencies,
) -> None:
    should_finalize, missing_prefixes = deps.preflight_release_manifest_publish(
        manifest_files,
        version=config.version,
        new_repo_paths=rel_paths,
        hf_repo_name=config.hf_repo_name,
        hf_repo_type=config.hf_repo_type,
    )
    if not should_finalize:
        raise RuntimeError(
            "Cannot finalize release; staged artifact set is incomplete. "
            "Missing local-area prefixes: " + ", ".join(missing_prefixes)
        )


def _upload_version_manifest(
    config: FullReleasePromotionConfig,
    release_manifest: ReleaseManifest,
    deps: FullReleasePromotionDependencies,
) -> None:
    deps.upload_final_version_manifest(
        version=config.version,
        released_paths=_released_paths(release_manifest),
        run_id=config.run_id,
        hf_repo_name=config.hf_repo_name,
    )


def _upload_release_completion_marker(
    *,
    config: FullReleasePromotionConfig,
    release_manifest: ReleaseManifest,
    rel_paths: Sequence[str],
    promoted_hf: int,
    uploaded_gcs: int,
    create_tag: bool,
    deps: FullReleasePromotionDependencies,
) -> ReleaseManifest:
    return deps.upload_release_completion_marker(
        version=config.version,
        run_id=config.run_id,
        released_paths=rel_paths,
        expected_paths=rel_paths,
        release_manifest=release_manifest,
        hf_repo_name=config.hf_repo_name,
        hf_repo_type=config.hf_repo_type,
        promoted_hf=promoted_hf,
        uploaded_gcs=uploaded_gcs,
        create_tag=create_tag,
    )


def _assert_finalized_release_has_completion_marker(
    *,
    config: FullReleasePromotionConfig,
    deps: FullReleasePromotionDependencies,
) -> str:
    marker_path = release_completion_marker_path(config.version)
    if deps.release_completion_marker_exists(
        version=config.version,
        hf_repo_name=config.hf_repo_name,
        hf_repo_type=config.hf_repo_type,
    ):
        return marker_path

    raise RuntimeError(
        f"Release {config.version} is already finalized, but {marker_path} "
        f"is not present at tag {config.version}. Refusing to mutate release "
        "state after finalization; repair or migrate this release manually."
    )


def _released_paths(release_manifest: ReleaseManifest) -> list[str]:
    return sorted(
        artifact["path"] for artifact in release_manifest["artifacts"].values()
    )


def _cleanup_staging_after_release(
    *,
    config: FullReleasePromotionConfig,
    rel_paths: Sequence[str],
    deps: FullReleasePromotionDependencies,
    warning: str,
) -> int:
    if not config.cleanup_staging:
        return 0

    cleanup_paths = deps.dedupe_preserving_order(
        [*rel_paths, *config.extra_cleanup_paths]
    )
    try:
        return deps.cleanup_staging_hf(
            cleanup_paths,
            version=config.version,
            hf_repo_name=config.hf_repo_name,
            hf_repo_type=config.hf_repo_type,
            run_id=config.run_id,
        )
    except Exception:
        logging.warning(
            warning,
            config.version,
            exc_info=True,
        )
        return 0
