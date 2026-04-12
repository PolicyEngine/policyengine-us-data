"""
Promote locally-built H5 files to production via HF staging.

Uploads local state/district H5s to HF staging/, promotes to
production paths, uploads to GCS, then cleans up staging.
No Modal required.

Usage:
    python -m policyengine_us_data.calibration.promote_local_h5s \
        --local-dir local_area_build \
        --area-types states,districts

    # Dry run (stage only, don't promote):
    python -m policyengine_us_data.calibration.promote_local_h5s \
        --local-dir local_area_build --stage-only

    # Promote previously staged files:
    python -m policyengine_us_data.calibration.promote_local_h5s \
        --local-dir local_area_build --promote-only
"""

import argparse
import logging
from importlib import metadata
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from policyengine_us_data.utils.data_upload import (
    upload_to_staging_hf,
    preflight_release_manifest_publish,
    promote_staging_to_production_hf,
    upload_from_hf_staging_to_gcs,
    cleanup_staging_hf,
    publish_release_manifest_to_hf,
)
from policyengine_us_data.utils.version_manifest import (
    HFVersionInfo,
    build_manifest,
    upload_manifest,
)

logger = logging.getLogger(__name__)


def collect_files(local_dir: Path, area_types: list) -> list:
    files = []
    for area_type in area_types:
        area_dir = local_dir / area_type
        if not area_dir.exists():
            logger.warning("Directory %s not found, skipping", area_dir)
            continue
        h5s = sorted(area_dir.glob("*.h5"))
        logger.info("Found %d H5 files in %s/", len(h5s), area_type)
        for f in h5s:
            files.append((f, f"{area_type}/{f.name}"))
    return files


def collect_staged_rel_paths(area_types: list, run_id: str = "") -> list:
    api = HfApi()
    prefix = f"staging/{run_id}" if run_id else "staging"
    repo_files = api.list_repo_files(
        repo_id="policyengine/policyengine-us-data",
        repo_type="model",
    )

    rel_paths = []
    for path in repo_files:
        for area_type in area_types:
            candidate_prefix = f"{prefix}/{area_type}/"
            if path.startswith(candidate_prefix) and path.endswith(".h5"):
                rel_paths.append(path.removeprefix(f"{prefix}/"))
                break

    return sorted(rel_paths)


def download_staged_files(rel_paths: list, run_id: str = "") -> list:
    prefix = f"staging/{run_id}" if run_id else "staging"
    files = []
    for rel_path in rel_paths:
        local_path = Path(
            hf_hub_download(
                repo_id="policyengine/policyengine-us-data",
                filename=f"{prefix}/{rel_path}",
                repo_type="model",
            )
        )
        files.append((local_path, rel_path))
    return files


def stage(files: list, version: str, run_id: str = ""):
    logger.info("Uploading %d files to HF staging/...", len(files))
    n = upload_to_staging_hf(files, version=version, run_id=run_id)
    logger.info("Staged %d files", n)


def promote(files: list, rel_paths: list, version: str, run_id: str = ""):
    manifest_files = (
        [(local_path, rel_path) for local_path, rel_path in files]
        if files
        else download_staged_files(rel_paths, run_id=run_id)
    )
    should_finalize, missing_prefixes = preflight_release_manifest_publish(
        manifest_files,
        version=version,
        new_repo_paths=rel_paths,
    )

    logger.info(
        "Promoting %d files from staging/ to production...",
        len(rel_paths),
    )
    promote_staging_to_production_hf(rel_paths, version=version, run_id=run_id)

    logger.info("Uploading %d files to GCS from HF staging...", len(rel_paths))
    upload_from_hf_staging_to_gcs(rel_paths, version=version, run_id=run_id)
    logger.info("Publishing release manifest...")
    manifest = publish_release_manifest_to_hf(
        manifest_files,
        version=version,
        create_tag=should_finalize,
    )
    if should_finalize:
        upload_manifest(
            build_manifest(
                version=version,
                blob_names=sorted(
                    artifact["path"] for artifact in manifest["artifacts"].values()
                ),
                hf_info=HFVersionInfo(
                    repo="policyengine/policyengine-us-data",
                    commit=version,
                ),
            )
        )
    else:
        logger.warning(
            "Promoted only a partial artifact set for %s; missing prefixes %s. "
            "Leaving the release untagged until all required local-area artifacts "
            "have been published.",
            version,
            ", ".join(missing_prefixes),
        )

    logger.info("Cleaning up staging/...")
    cleanup_staging_hf(rel_paths, version=version, run_id=run_id)
    logger.info("Done — %d files promoted to production", len(rel_paths))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Promote local H5 files to production via HF staging"
    )
    parser.add_argument(
        "--local-dir",
        default="local_area_build",
        help="Directory containing states/ and districts/ subdirs",
    )
    parser.add_argument(
        "--area-types",
        default="national,states,districts,cities",
        help="Comma-separated area types to publish (default: national,states,districts,cities)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version string (default: from package metadata)",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Upload to HF staging only, don't promote",
    )
    parser.add_argument(
        "--promote-only",
        action="store_true",
        help="Promote previously staged files (skip upload to staging)",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run ID to scope HF staging paths (e.g. staging/{run_id}/...)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args(argv)
    local_dir = Path(args.local_dir)
    area_types = [t.strip() for t in args.area_types.split(",")]
    version = args.version or metadata.version("policyengine-us-data")

    logger.info("Version: %s", version)
    logger.info("Local dir: %s", local_dir)
    logger.info("Area types: %s", area_types)

    files = collect_files(local_dir, area_types)
    if not files and not args.promote_only:
        logger.error("No H5 files found")
        return

    run_id = args.run_id

    if args.promote_only:
        rel_paths = collect_staged_rel_paths(area_types, run_id=run_id)
        if not rel_paths:
            logger.error("No staged H5 files found")
            return
        promote([], rel_paths, version, run_id=run_id)
    elif args.stage_only:
        stage(files, version, run_id=run_id)
    else:
        rel_paths = [rp for _, rp in files]
        stage(files, version, run_id=run_id)
        promote(files, rel_paths, version, run_id=run_id)


if __name__ == "__main__":
    main()
