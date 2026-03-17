"""Upload pipeline artifacts to the stage-organized pipeline repo.

Mirrors existing build artifacts to
``policyengine/policyengine-us-data-pipeline`` with a folder
structure that groups files by pipeline stage and timestamps
each run.  All operations are additive — the production repo
(``policyengine/policyengine-us-data``) is never modified.

Failures are logged but never raised so that mirror uploads
cannot block the main pipeline.
"""

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import CommitOperationAdd, HfApi

from policyengine_us_data.utils.data_upload import (
    hf_create_commit_with_retry,
)
from policyengine_us_data.utils.manifest import (
    compute_file_checksum,
)

logger = logging.getLogger(__name__)

PIPELINE_REPO = "policyengine/policyengine-us-data-pipeline"
PIPELINE_REPO_TYPE = "model"


def get_pipeline_run_id() -> str:
    """Return a UTC timestamp identifier for this pipeline run.

    Checks the ``PIPELINE_RUN_ID`` environment variable first so
    that a single identifier can be shared across processes (e.g.
    separate Modal containers).  If unset, generates a new
    timestamp.

    Returns:
        String like ``'20260317T143000Z'``.
    """
    env_id = os.environ.get("PIPELINE_RUN_ID")
    if env_id:
        return env_id
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_git_info() -> Dict[str, object]:
    """Capture lightweight git provenance.

    Reimplemented here (~15 lines) instead of importing
    ``unified_calibration.get_git_provenance`` to avoid pulling
    in torch / l0-python and other heavy calibration deps.
    """
    info: Dict[str, object] = {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }
    try:
        info["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["git_branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["git_dirty"] = len(status) > 0
    except Exception:
        pass
    return info


def generate_stage_manifest(
    stage_name: str,
    run_id: str,
    files: List[Path],
) -> Dict:
    """Build a manifest dict for a pipeline stage.

    Args:
        stage_name: Stage identifier (e.g. ``'stage_1_base'``).
        run_id: Pipeline run identifier (UTC timestamp).
        files: Paths to the artifact files.

    Returns:
        Manifest dictionary matching the schema documented in
        ``pipeline_improvements.md``.
    """
    manifest: Dict[str, object] = {
        "stage": stage_name,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **_get_git_info(),
        "files": {},
    }
    for f in files:
        p = Path(f)
        if not p.exists():
            logger.warning("Skipping missing file for manifest: %s", p)
            continue
        manifest["files"][p.name] = {
            "sha256": compute_file_checksum(p),
            "size_bytes": p.stat().st_size,
        }
    return manifest


def mirror_to_pipeline(
    stage_name: str,
    files: List[Path],
    run_id: Optional[str] = None,
    manifest_only: bool = False,
    repo: str = PIPELINE_REPO,
) -> str:
    """Upload artifacts and manifest to the pipeline repo.

    This is the single-call interface used at every hook point.
    It generates a manifest, optionally uploads the files
    themselves, and commits everything to the pipeline repo
    under ``{run_id}/{stage_name}/``.

    Failures are logged as warnings and never propagated so
    that mirror uploads cannot block the main pipeline.

    Args:
        stage_name: Stage identifier (e.g. ``'stage_6_weights'``).
        files: Paths to the artifact files.
        run_id: Pipeline run identifier.  If ``None``, one is
            generated via :func:`get_pipeline_run_id`.
        manifest_only: If ``True``, upload only the manifest
            (with checksums) but not the actual files.  Used
            for stage_7 where files are too large to
            double-upload.
        repo: HuggingFace repo ID.

    Returns:
        The ``run_id`` that was used (so callers can pass it
        to subsequent stages for consistency).
    """
    if run_id is None:
        run_id = get_pipeline_run_id()

    try:
        manifest = generate_stage_manifest(stage_name, run_id, files)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as tmp:
            json.dump(manifest, tmp, indent=2)
            manifest_path = tmp.name

        prefix = f"{run_id}/{stage_name}"
        operations = [
            CommitOperationAdd(
                path_in_repo=f"{prefix}/manifest.json",
                path_or_fileobj=manifest_path,
            )
        ]

        if not manifest_only:
            for f in files:
                p = Path(f)
                if not p.exists():
                    logger.warning("Skipping missing file: %s", p)
                    continue
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"{prefix}/{p.name}",
                        path_or_fileobj=str(p),
                    )
                )

        token = os.environ.get("HUGGING_FACE_TOKEN")
        api = HfApi()

        hf_create_commit_with_retry(
            api=api,
            operations=operations,
            repo_id=repo,
            repo_type=PIPELINE_REPO_TYPE,
            token=token,
            commit_message=(f"Upload {stage_name} artifacts for run {run_id}"),
        )

        n_files = len(operations) - 1  # exclude manifest
        mode = "manifest-only" if manifest_only else "with files"
        logger.info(
            "Mirrored %s to pipeline repo (%s, %d files)",
            stage_name,
            mode,
            n_files,
        )

        # Clean up temp manifest file
        try:
            os.unlink(manifest_path)
        except OSError:
            pass

    except Exception:
        logger.warning(
            "Failed to mirror %s to pipeline repo — continuing without blocking",
            stage_name,
            exc_info=True,
        )

    return run_id
