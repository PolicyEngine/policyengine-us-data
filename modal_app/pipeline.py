"""
End-to-end versioned pipeline orchestrator for Modal.

Chains all dataset-building steps (build datasets, build calibration
package, fit weights, build H5s, stage, promote) into a single
coordinated run with diagnostics, resume support, and atomic
promotion.

**Stability assumption**: This pipeline is designed for production
use when the target branch is stable and not expected to change
during the run. All steps clone from branch tip independently;
artifacts flow through the shared pipeline volume. The run's
metadata records the SHA at orchestrator start for auditability.
If the branch changes mid-run, intermediate artifacts may come
from different commits. For development branches that are actively
changing, run individual steps manually instead.

Usage:
    # Full pipeline run
    modal run --detach modal_app/pipeline.py::main \\
        --action run --branch main --gpu A100-80GB --epochs 200

    # Check status
    modal run modal_app/pipeline.py::main --action status

    # Resume a failed run
    modal run --detach modal_app/pipeline.py::main \\
        --action run --resume-run-id <RUN_ID>

    # Promote a completed run
    modal run modal_app/pipeline.py::main \\
        --action promote --run-id <RUN_ID>
"""

import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import modal

# ── Modal resources ──────────────────────────────────────────────

app = modal.App("policyengine-us-data-pipeline")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

pipeline_volume = modal.Volume.from_name("pipeline-artifacts", create_if_missing=True)
staging_volume = modal.Volume.from_name("local-area-staging", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("uv", "tomli")
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"
PIPELINE_MOUNT = "/pipeline"
STAGING_MOUNT = "/staging"
ARTIFACTS_DIR = f"{PIPELINE_MOUNT}/artifacts"
RUNS_DIR = f"{PIPELINE_MOUNT}/runs"


# ── Run metadata ─────────────────────────────────────────────────


@dataclass
class RunMetadata:
    """Metadata for a pipeline run.

    Tracks run identity, progress, and diagnostics for
    auditability and resume support.
    """

    run_id: str
    branch: str
    sha: str
    version: str
    start_time: str
    status: str  # running | completed | failed | promoted
    step_timings: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        return cls(**data)


def generate_run_id(version: str, sha: str) -> str:
    """Generate a unique run ID.

    Format: {version}_{sha[:8]}_{YYYYMMDD_HHMMSS}
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{version}_{sha[:8]}_{ts}"


def write_run_meta(
    meta: RunMetadata,
    vol: modal.Volume,
) -> None:
    """Write run metadata to the pipeline volume."""
    run_dir = Path(RUNS_DIR) / meta.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)
    vol.commit()


def read_run_meta(
    run_id: str,
    vol: modal.Volume,
) -> RunMetadata:
    """Read run metadata from the pipeline volume."""
    vol.reload()
    meta_path = Path(RUNS_DIR) / run_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found for run {run_id} at {meta_path}")
    with open(meta_path) as f:
        return RunMetadata.from_dict(json.load(f))


def get_pinned_sha(branch: str) -> str:
    """Get the current tip SHA for a branch from GitHub."""
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            REPO_URL,
            f"refs/heads/{branch}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get SHA for branch {branch}: {result.stderr}")
    line = result.stdout.strip()
    if not line:
        raise RuntimeError(f"Branch {branch} not found in remote")
    return line.split()[0]


def get_version_from_branch(branch: str) -> str:
    """Get the package version from pyproject.toml on a
    branch by fetching just that file."""
    result = subprocess.run(
        [
            "git",
            "archive",
            f"--remote={REPO_URL}",
            branch,
            "pyproject.toml",
        ],
        capture_output=True,
    )
    # git archive --remote may not work with HTTPS;
    # fall back to cloning
    if result.returncode != 0:
        # Use a lightweight approach: fetch and read
        clone_dir = "/tmp/version_check"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "-b",
                branch,
                REPO_URL,
                clone_dir,
            ],
            capture_output=True,
        )
        import tomli

        with open(f"{clone_dir}/pyproject.toml", "rb") as f:
            pyproject = tomli.load(f)
        import shutil

        shutil.rmtree(clone_dir, ignore_errors=True)
        return pyproject["project"]["version"]

    # Parse from tar
    import io
    import tarfile

    tar = tarfile.open(fileobj=io.BytesIO(result.stdout))
    member = tar.extractfile("pyproject.toml")
    import tomli

    pyproject = tomli.load(member)
    return pyproject["project"]["version"]


def archive_diagnostics(
    run_id: str,
    result_bytes: dict,
    vol: modal.Volume,
    prefix: str = "",
) -> None:
    """Archive calibration diagnostics to the run directory."""
    diag_dir = Path(RUNS_DIR) / run_id / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "log": f"{prefix}unified_diagnostics.csv",
        "cal_log": f"{prefix}calibration_log.csv",
        "config": f"{prefix}unified_run_config.json",
    }

    for key, filename in file_map.items():
        data = result_bytes.get(key)
        if data:
            path = diag_dir / filename
            with open(path, "wb") as f:
                f.write(data)
            print(f"  Archived {filename} ({len(data):,} bytes)")

    vol.commit()


def _step_completed(meta: RunMetadata, step: str) -> bool:
    """Check if a step is marked completed in metadata."""
    timing = meta.step_timings.get(step, {})
    return timing.get("status") == "completed"


def _record_step(
    meta: RunMetadata,
    step: str,
    start: float,
    vol: modal.Volume,
    status: str = "completed",
) -> None:
    """Record step timing and status in metadata."""
    meta.step_timings[step] = {
        "start": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
        "end": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - start, 1),
        "status": status,
    }
    write_run_meta(meta, vol)


# ── Imports from other Modal apps ────────────────────────────────
# These are imported at function call time to avoid
# cross-app import issues at module level.


def _get_data_build():
    """Import build_datasets from data_build app."""
    from modal_app.data_build import build_datasets

    return build_datasets


def _get_calibration_funcs():
    """Import calibration functions."""
    from modal_app.remote_calibration_runner import (
        build_package_remote,
        PACKAGE_GPU_FUNCTIONS,
    )

    return build_package_remote, PACKAGE_GPU_FUNCTIONS


def _get_local_area_funcs():
    """Import local area publishing functions."""
    from modal_app.local_area import (
        coordinate_publish,
        coordinate_national_publish,
        promote_publish,
        promote_national_publish,
    )

    return (
        coordinate_publish,
        coordinate_national_publish,
        promote_publish,
        promote_national_publish,
    )


# ── Stage base datasets ─────────────────────────────────────────


def stage_base_datasets(run_id: str, version: str) -> None:
    """Upload source_imputed + policy_data.db from pipeline
    volume to HF staging/.

    Reads artifacts from /pipeline/artifacts/ and uploads
    via upload_to_staging_hf().

    Args:
        run_id: The current run ID (for logging).
        version: Package version string for the commit.
    """
    artifacts = Path(ARTIFACTS_DIR)

    source_imputed = artifacts / "source_imputed_stratified_extended_cps.h5"
    policy_db = artifacts / "policy_data.db"

    files_with_paths = []
    if source_imputed.exists():
        files_with_paths.append(
            (
                source_imputed,
                "calibration/source_imputed_stratified_extended_cps.h5",
            )
        )
        print(f"  source_imputed: {source_imputed.stat().st_size:,} bytes")
    else:
        print("  WARNING: source_imputed not found, skipping")

    if policy_db.exists():
        files_with_paths.append((policy_db, "calibration/policy_data.db"))
        print(f"  policy_data.db: {policy_db.stat().st_size:,} bytes")
    else:
        print("  WARNING: policy_data.db not found, skipping")

    if not files_with_paths:
        print("  No base datasets to stage")
        return

    from policyengine_us_data.utils.data_upload import (
        upload_to_staging_hf,
    )

    count = upload_to_staging_hf(files_with_paths, version)
    print(f"  Staged {count} base dataset(s) to HF")


def upload_run_diagnostics(
    run_id: str,
) -> None:
    """Upload run diagnostics to HF for archival."""
    diag_dir = Path(RUNS_DIR) / run_id / "diagnostics"
    if not diag_dir.exists():
        print("  No diagnostics to upload")
        return

    files = list(diag_dir.glob("*"))
    if not files:
        print("  No diagnostic files found")
        return

    print(f"  Found {len(files)} diagnostic file(s) to upload")
    # Upload diagnostics via HF API
    from huggingface_hub import HfApi

    api = HfApi()
    token = os.environ.get("HUGGING_FACE_TOKEN")

    for f in files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=(f"calibration/runs/{run_id}/diagnostics/{f.name}"),
            repo_id="policyengine/policyengine-us-data",
            repo_type="model",
            token=token,
        )
        print(f"  Uploaded {f.name}")


# ── Orchestrator ─────────────────────────────────────────────────


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=172800,  # 48 hours
    volumes={
        PIPELINE_MOUNT: pipeline_volume,
        STAGING_MOUNT: staging_volume,
    },
    secrets=[hf_secret, gcp_secret],
)
def run_pipeline(
    branch: str = "main",
    gpu: str = "A100-80GB",
    epochs: int = 1000,
    national_gpu: str = "T4",
    national_epochs: int = 1000,
    num_workers: int = 8,
    n_clones: int = 430,
    skip_national: bool = False,
    resume_run_id: str = None,
) -> str:
    """Run the full pipeline end-to-end.

    Args:
        branch: Git branch to build from.
        gpu: GPU type for regional calibration.
        epochs: Training epochs for regional calibration.
        national_gpu: GPU type for national calibration.
        national_epochs: Training epochs for national.
        num_workers: Number of parallel H5 workers.
        n_clones: Number of clones for H5 building.
        skip_national: Skip national calibration/H5.
        resume_run_id: Resume a previously failed run.

    Returns:
        The run ID for use with promote.
    """
    # ── Setup GCP credentials ──
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_path = "/tmp/gcp-credentials.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    # ── Initialize or resume run ──
    sha = get_pinned_sha(branch)
    version = get_version_from_branch(branch)

    if resume_run_id:
        print(f"Resuming run {resume_run_id}...")
        meta = read_run_meta(resume_run_id, pipeline_volume)
        if meta.sha != sha:
            raise RuntimeError(
                f"Branch {branch} has moved since run "
                f"started.\n"
                f"  Run SHA:     {meta.sha[:12]}\n"
                f"  Current SHA: {sha[:12]}\n"
                f"Start a fresh run instead."
            )
        meta.status = "running"
        run_id = resume_run_id
    else:
        run_id = generate_run_id(version, sha)
        meta = RunMetadata(
            run_id=run_id,
            branch=branch,
            sha=sha,
            version=version,
            start_time=datetime.now(timezone.utc).isoformat(),
            status="running",
        )

    # Create run directory
    run_dir = Path(RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics").mkdir(exist_ok=True)

    # Create artifacts directory
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    write_run_meta(meta, pipeline_volume)

    print("=" * 60)
    print("PIPELINE RUN")
    print("=" * 60)
    print(f"  Run ID:  {run_id}")
    print(f"  Branch:  {branch}")
    print(f"  SHA:     {sha[:12]}")
    print(f"  Version: {version}")
    print(f"  GPU:     {gpu} (regional)")
    if not skip_national:
        print(f"  GPU:     {national_gpu} (national)")
    print(f"  Epochs:  {epochs}")
    print(f"  Workers: {num_workers}")
    if resume_run_id:
        completed = [
            s for s, t in meta.step_timings.items() if t.get("status") == "completed"
        ]
        print(f"  Resume:  skipping {completed}")
    print("=" * 60)

    try:
        # ── Step 1: Build datasets ──
        if not _step_completed(meta, "build_datasets"):
            print("\n[Step 1/5] Building datasets...")
            step_start = time.time()

            build_datasets = _get_data_build()
            build_datasets.remote(
                upload=False,
                branch=branch,
                sequential=False,
                skip_tests=True,
                skip_enhanced_cps=True,
            )

            # The build_datasets step produces files in its
            # own volume. Key outputs (source_imputed,
            # policy_data.db) are staged to HF in step 4.
            # TODO(#617): When pipeline_artifacts.py lands,
            # call mirror_to_pipeline() here for audit trail.
            _record_step(
                meta,
                "build_datasets",
                step_start,
                pipeline_volume,
            )
            print(
                f"  Completed in {meta.step_timings['build_datasets']['duration_s']}s"
            )
        else:
            print("\n[Step 1/5] Build datasets (skipped - completed)")

        # ── Step 2: Build calibration package ──
        if not _step_completed(meta, "build_package"):
            print("\n[Step 2/5] Building calibration package...")
            step_start = time.time()

            (
                build_package_remote,
                _,
            ) = _get_calibration_funcs()
            pkg_path = build_package_remote.remote(
                branch=branch,
                workers=num_workers,
                n_clones=n_clones,
            )
            print(f"  Package at: {pkg_path}")

            _record_step(
                meta,
                "build_package",
                step_start,
                pipeline_volume,
            )
            print(f"  Completed in {meta.step_timings['build_package']['duration_s']}s")
        else:
            print("\n[Step 2/5] Build package (skipped - completed)")

        # ── Step 3: Fit weights (parallel) ──
        if not _step_completed(meta, "fit_weights"):
            print("\n[Step 3/5] Fitting calibration weights...")
            step_start = time.time()

            _, PACKAGE_GPU_FUNCTIONS = _get_calibration_funcs()

            vol_path = "/calibration-data/calibration_package.pkl"

            # Spawn regional fit
            regional_func = PACKAGE_GPU_FUNCTIONS[gpu]
            print(f"  Spawning regional fit ({gpu}, {epochs} epochs)...")
            regional_handle = regional_func.spawn(
                branch=branch,
                epochs=epochs,
                volume_package_path=vol_path,
            )

            # Spawn national fit (if enabled)
            national_handle = None
            if not skip_national:
                national_func = PACKAGE_GPU_FUNCTIONS[national_gpu]
                print(
                    f"  Spawning national fit "
                    f"({national_gpu}, "
                    f"{national_epochs} epochs)..."
                )
                national_handle = national_func.spawn(
                    branch=branch,
                    epochs=national_epochs,
                    volume_package_path=vol_path,
                    target_config=None,
                )

            # Collect regional results
            print("  Waiting for regional fit...")
            regional_result = regional_handle.get()
            print("  Regional fit complete. Writing to volume...")

            # Write regional results to pipeline volume
            with pipeline_volume.batch_upload(force=True) as batch:
                batch.put(
                    BytesIO(regional_result["weights"]),
                    "artifacts/calibration_weights.npy",
                )
                if regional_result.get("config"):
                    batch.put(
                        BytesIO(regional_result["config"]),
                        "artifacts/unified_run_config.json",
                    )
                if regional_result.get("blocks"):
                    batch.put(
                        BytesIO(regional_result["blocks"]),
                        "artifacts/stacked_blocks.npy",
                    )
                if regional_result.get("geo_labels"):
                    batch.put(
                        BytesIO(regional_result["geo_labels"]),
                        "artifacts/geo_labels.json",
                    )
                if regional_result.get("geography"):
                    batch.put(
                        BytesIO(regional_result["geography"]),
                        "artifacts/geography.npz",
                    )

            # Also upload to HF for downstream steps
            # that download from HF
            from policyengine_us_data.utils.huggingface import (
                upload_calibration_artifacts,
            )

            # Save regional results locally for upload
            _save_result_locally(regional_result, prefix="")
            upload_calibration_artifacts(
                weights_path="/tmp/calibration_weights.npy",
                log_dir="/tmp",
                prefix="",
            )

            archive_diagnostics(
                run_id,
                regional_result,
                pipeline_volume,
                prefix="",
            )

            # Collect national results
            if national_handle is not None:
                print("  Waiting for national fit...")
                national_result = national_handle.get()
                print("  National fit complete. Writing to volume...")

                with pipeline_volume.batch_upload(force=True) as batch:
                    batch.put(
                        BytesIO(national_result["weights"]),
                        "artifacts/national_calibration_weights.npy",
                    )
                    if national_result.get("config"):
                        batch.put(
                            BytesIO(national_result["config"]),
                            "artifacts/national_unified_run_config.json",
                        )
                    if national_result.get("geography"):
                        batch.put(
                            BytesIO(national_result["geography"]),
                            "artifacts/national_geography.npz",
                        )

                # Upload national to HF
                _save_result_locally(
                    national_result,
                    prefix="national_",
                )
                upload_calibration_artifacts(
                    weights_path=("/tmp/national_calibration_weights.npy"),
                    log_dir="/tmp",
                    prefix="national_",
                )

                archive_diagnostics(
                    run_id,
                    national_result,
                    pipeline_volume,
                    prefix="national_",
                )

            _record_step(
                meta,
                "fit_weights",
                step_start,
                pipeline_volume,
            )
            print(f"  Completed in {meta.step_timings['fit_weights']['duration_s']}s")
        else:
            print("\n[Step 3/5] Fit weights (skipped - completed)")

        # ── Step 4: Build H5s + stage + diagnostics (parallel) ──
        # Per plan: all four tasks run in parallel:
        #   4a. coordinate_publish (regional H5s)
        #   4b. coordinate_national_publish (national H5)
        #   4c. stage_base_datasets (datasets → HF staging)
        #   4d. upload_run_diagnostics (diagnostics → HF)
        if not _step_completed(meta, "publish_and_stage"):
            print(
                "\n[Step 4/5] Building H5s, staging datasets, "
                "uploading diagnostics (parallel)..."
            )
            step_start = time.time()

            (
                coordinate_publish,
                coordinate_national_publish,
                _,
                _,
            ) = _get_local_area_funcs()

            # Spawn H5 builds (run on separate Modal containers)
            print(f"  Spawning regional H5 build ({num_workers} workers)...")
            regional_h5_handle = coordinate_publish.spawn(
                branch=branch,
                num_workers=num_workers,
                skip_upload=False,
                n_clones=n_clones,
            )

            national_h5_handle = None
            if not skip_national:
                print("  Spawning national H5 build...")
                national_h5_handle = coordinate_national_publish.spawn(
                    branch=branch,
                    n_clones=n_clones,
                )

            # While H5 builds run, stage base datasets
            # and upload diagnostics in this container
            pipeline_volume.reload()

            print("  Staging base datasets to HF...")
            stage_base_datasets(run_id, version)

            print("  Uploading run diagnostics...")
            upload_run_diagnostics(run_id)

            # Now wait for H5 builds to finish
            print("  Waiting for regional H5 build...")
            regional_h5_result = regional_h5_handle.get()
            print(f"  Regional H5: {regional_h5_result}")

            if national_h5_handle is not None:
                print("  Waiting for national H5 build...")
                national_h5_result = national_h5_handle.get()
                print(f"  National H5: {national_h5_result}")

            _record_step(
                meta,
                "publish_and_stage",
                step_start,
                pipeline_volume,
            )
            print(
                f"  Completed in "
                f"{meta.step_timings['publish_and_stage']['duration_s']}s"
            )
        else:
            print("\n[Step 4/5] Publish + stage (skipped - completed)")

        # ── Step 5: Finalize ──
        print("\n[Step 5/5] Finalizing run...")
        meta.status = "completed"
        write_run_meta(meta, pipeline_volume)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Run ID: {run_id}")
        print(f"  Status: {meta.status}")
        _print_step_timings(meta)
        print(
            f"\nTo promote, run:\n"
            f"  modal run modal_app/pipeline.py"
            f"::main --action promote "
            f"--run-id {run_id}"
        )
        print("=" * 60)

        return run_id

    except Exception as e:
        meta.status = "failed"
        meta.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        write_run_meta(meta, pipeline_volume)
        print(f"\nPIPELINE FAILED: {e}")
        print(f"Resume with: --resume-run-id {run_id}")
        raise


def _save_result_locally(result: dict, prefix: str) -> None:
    """Save calibration result bytes to /tmp for upload."""
    if result.get("weights"):
        with open(
            f"/tmp/{prefix}calibration_weights.npy",
            "wb",
        ) as f:
            f.write(result["weights"])
    if result.get("blocks"):
        with open(f"/tmp/{prefix}stacked_blocks.npy", "wb") as f:
            f.write(result["blocks"])
    if result.get("geo_labels"):
        with open(f"/tmp/{prefix}geo_labels.json", "wb") as f:
            f.write(result["geo_labels"])
    if result.get("geography"):
        with open(f"/tmp/{prefix}geography.npz", "wb") as f:
            f.write(result["geography"])
    if result.get("log"):
        with open(
            f"/tmp/{prefix}unified_diagnostics.csv",
            "wb",
        ) as f:
            f.write(result["log"])
    if result.get("cal_log"):
        with open(f"/tmp/{prefix}calibration_log.csv", "wb") as f:
            f.write(result["cal_log"])
    if result.get("config"):
        with open(
            f"/tmp/{prefix}unified_run_config.json",
            "wb",
        ) as f:
            f.write(result["config"])


def _print_step_timings(meta: RunMetadata) -> None:
    """Print formatted step timings."""
    total = 0.0
    for step, timing in meta.step_timings.items():
        dur = timing.get("duration_s", 0)
        total += dur
        status = timing.get("status", "unknown")
        print(f"  {step}: {dur}s ({status})")
    hours = total / 3600
    print(f"  TOTAL: {total:.0f}s ({hours:.1f}h)")


# ── Promote ──────────────────────────────────────────────────────


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=7200,
    volumes={
        PIPELINE_MOUNT: pipeline_volume,
        STAGING_MOUNT: staging_volume,
    },
    secrets=[hf_secret, gcp_secret],
)
def promote_run(
    run_id: str,
    version: str = None,
) -> str:
    """Promote a completed pipeline run to production.

    1. Verify run status is "completed"
    2. Promote H5s (regional + national) via existing
       promote functions
    3. Register version in version_manifest.json
    4. Update run status to "promoted"

    Args:
        run_id: The run ID to promote.
        version: Override version (default: from run
            metadata).

    Returns:
        Summary message.
    """
    # Setup GCP
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_path = "/tmp/gcp-credentials.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    meta = read_run_meta(run_id, pipeline_volume)

    if meta.status not in ("completed", "promoted"):
        raise RuntimeError(
            f"Run {run_id} has status "
            f"'{meta.status}'. Only completed runs "
            f"can be promoted."
        )

    if meta.status == "promoted":
        print(f"WARNING: Run {run_id} was already promoted. Re-promoting...")

    version = version or meta.version

    print("=" * 60)
    print("PROMOTING PIPELINE RUN")
    print("=" * 60)
    print(f"  Run ID:  {run_id}")
    print(f"  Version: {version}")
    print(f"  Branch:  {meta.branch}")
    print(f"  SHA:     {meta.sha[:12]}")
    print("=" * 60)

    # Promote base datasets from staging → production
    print("\nPromoting base datasets (staging → production)...")
    try:
        from policyengine_us_data.utils.data_upload import (
            promote_staging_to_production_hf,
        )

        base_files = [
            "calibration/source_imputed_stratified_extended_cps.h5",
            "calibration/policy_data.db",
        ]
        count = promote_staging_to_production_hf(base_files, version)
        print(f"  Promoted {count} base dataset(s)")
    except Exception as e:
        print(f"  WARNING: Base dataset promotion: {e}")

    # Promote H5s via existing functions
    (
        _,
        _,
        promote_publish,
        promote_national_publish,
    ) = _get_local_area_funcs()

    print("\nPromoting regional H5s...")
    try:
        regional_result = promote_publish.remote(
            branch=meta.branch,
            version=version,
        )
        print(f"  {regional_result}")
    except Exception as e:
        print(f"  WARNING: Regional promote: {e}")

    print("\nPromoting national H5...")
    try:
        national_result = promote_national_publish.remote(
            branch=meta.branch,
        )
        print(f"  {national_result}")
    except Exception as e:
        print(f"  WARNING: National promote: {e}")

    # Register version in manifest
    print("\nRegistering version in manifest...")
    try:
        from policyengine_us_data.utils.version_manifest import (
            build_manifest,
            upload_manifest,
        )

        # Build manifest from GCS blobs
        blob_names = [
            "calibration/source_imputed_stratified_extended_cps.h5",
            "calibration/policy_data.db",
            "calibration/calibration_weights.npy",
        ]
        manifest = build_manifest(
            version=version,
            blob_names=blob_names,
        )
        manifest.pipeline_run_id = run_id
        manifest.diagnostics_path = f"calibration/runs/{run_id}/diagnostics/"
        upload_manifest(manifest)
        print(f"  Registered version {version} in version_manifest.json")
    except Exception as e:
        print(f"  WARNING: Version registration failed: {e}")
        print("  This can be done manually later via version_manifest.py")

    # Update run status
    meta.status = "promoted"
    write_run_meta(meta, pipeline_volume)

    print("\n" + "=" * 60)
    print("PROMOTION COMPLETE")
    print("=" * 60)
    print(f"  Version {version} is now live.")
    print("=" * 60)

    return f"Promoted run {run_id} as version {version}"


# ── Status ───────────────────────────────────────────────────────


@app.function(
    image=image,
    timeout=60,
    volumes={PIPELINE_MOUNT: pipeline_volume},
)
def pipeline_status(
    run_id: str = None,
) -> str:
    """Get pipeline status.

    If run_id is provided, show that run's details.
    Otherwise, list all runs.
    """
    pipeline_volume.reload()
    runs_dir = Path(RUNS_DIR)

    if not runs_dir.exists():
        return "No pipeline runs found."

    if run_id:
        meta = read_run_meta(run_id, pipeline_volume)
        lines = [
            f"Run: {meta.run_id}",
            f"  Branch:  {meta.branch}",
            f"  SHA:     {meta.sha[:12]}",
            f"  Version: {meta.version}",
            f"  Status:  {meta.status}",
            f"  Started: {meta.start_time}",
        ]
        if meta.error:
            lines.append(f"  Error:   {meta.error[:200]}")
        if meta.step_timings:
            lines.append("  Steps:")
            for step, timing in meta.step_timings.items():
                dur = timing.get("duration_s", "?")
                status = timing.get("status", "unknown")
                lines.append(f"    {step}: {dur}s ({status})")
        return "\n".join(lines)

    # List all runs
    runs = []
    for entry in sorted(runs_dir.iterdir()):
        meta_path = entry / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
            runs.append(
                f"  {data['run_id']}: "
                f"{data['status']} "
                f"(branch={data['branch']}, "
                f"v={data['version']})"
            )

    if not runs:
        return "No pipeline runs found."

    return "Pipeline runs:\n" + "\n".join(runs)


# ── Local entrypoint ─────────────────────────────────────────────


@app.local_entrypoint()
def main(
    action: str = "run",
    branch: str = "main",
    run_id: str = None,
    resume_run_id: str = None,
    gpu: str = "A100-80GB",
    epochs: int = 1000,
    national_gpu: str = "T4",
    national_epochs: int = 1000,
    num_workers: int = 8,
    n_clones: int = 430,
    skip_national: bool = False,
    version: str = None,
):
    """Pipeline entrypoint.

    Actions:
        run     - Run the full pipeline
        status  - Show pipeline status
        promote - Promote a completed run
    """
    if action == "run":
        result = run_pipeline.remote(
            branch=branch,
            gpu=gpu,
            epochs=epochs,
            national_gpu=national_gpu,
            national_epochs=national_epochs,
            num_workers=num_workers,
            n_clones=n_clones,
            skip_national=skip_national,
            resume_run_id=resume_run_id,
        )
        print(f"\nPipeline run complete: {result}")

    elif action == "status":
        result = pipeline_status.remote(
            run_id=run_id,
        )
        print(result)

    elif action == "promote":
        if not run_id:
            raise ValueError("--run-id is required for promote")
        result = promote_run.remote(
            run_id=run_id,
            version=version,
        )
        print(result)

    else:
        raise ValueError(f"Unknown action: {action}. Use: run, status, promote")
