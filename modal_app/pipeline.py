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
        --action run --branch main --gpu T4 --epochs 200

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
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image as image

# ── Modal resources ──────────────────────────────────────────────

app = modal.App("policyengine-us-data-pipeline")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

pipeline_volume = modal.Volume.from_name("pipeline-artifacts", create_if_missing=True)
staging_volume = modal.Volume.from_name("local-area-staging", create_if_missing=True)

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
    """Get the package version from the pre-baked pyproject.toml.

    The branch parameter is kept for API compatibility but is
    no longer used -- version comes from the baked source.
    """
    import tomllib

    pyproject_path = "/root/policyengine-us-data/pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
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


def find_resumable_run(branch: str, sha: str, vol: modal.Volume) -> Optional[str]:
    """Find an existing running run for the same branch+sha."""
    vol.reload()
    runs_dir = Path(RUNS_DIR)
    if not runs_dir.exists():
        return None

    best_run_id = None
    best_start = ""

    for entry in runs_dir.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                data = json.load(f)
            if (
                data.get("branch") == branch
                and data.get("sha") == sha
                and data.get("status") == "running"
            ):
                start = data.get("start_time", "")
                if start > best_start:
                    best_start = start
                    best_run_id = data.get("run_id")
        except (json.JSONDecodeError, KeyError):
            continue

    return best_run_id


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


# ── Include other Modal apps ─────────────────────────────────────
# app.include() merges functions from other apps into this one,
# ensuring Modal mounts their files and registers their functions
# (with their GPU/memory/volume configs) in the ephemeral run.
# sys.path setup is handled at the top of this file.

from modal_app.data_build import app as _data_build_app
from modal_app.data_build import build_datasets

app.include(_data_build_app)

from modal_app.remote_calibration_runner import app as _calibration_app
from modal_app.remote_calibration_runner import (
    build_package_remote,
    PACKAGE_GPU_FUNCTIONS,
)

app.include(_calibration_app)

from modal_app.local_area import app as _local_area_app
from modal_app.local_area import (
    coordinate_publish,
    coordinate_national_publish,
    promote_publish,
    promote_national_publish,
    queue_coordinator,
)

app.include(_local_area_app)


# ── Stage base datasets ─────────────────────────────────────────


def _setup_repo() -> None:
    """Change to the pre-baked repo directory."""
    os.chdir("/root/policyengine-us-data")


def stage_base_datasets(
    run_id: str,
    version: str,
    branch: str,
) -> None:
    """Upload source_imputed + policy_data.db from pipeline
    volume to HF staging/.

    Clones the repo and shells out to upload_to_staging_hf()
    via subprocess, consistent with other Modal apps.

    Args:
        run_id: The current run ID (for logging).
        version: Package version string for the commit.
        branch: Git branch for repo clone.
    """
    artifacts = Path(ARTIFACTS_DIR)

    files_with_paths = []

    # Stage all intermediate H5 datasets for lineage tracing
    # source_imputed* goes to calibration/ (promote expects that path)
    for h5_file in sorted(artifacts.glob("*.h5")):
        if h5_file.name.startswith("source_imputed"):
            repo_path = f"calibration/{h5_file.name}"
        else:
            repo_path = f"datasets/{h5_file.name}"
        files_with_paths.append((str(h5_file), repo_path))
        print(f"  {h5_file.name} -> {repo_path}: {h5_file.stat().st_size:,} bytes")

    policy_db = artifacts / "policy_data.db"
    if policy_db.exists():
        files_with_paths.append((str(policy_db), "calibration/policy_data.db"))
        print(f"  policy_data.db: {policy_db.stat().st_size:,} bytes")
    else:
        print("  WARNING: policy_data.db not found, skipping")

    if not files_with_paths:
        print("  No base datasets to stage")
        return

    _setup_repo()

    # Build the upload script as a Python snippet
    import json as _json

    pairs_json = _json.dumps(files_with_paths)
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"""
import json
from policyengine_us_data.utils.data_upload import (
    upload_to_staging_hf,
)

pairs = json.loads('''{pairs_json}''')
files_with_paths = [(p, r) for p, r in pairs]
count = upload_to_staging_hf(files_with_paths, "{version}", run_id="{run_id}")
print(f"Staged {{count}} base dataset(s) to HF")
""",
        ],
        cwd="/root/policyengine-us-data",
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Base dataset staging failed: {result.stderr}")
    print(f"  {result.stdout.strip()}")


def upload_run_diagnostics(
    run_id: str,
    branch: str,
) -> None:
    """Upload run diagnostics to HF for archival.

    Shells out via subprocess for consistency with other
    Modal apps and to avoid package dependencies in the
    orchestrator image.

    Args:
        run_id: The current run ID.
        branch: Git branch for repo clone.
    """
    diag_dir = Path(RUNS_DIR) / run_id / "diagnostics"
    if not diag_dir.exists():
        print("  No diagnostics to upload")
        return

    files = list(diag_dir.glob("*"))
    if not files:
        print("  No diagnostic files found")
        return

    print(f"  Found {len(files)} diagnostic file(s) to upload")

    # Build file list as JSON for the subprocess
    import json as _json

    file_entries = [
        (str(f), f"calibration/runs/{run_id}/diagnostics/{f.name}") for f in files
    ]
    entries_json = _json.dumps(file_entries)

    _setup_repo()

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"""
import json, os
from huggingface_hub import HfApi

entries = json.loads('''{entries_json}''')
api = HfApi()
token = os.environ.get("HUGGING_FACE_TOKEN")
for local_path, repo_path in entries:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id="policyengine/policyengine-us-data",
        repo_type="model",
        token=token,
    )
    print(f"Uploaded {{repo_path}}")
""",
        ],
        cwd="/root/policyengine-us-data",
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Diagnostics upload failed: {result.stderr}")
    print(f"  {result.stdout.strip()}")


def _write_validation_diagnostics(
    run_id: str,
    regional_result,
    national_result,
    meta: RunMetadata,
    vol: modal.Volume,
) -> None:
    """Aggregate validation rows into a diagnostics CSV.

    Extracts validation_rows from coordinate_publish and
    national_validation from coordinate_national_publish,
    writes them to runs/{run_id}/diagnostics/validation_results.csv,
    and records a summary in meta.json.
    """
    import csv

    validation_rows = []

    # Extract regional validation rows
    if isinstance(regional_result, dict):
        v_rows = regional_result.get("validation_rows", [])
        if v_rows:
            validation_rows.extend(v_rows)
            print(f"  Collected {len(v_rows)} regional validation rows")

    # Extract national validation output
    national_output = ""
    if isinstance(national_result, dict):
        national_output = national_result.get("national_validation", "")
        if national_output:
            print("  National validation output captured")

    if not validation_rows and not national_output:
        print("  No validation data to write")
        return

    diag_dir = Path(RUNS_DIR) / run_id / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Write regional validation CSV
    if validation_rows:
        csv_columns = [
            "area_type",
            "area_id",
            "district",
            "variable",
            "target_name",
            "period",
            "target_value",
            "sim_value",
            "error",
            "rel_error",
            "abs_error",
            "rel_abs_error",
            "sanity_check",
            "sanity_reason",
            "in_training",
        ]
        csv_path = diag_dir / "validation_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for row in validation_rows:
                writer.writerow({k: row.get(k, "") for k in csv_columns})
        print(f"  Wrote {len(validation_rows)} rows to {csv_path}")

        # Compute summary
        n_sanity_fail = sum(
            1 for r in validation_rows if r.get("sanity_check") == "FAIL"
        )
        rae_vals = [
            r["rel_abs_error"]
            for r in validation_rows
            if isinstance(r.get("rel_abs_error"), (int, float))
            and r["rel_abs_error"] != float("inf")
        ]
        mean_rae = sum(rae_vals) / len(rae_vals) if rae_vals else 0.0

        # Per-area summaries for worst areas
        area_stats = {}
        for r in validation_rows:
            key = f"{r.get('area_type', '')}:{r.get('area_id', '')}"
            if key not in area_stats:
                area_stats[key] = {"rae_vals": [], "fails": 0}
            if r.get("sanity_check") == "FAIL":
                area_stats[key]["fails"] += 1
            rae = r.get("rel_abs_error")
            if isinstance(rae, (int, float)) and rae != float("inf"):
                area_stats[key]["rae_vals"].append(rae)

        worst_areas = sorted(
            area_stats.items(),
            key=lambda x: (
                sum(x[1]["rae_vals"]) / len(x[1]["rae_vals"]) if x[1]["rae_vals"] else 0
            ),
            reverse=True,
        )[:5]

        validation_summary = {
            "total_targets": len(validation_rows),
            "sanity_failures": n_sanity_fail,
            "mean_rel_abs_error": round(mean_rae, 4),
            "worst_areas": [
                {
                    "area": k,
                    "mean_rae": round(
                        (
                            sum(v["rae_vals"]) / len(v["rae_vals"])
                            if v["rae_vals"]
                            else 0
                        ),
                        4,
                    ),
                    "sanity_fails": v["fails"],
                }
                for k, v in worst_areas
            ],
        }

        print(
            f"  Validation summary: "
            f"{len(validation_rows)} targets, "
            f"{n_sanity_fail} sanity failures, "
            f"mean RAE={mean_rae:.4f}"
        )

        # Record in meta.json
        meta.step_timings["validation"] = validation_summary
        write_run_meta(meta, vol)

    # Write national validation output
    if national_output:
        nat_path = diag_dir / "national_validation.txt"
        with open(nat_path, "w") as f:
            f.write(national_output)
        print(f"  Wrote national validation to {nat_path}")

    vol.commit()


# ── Orchestrator ─────────────────────────────────────────────────


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=86400,  # 24 hours (Modal max)
    volumes={
        PIPELINE_MOUNT: pipeline_volume,
        STAGING_MOUNT: staging_volume,
    },
    secrets=[hf_secret, gcp_secret],
    nonpreemptible=True,
)
def run_pipeline(
    branch: str = "main",
    gpu: str = "T4",
    epochs: int = 1000,
    national_gpu: str = "T4",
    national_epochs: int = 4000,
    num_workers: int = 8,
    n_clones: int = 430,
    skip_national: bool = False,
    resume_run_id: str = None,
    clear_checkpoints: bool = False,
    scope: str = "all",
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
        clear_checkpoints: Wipe ALL checkpoints before building
            (default False). Normally not needed — checkpoints are
            scoped by commit SHA, so stale ones from other commits
            are cleaned automatically. Use True only to force a
            full rebuild of the current commit.

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

    if not resume_run_id:
        existing = find_resumable_run(branch, sha, pipeline_volume)
        if existing:
            print(f"Auto-resuming existing run {existing}")
            resume_run_id = existing

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
    print(f"  Clones:  {n_clones}")
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

            build_datasets.remote(
                upload=True,
                branch=branch,
                sequential=False,
                clear_checkpoints=clear_checkpoints,
                skip_tests=True,
                skip_enhanced_cps=False,
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

            vol_path = "/pipeline/artifacts/calibration_package.pkl"
            target_cfg = "policyengine_us_data/calibration/target_config.yaml"

            # Spawn regional fit
            regional_func = PACKAGE_GPU_FUNCTIONS[gpu]
            print(f"  Spawning regional fit ({gpu}, {epochs} epochs)...")
            regional_handle = regional_func.spawn(
                branch=branch,
                epochs=epochs,
                volume_package_path=vol_path,
                target_config=target_cfg,
                beta=0.65,
                lambda_l0=1e-7,
                lambda_l2=1e-8,
                log_freq=500,
            )
            print(f"    → regional fit fc: {regional_handle.object_id}")

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
                    target_config=target_cfg,
                    beta=0.65,
                    lambda_l0=1e-4,
                    lambda_l2=1e-12,
                    log_freq=500,
                )
                print(f"    → national fit fc: {national_handle.object_id}")

            # Collect regional results
            print("  Waiting for regional fit...")
            regional_result = regional_handle.get()
            print("  Regional fit complete. Writing to volume...")

            # Write regional results to pipeline volume
            with pipeline_volume.batch_upload(force=True) as batch:
                batch.put_file(
                    BytesIO(regional_result["weights"]),
                    "artifacts/calibration_weights.npy",
                )
                if regional_result.get("config"):
                    batch.put_file(
                        BytesIO(regional_result["config"]),
                        "artifacts/unified_run_config.json",
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
                    batch.put_file(
                        BytesIO(national_result["weights"]),
                        "artifacts/national_calibration_weights.npy",
                    )
                    if national_result.get("config"):
                        batch.put_file(
                            BytesIO(national_result["config"]),
                            "artifacts/national_unified_run_config.json",
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

            # Spawn H5 builds (run on separate Modal containers)
            if scope != "all":
                # Queue-based: one container per item, filtered by scope
                print(f"  Spawning queue-based H5 build (scope={scope})...")
                regional_h5_handle = queue_coordinator.spawn(
                    scope=scope,
                    branch=branch,
                    n_clones=n_clones,
                    validate=True,
                    run_id=run_id,
                )
                print(f"    → queue_coordinator fc: {regional_h5_handle.object_id}")
            else:
                # Legacy partition-based: N workers with chunked items
                print(f"  Spawning regional H5 build ({num_workers} workers)...")
                regional_h5_handle = coordinate_publish.spawn(
                    branch=branch,
                    num_workers=num_workers,
                    skip_upload=False,
                    n_clones=n_clones,
                    validate=True,
                    run_id=run_id,
                )
                print(f"    → coordinate_publish fc: {regional_h5_handle.object_id}")

            national_h5_handle = None
            if not skip_national:
                print("  Spawning national H5 build...")
                national_h5_handle = coordinate_national_publish.spawn(
                    branch=branch,
                    n_clones=n_clones,
                    validate=True,
                    run_id=run_id,
                )
                print(
                    f"    → coordinate_national_publish fc: {national_h5_handle.object_id}"
                )

            # While H5 builds run, stage base datasets
            # and upload diagnostics in this container
            pipeline_volume.reload()

            print("  Staging base datasets to HF...")
            stage_base_datasets(run_id, version, branch)

            print("  Uploading run diagnostics...")
            upload_run_diagnostics(run_id, branch)

            # Now wait for H5 builds to finish
            print("  Waiting for regional H5 build...")
            regional_h5_result = regional_h5_handle.get()
            regional_msg = (
                regional_h5_result.get("message", regional_h5_result)
                if isinstance(regional_h5_result, dict)
                else regional_h5_result
            )
            print(f"  Regional H5: {regional_msg}")

            national_h5_result = None
            if national_h5_handle is not None:
                print("  Waiting for national H5 build...")
                national_h5_result = national_h5_handle.get()
                national_msg = (
                    national_h5_result.get("message", national_h5_result)
                    if isinstance(national_h5_result, dict)
                    else national_h5_result
                )
                print(f"  National H5: {national_msg}")

            # ── Aggregate validation results ──
            _write_validation_diagnostics(
                run_id=run_id,
                regional_result=regional_h5_result,
                national_result=national_h5_result,
                meta=meta,
                vol=pipeline_volume,
            )

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
    nonpreemptible=True,
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

    # Clone repo for subprocess calls
    _setup_repo()

    # Promote base datasets from staging → production
    print("\nPromoting base datasets (staging → production)...")
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                f"""
from policyengine_us_data.utils.data_upload import (
    promote_staging_to_production_hf,
)

base_files = [
    "calibration/source_imputed_stratified_extended_cps.h5",
    "calibration/policy_data.db",
]
count = promote_staging_to_production_hf(base_files, "{version}", run_id="{run_id}")
print(f"Promoted {{count}} base dataset(s)")
""",
            ],
            cwd="/root/policyengine-us-data",
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        print(f"  {result.stdout.strip()}")
    except Exception as e:
        print(f"  WARNING: Base dataset promotion: {e}")

    # Promote H5s via existing functions
    print("\nPromoting regional H5s...")
    try:
        regional_result = promote_publish.remote(
            branch=meta.branch,
            version=version,
            run_id=run_id,
        )
        print(f"  {regional_result}")
    except Exception as e:
        print(f"  WARNING: Regional promote: {e}")

    print("\nPromoting national H5...")
    try:
        national_result = promote_national_publish.remote(
            branch=meta.branch,
            run_id=run_id,
        )
        print(f"  {national_result}")
    except Exception as e:
        print(f"  WARNING: National promote: {e}")

    # Register version in manifest
    print("\nRegistering version in manifest...")
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                f"""
from policyengine_us_data.utils.version_manifest import (
    build_manifest,
    upload_manifest,
)

blob_names = [
    "calibration/source_imputed_stratified_extended_cps.h5",
    "calibration/policy_data.db",
    "calibration/calibration_weights.npy",
]
manifest = build_manifest(
    version="{version}",
    blob_names=blob_names,
)
manifest.pipeline_run_id = "{run_id}"
manifest.diagnostics_path = "calibration/runs/{run_id}/diagnostics/"
upload_manifest(manifest)
print("Registered version {version} in version_manifest.json")
""",
            ],
            cwd="/root/policyengine-us-data",
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        print(f"  {result.stdout.strip()}")
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
    gpu: str = "T4",
    epochs: int = 1000,
    national_gpu: str = "T4",
    national_epochs: int = 4000,
    num_workers: int = 8,
    n_clones: int = 430,
    skip_national: bool = False,
    clear_checkpoints: bool = False,
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
            clear_checkpoints=clear_checkpoints,
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
