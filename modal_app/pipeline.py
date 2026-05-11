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
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image as image  # noqa: E402
from modal_app.resilience import ensure_resume_sha_compatible  # noqa: E402
from modal_app.step_manifests.specs import (  # noqa: E402
    BUILD_CALIBRATION_PACKAGE,
    BUILD_DATASETS,
    BUILD_OUTPUTS,
    LOCAL_AREA_H5_NATIONAL,
    LOCAL_AREA_H5_REGIONAL,
    STAGE_BASE_DATASETS,
    UPLOAD_DIAGNOSTICS,
    VALIDATE_AND_PROMOTE_RELEASE,
    WEIGHT_FITTING_NATIONAL,
    WEIGHT_FITTING_REGIONAL,
)
from modal_app.step_manifests.state import (  # noqa: E402
    PIPELINE_MOUNT,
    RUNS_DIR,
    RunMetadata,
    STAGING_MOUNT,
    apply_run_context_env as _apply_run_context_env,
    artifact_identities as _artifact_identities,
    artifacts_dir as _artifacts_dir,
    artifacts_dir_for_run,
    collect_diagnostics as _collect_diagnostics,
    collect_staging_outputs as _collect_staging_outputs,
    metadata_run_fields as _metadata_run_fields,
    run_dir as _run_dir,
)
from modal_app.step_manifests.store import (  # noqa: E402
    complete_step_manifest as _complete_step_manifest,
    fail_step_manifest as _fail_step_manifest,
    mark_step_reused as _mark_step_reused,
    read_run_meta,
    start_step_manifest as _start_step_manifest,
    step_reusable as _step_reusable,
    write_run_meta,
)
from policyengine_us_data.utils.run_context import RunContext, resolve_run_id  # noqa: E402
from policyengine_us_data.utils.step_manifest import (  # noqa: E402
    ArtifactReference,
    ReuseMeasurement,
    StepManifest,
    collect_artifacts,
    collect_directory_artifacts,
    completed_validated_outputs,
    read_step_manifest,
    run_manifest_path,
)
from policyengine_us_data.pipeline_metadata import pipeline_node  # noqa: E402
from policyengine_us_data.pipeline_schema import PipelineNode  # noqa: E402

# ── Modal resources ──────────────────────────────────────────────

app = modal.App(
    os.environ.get("US_DATA_PIPELINE_APP_NAME")
    or os.environ.get("US_DATA_MODAL_APP_NAME")
    or "policyengine-us-data-pipeline"
)

NATIONAL_FIT_LAMBDA_L0 = 1e-4

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

pipeline_volume = modal.Volume.from_name(
    os.environ.get("US_DATA_PIPELINE_VOLUME_NAME", "pipeline-artifacts"),
    create_if_missing=True,
    version=2,
)
staging_volume = modal.Volume.from_name(
    os.environ.get("US_DATA_STAGING_VOLUME_NAME", "local-area-staging"),
    create_if_missing=True,
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"


def _python_cmd(*args: str) -> list[str]:
    """Build a command that uses the current interpreter."""
    return [sys.executable, *args]


def _calibration_package_parameters(
    *,
    workers: int,
    n_clones: int,
    target_config: str | None,
    skip_county: bool,
    chunked_matrix: bool,
    chunk_size: int,
    parallel_matrix: bool,
    num_matrix_workers: int,
) -> dict:
    """Return manifest parameters that affect package construction."""
    effective_parallel = bool(chunked_matrix and parallel_matrix)
    params = {
        "workers": workers if not chunked_matrix else None,
        "n_clones": n_clones,
        "target_config": target_config,
        "skip_county": skip_county,
        "chunked_matrix": bool(chunked_matrix),
        "chunk_size": chunk_size if chunked_matrix else None,
        "parallel_matrix": effective_parallel,
        "num_matrix_workers": num_matrix_workers if effective_parallel else None,
    }
    return {key: value for key, value in params.items() if value is not None}


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


# ── Include other Modal apps ─────────────────────────────────────
# app.include() merges functions from other apps into this one,
# ensuring Modal mounts their files and registers their functions
# (with their GPU/memory/volume configs) in the ephemeral run.
# sys.path setup is handled at the top of this file.

from modal_app.data_build import app as _data_build_app  # noqa: E402
from modal_app.data_build import build_datasets  # noqa: E402

app.include(_data_build_app)

from modal_app.remote_calibration_runner import app as _calibration_app  # noqa: E402
from modal_app.remote_calibration_runner import (  # noqa: E402
    build_package_remote,
    PACKAGE_GPU_FUNCTIONS,
)

# Import registers ``build_matrix_chunk_worker`` on ``_calibration_app``
# so a single ``modal deploy modal_app/pipeline.py`` also deploys the
# worker via ``app.include(_calibration_app)`` below. Without this the
# dispatch layer's ``modal.Function.from_name`` lookup would fail at
# runtime.
from modal_app.matrix_chunk_worker import build_matrix_chunk_worker  # noqa: E402, F401

app.include(_calibration_app)

from modal_app.local_area import app as _local_area_app  # noqa: E402
from modal_app.local_area import (  # noqa: E402
    coordinate_publish,
    coordinate_national_publish,
    _build_publishing_input_bundle,
    _resolve_scope_fingerprint,
)

app.include(_local_area_app)


# ── Upload helpers ──────────────────────────────────────────────


def _setup_repo() -> None:
    """Change to the pre-baked repo directory."""
    os.chdir("/root/policyengine-us-data")


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
        _python_cmd(
            "-c",
            _build_diagnostics_upload_script(entries_json),
        ),
        cwd="/root/policyengine-us-data",
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Diagnostics upload failed: {result.stderr}")
    print(f"  {result.stdout.strip()}")


def _build_diagnostics_upload_script(entries_json: str) -> str:
    """Build the isolated diagnostics-upload script.

    Keep this snippet syntactically self-contained: it is passed directly to
    ``python -c`` inside the Modal orchestrator container.
    """
    return f"""
import json
import os
from huggingface_hub import HfApi

entries = json.loads({entries_json!r})
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
"""


def _run_required_promotion_subprocess(label: str, script: str) -> str:
    """Run a promotion subprocess and fail the release step on error."""
    result = subprocess.run(
        _python_cmd("-c", script),
        cwd="/root/policyengine-us-data",
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{label} failed: {detail}")
    return result.stdout.strip()


BASE_DATASET_STAGING_REL_PATHS = (
    "cps_2024.h5",
    "policy_data.db",
    "enhanced_cps_2024.h5",
    "small_enhanced_cps_2024.h5",
)


def _regional_h5_staging_rel_paths(run_id: str) -> list[str]:
    """Read regional H5 staged paths from the Modal staging manifest."""
    manifest_path = Path(STAGING_MOUNT) / run_id / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    return list(manifest.get("files", {}).keys())


def _full_release_staging_rel_paths(run_id: str) -> list[str]:
    """Return the full set of staged production artifact paths for a run."""
    return sorted(
        {
            *BASE_DATASET_STAGING_REL_PATHS,
            *_regional_h5_staging_rel_paths(run_id),
            "national/US.h5",
        }
    )


def _full_release_manifest_files(
    run_id: str, rel_paths: list[str]
) -> list[tuple[str, str]]:
    """Map staged release repo paths to local files for manifest checksums."""
    base_dir = _artifacts_dir(run_id)
    h5_dir = Path(STAGING_MOUNT) / run_id
    files = []
    for rel_path in rel_paths:
        if rel_path in BASE_DATASET_STAGING_REL_PATHS:
            local_path = base_dir / rel_path
        else:
            local_path = h5_dir / rel_path
        files.append((str(local_path), rel_path))
    return files


def _promote_full_release_from_staging(
    run_id: str,
    version: str,
    run_context: dict | None = None,
) -> str:
    """Promote all staged artifacts as one finalized release."""
    rel_paths = _full_release_staging_rel_paths(run_id)
    rel_paths_json = json.dumps(rel_paths)
    files_json = json.dumps(_full_release_manifest_files(run_id, rel_paths))
    run_context_json = json.dumps(run_context or {})
    return _run_required_promotion_subprocess(
        "Full release promotion",
        f"""
import json
from policyengine_us_data.utils.data_upload import promote_full_release_from_staging

rel_paths = json.loads({rel_paths_json!r})
files_with_paths = json.loads({files_json!r})
run_context = json.loads({run_context_json!r})
result = promote_full_release_from_staging(
    rel_paths=rel_paths,
    version="{version}",
    run_id="{run_id}",
    run_context=run_context,
    files_with_paths=files_with_paths,
    extra_cleanup_paths=["_run_context.json"],
)
print(json.dumps(result, indent=2, sort_keys=True))
""",
    )


@app.function(
    image=image,
    timeout=300,
)
@pipeline_node(
    PipelineNode(
        id="verify_runtime_seams",
        label="Verify Modal Runtime Seams",
        node_type="validation",
        description="Check import, subprocess, baked-file, and Modal function seams before heavy pipeline execution.",
        source_file="modal_app/pipeline.py",
        status="current",
        stability="moving",
        pathways=["orchestration"],
        validation_commands=["uv run pytest tests/unit/test_pipeline.py"],
    )
)
def verify_runtime_seams() -> dict:
    """Verify deployed-image imports and subprocess seams."""
    import importlib

    repo_root = "/root/policyengine-us-data"
    expected_files = (
        "pyproject.toml",
        "uv.lock",
        "modal_app/worker_script.py",
        "modal_app/local_area.py",
        "modal_app/h5_test_harness.py",
        "modal_app/step_manifests/specs.py",
        "modal_app/step_manifests/state.py",
        "modal_app/step_manifests/store.py",
        "modal_app/fixtures/h5_cases.py",
        "tests/integration/test_fixture_50hh.h5",
        "policyengine_us_data/calibration/target_config.yaml",
        "policyengine_us_data/calibration/target_config_full.yaml",
        "policyengine_us_data/utils/run_context.py",
        "policyengine_us_data/utils/step_manifest.py",
    )
    result = {
        "interpreter": {
            "parent": sys.executable,
        },
        "imports": {},
        "subprocess": {},
        "paths": {
            "cwd": os.getcwd(),
            "repo_root_exists": os.path.isdir(repo_root),
            "working_directory_is_repo_root": os.getcwd() == repo_root,
            "target_config_exists": os.path.exists(
                f"{repo_root}/policyengine_us_data/calibration/target_config.yaml"
            ),
            "expected_files": {
                rel_path: os.path.exists(f"{repo_root}/{rel_path}")
                for rel_path in expected_files
            },
        },
    }
    result["paths"]["all_expected_files_exist"] = all(
        result["paths"]["expected_files"].values()
    )

    for module_name in (
        "google.cloud.storage",
        "h5py",
        "huggingface_hub",
        "modal_app.fixtures.h5_cases",
        "modal_app.h5_test_harness",
        "modal_app.local_area",
        "modal_app.remote_calibration_runner",
        "modal_app.step_manifests.specs",
        "modal_app.step_manifests.state",
        "modal_app.step_manifests.store",
        "modal_app.worker_script",
        "numpy",
        "pandas",
        "policyengine_us",
        "policyengine_us_data",
        "policyengine_us_data.utils.run_context",
        "policyengine_us_data.utils.step_manifest",
        "spm_calculator",
        "sqlalchemy",
    ):
        try:
            imported = importlib.import_module(module_name)
            result["imports"][module_name] = {
                "ok": True,
                "version": getattr(imported, "__version__", None),
            }
        except Exception as exc:
            result["imports"][module_name] = {
                "ok": False,
                "error": repr(exc),
            }

    child_python = subprocess.run(
        _python_cmd(
            "-c",
            (
                "import json, os, sys; "
                "print(json.dumps({'executable': sys.executable, 'cwd': os.getcwd()}))"
            ),
        ),
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    child_runtime = json.loads(child_python.stdout)
    child_exec = child_runtime["executable"]
    result["interpreter"]["child"] = child_exec
    result["interpreter"]["child_cwd"] = child_runtime["cwd"]
    result["interpreter"]["child_matches_parent"] = child_exec == sys.executable
    result["interpreter"]["child_cwd_is_repo_root"] = child_runtime["cwd"] == repo_root

    for name, cmd in {
        "worker_import": _python_cmd("-c", "import modal_app.worker_script"),
        "worker_help": _python_cmd("-m", "modal_app.worker_script", "--help"),
        "local_area_import": _python_cmd("-c", "import modal_app.local_area"),
        "calibration_help": _python_cmd(
            "-m",
            "policyengine_us_data.calibration.unified_calibration",
            "--help",
        ),
    }.items():
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        result["subprocess"][name] = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-500:],
            "stderr_tail": proc.stderr[-500:],
        }

    import modal_app.remote_calibration_runner as calibration_runner

    runner_source = Path(calibration_runner.__file__).read_text()
    result["calibration_optimizer_checkpoint_policy"] = {
        "runner_exposes_checkpoint_name": "checkpoint_name" in runner_source,
        "runner_passes_checkpoint_output": "--checkpoint-output" in runner_source,
        "runner_collects_checkpoint_path": "CHECKPOINT_PATH:" in runner_source,
    }

    return result


@app.function(
    image=image,
    timeout=28800,
)
def run_seeded_h5_publish_seam(
    *,
    branch: str,
    run_id: str,
    n_clones: int,
    regional_work_items: list[dict],
) -> dict:
    """Run the pipeline-owned H5 publish seam against pre-seeded tiny artifacts."""

    regional_handle = coordinate_publish.spawn(
        branch=branch,
        num_workers=1,
        skip_upload=True,
        n_clones=n_clones,
        validate=False,
        run_id=run_id,
        work_items_override=regional_work_items,
    )
    national_handle = coordinate_national_publish.spawn(
        branch=branch,
        n_clones=n_clones,
        validate=False,
        run_id=run_id,
        skip_upload=True,
    )
    return {
        "regional": regional_handle.get(),
        "national": national_handle.get(),
    }


def _write_validation_diagnostics(
    run_id: str,
    regional_result,
    national_result,
    vol: modal.Volume,
) -> None:
    """Aggregate validation rows into a diagnostics CSV.

    Extracts validation_rows from coordinate_publish and
    national_validation from coordinate_national_publish,
    writes them to runs/{run_id}/diagnostics/validation_results.csv,
    and records a summary in diagnostics/validation_summary.json.
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
            "display_name",
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

        summary_path = diag_dir / "validation_summary.json"
        summary_path.write_text(json.dumps(validation_summary, indent=2) + "\n")
        print(f"  Wrote validation summary to {summary_path}")

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
@pipeline_node(
    PipelineNode(
        id="run_modal_pipeline",
        label="Run Modal Pipeline",
        node_type="entrypoint",
        description="Coordinate data build, calibration package, weight fit, local H5 publishing, validation, and promotion.",
        details="This is the current production orchestration surface. It remains documented as a bundled pathway while lower-level seams are migrated.",
        source_file="modal_app/pipeline.py",
        status="current",
        stability="moving",
        pathways=["orchestration"],
        artifacts_out=["run metadata", "diagnostics", "published H5 artifacts"],
        validation_commands=[
            "uv run pytest tests/unit/test_pipeline.py",
            "uv run pytest tests/integration/test_modal_pipeline_seams.py",
        ],
    )
)
def run_pipeline(
    branch: str = "main",
    gpu: str = "T4",
    epochs: int = 1000,
    national_gpu: str = "T4",
    national_epochs: int = 1000,
    num_workers: int = 50,
    n_clones: int = 430,
    skip_national: bool = False,
    resume_run_id: str = None,
    clear_checkpoints: bool = False,
    version_override: str = "",
    sha_override: str = "",
    run_id: str = "",
    run_context: dict | None = None,
    modal_app_name: str = "",
    modal_environment: str = "",
    chunked_matrix: bool = False,
    chunk_size: int = 25_000,
    parallel_matrix: bool = False,
    num_matrix_workers: int = 50,
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
        sha_override: Exact source SHA deployed by GitHub Actions. When
            provided, this is recorded instead of reading the current
            branch tip.
        run_id: Cross-system run ID created by GitHub.
        run_context: Serialized run context from the launcher workflow.
        modal_app_name: Deployed Modal app name for this run.
        modal_environment: Modal environment used for this run.
        chunked_matrix: Build the calibration matrix in clone-household
            chunks instead of the non-chunked path. Opt-in; default off.
        chunk_size: Clone-household columns per chunk when
            ``chunked_matrix`` is True.
        parallel_matrix: Fan chunked matrix building across Modal
            workers via ``build_matrix_chunk_worker``. Only meaningful
            when ``chunked_matrix`` is True; ignored otherwise.
        num_matrix_workers: Number of Modal workers when
            ``parallel_matrix`` is True.

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
    sha = sha_override or get_pinned_sha(branch)
    version = version_override or get_version_from_branch(branch)
    resolved_run_id = resolve_run_id(run_id)
    current_run_context = RunContext.from_mapping(
        run_context,
        run_id=resolved_run_id,
        modal_app_name=modal_app_name,
        modal_environment=modal_environment,
    )

    explicit_resume = bool(resume_run_id)

    if resume_run_id:
        print(f"Resuming run {resume_run_id}...")
        meta = read_run_meta(resume_run_id, pipeline_volume)
        current_run_context = RunContext.from_mapping(
            meta.run_context,
            run_id=meta.run_id,
            modal_app_name=meta.modal_app_name or current_run_context.modal_app_name,
            modal_environment=meta.modal_environment
            or current_run_context.modal_environment,
        )
        _apply_run_context_env(current_run_context)
        current_sha = sha
        sha_match = ensure_resume_sha_compatible(
            branch=branch,
            run_sha=meta.sha,
            current_sha=current_sha,
            force=explicit_resume,
        )
        sha = meta.sha
        version = meta.version
        if not hasattr(meta, "resume_history") or meta.resume_history is None:
            meta.resume_history = []
        meta.resume_history.append(
            {
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "code_sha": current_sha,
                "original_sha": meta.sha,
                "branch": branch,
                "mixed_provenance": not sha_match,
            }
        )
        meta.status = "running"
        if not meta.run_context:
            meta.run_context = current_run_context.to_dict()
        meta.modal_app_name = meta.modal_app_name or current_run_context.modal_app_name
        meta.modal_environment = (
            meta.modal_environment or current_run_context.modal_environment
        )
        meta.hf_staging_prefix = (
            meta.hf_staging_prefix or current_run_context.hf_staging_prefix
        )
        run_id = resume_run_id
    else:
        if not current_run_context.run_id:
            raise RuntimeError(
                "run_id is required. Production pipeline runs must receive the "
                "GitHub-created run ID through workflow_dispatch."
            )
        _apply_run_context_env(current_run_context)
        run_id = current_run_context.run_id
        meta = RunMetadata(
            run_id=run_id,
            branch=branch,
            sha=sha,
            version=version,
            start_time=datetime.now(timezone.utc).isoformat(),
            status="running",
            **_metadata_run_fields(current_run_context),
        )

    # Create run directory
    run_dir = Path(RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics").mkdir(exist_ok=True)

    # Create run-scoped artifacts directory
    Path(artifacts_dir_for_run(run_id)).mkdir(parents=True, exist_ok=True)

    write_run_meta(meta, pipeline_volume)

    print("=" * 60)
    print("PIPELINE RUN")
    print("=" * 60)
    print(f"  Run ID:  {run_id}")
    if meta.modal_app_name:
        print(f"  Modal app: {meta.modal_app_name}")
    if meta.hf_staging_prefix:
        print(f"  HF staging: {meta.hf_staging_prefix}")
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
        completed = _completed_step_manifest_ids(run_id)
        print(f"  Resume:  skipping {completed}")
    print("=" * 60)

    active_step_manifest: StepManifest | None = None

    try:
        # ── Step 1: Build datasets ──
        build_dataset_inputs = {"source": {"branch": branch, "sha": sha}}
        build_dataset_parameters = {
            "upload": True,
            "stage_only": True,
            "sequential": False,
            "clear_checkpoints": clear_checkpoints,
            "skip_tests": False,
            "skip_enhanced_cps": False,
            "run_id": run_id,
        }
        build_dataset_reuse = _step_reusable(
            meta,
            BUILD_DATASETS,
            expected_input_identities=build_dataset_inputs,
            expected_parameters=build_dataset_parameters,
        )
        if build_dataset_reuse.reusable:
            _mark_step_reused(
                meta,
                BUILD_DATASETS,
                build_dataset_reuse,
                vol=pipeline_volume,
            )
            print(f"\n[Step 1/5] {BUILD_DATASETS.title} (skipped - manifest valid)")
        else:
            print(f"\n[Step 1/5] {BUILD_DATASETS.title}...")
            active_step_manifest = _start_step_manifest(
                meta,
                BUILD_DATASETS,
                parameters=build_dataset_parameters,
                input_identities=build_dataset_inputs,
                vol=pipeline_volume,
            )

            build_datasets.remote(
                upload=True,
                branch=branch,
                sequential=False,
                clear_checkpoints=clear_checkpoints,
                skip_tests=False,
                skip_enhanced_cps=False,
                stage_only=True,
                run_id=run_id,
            )

            # Stage 1 uses the existing dataset upload machinery to validate
            # and write canonical dataset paths under staging/{run_id}/.
            # It also copies artifacts to the pipeline volume for downstream
            # calibration, H5 building, and manifest traceability.
            dataset_outputs = collect_directory_artifacts(
                _artifacts_dir(run_id),
                role="output",
            )
            build_manifest = active_step_manifest
            stage_base_manifest = _start_step_manifest(
                meta,
                STAGE_BASE_DATASETS,
                parameters={
                    "version": version,
                    "run_id": run_id,
                    "stage_only": True,
                },
                input_identities={
                    "dataset_outputs": [
                        artifact.to_dict() for artifact in dataset_outputs
                    ],
                },
                vol=pipeline_volume,
            )
            active_step_manifest = stage_base_manifest
            _complete_step_manifest(
                stage_base_manifest,
                outputs=dataset_outputs,
                reuse_decision="computed",
                reuse_measurement=ReuseMeasurement(
                    expected_outputs=len(dataset_outputs),
                    recomputed_outputs=len(dataset_outputs),
                ),
                vol=pipeline_volume,
            )
            active_step_manifest = build_manifest
            checkpoint_stats_path = (
                _artifacts_dir(run_id) / "data_build_checkpoint_stats.json"
            )
            checkpoint_stats = (
                json.loads(checkpoint_stats_path.read_text())
                if checkpoint_stats_path.exists()
                else {}
            )
            completed_build_manifest = _complete_step_manifest(
                active_step_manifest,
                outputs=dataset_outputs,
                reuse_measurement=ReuseMeasurement(
                    expected_outputs=checkpoint_stats.get(
                        "expected_outputs", len(dataset_outputs)
                    ),
                    valid_reused_outputs=checkpoint_stats.get(
                        "valid_reused_outputs", 0
                    ),
                    recomputed_outputs=checkpoint_stats.get(
                        "recomputed_outputs", len(dataset_outputs)
                    ),
                    invalid_outputs=checkpoint_stats.get("invalid_outputs", 0),
                ),
                vol=pipeline_volume,
            )
            active_step_manifest = None
            print(f"  Completed in {completed_build_manifest.duration_s}s")

        # ── Step 2: Build calibration package ──
        package_inputs = _artifact_identities(
            {
                "dataset": _artifacts_dir(run_id)
                / "source_imputed_stratified_extended_cps.h5",
                "database": _artifacts_dir(run_id) / "policy_data.db",
            }
        )
        package_parameters = _calibration_package_parameters(
            workers=num_workers,
            n_clones=n_clones,
            target_config=None,
            skip_county=True,
            chunked_matrix=chunked_matrix,
            chunk_size=chunk_size,
            parallel_matrix=parallel_matrix,
            num_matrix_workers=num_matrix_workers,
        )
        package_reuse = _step_reusable(
            meta,
            BUILD_CALIBRATION_PACKAGE,
            expected_input_identities=package_inputs,
            expected_parameters=package_parameters,
        )
        if not package_reuse.reusable:
            previous = package_reuse.manifest
            print(f"  Package reuse invalidated: {package_reuse.reason}")
            if previous is not None:
                print(f"    prior status: {previous.status}")
                print(f"    prior parameters: {previous.parameters}")
                print(f"    expected parameters: {package_parameters}")
                print(f"    prior inputs: {previous.input_identities}")
                print(f"    expected inputs: {package_inputs}")
        if package_reuse.reusable:
            _mark_step_reused(
                meta,
                BUILD_CALIBRATION_PACKAGE,
                package_reuse,
                vol=pipeline_volume,
            )
            print(
                f"\n[Step 2/5] {BUILD_CALIBRATION_PACKAGE.title} "
                "(skipped - manifest valid)"
            )
        else:
            print(f"\n[Step 2/5] {BUILD_CALIBRATION_PACKAGE.title}...")
            active_step_manifest = _start_step_manifest(
                meta,
                BUILD_CALIBRATION_PACKAGE,
                parameters=package_parameters,
                input_identities=package_inputs,
                vol=pipeline_volume,
            )

            pkg_path = build_package_remote.remote(
                branch=branch,
                workers=num_workers,
                n_clones=n_clones,
                run_id=run_id,
                modal_app_name=current_run_context.modal_app_name,
                modal_environment=current_run_context.modal_environment,
                pipeline_volume_name=current_run_context.pipeline_volume_name,
                chunked_matrix=chunked_matrix,
                chunk_size=chunk_size,
                parallel_matrix=parallel_matrix,
                num_matrix_workers=num_matrix_workers,
            )
            print(f"  Package at: {pkg_path}")

            completed_package_manifest = _complete_step_manifest(
                active_step_manifest,
                outputs=collect_artifacts(
                    [_artifacts_dir(run_id) / "calibration_package.pkl"],
                    missing_ok=True,
                ),
                vol=pipeline_volume,
            )
            active_step_manifest = None
            print(f"  Completed in {completed_package_manifest.duration_s}s")

        # ── Step 3: Fit weights (parallel) ──
        fit_inputs = _artifact_identities(
            {
                "calibration_package": _artifacts_dir(run_id)
                / "calibration_package.pkl",
            }
        )
        regional_fit_parameters = {
            "gpu": gpu,
            "epochs": epochs,
            "target_config": "policyengine_us_data/calibration/target_config.yaml",
            "beta": 0.65,
            "lambda_l0": 1e-7,
            "lambda_l2": 1e-8,
            "log_freq": 100,
        }
        national_fit_parameters = {
            "gpu": national_gpu,
            "epochs": national_epochs,
            "target_config": "policyengine_us_data/calibration/target_config.yaml",
            "beta": 0.65,
            "lambda_l0": NATIONAL_FIT_LAMBDA_L0,
            "lambda_l2": 1e-12,
            "log_freq": 100,
            "skip_national": skip_national,
        }
        regional_fit_reuse = _step_reusable(
            meta,
            WEIGHT_FITTING_REGIONAL,
            expected_input_identities=fit_inputs,
            expected_parameters=regional_fit_parameters,
        )
        national_fit_reuse = (
            _step_reusable(
                meta,
                WEIGHT_FITTING_NATIONAL,
                expected_input_identities=fit_inputs,
                expected_parameters=national_fit_parameters,
            )
            if not skip_national
            else None
        )
        fit_reusable = regional_fit_reuse.reusable and (
            skip_national or national_fit_reuse.reusable
        )
        if fit_reusable:
            _mark_step_reused(
                meta,
                WEIGHT_FITTING_REGIONAL,
                regional_fit_reuse,
                vol=pipeline_volume,
            )
            if national_fit_reuse is not None:
                _mark_step_reused(
                    meta,
                    WEIGHT_FITTING_NATIONAL,
                    national_fit_reuse,
                    vol=pipeline_volume,
                )
            print("\n[Step 3/5] Fit weights (skipped - manifests valid)")
        else:
            print("\n[Step 3/5] Fitting calibration weights...")
            step_start = time.time()

            vol_path = f"{artifacts_dir_for_run(run_id)}/calibration_package.pkl"
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
                log_freq=100,
            )
            print(f"    → regional fit fc: {regional_handle.object_id}")
            regional_fit_manifest = _start_step_manifest(
                meta,
                WEIGHT_FITTING_REGIONAL,
                scope="regional",
                parameters=regional_fit_parameters,
                input_identities=fit_inputs,
                modal_call_id=regional_handle.object_id,
                vol=pipeline_volume,
            )
            active_step_manifest = regional_fit_manifest

            # Spawn national fit (if enabled)
            national_handle = None
            national_fit_manifest = None
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
                    lambda_l0=NATIONAL_FIT_LAMBDA_L0,
                    lambda_l2=1e-12,
                    log_freq=100,
                )
                print(f"    → national fit fc: {national_handle.object_id}")
                national_fit_manifest = _start_step_manifest(
                    meta,
                    WEIGHT_FITTING_NATIONAL,
                    scope="national",
                    parameters=national_fit_parameters,
                    input_identities=fit_inputs,
                    modal_call_id=national_handle.object_id,
                    vol=pipeline_volume,
                )

            # Collect regional results
            print("  Waiting for regional fit...")
            regional_result = regional_handle.get()
            print("  Regional fit complete. Writing to volume...")

            # Write regional results to pipeline volume (run-scoped)
            artifacts_rel = f"artifacts/{run_id}" if run_id else "artifacts"
            with pipeline_volume.batch_upload(force=True) as batch:
                batch.put_file(
                    BytesIO(regional_result["weights"]),
                    f"{artifacts_rel}/calibration_weights.npy",
                )
                if regional_result.get("geography"):
                    batch.put_file(
                        BytesIO(regional_result["geography"]),
                        f"{artifacts_rel}/geography_assignment.npz",
                    )
                if regional_result.get("config"):
                    batch.put_file(
                        BytesIO(regional_result["config"]),
                        f"{artifacts_rel}/unified_run_config.json",
                    )

            archive_diagnostics(
                run_id,
                regional_result,
                pipeline_volume,
                prefix="",
            )
            regional_outputs = collect_artifacts(
                [
                    _artifacts_dir(run_id) / "calibration_weights.npy",
                    _artifacts_dir(run_id) / "geography_assignment.npz",
                    _artifacts_dir(run_id) / "unified_run_config.json",
                ],
                missing_ok=True,
            )
            regional_fit_reuse_measurement = ReuseMeasurement(
                expected_outputs=len(regional_outputs),
                recomputed_outputs=len(regional_outputs),
            )
            _complete_step_manifest(
                regional_fit_manifest,
                outputs=regional_outputs,
                diagnostics=_collect_diagnostics(run_id),
                reuse_decision="computed",
                reuse_measurement=regional_fit_reuse_measurement,
                vol=pipeline_volume,
            )
            active_step_manifest = national_fit_manifest

            # Collect national results
            if national_handle is not None:
                print("  Waiting for national fit...")
                national_result = national_handle.get()
                print("  National fit complete. Writing to volume...")

                with pipeline_volume.batch_upload(force=True) as batch:
                    batch.put_file(
                        BytesIO(national_result["weights"]),
                        f"{artifacts_rel}/national_calibration_weights.npy",
                    )
                    if national_result.get("geography"):
                        batch.put_file(
                            BytesIO(national_result["geography"]),
                            f"{artifacts_rel}/national_geography_assignment.npz",
                        )
                    if national_result.get("config"):
                        batch.put_file(
                            BytesIO(national_result["config"]),
                            f"{artifacts_rel}/national_unified_run_config.json",
                        )

                archive_diagnostics(
                    run_id,
                    national_result,
                    pipeline_volume,
                    prefix="national_",
                )
                national_outputs = collect_artifacts(
                    [
                        _artifacts_dir(run_id) / "national_calibration_weights.npy",
                        _artifacts_dir(run_id) / "national_geography_assignment.npz",
                        _artifacts_dir(run_id) / "national_unified_run_config.json",
                    ],
                    missing_ok=True,
                )
                _complete_step_manifest(
                    national_fit_manifest,
                    outputs=national_outputs,
                    diagnostics=_collect_diagnostics(run_id),
                    reuse_decision="computed",
                    reuse_measurement=ReuseMeasurement(
                        expected_outputs=len(national_outputs),
                        recomputed_outputs=len(national_outputs),
                    ),
                    vol=pipeline_volume,
                )
                active_step_manifest = None

            active_step_manifest = None
            print(f"  Completed in {round(time.time() - step_start, 1)}s")

        # ── Step 4: Build H5s + diagnostics (parallel) ──
        #   4a. coordinate_publish (regional H5s)
        #   4b. coordinate_national_publish (national H5)
        #   4c. upload_run_diagnostics (calibration diagnostics → HF)
        #   4d. _write_validation_diagnostics (after H5 builds)
        #   4e. upload_run_diagnostics (validation diagnostics → HF)
        regional_h5_inputs = _artifact_identities(
            {
                "weights": _artifacts_dir(run_id) / "calibration_weights.npy",
                "geography": _artifacts_dir(run_id) / "geography_assignment.npz",
                "dataset": _artifacts_dir(run_id)
                / "source_imputed_stratified_extended_cps.h5",
                "database": _artifacts_dir(run_id) / "policy_data.db",
                "run_config": _artifacts_dir(run_id) / "unified_run_config.json",
                "calibration_package": _artifacts_dir(run_id)
                / "calibration_package.pkl",
            }
        )
        regional_h5_parameters = {
            "num_workers": num_workers,
            "n_clones": n_clones,
            "validate": True,
            "skip_upload": False,
        }
        national_h5_inputs = _artifact_identities(
            {
                "weights": _artifacts_dir(run_id) / "national_calibration_weights.npy",
                "geography": _artifacts_dir(run_id)
                / "national_geography_assignment.npz",
                "dataset": _artifacts_dir(run_id)
                / "source_imputed_stratified_extended_cps.h5",
                "database": _artifacts_dir(run_id) / "policy_data.db",
                "run_config": _artifacts_dir(run_id)
                / "national_unified_run_config.json",
            }
        )
        national_h5_parameters = {
            "n_clones": n_clones,
            "validate": True,
            "skip_upload": False,
            "skip_national": skip_national,
        }
        regional_fingerprint_inputs = _build_publishing_input_bundle(
            weights_path=_artifacts_dir(run_id) / "calibration_weights.npy",
            dataset_path=_artifacts_dir(run_id)
            / "source_imputed_stratified_extended_cps.h5",
            db_path=_artifacts_dir(run_id) / "policy_data.db",
            geography_path=_artifacts_dir(run_id) / "geography_assignment.npz",
            calibration_package_path=_artifacts_dir(run_id) / "calibration_package.pkl",
            run_config_path=_artifacts_dir(run_id) / "unified_run_config.json",
            run_id=run_id,
            version=version,
            n_clones=n_clones,
            seed=42,
            legacy_blocks_path=_artifacts_dir(run_id) / "stacked_blocks.npy",
        )
        regional_scope_fingerprint = _resolve_scope_fingerprint(
            inputs=regional_fingerprint_inputs,
            scope="regional",
        )
        regional_h5_inputs["h5_scope_fingerprint"] = regional_scope_fingerprint

        national_scope_fingerprint = None
        if not skip_national:
            national_fingerprint_inputs = _build_publishing_input_bundle(
                weights_path=_artifacts_dir(run_id)
                / "national_calibration_weights.npy",
                dataset_path=_artifacts_dir(run_id)
                / "source_imputed_stratified_extended_cps.h5",
                db_path=_artifacts_dir(run_id) / "policy_data.db",
                geography_path=_artifacts_dir(run_id)
                / "national_geography_assignment.npz",
                calibration_package_path=None,
                run_config_path=_artifacts_dir(run_id)
                / "national_unified_run_config.json",
                run_id=run_id,
                version=version,
                n_clones=n_clones,
                seed=42,
            )
            national_scope_fingerprint = _resolve_scope_fingerprint(
                inputs=national_fingerprint_inputs,
                scope="national",
            )
            national_h5_inputs["h5_scope_fingerprint"] = national_scope_fingerprint

        regional_h5_reuse = _step_reusable(
            meta,
            LOCAL_AREA_H5_REGIONAL,
            expected_input_identities=regional_h5_inputs,
            expected_parameters=regional_h5_parameters,
        )
        national_h5_reuse = (
            _step_reusable(
                meta,
                LOCAL_AREA_H5_NATIONAL,
                expected_input_identities=national_h5_inputs,
                expected_parameters=national_h5_parameters,
            )
            if not skip_national
            else None
        )
        publish_reusable = regional_h5_reuse.reusable and (
            skip_national or national_h5_reuse.reusable
        )
        step_start = time.time()
        regional_h5_result = None
        national_h5_result = None
        if publish_reusable:
            _mark_step_reused(
                meta,
                LOCAL_AREA_H5_REGIONAL,
                regional_h5_reuse,
                vol=pipeline_volume,
            )
            if national_h5_reuse is not None:
                _mark_step_reused(
                    meta,
                    LOCAL_AREA_H5_NATIONAL,
                    national_h5_reuse,
                    vol=pipeline_volume,
                )
            print(
                f"\n[Step 4/5] {BUILD_OUTPUTS.title}: "
                "H5 outputs skipped - manifests valid; refreshing diagnostics..."
            )
        else:
            print(
                f"\n[Step 4/5] {BUILD_OUTPUTS.title}: "
                "building H5s and uploading diagnostics "
                "(parallel)..."
            )

            # Spawn H5 builds (run on separate Modal containers)
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
            regional_h5_manifest = _start_step_manifest(
                meta,
                LOCAL_AREA_H5_REGIONAL,
                scope="regional",
                parameters=regional_h5_parameters,
                input_identities=regional_h5_inputs,
                modal_call_id=regional_h5_handle.object_id,
                vol=pipeline_volume,
            )
            active_step_manifest = regional_h5_manifest

            national_h5_handle = None
            national_h5_manifest = None
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
                national_h5_manifest = _start_step_manifest(
                    meta,
                    LOCAL_AREA_H5_NATIONAL,
                    scope="national",
                    parameters=national_h5_parameters,
                    input_identities=national_h5_inputs,
                    modal_call_id=national_h5_handle.object_id,
                    vol=pipeline_volume,
                )

            # Now wait for H5 builds to finish. Do not reload the shared
            # volume until the child jobs release SQLite handles.
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

            pipeline_volume.reload()
            staging_volume.reload()

            if isinstance(regional_h5_result, dict) and regional_h5_result.get(
                "fingerprint"
            ):
                if regional_h5_result["fingerprint"] != regional_scope_fingerprint:
                    raise RuntimeError(
                        "Regional H5 fingerprint changed between pipeline "
                        "reuse planning and child publish completion.\n"
                        f"  Planned: {regional_scope_fingerprint}\n"
                        f"  Actual:  {regional_h5_result['fingerprint']}"
                    )
                regional_h5_manifest.input_identities["h5_scope_fingerprint"] = (
                    regional_h5_result["fingerprint"]
                )
            regional_reuse_measurement = ReuseMeasurement.from_dict(
                regional_h5_result.get("reuse_measurement", {})
                if isinstance(regional_h5_result, dict)
                else {}
            )
            _complete_step_manifest(
                regional_h5_manifest,
                outputs=_collect_staging_outputs(run_id, scope="regional"),
                diagnostics=_collect_diagnostics(run_id),
                reuse_decision=(
                    "partially_reused"
                    if regional_reuse_measurement.valid_reused_outputs
                    else "computed"
                ),
                reuse_measurement=regional_reuse_measurement,
                vol=pipeline_volume,
            )
            active_step_manifest = national_h5_manifest

            if national_h5_handle is not None:
                if isinstance(national_h5_result, dict) and national_h5_result.get(
                    "fingerprint"
                ):
                    if national_h5_result["fingerprint"] != national_scope_fingerprint:
                        raise RuntimeError(
                            "National H5 fingerprint changed between pipeline "
                            "reuse planning and child publish completion.\n"
                            f"  Planned: {national_scope_fingerprint}\n"
                            f"  Actual:  {national_h5_result['fingerprint']}"
                        )
                    national_h5_manifest.input_identities["h5_scope_fingerprint"] = (
                        national_h5_result["fingerprint"]
                    )
                national_reuse_measurement = ReuseMeasurement.from_dict(
                    national_h5_result.get("reuse_measurement", {})
                    if isinstance(national_h5_result, dict)
                    else {}
                )
                _complete_step_manifest(
                    national_h5_manifest,
                    outputs=_collect_staging_outputs(run_id, scope="national"),
                    diagnostics=_collect_diagnostics(run_id),
                    reuse_decision=(
                        "partially_reused"
                        if national_reuse_measurement.valid_reused_outputs
                        else "computed"
                    ),
                    reuse_measurement=national_reuse_measurement,
                    vol=pipeline_volume,
                )
                active_step_manifest = None

        # ── Aggregate validation results ──
        _write_validation_diagnostics(
            run_id=run_id,
            regional_result=regional_h5_result,
            national_result=national_h5_result,
            vol=pipeline_volume,
        )

        # Upload validation diagnostics even when H5 outputs are reused.
        print("  Uploading validation diagnostics...")
        diagnostics_manifest = _start_step_manifest(
            meta,
            UPLOAD_DIAGNOSTICS,
            parameters={"branch": branch, "run_id": run_id},
            input_identities={
                "diagnostics": [
                    artifact.to_dict() for artifact in _collect_diagnostics(run_id)
                ]
            },
            vol=pipeline_volume,
        )
        active_step_manifest = diagnostics_manifest
        upload_run_diagnostics(run_id, branch)
        diagnostic_outputs = _collect_diagnostics(run_id)
        _complete_step_manifest(
            diagnostics_manifest,
            outputs=diagnostic_outputs,
            diagnostics=diagnostic_outputs,
            reuse_decision="computed",
            reuse_measurement=ReuseMeasurement(
                expected_outputs=len(diagnostic_outputs),
                recomputed_outputs=len(diagnostic_outputs),
            ),
            vol=pipeline_volume,
        )

        active_step_manifest = None
        print(f"  Completed in {round(time.time() - step_start, 1)}s")

        # ── Step 5: Finalize ──
        print("\n[Step 5/5] Finalizing run...")
        meta.status = "completed"
        write_run_meta(meta, pipeline_volume)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Run ID: {run_id}")
        print(f"  Status: {meta.status}")
        _print_step_manifests(run_id)
        print(
            f"\nTo promote, run:\n"
            f"  modal run modal_app/pipeline.py"
            f"::main --action promote "
            f"--run-id {run_id}"
        )
        print("=" * 60)

        return run_id

    except Exception as e:
        _fail_step_manifest(active_step_manifest, e, pipeline_volume)
        meta.status = "failed"
        meta.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        write_run_meta(meta, pipeline_volume)
        print(f"\nPIPELINE FAILED: {e}")
        print(f"Resume with: --resume-run-id {run_id}")
        raise


def _read_step_manifests(run_id: str) -> list[StepManifest]:
    """Read all step manifests for a run."""
    steps_dir = _run_dir(run_id) / "steps"
    if not steps_dir.exists():
        return []
    return [read_step_manifest(path) for path in sorted(steps_dir.glob("*.json"))]


def _completed_step_manifest_ids(run_id: str) -> list[str]:
    """Return step IDs that have a completed/reusable manifest state."""
    return [
        manifest.step_id
        for manifest in _read_step_manifests(run_id)
        if manifest.status in {"completed", "reused", "partially_reused"}
    ]


def _print_step_manifests(run_id: str) -> None:
    """Print formatted step-manifest durations."""
    total = 0.0
    for manifest in _read_step_manifests(run_id):
        duration = manifest.duration_s or 0.0
        total += duration
        print(
            f"  {manifest.step_id}: "
            f"{duration}s ({manifest.status}, {manifest.reuse_decision})"
        )
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
@pipeline_node(
    PipelineNode(
        id="promote_pipeline_run",
        label="Promote Pipeline Run",
        node_type="entrypoint",
        description="Promote a completed staged pipeline run without re-running computation.",
        source_file="modal_app/pipeline.py",
        status="current",
        stability="moving",
        pathways=["orchestration", "local_h5"],
        artifacts_in=["staged H5 files", "run metadata"],
        artifacts_out=["production H5 release"],
        validation_commands=["uv run pytest tests/unit/test_pipeline.py"],
    )
)
def promote_run(
    run_id: str,
    version: str = None,
) -> str:
    """Promote a completed pipeline run to production.

    1. Verify run status is "completed"
    2. Promote every staged artifact in one Hugging Face commit
    3. Upload/copy every artifact to GCS
    4. Finalize release_manifest.json, tag the release, and update
       version_manifest.json
    5. Update run status to "promoted"

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
    promotion_context = RunContext.from_mapping(
        meta.run_context,
        run_id=run_id,
        modal_app_name=meta.modal_app_name,
        modal_environment=meta.modal_environment,
    )
    _apply_run_context_env(promotion_context)
    if not meta.run_context:
        meta.run_context = promotion_context.to_dict()
    meta.modal_app_name = meta.modal_app_name or promotion_context.modal_app_name
    meta.modal_environment = (
        meta.modal_environment or promotion_context.modal_environment
    )
    meta.hf_staging_prefix = (
        meta.hf_staging_prefix or promotion_context.hf_staging_prefix
    )

    if meta.status not in ("completed", "promoted"):
        raise RuntimeError(
            f"Run {run_id} has status "
            f"'{meta.status}'. Only completed runs "
            f"can be promoted."
        )

    if meta.status == "promoted":
        print(f"WARNING: Run {run_id} was already promoted. Re-promoting...")

    version = version or meta.version
    promote_inputs = {
        "validated_step_outputs": [
            artifact.to_dict()
            for artifact in completed_validated_outputs(
                _run_dir(run_id),
                step_ids=[
                    BUILD_DATASETS.id,
                    LOCAL_AREA_H5_REGIONAL.id,
                    LOCAL_AREA_H5_NATIONAL.id,
                ],
            )
        ]
    }
    if (
        run_manifest_path(_run_dir(run_id)).exists()
        and not promote_inputs["validated_step_outputs"]
    ):
        raise RuntimeError(
            "No validated completed step outputs found for release promotion. "
            "Run Phase 3c pipeline steps before promoting this run."
        )
    promote_manifest = _start_step_manifest(
        meta,
        VALIDATE_AND_PROMOTE_RELEASE,
        parameters={"version": version, "run_id": run_id},
        input_identities=promote_inputs,
        vol=pipeline_volume,
    )

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

    try:
        rel_paths = _full_release_staging_rel_paths(run_id)
        print(f"\nPromoting {len(rel_paths)} staged release artifact(s)...")
        promotion_stdout = _promote_full_release_from_staging(
            run_id,
            version,
            promotion_context.to_dict(),
        )
        print(f"  {promotion_stdout}")

        # Update run status only after all required promotion work succeeds.
        meta.status = "promoted"
        _complete_step_manifest(
            promote_manifest,
            outputs=[
                ArtifactReference.from_dict(artifact)
                for artifact in promote_inputs["validated_step_outputs"]
            ],
            reuse_decision="computed",
            vol=pipeline_volume,
        )
        write_run_meta(meta, pipeline_volume)
    except Exception as exc:
        _fail_step_manifest(promote_manifest, exc, pipeline_volume)
        raise

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
        steps_dir = _run_dir(run_id) / "steps"
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
        if steps_dir.exists():
            lines.append("  Step manifests:")
            for manifest_path in sorted(steps_dir.glob("*.json")):
                manifest = read_step_manifest(manifest_path)
                duration = (
                    manifest.duration_s if manifest.duration_s is not None else "?"
                )
                reuse = manifest.reuse_decision
                lines.append(
                    f"    {manifest.step_id}: {duration}s ({manifest.status}, {reuse})"
                )
        return "\n".join(lines)

    # List all runs
    runs = []
    for entry in sorted(runs_dir.iterdir()):
        manifest_path = entry / "run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
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
    national_epochs: int = 1000,
    num_workers: int = 50,
    n_clones: int = 430,
    skip_national: bool = False,
    clear_checkpoints: bool = False,
    version: str = None,
    sha_override: str = "",
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
            version_override=version or "",
            sha_override=sha_override,
            run_id=run_id or "",
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
