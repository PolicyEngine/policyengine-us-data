"""
Modal coordinator for local and national H5 publishing.

This module is now an adapter over the internal `local_h5` components:

1. Resolve concrete US publish requests from `USAreaCatalog`
2. Reconcile the staging run directory against the publish fingerprint
3. Partition request work across Modal workers
4. Invoke the worker script with serialized request payloads
5. Aggregate structured worker results, validation rows, and errors
6. Stage manifests and uploads

The one-area build logic no longer lives here. That now sits under
`policyengine_us_data.calibration.local_h5`.
"""

import heapq
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

    from modal_app.images import cpu_image as image
from modal_app.resilience import reconcile_run_dir_fingerprint
from policyengine_us_data.calibration.local_h5.area_catalog import (
    USAreaCatalog,
)
from policyengine_us_data.calibration.local_h5.contracts import (
    AreaBuildRequest,
    AreaBuildResult,
    ValidationIssue,
    WorkerResult,
)
from policyengine_us_data.calibration.local_h5.fingerprinting import (
    FingerprintService,
)
from policyengine_us_data.calibration.local_h5.package_geography import (
    CalibrationPackageGeographyLoader,
    require_calibration_package_path,
)
from policyengine_us_data.calibration.local_h5.partitioning import (
    partition_weighted_work_items,
    work_item_key,
)

app = modal.App("policyengine-us-data-local-area")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

staging_volume = modal.Volume.from_name(
    "local-area-staging",
    create_if_missing=True,
)

pipeline_volume = modal.Volume.from_name(
    "pipeline-artifacts",
    create_if_missing=True,
)

VOLUME_MOUNT = "/staging"


def setup_gcp_credentials():
    """Write GCP credentials JSON to a temp file for google.auth.default()."""
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_path = "/tmp/gcp-credentials.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        return creds_path
    return None


def setup_repo(branch: str):
    """Change to the pre-baked repo directory.

    The branch parameter is kept for API compatibility but is
    no longer used for cloning -- code is baked into the image.
    """
    os.chdir("/root/policyengine-us-data")


def validate_artifacts(
    config_path: Path,
    artifact_dir: Path,
    filename_remap: Dict[str, str] = None,
) -> None:
    """Verify artifact checksums against unified_run_config.json.

    Args:
        config_path: Path to unified_run_config.json.
        artifact_dir: Directory containing the artifact files.
        filename_remap: Optional mapping from config filenames to
            actual filenames on disk (e.g. national weights are
            stored as national_calibration_weights.npy but the
            config records calibration_weights.npy).

    Raises:
        RuntimeError: If any artifact is missing or has a
            checksum mismatch.
    """
    import hashlib

    if not config_path.exists():
        print(
            "WARNING: unified_run_config.json not found, "
            "skipping artifact validation "
            "(backwards compat with old runs)"
        )
        return

    with open(config_path) as f:
        config = json.load(f)

    artifacts = config.get("artifacts", {})
    if not artifacts:
        print("WARNING: No artifacts section in run config, skipping validation")
        return

    remap = filename_remap or {}
    for filename, expected_hash in artifacts.items():
        actual_filename = remap.get(filename, filename)
        filepath = artifact_dir / actual_filename
        if not filepath.exists():
            raise RuntimeError(
                f"Artifact validation failed: {actual_filename} not found in {artifact_dir}"
            )
        h = hashlib.sha256()
        with open(filepath, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        actual = f"sha256:{h.hexdigest()}"
        if actual != expected_hash:
            raise RuntimeError(
                f"Artifact validation failed: {filename} "
                f"checksum mismatch.\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual}"
            )

    print(f"Validated {len(artifacts)} artifact(s) against run config checksums")


def get_version() -> str:
    """Get package version from pyproject.toml."""
    import tomllib

    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    return pyproject["project"]["version"]


def get_completed_from_volume(run_dir: Path) -> set:
    """Scan volume to find already-built files."""
    completed = set()

    states_dir = run_dir / "states"
    if states_dir.exists():
        for f in states_dir.glob("*.h5"):
            completed.add(f"state:{f.stem}")

    districts_dir = run_dir / "districts"
    if districts_dir.exists():
        for f in districts_dir.glob("*.h5"):
            completed.add(f"district:{f.stem}")

    cities_dir = run_dir / "cities"
    if cities_dir.exists():
        for f in cities_dir.glob("*.h5"):
            completed.add(f"city:{f.stem}")

    return completed


def _derive_canonical_n_clones(
    *,
    weights_path: Path,
    package_path: Path,
    requested_n_clones: int,
) -> int:
    """Use weights length as the canonical clone-count source for publishing."""

    import numpy as np

    from policyengine_us_data.calibration.local_h5.package_geography import (
        CalibrationPackageGeographyLoader,
    )
    from policyengine_us_data.calibration.local_h5.weights import (
        infer_clone_count_from_weight_length,
    )

    weights = np.load(weights_path, mmap_mode="r")
    loader = CalibrationPackageGeographyLoader()
    loaded = loader.load(package_path)
    if loaded is None:
        raise RuntimeError(
            f"Calibration package at {package_path} does not contain usable geography"
        )

    canonical_n_clones = infer_clone_count_from_weight_length(
        weights.shape[0],
        loaded.geography.n_records,
    )
    if requested_n_clones != canonical_n_clones:
        print(
            f"WARNING: requested n_clones={requested_n_clones} but "
            f"weights imply {canonical_n_clones}; using weights-derived value"
    )
    return canonical_n_clones


def _request_key(request: AreaBuildRequest) -> str:
    return f"{request.area_type}:{request.area_id}"


def _phase_errors_from_worker_result(worker_result: WorkerResult) -> list[dict]:
    phase_errors: list[dict] = []

    for result in worker_result.failed:
        phase_errors.append(
            {
                "item": _request_key(result.request),
                "error": result.build_error,
            }
        )

    for issue in worker_result.worker_issues:
        phase_errors.append(
            {
                "item": "worker",
                "error": issue.message,
                "code": issue.code,
                "details": dict(issue.details),
            }
        )

    return phase_errors


def _validation_rows_from_worker_result(worker_result: WorkerResult) -> list[dict]:
    rows: list[dict] = []
    for result in worker_result.completed:
        if result.validation.status in ("passed", "failed"):
            rows.extend(dict(row) for row in result.validation.rows)
    return rows


def _validation_errors_from_worker_result(worker_result: WorkerResult) -> list[dict]:
    errors: list[dict] = []
    for result in worker_result.completed:
        if result.validation.status != "error":
            continue
        item_key = _request_key(result.request)
        for issue in result.validation.issues:
            errors.append(
                {
                    "item": item_key,
                    "error": issue.message,
                    "code": issue.code,
                    "details": dict(issue.details),
                }
            )
    return errors


def _worker_failure_result(
    requests: List[Dict],
    *,
    error: str,
    code: str,
) -> Dict:
    failed_results = []
    for payload in requests:
        request = AreaBuildRequest.from_dict(payload)
        failed_results.append(
            AreaBuildResult(
                request=request,
                build_status="failed",
                build_error=error,
            )
        )

    result = WorkerResult(
        completed=(),
        failed=tuple(failed_results),
        worker_issues=(
            ValidationIssue(
                code=code,
                message=error,
                severity="error",
            ),
        ),
    )
    return result.to_dict()


def _load_catalog_geography(package_path: Path):
    loader = CalibrationPackageGeographyLoader()
    loaded = loader.load(package_path)
    if loaded is None:
        raise RuntimeError(
            f"Calibration package at {package_path} does not contain usable geography"
        )
    return loaded.geography


def run_phase(
    phase_name: str,
    entries: List,
    num_workers: int,
    completed: set,
    branch: str,
    run_id: str,
    calibration_inputs: Dict[str, str],
    run_dir: Path,
    validate: bool = True,
) -> tuple:
    """Run a single build phase, spawning workers and collecting results.

    Returns:
        A tuple of (volume_completed, phase_errors, validation_rows,
        validation_errors)
        where phase_errors is a list of error dicts from workers
        and crashes, validation_rows is a list of per-target
        validation result dicts, and validation_errors is a list
        of structured validation execution failures.
    """
    work_items = [entry.to_partition_item() for entry in entries]
    requests_by_key = {
        entry.key: entry.request.to_dict() for entry in entries
    }
    work_chunks = partition_weighted_work_items(work_items, num_workers, completed)
    total_remaining = sum(len(c) for c in work_chunks)

    print(f"\n--- Phase: {phase_name} ---")
    print(f"Remaining work: {total_remaining} items across {len(work_chunks)} workers")

    if total_remaining == 0:
        print(f"All {phase_name} items already built!")
        return completed, [], [], []

    handles = []
    for i, chunk in enumerate(work_chunks):
        total_weight = sum(item["weight"] for item in chunk)
        print(f"  Worker {i}: {len(chunk)} items, weight {total_weight}")
        requests = [requests_by_key[work_item_key(item)] for item in chunk]
        handle = build_areas_worker.spawn(
            branch=branch,
            run_id=run_id,
            requests=requests,
            calibration_inputs=calibration_inputs,
            validate=validate,
        )
        print(f"    → fc: {handle.object_id}")
        handles.append(handle)

    print(f"Waiting for {phase_name} workers to complete...")
    all_results = []
    all_errors = []
    all_validation_rows = []
    all_validation_errors = []

    for i, handle in enumerate(handles):
        try:
            payload = handle.get()
            if payload is None:
                all_errors.append({"worker": i, "error": "Worker returned None"})
                print(f"  Worker {i}: returned None (no results)")
                continue
            result = WorkerResult.from_dict(payload)
            all_results.append(result)
            print(
                f"  Worker {i}: {len(result.completed)} completed, "
                f"{len(result.failed)} failed"
            )
            worker_errors = _phase_errors_from_worker_result(result)
            if worker_errors:
                all_errors.extend(worker_errors)
            v_rows = _validation_rows_from_worker_result(result)
            if v_rows:
                all_validation_rows.extend(v_rows)
                print(f"  Worker {i}: {len(v_rows)} validation rows")
            v_errors = _validation_errors_from_worker_result(result)
            if v_errors:
                all_validation_errors.extend(v_errors)
                print(f"  Worker {i}: {len(v_errors)} validation errors")
        except Exception as e:
            all_errors.append(
                {"worker": i, "error": str(e), "traceback": traceback.format_exc()}
            )
            print(f"  Worker {i}: CRASHED - {e}")

    total_completed = sum(len(r.completed) for r in all_results)
    total_failed = sum(len(r.failed) for r in all_results)

    staging_volume.reload()
    volume_completed = get_completed_from_volume(run_dir)
    volume_new = volume_completed - completed

    print(f"\n{phase_name} summary (worker-reported):")
    print(f"  Completed: {total_completed}")
    print(f"  Failed: {total_failed}")
    print(f"{phase_name} summary (volume verification):")
    print(f"  Files on volume: {len(volume_completed)}")
    print(f"  New files this run: {len(volume_new)}")

    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        for err in all_errors[:5]:
            err_msg = str(err.get("error") or "Unknown")[:200]
            print(f"  - {err.get('item', err.get('worker'))}: {err_msg}")
        if len(all_errors) > 5:
            print(f"  ... and {len(all_errors) - 5} more")

    return volume_completed, all_errors, all_validation_rows, all_validation_errors


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={
        VOLUME_MOUNT: staging_volume,
        "/pipeline": pipeline_volume,
    },
    memory=16384,
    cpu=1.0,
    timeout=28800,
    max_containers=50,
    nonpreemptible=True,
)
def build_areas_worker(
    branch: str,
    run_id: str,
    requests: List[Dict],
    calibration_inputs: Dict[str, str],
    validate: bool = True,
) -> Dict:
    """
    Worker function that builds a subset of H5 files.
    Uses subprocess to avoid import conflicts with Modal's environment.
    """
    setup_gcp_credentials()
    setup_repo(branch)

    output_dir = Path(VOLUME_MOUNT) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    requests_json = json.dumps(requests)

    worker_cmd = [
        "uv",
        "run",
        "python",
        "modal_app/worker_script.py",
        "--requests-json",
        requests_json,
        "--weights-path",
        calibration_inputs["weights"],
        "--dataset-path",
        calibration_inputs["dataset"],
        "--db-path",
        calibration_inputs["database"],
        "--output-dir",
        str(output_dir),
    ]
    if "package" in calibration_inputs:
        worker_cmd.extend(
            ["--calibration-package-path", calibration_inputs["package"]]
        )
    if "n_clones" in calibration_inputs:
        worker_cmd.extend(["--n-clones", str(calibration_inputs["n_clones"])])
    if "seed" in calibration_inputs:
        worker_cmd.extend(["--seed", str(calibration_inputs["seed"])])
    repo_root = Path("/root/policyengine-us-data")
    cal_dir = repo_root / "policyengine_us_data" / "calibration"
    worker_cmd.extend(
        [
            "--target-config",
            str(cal_dir / "target_config.yaml"),
        ]
    )
    worker_cmd.extend(
        [
            "--validation-config",
            str(cal_dir / "target_config_full.yaml"),
        ]
    )
    if not validate:
        worker_cmd.append("--no-validate")
    result = subprocess.run(
        worker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        print(f"Worker stderr:\n{result.stderr}", file=__import__("sys").stderr)
        return _worker_failure_result(
            requests,
            error=(result.stderr or "No stderr")[:2000],
            code="worker_subprocess_failed",
        )

    try:
        results = json.loads(result.stdout)
    except json.JSONDecodeError:
        results = _worker_failure_result(
            requests,
            error=f"Failed to parse output: {result.stdout}",
            code="worker_output_parse_failed",
        )

    results = WorkerResult.from_dict(results).to_dict()

    staging_volume.commit()
    return results


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=4096,
    timeout=1800,
    nonpreemptible=True,
)
def validate_staging(branch: str, run_id: str, version: str = "") -> Dict:
    """Validate all expected files and generate manifest."""
    setup_repo(branch)

    if not version:
        version = run_id.split("_", 1)[0]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"""
import json
from pathlib import Path
from policyengine_us_data.utils.manifest import generate_manifest, save_manifest

staging_dir = Path("{VOLUME_MOUNT}")
run_id = "{run_id}"
version = "{version}"
manifest = generate_manifest(staging_dir, run_id, version=version)
manifest["run_id"] = run_id
manifest_path = staging_dir / run_id / "manifest.json"
save_manifest(manifest, manifest_path)
print(json.dumps(manifest))
""",
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Validation failed: {result.stderr}")

    manifest = json.loads(result.stdout)
    staging_volume.commit()

    print(f"Generated manifest with {len(manifest['files'])} files")
    print(f"  States: {manifest['totals']['states']}")
    print(f"  Districts: {manifest['totals']['districts']}")
    print(f"  Cities: {manifest['totals']['cities']}")
    print(f"  Total size: {manifest['totals']['total_size_bytes'] / 1e9:.2f} GB")

    return manifest


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=8192,
    timeout=28800,
    nonpreemptible=True,
)
def upload_to_staging(
    branch: str, version: str, manifest: Dict, run_id: str = ""
) -> str:
    """
    Upload files to HuggingFace staging only.

    GCS is updated during promote_publish, not here.
    Promote must be run separately via promote_publish.
    """
    setup_repo(branch)

    manifest_json = json.dumps(manifest)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"""
import json
from pathlib import Path
from policyengine_us_data.utils.manifest import verify_manifest
from policyengine_us_data.utils.data_upload import upload_to_staging_hf

manifest = json.loads('''{manifest_json}''')
version = "{version}"
run_id = "{run_id}"
staging_dir = Path("{VOLUME_MOUNT}")
run_dir = staging_dir / run_id

print("Verifying manifest before upload...")
verification = verify_manifest(staging_dir, manifest, subdir=run_id)
if not verification["valid"]:
    print(
        f"WARNING: Manifest verification issues: "
        f"{{len(verification['missing'])}} missing, "
        f"{{len(verification['checksum_mismatch'])}} checksum mismatches. "
        f"Proceeding with upload anyway."
    )
else:
    print(f"Verified {{verification['verified']}} files")

files_with_paths = []
for rel_path in manifest["files"].keys():
    local_path = run_dir / rel_path
    files_with_paths.append((local_path, rel_path))

# Upload to HuggingFace staging/
print(f"Uploading {{len(files_with_paths)}} files to HuggingFace staging/...")
hf_count = upload_to_staging_hf(files_with_paths, version, run_id=run_id)
print(f"Uploaded {{hf_count}} files to HuggingFace staging/")

print(f"Staged version {{version}} for promotion")
""",
        ],
        text=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Upload failed: {result.stderr}")

    return (
        f"Staged version {version} with {len(manifest['files'])} files. "
        f"Run promote workflow to publish to HuggingFace production and GCS."
    )


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=4096,
    timeout=3600,
    nonpreemptible=True,
)
def promote_publish(branch: str = "main", version: str = "", run_id: str = "") -> str:
    """
    Promote staged files from HF staging/ to production paths,
    upload to GCS, then cleanup HF staging.

    Reads the manifest from the Modal staging volume to determine which
    files to promote.
    """
    setup_gcp_credentials()
    setup_repo(branch)

    if not run_id:
        raise ValueError("--run-id is required for promote")
    if not version:
        version = run_id.split("_", 1)[0]

    staging_dir = Path(VOLUME_MOUNT)
    staging_volume.reload()

    manifest_path = staging_dir / run_id / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"No manifest found at {manifest_path}. Run build+stage workflow first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    rel_paths_json = json.dumps(list(manifest["files"].keys()))

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"""
import json
from pathlib import Path
from policyengine_us_data.utils.data_upload import (
    promote_staging_to_production_hf,
    cleanup_staging_hf,
    upload_local_area_file,
)

rel_paths = json.loads('''{rel_paths_json}''')
version = "{version}"
run_id = "{run_id}"
run_dir = Path("{VOLUME_MOUNT}") / run_id

print(f"Promoting {{len(rel_paths)}} files from staging/ to production (run_id={{run_id!r}})...")
promoted = promote_staging_to_production_hf(rel_paths, version, run_id=run_id)
print(f"Promoted {{promoted}} files to HuggingFace production")

print(f"Uploading {{len(rel_paths)}} files to GCS...")
gcs_count = 0
for rel_path in rel_paths:
    local_path = run_dir / rel_path
    subdirectory = str(Path(rel_path).parent)
    upload_local_area_file(
        str(local_path),
        subdirectory,
        version=version,
        skip_hf=True,
    )
    gcs_count += 1
print(f"Uploaded {{gcs_count}} files to GCS")

print("Cleaning up staging/...")
cleaned = cleanup_staging_hf(rel_paths, version, run_id=run_id)
print(f"Cleaned up {{cleaned}} files from staging/")

print(f"Successfully published version {{version}}")
""",
        ],
        text=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Promote failed: {result.stderr}")

    return (
        f"Successfully promoted version {version} with {len(manifest['files'])} files"
    )


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={
        VOLUME_MOUNT: staging_volume,
        "/pipeline": pipeline_volume,
    },
    memory=8192,
    timeout=86400,
    nonpreemptible=True,
)
def coordinate_publish(
    branch: str = "main",
    num_workers: int = 50,
    skip_upload: bool = False,
    n_clones: int = 430,
    validate: bool = True,
    run_id: str = "",
    expected_fingerprint: str = "",
) -> Dict:
    """Coordinate the full publishing workflow."""
    setup_gcp_credentials()
    setup_repo(branch)

    version = get_version()

    if not run_id:
        from policyengine_us_data.utils.run_id import generate_run_id

        sha = os.environ.get("GIT_COMMIT", "unknown")
        run_id = generate_run_id(version, sha)

    print("=" * 60)
    print(f"Run ID: {run_id}")
    print("=" * 60)
    print(f"Publishing version {version} from branch {branch}")
    print(f"Using {num_workers} parallel workers")

    staging_dir = Path(VOLUME_MOUNT)
    run_dir = staging_dir / run_id

    pipeline_volume.reload()
    artifacts = (
        Path(f"/pipeline/artifacts/{run_id}") if run_id else Path("/pipeline/artifacts")
    )
    weights_path = artifacts / "calibration_weights.npy"
    db_path = artifacts / "policy_data.db"
    dataset_path = artifacts / "source_imputed_stratified_extended_cps.h5"
    package_path = require_calibration_package_path(
        artifacts / "calibration_package.pkl"
    )
    config_json_path = artifacts / "unified_run_config.json"

    required = {
        "weights": weights_path,
        "dataset": dataset_path,
        "database": db_path,
    }
    for label, p in required.items():
        if not p.exists():
            raise RuntimeError(
                f"Missing {label} on pipeline volume: {p}. "
                f"Run upstream pipeline steps first."
            )
    print("All required pipeline artifacts found on volume.")

    canonical_n_clones = _derive_canonical_n_clones(
        weights_path=weights_path,
        package_path=package_path,
        requested_n_clones=n_clones,
    )

    calibration_inputs = {
        "weights": str(weights_path),
        "dataset": str(dataset_path),
        "database": str(db_path),
        "n_clones": canonical_n_clones,
        "seed": 42,
        "package": str(package_path),
    }
    print(f"Using calibration package geography from {package_path}")
    validate_artifacts(config_json_path, artifacts)

    if validate:
        try:
            from sqlalchemy import create_engine as _create_engine
            from policyengine_us_data.calibration.validate_staging import (
                _query_all_active_targets,
            )

            _test_engine = _create_engine(f"sqlite:///{db_path}")
            _df = _query_all_active_targets(_test_engine, 2024)
            print(f"Validation pre-flight OK: {len(_df)} targets queryable")
            _test_engine.dispose()
        except Exception as e:
            print(f"WARNING: Validation pre-flight failed: {e}")
            print("Disabling validation to protect H5 builds")
            validate = False

    # Fingerprint-based cache invalidation
    if expected_fingerprint:
        print(f"Using pinned fingerprint from pipeline: {expected_fingerprint}")

    fingerprint_service = FingerprintService()
    fingerprint_record = fingerprint_service.create_publish_fingerprint(
        weights_path=weights_path,
        dataset_path=dataset_path,
        calibration_package_path=package_path,
        n_clones=canonical_n_clones,
        seed=42,
    )
    fingerprint = fingerprint_record.digest

    if expected_fingerprint and expected_fingerprint != fingerprint:
        raise RuntimeError(
            "Pinned fingerprint does not match current publish inputs.\n"
            f"  Expected: {expected_fingerprint}\n"
            f"  Current:  {fingerprint}\n"
            "Start a fresh run instead of resuming."
        )

    reconcile_action = reconcile_run_dir_fingerprint(run_dir, fingerprint_record)
    if reconcile_action == "resume":
        print(f"Inputs unchanged ({fingerprint}), resuming...")
    else:
        print(f"Prepared staging directory for fingerprint {fingerprint}")
    staging_volume.commit()
    catalog = USAreaCatalog()
    catalog_geography = _load_catalog_geography(package_path)
    entries = list(
        catalog.resolved_regional_entries(
            f"sqlite:///{db_path}",
            geography=catalog_geography,
        )
    )
    states = [e for e in entries if e.request.area_type == "state"]
    districts = [e for e in entries if e.request.area_type == "district"]
    cities = [e for e in entries if e.request.area_type == "city"]

    staging_volume.reload()
    completed = get_completed_from_volume(run_dir)
    print(f"Found {len(completed)} already-completed items on volume")

    phase_args = dict(
        num_workers=num_workers,
        branch=branch,
        run_id=run_id,
        calibration_inputs=calibration_inputs,
        run_dir=run_dir,
        validate=validate,
    )

    accumulated_errors = []
    accumulated_validation_rows = []
    accumulated_validation_errors = []

    completed, phase_errors, v_rows, v_errors = run_phase(
        "All areas",
        entries=entries,
        completed=completed,
        **phase_args,
    )
    accumulated_errors.extend(phase_errors)
    accumulated_validation_rows.extend(v_rows)
    accumulated_validation_errors.extend(v_errors)

    expected_total = len(entries)

    # If workers crashed but all files landed on the volume,
    # treat as transient infrastructure errors (e.g. gRPC stream resets).
    if accumulated_errors:
        crash_errors = [e for e in accumulated_errors if "worker" in e]
        if crash_errors and len(completed) >= expected_total:
            print(
                f"WARNING: {len(crash_errors)} worker error(s) occurred "
                f"but all {expected_total} files present on volume. "
                f"Treating as transient. Errors: {crash_errors[:3]}"
            )
        elif crash_errors:
            raise RuntimeError(
                f"Build failed: {len(crash_errors)} worker "
                f"crash(es) detected and only "
                f"{len(completed)}/{expected_total} files on volume. "
                f"Errors: {crash_errors[:3]}"
            )

    if len(completed) < expected_total:
        missing = expected_total - len(completed)
        raise RuntimeError(
            f"Build incomplete: {missing} files missing from "
            f"volume ({len(completed)}/{expected_total}). "
            f"Volume preserved for retry."
        )

    if skip_upload:
        print("\nSkipping upload (--skip-upload flag set)")
        return {
            "message": (f"Build complete for version {version}. Upload skipped."),
            "validation_rows": accumulated_validation_rows,
            "validation_errors": accumulated_validation_errors,
            "fingerprint": fingerprint,
        }

    print("\nValidating staging...")
    manifest = validate_staging.remote(branch=branch, run_id=run_id, version=version)

    expected_total = len(entries)
    actual_total = (
        manifest["totals"]["states"]
        + manifest["totals"]["districts"]
        + manifest["totals"]["cities"]
    )

    if actual_total < expected_total:
        print(f"WARNING: Expected {expected_total} files, found {actual_total}")

    print("\nStarting upload to staging...")
    result = upload_to_staging.remote(
        branch=branch, version=version, manifest=manifest, run_id=run_id
    )
    print(result)

    print("\n" + "=" * 60)
    print("BUILD + STAGE COMPLETE")
    print(f"Run ID: {run_id}")
    print("=" * 60)
    print(
        f"To promote: modal run modal_app/local_area.py::main_promote "
        f"--version={version} --run-id={run_id}"
    )
    print("=" * 60)

    return {
        "message": result,
        "run_id": run_id,
        "validation_rows": accumulated_validation_rows,
        "validation_errors": accumulated_validation_errors,
        "fingerprint": fingerprint,
    }


@app.local_entrypoint()
def main(
    branch: str = "main",
    num_workers: int = 50,
    skip_upload: bool = False,
    n_clones: int = 430,
    run_id: str = "",
):
    """Local entrypoint for Modal CLI."""
    result = coordinate_publish.remote(
        branch=branch,
        num_workers=num_workers,
        skip_upload=skip_upload,
        n_clones=n_clones,
        run_id=run_id,
    )
    if isinstance(result, dict):
        print(result.get("message", result))
    else:
        print(result)


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={
        VOLUME_MOUNT: staging_volume,
        "/pipeline": pipeline_volume,
    },
    memory=16384,
    timeout=28800,
    nonpreemptible=True,
)
def coordinate_national_publish(
    branch: str = "main",
    n_clones: int = 430,
    validate: bool = True,
    run_id: str = "",
) -> Dict:
    """Build and upload a national US.h5 from national weights."""
    setup_gcp_credentials()
    setup_repo(branch)

    version = get_version()

    if not run_id:
        from policyengine_us_data.utils.run_id import generate_run_id

        sha = os.environ.get("GIT_COMMIT", "unknown")
        run_id = generate_run_id(version, sha)

    print("=" * 60)
    print(f"Run ID: {run_id}")
    print("=" * 60)
    print(f"Building national H5 for version {version} from branch {branch}")

    staging_dir = Path(VOLUME_MOUNT)

    pipeline_volume.reload()
    artifacts = (
        Path(f"/pipeline/artifacts/{run_id}") if run_id else Path("/pipeline/artifacts")
    )
    weights_path = artifacts / "national_calibration_weights.npy"
    db_path = artifacts / "policy_data.db"
    dataset_path = artifacts / "source_imputed_stratified_extended_cps.h5"
    package_path = require_calibration_package_path(
        artifacts / "calibration_package.pkl"
    )
    config_json_path = artifacts / "national_unified_run_config.json"

    required = {
        "weights": weights_path,
        "dataset": dataset_path,
        "database": db_path,
    }
    for label, p in required.items():
        if not p.exists():
            raise RuntimeError(
                f"Missing {label} on pipeline volume: {p}. "
                f"Run upstream pipeline steps first."
            )
    print("All required national pipeline artifacts found.")

    canonical_n_clones = _derive_canonical_n_clones(
        weights_path=weights_path,
        package_path=package_path,
        requested_n_clones=n_clones,
    )

    calibration_inputs = {
        "weights": str(weights_path),
        "dataset": str(dataset_path),
        "database": str(db_path),
        "n_clones": canonical_n_clones,
        "seed": 42,
        "package": str(package_path),
    }
    print(f"Using calibration package geography from {package_path}")
    validate_artifacts(
        config_json_path,
        artifacts,
        filename_remap={
            "calibration_weights.npy": "national_calibration_weights.npy",
        },
    )
    run_dir = staging_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    catalog = USAreaCatalog()
    national_entry = catalog.resolved_national_entry()
    print("Spawning worker for national H5 build...")
    worker_payload = build_areas_worker.remote(
        branch=branch,
        run_id=run_id,
        requests=[national_entry.request.to_dict()],
        calibration_inputs=calibration_inputs,
        validate=validate,
    )
    worker_result = WorkerResult.from_dict(worker_payload)

    print(
        f"Worker result: "
        f"{len(worker_result.completed)} completed, "
        f"{len(worker_result.failed)} failed"
    )

    phase_errors = _phase_errors_from_worker_result(worker_result)
    validation_errors = _validation_errors_from_worker_result(worker_result)

    if worker_result.failed or phase_errors:
        raise RuntimeError(f"National build failed: {phase_errors}")

    staging_volume.reload()
    national_h5 = run_dir / "national" / "US.h5"
    if not national_h5.exists():
        raise RuntimeError(f"Expected {national_h5} not found after build")

    # Compute SHA256 checksum before upload for integrity verification
    import hashlib

    h = hashlib.sha256()
    with open(national_h5, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    national_checksum = f"sha256:{h.hexdigest()}"
    national_size = national_h5.stat().st_size
    print(f"National H5 checksum: {national_checksum} ({national_size:,} bytes)")

    # ── National validation ──
    national_validation_output = ""
    if validate:
        print("Running national H5 validation...")
        val_result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "policyengine_us_data.calibration.validate_national_h5",
                "--h5-path",
                str(national_h5),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        national_validation_output = val_result.stdout
        print(val_result.stdout)
        if val_result.stderr:
            print(val_result.stderr)
        if val_result.returncode != 0:
            print(
                "WARNING: National validation returned "
                f"non-zero exit code: {val_result.returncode}"
            )

    print(f"Uploading {national_h5} to HF staging...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"""
from policyengine_us_data.utils.data_upload import (
    upload_to_staging_hf,
)
upload_to_staging_hf(
    [("{national_h5}", "national/US.h5")],
    "{version}",
    run_id="{run_id}",
)
print("Done")
""",
        ],
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Staging upload failed: {result.stderr}")

    # Verify the file still exists on the volume after upload
    staging_volume.reload()
    if not national_h5.exists():
        raise RuntimeError("National H5 disappeared from staging volume after upload")
    print(
        f"Post-upload verification passed: {national_h5} "
        f"(checksum: {national_checksum})"
    )

    print("National H5 staged. Run promote workflow to publish.")
    return {
        "message": (
            f"National US.h5 built and staged for version "
            f"{version}. Run main_national_promote to publish."
        ),
        "run_id": run_id,
        "national_validation": national_validation_output,
        "validation_errors": validation_errors,
    }


@app.local_entrypoint()
def main_national(branch: str = "main", n_clones: int = 430, run_id: str = ""):
    """Build and stage national US.h5."""
    result = coordinate_national_publish.remote(
        branch=branch, n_clones=n_clones, run_id=run_id
    )
    if isinstance(result, dict):
        print(result.get("message", result))
    else:
        print(result)


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=4096,
    timeout=3600,
    nonpreemptible=True,
)
def promote_national_publish(
    branch: str = "main",
    version: str = "",
    run_id: str = "",
) -> str:
    """Promote national US.h5 from HF staging to production + GCS."""
    setup_gcp_credentials()
    setup_repo(branch)

    if not run_id:
        raise ValueError("--run-id is required for promote")
    if not version:
        version = run_id.split("_", 1)[0]
    rel_paths = ["national/US.h5"]

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"""
import json
from pathlib import Path
from policyengine_us_data.utils.data_upload import (
    promote_staging_to_production_hf,
    cleanup_staging_hf,
    upload_local_area_file,
)

version = "{version}"
run_id = "{run_id}"
rel_paths = {json.dumps(rel_paths)}
run_dir = Path("{VOLUME_MOUNT}") / run_id

print(f"Promoting national H5 from staging to production (run_id={{run_id!r}})...")
promoted = promote_staging_to_production_hf(rel_paths, version, run_id=run_id)
print(f"Promoted {{promoted}} files to HuggingFace production")

national_h5 = run_dir / "national" / "US.h5"
if national_h5.exists():
    print("Uploading national H5 to GCS...")
    upload_local_area_file(
        str(national_h5), "national", version=version, skip_hf=True
    )
    print("Uploaded national H5 to GCS")
else:
    print(f"WARNING: {{national_h5}} not on volume, skipping GCS")

print("Cleaning up staging...")
cleaned = cleanup_staging_hf(rel_paths, version, run_id=run_id)
print(f"Cleaned up {{cleaned}} files from staging")
print(f"Successfully promoted national H5 for version {{version}}")
""",
        ],
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"National promote failed: {result.stderr}")

    return f"National US.h5 promoted for version {version}"


@app.local_entrypoint()
def main_national_promote(branch: str = "main", run_id: str = ""):
    """Promote staged national US.h5 to production."""
    result = promote_national_publish.remote(branch=branch, run_id=run_id)
    print(result)


@app.local_entrypoint()
def main_promote(
    run_id: str = "",
    branch: str = "main",
):
    """Promote staged files to HuggingFace production."""
    if not run_id:
        raise ValueError("--run-id is required")
    result = promote_publish.remote(branch=branch, run_id=run_id)
    print(result)
