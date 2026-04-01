"""
Modal app for publishing local area H5 files with parallel workers.

Architecture:
1. Coordinator partitions work across N workers
2. Workers build H5 files in parallel, writing to shared Volume
3. Validation generates manifest with checksums
4. Atomic upload to versioned paths, updates latest.json last

Usage:
    modal run modal_app/local_area.py --branch=main --num-workers=8
"""

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
from modal_app.resilience import reconcile_version_dir_fingerprint

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


def generate_work_items(scope: str, db_path: str) -> List[Dict]:
    """Auto-generate a flat list of work items based on scope.

    Args:
        scope: One of 'all', 'national', 'state', 'congressional',
            'local', or 'test'.
        db_path: Path to policy_data.db for querying districts.

    Returns:
        List of work item dicts: [{"type": str, "id": str}, ...]
    """
    from policyengine_us_data.calibration.calibration_utils import (
        get_all_cds_from_database,
        STATE_CODES,
    )
    from policyengine_us_data.calibration.publish_local_area import (
        get_district_friendly_name,
    )

    all_states = list(STATE_CODES.values())
    db_uri = f"sqlite:///{db_path}"
    all_cds = get_all_cds_from_database(db_uri)
    all_districts = [get_district_friendly_name(cd) for cd in all_cds]
    all_cities = ["NYC"]

    items = []

    if scope == "national":
        items.append({"type": "national", "id": "US"})

    elif scope == "state":
        for s in all_states:
            items.append({"type": "state", "id": s})

    elif scope == "congressional":
        for d in all_districts:
            items.append({"type": "district", "id": d})

    elif scope == "local":
        for c in all_cities:
            items.append({"type": "city", "id": c})

    elif scope == "test":
        items.append({"type": "national", "id": "US"})
        items.append({"type": "state", "id": "NY"})
        items.append({"type": "district", "id": "NV-01"})

    else:  # "all" or unrecognized
        items.append({"type": "national", "id": "US"})
        for s in all_states:
            items.append({"type": "state", "id": s})
        for d in all_districts:
            items.append({"type": "district", "id": d})
        for c in all_cities:
            items.append({"type": "city", "id": c})

    return items


def partition_work(
    states: List[str],
    districts: List[str],
    cities: List[str],
    num_workers: int,
    completed: set,
) -> List[List[Dict]]:
    """Partition work items across N workers."""
    remaining = []

    for s in states:
        item_id = f"state:{s}"
        if item_id not in completed:
            remaining.append({"type": "state", "id": s, "weight": 5})

    for d in districts:
        item_id = f"district:{d}"
        if item_id not in completed:
            remaining.append({"type": "district", "id": d, "weight": 1})

    for c in cities:
        item_id = f"city:{c}"
        if item_id not in completed:
            remaining.append({"type": "city", "id": c, "weight": 3})

    remaining.sort(key=lambda x: -x["weight"])

    chunks = [[] for _ in range(num_workers)]
    for i, item in enumerate(remaining):
        chunks[i % num_workers].append(item)

    return [c for c in chunks if c]


def get_completed_from_volume(version_dir: Path) -> set:
    """Scan volume to find already-built files."""
    completed = set()

    states_dir = version_dir / "states"
    if states_dir.exists():
        for f in states_dir.glob("*.h5"):
            completed.add(f"state:{f.stem}")

    districts_dir = version_dir / "districts"
    if districts_dir.exists():
        for f in districts_dir.glob("*.h5"):
            completed.add(f"district:{f.stem}")

    cities_dir = version_dir / "cities"
    if cities_dir.exists():
        for f in cities_dir.glob("*.h5"):
            completed.add(f"city:{f.stem}")

    return completed


def run_phase(
    phase_name: str,
    states: List[str],
    districts: List[str],
    cities: List[str],
    num_workers: int,
    completed: set,
    branch: str,
    version: str,
    calibration_inputs: Dict[str, str],
    version_dir: Path,
    validate: bool = True,
) -> tuple:
    """Run a single build phase, spawning workers and collecting results.

    Returns:
        A tuple of (volume_completed, phase_errors, validation_rows)
        where phase_errors is a list of error dicts from workers
        and crashes, and validation_rows is a list of per-target
        validation result dicts.
    """
    work_chunks = partition_work(states, districts, cities, num_workers, completed)
    total_remaining = sum(len(c) for c in work_chunks)

    print(f"\n--- Phase: {phase_name} ---")
    print(f"Remaining work: {total_remaining} items across {len(work_chunks)} workers")

    if total_remaining == 0:
        print(f"All {phase_name} items already built!")
        return completed, [], []

    handles = []
    for i, chunk in enumerate(work_chunks):
        print(f"  Worker {i}: {len(chunk)} items")
        handle = build_areas_worker.spawn(
            branch=branch,
            version=version,
            work_items=chunk,
            calibration_inputs=calibration_inputs,
            validate=validate,
        )
        print(f"    → fc: {handle.object_id}")
        handles.append(handle)

    print(f"Waiting for {phase_name} workers to complete...")
    all_results = []
    all_errors = []
    all_validation_rows = []

    for i, handle in enumerate(handles):
        try:
            result = handle.get()
            if result is None:
                all_errors.append({"worker": i, "error": "Worker returned None"})
                print(f"  Worker {i}: returned None (no results)")
                continue
            all_results.append(result)
            print(
                f"  Worker {i}: {len(result['completed'])} completed, "
                f"{len(result['failed'])} failed"
            )
            if result["errors"]:
                all_errors.extend(result["errors"])
            # Collect validation rows
            v_rows = result.get("validation_rows", [])
            if v_rows:
                all_validation_rows.extend(v_rows)
                print(f"  Worker {i}: {len(v_rows)} validation rows")
        except Exception as e:
            all_errors.append(
                {"worker": i, "error": str(e), "traceback": traceback.format_exc()}
            )
            print(f"  Worker {i}: CRASHED - {e}")

    total_completed = sum(len(r["completed"]) for r in all_results)
    total_failed = sum(len(r["failed"]) for r in all_results)

    staging_volume.reload()
    volume_completed = get_completed_from_volume(version_dir)
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

    return volume_completed, all_errors, all_validation_rows


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={
        VOLUME_MOUNT: staging_volume,
        "/pipeline": pipeline_volume,
    },
    memory=16384,
    cpu=4.0,
    timeout=28800,
    nonpreemptible=True,
)
def build_areas_worker(
    branch: str,
    version: str,
    work_items: List[Dict],
    calibration_inputs: Dict[str, str],
    validate: bool = True,
) -> Dict:
    """
    Worker function that builds a subset of H5 files.
    Uses subprocess to avoid import conflicts with Modal's environment.
    """
    setup_gcp_credentials()
    setup_repo(branch)

    output_dir = Path(VOLUME_MOUNT) / version
    output_dir.mkdir(parents=True, exist_ok=True)

    work_items_json = json.dumps(work_items)

    worker_cmd = [
        "python",
        "modal_app/worker_script.py",
        "--work-items",
        work_items_json,
        "--weights-path",
        calibration_inputs["weights"],
        "--dataset-path",
        calibration_inputs["dataset"],
        "--db-path",
        calibration_inputs["database"],
        "--output-dir",
        str(output_dir),
    ]
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
        return {
            "completed": [],
            "failed": [f"{item['type']}:{item['id']}" for item in work_items],
            "errors": [{"error": (result.stderr or "No stderr")[:2000]}],
        }

    try:
        results = json.loads(result.stdout)
    except json.JSONDecodeError:
        results = {
            "completed": [],
            "failed": [],
            "errors": [{"error": f"Failed to parse output: {result.stdout}"}],
        }

    staging_volume.commit()
    return results


# ── Queue-based architecture ──────────────────────────────────
#
# build_single_area: processes ONE work item per container (1 CPU).
# queue_coordinator: generates items from scope, spawns workers,
#     collects results.


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={
        VOLUME_MOUNT: staging_volume,
        "/pipeline": pipeline_volume,
    },
    memory=16384,
    cpu=1.0,
    timeout=7200,
    nonpreemptible=True,
)
def build_single_area(
    work_item: Dict,
    branch: str,
    version: str,
    calibration_inputs: Dict[str, str],
    validate: bool = True,
) -> Dict:
    """Build a single H5 file for one area.

    Each container processes exactly one work item (state, district,
    city, or national), validates the output, and writes to the
    staging volume.

    Args:
        work_item: {"type": "state|district|city|national", "id": "XX"}
        branch: Git branch (for repo setup).
        version: Package version string.
        calibration_inputs: Dict with weights, dataset, database paths
            and n_clones/seed.
        validate: Whether to run per-item validation.

    Returns:
        Dict with completed, failed, errors, validation_rows keys.
    """
    setup_gcp_credentials()
    setup_repo(branch)

    output_dir = Path(VOLUME_MOUNT) / version
    output_dir.mkdir(parents=True, exist_ok=True)

    work_items_json = json.dumps([work_item])

    repo_root = Path("/root/policyengine-us-data")
    cal_dir = repo_root / "policyengine_us_data" / "calibration"

    worker_cmd = [
        "python",
        "modal_app/worker_script.py",
        "--work-items",
        work_items_json,
        "--weights-path",
        calibration_inputs["weights"],
        "--dataset-path",
        calibration_inputs["dataset"],
        "--db-path",
        calibration_inputs["database"],
        "--output-dir",
        str(output_dir),
        "--target-config",
        str(cal_dir / "target_config.yaml"),
        "--validation-config",
        str(cal_dir / "target_config_full.yaml"),
    ]
    if "n_clones" in calibration_inputs:
        worker_cmd.extend(["--n-clones", str(calibration_inputs["n_clones"])])
    if "seed" in calibration_inputs:
        worker_cmd.extend(["--seed", str(calibration_inputs["seed"])])
    if not validate:
        worker_cmd.append("--no-validate")

    item_key = f"{work_item['type']}:{work_item['id']}"
    print(f"Building {item_key}...")

    result = subprocess.run(
        worker_cmd,
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        print(f"FAILED {item_key}: {result.stderr[:200]}")
        return {
            "completed": [],
            "failed": [item_key],
            "errors": [{"item": item_key, "error": result.stderr}],
            "validation_rows": [],
        }

    try:
        results = json.loads(result.stdout)
    except json.JSONDecodeError:
        results = {
            "completed": [],
            "failed": [item_key],
            "errors": [
                {
                    "item": item_key,
                    "error": f"Failed to parse output: {result.stdout[:200]}",
                }
            ],
            "validation_rows": [],
        }

    staging_volume.commit()
    print(f"Completed {item_key}")
    return results


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={
        VOLUME_MOUNT: staging_volume,
        "/pipeline": pipeline_volume,
    },
    memory=8192,
    cpu=1.0,
    timeout=86400,
    nonpreemptible=True,
)
def queue_coordinator(
    scope: str = "all",
    branch: str = "main",
    n_clones: int = 430,
    validate: bool = True,
    max_parallel: int = 50,
    run_id: str = "",
) -> Dict:
    """Queue-based coordinator for H5 builds.

    Generates work items based on scope, spawns up to max_parallel
    single-item workers, and collects results.

    Args:
        scope: Dataset scope (all/national/state/congressional/local/test).
        branch: Git branch.
        n_clones: Number of clones for calibration.
        validate: Whether to run per-item validation.
        max_parallel: Maximum concurrent worker containers.
        run_id: Optional run identifier.

    Returns:
        Summary dict with completed count, failed items, and
        validation results.
    """
    setup_gcp_credentials()
    setup_repo(branch)

    version = get_version()
    if not run_id:
        from policyengine_us_data.utils.run_id import generate_run_id

        sha = os.environ.get("GIT_COMMIT", "unknown")
        run_id = generate_run_id(version, sha)

    print("=" * 60)
    print(f"Queue Coordinator")
    print(f"  Run ID: {run_id}")
    print(f"  Scope:  {scope}")
    print(f"  Branch: {branch}")
    print("=" * 60)

    # Load pipeline artifacts
    pipeline_volume.reload()
    artifacts = Path("/pipeline/artifacts")
    weights_path = artifacts / "calibration_weights.npy"
    db_path = artifacts / "policy_data.db"
    dataset_path = artifacts / "source_imputed_stratified_extended_cps.h5"

    for label, p in [
        ("weights", weights_path),
        ("dataset", dataset_path),
        ("database", db_path),
    ]:
        if not p.exists():
            raise RuntimeError(
                f"Missing {label} on pipeline volume: {p}. "
                f"Run upstream pipeline steps first."
            )

    calibration_inputs = {
        "weights": str(weights_path),
        "dataset": str(dataset_path),
        "database": str(db_path),
        "n_clones": n_clones,
        "seed": 42,
    }

    # Generate work items
    items = generate_work_items(scope, str(db_path))
    print(f"Generated {len(items)} work items for scope '{scope}'")

    # Check for already-completed items on volume
    version_dir = Path(VOLUME_MOUNT) / version
    staging_volume.reload()
    completed = get_completed_from_volume(version_dir)
    remaining = [
        item for item in items if f"{item['type']}:{item['id']}" not in completed
    ]
    print(f"Already completed: {len(completed)}, remaining: {len(remaining)}")

    if not remaining:
        print("All items already built!")
        return {
            "run_id": run_id,
            "total": len(items),
            "completed": len(items),
            "failed": 0,
            "errors": [],
            "validation_rows": [],
        }

    # Spawn workers — one per item, up to max_parallel
    handles = []
    for item in remaining:
        handle = build_single_area.spawn(
            work_item=item,
            branch=branch,
            version=version,
            calibration_inputs=calibration_inputs,
            validate=validate,
        )
        handles.append((item, handle))
        if len(handles) % 10 == 0:
            print(f"  Spawned {len(handles)}/{len(remaining)} workers...")

    print(f"Spawned {len(handles)} workers (max_parallel={max_parallel})")

    # Collect results
    all_completed = list(completed)
    all_errors = []
    all_validation_rows = []

    for i, (item, handle) in enumerate(handles):
        item_key = f"{item['type']}:{item['id']}"
        try:
            result = handle.get()
            all_completed.extend(result.get("completed", []))
            all_errors.extend(result.get("errors", []))
            all_validation_rows.extend(result.get("validation_rows", []))
            status = "OK" if result.get("completed") else "FAILED"
            print(f"  [{i + 1}/{len(handles)}] {item_key}: {status}")
        except Exception as e:
            all_errors.append({"item": item_key, "error": str(e)})
            print(f"  [{i + 1}/{len(handles)}] {item_key}: CRASHED - {e}")

    total_completed = len(all_completed)
    total_failed = len(all_errors)

    print(f"\nQueue complete: {total_completed} completed, {total_failed} failed")

    return {
        "run_id": run_id,
        "scope": scope,
        "total": len(items),
        "completed": total_completed,
        "failed": total_failed,
        "errors": all_errors[:10],
        "validation_rows": all_validation_rows,
    }


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=4096,
    timeout=1800,
    nonpreemptible=True,
)
def validate_staging(branch: str, version: str, run_id: str = "") -> Dict:
    """Validate all expected files and generate manifest."""
    setup_repo(branch)

    result = subprocess.run(
        [
            "python",
            "-c",
            f"""
import json
from pathlib import Path
from policyengine_us_data.utils.manifest import generate_manifest, save_manifest

staging_dir = Path("{VOLUME_MOUNT}")
version = "{version}"
manifest = generate_manifest(staging_dir, version)
manifest["run_id"] = "{run_id}"
manifest_path = staging_dir / version / "manifest.json"
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
            "python",
            "-c",
            f"""
import json
from pathlib import Path
from policyengine_us_data.utils.manifest import verify_manifest
from policyengine_us_data.utils.data_upload import upload_to_staging_hf

manifest = json.loads('''{manifest_json}''')
version = "{version}"
staging_dir = Path("{VOLUME_MOUNT}")
version_dir = staging_dir / version

print("Verifying manifest before upload...")
verification = verify_manifest(staging_dir, manifest)
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
    local_path = version_dir / rel_path
    files_with_paths.append((local_path, rel_path))

# Upload to HuggingFace staging/
run_id = "{run_id}"
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

    staging_dir = Path(VOLUME_MOUNT)
    staging_volume.reload()

    manifest_path = staging_dir / version / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"No manifest found at {manifest_path}. Run build+stage workflow first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    if not run_id:
        run_id = manifest.get("run_id", "")

    rel_paths_json = json.dumps(list(manifest["files"].keys()))

    result = subprocess.run(
        [
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
version_dir = Path("{VOLUME_MOUNT}") / version

run_id = "{run_id}"
print(f"Promoting {{len(rel_paths)}} files from staging/ to production (run_id={{run_id!r}})...")
promoted = promote_staging_to_production_hf(rel_paths, version, run_id=run_id)
print(f"Promoted {{promoted}} files to HuggingFace production")

print(f"Uploading {{len(rel_paths)}} files to GCS...")
gcs_count = 0
for rel_path in rel_paths:
    local_path = version_dir / rel_path
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
    num_workers: int = 8,
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
    version_dir = staging_dir / version

    pipeline_volume.reload()
    artifacts = (
        Path(f"/pipeline/artifacts/{run_id}") if run_id else Path("/pipeline/artifacts")
    )
    weights_path = artifacts / "calibration_weights.npy"
    db_path = artifacts / "policy_data.db"
    dataset_path = artifacts / "source_imputed_stratified_extended_cps.h5"
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

    calibration_inputs = {
        "weights": str(weights_path),
        "dataset": str(dataset_path),
        "database": str(db_path),
        "n_clones": n_clones,
        "seed": 42,
    }
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
        fingerprint = expected_fingerprint
        print(f"Using pinned fingerprint from pipeline: {fingerprint}")
    else:
        fp_result = subprocess.run(
            [
                "python",
                "-c",
                f"""
from policyengine_us_data.calibration.publish_local_area import (
    compute_input_fingerprint,
)
print(compute_input_fingerprint("{weights_path}", "{dataset_path}", {n_clones}, seed=42))
""",
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if fp_result.returncode != 0:
            raise RuntimeError(f"Failed to compute fingerprint: {fp_result.stderr}")
        fingerprint = fp_result.stdout.strip()
    reconcile_action = reconcile_version_dir_fingerprint(version_dir, fingerprint)
    if reconcile_action == "resume":
        print(f"Inputs unchanged ({fingerprint}), resuming...")
    else:
        print(f"Prepared staging directory for fingerprint {fingerprint}")
    staging_volume.commit()
    result = subprocess.run(
        [
            "python",
            "-c",
            f"""
import json
from policyengine_us_data.calibration.calibration_utils import (
    get_all_cds_from_database,
    STATE_CODES,
)
from policyengine_us_data.calibration.publish_local_area import (
    get_district_friendly_name,
)

db_uri = "sqlite:///{db_path}"
cds = get_all_cds_from_database(db_uri)
states = list(STATE_CODES.values())
districts = [get_district_friendly_name(cd) for cd in cds]
print(json.dumps({{"states": states, "districts": districts, "cities": ["NYC"]}}))
""",
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to get work items: {result.stderr}")

    work_info = json.loads(result.stdout)
    states = work_info["states"]
    districts = work_info["districts"]
    cities = work_info["cities"]

    staging_volume.reload()
    completed = get_completed_from_volume(version_dir)
    print(f"Found {len(completed)} already-completed items on volume")

    phase_args = dict(
        num_workers=num_workers,
        branch=branch,
        version=version,
        calibration_inputs=calibration_inputs,
        version_dir=version_dir,
        validate=validate,
    )

    accumulated_errors = []
    accumulated_validation_rows = []

    completed, phase_errors, v_rows = run_phase(
        "States",
        states=states,
        districts=[],
        cities=[],
        completed=completed,
        **phase_args,
    )
    accumulated_errors.extend(phase_errors)
    accumulated_validation_rows.extend(v_rows)

    completed, phase_errors, v_rows = run_phase(
        "Districts",
        states=[],
        districts=districts,
        cities=[],
        completed=completed,
        **phase_args,
    )
    accumulated_errors.extend(phase_errors)
    accumulated_validation_rows.extend(v_rows)

    completed, phase_errors, v_rows = run_phase(
        "Cities",
        states=[],
        districts=[],
        cities=cities,
        completed=completed,
        **phase_args,
    )
    accumulated_errors.extend(phase_errors)
    accumulated_validation_rows.extend(v_rows)

    expected_total = len(states) + len(districts) + len(cities)

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
            "fingerprint": fingerprint,
        }

    print("\nValidating staging...")
    manifest = validate_staging.remote(branch=branch, version=version, run_id=run_id)

    expected_total = len(states) + len(districts) + len(cities)
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
        "fingerprint": fingerprint,
    }


@app.local_entrypoint()
def main(
    branch: str = "main",
    num_workers: int = 8,
    skip_upload: bool = False,
    n_clones: int = 430,
    run_id: str = "",
):
    """Local entrypoint for Modal CLI (legacy partition-based)."""
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


@app.local_entrypoint()
def main_queue(
    scope: str = "all",
    branch: str = "main",
    n_clones: int = 430,
    max_parallel: int = 50,
    run_id: str = "",
):
    """Queue-based entrypoint: one container per work item.

    Usage:
        modal run modal_app/local_area.py::main_queue --scope=test
        modal run modal_app/local_area.py::main_queue --scope=all --max-parallel=50
    """
    result = queue_coordinator.remote(
        scope=scope,
        branch=branch,
        n_clones=n_clones,
        max_parallel=max_parallel,
        run_id=run_id,
    )
    import json

    print(json.dumps(result, indent=2, default=str))


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

    calibration_inputs = {
        "weights": str(weights_path),
        "dataset": str(dataset_path),
        "database": str(db_path),
        "n_clones": n_clones,
        "seed": 42,
    }
    validate_artifacts(
        config_json_path,
        artifacts,
        filename_remap={
            "calibration_weights.npy": "national_calibration_weights.npy",
        },
    )
    version_dir = staging_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    work_items = [{"type": "national", "id": "US"}]
    print("Spawning worker for national H5 build...")
    worker_result = build_areas_worker.remote(
        branch=branch,
        version=version,
        work_items=work_items,
        calibration_inputs=calibration_inputs,
        validate=validate,
    )

    print(
        f"Worker result: "
        f"{len(worker_result['completed'])} completed, "
        f"{len(worker_result['failed'])} failed"
    )

    if worker_result["failed"]:
        raise RuntimeError(f"National build failed: {worker_result['errors']}")

    staging_volume.reload()
    national_h5 = version_dir / "national" / "US.h5"
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
    run_id: str = "",
) -> str:
    """Promote national US.h5 from HF staging to production + GCS."""
    setup_gcp_credentials()
    setup_repo(branch)

    version = get_version()
    rel_paths = ["national/US.h5"]

    result = subprocess.run(
        [
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
rel_paths = {json.dumps(rel_paths)}
version_dir = Path("{VOLUME_MOUNT}") / version

run_id = "{run_id}"
print(f"Promoting national H5 from staging to production (run_id={{run_id!r}})...")
promoted = promote_staging_to_production_hf(rel_paths, version, run_id=run_id)
print(f"Promoted {{promoted}} files to HuggingFace production")

national_h5 = version_dir / "national" / "US.h5"
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
    version: str = "",
    branch: str = "main",
    run_id: str = "",
):
    """Promote staged files to HuggingFace production."""
    if not version:
        raise ValueError("--version is required")
    result = promote_publish.remote(branch=branch, version=version, run_id=run_id)
    print(result)
