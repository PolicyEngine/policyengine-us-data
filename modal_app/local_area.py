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

import os
import subprocess
import json
import modal
from pathlib import Path
from typing import List, Dict

app = modal.App("policyengine-us-data-local-area")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

staging_volume = modal.Volume.from_name(
    "local-area-staging",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("uv", "tomli")
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"
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
    """Clone repo and install dependencies."""
    repo_dir = Path("/root/policyengine-us-data")

    if not repo_dir.exists():
        os.chdir("/root")
        subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
        os.chdir("policyengine-us-data")
        subprocess.run(["uv", "sync", "--locked"], check=True)
    else:
        os.chdir(repo_dir)


def get_version() -> str:
    """Get package version from pyproject.toml."""
    import tomli

    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["version"]


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


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=16384,
    cpu=4.0,
    timeout=14400,
)
def build_areas_worker(
    branch: str,
    version: str,
    work_items: List[Dict],
    calibration_inputs: Dict[str, str],
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

    result = subprocess.run(
        [
            "uv",
            "run",
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
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    print(result.stderr)

    if result.returncode != 0:
        return {
            "completed": [],
            "failed": [f"{item['type']}:{item['id']}" for item in work_items],
            "errors": [{"error": result.stderr}],
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


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=4096,
    timeout=1800,
)
def validate_staging(branch: str, version: str) -> Dict:
    """Validate all expected files and generate manifest."""
    setup_repo(branch)

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
version = "{version}"
manifest = generate_manifest(staging_dir, version)
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
    print(
        f"  Total size: {manifest['totals']['total_size_bytes'] / 1e9:.2f} GB"
    )

    return manifest


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=8192,
    timeout=14400,
)
def atomic_upload(branch: str, version: str, manifest: Dict) -> str:
    """
    Upload files using staging approach for atomic deployment.

    1. Upload to GCS (direct, overwrites existing)
    2. Upload to HuggingFace staging/ folder
    3. Atomically promote staging/ to production paths
    4. Clean up staging/
    """
    setup_gcp_credentials()
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
from policyengine_us_data.utils.data_upload import (
    upload_local_area_file,
    upload_to_staging_hf,
    promote_staging_to_production_hf,
    cleanup_staging_hf,
)

manifest = json.loads('''{manifest_json}''')
version = "{version}"
staging_dir = Path("{VOLUME_MOUNT}")
version_dir = staging_dir / version

print("Verifying manifest before upload...")
verification = verify_manifest(staging_dir, manifest)
if not verification["valid"]:
    raise ValueError(
        f"Manifest verification failed: "
        f"{{len(verification['missing'])}} missing, "
        f"{{len(verification['checksum_mismatch'])}} checksum mismatches"
    )
print(f"Verified {{verification['verified']}} files")

files_with_paths = []
rel_paths = []
for rel_path in manifest["files"].keys():
    local_path = version_dir / rel_path
    files_with_paths.append((local_path, rel_path))
    rel_paths.append(rel_path)

# Upload to GCS (direct to production paths)
print(f"Uploading {{len(files_with_paths)}} files to GCS...")
gcs_count = 0
for local_path, rel_path in files_with_paths:
    subdirectory = str(Path(rel_path).parent)
    upload_local_area_file(
        str(local_path),
        subdirectory,
        version=version,
        skip_hf=True,
    )
    gcs_count += 1
print(f"Uploaded {{gcs_count}} files to GCS")

# Upload to HuggingFace staging/
print(f"Uploading {{len(files_with_paths)}} files to HuggingFace staging/...")
hf_count = upload_to_staging_hf(files_with_paths, version)
print(f"Uploaded {{hf_count}} files to HuggingFace staging/")

# Atomically promote staging to production
print("Promoting staging/ to production (atomic commit)...")
promoted = promote_staging_to_production_hf(rel_paths, version)
print(f"Promoted {{promoted}} files to production")

# Clean up staging
print("Cleaning up staging/...")
cleaned = cleanup_staging_hf(rel_paths, version)
print(f"Cleaned up {{cleaned}} files from staging/")

print(f"Successfully published version {{version}}")
""",
        ],
        text=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Upload failed: {result.stderr}")

    return f"Successfully published version {version} with {len(manifest['files'])} files"


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=8192,
    timeout=86400,
)
def coordinate_publish(
    branch: str = "main",
    num_workers: int = 8,
    skip_upload: bool = False,
) -> str:
    """Coordinate the full publishing workflow."""
    setup_gcp_credentials()
    setup_repo(branch)

    version = get_version()
    print(f"Publishing version {version} from branch {branch}")
    print(f"Using {num_workers} parallel workers")

    staging_dir = Path(VOLUME_MOUNT)
    version_dir = staging_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    calibration_dir = staging_dir / "calibration_inputs"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    # hf_hub_download preserves directory structure, so files are in calibration/ subdir
    weights_path = (
        calibration_dir / "calibration" / "w_district_calibration.npy"
    )
    dataset_path = (
        calibration_dir / "calibration" / "stratified_extended_cps.h5"
    )
    db_path = calibration_dir / "calibration" / "policy_data.db"

    if not all(p.exists() for p in [weights_path, dataset_path, db_path]):
        print("Downloading calibration inputs...")
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                f"""
from policyengine_us_data.utils.huggingface import download_calibration_inputs
download_calibration_inputs("{calibration_dir}")
print("Done")
""",
            ],
            text=True,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")
        staging_volume.commit()
        print("Calibration inputs downloaded and cached on volume")
    else:
        print("Using cached calibration inputs from volume")

    calibration_inputs = {
        "weights": str(weights_path),
        "dataset": str(dataset_path),
        "database": str(db_path),
    }

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"""
import json
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_all_cds_from_database,
    STATE_CODES,
)
from policyengine_us_data.datasets.cps.local_area_calibration.publish_local_area import (
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

    work_chunks = partition_work(
        states, districts, cities, num_workers, completed
    )

    total_remaining = sum(len(c) for c in work_chunks)
    print(
        f"Remaining work: {total_remaining} items "
        f"across {len(work_chunks)} workers"
    )

    if total_remaining == 0:
        print("All items already built!")
    else:
        print("\nSpawning workers...")
        handles = []
        for i, chunk in enumerate(work_chunks):
            print(f"  Worker {i}: {len(chunk)} items")
            handle = build_areas_worker.spawn(
                branch=branch,
                version=version,
                work_items=chunk,
                calibration_inputs=calibration_inputs,
            )
            handles.append(handle)

        print("\nWaiting for workers to complete...")
        all_results = []
        all_errors = []

        for i, handle in enumerate(handles):
            try:
                result = handle.get()
                all_results.append(result)
                print(
                    f"  Worker {i}: {len(result['completed'])} completed, "
                    f"{len(result['failed'])} failed"
                )
                if result["errors"]:
                    all_errors.extend(result["errors"])
            except Exception as e:
                all_errors.append({"worker": i, "error": str(e)})
                print(f"  Worker {i}: CRASHED - {e}")

        total_completed = sum(len(r["completed"]) for r in all_results)
        total_failed = sum(len(r["failed"]) for r in all_results)

        print(f"\nBuild summary:")
        print(f"  Completed: {total_completed}")
        print(f"  Failed: {total_failed}")
        print(f"  Previously completed: {len(completed)}")

        if all_errors:
            print(f"\nErrors ({len(all_errors)}):")
            for err in all_errors[:5]:
                err_msg = err.get("error", "Unknown")[:100]
                print(f"  - {err.get('item', err.get('worker'))}: {err_msg}")
            if len(all_errors) > 5:
                print(f"  ... and {len(all_errors) - 5} more")

        if total_failed > 0:
            raise RuntimeError(
                f"Build incomplete: {total_failed} failures. "
                f"Volume preserved for retry."
            )

    if skip_upload:
        print("\nSkipping upload (--skip-upload flag set)")
        return f"Build complete for version {version}. Upload skipped."

    print("\nValidating staging...")
    manifest = validate_staging.remote(branch=branch, version=version)

    expected_total = len(states) + len(districts) + len(cities)
    actual_total = (
        manifest["totals"]["states"]
        + manifest["totals"]["districts"]
        + manifest["totals"]["cities"]
    )

    if actual_total < expected_total:
        print(
            f"WARNING: Expected {expected_total} files, found {actual_total}"
        )

    print("\nStarting atomic upload...")
    result = atomic_upload.remote(
        branch=branch, version=version, manifest=manifest
    )

    return result


@app.local_entrypoint()
def main(
    branch: str = "main",
    num_workers: int = 8,
    skip_upload: bool = False,
):
    """Local entrypoint for Modal CLI."""
    result = coordinate_publish.remote(
        branch=branch,
        num_workers=num_workers,
        skip_upload=skip_upload,
    )
    print(result)
