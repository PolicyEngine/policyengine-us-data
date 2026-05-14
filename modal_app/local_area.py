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
from typing import Dict, List, Mapping, Sequence

import modal
import numpy as np

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image as image  # noqa: E402
from modal_app.resilience import reconcile_run_dir_fingerprint  # noqa: E402
from policyengine_us_data.build_outputs.bootstrap import (  # noqa: E402
    WorkerBootstrapBuilder,
)
from policyengine_us_data.build_outputs.area_catalog import USAreaCatalog  # noqa: E402
from policyengine_us_data.build_outputs.fingerprinting import (  # noqa: E402
    FingerprintingService,
    PublishingInputBundle,
)
from policyengine_us_data.build_outputs.geography_loader import (  # noqa: E402
    CalibrationGeographyLoader,
)
from policyengine_us_data.build_outputs.partitioning import (  # noqa: E402
    WeightedAreaRequest,
    partition_weighted_area_requests,
    partition_weighted_work_items,
)
from policyengine_us_data.build_outputs.worker_responses import (  # noqa: E402
    normalize_worker_response,
)
from policyengine_us_data.build_outputs.worker_inputs import (  # noqa: E402
    WorkerCalibrationInputs,
)
from policyengine_us_data.pipeline_metadata import pipeline_node  # noqa: E402
from policyengine_us_data.pipeline_schema import PipelineNode  # noqa: E402
from policyengine_us_data.utils.run_context import (  # noqa: E402
    resolve_candidate_version,
    resolve_run_id,
)

app = modal.App(
    os.environ.get("US_DATA_LOCAL_AREA_APP_NAME") or "policyengine-us-data-local-area"
)

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

staging_volume = modal.Volume.from_name(
    os.environ.get("US_DATA_STAGING_VOLUME_NAME", "local-area-staging"),
    create_if_missing=True,
)

pipeline_volume = modal.Volume.from_name(
    os.environ.get("US_DATA_PIPELINE_VOLUME_NAME", "pipeline-artifacts"),
    create_if_missing=True,
    version=2,
)

VOLUME_MOUNT = "/staging"


def _python_cmd(*args: str) -> list[str]:
    """Build a command that uses the current interpreter."""
    return [sys.executable, *args]


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


def _build_promote_national_publish_script(
    *,
    version: str,
    run_id: str,
    rel_paths: list[str],
    cleanup_staging: bool = True,
) -> str:
    rel_paths_json = json.dumps(rel_paths)
    cleanup_staging_json = json.dumps(cleanup_staging)
    return f"""
import json
import os
from pathlib import Path
from policyengine_us_data.utils.data_upload import (
    promote_staging_to_production_hf,
    cleanup_staging_hf,
    upload_local_area_file,
    publish_release_manifest_to_hf,
    preflight_release_manifest_publish,
)
from policyengine_us_data.utils.version_manifest import (
    HFVersionInfo,
    build_manifest,
    upload_manifest,
)

version = "{version}"
run_id = "{run_id}"
os.environ["US_DATA_RUN_ID"] = run_id
rel_paths = json.loads('''{rel_paths_json}''')
cleanup_staging = json.loads('''{cleanup_staging_json}''')
run_dir = Path("{VOLUME_MOUNT}") / run_id
national_h5 = run_dir / "national" / "US.h5"
if not national_h5.exists():
    raise RuntimeError(f"Expected national H5 at {{national_h5}}")

print("Preflighting release manifest...")
should_finalize, missing_prefixes = preflight_release_manifest_publish(
    [(national_h5, "national/US.h5")],
    version=version,
    new_repo_paths=["national/US.h5"],
    pipeline_run_id=run_id,
)

print(f"Promoting national H5 from staging to production (run_id={{run_id!r}})...")
promoted = promote_staging_to_production_hf(rel_paths, version, run_id=run_id)
print(f"Promoted {{promoted}} files to HuggingFace production")

print("Uploading national H5 to GCS...")
upload_local_area_file(
    str(national_h5), "national", version=version, skip_hf=True
)
print("Uploaded national H5 to GCS")

print("Updating release manifest...")
manifest = publish_release_manifest_to_hf(
    [(national_h5, "national/US.h5")],
    version=version,
    create_tag=should_finalize,
    pipeline_run_id=run_id,
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
            run_id=run_id or None,
        )
    )
    print("Updated release manifest and created tag")
else:
    print(
        "Updated release manifest without creating a tag; "
        f"missing prefixes: {{', '.join(missing_prefixes)}}"
    )

if cleanup_staging:
    print("Cleaning up staging...")
    cleaned = cleanup_staging_hf(rel_paths, version, run_id=run_id)
    print(f"Cleaned up {{cleaned}} files from staging")
else:
    print("Deferring staged national cleanup until full release promotion succeeds")
print(f"Successfully promoted national H5 for version {{version}}")
"""


def _build_promote_publish_script(
    *,
    version: str,
    run_id: str,
    rel_paths: list[str],
    cleanup_staging: bool = True,
) -> str:
    rel_paths_json = json.dumps(rel_paths)
    cleanup_staging_json = json.dumps(cleanup_staging)
    return f"""
import json
import os
from pathlib import Path
from policyengine_us_data.utils.data_upload import (
    promote_staging_to_production_hf,
    cleanup_staging_hf,
    upload_local_area_file,
    publish_release_manifest_to_hf,
    preflight_release_manifest_publish,
)
from policyengine_us_data.utils.version_manifest import (
    HFVersionInfo,
    build_manifest,
    upload_manifest,
)

rel_paths = json.loads('''{rel_paths_json}''')
version = "{version}"
run_id = "{run_id}"
os.environ["US_DATA_RUN_ID"] = run_id
cleanup_staging = json.loads('''{cleanup_staging_json}''')
run_dir = Path("{VOLUME_MOUNT}") / run_id
manifest_files = [(run_dir / rel_path, rel_path) for rel_path in rel_paths]
missing_local_paths = [str(path) for path, _ in manifest_files if not path.exists()]
if missing_local_paths:
    raise RuntimeError(
        "Expected local-area artifacts before promotion: "
        + ", ".join(missing_local_paths)
    )

print("Preflighting release manifest...")
should_finalize, missing_prefixes = preflight_release_manifest_publish(
    manifest_files,
    version=version,
    new_repo_paths=rel_paths,
    pipeline_run_id=run_id,
)

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

print("Updating release manifest...")
manifest = publish_release_manifest_to_hf(
    manifest_files,
    version=version,
    create_tag=should_finalize,
    pipeline_run_id=run_id,
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
            run_id=run_id or None,
        )
    )
    print("Updated release manifest and created tag")
else:
    print(
        "Updated release manifest without final tag; missing local-area prefixes: "
        + ", ".join(missing_prefixes)
    )
    print("Deferring version_manifest.json update until release finalization")

if cleanup_staging:
    print("Cleaning up staging/...")
    cleaned = cleanup_staging_hf(rel_paths, version, run_id=run_id)
    print(f"Cleaned up {{cleaned}} files from staging/")
else:
    print("Deferring staged regional cleanup until full release promotion succeeds")

print(f"Successfully published version {{version}}")
"""


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


def get_staging_candidate_version(
    fallback_version: str,
    explicit_candidate_version: str = "",
) -> str:
    """Resolve the HF staging candidate scope for local-area artifacts."""
    return (
        resolve_candidate_version(explicit_candidate_version, env=os.environ)
        or fallback_version
    )


@pipeline_node(
    PipelineNode(
        id="build_publishing_input_bundle",
        label="Build Publishing Input Bundle",
        node_type="library",
        description="Assemble artifact paths and run metadata for scope fingerprinting.",
        source_file="modal_app/local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        api_refs=[
            "policyengine_us_data.build_outputs.fingerprinting.PublishingInputBundle"
        ],
        validation_commands=["uv run pytest tests/unit/test_modal_local_area.py"],
    )
)
def _build_publishing_input_bundle(
    *,
    weights_path: Path,
    dataset_path: Path,
    db_path: Path | None,
    geography_path: Path | None,
    calibration_package_path: Path | None,
    run_config_path: Path | None,
    run_id: str,
    version: str,
    n_clones: int | None,
    seed: int,
    legacy_blocks_path: Path | None = None,
) -> PublishingInputBundle:
    """Build the normalized coordinator input bundle for one publish scope."""

    return PublishingInputBundle(
        weights_path=weights_path,
        source_dataset_path=dataset_path,
        target_db_path=db_path,
        exact_geography_path=geography_path,
        calibration_package_path=calibration_package_path,
        run_config_path=run_config_path,
        run_id=run_id,
        version=version,
        n_clones=n_clones,
        seed=seed,
        legacy_blocks_path=legacy_blocks_path,
    )


@pipeline_node(
    PipelineNode(
        id="resolve_scope_fingerprint",
        label="Resolve Scope Fingerprint",
        node_type="library",
        description="Compute the regional or national local H5 fingerprint from publishing inputs.",
        source_file="modal_app/local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        api_refs=[
            "policyengine_us_data.build_outputs.fingerprinting.FingerprintingService"
        ],
        validation_commands=["uv run pytest tests/unit/test_modal_local_area.py"],
    )
)
def _resolve_scope_fingerprint(
    *,
    inputs: PublishingInputBundle,
    scope: str,
    expected_fingerprint: str = "",
) -> str:
    """Compute the scope fingerprint while preserving pinned resume values."""

    service = FingerprintingService()
    traceability = service.build_traceability(inputs=inputs, scope=scope)
    computed_fingerprint = service.compute_scope_fingerprint(traceability)
    if expected_fingerprint:
        if expected_fingerprint != computed_fingerprint:
            print(
                "WARNING: Pinned fingerprint differs from current "
                f"{scope} scope fingerprint. "
                "Preserving pinned value for backward-compatible resume.\n"
                f"  Pinned:   {expected_fingerprint}\n"
                f"  Current:  {computed_fingerprint}"
            )
        else:
            print(f"Using pinned fingerprint from pipeline: {expected_fingerprint}")
        return expected_fingerprint
    return computed_fingerprint


@pipeline_node(
    PipelineNode(
        id="build_worker_bootstrap",
        label="Build Worker Bootstrap",
        node_type="library",
        description="Persist deterministic local H5 worker setup artifacts for one scope.",
        source_file="modal_app/local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        api_refs=[
            "policyengine_us_data.build_outputs.bootstrap.WorkerBootstrapBuilder"
        ],
        artifacts_out=[
            "bootstrap/{scope}/worker_bootstrap.json",
            "bootstrap/{scope}/entity_graph.npz",
        ],
        validation_commands=["uv run pytest tests/unit/test_modal_local_area.py"],
    )
)
def _build_worker_bootstrap(
    *,
    inputs: PublishingInputBundle,
    scope: str,
    artifacts_dir: Path,
    scope_fingerprint: str | None = None,
):
    """Persist optional worker bootstrap artifacts for one local H5 scope."""

    bundle = WorkerBootstrapBuilder().build(
        inputs=inputs,
        scope=scope,
        artifacts_dir=artifacts_dir,
        scope_fingerprint=scope_fingerprint,
    )
    print(
        f"Worker bootstrap ready for {scope}: "
        f"{bundle.manifest_path.relative_to(artifacts_dir)}"
    )
    return bundle


def _build_worker_calibration_inputs(
    *,
    weights_path: Path,
    geography_path: Path,
    dataset_path: Path,
    db_path: Path,
    n_clones: int,
    seed: int,
    run_config_path: Path | None = None,
    calibration_package_path: Path | None = None,
) -> WorkerCalibrationInputs:
    """Build the normalized H5 worker input payload."""

    return WorkerCalibrationInputs.from_artifact_paths(
        weights_path=weights_path,
        geography_path=geography_path,
        dataset_path=dataset_path,
        database_path=db_path,
        n_clones=n_clones,
        seed=seed,
        run_config_path=run_config_path,
        calibration_package_path=calibration_package_path,
    )


def _existing_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if Path(path).exists() else None


def _infer_weight_record_count(*, weights_path: Path, n_clones: int) -> int:
    """Infer source-record count from a flat weight vector without loading it."""

    if isinstance(n_clones, bool) or not isinstance(n_clones, int | np.integer):
        raise TypeError("n_clones must be an integer")
    normalized_clones = int(n_clones)
    if normalized_clones <= 0:
        raise ValueError("n_clones must be positive")

    weights = np.load(weights_path, mmap_mode="r")
    if weights.ndim != 1:
        raise ValueError("Weight vector must be one-dimensional")
    if weights.size == 0:
        raise ValueError("Weight vector must be non-empty")
    if not np.issubdtype(weights.dtype, np.number):
        raise TypeError("Weight vector must have a numeric dtype")
    if np.issubdtype(weights.dtype, np.complexfloating):
        raise TypeError("Weight vector must have a real numeric dtype")
    if weights.size % normalized_clones != 0:
        raise ValueError(
            f"Weight vector length {weights.size} is not divisible by "
            f"n_clones={normalized_clones}"
        )
    return weights.size // normalized_clones


def _load_area_catalog_geography(
    *,
    weights_path: Path,
    n_clones: int,
    geography_path: Path | None,
    calibration_package_path: Path | None = None,
    legacy_blocks_path: Path | None = None,
):
    """Load geography once for coordinator-side typed request construction."""

    n_records = _infer_weight_record_count(
        weights_path=weights_path,
        n_clones=n_clones,
    )
    return CalibrationGeographyLoader().load(
        weights_path=weights_path,
        n_records=n_records,
        n_clones=n_clones,
        geography_path=_existing_path(geography_path),
        blocks_path=_existing_path(legacy_blocks_path),
        calibration_package_path=_existing_path(calibration_package_path),
    )


def _load_target_cd_geoids(db_path: Path) -> tuple[str, ...]:
    """Load the congressional district target universe for regional H5s."""

    from policyengine_us_data.calibration.calibration_utils import (
        get_all_cds_from_database,
    )

    return tuple(
        str(cd_geoid) for cd_geoid in get_all_cds_from_database(f"sqlite:///{db_path}")
    )


def _build_regional_weighted_requests(
    *,
    geography,
    target_cd_geoids: Sequence[str],
    catalog: USAreaCatalog | None = None,
) -> tuple[WeightedAreaRequest, ...]:
    """Build canonical weighted regional H5 requests from release targets."""

    catalog = catalog or USAreaCatalog.default()
    requests = catalog.build_expected_regional_requests(
        target_cd_geoids=target_cd_geoids,
        geography=geography,
    )

    from collections import Counter

    districts_by_state = Counter(
        request.area_id.split("-", 1)[0]
        for request in requests
        if request.area_type == "district"
    )
    city_weights = {"NYC": 11}

    weighted: list[WeightedAreaRequest] = []
    for request in requests:
        if request.area_type == "state":
            weight = districts_by_state.get(request.area_id, 1)
        elif request.area_type == "city":
            weight = city_weights.get(request.area_id, 3)
        else:
            weight = 1
        weighted.append(
            WeightedAreaRequest(
                request=request,
                weight=weight,
            )
        )
    return tuple(weighted)


def _build_weighted_requests_from_work_items(
    *,
    work_items: Sequence[Mapping[str, object]],
    geography,
    catalog: USAreaCatalog | None = None,
) -> tuple[WeightedAreaRequest, ...]:
    """Convert legacy override work items into canonical weighted requests."""

    catalog = catalog or USAreaCatalog.default()
    weighted: list[WeightedAreaRequest] = []
    for item in work_items:
        request = catalog.build_request_from_work_item(item, geography=geography)
        if request is None:
            continue
        weighted.append(
            WeightedAreaRequest(
                request=request,
                weight=item.get("weight", 1),
            )
        )
    return tuple(weighted)


@pipeline_node(
    PipelineNode(
        id="coordinate_work_partition",
        label="Coordinate Work Partition",
        node_type="library",
        description="Compatibility wrapper for local H5 weighted work partitioning.",
        source_file="modal_app/local_area.py",
        status="legacy",
        stability="moving",
        pathways=["local_h5"],
        api_refs=[
            "policyengine_us_data.build_outputs.partitioning.partition_weighted_work_items"
        ],
        validation_commands=["uv run pytest tests/unit/test_modal_local_area.py"],
    )
)
def partition_work(
    work_items: List[Dict],
    num_workers: int,
    completed: set,
) -> List[List[Dict]]:
    """Compatibility wrapper over the extracted pure partitioning seam."""
    return partition_weighted_work_items(
        work_items=work_items,
        num_workers=num_workers,
        completed=completed,
    )


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


def _measure_expected_completion(
    *,
    expected_keys: set[str],
    initially_completed: set[str],
    completed: set[str],
) -> tuple[set[str], dict[str, int]]:
    """Measure completion against the explicit expected request set."""

    missing_keys = expected_keys - completed
    reused_outputs = initially_completed & completed & expected_keys
    recomputed_outputs = (completed - initially_completed) & expected_keys
    return missing_keys, {
        "expected_outputs": len(expected_keys),
        "valid_reused_outputs": len(reused_outputs),
        "recomputed_outputs": len(recomputed_outputs),
        "invalid_outputs": len(missing_keys),
    }


@pipeline_node(
    PipelineNode(
        id="run_local_h5_phase",
        label="Run Local H5 Worker Phase",
        node_type="entrypoint",
        description="Spawn local H5 workers for one partitioned build phase and record completed outputs.",
        source_file="modal_app/local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        artifacts_out=["staged phase outputs"],
        validation_commands=["uv run pytest tests/unit/test_modal_local_area.py"],
    )
)
def run_phase(
    phase_name: str,
    weighted_requests: Sequence[WeightedAreaRequest] | None,
    num_workers: int,
    completed: set,
    branch: str,
    run_id: str,
    calibration_inputs: WorkerCalibrationInputs | Mapping[str, object],
    run_dir: Path,
    validate: bool = True,
    scope_fingerprint: str | None = None,
    work_items: List[Dict] | None = None,
) -> tuple:
    """Run a single build phase, spawning workers and collecting results.

    Returns:
        A tuple of (volume_completed, phase_errors, validation_rows)
        where phase_errors is a list of error dicts from workers
        and crashes, and validation_rows is a list of per-target
        validation result dicts.
    """
    if weighted_requests is not None:
        work_chunks = partition_weighted_area_requests(
            weighted_requests,
            num_workers,
            completed,
        )
    else:
        work_chunks = partition_work(work_items or [], num_workers, completed)
    total_remaining = sum(len(c) for c in work_chunks)
    worker_input_payload = WorkerCalibrationInputs.from_wire_dict(
        calibration_inputs
    ).to_wire_dict()

    print(f"\n--- Phase: {phase_name} ---")
    print(f"Remaining work: {total_remaining} items across {len(work_chunks)} workers")

    if total_remaining == 0:
        print(f"All {phase_name} items already built!")
        return completed, [], []

    handles = []
    for i, chunk in enumerate(work_chunks):
        if weighted_requests is not None:
            total_weight = sum(item.weight for item in chunk)
            request_payloads = [item.to_worker_payload() for item in chunk]
            legacy_work_items = None
        else:
            total_weight = sum(item["weight"] for item in chunk)
            request_payloads = None
            legacy_work_items = chunk
        print(f"  Worker {i}: {len(chunk)} items, weight {total_weight}")
        handle = build_areas_worker.spawn(
            branch=branch,
            run_id=run_id,
            scope="regional",
            work_items=legacy_work_items,
            request_payloads=request_payloads,
            calibration_inputs=worker_input_payload,
            validate=validate,
            scope_fingerprint=scope_fingerprint,
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
            worker_result = normalize_worker_response(
                worker_index=i,
                result=result,
            )
            all_results.append(worker_result)
            print(
                f"  Worker {i}: {len(worker_result.completed)} completed, "
                f"{len(worker_result.failed)} failed"
            )
            if worker_result.fatal_errors:
                all_errors.extend(worker_result.fatal_errors)
            if worker_result.issues:
                all_errors.extend(worker_result.issues)
            if worker_result.validation_rows:
                all_validation_rows.extend(worker_result.validation_rows)
                print(
                    f"  Worker {i}: {len(worker_result.validation_rows)} validation rows"
                )
        except Exception as e:
            all_errors.append(
                {
                    "worker": i,
                    "phase": "transport",
                    "severity": "transport",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  Worker {i}: CRASHED - {e}")

    total_completed = sum(len(result.completed) for result in all_results)
    total_failed = sum(len(result.failed) for result in all_results)

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

    return volume_completed, all_errors, all_validation_rows


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
@pipeline_node(
    PipelineNode(
        id="build_areas_worker",
        label="Build Areas Worker",
        node_type="entrypoint",
        description="Modal worker entrypoint for state, district, city, or typed local H5 requests.",
        source_file="modal_app/local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        artifacts_out=["one or more H5 files"],
        validation_commands=[
            "uv run pytest tests/unit/test_modal_local_area.py",
            "uv run pytest tests/integration/build_outputs/h5_worker_runtime/test_worker_script_tiny_fixture.py",
        ],
    )
)
def build_areas_worker(
    branch: str,
    run_id: str,
    scope: str,
    work_items: List[Dict] | None = None,
    calibration_inputs: WorkerCalibrationInputs | Mapping[str, object] | None = None,
    validate: bool = True,
    scope_fingerprint: str | None = None,
    request_payloads: List[Dict] | None = None,
) -> Dict:
    """
    Worker function that builds a subset of H5 files.
    Uses subprocess to avoid import conflicts with Modal's environment.
    """
    setup_gcp_credentials()
    setup_repo(branch)
    pipeline_volume.reload()
    staging_volume.reload()

    output_dir = Path(VOLUME_MOUNT) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if calibration_inputs is None:
        raise ValueError("calibration_inputs must be provided")
    worker_inputs = WorkerCalibrationInputs.from_wire_dict(calibration_inputs)
    if request_payloads is not None:
        request_args = ["--requests-json", json.dumps(request_payloads)]
        failed_items = [
            f"{item.get('area_type', '<missing-type>')}:"
            f"{item.get('area_id', '<missing-id>')}"
            for item in request_payloads
        ]
    elif work_items is not None:
        request_args = ["--work-items", json.dumps(work_items)]
        failed_items = [f"{item['type']}:{item['id']}" for item in work_items]
    else:
        raise ValueError("Either request_payloads or work_items must be provided")

    worker_cmd = [
        *_python_cmd("-m", "modal_app.worker_script"),
        *request_args,
        *worker_inputs.to_worker_cli_args(),
        "--output-dir",
        str(output_dir),
        "--scope",
        scope,
        "--run-id",
        run_id,
        "--artifacts-dir",
        str(Path("/pipeline/artifacts") / run_id),
    ]
    if scope_fingerprint:
        worker_cmd.extend(["--scope-fingerprint", scope_fingerprint])
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

    if result.stderr:
        print(f"Worker stderr:\n{result.stderr}", file=__import__("sys").stderr)

    if result.returncode != 0:
        return {
            "completed": [],
            "failed": failed_items,
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


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={VOLUME_MOUNT: staging_volume},
    memory=4096,
    timeout=1800,
    nonpreemptible=True,
)
@pipeline_node(
    PipelineNode(
        id="validate_staging",
        label="Validate Staged H5 Files",
        node_type="validation",
        description="Run staged H5 validation before upload and promotion.",
        source_file="modal_app/local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        validation_commands=["uv run pytest tests/unit/test_modal_local_area.py"],
    )
)
def validate_staging(branch: str, run_id: str, version: str = "") -> Dict:
    """Validate all expected files and generate manifest."""
    setup_repo(branch)
    staging_volume.reload()

    if not version:
        version = run_id.split("_", 1)[0]

    result = subprocess.run(
        _python_cmd(
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
        ),
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
@pipeline_node(
    PipelineNode(
        id="staging_upload",
        label="Upload Local H5s To Staging",
        node_type="entrypoint",
        description="Upload completed local H5 outputs to staging storage before promotion.",
        source_file="modal_app/local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        artifacts_in=["staged local-area H5 files"],
        validation_commands=["uv run pytest tests/unit/test_modal_local_area.py"],
    )
)
def upload_to_staging(
    branch: str,
    version: str,
    manifest: Dict,
    run_id: str = "",
    candidate_version: str = "",
) -> str:
    """
    Upload files to HuggingFace staging only.

    GCS is updated during promote_publish, not here.
    Promote must be run separately via promote_publish.
    """
    setup_repo(branch)

    manifest_json = json.dumps(manifest)
    staging_candidate_version = get_staging_candidate_version(
        version,
        explicit_candidate_version=candidate_version,
    )

    result = subprocess.run(
        _python_cmd(
            "-c",
            f"""
import json
from pathlib import Path
from policyengine_us_data.utils.manifest import verify_manifest
from policyengine_us_data.utils.data_upload import upload_to_staging_hf

manifest = json.loads('''{manifest_json}''')
version = "{version}"
staging_candidate_version = "{staging_candidate_version}"
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
hf_count = upload_to_staging_hf(
    files_with_paths,
    candidate_version=staging_candidate_version,
    run_id=run_id,
)
print(f"Uploaded {{hf_count}} files to HuggingFace staging/")

print(f"Staged candidate {{staging_candidate_version}} for promotion")
""",
        ),
        text=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Upload failed: {result.stderr}")

    return (
        f"Staged candidate {staging_candidate_version} with "
        f"{len(manifest['files'])} files. "
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
def promote_publish(
    branch: str = "main",
    version: str = "",
    run_id: str = "",
    cleanup_staging: bool = True,
) -> str:
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
        version = get_version()

    staging_dir = Path(VOLUME_MOUNT)
    staging_volume.reload()

    manifest_path = staging_dir / run_id / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"No manifest found at {manifest_path}. Run build+stage workflow first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    result = subprocess.run(
        _python_cmd(
            "-c",
            _build_promote_publish_script(
                version=version,
                run_id=run_id,
                rel_paths=list(manifest["files"].keys()),
                cleanup_staging=cleanup_staging,
            ),
        ),
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
@pipeline_node(
    PipelineNode(
        id="coordinate_publish",
        label="Coordinate Local H5 Publish",
        node_type="entrypoint",
        description="Coordinate local H5 partitioning, worker phases, validation, staging, and promotion.",
        source_file="modal_app/local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5", "orchestration"],
        artifacts_in=[
            "calibration_weights.npy",
            "source_imputed_stratified_extended_cps*.h5",
        ],
        artifacts_out=["staged local-area H5 files"],
        validation_commands=["uv run pytest tests/unit/test_modal_local_area.py"],
    )
)
def coordinate_publish(
    branch: str = "main",
    num_workers: int = 50,
    skip_upload: bool = False,
    n_clones: int = 430,
    validate: bool = True,
    run_id: str = "",
    candidate_version: str = "",
    expected_fingerprint: str = "",
    work_items_override: List[Dict] | None = None,
) -> Dict:
    """Coordinate the full publishing workflow."""
    setup_gcp_credentials()
    setup_repo(branch)

    version = get_version()
    staging_candidate_version = get_staging_candidate_version(
        version,
        explicit_candidate_version=candidate_version,
    )

    run_id = run_id or resolve_run_id()
    if not run_id:
        raise RuntimeError(
            "run_id is required. Local-area publishing must receive the "
            "GitHub-created run ID from the pipeline."
        )

    print("=" * 60)
    print(f"Run ID: {run_id}")
    print("=" * 60)
    print(f"Publishing version {version} from branch {branch}")
    print(f"Staging candidate {staging_candidate_version}")
    print(f"Using {num_workers} parallel workers")

    staging_dir = Path(VOLUME_MOUNT)
    run_dir = staging_dir / run_id

    pipeline_volume.reload()
    artifacts = (
        Path(f"/pipeline/artifacts/{run_id}") if run_id else Path("/pipeline/artifacts")
    )
    weights_path = artifacts / "calibration_weights.npy"
    geography_path = artifacts / "geography_assignment.npz"
    db_path = artifacts / "policy_data.db"
    dataset_path = artifacts / "source_imputed_stratified_extended_cps.h5"
    config_json_path = artifacts / "unified_run_config.json"
    calibration_package_path = artifacts / "calibration_package.pkl"

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

    calibration_inputs = _build_worker_calibration_inputs(
        weights_path=weights_path,
        geography_path=geography_path,
        dataset_path=dataset_path,
        db_path=db_path,
        n_clones=n_clones,
        seed=42,
        run_config_path=config_json_path,
        calibration_package_path=calibration_package_path,
    )
    validate_artifacts(config_json_path, artifacts)
    regional_geography = _load_area_catalog_geography(
        weights_path=weights_path,
        n_clones=n_clones,
        geography_path=geography_path,
        calibration_package_path=calibration_package_path,
        legacy_blocks_path=artifacts / "stacked_blocks.npy",
    )

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
    fingerprint_inputs = _build_publishing_input_bundle(
        weights_path=weights_path,
        dataset_path=dataset_path,
        db_path=db_path,
        geography_path=geography_path,
        calibration_package_path=(
            calibration_package_path if calibration_package_path.exists() else None
        ),
        run_config_path=config_json_path if config_json_path.exists() else None,
        run_id=run_id,
        version=version,
        n_clones=n_clones,
        seed=42,
        legacy_blocks_path=artifacts / "stacked_blocks.npy",
    )
    fingerprint = _resolve_scope_fingerprint(
        inputs=fingerprint_inputs,
        scope="regional",
        expected_fingerprint=expected_fingerprint,
    )
    reconcile_action = reconcile_run_dir_fingerprint(run_dir, fingerprint)
    if reconcile_action == "resume":
        print(f"Inputs unchanged ({fingerprint}), resuming...")
    else:
        print(f"Prepared staging directory for fingerprint {fingerprint}")
    _build_worker_bootstrap(
        inputs=fingerprint_inputs,
        scope="regional",
        artifacts_dir=artifacts,
        scope_fingerprint=fingerprint,
    )
    pipeline_volume.commit()
    staging_volume.commit()
    if work_items_override is None:
        target_cd_geoids = _load_target_cd_geoids(db_path)
        weighted_requests = _build_regional_weighted_requests(
            geography=regional_geography,
            target_cd_geoids=target_cd_geoids,
        )
    else:
        weighted_requests = _build_weighted_requests_from_work_items(
            work_items=work_items_override,
            geography=regional_geography,
        )
    if not weighted_requests:
        raise RuntimeError("No regional H5 requests found for coordinator geography")

    expected_total = len(weighted_requests)
    expected_keys = {item.key for item in weighted_requests}

    staging_volume.reload()
    completed = get_completed_from_volume(run_dir)
    initially_completed = set(completed)
    print(f"Found {len(completed)} already-completed items on volume")

    phase_args = dict(
        num_workers=num_workers,
        branch=branch,
        run_id=run_id,
        calibration_inputs=calibration_inputs,
        run_dir=run_dir,
        validate=validate,
        scope_fingerprint=fingerprint,
    )

    accumulated_errors = []
    accumulated_validation_rows = []

    completed, phase_errors, v_rows = run_phase(
        "All areas",
        weighted_requests=weighted_requests,
        completed=completed,
        **phase_args,
    )
    accumulated_errors.extend(phase_errors)
    accumulated_validation_rows.extend(v_rows)

    # If workers crashed but all files landed on the volume,
    # treat as transient infrastructure errors (e.g. gRPC stream resets).
    missing_keys, reuse_measurement = _measure_expected_completion(
        expected_keys=expected_keys,
        initially_completed=initially_completed,
        completed=completed,
    )
    if accumulated_errors:
        fatal_worker_errors = [
            error
            for error in accumulated_errors
            if error.get("severity") in {"protocol", "worker_failure"}
        ]
        transport_errors = [
            error
            for error in accumulated_errors
            if error.get("severity") == "transport"
        ]
        if fatal_worker_errors:
            raise RuntimeError(
                f"Build failed: {len(fatal_worker_errors)} fatal worker "
                f"error(s) detected. Errors: {fatal_worker_errors[:3]}"
            )
        if transport_errors and not missing_keys:
            print(
                f"WARNING: {len(transport_errors)} worker transport error(s) occurred "
                f"but all {expected_total} files present on volume. "
                f"Treating as transient. Errors: {transport_errors[:3]}"
            )
        elif transport_errors:
            raise RuntimeError(
                f"Build failed: {len(transport_errors)} worker "
                f"transport error(s) detected and only "
                f"{expected_total - len(missing_keys)}/{expected_total} "
                f"expected files on volume. "
                f"Errors: {transport_errors[:3]}"
            )

    if missing_keys:
        raise RuntimeError(
            f"Build incomplete: {len(missing_keys)} expected files missing from "
            f"volume ({expected_total - len(missing_keys)}/{expected_total}). "
            f"Missing: {sorted(missing_keys)[:5]}. "
            f"Volume preserved for retry."
        )

    if skip_upload:
        print("\nSkipping upload (--skip-upload flag set)")
        return {
            "message": (f"Build complete for version {version}. Upload skipped."),
            "validation_rows": accumulated_validation_rows,
            "fingerprint": fingerprint,
            "reuse_measurement": reuse_measurement,
        }

    print("\nValidating staging...")
    manifest = validate_staging.remote(branch=branch, run_id=run_id, version=version)

    actual_total = (
        manifest["totals"]["states"]
        + manifest["totals"]["districts"]
        + manifest["totals"]["cities"]
    )

    if actual_total < expected_total:
        print(f"WARNING: Expected {expected_total} files, found {actual_total}")

    print("\nStarting upload to staging...")
    result = upload_to_staging.remote(
        branch=branch,
        version=version,
        manifest=manifest,
        run_id=run_id,
        candidate_version=staging_candidate_version,
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
        "reuse_measurement": reuse_measurement,
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
    skip_upload: bool = False,
    candidate_version: str = "",
) -> Dict:
    """Build and upload a national US.h5 from national weights."""
    setup_gcp_credentials()
    setup_repo(branch)

    version = get_version()
    staging_candidate_version = get_staging_candidate_version(
        version,
        explicit_candidate_version=candidate_version,
    )

    run_id = run_id or resolve_run_id()
    if not run_id:
        raise RuntimeError(
            "run_id is required. National publishing must receive the "
            "GitHub-created run ID from the pipeline."
        )

    print("=" * 60)
    print(f"Run ID: {run_id}")
    print("=" * 60)
    print(f"Building national H5 for version {version} from branch {branch}")
    print(f"Staging candidate {staging_candidate_version}")

    staging_dir = Path(VOLUME_MOUNT)

    pipeline_volume.reload()
    artifacts = (
        Path(f"/pipeline/artifacts/{run_id}") if run_id else Path("/pipeline/artifacts")
    )
    weights_path = artifacts / "national_calibration_weights.npy"
    geography_path = artifacts / "national_geography_assignment.npz"
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

    calibration_inputs = _build_worker_calibration_inputs(
        weights_path=weights_path,
        geography_path=geography_path,
        dataset_path=dataset_path,
        db_path=db_path,
        n_clones=n_clones,
        seed=42,
        run_config_path=config_json_path,
    )
    validate_artifacts(
        config_json_path,
        artifacts,
        filename_remap={
            "calibration_weights.npy": "national_calibration_weights.npy",
            "geography_assignment.npz": "national_geography_assignment.npz",
        },
    )
    fingerprint_inputs = _build_publishing_input_bundle(
        weights_path=weights_path,
        dataset_path=dataset_path,
        db_path=db_path,
        geography_path=geography_path,
        calibration_package_path=None,
        run_config_path=config_json_path if config_json_path.exists() else None,
        run_id=run_id,
        version=version,
        n_clones=n_clones,
        seed=42,
    )
    fingerprint = _resolve_scope_fingerprint(
        inputs=fingerprint_inputs,
        scope="national",
    )
    run_dir = staging_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _build_worker_bootstrap(
        inputs=fingerprint_inputs,
        scope="national",
        artifacts_dir=artifacts,
        scope_fingerprint=fingerprint,
    )
    pipeline_volume.commit()
    national_h5 = run_dir / "national" / "US.h5"

    national_request = USAreaCatalog.default().build_national_request()
    print("Spawning worker for national H5 build...")
    raw_worker_result = build_areas_worker.remote(
        branch=branch,
        run_id=run_id,
        scope="national",
        request_payloads=[national_request.to_dict()],
        calibration_inputs=calibration_inputs.to_wire_dict(),
        validate=validate,
        scope_fingerprint=fingerprint,
    )
    worker_result = normalize_worker_response(worker_index=0, result=raw_worker_result)

    print(
        f"Worker result: "
        f"{len(worker_result.completed)} completed, "
        f"{len(worker_result.failed)} failed"
    )

    if worker_result.fatal_errors:
        raise RuntimeError(f"National build failed: {worker_result.fatal_errors}")

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
            _python_cmd(
                "-m",
                "policyengine_us_data.calibration.validate_national_h5",
                "--h5-path",
                str(national_h5),
            ),
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

    if skip_upload:
        print("\nSkipping national upload (--skip-upload flag set)")
        return {
            "message": (f"National US.h5 built for version {version}. Upload skipped."),
            "run_id": run_id,
            "fingerprint": fingerprint,
            "national_validation": national_validation_output,
            "reuse_measurement": {
                "expected_outputs": 1,
                "valid_reused_outputs": 0,
                "recomputed_outputs": 1,
                "invalid_outputs": 0,
            },
        }

    print(f"Uploading {national_h5} to HF staging...")
    result = subprocess.run(
        _python_cmd(
            "-c",
            f"""
from policyengine_us_data.utils.data_upload import (
    upload_to_staging_hf,
)
upload_to_staging_hf(
    [("{national_h5}", "national/US.h5")],
    candidate_version="{staging_candidate_version}",
    run_id="{run_id}",
)
print("Done")
""",
        ),
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
        "fingerprint": fingerprint,
        "national_validation": national_validation_output,
        "reuse_measurement": {
            "expected_outputs": 1,
            "valid_reused_outputs": 0,
            "recomputed_outputs": 1,
            "invalid_outputs": 0,
        },
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
    cleanup_staging: bool = True,
) -> str:
    """Promote national US.h5 from HF staging to production + GCS."""
    setup_gcp_credentials()
    setup_repo(branch)

    if not run_id:
        raise ValueError("--run-id is required for promote")
    if not version:
        version = get_version()
    rel_paths = ["national/US.h5"]

    result = subprocess.run(
        _python_cmd(
            "-c",
            _build_promote_national_publish_script(
                version=version,
                run_id=run_id,
                rel_paths=rel_paths,
                cleanup_staging=cleanup_staging,
            ),
        ),
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
