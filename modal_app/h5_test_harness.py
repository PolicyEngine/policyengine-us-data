"""Tiny-fixture Modal harness for H5 publish end-to-end tests."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image as image  # noqa: E402
from modal_app.local_area import VOLUME_MOUNT, pipeline_volume, staging_volume  # noqa: E402


app = modal.App("policyengine-us-data-h5-test-harness")


@app.function(
    image=image,
    volumes={
        "/pipeline": pipeline_volume,
        VOLUME_MOUNT: staging_volume,
    },
    timeout=600,
    memory=4096,
    cpu=1.0,
)
def seed_h5_case(run_id: str, case_name: str) -> dict:
    from modal_app.fixtures.h5_cases import seed_case

    pipeline_volume.reload()
    staging_volume.reload()
    artifact_dir = Path(f"/pipeline/artifacts/{run_id}")
    staging_dir = Path(VOLUME_MOUNT) / run_id
    seeded = seed_case(
        run_id=run_id,
        artifact_dir=artifact_dir,
        staging_dir=staging_dir,
        case_name=case_name,
    )
    pipeline_volume.commit()
    staging_volume.commit()
    return {
        "name": seeded.name,
        "calibration_inputs": seeded.calibration_inputs,
        "expected_district_name": seeded.expected_district_name,
        "n_clones": seeded.n_clones,
        "seed": seeded.seed,
    }


@app.function(
    image=image,
    volumes={
        "/pipeline": pipeline_volume,
        VOLUME_MOUNT: staging_volume,
    },
    timeout=600,
    memory=4096,
    cpu=1.0,
)
def preflight_h5_case(run_id: str, *, n_clones: int = 1) -> dict:
    from modal_app.local_area import validate_artifacts
    from modal_app.fixtures.h5_cases import SEED
    from policyengine_us_data.calibration.publish_local_area import (
        compute_input_fingerprint,
    )
    from policyengine_us_data.calibration.local_h5.geography_loader import (
        CalibrationGeographyLoader,
    )

    pipeline_volume.reload()
    staging_volume.reload()
    artifact_dir = Path(f"/pipeline/artifacts/{run_id}")
    config_path = artifact_dir / "unified_run_config.json"
    weights_path = artifact_dir / "calibration_weights.npy"
    dataset_path = artifact_dir / "source_imputed_stratified_extended_cps.h5"
    db_path = artifact_dir / "policy_data.db"
    geography_path = artifact_dir / "geography_assignment.npz"
    package_path = artifact_dir / "calibration_package.pkl"

    validate_artifacts(config_path, artifact_dir)
    fingerprint = compute_input_fingerprint(
        weights_path=weights_path,
        dataset_path=dataset_path,
        n_clones=n_clones,
        seed=SEED,
        geography_path=geography_path if geography_path.exists() else None,
        blocks_path=artifact_dir / "stacked_blocks.npy",
        calibration_package_path=package_path if package_path.exists() else None,
    )
    loader = CalibrationGeographyLoader()
    resolved = loader.resolve_source(
        weights_path=weights_path,
        geography_path=geography_path if geography_path.exists() else None,
        calibration_package_path=package_path if package_path.exists() else None,
        blocks_path=artifact_dir / "stacked_blocks.npy",
    )
    calibration_inputs = {
        "weights": str(weights_path),
        "dataset": str(dataset_path),
        "database": str(db_path),
        "n_clones": n_clones,
        "seed": SEED,
    }
    if geography_path.exists():
        calibration_inputs["geography"] = str(geography_path)
    if package_path.exists():
        calibration_inputs["calibration_package"] = str(package_path)
    return {
        "fingerprint": fingerprint,
        "geography_source": resolved.kind if resolved is not None else None,
        "calibration_inputs": calibration_inputs,
    }


@app.function(
    image=image,
    volumes={
        "/pipeline": pipeline_volume,
        VOLUME_MOUNT: staging_volume,
    },
    timeout=600,
    memory=2048,
    cpu=1.0,
)
def cleanup_h5_case(run_id: str) -> None:
    pipeline_volume.reload()
    staging_volume.reload()
    artifact_dir = Path(f"/pipeline/artifacts/{run_id}")
    staging_dir = Path(VOLUME_MOUNT) / run_id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    pipeline_volume.commit()
    staging_volume.commit()
