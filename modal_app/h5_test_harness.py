"""Tiny-fixture Modal harness for H5 publish end-to-end tests."""

from __future__ import annotations

import hashlib
import json
import os
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


app = modal.App(
    os.environ.get("US_DATA_H5_HARNESS_APP_NAME")
    or "policyengine-us-data-h5-test-harness"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _copy_to_artifact_root(source: Path, artifact_dir: Path, name: str) -> Path:
    destination = artifact_dir / name
    shutil.copy2(source, destination)
    return destination


def _write_flat_run_config(
    artifact_dir: Path,
    *,
    artifact_paths: dict[str, Path],
    filename: str = "unified_run_config.json",
) -> Path:
    payload = {
        "git_commit": "deadbeefcafebabe",
        "git_branch": "main",
        "git_dirty": False,
        "package_version": "0.0.0",
        "artifacts": {
            name: _sha256(path) for name, path in sorted(artifact_paths.items())
        },
    }
    config_path = artifact_dir / filename
    config_path.write_text(json.dumps(payload, indent=2))
    return config_path


def _calibration_inputs(
    *,
    weights_path: Path,
    dataset_path: Path,
    db_path: Path,
    geography_path: Path | None = None,
    calibration_package_path: Path | None = None,
    n_clones: int = 1,
    seed: int = 42,
) -> dict:
    inputs = {
        "weights": str(weights_path),
        "dataset": str(dataset_path),
        "database": str(db_path),
        "n_clones": n_clones,
        "seed": seed,
    }
    if geography_path is not None:
        inputs["geography"] = str(geography_path)
    if calibration_package_path is not None:
        inputs["calibration_package"] = str(calibration_package_path)
    return inputs


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
def seed_tiny_pipeline_case(
    run_id: str, case_name: str = "saved_geography_success"
) -> dict:
    """Seed Modal volumes with the shared fixture-scale Stage 1-5 artifact shape."""

    from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
    from tests.integration.support.tiny_h5 import create_tiny_h5_artifacts
    from tests.integration.support.tiny_pipeline import create_tiny_pipeline_artifacts

    pipeline_volume.reload()
    staging_volume.reload()
    artifact_dir = Path(f"/pipeline/artifacts/{run_id}")
    staging_dir = Path(VOLUME_MOUNT) / run_id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    workspace = TinyPipelineWorkspace.create(artifact_dir / "tiny_workspace")
    pipeline_artifacts = create_tiny_pipeline_artifacts(workspace)
    h5_artifacts = create_tiny_h5_artifacts(workspace, pipeline_artifacts)

    dataset_path = _copy_to_artifact_root(
        h5_artifacts.dataset_path,
        artifact_dir,
        "source_imputed_stratified_extended_cps.h5",
    )
    weights_path = _copy_to_artifact_root(
        h5_artifacts.weights_path,
        artifact_dir,
        "calibration_weights.npy",
    )
    national_weights_path = _copy_to_artifact_root(
        h5_artifacts.weights_path,
        artifact_dir,
        "national_calibration_weights.npy",
    )
    db_path = _copy_to_artifact_root(
        h5_artifacts.db_path,
        artifact_dir,
        "policy_data.db",
    )

    artifact_paths = {
        "calibration_weights.npy": weights_path,
    }
    geography_path = None
    calibration_package_path = None

    if case_name == "saved_geography_success":
        geography_path = _copy_to_artifact_root(
            h5_artifacts.geography_path,
            artifact_dir,
            "geography_assignment.npz",
        )
        national_geography_path = _copy_to_artifact_root(
            h5_artifacts.geography_path,
            artifact_dir,
            "national_geography_assignment.npz",
        )
        artifact_paths["geography_assignment.npz"] = geography_path
    elif case_name == "package_fallback_success":
        national_geography_path = _copy_to_artifact_root(
            h5_artifacts.geography_path,
            artifact_dir,
            "national_geography_assignment.npz",
        )
        calibration_package_path = _copy_to_artifact_root(
            h5_artifacts.calibration_package_path,
            artifact_dir,
            "calibration_package.pkl",
        )
        artifact_paths["calibration_package.pkl"] = calibration_package_path
    else:
        raise ValueError(f"Unknown tiny pipeline H5 case: {case_name}")

    run_config_path = _write_flat_run_config(
        artifact_dir,
        artifact_paths=artifact_paths,
    )
    national_run_config_path = _write_flat_run_config(
        artifact_dir,
        artifact_paths={
            "calibration_weights.npy": national_weights_path,
            "geography_assignment.npz": national_geography_path,
        },
        filename="national_unified_run_config.json",
    )

    pipeline_volume.commit()
    staging_volume.commit()
    return {
        "name": case_name,
        "workspace_root": str(workspace.root),
        "stage_5_source": str(pipeline_artifacts.stage_5.source_imputed_alias_path),
        "calibration_inputs": _calibration_inputs(
            weights_path=weights_path,
            dataset_path=dataset_path,
            db_path=db_path,
            geography_path=geography_path,
            calibration_package_path=calibration_package_path,
            n_clones=h5_artifacts.n_clones,
            seed=42,
        ),
        "run_config": str(run_config_path),
        "national_run_config": str(national_run_config_path),
        "expected_district_name": "NC-01",
        "n_clones": h5_artifacts.n_clones,
        "seed": 42,
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
        VOLUME_MOUNT: staging_volume,
    },
    timeout=300,
    memory=2048,
    cpu=1.0,
)
def inspect_h5_outputs(run_id: str, relative_paths: list[str]) -> dict:
    """Inspect staged H5 outputs and manifest contents inside the Modal volume."""

    import h5py

    staging_volume.reload()
    run_dir = Path(VOLUME_MOUNT) / run_id
    manifest_path = run_dir / "manifest.json"
    outputs = {}
    for relative_path in relative_paths:
        output_path = run_dir / relative_path
        output = {
            "exists": output_path.exists(),
            "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "variables": {},
        }
        if output_path.exists():
            with h5py.File(output_path, mode="r") as h5:
                output["top_level_keys"] = sorted(h5.keys())
                for variable in (
                    "household_id",
                    "person_id",
                    "household_weight",
                    "state_fips",
                    "congressional_district_geoid",
                ):
                    output["variables"][variable] = _inspect_h5_variable(h5, variable)
        outputs[relative_path] = output

    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    return {
        "run_dir_exists": run_dir.exists(),
        "manifest_exists": manifest_path.exists(),
        "manifest": manifest,
        "outputs": outputs,
    }


def _inspect_h5_variable(h5, variable: str) -> dict:
    """Return row-count metadata for either grouped or direct H5 datasets."""

    import h5py

    if variable not in h5:
        return {
            "exists": False,
            "rows": 0,
            "periods": [],
        }

    node = h5[variable]
    if isinstance(node, h5py.Dataset):
        rows = _h5_dataset_rows(node)
        return {
            "exists": rows > 0,
            "rows": rows,
            "periods": [],
        }

    period_rows = {
        str(period): _h5_dataset_rows(dataset)
        for period, dataset in node.items()
        if isinstance(dataset, h5py.Dataset)
    }
    rows = max(period_rows.values(), default=0)
    return {
        "exists": rows > 0,
        "rows": rows,
        "periods": sorted(period_rows),
    }


def _h5_dataset_rows(dataset) -> int:
    if dataset.shape == ():
        return int(dataset.size)
    return int(dataset.shape[0])


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
