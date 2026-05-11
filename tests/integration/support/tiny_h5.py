"""Fixture-scale H5 continuation helpers for integration tests."""

from __future__ import annotations

import json
import pickle
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policyengine_us_data.calibration.clone_and_assign import (
    GeographyAssignment,
    save_geography,
)
from policyengine_us_data.build_outputs.fingerprinting import (
    PublishingInputBundle,
)
from policyengine_us_data.build_outputs.requests import (
    AreaBuildRequest,
    AreaFilter,
)
from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_pipeline import TinyPipelineArtifacts

__test__ = False


DISTRICT_GEOID = "3701"
COUNTY_FIPS = "37183"
STATE_CODE = "NC"
STATE_FIPS = 37
N_CLONES = 1
SEED = 42
VERSION = "0.0.0"


@dataclass(frozen=True)
class TinyH5Artifacts:
    """Artifacts needed to continue a tiny Stage 5 build into H5 generation."""

    dataset_path: Path
    weights_path: Path
    db_path: Path
    run_config_path: Path
    geography_path: Path
    calibration_package_path: Path
    geography: GeographyAssignment
    n_records: int
    n_clones: int


def create_tiny_h5_artifacts(
    workspace: TinyPipelineWorkspace,
    pipeline_artifacts: TinyPipelineArtifacts,
    *,
    n_clones: int = N_CLONES,
) -> TinyH5Artifacts:
    """Seed calibration inputs using the shared tiny Stage 5 dataset."""

    dataset_path = pipeline_artifacts.stage_5.source_imputed_alias_path
    n_records = _household_count(dataset_path)

    weights_path = workspace.artifact_path("calibration", "calibration_weights.npy")
    db_path = workspace.artifact_path("calibration", "policy_data.db")
    run_config_path = workspace.artifact_path("calibration", "unified_run_config.json")
    geography_path = workspace.artifact_path("calibration", "geography_assignment.npz")
    calibration_package_path = workspace.artifact_path(
        "calibration",
        "calibration_package.pkl",
    )

    np.save(weights_path, np.ones(n_records * n_clones, dtype=np.float32))

    geography = base_geography(n_records=n_records, n_clones=n_clones)
    save_geography(geography, geography_path)
    _write_calibration_package(calibration_package_path, geography=geography)
    _write_policy_data_db(db_path)
    run_config_path.write_text(json.dumps(_run_metadata()))

    return TinyH5Artifacts(
        dataset_path=dataset_path,
        weights_path=weights_path,
        db_path=db_path,
        run_config_path=run_config_path,
        geography_path=geography_path,
        calibration_package_path=calibration_package_path,
        geography=geography,
        n_records=n_records,
        n_clones=n_clones,
    )


def base_geography(*, n_records: int, n_clones: int = N_CLONES) -> GeographyAssignment:
    """Create one deterministic NC-01 geography assignment."""

    total_rows = n_records * n_clones
    block_geoids = np.array(
        [f"{COUNTY_FIPS}{i:06d}{i:04d}"[:15] for i in range(total_rows)],
        dtype="U15",
    )
    return GeographyAssignment(
        block_geoid=block_geoids,
        cd_geoid=np.full(total_rows, DISTRICT_GEOID, dtype="U4"),
        county_fips=np.full(total_rows, COUNTY_FIPS, dtype="U5"),
        state_fips=np.full(total_rows, STATE_FIPS, dtype=np.int32),
        n_records=n_records,
        n_clones=n_clones,
    )


def build_h5_request(area_type: str) -> AreaBuildRequest:
    """Return a typed worker request for the tiny H5 fixture geography."""

    if area_type == "district":
        return AreaBuildRequest(
            area_type="district",
            area_id=f"{STATE_CODE}-01",
            display_name=f"{STATE_CODE}-01",
            output_relative_path=f"districts/{STATE_CODE}-01.h5",
            filters=(
                AreaFilter(
                    geography_field="cd_geoid",
                    op="in",
                    value=(DISTRICT_GEOID,),
                ),
            ),
            validation_geo_level="district",
            validation_geographic_ids=(DISTRICT_GEOID,),
        )
    if area_type == "state":
        return AreaBuildRequest(
            area_type="state",
            area_id=STATE_CODE,
            display_name=STATE_CODE,
            output_relative_path=f"states/{STATE_CODE}.h5",
            filters=(
                AreaFilter(
                    geography_field="cd_geoid",
                    op="in",
                    value=(DISTRICT_GEOID,),
                ),
            ),
            validation_geo_level="state",
            validation_geographic_ids=(str(STATE_FIPS),),
        )
    if area_type == "national":
        return AreaBuildRequest(
            area_type="national",
            area_id="US",
            display_name="US",
            output_relative_path="national/US.h5",
            validation_geo_level="national",
            validation_geographic_ids=("US",),
        )
    raise ValueError(f"Unsupported tiny H5 request type: {area_type}")


def build_publishing_input_bundle(
    artifacts: TinyH5Artifacts,
    *,
    run_id: str,
    scope: str,
) -> PublishingInputBundle:
    """Build the same traceability input shape used by local H5 publication."""

    return PublishingInputBundle(
        weights_path=artifacts.weights_path,
        source_dataset_path=artifacts.dataset_path,
        target_db_path=artifacts.db_path,
        exact_geography_path=artifacts.geography_path,
        calibration_package_path=(
            artifacts.calibration_package_path if scope == "regional" else None
        ),
        run_config_path=artifacts.run_config_path,
        run_id=run_id,
        version=VERSION,
        n_clones=artifacts.n_clones,
        seed=SEED,
    )


def run_local_h5_worker(
    *,
    requests: tuple[AreaBuildRequest, ...],
    artifacts: TinyH5Artifacts,
    output_dir: Path,
    use_saved_geography: bool,
    use_package_geography: bool,
) -> dict:
    """Run the real local H5 worker subprocess for tiny fixture requests."""

    cmd = [
        sys.executable,
        "-m",
        "modal_app.worker_script",
        "--requests-json",
        json.dumps([request.to_dict() for request in requests]),
        "--weights-path",
        str(artifacts.weights_path),
        "--dataset-path",
        str(artifacts.dataset_path),
        "--db-path",
        str(artifacts.db_path),
        "--output-dir",
        str(output_dir),
        "--scope",
        "regional",
        "--n-clones",
        str(artifacts.n_clones),
        "--no-validate",
    ]
    if use_saved_geography:
        cmd.extend(["--geography-path", str(artifacts.geography_path)])
    if use_package_geography:
        cmd.extend(
            [
                "--calibration-package-path",
                str(artifacts.calibration_package_path),
            ]
        )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _household_count(dataset_path: Path) -> int:
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=str(dataset_path))
    try:
        return int(len(sim.calculate("household_id", map_to="household").values))
    finally:
        del sim


def _run_metadata() -> dict[str, object]:
    return {
        "git_commit": "deadbeefcafebabe",
        "git_branch": "main",
        "git_dirty": False,
        "package_version": VERSION,
    }


def _write_calibration_package(
    path: Path,
    *,
    geography: GeographyAssignment,
) -> None:
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "block_geoid": geography.block_geoid,
                "cd_geoid": geography.cd_geoid,
                "metadata": _run_metadata(),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _write_policy_data_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE stratum_constraints (
                stratum_id INTEGER,
                constraint_variable TEXT,
                value TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO stratum_constraints (stratum_id, constraint_variable, value)
            VALUES (?, ?, ?)
            """,
            (1, "congressional_district_geoid", DISTRICT_GEOID),
        )
        conn.commit()
    finally:
        conn.close()
