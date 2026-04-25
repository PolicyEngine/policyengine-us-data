"""Shared tiny-artifact fixtures for local H5 integration tests."""

from __future__ import annotations

import json
import pickle
import shutil
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from policyengine_us_data.calibration.clone_and_assign import (
    GeographyAssignment,
    save_geography,
)
from policyengine_us_data.calibration.local_h5.requests import (
    AreaBuildRequest,
    AreaFilter,
)

FIXTURE_DATASET_PATH = Path(__file__).resolve().parents[1] / "test_fixture_50hh.h5"
DISTRICT_GEOID = "3701"
COUNTY_FIPS = "37183"
STATE_FIPS = 37
N_CLONES = 1
SEED = 42
VERSION = "0.0.0"


@dataclass(frozen=True)
class LocalH5Artifacts:
    dataset_path: Path
    weights_path: Path
    db_path: Path
    run_config_path: Path
    geography_path: Path
    calibration_package_path: Path
    geography: GeographyAssignment
    n_records: int
    n_clones: int


@lru_cache(maxsize=1)
def fixture_household_count() -> int:
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=str(FIXTURE_DATASET_PATH))
    try:
        return int(len(sim.calculate("household_id", map_to="household").values))
    finally:
        del sim


def base_geography(*, n_records: int, n_clones: int = N_CLONES) -> GeographyAssignment:
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


def seed_local_h5_artifacts(
    tmp_path: Path,
    *,
    n_clones: int = N_CLONES,
) -> LocalH5Artifacts:
    artifact_dir = tmp_path / "artifacts"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = artifact_dir / "source.h5"
    weights_path = artifact_dir / "calibration_weights.npy"
    db_path = artifact_dir / "policy_data.db"
    run_config_path = artifact_dir / "unified_run_config.json"
    geography_path = artifact_dir / "geography_assignment.npz"
    calibration_package_path = artifact_dir / "calibration_package.pkl"

    shutil.copy2(FIXTURE_DATASET_PATH, dataset_path)
    n_records = fixture_household_count()
    np.save(weights_path, np.ones(n_records * n_clones, dtype=np.float32))

    geography = base_geography(n_records=n_records, n_clones=n_clones)
    save_geography(geography, geography_path)

    with open(calibration_package_path, "wb") as handle:
        pickle.dump(
            {
                "block_geoid": geography.block_geoid,
                "cd_geoid": geography.cd_geoid,
                "metadata": {
                    "git_commit": "deadbeefcafebabe",
                    "git_branch": "main",
                    "git_dirty": False,
                    "package_version": VERSION,
                },
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    conn = sqlite3.connect(db_path)
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

    run_config_path.write_text(
        json.dumps(
            {
                "git_commit": "deadbeefcafebabe",
                "git_branch": "main",
                "git_dirty": False,
                "package_version": VERSION,
            }
        )
    )

    return LocalH5Artifacts(
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


def build_request(
    area_type: str, *, geography: GeographyAssignment
) -> AreaBuildRequest:
    if area_type == "district":
        return AreaBuildRequest(
            area_type="district",
            area_id="NC-01",
            display_name="NC-01",
            output_relative_path="districts/NC-01.h5",
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
            area_id="NC",
            display_name="NC",
            output_relative_path="states/NC.h5",
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
    raise ValueError(f"Unsupported area_type for test fixture: {area_type}")
