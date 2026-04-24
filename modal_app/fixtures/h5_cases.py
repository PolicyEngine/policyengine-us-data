"""Tiny fixture writers for Modal H5 end-to-end tests."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXTURE_DATASET_PATH = Path(
    "/root/policyengine-us-data/tests/integration/test_fixture_50hh.h5"
)
DISTRICT_GEOID = "3701"
DISTRICT_NAME = "NC-01"
COUNTY_FIPS = "37183"
STATE_FIPS = 37
N_CLONES = 1
SEED = 42
VERSION = "0.0.0"


@dataclass(frozen=True)
class SeededCase:
    """Description of one tiny end-to-end H5 test case."""

    name: str
    calibration_inputs: dict[str, Any]
    expected_district_name: str = DISTRICT_NAME
    n_clones: int = N_CLONES
    seed: int = SEED


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _fixture_n_households() -> int:
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=str(FIXTURE_DATASET_PATH))
    try:
        return int(len(sim.calculate("household_id", map_to="household").values))
    finally:
        del sim


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_dataset(artifact_dir: Path) -> Path:
    dataset_path = artifact_dir / "source_imputed_stratified_extended_cps.h5"
    shutil.copy2(FIXTURE_DATASET_PATH, dataset_path)
    return dataset_path


def _write_weights(artifact_dir: Path, *, n_records: int) -> Path:
    import numpy as np

    weights_path = artifact_dir / "calibration_weights.npy"
    np.save(weights_path, np.ones(n_records * N_CLONES, dtype=np.float32))
    return weights_path


def _base_geography(*, n_records: int):
    import numpy as np

    from policyengine_us_data.calibration.clone_and_assign import GeographyAssignment

    total_rows = n_records * N_CLONES
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
        n_clones=N_CLONES,
    )


def _write_saved_geography(artifact_dir: Path, *, n_records: int) -> Path:
    from policyengine_us_data.calibration.clone_and_assign import save_geography

    geography_path = artifact_dir / "geography_assignment.npz"
    save_geography(_base_geography(n_records=n_records), geography_path)
    return geography_path


def _write_calibration_package(artifact_dir: Path, *, n_records: int) -> Path:
    package_path = artifact_dir / "calibration_package.pkl"
    geography = _base_geography(n_records=n_records)
    payload = {
        "block_geoid": geography.block_geoid,
        "cd_geoid": geography.cd_geoid,
        "metadata": {
            "git_commit": "deadbeefcafebabe",
            "git_branch": "main",
            "git_dirty": False,
            "package_version": VERSION,
        },
    }
    with open(package_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return package_path


def _write_misnamed_package(artifact_dir: Path, *, n_records: int) -> Path:
    wrong_path = artifact_dir / "calibration_package_typo.pkl"
    geography = _base_geography(n_records=n_records)
    payload = {
        "block_geoid": geography.block_geoid,
        "cd_geoid": geography.cd_geoid,
    }
    with open(wrong_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return wrong_path


def _write_db(artifact_dir: Path) -> Path:
    db_path = artifact_dir / "policy_data.db"
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
    return db_path


def _write_run_config(
    artifact_dir: Path,
    *,
    weights_path: Path,
    geography_path: Path | None = None,
    checkpoint_key: str | None = None,
    checkpoint_path: Path | None = None,
) -> Path:
    payload = {
        "git_commit": "deadbeefcafebabe",
        "git_branch": "main",
        "git_dirty": False,
        "package_version": VERSION,
        "artifacts": {
            "calibration_weights.npy": _sha256(weights_path),
        },
    }
    if geography_path is not None:
        payload["artifacts"]["geography_assignment.npz"] = _sha256(geography_path)
    if checkpoint_key is not None and checkpoint_path is not None:
        payload["artifacts"][checkpoint_key] = _sha256(checkpoint_path)

    config_path = artifact_dir / "unified_run_config.json"
    config_path.write_text(json.dumps(payload, indent=2))
    return config_path


def seed_case(
    *, run_id: str, artifact_dir: Path, staging_dir: Path, case_name: str
) -> SeededCase:
    """Write one tiny artifact bundle for the requested end-to-end case."""

    _reset_dir(artifact_dir)
    _reset_dir(staging_dir)

    n_records = _fixture_n_households()
    dataset_path = _write_dataset(artifact_dir)
    weights_path = _write_weights(artifact_dir, n_records=n_records)
    db_path = _write_db(artifact_dir)

    calibration_inputs: dict[str, Any] = {
        "weights": str(weights_path),
        "dataset": str(dataset_path),
        "database": str(db_path),
        "n_clones": N_CLONES,
        "seed": SEED,
    }

    if case_name == "saved_geography_success":
        geography_path = _write_saved_geography(artifact_dir, n_records=n_records)
        _write_run_config(
            artifact_dir,
            weights_path=weights_path,
            geography_path=geography_path,
        )
        calibration_inputs["geography"] = str(geography_path)
    elif case_name == "package_fallback_success":
        package_path = _write_calibration_package(artifact_dir, n_records=n_records)
        _write_run_config(artifact_dir, weights_path=weights_path)
        calibration_inputs["calibration_package"] = str(package_path)
    elif case_name == "checkpoint_name_mismatch":
        geography_path = _write_saved_geography(artifact_dir, n_records=n_records)
        checkpoint_path = artifact_dir / "calibration_weights.checkpoint.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        _write_run_config(
            artifact_dir,
            weights_path=weights_path,
            geography_path=geography_path,
            checkpoint_key="calibration_checkpoint.pt",
            checkpoint_path=checkpoint_path,
        )
        calibration_inputs["geography"] = str(geography_path)
    elif case_name == "misnamed_package":
        _write_misnamed_package(artifact_dir, n_records=n_records)
        _write_run_config(artifact_dir, weights_path=weights_path)
    else:
        raise ValueError(f"Unknown H5 test case: {case_name}")

    return SeededCase(
        name=case_name,
        calibration_inputs=calibration_inputs,
    )
