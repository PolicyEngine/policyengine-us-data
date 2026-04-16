import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from policyengine_us import Microsimulation
from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
    County,
)
from sqlalchemy import create_engine, text

from policyengine_us_data.calibration.clone_and_assign import GeographyAssignment
from policyengine_us_data.calibration.unified_matrix_builder import (
    UnifiedMatrixBuilder,
)
from policyengine_us_data.db.create_database_tables import TARGET_OVERVIEW_VIEW


FIXTURE_PATH = Path(__file__).with_name("test_fixture_50hh.h5")


def _create_chunked_smoke_db(db_path):
    db_uri = f"sqlite:///{db_path}"
    engine = create_engine(db_uri)

    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE strata ("
                "stratum_id INTEGER PRIMARY KEY, "
                "definition_hash VARCHAR(64), "
                "parent_stratum_id INTEGER, "
                "notes VARCHAR)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE stratum_constraints ("
                "constraint_id INTEGER PRIMARY KEY, "
                "stratum_id INTEGER, "
                "constraint_variable TEXT, "
                "operation TEXT, "
                "value TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE targets ("
                "target_id INTEGER PRIMARY KEY, "
                "stratum_id INTEGER, "
                "variable TEXT, "
                "reform_id INTEGER DEFAULT 0, "
                "value REAL, "
                "period INTEGER, "
                "active INTEGER DEFAULT 1)"
            )
        )
        conn.execute(text(TARGET_OVERVIEW_VIEW))

        conn.execute(text("INSERT INTO strata VALUES (1, NULL, NULL, NULL)"))
        conn.execute(text("INSERT INTO strata VALUES (2, NULL, NULL, NULL)"))
        conn.execute(text("INSERT INTO strata VALUES (3, NULL, NULL, NULL)"))
        conn.execute(
            text(
                "INSERT INTO stratum_constraints VALUES "
                "(1, 2, 'congressional_district_geoid', '=', '3701')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO stratum_constraints VALUES (2, 3, 'state_fips', '=', '35')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO targets "
                "(target_id, stratum_id, variable, reform_id, value, period, active) "
                "VALUES "
                "(1, 1, 'household_count', 0, 100, 2023, 1), "
                "(2, 2, 'household_count', 0, 50, 2023, 1), "
                "(3, 3, 'household_count', 0, 50, 2023, 1)"
            )
        )
        conn.commit()

    return db_uri


def _create_chunked_entity_target_db(db_path):
    db_uri = _create_chunked_smoke_db(db_path)
    engine = create_engine(db_uri)

    with engine.connect() as conn:
        conn.execute(text("INSERT INTO strata VALUES (4, NULL, NULL, NULL)"))
        conn.execute(text("INSERT INTO strata VALUES (5, NULL, NULL, NULL)"))
        conn.execute(text("INSERT INTO strata VALUES (6, NULL, NULL, NULL)"))
        conn.execute(
            text(
                "INSERT INTO stratum_constraints VALUES "
                "(3, 4, 'aca_ptc', '>', '0'), "
                "(4, 5, 'aca_ptc', '>', '0'), "
                "(5, 5, 'congressional_district_geoid', '=', '3701'), "
                "(6, 6, 'aca_ptc', '>', '0'), "
                "(7, 6, 'state_fips', '=', '35')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO targets "
                "(target_id, stratum_id, variable, reform_id, value, period, active) "
                "VALUES "
                "(4, 4, 'aca_ptc', 0, 100, 2023, 1), "
                "(5, 5, 'aca_ptc', 0, 50, 2023, 1), "
                "(6, 6, 'aca_ptc', 0, 50, 2023, 1)"
            )
        )
        conn.commit()

    return db_uri


def _fake_geography_from_blocks(blocks):
    blocks = np.asarray(blocks, dtype=str)
    county_fips = np.array([block[:5] for block in blocks], dtype="U5")
    county_index = np.array(
        [
            County._member_names_.index("DOÑA_ANA_COUNTY_NM")
            if county == "35013"
            else County._member_names_.index("WAKE_COUNTY_NC")
            for county in county_fips
        ],
        dtype=np.int32,
    )
    return {
        "state_fips": np.array([int(block[:2]) for block in blocks], dtype=np.int32),
        "county_fips": county_fips,
        "county_index": county_index,
        "block_geoid": blocks,
        "tract_geoid": np.array([block[:11] for block in blocks], dtype="U11"),
        "cbsa_code": np.array(["00000"] * len(blocks), dtype="U5"),
        "sldu": np.array(["000"] * len(blocks), dtype="U3"),
        "sldl": np.array(["000"] * len(blocks), dtype="U3"),
        "place_fips": np.array(["00000"] * len(blocks), dtype="U5"),
        "vtd": np.array(["000000"] * len(blocks), dtype="U6"),
        "puma": np.array(["00000"] * len(blocks), dtype="U5"),
        "zcta": np.array(["00000"] * len(blocks), dtype="U5"),
    }


def _build_chunked_test_geography(sim):
    n_records = sim.calculate("household_id", map_to="household").shape[0]
    geography = GeographyAssignment(
        block_geoid=np.concatenate(
            [
                np.full(n_records, "371830501001001", dtype="U15"),
                np.full(n_records, "350130001001001", dtype="U15"),
            ]
        ),
        cd_geoid=np.concatenate(
            [
                np.full(n_records, "3701", dtype="U4"),
                np.full(n_records, "3502", dtype="U4"),
            ]
        ),
        county_fips=np.concatenate(
            [
                np.full(n_records, "37183", dtype="U5"),
                np.full(n_records, "35013", dtype="U5"),
            ]
        ),
        state_fips=np.concatenate(
            [
                np.full(n_records, 37, dtype=np.int32),
                np.full(n_records, 35, dtype=np.int32),
            ]
        ),
        n_records=n_records,
        n_clones=2,
    )
    return n_records, geography


def _build_chunked_test_builder(db_uri):
    return UnifiedMatrixBuilder(
        db_uri=db_uri,
        time_period=2023,
        dataset_path=str(FIXTURE_PATH),
    )


@pytest.fixture
def chunked_smoke_db():
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    try:
        yield _create_chunked_smoke_db(temp_db.name)
    finally:
        os.unlink(temp_db.name)


@pytest.fixture
def chunked_entity_target_db():
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    try:
        yield _create_chunked_entity_target_db(temp_db.name)
    finally:
        os.unlink(temp_db.name)


def test_build_matrix_chunked_smoke_on_fixture(
    tmp_path,
    monkeypatch,
    chunked_smoke_db,
):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.derive_geography_from_blocks",
        _fake_geography_from_blocks,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.load_cd_geoadj_values",
        lambda cds: {cd: 1.0 for cd in cds},
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone."
        "calculate_spm_thresholds_vectorized",
        lambda **kwargs: np.ones(
            len(kwargs["spm_unit_tenure_types"]),
            dtype=np.float32,
        ),
    )

    sim = Microsimulation(dataset=str(FIXTURE_PATH))
    n_records, geography = _build_chunked_test_geography(sim)
    builder = _build_chunked_test_builder(chunked_smoke_db)

    targets_df, matrix, target_names = builder.build_matrix_chunked(
        geography=geography,
        sim=sim,
        chunk_size=20,
        chunk_dir=str(tmp_path / "chunks"),
        rerandomize_takeup=False,
    )

    assert len(targets_df) == 3
    assert matrix.shape == (3, n_records * 2)
    assert matrix.nnz == n_records * 2 + n_records + n_records

    row_sums = {name: matrix[index].sum() for index, name in enumerate(target_names)}
    assert row_sums["national/household_count"] == n_records * 2
    assert row_sums["cd_3701/household_count"] == n_records
    assert row_sums["state_35/household_count"] == n_records


def test_build_matrix_chunked_matches_precomputed_builder(
    tmp_path,
    monkeypatch,
    chunked_smoke_db,
):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.derive_geography_from_blocks",
        _fake_geography_from_blocks,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.load_cd_geoadj_values",
        lambda cds: {cd: 1.0 for cd in cds},
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone."
        "calculate_spm_thresholds_vectorized",
        lambda **kwargs: np.ones(
            len(kwargs["spm_unit_tenure_types"]),
            dtype=np.float32,
        ),
    )

    sim = Microsimulation(dataset=str(FIXTURE_PATH))
    _, geography = _build_chunked_test_geography(sim)
    builder = _build_chunked_test_builder(chunked_smoke_db)

    expected_targets, expected_matrix, expected_names = builder.build_matrix(
        geography=geography,
        sim=sim,
        rerandomize_takeup=False,
        workers=1,
    )

    chunked_targets, chunked_matrix, chunked_names = builder.build_matrix_chunked(
        geography=geography,
        sim=sim,
        chunk_size=20,
        chunk_dir=str(tmp_path / "chunks"),
        rerandomize_takeup=False,
    )

    assert chunked_names == expected_names
    pd.testing.assert_frame_equal(chunked_targets, expected_targets)
    np.testing.assert_array_equal(
        chunked_matrix.toarray(),
        expected_matrix.toarray(),
    )


def test_build_matrix_chunked_matches_precomputed_builder_for_aca_ptc(
    tmp_path,
    monkeypatch,
    chunked_entity_target_db,
):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.derive_geography_from_blocks",
        _fake_geography_from_blocks,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.load_cd_geoadj_values",
        lambda cds: {cd: 1.0 for cd in cds},
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone."
        "calculate_spm_thresholds_vectorized",
        lambda **kwargs: np.ones(
            len(kwargs["spm_unit_tenure_types"]),
            dtype=np.float32,
        ),
    )

    sim = Microsimulation(dataset=str(FIXTURE_PATH))
    _, geography = _build_chunked_test_geography(sim)
    builder = _build_chunked_test_builder(chunked_entity_target_db)

    expected_targets, expected_matrix, expected_names = builder.build_matrix(
        geography=geography,
        sim=sim,
        rerandomize_takeup=False,
        workers=1,
    )

    chunked_targets, chunked_matrix, chunked_names = builder.build_matrix_chunked(
        geography=geography,
        sim=sim,
        chunk_size=20,
        chunk_dir=str(tmp_path / "chunks"),
        rerandomize_takeup=False,
    )

    assert chunked_names == expected_names
    pd.testing.assert_frame_equal(chunked_targets, expected_targets)
    np.testing.assert_array_equal(
        chunked_matrix.toarray(),
        expected_matrix.toarray(),
    )


def test_build_matrix_chunked_resume_reuses_matching_manifest(
    tmp_path,
    monkeypatch,
    chunked_smoke_db,
):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.derive_geography_from_blocks",
        _fake_geography_from_blocks,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.load_cd_geoadj_values",
        lambda cds: {cd: 1.0 for cd in cds},
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone."
        "calculate_spm_thresholds_vectorized",
        lambda **kwargs: np.ones(
            len(kwargs["spm_unit_tenure_types"]),
            dtype=np.float32,
        ),
    )

    sim = Microsimulation(dataset=str(FIXTURE_PATH))
    _, geography = _build_chunked_test_geography(sim)
    builder = _build_chunked_test_builder(chunked_smoke_db)
    chunk_dir = tmp_path / "chunks"

    first_targets, first_matrix, first_names = builder.build_matrix_chunked(
        geography=geography,
        sim=sim,
        chunk_size=20,
        chunk_dir=str(chunk_dir),
        rerandomize_takeup=False,
    )

    second_targets, second_matrix, second_names = builder.build_matrix_chunked(
        geography=geography,
        sim=sim,
        chunk_size=20,
        chunk_dir=str(chunk_dir),
        resume_chunks=True,
        rerandomize_takeup=False,
    )

    assert first_names == second_names
    pd.testing.assert_frame_equal(first_targets, second_targets)
    np.testing.assert_array_equal(first_matrix.toarray(), second_matrix.toarray())


def test_build_matrix_chunked_resume_rejects_lineage_mismatch(
    tmp_path,
    monkeypatch,
    chunked_smoke_db,
):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.derive_geography_from_blocks",
        _fake_geography_from_blocks,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.load_cd_geoadj_values",
        lambda cds: {cd: 1.0 for cd in cds},
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone."
        "calculate_spm_thresholds_vectorized",
        lambda **kwargs: np.ones(
            len(kwargs["spm_unit_tenure_types"]),
            dtype=np.float32,
        ),
    )

    sim = Microsimulation(dataset=str(FIXTURE_PATH))
    _, geography = _build_chunked_test_geography(sim)
    builder = _build_chunked_test_builder(chunked_smoke_db)
    chunk_dir = tmp_path / "chunks"

    builder.build_matrix_chunked(
        geography=geography,
        sim=sim,
        chunk_size=20,
        chunk_dir=str(chunk_dir),
        rerandomize_takeup=False,
    )

    with pytest.raises(ValueError, match="Chunk cache lineage mismatch"):
        builder.build_matrix_chunked(
            geography=geography,
            sim=sim,
            chunk_size=40,
            chunk_dir=str(chunk_dir),
            resume_chunks=True,
            rerandomize_takeup=False,
        )


def test_build_matrix_chunked_resume_rejects_cached_chunk_range_mismatch(
    tmp_path,
    monkeypatch,
    chunked_smoke_db,
):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.derive_geography_from_blocks",
        _fake_geography_from_blocks,
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone.load_cd_geoadj_values",
        lambda cds: {cd: 1.0 for cd in cds},
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.entity_clone."
        "calculate_spm_thresholds_vectorized",
        lambda **kwargs: np.ones(
            len(kwargs["spm_unit_tenure_types"]),
            dtype=np.float32,
        ),
    )

    sim = Microsimulation(dataset=str(FIXTURE_PATH))
    _, geography = _build_chunked_test_geography(sim)
    builder = _build_chunked_test_builder(chunked_smoke_db)
    chunk_dir = tmp_path / "chunks"

    builder.build_matrix_chunked(
        geography=geography,
        sim=sim,
        chunk_size=20,
        chunk_dir=str(chunk_dir),
        rerandomize_takeup=False,
    )

    cached_chunk = chunk_dir / "coo" / "chunk_000000.npz"
    with np.load(cached_chunk) as data:
        np.savez_compressed(
            cached_chunk,
            rows=data["rows"],
            cols=data["cols"],
            vals=data["vals"],
            col_start=np.array([1], dtype=np.int64),
            col_end=data["col_end"],
        )

    with pytest.raises(ValueError, match="covers cols"):
        builder.build_matrix_chunked(
            geography=geography,
            sim=sim,
            chunk_size=20,
            chunk_dir=str(chunk_dir),
            resume_chunks=True,
            rerandomize_takeup=False,
        )
