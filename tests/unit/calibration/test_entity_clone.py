from pathlib import Path

import h5py
import numpy as np
import pytest
from policyengine_us import Microsimulation
from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
    County,
)

from policyengine_us_data.calibration.entity_clone import (
    build_household_entity_maps,
    materialize_clone_household_chunk,
)


FIXTURE_PATH = Path(__file__).parents[2] / "integration" / "test_fixture_50hh.h5"


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


@pytest.fixture(scope="module")
def fixture_sim():
    return Microsimulation(dataset=str(FIXTURE_PATH))


@pytest.fixture(scope="module")
def fixture_entity_maps(fixture_sim):
    return build_household_entity_maps(fixture_sim)


def test_materialize_clone_household_chunk_preserves_entity_joins(
    tmp_path,
    monkeypatch,
    fixture_sim,
    fixture_entity_maps,
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
        lambda **kwargs: np.full(
            len(kwargs["spm_unit_tenure_types"]),
            12000.0,
            dtype=np.float32,
        ),
    )

    output_path = tmp_path / "clone_chunk.h5"
    summary = materialize_clone_household_chunk(
        sim=fixture_sim,
        entity_maps=fixture_entity_maps,
        active_hh=np.array([0, 1, 0], dtype=np.int64),
        active_blocks=np.array(
            [
                "371830501001001",
                "371830502002002",
                "371830503003003",
            ],
            dtype="U15",
        ),
        active_cd_geoids=np.array(["3701", "3701", "3702"], dtype=str),
        active_clone_indices=np.array([0, 0, 1], dtype=np.int64),
        output_path=output_path,
        apply_takeup=False,
    )

    assert summary.n_households == 3
    assert summary.n_persons > 3

    with h5py.File(output_path, "r") as h5:
        person_household_id = h5["person_household_id"]["2023"][:]
        household_id = h5["household_id"]["2023"][:]
        assert np.array_equal(household_id, np.array([0, 1, 2], dtype=np.int32))
        assert set(person_household_id).issubset(set(household_id))

        for entity_key in ("tax_unit", "spm_unit", "family", "marital_unit"):
            entity_ids = h5[f"{entity_key}_id"]["2023"][:]
            person_entity_ids = h5[f"person_{entity_key}_id"]["2023"][:]
            assert len(entity_ids) == len(set(entity_ids))
            assert set(person_entity_ids).issubset(set(entity_ids))


def test_materialize_clone_household_chunk_keeps_clone_specific_block_geoids(
    tmp_path,
    monkeypatch,
    fixture_sim,
    fixture_entity_maps,
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

    output_path = tmp_path / "clone_specific_blocks.h5"
    materialize_clone_household_chunk(
        sim=fixture_sim,
        entity_maps=fixture_entity_maps,
        active_hh=np.array([2, 2], dtype=np.int64),
        active_blocks=np.array(
            ["371830501001001", "371830501001999"],
            dtype="U15",
        ),
        active_cd_geoids=np.array(["3701", "3701"], dtype=str),
        active_clone_indices=np.array([0, 1], dtype=np.int64),
        output_path=output_path,
        apply_takeup=False,
    )

    with h5py.File(output_path, "r") as h5:
        blocks = h5["block_geoid"]["2023"][:].astype(str)
        assert blocks.tolist() == ["371830501001001", "371830501001999"]


def test_materialize_clone_household_chunk_writes_non_ascii_county_as_index(
    tmp_path,
    monkeypatch,
    fixture_sim,
    fixture_entity_maps,
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

    output_path = tmp_path / "non_ascii_county.h5"
    materialize_clone_household_chunk(
        sim=fixture_sim,
        entity_maps=fixture_entity_maps,
        active_hh=np.array([0], dtype=np.int64),
        active_blocks=np.array(["350130001001001"], dtype="U15"),
        active_cd_geoids=np.array(["3502"], dtype=str),
        active_clone_indices=np.array([0], dtype=np.int64),
        output_path=output_path,
        apply_takeup=False,
    )

    with h5py.File(output_path, "r") as h5:
        county = h5["county"]["2023"][:]
        assert np.issubdtype(county.dtype, np.integer)
        assert county.tolist() == [County._member_names_.index("DOÑA_ANA_COUNTY_NM")]


def test_materialize_clone_household_chunk_rejects_empty_blocks(
    tmp_path,
    fixture_sim,
    fixture_entity_maps,
):
    with pytest.raises(ValueError, match="empty block GEOIDs"):
        materialize_clone_household_chunk(
            sim=fixture_sim,
            entity_maps=fixture_entity_maps,
            active_hh=np.array([0], dtype=np.int64),
            active_blocks=np.array([""], dtype=str),
            active_cd_geoids=np.array(["3701"], dtype=str),
            active_clone_indices=np.array([0], dtype=np.int64),
            output_path=tmp_path / "bad.h5",
            apply_takeup=False,
        )
