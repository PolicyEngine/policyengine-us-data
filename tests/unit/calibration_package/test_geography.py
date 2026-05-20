import json
from types import SimpleNamespace

import numpy as np
import pytest

from policyengine_us_data.calibration_package.geography import (
    GEOGRAPHY_ASSIGNMENT_ORDERING,
    GeographyAssignmentResult,
    GeographyAssignmentSpec,
)
from policyengine_us_data.stage_contracts.calibration_package_schema import (
    GeographyAssignmentSummary,
)


def _assignment_for_seed(**kwargs):
    seed = int(kwargs["seed"])
    n_records = int(kwargs["n_records"])
    n_clones = int(kwargs["n_clones"])
    n_rows = n_records * n_clones
    start_state = (seed % 50) + 1
    states = np.array(
        [((start_state + index) % 50) + 1 for index in range(n_rows)],
        dtype=np.int32,
    )
    blocks = np.array(
        [f"{state:02d}0010001001000" for state in states],
        dtype="<U15",
    )
    return SimpleNamespace(
        block_geoid=blocks,
        cd_geoid=np.array([f"{state:02d}01" for state in states], dtype="<U4"),
        county_fips=np.array([block[:5] for block in blocks], dtype="<U5"),
        state_fips=states,
    )


def _spec(seed: int = 42) -> GeographyAssignmentSpec:
    return GeographyAssignmentSpec.from_runtime_inputs(
        n_records=2,
        n_clones=2,
        seed=seed,
        household_agi=np.array([10_000.0, 250_000.0]),
        cd_agi_targets={"0101": 1_000.0, "0201": 2_000.0},
        fixed_state_fips=np.array([1, 0], dtype=np.int32),
    )


def test_geography_assignment_spec_records_runtime_input_identity():
    spec = _spec()
    payload = spec.to_dict()

    assert payload["n_records"] == 2
    assert payload["n_clones"] == 2
    assert payload["seed"] == 42
    assert payload["household_agi_sha256"].startswith("sha256:")
    assert payload["cd_agi_targets_sha256"].startswith("sha256:")
    assert payload["fixed_state_fips_present_count"] == 1
    assert GeographyAssignmentSpec.from_dict(payload) == spec


def test_geography_assignment_is_deterministic_for_fixed_seed():
    first = _spec(seed=42).assign(
        household_agi=np.array([10_000.0, 250_000.0]),
        cd_agi_targets={"0101": 1_000.0, "0201": 2_000.0},
        fixed_state_fips=np.array([1, 0], dtype=np.int32),
        assigner=_assignment_for_seed,
    )
    second = _spec(seed=42).assign(
        household_agi=np.array([10_000.0, 250_000.0]),
        cd_agi_targets={"0101": 1_000.0, "0201": 2_000.0},
        fixed_state_fips=np.array([1, 0], dtype=np.int32),
        assigner=_assignment_for_seed,
    )

    assert first.canonical_geography_sha256 == second.canonical_geography_sha256
    assert first.summary() == second.summary()


def test_geography_assignment_checksum_changes_with_seed():
    first = _spec(seed=42).assign(
        household_agi=np.array([10_000.0, 250_000.0]),
        cd_agi_targets={"0101": 1_000.0, "0201": 2_000.0},
        fixed_state_fips=np.array([1, 0], dtype=np.int32),
        assigner=_assignment_for_seed,
    )
    second = _spec(seed=43).assign(
        household_agi=np.array([10_000.0, 250_000.0]),
        cd_agi_targets={"0101": 1_000.0, "0201": 2_000.0},
        fixed_state_fips=np.array([1, 0], dtype=np.int32),
        assigner=_assignment_for_seed,
    )

    assert first.canonical_geography_sha256 != second.canonical_geography_sha256


def test_geography_assignment_validates_row_count_and_ordering():
    spec = _spec()

    with pytest.raises(ValueError, match="block_geoid length 3"):
        GeographyAssignmentResult.from_arrays(
            spec=spec,
            block_geoid=np.array(["010010001", "010010002", "010010003"]),
            cd_geoid=np.array(["0101", "0101", "0101"]),
        )

    result = GeographyAssignmentResult.from_arrays(
        spec=spec,
        block_geoid=np.array(
            [
                "010010001001000",
                "020010001001000",
                "010010001001001",
                "020010001001001",
            ]
        ),
        cd_geoid=np.array(["0101", "0201", "0101", "0201"]),
    )

    assert result.summary()["ordering"] == GEOGRAPHY_ASSIGNMENT_ORDERING
    assert result.summary()["n_rows"] == 4


def test_geography_summary_json_round_trips_through_contract_schema(tmp_path):
    result = _spec().assign(
        household_agi=np.array([10_000.0, 250_000.0]),
        cd_agi_targets={"0101": 1_000.0, "0201": 2_000.0},
        fixed_state_fips=np.array([1, 0], dtype=np.int32),
        assigner=_assignment_for_seed,
    )

    summary_path = result.write_summary(tmp_path / "geography_assignment_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert GeographyAssignmentSummary.from_dict(summary).to_dict() == summary
    assert summary["block_geoid_unique_count"] == 4
    assert summary["county_fips_sha256"].startswith("sha256:")
    assert summary["state_fips_sha256"].startswith("sha256:")
