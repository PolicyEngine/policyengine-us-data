import pytest
from pathlib import Path
from policyengine_us import Microsimulation
from policyengine_core.data import Dataset


STATE_FILES_DIR = Path("policyengine_us_data/storage/cd_states")

EXPECTED_CONGRESSIONAL_DISTRICTS = {
    "NC": 14,
    "CA": 52,
    "TX": 38,
    "FL": 28,
    "NY": 26,
    "PA": 17,
}


@pytest.mark.district_level_validation
@pytest.mark.parametrize(
    "state_code,expected_districts",
    [
        ("NC", 14),
        ("CA", 52),
        ("TX", 38),
        ("FL", 28),
        ("NY", 26),
        ("PA", 17),
    ],
)
def test_state_congressional_districts(state_code, expected_districts):
    state_file = STATE_FILES_DIR / f"{state_code}.h5"

    if not state_file.exists():
        pytest.skip(f"State file {state_code}.h5 not yet generated")

    dataset = Dataset.from_file(state_file)
    sim = Microsimulation(dataset=dataset)

    cd_geoids = sim.calculate("congressional_district_geoid")
    unique_districts = len(set(cd_geoids))

    assert unique_districts == expected_districts, (
        f"{state_code} should have {expected_districts} congressional districts, "
        f"but found {unique_districts}"
    )


@pytest.mark.district_level_validation
def test_nc_has_positive_weights():
    state_file = STATE_FILES_DIR / "NC.h5"

    if not state_file.exists():
        pytest.skip("NC.h5 not yet generated")

    dataset = Dataset.from_file(state_file)
    data = dataset.load_dataset()
    weights = data["household_weight"]["2023"]

    assert (weights > 0).all(), "All household weights should be positive"
    assert weights.sum() > 0, "Total weight should be positive"


@pytest.mark.district_level_validation
def test_nc_household_count_reasonable():
    state_file = STATE_FILES_DIR / "NC.h5"

    if not state_file.exists():
        pytest.skip("NC.h5 not yet generated")

    dataset = Dataset.from_file(state_file)
    data = dataset.load_dataset()
    weights = data["household_weight"]["2023"]

    total_households = weights.sum()

    NC_MIN_HOUSEHOLDS = 3_500_000
    NC_MAX_HOUSEHOLDS = 5_000_000

    assert NC_MIN_HOUSEHOLDS < total_households < NC_MAX_HOUSEHOLDS, (
        f"NC total weighted households ({total_households:,.0f}) outside "
        f"expected range ({NC_MIN_HOUSEHOLDS:,} - {NC_MAX_HOUSEHOLDS:,})"
    )


@pytest.mark.district_level_validation
def test_all_state_files_have_mapping_csv():
    state_files = list(STATE_FILES_DIR.glob("*.h5"))

    if not state_files:
        pytest.skip("No state files generated yet")

    for state_file in state_files:
        state_code = state_file.stem
        if state_code == "cd_calibration":
            continue

        mapping_file = STATE_FILES_DIR / f"{state_code}_household_mapping.csv"
        assert (
            mapping_file.exists()
        ), f"Missing household mapping CSV for {state_code}"
