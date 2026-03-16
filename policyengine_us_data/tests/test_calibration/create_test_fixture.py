"""
Create a minimal deterministic H5 test fixture for stacked_dataset_builder tests.

This creates a tiny dataset (~50 households) with known, fixed values that
won't change between test runs. The fixture is used to test the H5 creation
pipeline without depending on the full stratified CPS.

Run this script to regenerate the fixture:
    python create_test_fixture.py
"""

import numpy as np
import h5py
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "test_fixture_50hh.h5"
TIME_PERIOD = "2023"
N_HOUSEHOLDS = 50
SEED = 42


def create_test_fixture():
    np.random.seed(SEED)

    # Create household structure: 50 households with 1-4 persons each
    persons_per_hh = np.random.choice(
        [1, 2, 3, 4], size=N_HOUSEHOLDS, p=[0.3, 0.4, 0.2, 0.1]
    )
    n_persons = persons_per_hh.sum()

    # Household-level arrays
    household_ids = np.arange(N_HOUSEHOLDS, dtype=np.int32)
    household_weights = np.random.uniform(500, 3000, N_HOUSEHOLDS).astype(
        np.float32
    )

    # Assign households to states (use NC=37 and AK=2 for testing)
    # 40 households in NC, 10 in AK
    state_fips_hh = np.array([37] * 40 + [2] * 10, dtype=np.int32)

    # SNAP values: some households get SNAP, values differ by state
    # NC SNAP ~$200-800, AK SNAP ~$300-900 (higher cost of living)
    snap_hh = np.zeros(N_HOUSEHOLDS, dtype=np.float32)
    snap_recipients = np.random.choice(N_HOUSEHOLDS, size=15, replace=False)
    for hh_idx in snap_recipients:
        if state_fips_hh[hh_idx] == 37:  # NC
            snap_hh[hh_idx] = np.random.uniform(200, 800)
        else:  # AK
            snap_hh[hh_idx] = np.random.uniform(300, 900)

    # Person-level arrays
    person_ids = np.arange(n_persons, dtype=np.int32)
    person_household_ids = np.repeat(household_ids, persons_per_hh)
    person_weights = np.repeat(household_weights, persons_per_hh)

    # Age: mix of ages
    ages = np.random.choice(
        [5, 15, 25, 35, 45, 55, 65, 75],
        size=n_persons,
        p=[0.1, 0.1, 0.15, 0.2, 0.15, 0.15, 0.1, 0.05],
    ).astype(np.int32)

    # Employment income: working-age adults have income
    employment_income = np.zeros(n_persons, dtype=np.float32)
    working_age_mask = (ages >= 18) & (ages < 65)
    employment_income[working_age_mask] = np.random.uniform(
        0, 80000, working_age_mask.sum()
    )

    # Tax unit structure: simplify - each household is one tax unit
    tax_unit_ids = np.arange(N_HOUSEHOLDS, dtype=np.int32)
    person_tax_unit_ids = np.repeat(tax_unit_ids, persons_per_hh)
    tax_unit_weights = household_weights.copy()

    # SPM unit: same as household for simplicity
    spm_unit_ids = np.arange(N_HOUSEHOLDS, dtype=np.int32)
    person_spm_unit_ids = np.repeat(spm_unit_ids, persons_per_hh)
    spm_unit_weights = household_weights.copy()

    # Family: same as household
    family_ids = np.arange(N_HOUSEHOLDS, dtype=np.int32)
    person_family_ids = np.repeat(family_ids, persons_per_hh)
    family_weights = household_weights.copy()

    # Marital unit: each person is their own marital unit for simplicity
    marital_unit_ids = np.arange(n_persons, dtype=np.int32)
    person_marital_unit_ids = marital_unit_ids.copy()
    marital_unit_weights = person_weights.copy()

    # SNAP is at SPM unit level (same as household in our simple structure)
    snap_spm = snap_hh.copy()
    # state_fips is at household level

    # Write H5 file
    print(f"Creating test fixture: {FIXTURE_PATH}")
    print(f"  Households: {N_HOUSEHOLDS}")
    print(f"  Persons: {n_persons}")

    with h5py.File(FIXTURE_PATH, "w") as f:
        # Household variables
        f.create_group("household_id")
        f["household_id"].create_dataset(TIME_PERIOD, data=household_ids)

        f.create_group("household_weight")
        f["household_weight"].create_dataset(
            TIME_PERIOD, data=household_weights
        )

        # Person variables
        f.create_group("person_id")
        f["person_id"].create_dataset(TIME_PERIOD, data=person_ids)

        f.create_group("person_household_id")
        f["person_household_id"].create_dataset(
            TIME_PERIOD, data=person_household_ids
        )

        f.create_group("person_weight")
        f["person_weight"].create_dataset(TIME_PERIOD, data=person_weights)

        f.create_group("age")
        f["age"].create_dataset(TIME_PERIOD, data=ages)

        f.create_group("employment_income")
        f["employment_income"].create_dataset(
            TIME_PERIOD, data=employment_income
        )

        # Tax unit
        f.create_group("tax_unit_id")
        f["tax_unit_id"].create_dataset(TIME_PERIOD, data=tax_unit_ids)

        f.create_group("person_tax_unit_id")
        f["person_tax_unit_id"].create_dataset(
            TIME_PERIOD, data=person_tax_unit_ids
        )

        f.create_group("tax_unit_weight")
        f["tax_unit_weight"].create_dataset(TIME_PERIOD, data=tax_unit_weights)

        # SPM unit
        f.create_group("spm_unit_id")
        f["spm_unit_id"].create_dataset(TIME_PERIOD, data=spm_unit_ids)

        f.create_group("person_spm_unit_id")
        f["person_spm_unit_id"].create_dataset(
            TIME_PERIOD, data=person_spm_unit_ids
        )

        f.create_group("spm_unit_weight")
        f["spm_unit_weight"].create_dataset(TIME_PERIOD, data=spm_unit_weights)

        # Family
        f.create_group("family_id")
        f["family_id"].create_dataset(TIME_PERIOD, data=family_ids)

        f.create_group("person_family_id")
        f["person_family_id"].create_dataset(
            TIME_PERIOD, data=person_family_ids
        )

        f.create_group("family_weight")
        f["family_weight"].create_dataset(TIME_PERIOD, data=family_weights)

        # Marital unit
        f.create_group("marital_unit_id")
        f["marital_unit_id"].create_dataset(TIME_PERIOD, data=marital_unit_ids)

        f.create_group("person_marital_unit_id")
        f["person_marital_unit_id"].create_dataset(
            TIME_PERIOD, data=person_marital_unit_ids
        )

        f.create_group("marital_unit_weight")
        f["marital_unit_weight"].create_dataset(
            TIME_PERIOD, data=marital_unit_weights
        )

        # Geography (household level)
        f.create_group("state_fips")
        f["state_fips"].create_dataset(TIME_PERIOD, data=state_fips_hh)

        # SNAP (at SPM unit level)
        f.create_group("snap")
        f["snap"].create_dataset(TIME_PERIOD, data=snap_spm)

    print("Done!")

    # Verify
    with h5py.File(FIXTURE_PATH, "r") as f:
        print(f"\nVerification:")
        print(f"  Variables: {list(f.keys())}")
        print(f"  household_id shape: {f['household_id'][TIME_PERIOD].shape}")
        print(f"  person_id shape: {f['person_id'][TIME_PERIOD].shape}")


if __name__ == "__main__":
    create_test_fixture()
