"""
Create a representative test fixture (~4,000 households) for local area
calibration testing.

This creates a dataset sampled from real CPS data (extended_cps_2024.h5) that:
- Covers all 436 congressional districts (min 9 households per CD)
- Covers all 51 states (follows from CD coverage)
- Includes income diversity (all quintiles + top 1%)
- Includes entity composition diversity (multi-person, multi-tax-unit, etc.)
- Has pre-assigned counties via block_assignment.py

Unlike the synthetic test_fixture_50hh.h5, this uses real CPS microdata for
more realistic testing of the calibration pipeline.

Run this script to regenerate the fixture:
    python create_representative_fixture.py

Requirements:
    - extended_cps_2024.h5 in storage folder
    - policy_data.db in storage/calibration folder
    - block_cd_distributions.csv.gz in storage folder
"""

import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict

from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import Enum

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_all_cds_from_database,
    get_pseudo_input_variables,
)
from policyengine_us_data.datasets.cps.local_area_calibration.block_assignment import (
    assign_geography_for_cd,
    _get_block_distributions,
    get_county_fips_from_block,
    get_county_enum_index_from_block,
    get_state_fips_from_block,
    get_tract_geoid_from_block,
    get_cbsa_from_county,
    get_all_geography_from_block,
)


def _make_geo_from_block(cd_geoid: str, block_geoid: str) -> dict:
    """Create geography dict from a single block GEOID."""
    county_fips = get_county_fips_from_block(block_geoid)
    extra_geo = get_all_geography_from_block(block_geoid)

    return {
        "block_geoid": np.array([block_geoid]),
        "county_fips": np.array([county_fips]),
        "tract_geoid": np.array([get_tract_geoid_from_block(block_geoid)]),
        "state_fips": np.array([get_state_fips_from_block(block_geoid)]),
        "cbsa_code": np.array([get_cbsa_from_county(county_fips) or ""]),
        "county_index": np.array(
            [get_county_enum_index_from_block(block_geoid)], dtype=np.int32
        ),
        "sldu": np.array([extra_geo["sldu"] or ""]),
        "sldl": np.array([extra_geo["sldl"] or ""]),
        "place_fips": np.array([extra_geo["place_fips"] or ""]),
        "vtd": np.array([extra_geo["vtd"] or ""]),
        "puma": np.array([extra_geo["puma"] or ""]),
        "zcta": np.array([extra_geo["zcta"] or ""]),
    }


# Configuration
FIXTURE_PATH = Path(__file__).parent / "test_fixture_4k_representative.h5"
TARGET_HOUSEHOLDS = 4000
MIN_HOUSEHOLDS_PER_CD = (
    1  # Minimum floor to ensure every CD has representation
)
SEED = 12345
TIME_PERIOD = 2024


def get_household_entity_composition(sim):
    """
    Calculate entity composition metrics for each household.

    Returns:
        Dict with arrays for each household:
        - n_persons: number of persons
        - n_tax_units: number of tax units
        - n_spm_units: number of SPM units
    """
    household_ids = sim.calculate("household_id", map_to="household").values
    n_households = len(household_ids)
    hh_id_to_idx = {hh_id: idx for idx, hh_id in enumerate(household_ids)}

    # Get person-level mappings
    person_hh_ids = sim.calculate(
        "person_household_id", map_to="person"
    ).values
    person_tax_unit_ids = sim.calculate(
        "person_tax_unit_id", map_to="person"
    ).values
    person_spm_unit_ids = sim.calculate(
        "person_spm_unit_id", map_to="person"
    ).values

    # Initialize counters
    n_persons = np.zeros(n_households, dtype=np.int32)
    # Use sets to track unique tax units and SPM units per household
    hh_tax_units = {hh_id: set() for hh_id in household_ids}
    hh_spm_units = {hh_id: set() for hh_id in household_ids}

    # Count persons and collect unique entity IDs per household
    for i, hh_id in enumerate(person_hh_ids):
        if hh_id in hh_id_to_idx:
            idx = hh_id_to_idx[hh_id]
            n_persons[idx] += 1
            hh_tax_units[hh_id].add(person_tax_unit_ids[i])
            hh_spm_units[hh_id].add(person_spm_unit_ids[i])

    # Convert sets to counts
    n_tax_units = np.array(
        [len(hh_tax_units[hh_id]) for hh_id in household_ids], dtype=np.int32
    )
    n_spm_units = np.array(
        [len(hh_spm_units[hh_id]) for hh_id in household_ids], dtype=np.int32
    )

    return {
        "n_persons": n_persons,
        "n_tax_units": n_tax_units,
        "n_spm_units": n_spm_units,
    }


def create_representative_fixture(
    base_dataset=None,
    output_path=None,
    target_households=TARGET_HOUSEHOLDS,
    min_per_cd=MIN_HOUSEHOLDS_PER_CD,
    seed=SEED,
):
    """
    Create a representative test fixture by sampling from real CPS data.

    Sampling strategy:
    1. CD-Based Sampling: Sample households proportionally to county count
       (CDs with more counties get more households for better coverage)
    2. Income Enrichment: Add households from top 1% and bottom 20% if budget
    3. Entity Enrichment: Ensure multi-person/tax-unit/SPM-unit households

    Args:
        base_dataset: Path to source h5 file (default: extended_cps_2024.h5)
        output_path: Where to save the fixture (default: test_fixture_4k.h5)
        target_households: Target number of households
        min_per_cd: Minimum households per congressional district
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    print("\n" + "=" * 70)
    print("CREATING REPRESENTATIVE TEST FIXTURE")
    print("=" * 70)

    # Set defaults
    if base_dataset is None:
        base_dataset = str(STORAGE_FOLDER / "extended_cps_2024.h5")
    if output_path is None:
        output_path = FIXTURE_PATH

    # Load source dataset
    print(f"\nLoading source dataset: {base_dataset}")
    sim = Microsimulation(dataset=base_dataset)

    # Get household-level data
    household_ids = sim.calculate("household_id", map_to="household").values
    household_weights = sim.calculate(
        "household_weight", map_to="household"
    ).values
    state_fips = sim.calculate("state_fips", map_to="household").values
    n_households_orig = len(household_ids)

    print(f"Source dataset: {n_households_orig:,} households")
    print(f"Target dataset: {target_households:,} households")

    # Calculate household AGI for income stratification
    agi = sim.calculate("adjusted_gross_income", map_to="household").values

    # Calculate entity composition
    print("Analyzing entity composition...")
    entity_comp = get_household_entity_composition(sim)

    # Get all CDs from database
    print("\nLoading congressional districts from database...")
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    db_uri = f"sqlite:///{db_path}"

    try:
        all_cds = get_all_cds_from_database(db_uri)
        print(f"Found {len(all_cds)} congressional districts")
    except Exception as e:
        print(f"Warning: Could not load CDs from database: {e}")
        print("Using state-based sampling fallback...")
        all_cds = None

    # Build state -> household indices mapping
    state_to_hh_indices = defaultdict(list)
    for idx, st in enumerate(state_fips):
        state_to_hh_indices[int(st)].append(idx)

    print(f"Households distributed across {len(state_to_hh_indices)} states")

    # Phase 1: CD-Based Sampling (ensure all counties covered)
    print("\n--- Phase 1: CD-Based Sampling (county coverage) ---")
    selected_indices = set()
    cd_assignments = {}  # hh_idx -> cd_geoid

    # Initialize for use in Phase 4
    block_distributions = _get_block_distributions()
    cd_county_sets = {}
    counties_per_cd = {}

    if all_cds is not None:
        # Count counties per CD from block distributions
        print("  Analyzing counties per CD...")
        for cd in all_cds:
            cd_key = str(int(cd))
            if cd_key in block_distributions:
                blocks = block_distributions[cd_key].keys()
                counties = set(get_county_fips_from_block(b) for b in blocks)
                counties_per_cd[cd] = len(counties)
                cd_county_sets[cd] = counties
            else:
                counties_per_cd[cd] = 1
                cd_county_sets[cd] = set()

        total_county_cd_pairs = sum(counties_per_cd.values())
        print(f"  Total county-CD pairs: {total_county_cd_pairs}")

        # Sample at least 1 household per county in each CD to ensure coverage
        # This guarantees we can assign one household to each county
        # Track remaining households per state (to avoid overwriting assignments)
        state_remaining_indices = {
            state: set(indices)
            for state, indices in state_to_hh_indices.items()
        }

        for cd_geoid in all_cds:
            state = int(cd_geoid) // 100
            remaining = state_remaining_indices.get(state, set())

            if len(remaining) == 0:
                print(f"  Warning: No remaining households for CD {cd_geoid}")
                continue

            # Sample at least as many households as counties in this CD
            n_counties = counties_per_cd.get(cd_geoid, 1)
            n_to_sample = max(min_per_cd, n_counties)

            # Weight-proportional sampling from REMAINING households
            hh_indices = np.array(list(remaining))
            weights = household_weights[hh_indices]
            weights = weights / weights.sum()

            n_to_sample = min(n_to_sample, len(hh_indices))
            sampled = np.random.choice(
                hh_indices,
                size=n_to_sample,
                replace=False,
                p=weights,
            )

            for idx in sampled:
                selected_indices.add(idx)
                cd_assignments[idx] = cd_geoid
                remaining.discard(idx)  # Remove from available pool

        print(f"  Selected {len(selected_indices):,} households across CDs")
    else:
        # Fallback: sample proportionally from each state
        for state, hh_indices in state_to_hh_indices.items():
            hh_indices = np.array(hh_indices)
            weights = household_weights[hh_indices]
            weights = weights / weights.sum()

            n_to_sample = max(
                min_per_cd,
                int(len(hh_indices) * target_households / n_households_orig),
            )
            n_to_sample = min(n_to_sample, len(hh_indices))

            sampled = np.random.choice(
                hh_indices,
                size=n_to_sample,
                replace=False,
                p=weights,
            )
            selected_indices.update(sampled)

        print(f"  Selected {len(selected_indices):,} households (state-based)")

    # Phase 2: Income Enrichment
    print("\n--- Phase 2: Income Enrichment ---")
    remaining_budget = target_households - len(selected_indices)

    if remaining_budget > 0:
        # Identify top 1% and bottom 20%
        top_1_threshold = np.percentile(agi, 99)
        bottom_20_threshold = np.percentile(agi, 20)

        top_1_indices = set(np.where(agi >= top_1_threshold)[0])
        bottom_20_indices = set(np.where(agi <= bottom_20_threshold)[0])

        # Add missing top 1% households
        missing_top_1 = top_1_indices - selected_indices
        n_add_top = min(len(missing_top_1), remaining_budget // 3)
        if n_add_top > 0:
            to_add = np.random.choice(
                list(missing_top_1), size=n_add_top, replace=False
            )
            selected_indices.update(to_add)
            print(f"  Added {n_add_top} top 1% households")

        # Add missing bottom 20% households
        remaining_budget = target_households - len(selected_indices)
        missing_bottom = bottom_20_indices - selected_indices
        n_add_bottom = min(len(missing_bottom), remaining_budget // 2)
        if n_add_bottom > 0:
            to_add = np.random.choice(
                list(missing_bottom), size=n_add_bottom, replace=False
            )
            selected_indices.update(to_add)
            print(f"  Added {n_add_bottom} bottom 20% households")
    else:
        print("  No budget remaining for income enrichment")

    # Phase 3: Entity Composition Diversity
    print("\n--- Phase 3: Entity Composition Diversity ---")
    remaining_budget = target_households - len(selected_indices)

    if remaining_budget > 0:
        # Ensure diverse household sizes (1, 2, 3, 4+ persons)
        for size, label in [(1, "1-person"), (2, "2-person"), (3, "3-person")]:
            if remaining_budget <= 0:
                break
            size_mask = entity_comp["n_persons"] == size
            size_indices = set(np.where(size_mask)[0])
            missing = size_indices - selected_indices
            # Add a few of each size to ensure diversity
            n_add = min(len(missing), max(10, remaining_budget // 10))
            if n_add > 0:
                to_add = np.random.choice(
                    list(missing), size=n_add, replace=False
                )
                selected_indices.update(to_add)
                print(f"  Added {n_add} {label} households")
                remaining_budget = target_households - len(selected_indices)

        # Find multi-tax-unit households (important for tax analysis)
        remaining_budget = target_households - len(selected_indices)
        multi_tu_mask = entity_comp["n_tax_units"] >= 2
        multi_tu_indices = set(np.where(multi_tu_mask)[0])
        missing_multi_tu = multi_tu_indices - selected_indices
        n_add = min(len(missing_multi_tu), remaining_budget // 3)
        if n_add > 0:
            to_add = np.random.choice(
                list(missing_multi_tu), size=n_add, replace=False
            )
            selected_indices.update(to_add)
            print(f"  Added {n_add} multi-tax-unit households")

        # Find multi-SPM-unit households (important for poverty analysis)
        remaining_budget = target_households - len(selected_indices)
        multi_spm_mask = entity_comp["n_spm_units"] >= 2
        multi_spm_indices = set(np.where(multi_spm_mask)[0])
        missing_multi_spm = multi_spm_indices - selected_indices
        n_add = min(len(missing_multi_spm), remaining_budget // 3)
        if n_add > 0:
            to_add = np.random.choice(
                list(missing_multi_spm), size=n_add, replace=False
            )
            selected_indices.update(to_add)
            print(f"  Added {n_add} multi-SPM-unit households")

    print(f"\nTotal selected: {len(selected_indices):,} households")

    # Phase 4: Assign CD and geography for each household
    print(
        "\n--- Phase 4: Geographic Assignment (ensuring county coverage) ---"
    )

    # For households without CD assignment, assign based on state
    selected_list = sorted(selected_indices)

    # Build state -> CDs mapping
    state_to_cds = defaultdict(list)
    if all_cds is not None:
        for cd in all_cds:
            state = int(cd) // 100
            state_to_cds[state].append(cd)

    # Assign CDs to households that don't have one
    for idx in selected_list:
        if idx not in cd_assignments:
            state = int(state_fips[idx])
            cds_in_state = state_to_cds.get(state, [])
            if cds_in_state:
                cd_assignments[idx] = np.random.choice(cds_in_state)
            else:
                # Create synthetic CD (state * 100 + 1 for at-large)
                cd_assignments[idx] = str(state * 100 + 1)

    # Group households by CD for smart county assignment
    cd_to_household_indices = defaultdict(list)
    for i, idx in enumerate(selected_list):
        cd = cd_assignments[idx]
        cd_to_household_indices[cd].append(i)

    # Assign geography ensuring each county in a CD gets at least one household
    print("  Assigning block-level geography with county coverage...")
    geography_data = {
        "congressional_district_geoid": [None] * len(selected_list),
        "county": [None] * len(selected_list),
        "county_fips": [None] * len(selected_list),
        "tract_geoid": [None] * len(selected_list),
        "block_geoid": [None] * len(selected_list),
        "state_fips_assigned": [None] * len(selected_list),
        "cbsa_code": [None] * len(selected_list),
        "sldu": [None] * len(selected_list),
        "sldl": [None] * len(selected_list),
        "place_fips": [None] * len(selected_list),
        "vtd": [None] * len(selected_list),
        "puma": [None] * len(selected_list),
        "zcta": [None] * len(selected_list),
    }

    processed = 0
    for cd, hh_list_indices in cd_to_household_indices.items():
        cd_key = str(int(cd))
        counties_in_cd = list(cd_county_sets.get(cd, set()))

        # Assign one household to each county first (if we have enough)
        county_assigned = 0
        for i, list_idx in enumerate(hh_list_indices):
            idx = selected_list[list_idx]

            if i < len(counties_in_cd):
                # Deterministic assignment to ensure county coverage
                target_county_fips = counties_in_cd[i]
                # Get a block in this county from the distribution
                if cd_key in block_distributions:
                    blocks_in_county = [
                        b
                        for b in block_distributions[cd_key].keys()
                        if get_county_fips_from_block(b) == target_county_fips
                    ]
                    if blocks_in_county:
                        # Pick a block weighted by population
                        block_weights = [
                            block_distributions[cd_key][b]
                            for b in blocks_in_county
                        ]
                        total_w = sum(block_weights)
                        block_weights = [w / total_w for w in block_weights]
                        np.random.seed(int(seed + idx))
                        block = np.random.choice(
                            blocks_in_county, p=block_weights
                        )
                        geo = _make_geo_from_block(cd, block)
                        county_assigned += 1
                    else:
                        geo = assign_geography_for_cd(
                            cd, 1, seed=int(seed + idx)
                        )
                else:
                    geo = assign_geography_for_cd(cd, 1, seed=int(seed + idx))
            else:
                # Random assignment for remaining households
                geo = assign_geography_for_cd(cd, 1, seed=int(seed + idx))

            geography_data["congressional_district_geoid"][list_idx] = int(cd)
            geography_data["county"][list_idx] = geo["county_index"][0]
            geography_data["county_fips"][list_idx] = geo["county_fips"][0]
            geography_data["tract_geoid"][list_idx] = geo["tract_geoid"][0]
            geography_data["block_geoid"][list_idx] = geo["block_geoid"][0]
            geography_data["state_fips_assigned"][list_idx] = int(
                geo["state_fips"][0]
            )
            geography_data["cbsa_code"][list_idx] = geo["cbsa_code"][0]
            geography_data["sldu"][list_idx] = geo["sldu"][0]
            geography_data["sldl"][list_idx] = geo["sldl"][0]
            geography_data["place_fips"][list_idx] = geo["place_fips"][0]
            geography_data["vtd"][list_idx] = geo["vtd"][0]
            geography_data["puma"][list_idx] = geo["puma"][0]
            geography_data["zcta"][list_idx] = geo["zcta"][0]

            processed += 1
            if processed % 500 == 0:
                print(f"  Processed {processed:,}/{len(selected_list):,}")

    print(f"  Geographic assignment complete")

    # Convert to numpy arrays (use bytes for strings to be h5py compatible)
    for key in geography_data:
        if key in [
            "county",
            "congressional_district_geoid",
            "state_fips_assigned",
        ]:
            geography_data[key] = np.array(geography_data[key], dtype=np.int32)
        else:
            # Convert strings to bytes for h5py compatibility
            geography_data[key] = np.array(geography_data[key], dtype="S")

    # Create filtered dataset
    print("\n--- Creating Filtered Dataset ---")
    selected_household_ids = set(household_ids[selected_list])
    time_period = int(sim.default_calculation_period)

    # Convert to DataFrame and filter
    df = sim.to_input_dataframe()
    hh_id_col = f"household_id__{time_period}"
    df_filtered = df[df[hh_id_col].isin(selected_household_ids)].copy()

    print(f"Filtered DataFrame: {len(df_filtered):,} persons")

    # Create Dataset from filtered DataFrame
    print("Creating Dataset from filtered DataFrame...")
    filtered_dataset = Dataset.from_dataframe(df_filtered, time_period)

    # Build simulation
    print("Building simulation from Dataset...")
    filtered_sim = Microsimulation()
    filtered_sim.dataset = filtered_dataset
    filtered_sim.build_from_dataset()

    # Save to H5 file
    print(f"\nSaving to {output_path}...")
    data = {}

    # Get input variables (excluding pseudo-inputs)
    input_vars = set(sim.input_variables)
    pseudo_inputs = get_pseudo_input_variables(sim)
    if pseudo_inputs:
        print(f"Excluding {len(pseudo_inputs)} pseudo-input variables")
        input_vars = input_vars - pseudo_inputs

    print(f"Saving {len(input_vars)} input variables")

    for variable in filtered_sim.tax_benefit_system.variables:
        if variable not in input_vars:
            continue

        data[variable] = {}
        for period in filtered_sim.get_holder(variable).get_known_periods():
            values = filtered_sim.get_holder(variable).get_array(period)

            # Handle different value types
            if variable == "county_fips":
                values = values.astype("int32")
            elif filtered_sim.tax_benefit_system.variables.get(
                variable
            ).value_type in (Enum, str):
                if hasattr(values, "decode_to_str"):
                    values = values.decode_to_str().astype("S")
                else:
                    values = values.astype("S")
            else:
                values = np.array(values)

            if values is not None:
                data[variable][period] = values

        if len(data[variable]) == 0:
            del data[variable]

    # Write to H5
    with h5py.File(output_path, "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    print("Dataset saved successfully!")

    # Note: Geographic assignments are not stored in H5 to avoid conflicts
    # with policyengine loading. They can be regenerated using the same seed.

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION STATISTICS")
    print("=" * 70)

    # Load and verify
    with h5py.File(output_path, "r") as f:
        hh_ids = f["household_id"][str(time_period)][:]
        person_ids = f["person_id"][str(time_period)][:]
        print(f"\nTotal households: {len(hh_ids):,}")
        print(f"Total persons: {len(person_ids):,}")

    # Geographic coverage
    selected_states_unique = np.unique(geography_data["state_fips_assigned"])
    selected_cds_unique = np.unique(
        geography_data["congressional_district_geoid"]
    )
    selected_counties_unique = np.unique(geography_data["county"])

    # Total counts for percentages
    total_states = 51
    total_cds = 436
    total_counties = 3172  # US counties (excluding UNKNOWN)

    print(f"\nGeographic Coverage:")
    print(
        f"  States covered: {len(selected_states_unique)}/{total_states} "
        f"({100*len(selected_states_unique)/total_states:.1f}%)"
    )
    print(
        f"  CDs covered: {len(selected_cds_unique)}/{total_cds} "
        f"({100*len(selected_cds_unique)/total_cds:.1f}%)"
    )
    print(
        f"  Unique counties: {len(selected_counties_unique)}/{total_counties} "
        f"({100*len(selected_counties_unique)/total_counties:.1f}%)"
    )

    # Households per state
    state_counts = np.bincount(geography_data["state_fips_assigned"])
    state_counts = state_counts[state_counts > 0]
    print(
        f"  Households per state: min={state_counts.min()}, "
        f"max={state_counts.max()}, median={np.median(state_counts):.0f}"
    )

    # Income distribution
    selected_agi = agi[selected_list]
    print(f"\nIncome Distribution:")
    for p in [0, 20, 40, 60, 80, 99, 100]:
        val = np.percentile(selected_agi, p)
        print(f"  {p:3d}th percentile: ${val:>12,.0f}")

    top_1_in_selected = np.sum(selected_agi >= np.percentile(agi, 99))
    bottom_20_in_selected = np.sum(selected_agi <= np.percentile(agi, 20))
    print(f"  Top 1% households: {top_1_in_selected}")
    print(f"  Bottom 20% households: {bottom_20_in_selected}")

    # Entity composition
    selected_n_persons = entity_comp["n_persons"][selected_list]
    selected_n_tu = entity_comp["n_tax_units"][selected_list]
    selected_n_spm = entity_comp["n_spm_units"][selected_list]

    print(f"\nEntity Composition:")
    print(f"  1-person households: {np.sum(selected_n_persons == 1)}")
    print(f"  2-person households: {np.sum(selected_n_persons == 2)}")
    print(f"  3-person households: {np.sum(selected_n_persons == 3)}")
    print(f"  4+ person households: {np.sum(selected_n_persons >= 4)}")
    print(f"  Multi-tax-unit households: {np.sum(selected_n_tu >= 2)}")
    print(f"  Multi-SPM-unit households: {np.sum(selected_n_spm >= 2)}")

    print("\n" + "=" * 70)
    print("FIXTURE CREATION COMPLETE")
    print("=" * 70)

    return str(output_path)


if __name__ == "__main__":
    import sys

    target = TARGET_HOUSEHOLDS
    min_cd = MIN_HOUSEHOLDS_PER_CD
    seed_val = SEED

    for arg in sys.argv[1:]:
        if arg.startswith("--target="):
            target = int(arg.split("=")[1])
        elif arg.startswith("--min-cd="):
            min_cd = int(arg.split("=")[1])
        elif arg.startswith("--seed="):
            seed_val = int(arg.split("=")[1])

    print(f"Creating representative fixture:")
    print(f"  Target households: {target:,}")
    print(f"  Min per CD: {min_cd}")
    print(f"  Seed: {seed_val}")

    output = create_representative_fixture(
        target_households=target,
        min_per_cd=min_cd,
        seed=seed_val,
    )

    print(f"\nDone! Created: {output}")
