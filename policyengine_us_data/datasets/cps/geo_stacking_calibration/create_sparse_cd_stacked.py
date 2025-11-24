"""
Create a sparse congressional district-stacked dataset with only non-zero weight households.
Standalone version that doesn't modify the working state stacking code.
"""

## Testing with this:
#output_dir = "national"
#dataset_path_str = "/home/baogorek/devl/stratified_10k.h5"
#db_path = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"
#weights_path_str = "national/w_cd_20251031_122119.npy"
#include_full_dataset = True
## end testing lines --

import sys
import numpy as np
import pandas as pd
import h5py
import os
import json
import random
from pathlib import Path
from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import Enum
from sqlalchemy import create_engine, text
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import (
    download_from_huggingface,
    get_cd_index_mapping,
    get_id_range_for_cd,
    get_cd_from_id,
)
from policyengine_us.variables.household.demographic.geographic.state_name import (
    StateName,
)
from policyengine_us.variables.household.demographic.geographic.state_code import (
    StateCode,
)
from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
    County,
)


# TODO: consolidate mappings
STATE_CODES = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}

# State FIPS to StateName and StateCode mappings
STATE_FIPS_TO_NAME = {
    1: StateName.AL,
    2: StateName.AK,
    4: StateName.AZ,
    5: StateName.AR,
    6: StateName.CA,
    8: StateName.CO,
    9: StateName.CT,
    10: StateName.DE,
    11: StateName.DC,
    12: StateName.FL,
    13: StateName.GA,
    15: StateName.HI,
    16: StateName.ID,
    17: StateName.IL,
    18: StateName.IN,
    19: StateName.IA,
    20: StateName.KS,
    21: StateName.KY,
    22: StateName.LA,
    23: StateName.ME,
    24: StateName.MD,
    25: StateName.MA,
    26: StateName.MI,
    27: StateName.MN,
    28: StateName.MS,
    29: StateName.MO,
    30: StateName.MT,
    31: StateName.NE,
    32: StateName.NV,
    33: StateName.NH,
    34: StateName.NJ,
    35: StateName.NM,
    36: StateName.NY,
    37: StateName.NC,
    38: StateName.ND,
    39: StateName.OH,
    40: StateName.OK,
    41: StateName.OR,
    42: StateName.PA,
    44: StateName.RI,
    45: StateName.SC,
    46: StateName.SD,
    47: StateName.TN,
    48: StateName.TX,
    49: StateName.UT,
    50: StateName.VT,
    51: StateName.VA,
    53: StateName.WA,
    54: StateName.WV,
    55: StateName.WI,
    56: StateName.WY,
}

# Note that this is not exactly the same as above: StateName vs StateCode
STATE_FIPS_TO_CODE = {
    1: StateCode.AL,
    2: StateCode.AK,
    4: StateCode.AZ,
    5: StateCode.AR,
    6: StateCode.CA,
    8: StateCode.CO,
    9: StateCode.CT,
    10: StateCode.DE,
    11: StateCode.DC,
    12: StateCode.FL,
    13: StateCode.GA,
    15: StateCode.HI,
    16: StateCode.ID,
    17: StateCode.IL,
    18: StateCode.IN,
    19: StateCode.IA,
    20: StateCode.KS,
    21: StateCode.KY,
    22: StateCode.LA,
    23: StateCode.ME,
    24: StateCode.MD,
    25: StateCode.MA,
    26: StateCode.MI,
    27: StateCode.MN,
    28: StateCode.MS,
    29: StateCode.MO,
    30: StateCode.MT,
    31: StateCode.NE,
    32: StateCode.NV,
    33: StateCode.NH,
    34: StateCode.NJ,
    35: StateCode.NM,
    36: StateCode.NY,
    37: StateCode.NC,
    38: StateCode.ND,
    39: StateCode.OH,
    40: StateCode.OK,
    41: StateCode.OR,
    42: StateCode.PA,
    44: StateCode.RI,
    45: StateCode.SC,
    46: StateCode.SD,
    47: StateCode.TN,
    48: StateCode.TX,
    49: StateCode.UT,
    50: StateCode.VT,
    51: StateCode.VA,
    53: StateCode.WA,
    54: StateCode.WV,
    55: StateCode.WI,
    56: StateCode.WY,
}


def load_cd_county_mappings():
    """Load CD to county mappings from JSON file."""
    #script_dir = Path(__file__).parent
    #mapping_file = script_dir / "cd_county_mappings.json"
    mapping_file = Path.cwd() / "cd_county_mappings.json"
    if not mapping_file.exists():
        print(
            "WARNING: cd_county_mappings.json not found. Counties will not be updated."
        )
        return None

    with open(mapping_file, "r") as f:
        return json.load(f)


def get_county_for_cd(cd_geoid, cd_county_mappings):
    """
    Get a county FIPS code for a given congressional district.
    Uses weighted random selection based on county proportions.
    """
    if not cd_county_mappings or str(cd_geoid) not in cd_county_mappings:
        return None

    county_props = cd_county_mappings[str(cd_geoid)]
    if not county_props:
        return None

    counties = list(county_props.keys())
    weights = list(county_props.values())

    # Normalize weights to ensure they sum to 1
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
        return random.choices(counties, weights=weights)[0]

    return None


def create_sparse_cd_stacked_dataset(
    w,
    cds_to_calibrate,
    cd_subset=None,
    output_path=None,
    dataset_path=None,
    freeze_calculated_vars=False,
):
    """
    Create a SPARSE congressional district-stacked dataset using DataFrame approach.

    Args:
        w: Calibrated weight vector from L0 calibration (length = n_households * n_cds)
        cds_to_calibrate: List of CD GEOID codes used in calibration
        cd_subset: Optional list of CD GEOIDs to include (subset of cds_to_calibrate)
        output_path: Where to save the sparse CD-stacked h5 file
        dataset_path: Path to the base .h5 dataset used to create the training matrices
        freeze_calculated_vars: If True, save calculated variables (like SNAP) to h5 file so they're not recalculated on load.
                               If False (default), calculated variables are omitted and will be recalculated on load.
    """

    # Handle CD subset filtering
    if cd_subset is not None:
        # Validate that requested CDs are in the calibration
        for cd in cd_subset:
            if cd not in cds_to_calibrate:
                raise ValueError(f"CD {cd} not in calibrated CDs list")

        # Get indices of requested CDs
        cd_indices = [cds_to_calibrate.index(cd) for cd in cd_subset]
        cds_to_process = cd_subset

        print(
            f"Processing subset of {len(cd_subset)} CDs: {', '.join(cd_subset[:5])}..."
        )
    else:
        # Process all CDs
        cd_indices = list(range(len(cds_to_calibrate)))
        cds_to_process = cds_to_calibrate
        print(
            f"Processing all {len(cds_to_calibrate)} congressional districts"
        )

    # Generate output path if not provided
    if output_path is None:
        raise ValueError("No output .h5 path given")
    print(f"Output path: {output_path}")

    # Load the original simulation
    base_sim = Microsimulation(dataset=dataset_path)

    cd_county_mappings = load_cd_county_mappings()

    household_ids = base_sim.calculate(
        "household_id", map_to="household"
    ).values
    n_households_orig = len(household_ids)

    # From the base sim, create mapping from household ID to index for proper filtering
    hh_id_to_idx = {int(hh_id): idx for idx, hh_id in enumerate(household_ids)}

    # I.e.,  
    # {25: 0,
    #  78: 1,
    #  103: 2,
    #  125: 3,

    # Infer the number of households from weight vector and CD count
    if len(w) % len(cds_to_calibrate) != 0:
        raise ValueError(
            f"Weight vector length ({len(w):,}) is not evenly divisible by "
            f"number of CDs ({len(cds_to_calibrate)}). Cannot determine household count."
        )
    n_households_from_weights = len(w) // len(cds_to_calibrate)

    if n_households_from_weights != n_households_orig:
        raise ValueError("Households from base data set do not match households from weights")

    print(f"\nOriginal dataset has {n_households_orig:,} households")

    # Process the weight vector to understand active household-CD pairs
    W_full = w.reshape(len(cds_to_calibrate), n_households_orig)
    # (436, 10580)

    # Extract only the CDs we want to process
    if cd_subset is not None:
        W = W_full[cd_indices, :]
        print(
            f"Extracted weights for {len(cd_indices)} CDs from full weight matrix"
        )
    else:
        W = W_full

    # Count total active weights: i.e., number of active households
    total_active_weights = np.sum(W > 0)
    total_weight_in_W = np.sum(W)
    print(f"Total active household-CD pairs: {total_active_weights:,}")
    print(f"Total weight in W matrix: {total_weight_in_W:,.0f}")

    # Collect DataFrames for each CD
    cd_dfs = []
    total_kept_households = 0
    #total_calibrated_weight = 0
    #total_kept_weight = 0
    time_period = int(base_sim.default_calculation_period)

    for idx, cd_geoid in enumerate(cds_to_process):
        # Progress every 10 CDs and at the end ----
        if (idx + 1) % 10 == 0 or (idx + 1) == len(cds_to_process):
            print(
                f"Processing CD {cd_geoid} ({idx + 1}/{len(cds_to_process)})..."
            )

        # Get the correct index in the weight matrix
        cd_idx = idx  # Index in our filtered W matrix

        # Get ALL households with non-zero weight in this CD
        active_household_indices = np.where(W[cd_idx, :] > 0)[0]

        if len(active_household_indices) == 0:
            continue

        # Get the household IDs for active households
        active_household_ids = set(
            household_ids[hh_idx] for hh_idx in active_household_indices
        )

        # Fresh simulation
        cd_sim = Microsimulation(dataset=dataset_path)

        # First, create hh_df with CALIBRATED weights from the W matrix
        household_ids_in_sim = cd_sim.calculate(
            "household_id", map_to="household"
        ).values

        # Get this CD's calibrated weights from the weight matrix
        calibrated_weights_for_cd = W[cd_idx, :]  # Get this CD's row from weight matrix

        # Map the calibrated weights to household IDs
        hh_weight_values = []
        for hh_id in household_ids_in_sim:
            hh_idx = hh_id_to_idx[int(hh_id)]  # Get index in weight matrix
            hh_weight_values.append(calibrated_weights_for_cd[hh_idx])

        # TODO: do I need this?
        entity_rel = pd.DataFrame(
            {
                "person_id": cd_sim.calculate(
                    "person_id", map_to="person"
                ).values,
                "household_id": cd_sim.calculate(
                    "household_id", map_to="person"
                ).values,
                "tax_unit_id": cd_sim.calculate(
                    "tax_unit_id", map_to="person"
                ).values,
                "spm_unit_id": cd_sim.calculate(
                    "spm_unit_id", map_to="person"
                ).values,
                "family_id": cd_sim.calculate(
                    "family_id", map_to="person"
                ).values,
                "marital_unit_id": cd_sim.calculate(
                    "marital_unit_id", map_to="person"
                ).values,
            }
        )

        hh_df = pd.DataFrame(
            {
                "household_id": household_ids_in_sim,
                "household_weight": hh_weight_values,
            }
        )
        counts = entity_rel.groupby('household_id')['person_id'].size().reset_index(name="persons_per_hh")
        hh_df = hh_df.merge(counts)
        hh_df['per_person_hh_weight'] = hh_df.household_weight / hh_df.persons_per_hh

        ## Now create person_rel with calibrated household weights
        #person_ids = cd_sim.calculate("person_id", map_to="person").values
        #person_household_ids = cd_sim.calculate("household_id", map_to="person").values
        #person_tax_unit_ids = cd_sim.calculate("tax_unit_id", map_to="person").values

        ## Map calibrated household weights to person level
        #hh_weight_map = dict(zip(hh_df['household_id'], hh_df['household_weight']))
        #person_household_weights = [hh_weight_map[int(hh_id)] for hh_id in person_household_ids]

        #person_rel = pd.DataFrame(
        #    {
        #        "person_id": person_ids,
        #        "household_id": person_household_ids,
        #        "household_weight": person_household_weights,
        #        "tax_unit_id": person_tax_unit_ids,
        #    }
        #)

        ## Calculate person weights based on calibrated household weights
        ## Person weight equals household weight (each person represents the household weight)
        #person_rel['person_weight'] = person_rel['household_weight']

        ## Tax unit weight: each tax unit gets the weight of its household
        #tax_unit_df = person_rel.groupby('tax_unit_id').agg(
        #    tax_unit_weight=('household_weight', 'first')
        #).reset_index()

        ## SPM unit weight: each SPM unit gets the weight of its household
        #person_spm_ids = cd_sim.calculate('spm_unit_id', map_to='person').values
        #person_rel['spm_unit_id'] = person_spm_ids
        #spm_unit_df = person_rel.groupby('spm_unit_id').agg(
        #    spm_unit_weight=('household_weight', 'first')
        #).reset_index()

        ## Marital unit weight: each marital unit gets the weight of its household
        #person_marital_ids = cd_sim.calculate('marital_unit_id', map_to='person').values
        #person_rel['marital_unit_id'] = person_marital_ids
        #marital_unit_df = person_rel.groupby('marital_unit_id').agg(
        #    marital_unit_weight=('household_weight', 'first')
        #).reset_index()

        ## Track calibrated weight for this CD
        #cd_calibrated_weight = calibrated_weights_for_cd.sum()
        #cd_active_weight = calibrated_weights_for_cd[calibrated_weights_for_cd > 0].sum()

        # SET WEIGHTS IN SIMULATION BEFORE EXTRACTING DATAFRAME
        # This is the key - set_input updates the simulation's internal state

        non_household_cols = ['person_id', 'tax_unit_id', 'spm_unit_id', 'family_id', 'marital_unit_id']

        new_weights_per_id = {}
        for col in non_household_cols:
            person_counts = entity_rel.groupby(col)['person_id'].size().reset_index(name="person_id_count")
            # Below: drop duplicates to undo the broadcast join done in entity_rel
            id_link = entity_rel[['household_id', col]].drop_duplicates()
            hh_info = id_link.merge(hh_df)

            hh_info2 = hh_info.merge(person_counts, on=col)
            hh_info2["id_weight"] = hh_info2.per_person_hh_weight * hh_info2.person_id_count
            new_weights_per_id[col] = hh_info2.id_weight

        for key in new_weights_per_id.keys():
            assert np.isclose(np.sum(hh_weight_values), np.sum(new_weights_per_id[key]), atol=5)
        
        cd_sim.set_input("household_weight", time_period, hh_df.household_weight.values)
        cd_sim.set_input("person_weight", time_period, new_weights_per_id['person_id'])
        cd_sim.set_input("tax_unit_weight", time_period, new_weights_per_id['tax_unit_id'])
        cd_sim.set_input("spm_unit_weight", time_period, new_weights_per_id['spm_unit_id'])
        cd_sim.set_input("marital_unit_weight", time_period, new_weights_per_id['marital_unit_id'])
        cd_sim.set_input("family_weight", time_period,  new_weights_per_id['family_id'])

        # Now extract the dataframe with updated weights
        df = cd_sim.to_input_dataframe()

        assert df.shape[0] == entity_rel.shape[0]  # df is at the person level

        # Column names follow pattern: variable__year
        hh_id_col = f"household_id__{time_period}"
        cd_geoid_col = f"congressional_district_geoid__{time_period}"
        hh_weight_col = f"household_weight__{time_period}"
        person_weight_col = f"person_weight__{time_period}"
        tax_unit_weight_col = f"tax_unit_weight__{time_period}"
        person_id_col = f"person_id__{time_period}"
        tax_unit_id_col = f"tax_unit_id__{time_period}"

        state_fips_col = f"state_fips__{time_period}"
        state_name_col = f"state_name__{time_period}"
        state_code_col = f"state_code__{time_period}"
        county_fips_col = f"county_fips__{time_period}"
        county_col = f"county__{time_period}"
        county_str_col = f"county_str__{time_period}"

        # Filter to only active households in this CD
        df_filtered = df[df[hh_id_col].isin(active_household_ids)].copy()

        ## Track weight after filtering - need to group by household since df_filtered is person-level
        #df_filtered_weight = df_filtered.groupby(hh_id_col)[hh_weight_col].first().sum()

        #if abs(cd_active_weight - df_filtered_weight) > 10:
        #    print(f"  CD {cd_geoid}: Calibrated active weight = {cd_active_weight:,.0f}, df_filtered weight = {df_filtered_weight:,.0f}, LOST {cd_active_weight - df_filtered_weight:,.0f}")

        #total_calibrated_weight += cd_active_weight
        #total_kept_weight += df_filtered_weight

        # Update congressional_district_geoid to target CD
        df_filtered[cd_geoid_col] = int(cd_geoid)

        # Extract state FIPS from CD GEOID (first 1-2 digits)
        cd_geoid_int = int(cd_geoid)
        state_fips = cd_geoid_int // 100

        # Update state variables for consistency
        df_filtered[state_fips_col] = state_fips
        if state_fips in STATE_FIPS_TO_NAME:
            df_filtered[state_name_col] = STATE_FIPS_TO_NAME[state_fips]
        if state_fips in STATE_FIPS_TO_CODE:
            df_filtered[state_code_col] = STATE_FIPS_TO_CODE[state_fips]

        # Update county variables if we have mappings
        if cd_county_mappings:
            # For each household, assign a county based on CD proportions
            unique_hh_ids = df_filtered[hh_id_col].unique()
            hh_to_county = {}

            for hh_id in unique_hh_ids:
                county_fips = get_county_for_cd(cd_geoid, cd_county_mappings)
                if county_fips:
                    hh_to_county[hh_id] = county_fips
                else:
                    hh_to_county[hh_id] = ""

            if hh_to_county and any(hh_to_county.values()):
                # Map household to county FIPS string
                county_fips_str = df_filtered[hh_id_col].map(hh_to_county)

                # Convert FIPS string to integer for county_fips column
                # Handle empty strings by converting to 0
                df_filtered[county_fips_col] = county_fips_str.apply(
                    lambda x: int(x) if x and x != "" else 0
                )

                # Set county enum to UNKNOWN (since we don't have specific enum values)
                df_filtered[county_col] = County.UNKNOWN

                # Set county_str to the string representation of FIPS
                df_filtered[county_str_col] = county_fips_str

        cd_dfs.append(df_filtered)
        total_kept_households += len(df_filtered[hh_id_col].unique())

    print(f"\nCombining {len(cd_dfs)} CD DataFrames...")
    print(f"Total households across all CDs: {total_kept_households:,}")
    #print(f"\nWeight tracking:")
    #print(f"  Total calibrated active weight: {total_calibrated_weight:,.0f}")
    #print(f"  Total kept weight in df_filtered: {total_kept_weight:,.0f}")
    #print(f"  Weight retention: {100 * total_kept_weight / total_calibrated_weight:.2f}%")

    # Combine all CD DataFrames
    combined_df = pd.concat(cd_dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")

    # Check weights in combined_df before any reindexing
    hh_weight_col = f"household_weight__{time_period}"
    person_weight_col = f"person_weight__{time_period}"
    print(f"\nWeights in combined_df BEFORE reindexing:")
    print(f"  HH weight sum: {combined_df[hh_weight_col].sum()/1e6:.2f}M")
    print(f"  Person weight sum: {combined_df[person_weight_col].sum()/1e6:.2f}M")
    print(f"  Ratio: {combined_df[person_weight_col].sum() / combined_df[hh_weight_col].sum():.2f}")

    # REINDEX ALL IDs TO PREVENT OVERFLOW AND HANDLE DUPLICATES
    print("\nReindexing all entity IDs using 10k ranges per CD...")

    # Column names
    hh_id_col = f"household_id__{time_period}"
    person_id_col = f"person_id__{time_period}"
    person_hh_id_col = f"person_household_id__{time_period}"
    tax_unit_id_col = f"tax_unit_id__{time_period}"
    person_tax_unit_col = f"person_tax_unit_id__{time_period}"
    spm_unit_id_col = f"spm_unit_id__{time_period}"
    person_spm_unit_col = f"person_spm_unit_id__{time_period}"
    marital_unit_id_col = f"marital_unit_id__{time_period}"
    person_marital_unit_col = f"person_marital_unit_id__{time_period}"
    cd_geoid_col = f"congressional_district_geoid__{time_period}"

    # Cache the CD mapping to avoid thousands of database calls!
    cd_to_index, _, _ = get_cd_index_mapping()

    # Create household mapping for CSV export
    household_mapping = []

    # First, create a unique row identifier to track relationships
    combined_df["_row_idx"] = range(len(combined_df))

    # Group by household ID AND congressional district to create unique household-CD pairs
    hh_groups = (
        combined_df.groupby([hh_id_col, cd_geoid_col])["_row_idx"]
        .apply(list)
        .to_dict()
    )

    # Assign new household IDs using 10k ranges per CD
    hh_row_to_new_id = {}
    cd_hh_counters = {}  # Track how many households assigned per CD

    for (old_hh_id, cd_geoid), row_indices in hh_groups.items():
        # Calculate the ID range for this CD directly (avoiding function call)
        cd_str = str(int(cd_geoid))
        cd_idx = cd_to_index[cd_str]
        start_id = cd_idx * 10_000
        end_id = start_id + 9_999

        # Get the next available ID in this CD's range
        if cd_str not in cd_hh_counters:
            cd_hh_counters[cd_str] = 0

        new_hh_id = start_id + cd_hh_counters[cd_str]

        # Check we haven't exceeded the range
        if new_hh_id > end_id:
            raise ValueError(
                f"CD {cd_str} exceeded its 10k household allocation"
            )

        # All rows in the same household-CD pair get the SAME new ID
        for row_idx in row_indices:
            hh_row_to_new_id[row_idx] = new_hh_id

        # Save the mapping
        household_mapping.append(
            {
                "new_household_id": new_hh_id,
                "original_household_id": int(old_hh_id),
                "congressional_district": cd_str,
                "state_fips": int(cd_str) // 100,
            }
        )

        cd_hh_counters[cd_str] += 1

    # Apply new household IDs based on row index
    combined_df["_new_hh_id"] = combined_df["_row_idx"].map(hh_row_to_new_id)

    # Update household IDs
    combined_df[hh_id_col] = combined_df["_new_hh_id"]

    # Update person household references - since persons are already in their households,
    # person_household_id should just match the household_id of their row
    combined_df[person_hh_id_col] = combined_df["_new_hh_id"]

    # Report statistics
    total_households = sum(cd_hh_counters.values())
    print(
        f"  Created {total_households:,} unique households across {len(cd_hh_counters)} CDs"
    )

    # Now handle persons with same 10k range approach - VECTORIZED
    print("  Reindexing persons using 10k ranges...")

    # OFFSET PERSON IDs by 5 million to avoid collision with household IDs
    PERSON_ID_OFFSET = 5_000_000

    # Group by CD and assign IDs in bulk for each CD
    for cd_geoid_val in combined_df[cd_geoid_col].unique():
        cd_str = str(int(cd_geoid_val))

        # Calculate the ID range for this CD directly
        cd_idx = cd_to_index[cd_str]
        start_id = cd_idx * 10_000 + PERSON_ID_OFFSET  # Add offset for persons
        end_id = start_id + 9_999

        # Get all rows for this CD
        cd_mask = combined_df[cd_geoid_col] == cd_geoid_val
        n_persons_in_cd = cd_mask.sum()

        # Check we won't exceed the range
        if n_persons_in_cd > (end_id - start_id + 1):
            raise ValueError(
                f"CD {cd_str} has {n_persons_in_cd} persons, exceeds 10k allocation"
            )

        # Create sequential IDs for this CD
        new_person_ids = np.arange(start_id, start_id + n_persons_in_cd)

        # Assign all at once using loc
        combined_df.loc[cd_mask, person_id_col] = new_person_ids

    # Tax units - preserve structure within households
    print("  Reindexing tax units...")
    # Group by household first, then handle tax units within each household
    new_tax_id = 0
    for hh_id in combined_df[hh_id_col].unique():
        hh_mask = combined_df[hh_id_col] == hh_id
        hh_df = combined_df[hh_mask]

        # Get unique tax units within this household
        unique_tax_in_hh = hh_df[person_tax_unit_col].unique()

        # Create mapping for this household's tax units
        for old_tax in unique_tax_in_hh:
            # Update all persons with this tax unit ID in this household
            mask = (combined_df[hh_id_col] == hh_id) & (
                combined_df[person_tax_unit_col] == old_tax
            )
            combined_df.loc[mask, person_tax_unit_col] = new_tax_id
            # Also update tax_unit_id if it exists in the DataFrame
            if tax_unit_id_col in combined_df.columns:
                combined_df.loc[mask, tax_unit_id_col] = new_tax_id
            new_tax_id += 1

    # SPM units - preserve structure within households
    print("  Reindexing SPM units...")
    new_spm_id = 0
    for hh_id in combined_df[hh_id_col].unique():
        hh_mask = combined_df[hh_id_col] == hh_id
        hh_df = combined_df[hh_mask]

        # Get unique SPM units within this household
        unique_spm_in_hh = hh_df[person_spm_unit_col].unique()

        for old_spm in unique_spm_in_hh:
            # Update all persons with this SPM unit ID in this household
            mask = (combined_df[hh_id_col] == hh_id) & (
                combined_df[person_spm_unit_col] == old_spm
            )
            combined_df.loc[mask, person_spm_unit_col] = new_spm_id
            # Also update spm_unit_id if it exists
            if spm_unit_id_col in combined_df.columns:
                combined_df.loc[mask, spm_unit_id_col] = new_spm_id
            new_spm_id += 1

    # Marital units - preserve structure within households
    print("  Reindexing marital units...")
    new_marital_id = 0
    for hh_id in combined_df[hh_id_col].unique():
        hh_mask = combined_df[hh_id_col] == hh_id
        hh_df = combined_df[hh_mask]

        # Get unique marital units within this household
        unique_marital_in_hh = hh_df[person_marital_unit_col].unique()

        for old_marital in unique_marital_in_hh:
            # Update all persons with this marital unit ID in this household
            mask = (combined_df[hh_id_col] == hh_id) & (
                combined_df[person_marital_unit_col] == old_marital
            )
            combined_df.loc[mask, person_marital_unit_col] = new_marital_id
            # Also update marital_unit_id if it exists
            if marital_unit_id_col in combined_df.columns:
                combined_df.loc[mask, marital_unit_id_col] = new_marital_id
            new_marital_id += 1

    # Clean up temporary columns
    temp_cols = [col for col in combined_df.columns if col.startswith("_")]
    combined_df = combined_df.drop(columns=temp_cols)

    print(f"  Final persons: {len(combined_df):,}")
    print(f"  Final households: {total_households:,}")
    print(f"  Final tax units: {new_tax_id:,}")
    print(f"  Final SPM units: {new_spm_id:,}")
    print(f"  Final marital units: {new_marital_id:,}")

    # Check weights in combined_df AFTER reindexing
    print(f"\nWeights in combined_df AFTER reindexing:")
    print(f"  HH weight sum: {combined_df[hh_weight_col].sum()/1e6:.2f}M")
    print(f"  Person weight sum: {combined_df[person_weight_col].sum()/1e6:.2f}M")
    print(f"  Ratio: {combined_df[person_weight_col].sum() / combined_df[hh_weight_col].sum():.2f}")

    # Verify no overflow risk
    max_person_id = combined_df[person_id_col].max()
    print(f"\nOverflow check:")
    print(f"  Max person ID after reindexing: {max_person_id:,}")
    print(f"  Max person ID × 100: {max_person_id * 100:,}")
    print(f"  int32 max: {2_147_483_647:,}")
    if max_person_id * 100 < 2_147_483_647:
        print("  ✓ No overflow risk!")
    else:
        print("  ⚠️ WARNING: Still at risk of overflow!")

    # Create Dataset from combined DataFrame
    print("\nCreating Dataset from combined DataFrame...")
    sparse_dataset = Dataset.from_dataframe(combined_df, time_period)

    # Build a simulation to convert to h5
    print("Building simulation from Dataset...")
    sparse_sim = Microsimulation()
    sparse_sim.dataset = sparse_dataset
    sparse_sim.build_from_dataset()

    # Save to h5 file
    print(f"\nSaving to {output_path}...")
    data = {}

    # Load the base dataset to see what variables were available during training
    import h5py as h5py_check
    with h5py_check.File(dataset_path, 'r') as base_file:
        base_dataset_vars = set(base_file.keys())
    print(f"Base dataset has {len(base_dataset_vars)} variables")

    # Define essential variables that must be kept even if they have formulas
    essential_vars = {
        'person_id', 'household_id', 'tax_unit_id', 'spm_unit_id',
        'marital_unit_id', 'person_weight', 'household_weight', 'tax_unit_weight',
        'person_household_id', 'person_tax_unit_id', 'person_spm_unit_id',
        'person_marital_unit_id',
        'congressional_district_geoid',
        'state_fips', 'state_name', 'state_code',
        'county_fips', 'county', 'county_str'
    }

    # If freeze_calculated_vars is True, add all calculated variables to essential vars
    if freeze_calculated_vars:
        from metrics_matrix_geo_stacking_sparse import get_calculated_variables
        calculated_vars = get_calculated_variables(sparse_sim)
        essential_vars.update(calculated_vars)
        print(f"Freezing {len(calculated_vars)} calculated variables (will be saved to h5)")

    variables_saved = 0
    variables_skipped = 0

    for variable in sparse_sim.tax_benefit_system.variables:
        var_def = sparse_sim.tax_benefit_system.variables[variable]

        # Save if it's essential OR if it was in the base dataset
        if variable in essential_vars or variable in base_dataset_vars:
            pass  # Will try to save below
        else:
            # Skip other calculated/aggregate variables
            if var_def.formulas or \
               (hasattr(var_def, 'adds') and var_def.adds) or \
               (hasattr(var_def, 'subtracts') and var_def.subtracts):
                variables_skipped += 1
                continue

        # Only process variables that have actual data
        data[variable] = {}
        for period in sparse_sim.get_holder(variable).get_known_periods():
            values = sparse_sim.get_holder(variable).get_array(period)

            # Handle different value types
            if (
                sparse_sim.tax_benefit_system.variables.get(
                    variable
                ).value_type
                in (Enum, str)
                and variable != "county_fips"
            ):
                # Handle EnumArray objects
                if hasattr(values, "decode_to_str"):
                    values = values.decode_to_str().astype("S")
                else:
                    # Already a regular numpy array, just convert to string type
                    values = values.astype("S")
            elif variable == "county_fips":
                values = values.astype("int32")
            else:
                values = np.array(values)

            if values is not None:
                data[variable][period] = values
                variables_saved += 1

        if len(data[variable]) == 0:
            del data[variable]

    print(f"Variables saved: {variables_saved}")
    print(f"Variables skipped: {variables_skipped}")

    # Write to h5
    with h5py.File(output_path, "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    print(f"Sparse CD-stacked dataset saved successfully!")

    # Save household mapping to CSV
    mapping_df = pd.DataFrame(household_mapping)
    csv_path = output_path.replace(".h5", "_household_mapping.csv")
    mapping_df.to_csv(csv_path, index=False)
    print(f"Household mapping saved to {csv_path}")

    # Verify the saved file
    print("\nVerifying saved file...")
    with h5py.File(output_path, "r") as f:
        if "household_id" in f and str(time_period) in f["household_id"]:
            hh_ids = f["household_id"][str(time_period)][:]
            print(f"  Final households: {len(hh_ids):,}")
        if "person_id" in f and str(time_period) in f["person_id"]:
            person_ids = f["person_id"][str(time_period)][:]
            print(f"  Final persons: {len(person_ids):,}")
        if (
            "household_weight" in f
            and str(time_period) in f["household_weight"]
        ):
            weights = f["household_weight"][str(time_period)][:]
            print(
                f"  Total population (from household weights): {np.sum(weights):,.0f}"
            )
        if "person_weight" in f and str(time_period) in f["person_weight"]:
            person_weights = f["person_weight"][str(time_period)][:]
            print(
                f"  Total population (from person weights): {np.sum(person_weights):,.0f}"
            )
            print(
                f"  Average persons per household: {np.sum(person_weights) / np.sum(weights):.2f}"
            )

    return output_path


def main(dataset_path, w, db_uri):
    #dataset_path = Dataset.from_file(dataset_path_str)
    #w = np.load(weights_path_str)
    #db_uri = f"sqlite:///{db_path}"

    engine = create_engine(db_uri)
    
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM strata s
    JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
    WHERE s.stratum_group_id = 1
      AND sc.constraint_variable = "congressional_district_geoid"
    ORDER BY sc.value
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        cds_to_calibrate = [row[0] for row in result]
    
    ## Verify dimensions match
    # Note: this is the base dataset that was stacked repeatedly
    assert_sim = Microsimulation(dataset=dataset_path)
    n_hh = assert_sim.calculate("household_id", map_to="household").shape[0]
    expected_length = len(cds_to_calibrate) * n_hh
    
    # Ensure that the data set we're rebuilding has a shape that's consistent with training
    if len(w) != expected_length:
        raise ValueError(
            f"Weight vector length ({len(w):,}) doesn't match expected ({expected_length:,})"
        )
    
    # Create the .h5 files ---------------------------------------------
    # National Dataset with all districts ------------------------------------------------
    # TODO: what is the cds_to_calibrate doing for us if we have the cd_subset command?
    if include_full_dataset:
        output_path = f"{output_dir}/national.h5"
        print(f"\nCreating combined dataset with all CDs in {output_path}")
        output_file = create_sparse_cd_stacked_dataset(
            w,
            cds_to_calibrate,
            dataset_path=dataset_path,
            output_path=output_path,
        )
    
    # State Datasets with state districts ---------
    if False:
        for state_fips, state_code in STATE_CODES.items():
            cd_subset = [
                cd for cd in cds_to_calibrate if int(cd) // 100 == state_fips
            ]
    
            output_path = f"{output_dir}/{state_code}.h5"
            output_file = create_sparse_cd_stacked_dataset(
                w,
                cds_to_calibrate,
                cd_subset=cd_subset,
                dataset_path=dataset_path,
                output_path=output_path,
            )
            print(f"Created {state_code}.h5")


#if __name__ == "__main__":
#    import argparse
#
#    parser = argparse.ArgumentParser(
#        description="Create sparse CD-stacked state datasets"
#    )
#    parser.add_argument(
#        "--weights-path", required=True, help="Path to w_cd.npy file"
#    )
#    parser.add_argument(
#        "--dataset-path",
#        required=True,
#        help="Path to stratified dataset .h5 file",
#    )
#    parser.add_argument(
#        "--db-path", required=True, help="Path to policy_data.db"
#    )
#    parser.add_argument(
#        "--output-dir",
#        default="./temp",
#        help="Output directory for state files",
#    )
#    parser.add_argument(
#        "--include-full-dataset",
#        action="store_true",
#        help="Also create the combined dataset with all CDs (memory intensive)",
#    )
#
#    args = parser.parse_args()
#    dataset_path_str = args.dataset_path
#    weights_path_str = args.weights_path
#    db_path = Path(args.db_path).resolve()
#    output_dir = args.output_dir
#    include_full_dataset = args.include_full_dataset
#
#    # All args read in ---------
#    os.makedirs(output_dir, exist_ok=True)


