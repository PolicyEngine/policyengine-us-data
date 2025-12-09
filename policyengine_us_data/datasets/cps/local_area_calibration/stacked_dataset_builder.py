"""
Create a sparse congressional district-stacked dataset with non-zero weight
households.
"""

import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import Enum
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_all_cds_from_database,
    get_calculated_variables,
    STATE_CODES,
    STATE_FIPS_TO_NAME,
    STATE_FIPS_TO_CODE,
)
from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
    County,
)
from policyengine_us_data.datasets.cps.local_area_calibration.county_assignment import (
    assign_counties_for_cd,
)


def create_sparse_cd_stacked_dataset(
    w,
    cds_to_calibrate,
    cd_subset=None,
    output_path=None,
    dataset_path=None,
):
    """
    Create a SPARSE congressional district-stacked dataset using DataFrame approach.

    Args:
        w: Calibrated weight vector from L0 calibration (length = n_households * n_cds)
        cds_to_calibrate: List of CD GEOID codes used in calibration
        cd_subset: Optional list of CD GEOIDs to include (subset of cds_to_calibrate)
        output_path: Where to save the sparse CD-stacked h5 file
        dataset_path: Path to the base .h5 dataset used to create the training matrices
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

    # Check that output directory exists, create if needed
    output_dir_path = os.path.dirname(output_path)
    if output_dir_path and not os.path.exists(output_dir_path):
        print(f"Creating output directory: {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True)

    # Load the original simulation
    base_sim = Microsimulation(dataset=dataset_path)

    household_ids = base_sim.calculate(
        "household_id", map_to="household"
    ).values
    n_households_orig = len(household_ids)

    # From the base sim, create mapping from household ID to index for proper filtering
    hh_id_to_idx = {int(hh_id): idx for idx, hh_id in enumerate(household_ids)}

    # Infer the number of households from weight vector and CD count
    if len(w) % len(cds_to_calibrate) != 0:
        raise ValueError(
            f"Weight vector length ({len(w):,}) is not evenly divisible by "
            f"number of CDs ({len(cds_to_calibrate)}). Cannot determine household count."
        )
    n_households_from_weights = len(w) // len(cds_to_calibrate)

    if n_households_from_weights != n_households_orig:
        raise ValueError(
            "Households from base data set do not match households from weights"
        )

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
        calibrated_weights_for_cd = W[
            cd_idx, :
        ]  # Get this CD's row from weight matrix

        # Map the calibrated weights to household IDs
        hh_weight_values = []
        for hh_id in household_ids_in_sim:
            hh_idx = hh_id_to_idx[int(hh_id)]  # Get index in weight matrix
            hh_weight_values.append(calibrated_weights_for_cd[hh_idx])

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
        counts = (
            entity_rel.groupby("household_id")["person_id"]
            .size()
            .reset_index(name="persons_per_hh")
        )
        hh_df = hh_df.merge(counts)
        hh_df["per_person_hh_weight"] = (
            hh_df.household_weight / hh_df.persons_per_hh
        )

        # SET WEIGHTS IN SIMULATION BEFORE EXTRACTING DATAFRAME
        # This is the key - set_input updates the simulation's internal state

        non_household_cols = [
            "person_id",
            "tax_unit_id",
            "spm_unit_id",
            "family_id",
            "marital_unit_id",
        ]

        new_weights_per_id = {}
        for col in non_household_cols:
            person_counts = (
                entity_rel.groupby(col)["person_id"]
                .size()
                .reset_index(name="person_id_count")
            )
            # Below: drop duplicates to undo the broadcast join done in entity_rel
            id_link = entity_rel[["household_id", col]].drop_duplicates()
            hh_info = id_link.merge(hh_df)

            hh_info2 = hh_info.merge(person_counts, on=col)
            if col == "person_id":
                # Person weight = household weight (each person represents same count as their household)
                hh_info2["id_weight"] = hh_info2.household_weight
            else:
                hh_info2["id_weight"] = (
                    hh_info2.per_person_hh_weight * hh_info2.person_id_count
                )
            new_weights_per_id[col] = hh_info2.id_weight

        cd_sim.set_input(
            "household_weight", time_period, hh_df.household_weight.values
        )
        cd_sim.set_input(
            "person_weight", time_period, new_weights_per_id["person_id"]
        )
        cd_sim.set_input(
            "tax_unit_weight", time_period, new_weights_per_id["tax_unit_id"]
        )
        cd_sim.set_input(
            "spm_unit_weight", time_period, new_weights_per_id["spm_unit_id"]
        )
        cd_sim.set_input(
            "marital_unit_weight",
            time_period,
            new_weights_per_id["marital_unit_id"],
        )
        cd_sim.set_input(
            "family_weight", time_period, new_weights_per_id["family_id"]
        )

        # Extract state from CD GEOID and update simulation BEFORE calling to_input_dataframe()
        # This ensures calculated variables (SNAP, Medicaid) use the correct state
        cd_geoid_int = int(cd_geoid)
        state_fips = cd_geoid_int // 100

        cd_sim.set_input(
            "state_fips",
            time_period,
            np.full(n_households_orig, state_fips, dtype=np.int32),
        )
        cd_sim.set_input(
            "congressional_district_geoid",
            time_period,
            np.full(n_households_orig, cd_geoid_int, dtype=np.int32),
        )

        # Set county for this CD
        county_indices = assign_counties_for_cd(
            cd_geoid=cd_geoid, n_households=n_households_orig, seed=42 + idx
        )
        cd_sim.set_input("county", time_period, county_indices)

        # Delete cached calculated variables to ensure they're recalculated
        # with new state and county. Exclude 'county' itself since we just set it.
        for var in get_calculated_variables(cd_sim):
            if var != "county":
                cd_sim.delete_arrays(var)

        # Now extract the dataframe - calculated vars will use the updated state
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

        # Filter to only active households in this CD
        df_filtered = df[df[hh_id_col].isin(active_household_ids)].copy()

        # Update congressional_district_geoid to target CD
        df_filtered[cd_geoid_col] = int(cd_geoid)

        # Update state variables for consistency
        df_filtered[state_fips_col] = state_fips
        if state_fips in STATE_FIPS_TO_NAME:
            df_filtered[state_name_col] = STATE_FIPS_TO_NAME[state_fips]
        if state_fips in STATE_FIPS_TO_CODE:
            df_filtered[state_code_col] = STATE_FIPS_TO_CODE[state_fips]

        cd_dfs.append(df_filtered)
        total_kept_households += len(df_filtered[hh_id_col].unique())

    print(f"\nCombining {len(cd_dfs)} CD DataFrames...")
    print(f"Total households across all CDs: {total_kept_households:,}")

    # Combine all CD DataFrames
    combined_df = pd.concat(cd_dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")

    # REINDEX ALL IDs TO PREVENT OVERFLOW AND HANDLE DUPLICATES
    print("\nReindexing all entity IDs using 25k ranges per CD...")

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

    # Build CD index mapping from cds_to_calibrate (avoids database dependency)
    cds_sorted = sorted(cds_to_calibrate)
    cd_to_index = {cd: idx for idx, cd in enumerate(cds_sorted)}

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

    # Assign new household IDs using 25k ranges per CD
    hh_row_to_new_id = {}
    cd_hh_counters = {}  # Track how many households assigned per CD

    for (old_hh_id, cd_geoid), row_indices in hh_groups.items():
        # Calculate the ID range for this CD directly (avoiding function call)
        cd_str = str(int(cd_geoid))
        cd_idx = cd_to_index[cd_str]
        start_id = cd_idx * 25_000
        end_id = start_id + 24_999

        # Get the next available ID in this CD's range
        if cd_str not in cd_hh_counters:
            cd_hh_counters[cd_str] = 0

        new_hh_id = start_id + cd_hh_counters[cd_str]

        # Check we haven't exceeded the range
        if new_hh_id > end_id:
            raise ValueError(
                f"CD {cd_str} exceeded its 25k household allocation"
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

    # Now handle persons with same 25k range approach - VECTORIZED
    print("  Reindexing persons using 25k ranges...")

    # OFFSET PERSON IDs by 5 million to avoid collision with household IDs
    PERSON_ID_OFFSET = 5_000_000

    # Group by CD and assign IDs in bulk for each CD
    for cd_geoid_val in combined_df[cd_geoid_col].unique():
        cd_str = str(int(cd_geoid_val))

        # Calculate the ID range for this CD directly
        cd_idx = cd_to_index[cd_str]
        start_id = cd_idx * 25_000 + PERSON_ID_OFFSET  # Add offset for persons
        end_id = start_id + 24_999

        # Get all rows for this CD
        cd_mask = combined_df[cd_geoid_col] == cd_geoid_val
        n_persons_in_cd = cd_mask.sum()

        # Check we won't exceed the range
        if n_persons_in_cd > (end_id - start_id + 1):
            raise ValueError(
                f"CD {cd_str} has {n_persons_in_cd} persons, exceeds 25k allocation"
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
    print(
        f"  Person weight sum: {combined_df[person_weight_col].sum()/1e6:.2f}M"
    )
    print(
        f"  Ratio: {combined_df[person_weight_col].sum() / combined_df[hh_weight_col].sum():.2f}"
    )

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

    # Only save input variables (not calculated/derived variables)
    # Calculated variables like state_name, state_code will be recalculated on load
    input_vars = set(sparse_sim.input_variables)
    print(
        f"Found {len(input_vars)} input variables (excluding calculated variables)"
    )

    vars_to_save = input_vars.copy()

    # congressional_district_geoid isn't in the original microdata and has no formula,
    # so it's not in input_vars. Since we set it explicitly during stacking, save it.
    vars_to_save.add("congressional_district_geoid")

    # county is set explicitly with assign_counties_for_cd, must be saved
    vars_to_save.add("county")

    variables_saved = 0
    variables_skipped = 0

    for variable in sparse_sim.tax_benefit_system.variables:
        if variable not in vars_to_save:
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

    # Save household mapping to CSV in a mappings subdirectory
    mapping_df = pd.DataFrame(household_mapping)
    output_dir = os.path.dirname(output_path)
    mappings_dir = (
        os.path.join(output_dir, "mappings") if output_dir else "mappings"
    )
    os.makedirs(mappings_dir, exist_ok=True)
    csv_filename = os.path.basename(output_path).replace(
        ".h5", "_household_mapping.csv"
    )
    csv_path = os.path.join(mappings_dir, csv_filename)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create sparse CD-stacked datasets"
    )
    parser.add_argument(
        "--weights-path", required=True, help="Path to w_cd.npy file"
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to stratified dataset .h5 file",
    )
    parser.add_argument(
        "--db-path", required=True, help="Path to policy_data.db"
    )
    parser.add_argument(
        "--output-dir",
        default="./temp",
        help="Output directory for files",
    )
    parser.add_argument(
        "--mode",
        choices=["national", "states", "cds", "single-cd", "single-state"],
        default="national",
        help="Output mode: national (one file), states (per-state files), cds (per-CD files), single-cd (one CD), single-state (one state)",
    )
    parser.add_argument(
        "--cd",
        type=str,
        help="Single CD GEOID to process (only used with --mode single-cd)",
    )
    parser.add_argument(
        "--state",
        type=str,
        help="State code to process, e.g. RI, CA, NC (only used with --mode single-state)",
    )

    args = parser.parse_args()
    dataset_path_str = args.dataset_path
    weights_path_str = args.weights_path
    db_path = Path(args.db_path).resolve()
    output_dir = args.output_dir
    mode = args.mode

    os.makedirs(output_dir, exist_ok=True)

    # Load weights
    w = np.load(weights_path_str)
    db_uri = f"sqlite:///{db_path}"

    # Get list of CDs from database
    cds_to_calibrate = get_all_cds_from_database(db_uri)
    print(f"Found {len(cds_to_calibrate)} congressional districts")

    # Verify dimensions
    assert_sim = Microsimulation(dataset=dataset_path_str)
    n_hh = assert_sim.calculate("household_id", map_to="household").shape[0]
    expected_length = len(cds_to_calibrate) * n_hh

    if len(w) != expected_length:
        raise ValueError(
            f"Weight vector length ({len(w):,}) doesn't match expected ({expected_length:,})"
        )

    if mode == "national":
        output_path = f"{output_dir}/national.h5"
        print(f"\nCreating national dataset with all CDs: {output_path}")
        create_sparse_cd_stacked_dataset(
            w,
            cds_to_calibrate,
            dataset_path=dataset_path_str,
            output_path=output_path,
        )

    elif mode == "states":
        for state_fips, state_code in STATE_CODES.items():
            cd_subset = [
                cd for cd in cds_to_calibrate if int(cd) // 100 == state_fips
            ]
            if not cd_subset:
                continue
            output_path = f"{output_dir}/{state_code}.h5"
            print(f"\nCreating {state_code} dataset: {output_path}")
            create_sparse_cd_stacked_dataset(
                w,
                cds_to_calibrate,
                cd_subset=cd_subset,
                dataset_path=dataset_path_str,
                output_path=output_path,
            )

    elif mode == "cds":
        for i, cd_geoid in enumerate(cds_to_calibrate):
            # Convert GEOID to friendly name: 3705 -> NC-05
            cd_int = int(cd_geoid)
            state_fips = cd_int // 100
            district_num = cd_int % 100
            state_code = STATE_CODES.get(state_fips, str(state_fips))
            friendly_name = f"{state_code}-{district_num:02d}"

            output_path = f"{output_dir}/{friendly_name}.h5"
            print(
                f"\n[{i+1}/{len(cds_to_calibrate)}] Creating {friendly_name}.h5 (GEOID {cd_geoid})"
            )
            create_sparse_cd_stacked_dataset(
                w,
                cds_to_calibrate,
                cd_subset=[cd_geoid],
                dataset_path=dataset_path_str,
                output_path=output_path,
            )

    elif mode == "single-cd":
        if not args.cd:
            raise ValueError("--cd required with --mode single-cd")
        if args.cd not in cds_to_calibrate:
            raise ValueError(f"CD {args.cd} not in calibrated CDs list")
        output_path = f"{output_dir}/{args.cd}.h5"
        print(f"\nCreating single CD dataset: {output_path}")
        create_sparse_cd_stacked_dataset(
            w,
            cds_to_calibrate,
            cd_subset=[args.cd],
            dataset_path=dataset_path_str,
            output_path=output_path,
        )

    elif mode == "single-state":
        if not args.state:
            raise ValueError("--state required with --mode single-state")
        # Find FIPS code for this state
        state_code_upper = args.state.upper()
        state_fips = None
        for fips, code in STATE_CODES.items():
            if code == state_code_upper:
                state_fips = fips
                break
        if state_fips is None:
            raise ValueError(f"Unknown state code: {args.state}")

        cd_subset = [
            cd for cd in cds_to_calibrate if int(cd) // 100 == state_fips
        ]
        if not cd_subset:
            raise ValueError(f"No CDs found for state {state_code_upper}")

        output_path = f"{output_dir}/{state_code_upper}.h5"
        print(
            f"\nCreating {state_code_upper} dataset with {len(cd_subset)} CDs: {output_path}"
        )
        create_sparse_cd_stacked_dataset(
            w,
            cds_to_calibrate,
            cd_subset=cd_subset,
            dataset_path=dataset_path_str,
            output_path=output_path,
        )

    print("\nDone!")
