"""
Create a sparse congressional district-stacked dataset with only non-zero weight households.
Standalone version that doesn't modify the working state stacking code.
"""

import numpy as np
import pandas as pd
import h5py
import os
from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import Enum
from sqlalchemy import create_engine, text
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import download_from_huggingface


def create_sparse_cd_stacked_dataset(
    w, 
    cds_to_calibrate,
    cd_subset=None,
    output_path=None,
    dataset_path="hf://policyengine/test/extended_cps_2023.h5"
):
    """
    Create a SPARSE congressional district-stacked dataset using DataFrame approach.
    
    Args:
        w: Calibrated weight vector from L0 calibration (length = n_households * n_cds)
        cds_to_calibrate: List of CD GEOID codes used in calibration
        cd_subset: Optional list of CD GEOIDs to include (subset of cds_to_calibrate)
        output_path: Where to save the sparse CD-stacked h5 file (auto-generated if None)
        dataset_path: Path to the input dataset (default is standard extended CPS)
    """
    print("\n" + "=" * 70)
    print("CREATING SPARSE CD-STACKED DATASET (DataFrame approach)")
    print("=" * 70)
    
    # Handle CD subset filtering
    if cd_subset is not None:
        # Validate that requested CDs are in the calibration
        for cd in cd_subset:
            if cd not in cds_to_calibrate:
                raise ValueError(f"CD {cd} not in calibrated CDs list")
        
        # Get indices of requested CDs
        cd_indices = [cds_to_calibrate.index(cd) for cd in cd_subset]
        cds_to_process = cd_subset
        
        print(f"Processing subset of {len(cd_subset)} CDs: {', '.join(cd_subset[:5])}...")
    else:
        # Process all CDs
        cd_indices = list(range(len(cds_to_calibrate)))
        cds_to_process = cds_to_calibrate
        print(f"Processing all {len(cds_to_calibrate)} congressional districts")
    
    # Generate output path if not provided
    if output_path is None:
        base_dir = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage"
        if cd_subset is None:
            # Default name for all CDs
            output_path = f"{base_dir}/sparse_cd_stacked_2023.h5"
        else:
            # CD-specific name
            suffix = "_".join(cd_subset[:3])  # Use first 3 CDs for naming
            if len(cd_subset) > 3:
                suffix += f"_plus{len(cd_subset)-3}"
            output_path = f"{base_dir}/sparse_cd_stacked_2023_{suffix}.h5"
    
    print(f"Output path: {output_path}")
    
    # Load the original simulation
    base_sim = Microsimulation(dataset=dataset_path)
    
    # Get household IDs and create mapping
    household_ids = base_sim.calculate("household_id", map_to="household").values
    n_households_orig = len(household_ids)
    
    # Create mapping from household ID to index for proper filtering
    hh_id_to_idx = {int(hh_id): idx for idx, hh_id in enumerate(household_ids)}
    
    # Validate weight vector
    expected_weight_length = n_households_orig * len(cds_to_calibrate)
    assert len(w) == expected_weight_length, (
        f"Weight vector length mismatch! Expected {expected_weight_length:,} "
        f"(={n_households_orig:,} households × {len(cds_to_calibrate)} CDs), "
        f"but got {len(w):,}"
    )
    
    print(f"\nOriginal dataset has {n_households_orig:,} households")
    
    # Process the weight vector to understand active household-CD pairs
    print("\nProcessing weight vector...")
    W_full = w.reshape(len(cds_to_calibrate), n_households_orig)
    
    # Extract only the CDs we want to process
    if cd_subset is not None:
        W = W_full[cd_indices, :]
        print(f"Extracted weights for {len(cd_indices)} CDs from full weight matrix")
    else:
        W = W_full
    
    # Count total active weights
    total_active_weights = np.sum(W > 0)
    print(f"Total active household-CD pairs: {total_active_weights:,}")
    
    # Collect DataFrames for each CD
    cd_dfs = []
    total_kept_households = 0
    time_period = int(base_sim.default_calculation_period)
    
    for idx, cd_geoid in enumerate(cds_to_process):
        if (idx + 1) % 10 == 0 or (idx + 1) == len(cds_to_process):  # Progress every 10 CDs and at the end
            print(f"Processing CD {cd_geoid} ({idx + 1}/{len(cds_to_process)})...")
        
        # Get the correct index in the weight matrix
        cd_idx = idx  # Index in our filtered W matrix
        
        # Get ALL households with non-zero weight in this CD
        active_household_indices = np.where(W[cd_idx, :] > 0)[0]
        
        if len(active_household_indices) == 0:
            continue
        
        # Get the household IDs for active households
        active_household_ids = set(household_ids[idx] for idx in active_household_indices)
        
        # Create weight vector with weights for this CD
        cd_weights = np.zeros(n_households_orig)
        cd_weights[active_household_indices] = W[cd_idx, active_household_indices]
        
        # Create a simulation with these weights
        cd_sim = Microsimulation(dataset=dataset_path)
        cd_sim.set_input("household_weight", time_period, cd_weights)
        
        # Convert to DataFrame
        df = cd_sim.to_input_dataframe()
        
        # Column names follow pattern: variable__year
        hh_weight_col = f"household_weight__{time_period}"
        hh_id_col = f"household_id__{time_period}"
        cd_geoid_col = f"congressional_district_geoid__{time_period}"
        
        # Filter to only active households in this CD
        df_filtered = df[df[hh_id_col].isin(active_household_ids)].copy()
        
        # Update congressional_district_geoid to target CD
        df_filtered[cd_geoid_col] = cd_geoid
        
        cd_dfs.append(df_filtered)
        total_kept_households += len(df_filtered[hh_id_col].unique())
    
    print(f"\nCombining {len(cd_dfs)} CD DataFrames...")
    print(f"Total households across all CDs: {total_kept_households:,}")
    
    # Combine all CD DataFrames
    combined_df = pd.concat(cd_dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    # REINDEX ALL IDs TO PREVENT OVERFLOW AND HANDLE DUPLICATES
    print("\nReindexing all entity IDs to handle duplicates and prevent overflow...")
    
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
    
    # First, create a unique row identifier to track relationships
    combined_df['_row_idx'] = range(len(combined_df))
    
    # Group by household ID to track which rows belong to same original household
    hh_groups = combined_df.groupby(hh_id_col)['_row_idx'].apply(list).to_dict()
    
    # Create new unique household IDs (one per row group)
    new_hh_id = 0
    hh_row_to_new_id = {}
    for old_hh_id, row_indices in hh_groups.items():
        for row_idx in row_indices:
            hh_row_to_new_id[row_idx] = new_hh_id
            new_hh_id += 1
    
    # Apply new household IDs based on row index
    combined_df['_new_hh_id'] = combined_df['_row_idx'].map(hh_row_to_new_id)
    
    # Now update person household references to point to new household IDs
    # Create mapping from old household ID + CD context to new household ID
    old_to_new_hh = {}
    for idx, row in combined_df.iterrows():
        old_hh = row[hh_id_col]
        new_hh = row['_new_hh_id']
        # Store mapping for this specific occurrence
        if old_hh not in old_to_new_hh:
            old_to_new_hh[old_hh] = {}
        cd = row[cd_geoid_col]
        old_to_new_hh[old_hh][cd] = new_hh
    
    # Update household IDs
    combined_df[hh_id_col] = combined_df['_new_hh_id']
    
    # For person household references, we need to match based on CD
    def map_person_hh(row):
        old_hh = row[person_hh_id_col]
        cd = row[cd_geoid_col]
        if old_hh in old_to_new_hh and cd in old_to_new_hh[old_hh]:
            return old_to_new_hh[old_hh][cd]
        # Fallback
        return row['_new_hh_id']
    
    combined_df[person_hh_id_col] = combined_df.apply(map_person_hh, axis=1)
    
    print(f"  Created {new_hh_id:,} unique households from duplicates")
    
    # Now handle other entities - they also need unique IDs
    # Persons - each occurrence needs a unique ID
    print("  Reindexing persons...")
    combined_df['_new_person_id'] = range(len(combined_df))
    old_person_to_new = dict(zip(combined_df[person_id_col], combined_df['_new_person_id']))
    combined_df[person_id_col] = combined_df['_new_person_id']
    
    # Tax units - similar approach
    print("  Reindexing tax units...")
    tax_groups = combined_df.groupby([tax_unit_id_col, hh_id_col]).groups
    new_tax_id = 0
    tax_map = {}
    for (old_tax, hh), indices in tax_groups.items():
        for idx in indices:
            tax_map[idx] = new_tax_id
        new_tax_id += 1
    combined_df['_new_tax_id'] = combined_df.index.map(tax_map)
    combined_df[tax_unit_id_col] = combined_df['_new_tax_id']
    combined_df[person_tax_unit_col] = combined_df['_new_tax_id']
    
    # SPM units
    print("  Reindexing SPM units...")
    spm_groups = combined_df.groupby([spm_unit_id_col, hh_id_col]).groups
    new_spm_id = 0
    spm_map = {}
    for (old_spm, hh), indices in spm_groups.items():
        for idx in indices:
            spm_map[idx] = new_spm_id
        new_spm_id += 1
    combined_df['_new_spm_id'] = combined_df.index.map(spm_map)
    combined_df[spm_unit_id_col] = combined_df['_new_spm_id']
    combined_df[person_spm_unit_col] = combined_df['_new_spm_id']
    
    # Marital units
    print("  Reindexing marital units...")
    marital_groups = combined_df.groupby([marital_unit_id_col, hh_id_col]).groups
    new_marital_id = 0
    marital_map = {}
    for (old_marital, hh), indices in marital_groups.items():
        for idx in indices:
            marital_map[idx] = new_marital_id
        new_marital_id += 1
    combined_df['_new_marital_id'] = combined_df.index.map(marital_map)
    combined_df[marital_unit_id_col] = combined_df['_new_marital_id']
    combined_df[person_marital_unit_col] = combined_df['_new_marital_id']
    
    # Clean up temporary columns
    temp_cols = [col for col in combined_df.columns if col.startswith('_')]
    combined_df = combined_df.drop(columns=temp_cols)
    
    print(f"  Final persons: {len(combined_df):,}")
    print(f"  Final households: {new_hh_id:,}")
    print(f"  Final tax units: {new_tax_id:,}")
    print(f"  Final SPM units: {new_spm_id:,}")
    print(f"  Final marital units: {new_marital_id:,}")
    
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
    
    for variable in sparse_sim.tax_benefit_system.variables:
        data[variable] = {}
        for period in sparse_sim.get_holder(variable).get_known_periods():
            values = sparse_sim.get_holder(variable).get_array(period)
            
            # Handle different value types
            if (
                sparse_sim.tax_benefit_system.variables.get(variable).value_type
                in (Enum, str)
                and variable != "county_fips"
            ):
                # Handle EnumArray objects
                if hasattr(values, 'decode_to_str'):
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
        
        if len(data[variable]) == 0:
            del data[variable]
    
    # Write to h5
    with h5py.File(output_path, "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)
    
    print(f"Sparse CD-stacked dataset saved successfully!")
    
    # Verify the saved file
    print("\nVerifying saved file...")
    with h5py.File(output_path, "r") as f:
        if "household_id" in f and str(time_period) in f["household_id"]:
            hh_ids = f["household_id"][str(time_period)][:]
            print(f"  Final households: {len(hh_ids):,}")
        if "person_id" in f and str(time_period) in f["person_id"]:
            person_ids = f["person_id"][str(time_period)][:]
            print(f"  Final persons: {len(person_ids):,}")
        if "household_weight" in f and str(time_period) in f["household_weight"]:
            weights = f["household_weight"][str(time_period)][:]
            print(f"  Total population: {np.sum(weights):,.0f}")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    # Load the calibrated CD weights
    print("Loading calibrated CD weights...")
    w = np.load("w_cd_20250911_102023.npy")
    
    print(f"Weight array shape: {w.shape}")
    print(f"Non-zero weights: {np.sum(w != 0):,}")
    print(f"Sparsity: {100*np.sum(w != 0)/len(w):.2f}%")
    
    # Get all CD GEOIDs from database (must match calibration order)
    print("\nRetrieving CD list from database...")
    db_path = download_from_huggingface('policy_data.db')
    db_uri = f'sqlite:///{db_path}'
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
    
    print(f"Found {len(cds_to_calibrate)} congressional districts")
    
    # Determine dataset path (stratified CPS was used for calibration)
    dataset_path = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/stratified_extended_cps_2023.h5"
    
    # Verify dimensions match
    expected_length = 436 * 13089  # 436 CDs × 13,089 households
    if len(w) != expected_length:
        print(f"WARNING: Weight vector length ({len(w):,}) doesn't match expected ({expected_length:,})")
        print("Attempting to continue anyway...")
    
    # Check for command line arguments for CD subset
    if len(sys.argv) > 1:
        if sys.argv[1] == "test10":
            # Test case: 10 diverse CDs from different states
            cd_subset = [
                '601',   # California CD 1
                '652',   # California CD 52
                '3601',  # New York CD 1
                '3626',  # New York CD 26
                '4801',  # Texas CD 1
                '4838',  # Texas CD 38
                '1201',  # Florida CD 1
                '1228',  # Florida CD 28
                '1701',  # Illinois CD 1
                '1101',  # DC at-large
            ]
            print(f"\nCreating dataset for 10 test CDs...")
            output_file = create_sparse_cd_stacked_dataset(
                w, cds_to_calibrate, 
                cd_subset=cd_subset,
                dataset_path=dataset_path
            )
        elif sys.argv[1] == "CA":
            # Test case: All California CDs (start with '6')
            cd_subset = [cd for cd in cds_to_calibrate if cd.startswith('6')]
            print(f"\nCreating dataset for {len(cd_subset)} California CDs...")
            output_file = create_sparse_cd_stacked_dataset(
                w, cds_to_calibrate, 
                cd_subset=cd_subset,
                dataset_path=dataset_path
            )
        elif sys.argv[1] == "test1":
            # Single CD test
            cd_subset = ['601']  # California CD 1
            print(f"\nCreating dataset for single test CD (CA-01)...")
            output_file = create_sparse_cd_stacked_dataset(
                w, cds_to_calibrate, 
                cd_subset=cd_subset,
                dataset_path=dataset_path
            )
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python create_sparse_cd_stacked_standalone.py [test1|test10|CA]")
            sys.exit(1)
    else:
        # Default: all CDs (WARNING: This will be large!)
        print("\nCreating dataset for ALL 436 congressional districts...")
        print("WARNING: This will create a large dataset with ~89K households!")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        
        output_file = create_sparse_cd_stacked_dataset(
            w, cds_to_calibrate,
            dataset_path=dataset_path
        )
    
    print(f"\nDone! Created: {output_file}")
    print("\nTo test loading:")
    print("  from policyengine_us import Microsimulation")
    print(f"  sim = Microsimulation(dataset='{output_file}')")
    print("  sim.build_from_dataset()")