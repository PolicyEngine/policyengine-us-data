"""
Create a sparse state-stacked dataset with only non-zero weight households.
Uses DataFrame approach to ensure all entity relationships are preserved correctly.

IMPORTANT: This must use the same simulation that was used for calibration:
- extended_cps_2023.h5 from HuggingFace or local storage
- This dataset has 112,502 households
"""

import numpy as np
import pandas as pd
import h5py
import os
from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import Enum


def create_sparse_state_stacked_dataset(
    w, 
    states_to_calibrate, 
    output_path="/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/sparse_state_stacked_2023.h5"
):
    """
    Create a SPARSE state-stacked dataset using DataFrame approach.
    
    This method:
    1. Creates a simulation for each state with calibrated weights
    2. Converts to DataFrame (which handles all entity relationships)
    3. Modifies IDs to be unique across states
    4. Filters to only non-zero weight households
    5. Combines all states and saves as h5
    
    Args:
        w: Calibrated weight vector from L0 calibration (length = n_households * n_states)
        states_to_calibrate: List of state FIPS codes used in calibration
        output_path: Where to save the sparse state-stacked h5 file
    """
    print("\n" + "=" * 70)
    print("CREATING SPARSE STATE-STACKED DATASET (DataFrame approach)")
    print("=" * 70)
    
    # Load the original simulation
    base_sim = Microsimulation(dataset="hf://policyengine/test/extended_cps_2023.h5")
    
    # Get household IDs and create mapping
    household_ids = base_sim.calculate("household_id", map_to="household").values
    n_households_orig = len(household_ids)
    
    # Create mapping from household ID to index for proper filtering
    hh_id_to_idx = {int(hh_id): idx for idx, hh_id in enumerate(household_ids)}
    
    # Validate weight vector
    expected_weight_length = n_households_orig * len(states_to_calibrate)
    assert len(w) == expected_weight_length, (
        f"Weight vector length mismatch! Expected {expected_weight_length:,} "
        f"(={n_households_orig:,} households × {len(states_to_calibrate)} states), "
        f"but got {len(w):,}"
    )
    
    print(f"\nOriginal dataset has {n_households_orig:,} households")
    print(f"Processing {len(states_to_calibrate)} states...")
    
    # Process the weight vector to understand active household-state pairs
    print("\nProcessing weight vector...")
    W = w.reshape(len(states_to_calibrate), n_households_orig)
    
    # Count total active weights
    total_active_weights = np.sum(W > 0)
    print(f"Total active household-state pairs: {total_active_weights:,}")
    
    # Collect DataFrames for each state
    state_dfs = []
    total_kept_households = 0
    time_period = int(base_sim.default_calculation_period)
    
    for state_idx, state_fips in enumerate(states_to_calibrate):
        print(f"\nProcessing state {state_fips} ({state_idx + 1}/{len(states_to_calibrate)})...")
        
        # Get ALL households with non-zero weight in this state
        # (not just those "assigned" to this state)
        active_household_indices = np.where(W[state_idx, :] > 0)[0]
        
        if len(active_household_indices) == 0:
            print(f"  No households active in state {state_fips}, skipping...")
            continue
        
        print(f"  Households active in this state: {len(active_household_indices):,}")
        
        # Get the household IDs for active households
        active_household_ids = set(household_ids[idx] for idx in active_household_indices)
        
        # Create weight vector with weights for this state
        state_weights = np.zeros(n_households_orig)
        state_weights[active_household_indices] = W[state_idx, active_household_indices]
        
        # Create a simulation with these weights
        state_sim = Microsimulation(dataset="hf://policyengine/test/extended_cps_2023.h5")
        state_sim.set_input("household_weight", time_period, state_weights)
        
        # Convert to DataFrame
        df = state_sim.to_input_dataframe()
        
        # Column names follow pattern: variable__year
        hh_weight_col = f"household_weight__{time_period}"
        hh_id_col = f"household_id__{time_period}"
        person_id_col = f"person_id__{time_period}"
        person_hh_id_col = f"person_household_id__{time_period}"
        tax_unit_id_col = f"tax_unit_id__{time_period}"
        person_tax_unit_col = f"person_tax_unit_id__{time_period}"
        spm_unit_id_col = f"spm_unit_id__{time_period}"
        person_spm_unit_col = f"person_spm_unit_id__{time_period}"
        marital_unit_id_col = f"marital_unit_id__{time_period}"
        person_marital_unit_col = f"person_marital_unit_id__{time_period}"
        state_fips_col = f"state_fips__{time_period}"
        
        # Filter to only active households in this state
        df_filtered = df[df[hh_id_col].isin(active_household_ids)].copy()
        
        # Verify filtering worked correctly
        kept_hh_ids = df_filtered[hh_id_col].unique()
        if len(kept_hh_ids) != len(active_household_ids):
            print(f"  WARNING: Expected {len(active_household_ids)} households, but got {len(kept_hh_ids)}")
        
        # Skip ID modification - we'll reindex everything at the end anyway
        # This avoids any risk of overflow from large offsets
        
        # Update state_fips to target state
        df_filtered[state_fips_col] = state_fips
        
        state_dfs.append(df_filtered)
        total_kept_households += len(kept_hh_ids)
        
        print(f"  Kept {len(kept_hh_ids):,} households")
    
    print(f"\nCombining {len(state_dfs)} state DataFrames...")
    print(f"Total households across all states: {total_kept_households:,}")
    
    # Combine all state DataFrames
    combined_df = pd.concat(state_dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    # REINDEX ALL IDs TO PREVENT OVERFLOW AND HANDLE DUPLICATES
    # After combining, we have duplicate IDs (same household in multiple states)
    # We need to treat each occurrence as a unique entity
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
    
    # IMPORTANT: We need to treat each row as unique, even if IDs repeat
    # because the same household can appear in multiple states
    
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
    # Create mapping from old household ID + row context to new household ID
    old_to_new_hh = {}
    for idx, row in combined_df.iterrows():
        old_hh = row[hh_id_col]
        new_hh = row['_new_hh_id']
        # Store mapping for this specific occurrence
        if old_hh not in old_to_new_hh:
            old_to_new_hh[old_hh] = {}
        state = row[f"state_fips__{time_period}"]
        old_to_new_hh[old_hh][state] = new_hh
    
    # Update household IDs
    combined_df[hh_id_col] = combined_df['_new_hh_id']
    
    # For person household references, we need to match based on state
    state_col = f"state_fips__{time_period}"
    def map_person_hh(row):
        old_hh = row[person_hh_id_col]
        state = row[state_col]
        if old_hh in old_to_new_hh and state in old_to_new_hh[old_hh]:
            return old_to_new_hh[old_hh][state]
        # Fallback - this shouldn't happen
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
                values = values.decode_to_str().astype("S")
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
    
    print(f"Sparse state-stacked dataset saved successfully!")
    
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
    # Load the calibrated weights
    print("Loading calibrated weights...")
    w = np.load("/home/baogorek/Downloads/w_array_20250908_185748.npy")
    
    # Define states in calibration order (MUST match calibration)
    states_to_calibrate = [
        '1', '2', '4', '5', '6', '8', '9', '10', '11', '12', '13', '15', '16', '17', '18', 
        '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', 
        '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', 
        '48', '49', '50', '51', '53', '54', '55', '56'
    ]
    
    n_active = sum(w != 0)
    print(f"Sparsity: {n_active} active weights out of {len(w)} ({100*n_active/len(w):.2f}%)")
    
    # Create sparse state-stacked dataset
    output_file = create_sparse_state_stacked_dataset(w, states_to_calibrate)
    
    print(f"\nDone! Created: {output_file}")
    print("\nTo test loading:")
    print("  from policyengine_us import Microsimulation")
    print(f"  sim = Microsimulation(dataset='{output_file}')")
    print("  sim.build_from_dataset()")
