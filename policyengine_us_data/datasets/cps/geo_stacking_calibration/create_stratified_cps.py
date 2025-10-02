"""
Create a stratified sample of extended_cps_2023.h5 that preserves high-income households.
This is needed for congressional district geo-stacking where the full dataset is too large.

Strategy:
- Keep ALL households above a high income threshold (e.g., top 1%)
- Sample progressively less from lower income strata
- Ensure representation across all income levels
"""

import numpy as np
import pandas as pd
import h5py
from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import Enum


def create_stratified_cps_dataset(
    target_households=30_000,
    high_income_percentile=99,  # Keep ALL households above this percentile
    output_path=None
):
    """
    Create a stratified sample of CPS data preserving high-income households.
    
    Args:
        target_households: Target number of households in output (approximate)
        high_income_percentile: Keep ALL households above this AGI percentile
        output_path: Where to save the stratified h5 file
    """
    print("\n" + "=" * 70)
    print("CREATING STRATIFIED CPS DATASET")
    print("=" * 70)
    
    # Load the original simulation
    print("Loading original dataset...")
    sim = Microsimulation(dataset="hf://policyengine/test/extended_cps_2023.h5")
    
    # Calculate AGI for all households
    print("Calculating household AGI...")
    agi = sim.calculate('adjusted_gross_income', map_to="household").values
    household_ids = sim.calculate("household_id", map_to="household").values
    n_households_orig = len(household_ids)
    
    print(f"Original dataset: {n_households_orig:,} households")
    print(f"Target dataset: {target_households:,} households")
    print(f"Reduction ratio: {target_households/n_households_orig:.1%}")
    
    # Calculate AGI percentiles
    print("\nAnalyzing income distribution...")
    percentiles = [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    agi_percentiles = np.percentile(agi, percentiles)
    
    print("AGI Percentiles:")
    for p, val in zip(percentiles, agi_percentiles):
        print(f"  {p:5.1f}%: ${val:,.0f}")
    
    # Define sampling strategy
    # Keep ALL high earners, sample progressively less from lower strata
    high_income_threshold = np.percentile(agi, high_income_percentile)
    print(f"\nHigh-income threshold (top {100-high_income_percentile}%): ${high_income_threshold:,.0f}")
    
    # Create strata with sampling rates
    strata = [
        (99.9, 100, 1.00),    # Top 0.1% - keep ALL
        (99.5, 99.9, 1.00),   # 99.5-99.9% - keep ALL
        (99, 99.5, 1.00),     # 99-99.5% - keep ALL
        (95, 99, 0.80),       # 95-99% - keep 80%
        (90, 95, 0.60),       # 90-95% - keep 60%
        (75, 90, 0.40),       # 75-90% - keep 40%
        (50, 75, 0.25),       # 50-75% - keep 25%
        (25, 50, 0.15),       # 25-50% - keep 15%
        (0, 25, 0.10),        # Bottom 25% - keep 10%
    ]
    
    # Adjust sampling rates to hit target
    print("\nInitial sampling strategy:")
    expected_count = 0
    for low_p, high_p, rate in strata:
        low_val = np.percentile(agi, low_p) if low_p > 0 else -np.inf
        high_val = np.percentile(agi, high_p) if high_p < 100 else np.inf
        in_stratum = np.sum((agi > low_val) & (agi <= high_val))
        expected = int(in_stratum * rate)
        expected_count += expected
        print(f"  {low_p:5.1f}-{high_p:5.1f}%: {in_stratum:6,} households × {rate:.0%} = {expected:6,}")
    
    print(f"Expected total: {expected_count:,} households")
    
    # Adjust rates if needed
    if expected_count > target_households * 1.1:  # Allow 10% overage
        adjustment = target_households / expected_count
        print(f"\nAdjusting rates by factor of {adjustment:.2f} to meet target...")
        
        # Never reduce the top percentiles
        strata_adjusted = []
        for low_p, high_p, rate in strata:
            if high_p >= 99:  # Never reduce top 1%
                strata_adjusted.append((low_p, high_p, rate))
            else:
                strata_adjusted.append((low_p, high_p, min(1.0, rate * adjustment)))
        strata = strata_adjusted
    
    # Select households based on strata
    print("\nSelecting households...")
    selected_mask = np.zeros(n_households_orig, dtype=bool)
    
    for low_p, high_p, rate in strata:
        low_val = np.percentile(agi, low_p) if low_p > 0 else -np.inf
        high_val = np.percentile(agi, high_p) if high_p < 100 else np.inf
        
        in_stratum = (agi > low_val) & (agi <= high_val)
        stratum_indices = np.where(in_stratum)[0]
        n_in_stratum = len(stratum_indices)
        
        if rate >= 1.0:
            # Keep all
            selected_mask[stratum_indices] = True
            n_selected = n_in_stratum
        else:
            # Random sample within stratum
            n_to_select = int(n_in_stratum * rate)
            if n_to_select > 0:
                np.random.seed(42)  # For reproducibility
                selected_indices = np.random.choice(stratum_indices, n_to_select, replace=False)
                selected_mask[selected_indices] = True
                n_selected = n_to_select
            else:
                n_selected = 0
        
        print(f"  {low_p:5.1f}-{high_p:5.1f}%: Selected {n_selected:6,} / {n_in_stratum:6,} ({n_selected/max(1,n_in_stratum):.0%})")
    
    n_selected = np.sum(selected_mask)
    print(f"\nTotal selected: {n_selected:,} households ({n_selected/n_households_orig:.1%} of original)")
    
    # Verify high earners are preserved
    high_earners_mask = agi >= high_income_threshold
    n_high_earners = np.sum(high_earners_mask)
    n_high_earners_selected = np.sum(selected_mask & high_earners_mask)
    print(f"\nHigh earners (>=${high_income_threshold:,.0f}):")
    print(f"  Original: {n_high_earners:,}")
    print(f"  Selected: {n_high_earners_selected:,} ({n_high_earners_selected/n_high_earners:.0%})")
    
    # Get the selected household IDs
    selected_household_ids = set(household_ids[selected_mask])
    
    # Now filter the dataset using DataFrame approach (similar to create_sparse_state_stacked.py)
    print("\nCreating filtered dataset...")
    time_period = int(sim.default_calculation_period)
    
    # Convert full simulation to DataFrame
    df = sim.to_input_dataframe()
    
    # Filter to selected households
    hh_id_col = f"household_id__{time_period}"
    df_filtered = df[df[hh_id_col].isin(selected_household_ids)].copy()
    
    print(f"Filtered DataFrame: {len(df_filtered):,} persons")
    
    # Create Dataset from filtered DataFrame
    print("Creating Dataset from filtered DataFrame...")
    stratified_dataset = Dataset.from_dataframe(df_filtered, time_period)
    
    # Build a simulation to convert to h5
    print("Building simulation from Dataset...")
    stratified_sim = Microsimulation()
    stratified_sim.dataset = stratified_dataset
    stratified_sim.build_from_dataset()
    
    # Generate output path if not provided
    if output_path is None:
        output_path = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/stratified_extended_cps_2023.h5"
    
    # Save to h5 file
    print(f"\nSaving to {output_path}...")
    data = {}
    
    for variable in stratified_sim.tax_benefit_system.variables:
        data[variable] = {}
        for period in stratified_sim.get_holder(variable).get_known_periods():
            values = stratified_sim.get_holder(variable).get_array(period)
            
            # Handle different value types
            if variable == "county_fips":
                values = values.astype("int32")
            elif stratified_sim.tax_benefit_system.variables.get(variable).value_type in (Enum, str):
                # Check if it's an EnumArray with decode_to_str method
                if hasattr(values, 'decode_to_str'):
                    values = values.decode_to_str().astype("S")
                else:
                    # Already a numpy array, just ensure it's string type
                    values = values.astype("S")
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
    
    print(f"Stratified CPS dataset saved successfully!")
    
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
            print(f"  Final household weights sum: {np.sum(weights):,.0f}")
    
    # Final income distribution check
    print("\nVerifying income distribution in stratified dataset...")
    stratified_sim_verify = Microsimulation(dataset=output_path)
    agi_stratified = stratified_sim_verify.calculate('adjusted_gross_income', map_to="household").values
    
    print("AGI Percentiles in stratified dataset:")
    for p in [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        val = np.percentile(agi_stratified, p)
        print(f"  {p:5.1f}%: ${val:,.0f}")
    
    max_agi_original = np.max(agi)
    max_agi_stratified = np.max(agi_stratified)
    print(f"\nMaximum AGI:")
    print(f"  Original: ${max_agi_original:,.0f}")
    print(f"  Stratified: ${max_agi_stratified:,.0f}")
    
    if max_agi_stratified < max_agi_original * 0.9:
        print("WARNING: May have lost some ultra-high earners!")
    else:
        print("Ultra-high earners preserved!")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
            print(f"Creating stratified dataset with target of {target:,} households...")
            output_file = create_stratified_cps_dataset(target_households=target)
        except ValueError:
            print(f"Invalid target households: {sys.argv[1]}")
            print("Usage: python create_stratified_cps.py [target_households]")
            sys.exit(1)
    else:
        # Default target
        print("Creating stratified dataset with default target of 30,000 households...")
        output_file = create_stratified_cps_dataset(target_households=30_000)
    
    print(f"\nDone! Created: {output_file}")
    print("\nTo test loading:")
    print("  from policyengine_us import Microsimulation")
    print(f"  sim = Microsimulation(dataset='{output_file}')")
