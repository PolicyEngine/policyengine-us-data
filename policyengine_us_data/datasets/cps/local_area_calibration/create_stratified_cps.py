"""
Create a stratified sample of extended_cps_2023.h5 that preserves high-income households
while maintaining diversity in lower income strata for poverty analysis.

Strategy:
- Keep ALL households in top 1% (for high-income tax analysis)
- Uniform sample from the remaining 99% (preserves low-income diversity)
- Optional: slight oversample of bottom quartile for poverty-focused analysis
"""

import numpy as np
import h5py
from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import Enum
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_pseudo_input_variables,
)


def create_stratified_cps_dataset(
    target_households=30_000,
    high_income_percentile=99,
    oversample_poor=False,
    seed=None,
    base_dataset=None,
    output_path=None,
):
    """
    Create a stratified sample of CPS data preserving high-income households
    while maintaining low-income diversity for poverty analysis.

    Args:
        target_households: Target number of households in output (approximate)
        high_income_percentile: Keep ALL households above this AGI percentile (e.g., 99 or 99.5)
        oversample_poor: If True, boost sampling rate for bottom 25% by 1.5x
        seed: Random seed for reproducibility (default: None for random)
        base_dataset: Path to source h5 file (default: extended_cps_2023.h5)
        output_path: Where to save the stratified h5 file
    """
    print("\n" + "=" * 70)
    print("CREATING STRATIFIED CPS DATASET")
    print("=" * 70)

    # Default to local storage if no base_dataset specified
    if base_dataset is None:
        from policyengine_us_data.storage import STORAGE_FOLDER

        base_dataset = str(STORAGE_FOLDER / "extended_cps_2023.h5")

    # Load the original simulation
    print("Loading original dataset...")
    sim = Microsimulation(dataset=base_dataset)

    # Calculate AGI for all households
    print("Calculating household AGI...")
    agi = sim.calculate("adjusted_gross_income", map_to="household").values
    household_ids = sim.calculate("household_id", map_to="household").values
    n_households_orig = len(household_ids)

    print(f"Original dataset: {n_households_orig:,} households")
    print(f"Target dataset: {target_households:,} households")
    print(f"Reduction ratio: {target_households/n_households_orig:.1%}")

    # Show income distribution
    print("\nAGI Percentiles (original):")
    for p in [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        val = np.percentile(agi, p)
        print(f"  {p:5.1f}%: ${val:>12,.0f}")

    # Define strata thresholds
    high_income_threshold = np.percentile(agi, high_income_percentile)
    bottom_25_pct_threshold = np.percentile(agi, 25)

    # Count households in each stratum
    n_top = np.sum(agi >= high_income_threshold)
    n_bottom_25 = np.sum(agi < bottom_25_pct_threshold)
    n_middle = n_households_orig - n_top - n_bottom_25

    print(f"\nStratum sizes:")
    print(f"  Top {100 - high_income_percentile}% (AGI >= ${high_income_threshold:,.0f}): {n_top:,}")
    print(f"  Middle 25-{high_income_percentile}%: {n_middle:,}")
    print(f"  Bottom 25% (AGI < ${bottom_25_pct_threshold:,.0f}): {n_bottom_25:,}")

    # Calculate sampling rates
    # Keep ALL top earners, distribute remaining quota between middle and bottom
    remaining_quota = target_households - n_top
    if remaining_quota <= 0:
        raise ValueError(
            f"Target ({target_households:,}) is less than top {100-high_income_percentile}% "
            f"count ({n_top:,}). Increase target_households."
        )

    if oversample_poor:
        # Give bottom 25% a 1.5x boost relative to middle
        r_middle = remaining_quota / (1.5 * n_bottom_25 + n_middle)
        r_bottom = 1.5 * r_middle
        r_middle = min(1.0, r_middle)
        r_bottom = min(1.0, r_bottom)
    else:
        # Uniform sampling for the rest
        r_middle = remaining_quota / (n_bottom_25 + n_middle)
        r_bottom = r_middle
        r_middle = min(1.0, r_middle)
        r_bottom = min(1.0, r_bottom)

    print(f"\nSampling rates:")
    print(f"  Top {100 - high_income_percentile}%: 100%")
    print(f"  Middle 25-{high_income_percentile}%: {r_middle:.1%}")
    print(f"  Bottom 25%: {r_bottom:.1%}")

    # Expected counts
    expected_top = n_top
    expected_middle = int(n_middle * r_middle)
    expected_bottom = int(n_bottom_25 * r_bottom)
    expected_total = expected_top + expected_middle + expected_bottom

    print(f"\nExpected selection:")
    print(f"  Top {100 - high_income_percentile}%: {expected_top:,}")
    print(f"  Middle 25-{high_income_percentile}%: {expected_middle:,}")
    print(f"  Bottom 25%: {expected_bottom:,}")
    print(f"  Total: {expected_total:,}")

    # Select households
    print("\nSelecting households...")
    if seed is not None:
        np.random.seed(seed)
        print(f"  Using random seed: {seed}")
    selected_mask = np.zeros(n_households_orig, dtype=bool)

    # Top earners - keep all
    top_mask = agi >= high_income_threshold
    selected_mask[top_mask] = True
    print(f"  Top {100 - high_income_percentile}%: selected {np.sum(top_mask):,}")

    # Bottom 25%
    bottom_mask = agi < bottom_25_pct_threshold
    bottom_indices = np.where(bottom_mask)[0]
    n_select_bottom = int(len(bottom_indices) * r_bottom)
    if r_bottom >= 1.0:
        selected_mask[bottom_indices] = True
    elif n_select_bottom > 0:
        selected_bottom = np.random.choice(bottom_indices, n_select_bottom, replace=False)
        selected_mask[selected_bottom] = True
    else:
        print(f"  WARNING: Bottom 25% selection rounded to 0 (rate={r_bottom:.4f}, n={len(bottom_indices)})")
    print(f"  Bottom 25%: selected {np.sum(selected_mask & bottom_mask):,} / {len(bottom_indices):,}")

    # Middle
    middle_mask = ~top_mask & ~bottom_mask
    middle_indices = np.where(middle_mask)[0]
    n_select_middle = int(len(middle_indices) * r_middle)
    if r_middle >= 1.0:
        selected_mask[middle_indices] = True
    elif n_select_middle > 0:
        selected_middle = np.random.choice(middle_indices, n_select_middle, replace=False)
        selected_mask[selected_middle] = True
    else:
        print(f"  WARNING: Middle selection rounded to 0 (rate={r_middle:.4f}, n={len(middle_indices)})")
    print(f"  Middle 25-{high_income_percentile}%: selected {np.sum(selected_mask & middle_mask):,} / {len(middle_indices):,}")

    n_selected = np.sum(selected_mask)
    print(f"\nTotal selected: {n_selected:,} households ({n_selected/n_households_orig:.1%} of original)")

    # Verify high earners are preserved
    print(f"\nHigh earners (>=${high_income_threshold:,.0f}): {np.sum(selected_mask & top_mask):,} / {n_top:,} (100%)")

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
        from policyengine_us_data.storage import STORAGE_FOLDER

        output_path = str(STORAGE_FOLDER / "stratified_extended_cps_2023.h5")

    # Save to h5 file
    print(f"\nSaving to {output_path}...")
    data = {}

    # Only save input variables (not calculated/derived variables)
    input_vars = set(sim.input_variables)

    # Filter out pseudo-inputs: variables with adds/subtracts that aggregate
    # formula-based components. These have stale values that corrupt calculations.
    pseudo_inputs = get_pseudo_input_variables(sim)
    if pseudo_inputs:
        print(f"Excluding {len(pseudo_inputs)} pseudo-input variables:")
        for var in sorted(pseudo_inputs):
            print(f"  - {var}")
        input_vars = input_vars - pseudo_inputs

    print(f"Found {len(input_vars)} input variables to save")

    for variable in stratified_sim.tax_benefit_system.variables:
        if variable not in input_vars:
            continue

        data[variable] = {}
        for period in stratified_sim.get_holder(variable).get_known_periods():
            values = stratified_sim.get_holder(variable).get_array(period)

            # Handle different value types
            if variable == "county_fips":
                values = values.astype("int32")
            elif stratified_sim.tax_benefit_system.variables.get(
                variable
            ).value_type in (Enum, str):
                # Check if it's an EnumArray with decode_to_str method
                if hasattr(values, "decode_to_str"):
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
        if (
            "household_weight" in f
            and str(time_period) in f["household_weight"]
        ):
            weights = f["household_weight"][str(time_period)][:]
            print(f"  Final household weights sum: {np.sum(weights):,.0f}")

    # Final income distribution check
    print("\nVerifying income distribution in stratified dataset...")
    stratified_sim_verify = Microsimulation(dataset=output_path)
    agi_stratified = stratified_sim_verify.calculate(
        "adjusted_gross_income", map_to="household"
    ).values

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

    target = 30_000
    high_pct = 99
    oversample = False
    seed = None

    for arg in sys.argv[1:]:
        if arg == "--oversample-poor":
            oversample = True
        elif arg.startswith("--top="):
            high_pct = float(arg.split("=")[1])
        elif arg.startswith("--seed="):
            seed = int(arg.split("=")[1])
        elif arg.isdigit():
            target = int(arg)

    print(f"Creating stratified dataset:")
    print(f"  Target households: {target:,}")
    print(f"  Keep all above: {high_pct}th percentile")
    print(f"  Oversample poor: {oversample}")
    print(f"  Seed: {seed if seed is not None else 'random'}")

    output_file = create_stratified_cps_dataset(
        target_households=target,
        high_income_percentile=high_pct,
        oversample_poor=oversample,
        seed=seed,
    )

    print(f"\nDone! Created: {output_file}")
    print("\nUsage:")
    print("  python create_stratified_cps.py [target] [--top=99] [--oversample-poor] [--seed=N]")
    print("\nExamples:")
    print("  python create_stratified_cps.py 30000")
    print("  python create_stratified_cps.py 50000 --top=99.5 --oversample-poor")
    print("  python create_stratified_cps.py 30000 --seed=123  # reproducible")
