"""
Create a stratified sample of extended_cps_2024.h5 that preserves high-income households
while maintaining diversity in lower income strata for poverty analysis.

Strategy:
- Per-bracket caps on the high-AGI tail (avoids PUF-template pile-up above $10M
  and ensures the $1M-$10M middle-high range has enough records for calibration)
- Uniform sample from the middle range below $500k
- Optional: slight oversample of bottom quartile for poverty-focused analysis
"""

import numpy as np
import h5py
from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import Enum


# Per-bracket caps for the high-AGI tail. The extended_cps passes through PUF
# templates with household_weight=0 whose only role is to be cloned by
# calibration; keeping thousands of them in the >$10M bracket dilutes optimizer
# leverage and leaves the $1.5M-$10M middle-high range starved. These caps give
# each middle-high band enough records to hit SOI bracket targets while keeping
# the $10M+ tail at a manageable size. Weighted CPS records are preferred over
# PUF templates when a bracket exceeds its cap.
HIGH_AGI_BRACKETS = [
    (500_000, 1_000_000, 400),
    (1_000_000, 2_000_000, 400),
    (2_000_000, 5_000_000, 400),
    (5_000_000, 10_000_000, 300),
    (10_000_000, float("inf"), 300),
]

TOP_AGI_FLOOR = HIGH_AGI_BRACKETS[0][0]  # $500k — boundary between top and middle


def _format_agi(x):
    if x == float("inf"):
        return "inf"
    if x >= 1e6:
        return f"${x / 1e6:.1f}M"
    return f"${x / 1e3:.0f}k"


def create_stratified_cps_dataset(
    target_households=30_000,
    oversample_poor=False,
    seed=None,
    base_dataset=None,
    output_path=None,
    high_agi_brackets=None,
):
    """
    Create a stratified sample of CPS data preserving high-income households
    while maintaining low-income diversity for poverty analysis.

    Args:
        target_households: Target number of households in output (approximate)
        oversample_poor: If True, boost sampling rate for bottom 25% by 1.5x
        seed: Random seed for reproducibility (default: None for random)
        base_dataset: Path to source h5 file (default: extended_cps_2024.h5)
        output_path: Where to save the stratified h5 file
        high_agi_brackets: List of (lo, hi, cap) tuples defining per-bracket
            caps for the high-AGI tail. Defaults to HIGH_AGI_BRACKETS.
    """
    if high_agi_brackets is None:
        high_agi_brackets = HIGH_AGI_BRACKETS
    print("\n" + "=" * 70)
    print("CREATING STRATIFIED CPS DATASET")
    print("=" * 70)

    # Default to local storage if no base_dataset specified
    if base_dataset is None:
        from policyengine_us_data.storage import STORAGE_FOLDER

        base_dataset = str(STORAGE_FOLDER / "extended_cps_2024.h5")

    # Load the original simulation
    print("Loading original dataset...")
    sim = Microsimulation(dataset=base_dataset)

    # Calculate AGI and household weights
    print("Calculating household AGI and weights...")
    agi = sim.calculate("adjusted_gross_income", map_to="household").values
    household_weight = sim.calculate("household_weight", map_to="household").values
    household_ids = sim.calculate("household_id", map_to="household").values
    n_households_orig = len(household_ids)

    print(f"Original dataset: {n_households_orig:,} households")
    print(f"Target dataset: {target_households:,} households")
    print(f"Reduction ratio: {target_households / n_households_orig:.1%}")

    # Show income distribution
    print("\nAGI Percentiles (original):")
    for p in [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        val = np.percentile(agi, p)
        print(f"  {p:5.1f}%: ${val:>12,.0f}")

    # Initialize random state and selection mask
    if seed is not None:
        np.random.seed(seed)
        print(f"\nUsing random seed: {seed}")
    selected_mask = np.zeros(n_households_orig, dtype=bool)

    # === Stratum 1: High-AGI tail (per-bracket caps) ===
    # CPS top-codes earnings around $1M, so essentially all records above that
    # are PUF templates with household_weight=0. Uncapped, they pile up in the
    # >$10M bracket and dominate the stratified dataset. Cap each bracket to a
    # target count, preferring weighted CPS records when available.
    print(f"\nSelecting high-AGI records by bracket:")
    print(
        f"  {'bracket':<22s} {'selected':>10s} {'available':>10s} "
        f"{'cap':>6s} {'weighted':>10s}"
    )
    for lo, hi, cap in high_agi_brackets:
        bracket_mask = (agi >= lo) & (agi < hi)
        bracket_indices = np.where(bracket_mask)[0]
        n_available = len(bracket_indices)
        label = f"[{_format_agi(lo)}, {_format_agi(hi)})"

        if n_available == 0:
            print(f"  {label:<22s} {0:>10,} {0:>10,} {cap:>6,} {0:>10,}")
            continue

        is_weighted = household_weight[bracket_indices] > 0
        weighted_idx = bracket_indices[is_weighted]
        unweighted_idx = bracket_indices[~is_weighted]

        if n_available <= cap:
            chosen = bracket_indices
        elif len(weighted_idx) >= cap:
            chosen = np.random.choice(weighted_idx, cap, replace=False)
        else:
            n_from_puf = cap - len(weighted_idx)
            chosen_puf = np.random.choice(unweighted_idx, n_from_puf, replace=False)
            chosen = np.concatenate([weighted_idx, chosen_puf])

        selected_mask[chosen] = True
        n_chosen_weighted = int((household_weight[chosen] > 0).sum())
        print(
            f"  {label:<22s} {len(chosen):>10,} {n_available:>10,} "
            f"{cap:>6,} {n_chosen_weighted:>10,}"
        )

    n_top_selected = int(selected_mask.sum())
    print(f"\n  High-AGI total selected: {n_top_selected:,}")

    # === Strata 2 & 3: Middle and bottom sampling ===
    # Everything below the top-bracket floor ($500k) is split by the 25th
    # percentile of the non-top records.
    non_top_mask = agi < TOP_AGI_FLOOR
    non_top_agi = agi[non_top_mask]
    if len(non_top_agi) == 0:
        bottom_25_threshold = 0.0
    else:
        bottom_25_threshold = float(np.percentile(non_top_agi, 25))
    bottom_mask = non_top_mask & (agi < bottom_25_threshold)
    middle_mask = non_top_mask & (agi >= bottom_25_threshold)
    n_bottom_25 = int(bottom_mask.sum())
    n_middle = int(middle_mask.sum())

    print(f"\nStratum sizes (below ${TOP_AGI_FLOOR:,.0f}):")
    print(f"  Bottom 25% (AGI < ${bottom_25_threshold:,.0f}): {n_bottom_25:,}")
    print(
        f"  Middle [${bottom_25_threshold:,.0f}, ${TOP_AGI_FLOOR:,.0f}): {n_middle:,}"
    )

    remaining_quota = target_households - n_top_selected
    if remaining_quota <= 0:
        print(
            f"\nWARNING: high-AGI bracket caps ({n_top_selected:,}) already "
            f"exceed target_households ({target_households:,}); no middle/bottom "
            f"sampling."
        )
        r_middle = 0.0
        r_bottom = 0.0
    elif oversample_poor:
        # Give bottom 25% a 1.5x boost relative to middle
        r_middle = remaining_quota / (1.5 * n_bottom_25 + n_middle)
        r_bottom = 1.5 * r_middle
        r_middle = min(1.0, r_middle)
        r_bottom = min(1.0, r_bottom)
    else:
        r_middle = remaining_quota / (n_bottom_25 + n_middle)
        r_bottom = r_middle
        r_middle = min(1.0, r_middle)
        r_bottom = min(1.0, r_bottom)

    print(f"\nSampling rates:")
    print(f"  Bottom 25%: {r_bottom:.1%}")
    print(f"  Middle: {r_middle:.1%}")

    # Select bottom 25%
    bottom_indices = np.where(bottom_mask)[0]
    n_select_bottom = int(len(bottom_indices) * r_bottom)
    if r_bottom >= 1.0:
        selected_mask[bottom_indices] = True
    elif n_select_bottom > 0:
        selected_bottom = np.random.choice(
            bottom_indices, n_select_bottom, replace=False
        )
        selected_mask[selected_bottom] = True

    # Select middle
    middle_indices = np.where(middle_mask)[0]
    n_select_middle = int(len(middle_indices) * r_middle)
    if r_middle >= 1.0:
        selected_mask[middle_indices] = True
    elif n_select_middle > 0:
        selected_middle = np.random.choice(
            middle_indices, n_select_middle, replace=False
        )
        selected_mask[selected_middle] = True

    print(f"\nFinal selection:")
    print(
        f"  High-AGI (>= ${TOP_AGI_FLOOR:,.0f}): "
        f"{int((selected_mask & ~non_top_mask).sum()):,}"
    )
    print(f"  Middle: {int((selected_mask & middle_mask).sum()):,} / {n_middle:,}")
    print(
        f"  Bottom 25%: {int((selected_mask & bottom_mask).sum()):,} / {n_bottom_25:,}"
    )
    n_selected = int(selected_mask.sum())
    print(
        f"  Total: {n_selected:,} households "
        f"({n_selected / n_households_orig:.1%} of original)"
    )

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
    stratified_sim = Microsimulation(dataset=base_dataset)
    stratified_sim.dataset = stratified_dataset
    stratified_sim.build_from_dataset()

    # Generate output path if not provided
    if output_path is None:
        from policyengine_us_data.storage import STORAGE_FOLDER

        output_path = str(STORAGE_FOLDER / "stratified_extended_cps_2024.h5")

    # Save to h5 file
    print(f"\nSaving to {output_path}...")
    data = {}

    # Only save input variables (not calculated/derived variables)
    input_vars = set(sim.input_variables)
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
        if "household_weight" in f and str(time_period) in f["household_weight"]:
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

    target = 12_000
    oversample = False
    seed = 3526

    for arg in sys.argv[1:]:
        if arg == "--oversample-poor":
            oversample = True
        elif arg.startswith("--top="):
            print(
                f"WARNING: --top is no longer supported; high-AGI records are now "
                f"selected via per-bracket caps (HIGH_AGI_BRACKETS). Ignoring '{arg}'."
            )
        elif arg.startswith("--seed="):
            seed = int(arg.split("=")[1])
        elif arg.isdigit():
            target = int(arg)

    print(f"Creating stratified dataset:")
    print(f"  Target households: {target:,}")
    print(f"  High-AGI bracket caps: {HIGH_AGI_BRACKETS}")
    print(f"  Oversample poor: {oversample}")
    print(f"  Seed: {seed if seed is not None else 'random'}")

    output_file = create_stratified_cps_dataset(
        target_households=target,
        oversample_poor=oversample,
        seed=seed,
    )

    print(f"\nDone! Created: {output_file}")
    print("\nUsage:")
    print("  python create_stratified_cps.py [target] [--oversample-poor] [--seed=N]")
    print("\nExamples:")
    print("  python create_stratified_cps.py 30000")
    print("  python create_stratified_cps.py 12000 --seed=123  # reproducible")
