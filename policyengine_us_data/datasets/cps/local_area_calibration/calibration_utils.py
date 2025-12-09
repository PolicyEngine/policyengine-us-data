"""
Shared utilities for calibration scripts.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def get_calculated_variables(sim) -> List[str]:
    """
    Return variables that should be cleared for state-swap recalculation.

    Includes variables with formulas, adds, or subtracts.

    Excludes ID variables (person_id, household_id, etc.) because:
    1. They have formulas that generate sequential IDs (0, 1, 2, ...)
    2. We need the original H5 values, not regenerated sequences
    3. PolicyEngine's random() function uses entity IDs as seeds:
       seed = abs(entity_id * 100 + count_random_calls)
       If IDs change, random-dependent variables (SSI resource test,
       WIC nutritional risk, WIC takeup) produce different results.
    """
    exclude_ids = {
        "person_id",
        "household_id",
        "tax_unit_id",
        "spm_unit_id",
        "family_id",
        "marital_unit_id",
    }
    return [
        name
        for name, var in sim.tax_benefit_system.variables.items()
        if (
            var.formulas
            or getattr(var, "adds", None)
            or getattr(var, "subtracts", None)
        )
        and name not in exclude_ids
    ]


def apply_op(values: np.ndarray, op: str, val: str) -> np.ndarray:
    """Apply constraint operation to values array."""
    try:
        parsed = float(val)
        if parsed.is_integer():
            parsed = int(parsed)
    except ValueError:
        if val == "True":
            parsed = True
        elif val == "False":
            parsed = False
        else:
            parsed = val

    if op in ("==", "="):
        return values == parsed
    if op == ">":
        return values > parsed
    if op == ">=":
        return values >= parsed
    if op == "<":
        return values < parsed
    if op == "<=":
        return values <= parsed
    if op == "!=":
        return values != parsed
    return np.ones(len(values), dtype=bool)


def _get_geo_level(geo_id) -> int:
    """Return geographic level: 0=National, 1=State, 2=District."""
    if geo_id == "US":
        return 0
    try:
        val = int(geo_id)
        return 1 if val < 100 else 2
    except (ValueError, TypeError):
        return 3


def create_target_groups(
    targets_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Automatically create target groups based on metadata.

    Grouping rules:
    1. Groups are ordered by geographic level: National -> State -> District
    2. Within each level, targets are grouped by variable type
    3. Each group contributes equally to the total loss

    Parameters
    ----------
    targets_df : pd.DataFrame
        DataFrame containing target metadata with columns:
        - stratum_group_id: Identifier for the type of target
        - geographic_id: Geographic identifier (US, state FIPS, CD GEOID)
        - variable: Variable name
        - value: Target value

    Returns
    -------
    target_groups : np.ndarray
        Array of group IDs for each target
    group_info : List[str]
        List of descriptive strings for each group
    """
    target_groups = np.zeros(len(targets_df), dtype=int)
    group_id = 0
    group_info = []
    processed_mask = np.zeros(len(targets_df), dtype=bool)

    print("\n=== Creating Target Groups ===")

    # Add geo_level column for sorting
    targets_df = targets_df.copy()
    targets_df["_geo_level"] = targets_df["geographic_id"].apply(
        _get_geo_level
    )

    geo_level_names = {0: "National", 1: "State", 2: "District"}

    # Process by geographic level: National (0) -> State (1) -> District (2)
    for level in [0, 1, 2]:
        level_mask = targets_df["_geo_level"] == level
        if not level_mask.any():
            continue

        level_name = geo_level_names.get(level, f"Level {level}")
        print(f"\n{level_name} targets:")

        # Get unique variables at this level
        level_df = targets_df[level_mask & ~processed_mask]
        unique_vars = sorted(level_df["variable"].unique())

        for var_name in unique_vars:
            var_mask = (
                (targets_df["variable"] == var_name)
                & level_mask
                & ~processed_mask
            )

            if not var_mask.any():
                continue

            matching = targets_df[var_mask]
            n_targets = var_mask.sum()

            # Assign group
            target_groups[var_mask] = group_id
            processed_mask |= var_mask

            # Create descriptive label
            stratum_group = matching["stratum_group_id"].iloc[0]
            if var_name == "household_count" and stratum_group == 4:
                label = "SNAP Household Count"
            elif var_name == "snap":
                label = "Snap"
            else:
                label = var_name.replace("_", " ").title()

            # Format output based on level and count
            if n_targets == 1:
                value = matching["value"].iloc[0]
                info_str = (
                    f"{level_name} {label} (1 target, value={value:,.0f})"
                )
                print_str = f"  Group {group_id}: {label} = {value:,.0f}"
            else:
                info_str = f"{level_name} {label} ({n_targets} targets)"
                print_str = (
                    f"  Group {group_id}: {label} ({n_targets} targets)"
                )

            group_info.append(f"Group {group_id}: {info_str}")
            print(print_str)
            group_id += 1

    print(f"\nTotal groups created: {group_id}")
    print("=" * 40)

    return target_groups, group_info
