"""
Shared utilities for calibration scripts.
"""

import os
import urllib
import tempfile
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd


def _get_geo_level(geo_id) -> int:
    """Return geographic level: 0=National, 1=State, 2=District."""
    if geo_id == 'US':
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
    1. Groups are ordered by geographic level: National → State → District
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
    targets_df['_geo_level'] = targets_df['geographic_id'].apply(_get_geo_level)

    geo_level_names = {0: "National", 1: "State", 2: "District"}

    # Process by geographic level: National (0) → State (1) → District (2)
    for level in [0, 1, 2]:
        level_mask = targets_df['_geo_level'] == level
        if not level_mask.any():
            continue

        level_name = geo_level_names.get(level, f"Level {level}")
        print(f"\n{level_name} targets:")

        # Get unique variables at this level
        level_df = targets_df[level_mask & ~processed_mask]
        unique_vars = sorted(level_df['variable'].unique())

        for var_name in unique_vars:
            var_mask = (
                (targets_df['variable'] == var_name)
                & level_mask
                & ~processed_mask
            )

            if not var_mask.any():
                continue

            matching = targets_df[var_mask]
            n_targets = var_mask.sum()
            n_geos = matching['geographic_id'].nunique()

            # Assign group
            target_groups[var_mask] = group_id
            processed_mask |= var_mask

            # Create descriptive label
            stratum_group = matching['stratum_group_id'].iloc[0]
            if var_name == "household_count" and stratum_group == 4:
                label = "SNAP Household Count"
            elif var_name == "snap":
                label = "Snap"
            else:
                label = var_name.replace("_", " ").title()

            # Format output based on level and count
            if n_targets == 1:
                value = matching['value'].iloc[0]
                info_str = f"{level_name} {label} (1 target, value={value:,.0f})"
                print_str = f"  Group {group_id}: {label} = {value:,.0f}"
            else:
                info_str = f"{level_name} {label} ({n_targets} targets)"
                print_str = f"  Group {group_id}: {label} ({n_targets} targets)"

            group_info.append(f"Group {group_id}: {info_str}")
            print(print_str)
            group_id += 1

    print(f"\nTotal groups created: {group_id}")
    print("=" * 40)

    return target_groups, group_info


# NOTE: this is for public files. A TODO is to contrast it with what we already have
def download_from_huggingface(file_name):
    """Download a file from HuggingFace to a temporary location."""
    base_url = "https://huggingface.co/policyengine/test/resolve/main/"
    url = base_url + file_name

    # Create temporary file
    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, file_name)

    # Check if already downloaded
    if not os.path.exists(local_path):
        print(f"Downloading {file_name} from HuggingFace...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded to {local_path}")
    else:
        print(f"Using cached {local_path}")

    return local_path


def uprate_target_value(
    value: float, variable_name: str, from_year: int, to_year: int, sim=None
) -> float:
    """
    Uprate a target value from source year to dataset year.

    Parameters
    ----------
    value : float
        The value to uprate
    variable_name : str
        Name of the variable (used to determine uprating type)
    from_year : int
        Source year of the value
    to_year : int
        Target year to uprate to
    sim : Microsimulation, optional
        Existing microsimulation instance for getting parameters

    Returns
    -------
    float
        Uprated value
    """
    if from_year == to_year:
        return value

    # Need PolicyEngine parameters for uprating factors
    if sim is None:
        from policyengine_us import Microsimulation

        sim = Microsimulation(
            dataset="hf://policyengine/test/extended_cps_2023.h5"
        )

    params = sim.tax_benefit_system.parameters

    # Determine uprating type based on variable
    # Count variables use population uprating
    count_variables = [
        "person_count",
        "household_count",
        "tax_unit_count",
        "spm_unit_count",
        "family_count",
        "marital_unit_count",
    ]

    if variable_name in count_variables:
        # Use population uprating for counts
        try:
            pop_from = params.calibration.gov.census.populations.total(
                from_year
            )
            pop_to = params.calibration.gov.census.populations.total(to_year)
            factor = pop_to / pop_from
        except Exception as e:
            print(
                f"Warning: Could not get population uprating for {from_year}->{to_year}: {e}"
            )
            factor = 1.0
    else:
        # Use CPI-U for monetary values (default)
        try:
            cpi_from = params.gov.bls.cpi.cpi_u(from_year)
            cpi_to = params.gov.bls.cpi.cpi_u(to_year)
            factor = cpi_to / cpi_from
        except Exception as e:
            print(
                f"Warning: Could not get CPI uprating for {from_year}->{to_year}: {e}"
            )
            factor = 1.0

    return value * factor


def uprate_targets_df(
    targets_df: pd.DataFrame, target_year: int, sim=None
) -> pd.DataFrame:
    """
    Uprate all targets in a DataFrame to the target year.

    Parameters
    ----------
    targets_df : pd.DataFrame
        DataFrame containing targets with 'period', 'variable', and 'value' columns
    target_year : int
        Year to uprate all targets to
    sim : Microsimulation, optional
        Existing microsimulation instance for getting parameters

    Returns
    -------
    pd.DataFrame
        DataFrame with uprated values and tracking columns:
        - original_value: The value before uprating
        - uprating_factor: The factor applied
        - uprating_source: 'CPI-U', 'Population', or 'None'
    """
    if "period" not in targets_df.columns:
        return targets_df

    df = targets_df.copy()

    # Check if already uprated (avoid double uprating)
    if "uprating_factor" in df.columns:
        return df

    # Store original values and initialize tracking columns
    df["original_value"] = df["value"]
    df["uprating_factor"] = 1.0
    df["uprating_source"] = "None"

    # Identify rows needing uprating
    needs_uprating = df["period"] != target_year

    if not needs_uprating.any():
        return df

    # Get parameters once
    if sim is None:
        from policyengine_us import Microsimulation

        sim = Microsimulation(
            dataset="hf://policyengine/test/extended_cps_2023.h5"
        )
    params = sim.tax_benefit_system.parameters

    # Get unique years that need uprating
    unique_years = set(df.loc[needs_uprating, "period"].unique())

    # Remove NaN values if any
    unique_years = {year for year in unique_years if pd.notna(year)}

    # Pre-calculate all uprating factors
    factors = {}
    for from_year in unique_years:
        # Convert numpy int64 to Python int for parameter lookups
        from_year_int = int(from_year)
        target_year_int = int(target_year)

        if from_year_int == target_year_int:
            factors[(from_year, "cpi")] = 1.0
            factors[(from_year, "population")] = 1.0
            continue

        # CPI-U factor
        try:
            cpi_from = params.gov.bls.cpi.cpi_u(from_year_int)
            cpi_to = params.gov.bls.cpi.cpi_u(target_year_int)
            factors[(from_year, "cpi")] = cpi_to / cpi_from
        except Exception as e:
            print(
                f"  Warning: CPI uprating failed for {from_year_int}->{target_year_int}: {e}"
            )
            factors[(from_year, "cpi")] = 1.0

        # Population factor
        try:
            pop_from = params.calibration.gov.census.populations.total(
                from_year_int
            )
            pop_to = params.calibration.gov.census.populations.total(
                target_year_int
            )
            factors[(from_year, "population")] = pop_to / pop_from
        except Exception as e:
            print(
                f"  Warning: Population uprating failed for {from_year_int}->{target_year_int}: {e}"
            )
            factors[(from_year, "population")] = 1.0

    # Define count variables (use population uprating)
    count_variables = {
        "person_count",
        "household_count",
        "tax_unit_count",
        "spm_unit_count",
        "family_count",
        "marital_unit_count",
    }

    # Vectorized application of uprating factors
    for from_year in unique_years:
        year_mask = (df["period"] == from_year) & needs_uprating

        # Population-based variables
        pop_mask = year_mask & df["variable"].isin(count_variables)
        if pop_mask.any():
            factor = factors[(from_year, "population")]
            df.loc[pop_mask, "value"] *= factor
            df.loc[pop_mask, "uprating_factor"] = factor
            df.loc[pop_mask, "uprating_source"] = "Population"

        # CPI-based variables (everything else)
        cpi_mask = year_mask & ~df["variable"].isin(count_variables)
        if cpi_mask.any():
            factor = factors[(from_year, "cpi")]
            df.loc[cpi_mask, "value"] *= factor
            df.loc[cpi_mask, "uprating_factor"] = factor
            df.loc[cpi_mask, "uprating_source"] = "CPI-U"

    # Summary logging (only if factors are not all 1.0)
    uprated_count = needs_uprating.sum()
    if uprated_count > 0:
        # Check if any real uprating happened
        cpi_factors = df.loc[
            df["uprating_source"] == "CPI-U", "uprating_factor"
        ]
        pop_factors = df.loc[
            df["uprating_source"] == "Population", "uprating_factor"
        ]

        cpi_changed = len(cpi_factors) > 0 and (cpi_factors != 1.0).any()
        pop_changed = len(pop_factors) > 0 and (pop_factors != 1.0).any()

        if cpi_changed or pop_changed:
            # Count unique source years (excluding NaN and target year)
            source_years = df.loc[needs_uprating, "period"].dropna().unique()
            source_years = [y for y in source_years if y != target_year]
            unique_sources = len(source_years)

            print(
                f"\n  ✓ Uprated {uprated_count:,} targets from year(s) {sorted(source_years)} to {target_year}"
            )

            if cpi_changed:
                cpi_count = (df["uprating_source"] == "CPI-U").sum()
                print(
                    f"    - {cpi_count:,} monetary targets: CPI factors {cpi_factors.min():.4f} - {cpi_factors.max():.4f}"
                )
            if pop_changed:
                pop_count = (df["uprating_source"] == "Population").sum()
                print(
                    f"    - {pop_count:,} count targets: Population factors {pop_factors.min():.4f} - {pop_factors.max():.4f}"
                )

    return df


def filter_target_groups(
    targets_df: pd.DataFrame,
    X_sparse,
    target_groups: np.ndarray,
    groups_to_exclude: List[int],
) -> Tuple[pd.DataFrame, any, np.ndarray]:
    """
    Filter out specified target groups from targets_df and X_sparse.

    Parameters
    ----------
    targets_df : pd.DataFrame
        DataFrame containing target metadata
    X_sparse : scipy.sparse matrix
        Sparse calibration matrix (rows = targets, cols = households)
    target_groups : np.ndarray
        Array of group IDs for each target
    groups_to_exclude : List[int]
        List of group IDs to exclude

    Returns
    -------
    filtered_targets_df : pd.DataFrame
        Filtered targets dataframe
    filtered_X_sparse : scipy.sparse matrix
        Filtered sparse matrix
    filtered_target_groups : np.ndarray
        Filtered target groups array
    """
    if len(groups_to_exclude) == 0:
        return targets_df, X_sparse, target_groups

    keep_mask = ~np.isin(target_groups, groups_to_exclude)

    n_to_remove = (~keep_mask).sum()
    is_national = targets_df["geographic_id"] == "US"
    n_national_removed = is_national[~keep_mask].sum()
    n_cd_removed = n_to_remove - n_national_removed

    print(f"\nExcluding groups: {groups_to_exclude}")
    print(f"Total targets removed: {n_to_remove} out of {len(targets_df)}")
    print(f"  - CD/state-level targets removed: {n_cd_removed}")
    print(f"  - National-level targets removed: {n_national_removed}")

    filtered_targets_df = targets_df[keep_mask].reset_index(drop=True)
    filtered_X_sparse = X_sparse[keep_mask, :]
    filtered_target_groups = target_groups[keep_mask]

    print(
        f"After filtering: {len(filtered_targets_df)} targets, matrix shape: {filtered_X_sparse.shape}"
    )

    return filtered_targets_df, filtered_X_sparse, filtered_target_groups


def get_cd_index_mapping():
    """
    Get the canonical CD GEOID to index mapping.
    This MUST be consistent across all uses!
    Each CD gets 10,000 IDs for each entity type.

    Returns:
        dict: Maps CD GEOID string to index (0-435)
        dict: Maps index to CD GEOID string
        list: Ordered list of CD GEOIDs
    """
    from sqlalchemy import create_engine, text
    from pathlib import Path
    import os

    script_dir = Path(__file__).parent
    db_path = script_dir.parent.parent.parent / "storage" / "policy_data.db"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database file not found at {db_path}. "
            f"Current working directory: {os.getcwd()}"
        )

    db_uri = f"sqlite:///{db_path}"
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
        cds_ordered = [row[0] for row in result]

    # Create bidirectional mappings
    cd_to_index = {cd: idx for idx, cd in enumerate(cds_ordered)}
    index_to_cd = {idx: cd for idx, cd in enumerate(cds_ordered)}

    return cd_to_index, index_to_cd, cds_ordered


def get_id_range_for_cd(cd_geoid, entity_type="household"):
    """
    Get the ID range for a specific CD and entity type.

    Args:
        cd_geoid: Congressional district GEOID string (e.g., '3701')
        entity_type: Entity type ('household', 'person', 'tax_unit', 'spm_unit', 'marital_unit')

    Returns:
        tuple: (start_id, end_id) inclusive
    """
    cd_to_index, _, _ = get_cd_index_mapping()

    if cd_geoid not in cd_to_index:
        raise ValueError(f"Unknown CD GEOID: {cd_geoid}")

    idx = cd_to_index[cd_geoid]
    base_start = idx * 10_000
    base_end = base_start + 9_999

    # Offset different entities to avoid ID collisions
    # Max base ID is 435 * 10,000 + 9,999 = 4,359,999
    # Must ensure max_id * 100 < 2,147,483,647 (int32 max)
    # So max_id must be < 21,474,836
    # NOTE: Currently only household/person use CD-based ranges
    # Tax/SPM/marital units still use sequential numbering from 0
    offsets = {
        "household": 0,  # Max: 4,359,999
        "person": 5_000_000,  # Max: 9,359,999
        "tax_unit": 0,  # Not implemented yet
        "spm_unit": 0,  # Not implemented yet
        "marital_unit": 0,  # Not implemented yet
    }

    offset = offsets.get(entity_type, 0)
    return base_start + offset, base_end + offset


def get_cd_from_id(entity_id):
    """
    Determine which CD an entity ID belongs to.

    Args:
        entity_id: The household/person/etc ID

    Returns:
        str: CD GEOID
    """
    # Remove offset to get base ID
    # Currently only persons have offset (5M)
    if entity_id >= 5_000_000:
        base_id = entity_id - 5_000_000  # Person
    else:
        base_id = entity_id  # Household (or tax/spm/marital unit)

    idx = base_id // 10_000
    _, index_to_cd, _ = get_cd_index_mapping()

    if idx not in index_to_cd:
        raise ValueError(
            f"ID {entity_id} (base {base_id}) maps to invalid CD index {idx}"
        )

    return index_to_cd[idx]
