"""
Shared utilities for calibration scripts.
"""

from typing import List, Tuple
import json
import numpy as np
import pandas as pd
from scipy import sparse

from policyengine_us.variables.household.demographic.geographic.state_name import (
    StateName,
)
from policyengine_us.variables.household.demographic.geographic.state_code import (
    StateCode,
)

# State/Geographic Mappings
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


def get_calculated_variables(sim) -> List[str]:
    """
    Return variables that should be cleared for state-swap recalculation.

    Includes variables with formulas, or adds/subtracts that are lists.

    Excludes:
    1. ID variables (person_id, household_id, etc.) - needed for random seeds
    2. Variables with string adds/subtracts (parameter paths) - these are
       pseudo-inputs stored in H5 that would recalculate differently using
       parameter lookups. Examples: pre_tax_contributions.
    3. Variables in input_variables (have stored H5 values) even if they
       have formulas - the stored values represent original survey data
       that should be preserved. Examples: cdcc_relevant_expenses, rent.

    The exclusions are critical because:
    - The H5 file stores pre-computed values from original CPS processing
    - If deleted, recalculation produces different values, corrupting
      downstream calculations like income_tax
    """
    exclude_ids = {
        "person_id",
        "household_id",
        "tax_unit_id",
        "spm_unit_id",
        "family_id",
        "marital_unit_id",
    }

    # Get stored input variables to exclude
    input_vars = set(sim.input_variables)

    result = []
    for name, var in sim.tax_benefit_system.variables.items():
        if name in exclude_ids:
            continue

        # Exclude variables that have stored values (input_variables)
        # These represent original survey data that should be preserved
        if name in input_vars:
            continue

        # Include if has formulas
        if var.formulas:
            result.append(name)
            continue

        # Include if adds/subtracts is a list (explicit component aggregation)
        # Exclude if adds/subtracts is a string (parameter path - pseudo-input)
        adds = getattr(var, "adds", None)
        subtracts = getattr(var, "subtracts", None)

        if adds and isinstance(adds, list):
            result.append(name)
        elif subtracts and isinstance(subtracts, list):
            result.append(name)

    return result


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

    values = np.asarray(values)
    if values.dtype.kind == "S" and isinstance(parsed, str):
        parsed = parsed.encode()

    if op == "in":
        allowed = [part.strip() for part in val.split("|")]
        if values.dtype.kind == "S":
            allowed = [part.encode() for part in allowed]
        return np.isin(values, allowed)
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


def get_geo_level(geo_id) -> int:
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
        - domain_variable: Domain variable from stratum constraints
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
    targets_df["_geo_level"] = targets_df["geographic_id"].apply(get_geo_level)

    geo_level_names = {0: "National", 1: "State", 2: "District"}

    # Process by geographic level: National (0) -> State (1) -> District (2)
    for level in [0, 1, 2]:
        level_mask = targets_df["_geo_level"] == level
        if not level_mask.any():
            continue

        level_name = geo_level_names.get(level, f"Level {level}")
        print(f"\n{level_name} targets:")

        # Get unique (domain_variable, variable) pairs at this level
        level_df = targets_df[level_mask & ~processed_mask]
        has_domain = "domain_variable" in level_df.columns
        if has_domain:
            pairs = sorted(
                level_df[["domain_variable", "variable"]]
                .drop_duplicates()
                .itertuples(index=False, name=None),
                key=lambda pair: (
                    pair[0] is not None,
                    "" if pair[0] is None else str(pair[0]),
                    str(pair[1]),
                ),
            )
        else:
            pairs = [(None, v) for v in sorted(level_df["variable"].unique())]

        for domain_var, var_name in pairs:
            var_mask = (
                (targets_df["variable"] == var_name) & level_mask & ~processed_mask
            )
            if has_domain:
                if domain_var is None:
                    var_mask &= targets_df["domain_variable"].isna()
                else:
                    var_mask &= targets_df["domain_variable"] == domain_var

            if not var_mask.any():
                continue

            matching = targets_df[var_mask]
            n_targets = var_mask.sum()

            # Assign group
            target_groups[var_mask] = group_id
            processed_mask |= var_mask

            # Create descriptive label using domain_variable for context
            var_label = var_name.replace("_", " ").title()
            if domain_var and domain_var != var_name:
                domain_label = domain_var.replace("_", " ").upper()
                label = f"{domain_label} {var_label}"
            else:
                label = var_label

            # Format output based on level and count
            if n_targets == 1:
                value = matching["value"].iloc[0]
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


_GEO_LEVEL_NAMES = {0: "National", 1: "State", 2: "District"}


def drop_target_groups(
    targets_df: pd.DataFrame,
    X_sparse,
    target_groups: np.ndarray,
    group_info: List[str],
    drop_specs: List[Tuple[str, str]],
) -> Tuple[pd.DataFrame, "sparse.csr_matrix"]:
    """Drop target groups by (label_substring, geo_level_name).

    Args:
        targets_df: Target metadata from build_matrix.
        X_sparse: Sparse calibration matrix (n_targets x n_cols).
        target_groups: Group ID per row from create_target_groups.
        group_info: Group descriptions from create_target_groups.
        drop_specs: List of (label_substring, geo_level_name)
            tuples. geo_level_name is "National", "State", or
            "District". label_substring is matched case-insensitive
            against group descriptions.

    Returns:
        (filtered_targets_df, filtered_X_sparse)
    """
    geo_levels = targets_df["geographic_id"].apply(get_geo_level)
    name_to_level = {v: k for k, v in _GEO_LEVEL_NAMES.items()}
    drop_ids = set()

    for label_substr, geo_name in drop_specs:
        level = name_to_level[geo_name]
        matched = False
        for gid, info in enumerate(group_info):
            group_mask = target_groups == gid
            group_geo = geo_levels[group_mask]
            if not (group_geo == level).all():
                continue
            if label_substr.lower() in info.lower():
                drop_ids.add(gid)
                matched = True
        if not matched:
            print(f"  WARNING: no match for ({label_substr!r}, {geo_name!r})")

    keep_mask = ~np.isin(target_groups, list(drop_ids))

    print(f"Matrix before: {X_sparse.shape[0]} rows")
    for gid in sorted(drop_ids):
        n = (target_groups == gid).sum()
        print(f"  DROPPING {group_info[gid]} ({n} rows)")
    print()

    kept_ids = sorted(set(range(len(group_info))) - drop_ids)
    for gid in kept_ids:
        n = (target_groups == gid).sum()
        print(f"  KEEPING  {group_info[gid]} ({n} rows)")

    X_out = X_sparse[keep_mask, :]
    targets_out = targets_df[keep_mask].reset_index(drop=True)
    print(f"\nMatrix after: {X_out.shape[0]} rows")
    return targets_out, X_out


def get_all_cds_from_database(db_uri: str) -> List[str]:
    """
    Get ordered list of all CD GEOIDs from database.

    Args:
        db_uri: SQLAlchemy database URI (e.g., "sqlite:///path/to/db")

    Returns:
        List of CD GEOID strings ordered by value
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_uri)
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM stratum_constraints sc
    WHERE sc.constraint_variable = 'congressional_district_geoid'
    ORDER BY sc.value
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        return [row[0] for row in result]


def get_cd_index_mapping(db_uri: str = None):
    """
    Get the canonical CD GEOID to index mapping.

    Args:
        db_uri: SQLAlchemy database URI. If None, uses default db location.

    Returns:
        tuple: (cd_to_index dict, index_to_cd dict, cds_ordered list)
    """
    from sqlalchemy import create_engine, text
    from policyengine_us_data.storage import STORAGE_FOLDER

    if db_uri is None:
        db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
        db_uri = f"sqlite:///{db_path}"

    engine = create_engine(db_uri)
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM stratum_constraints sc
    WHERE sc.constraint_variable = 'congressional_district_geoid'
    ORDER BY sc.value
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        cds_ordered = [row[0] for row in result]

    cd_to_index = {cd: idx for idx, cd in enumerate(cds_ordered)}
    index_to_cd = {idx: cd for idx, cd in enumerate(cds_ordered)}
    return cd_to_index, index_to_cd, cds_ordered


def save_geo_labels(labels: List[str], path) -> None:
    """Save geo unit labels to JSON."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f)


def load_geo_labels(path) -> List[str]:
    """Load geo unit labels from JSON."""
    with open(path) as f:
        return json.load(f)
