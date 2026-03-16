"""
HDFStore serialization utilities.

Converts variable-centric data dicts (``{var: {period: array}}``) into
entity-level Pandas HDFStore files consumed by API v2 and
``extend_single_year_dataset()``.
"""

import warnings
from typing import Dict

import numpy as np
import pandas as pd

ENTITIES = [
    "person",
    "household",
    "tax_unit",
    "spm_unit",
    "family",
    "marital_unit",
]


def split_data_into_entity_dfs(
    data: Dict[str, dict],
    system,
    time_period: int,
) -> Dict[str, pd.DataFrame]:
    """Split the data dict into per-entity DataFrames.

    Args:
        data: Maps variable names to ``{period: array}`` dicts.
        system: A PolicyEngine tax-benefit system.
        time_period: Year to extract from each variable's period dict.

    Returns:
        One DataFrame per entity, keyed by entity name.
        Group entities are deduplicated by their ID column.
    """
    entity_vars: Dict[str, list] = {e: [] for e in ENTITIES}

    for var_name in sorted(data.keys()):
        if var_name in system.variables:
            ek = system.variables[var_name].entity.key
            if ek in entity_vars:
                entity_vars[ek].append(var_name)
        else:
            entity_vars["household"].append(var_name)

    entity_dfs: Dict[str, pd.DataFrame] = {}
    for entity in ENTITIES:
        id_col = f"{entity}_id"
        cols = {}
        for var_name in entity_vars[entity]:
            periods = data[var_name]
            tp_key = time_period if time_period in periods else str(time_period)
            if tp_key not in periods:
                continue
            arr = periods[tp_key]
            if hasattr(arr, "dtype") and arr.dtype.kind == "S":
                arr = np.char.decode(arr, "utf-8")
            cols[var_name] = arr

        if entity == "person":
            for ref_entity in ENTITIES[1:]:
                ref_col = f"person_{ref_entity}_id"
                if ref_col in data:
                    periods = data[ref_col]
                    tp_key = time_period if time_period in periods else str(time_period)
                    if tp_key in periods:
                        cols[ref_col] = periods[tp_key]

        if not cols:
            continue

        df = pd.DataFrame(cols)
        if entity != "person" and id_col in df.columns:
            df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
        entity_dfs[entity] = df

    return entity_dfs


def build_uprating_manifest(
    data: Dict[str, dict],
    system,
) -> pd.DataFrame:
    """Build manifest of variable metadata for embedding in HDFStore.

    Args:
        data: Maps variable names to ``{period: array}`` dicts.
        system: A PolicyEngine tax-benefit system.

    Returns:
        DataFrame with columns: variable, entity, uprating.
    """
    records = []
    for var_name in sorted(data.keys()):
        entity = (
            system.variables[var_name].entity.key
            if var_name in system.variables
            else "unknown"
        )
        uprating = ""
        if var_name in system.variables:
            uprating = getattr(system.variables[var_name], "uprating", None) or ""
        records.append(
            {
                "variable": var_name,
                "entity": entity,
                "uprating": uprating,
            }
        )
    return pd.DataFrame(records)


def save_hdfstore(
    entity_dfs: Dict[str, pd.DataFrame],
    manifest_df: pd.DataFrame,
    output_path: str,
    time_period: int,
) -> str:
    """Save entity DataFrames and manifest to a Pandas HDFStore file.

    Args:
        entity_dfs: One DataFrame per entity from
            :func:`split_data_into_entity_dfs`.
        manifest_df: Variable metadata from
            :func:`build_uprating_manifest`.
        output_path: Path to the base ``.h5`` file.  The HDFStore is
            written alongside it with a ``.hdfstore.h5`` suffix.
        time_period: Year stored as metadata inside the HDFStore.

    Returns:
        Path to the created HDFStore file.
    """
    hdfstore_path = str(output_path).replace(".h5", ".hdfstore.h5")

    print(f"\nSaving HDFStore to {hdfstore_path}...")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=pd.errors.PerformanceWarning,
            message=".*PyTables will pickle object types.*",
        )
        with pd.HDFStore(hdfstore_path, mode="w") as store:
            for entity_name, df in entity_dfs.items():
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str)
                store.put(entity_name, df, format="table")

            store.put("_variable_metadata", manifest_df, format="table")
            store.put(
                "_time_period",
                pd.Series([time_period]),
                format="table",
            )

    for entity_name, df in entity_dfs.items():
        print(f"  {entity_name}: {len(df):,} rows, {len(df.columns)} cols")
    print(f"  manifest: {len(manifest_df)} variables")
    print("HDFStore saved successfully!")

    return hdfstore_path
