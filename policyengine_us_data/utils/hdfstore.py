"""
Dataset serialization utilities.

Provides two serializers used by ``build_output_dataset``:

* ``save_h5``  – variable-centric h5py format
* ``save_hdfstore`` – entity-level Pandas HDFStore consumed by API v2
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict

import h5py
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


@dataclass
class DatasetResult:
    """Typed container returned by ``build_output_dataset``."""

    data: Dict[str, Dict]  # {var_name: {period: np.ndarray}}
    time_period: int
    system: Any  # TaxBenefitSystem


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _resolve_period_key(periods: dict, time_period: int):
    """Find the best matching key in a variable's period dict.

    Tries ``time_period`` (int), ``str(time_period)``, then falls back
    to the first available key (handles ETERNITY and Period objects).
    Returns ``None`` when *periods* is empty.
    """
    if time_period in periods:
        return time_period
    s = str(time_period)
    if s in periods:
        return s
    if periods:
        return next(iter(periods))
    return None


def _split_data_into_entity_dfs(
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
            tp_key = _resolve_period_key(periods, time_period)
            if tp_key is None:
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
                    tp_key = _resolve_period_key(periods, time_period)
                    if tp_key is not None:
                        cols[ref_col] = periods[tp_key]

        if not cols:
            continue

        df = pd.DataFrame(cols)
        if entity != "person" and id_col in df.columns:
            df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
        entity_dfs[entity] = df

    return entity_dfs


def _build_uprating_manifest(
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


# -------------------------------------------------------------------
# Serializers
# -------------------------------------------------------------------


def save_h5(result: DatasetResult, output_base: str) -> str:
    """Write variable-centric h5py file.

    Args:
        result: The assembled dataset.
        output_base: Path stem **without** file extension.

    Returns:
        Path to the created ``.h5`` file.
    """
    h5_path = str(output_base) + ".h5"
    with h5py.File(h5_path, "w") as f:
        for variable, periods in result.data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    print(f"\nH5 saved to {h5_path}")

    with h5py.File(h5_path, "r") as f:
        tp = str(result.time_period)
        if "household_id" in f and tp in f["household_id"]:
            n = len(f["household_id"][tp][:])
            print(f"Verified: {n:,} households in output")
        if "person_id" in f and tp in f["person_id"]:
            n = len(f["person_id"][tp][:])
            print(f"Verified: {n:,} persons in output")
        if "household_weight" in f and tp in f["household_weight"]:
            hw = f["household_weight"][tp][:]
            print(f"Total population (HH weights): {hw.sum():,.0f}")
        if "person_weight" in f and tp in f["person_weight"]:
            pw = f["person_weight"][tp][:]
            print(f"Total population (person weights): {pw.sum():,.0f}")

    return h5_path


def save_hdfstore(result: DatasetResult, output_base: str) -> str:
    """Write entity-level Pandas HDFStore file.

    Args:
        result: The assembled dataset.
        output_base: Path stem **without** file extension.

    Returns:
        Path to the created ``.hdfstore.h5`` file.
    """
    hdfstore_path = str(output_base) + ".hdfstore.h5"

    entity_dfs = _split_data_into_entity_dfs(
        result.data, result.system, result.time_period
    )
    manifest_df = _build_uprating_manifest(result.data, result.system)

    print(f"\nSaving HDFStore to {hdfstore_path}...")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=pd.errors.PerformanceWarning,
            message=".*PyTables will pickle object types.*",
        )
        with pd.HDFStore(hdfstore_path, mode="w") as store:
            for entity_name, df in entity_dfs.items():
                df = df.copy()
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str)
                store.put(entity_name, df, format="table")

            store.put("_variable_metadata", manifest_df, format="table")
            store.put(
                "_time_period",
                pd.Series([result.time_period]),
                format="table",
            )

    for entity_name, df in entity_dfs.items():
        print(f"  {entity_name}: {len(df):,} rows, {len(df.columns)} cols")
    print(f"  manifest: {len(manifest_df)} variables")
    print("HDFStore saved successfully!")

    return hdfstore_path
