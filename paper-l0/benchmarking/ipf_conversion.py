from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from benchmark_manifest import BenchmarkManifest
from policyengine_us_data.calibration.calibration_utils import apply_op


_GEO_VARS = {"state_fips", "congressional_district_geoid"}
_SUPPORTED_TARGET_VARIABLES = {"person_count", "household_count"}


def _detect_time_period(sim) -> int:
    raw_keys = sim.dataset.load_dataset()["household_id"]
    try:
        return int(next(iter(raw_keys)))
    except Exception:
        return 2024


def _load_stratum_constraints(
    db_path: str | Path,
    stratum_ids: Iterable[int],
) -> Dict[int, List[dict]]:
    ids = sorted({int(sid) for sid in stratum_ids})
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    query = f"""
        SELECT
            stratum_id,
            constraint_variable AS variable,
            operation,
            value
        FROM stratum_constraints
        WHERE stratum_id IN ({placeholders})
        ORDER BY stratum_id
    """
    with sqlite3.connect(str(db_path)) as conn:
        rows = pd.read_sql_query(query, conn, params=ids)
    grouped: Dict[int, List[dict]] = {}
    for stratum_id, group in rows.groupby("stratum_id", sort=False):
        grouped[int(stratum_id)] = group[["variable", "operation", "value"]].to_dict(
            "records"
        )
    return grouped


def _ensure_supported_targets(targets_df: pd.DataFrame) -> None:
    unsupported = sorted(
        set(targets_df["variable"].astype(str)) - _SUPPORTED_TARGET_VARIABLES
    )
    if unsupported:
        raise ValueError(
            "Automatic IPF conversion currently supports only "
            f"{sorted(_SUPPORTED_TARGET_VARIABLES)} targets. "
            f"Unsupported target variables in manifest selection: {unsupported}"
        )


def _required_constraint_variables(
    stratum_constraints: Dict[int, List[dict]],
) -> List[str]:
    variables = set()
    for constraints in stratum_constraints.values():
        for constraint in constraints:
            variable = str(constraint["variable"])
            if variable not in _GEO_VARS:
                variables.add(variable)
    return sorted(variables)


def _evaluate_constraints(
    constraints: List[dict],
    columns: Dict[str, np.ndarray],
) -> np.ndarray:
    n_rows = len(next(iter(columns.values())))
    mask = np.ones(n_rows, dtype=bool)
    for constraint in constraints:
        variable = str(constraint["variable"])
        if variable not in columns:
            raise KeyError(f"Missing column for constraint variable: {variable}")
        mask &= apply_op(
            np.asarray(columns[variable]),
            str(constraint["operation"]),
            str(constraint["value"]),
        )
    return mask


def _build_household_clone_arrays(
    package: Dict,
    sim,
) -> Tuple[pd.DataFrame, np.ndarray]:
    household_ids = sim.calculate("household_id", map_to="household").values
    n_households = len(household_ids)
    n_clones = int(package["metadata"]["n_clones"])
    expected_units = n_households * n_clones
    initial_weights = np.asarray(package["initial_weights"], dtype=np.float64)
    if len(initial_weights) != expected_units:
        raise ValueError(
            "Initial weight length does not match dataset households x n_clones: "
            f"{len(initial_weights)} != {n_households} * {n_clones}"
        )

    if package.get("cd_geoid") is None or package.get("block_geoid") is None:
        raise ValueError(
            "Automatic IPF conversion requires cd_geoid and block_geoid in the package"
        )

    unit_index = np.arange(expected_units, dtype=np.int64)
    block_geoid = np.asarray(package["block_geoid"]).astype(str)
    cd_geoid = np.asarray(package["cd_geoid"]).astype(str)
    if len(block_geoid) != expected_units or len(cd_geoid) != expected_units:
        raise ValueError("Geography arrays do not match expected cloned unit count")

    household_df = pd.DataFrame(
        {
            "unit_index": unit_index,
            "household_id": unit_index,
            "base_weight": initial_weights,
            "benchmark_all": "all",
            "state_fips": block_geoid,
            "congressional_district_geoid": cd_geoid,
        }
    )
    household_df["state_fips"] = household_df["state_fips"].str.slice(0, 2).astype(int)
    household_df["congressional_district_geoid"] = household_df[
        "congressional_district_geoid"
    ].astype(int)
    return household_df, household_ids


def _load_sim_columns(
    sim,
    variables: List[str],
    level: str,
) -> Dict[str, np.ndarray]:
    columns: Dict[str, np.ndarray] = {}
    for variable in variables:
        try:
            values = sim.calculate(variable, map_to=level).values
        except Exception as exc:
            raise RuntimeError(
                f"Failed to calculate benchmark variable '{variable}' at level '{level}'"
            ) from exc
        values = np.asarray(values)
        if hasattr(values, "decode_to_str"):
            values = values.decode_to_str()
        if values.dtype.kind == "S":
            values = values.astype(str)
        columns[variable] = values
    return columns


def _build_person_level_unit_data(
    package: Dict,
    household_df: pd.DataFrame,
    sim,
    needed_variables: List[str],
) -> pd.DataFrame:
    household_ids = sim.calculate("household_id", map_to="household").values
    person_hh_ids = sim.calculate("household_id", map_to="person").values
    hh_index = {int(hid): idx for idx, hid in enumerate(household_ids)}
    person_hh_index = np.array(
        [hh_index[int(hid)] for hid in person_hh_ids], dtype=np.int64
    )

    n_households = len(household_ids)
    n_clones = int(package["metadata"]["n_clones"])

    person_columns = _load_sim_columns(sim, needed_variables, level="person")
    person_frames = []
    for clone_idx in range(n_clones):
        unit_index = person_hh_index + clone_idx * n_households
        frame = pd.DataFrame(
            {
                "unit_index": unit_index,
                "household_id": unit_index,
                "base_weight": household_df["base_weight"].to_numpy()[unit_index],
                "benchmark_all": household_df["benchmark_all"].to_numpy()[unit_index],
                "state_fips": household_df["state_fips"].to_numpy()[unit_index],
                "congressional_district_geoid": household_df[
                    "congressional_district_geoid"
                ].to_numpy()[unit_index],
            }
        )
        for variable, values in person_columns.items():
            frame[variable] = values
        person_frames.append(frame)
    return pd.concat(person_frames, ignore_index=True)


def _build_household_level_unit_data(
    household_df: pd.DataFrame,
    sim,
    needed_variables: List[str],
) -> pd.DataFrame:
    frame = household_df.copy()
    household_columns = _load_sim_columns(sim, needed_variables, level="household")
    repeated_columns = {
        name: np.tile(values, len(frame) // len(values))
        for name, values in household_columns.items()
    }
    for name, values in repeated_columns.items():
        frame[name] = values
    return frame


def _target_scope(target_variable: str) -> str:
    if target_variable == "person_count":
        return "person"
    if target_variable == "household_count":
        return "household"
    raise ValueError(f"Unsupported IPF target variable: {target_variable}")


def build_ipf_inputs(
    package: Dict,
    manifest: BenchmarkManifest,
    filtered_targets: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_supported_targets(filtered_targets)

    metadata = package.get("metadata", {})
    dataset_path = metadata.get("dataset_path")
    db_path = metadata.get("db_path")
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(
            "Automatic IPF conversion requires metadata.dataset_path to exist locally"
        )
    if not db_path or not Path(db_path).exists():
        raise FileNotFoundError(
            "Automatic IPF conversion requires metadata.db_path to exist locally"
        )

    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=str(dataset_path))
    _ = _detect_time_period(sim)

    stratum_constraints = _load_stratum_constraints(
        db_path=db_path,
        stratum_ids=filtered_targets["stratum_id"].astype(int).tolist(),
    )
    needed_variables = _required_constraint_variables(stratum_constraints)
    has_person_targets = (
        filtered_targets["variable"].astype(str).eq("person_count").any()
    )

    household_df, _ = _build_household_clone_arrays(package, sim)
    if has_person_targets:
        unit_data = _build_person_level_unit_data(
            package=package,
            household_df=household_df,
            sim=sim,
            needed_variables=needed_variables,
        )
    else:
        unit_data = _build_household_level_unit_data(
            household_df=household_df,
            sim=sim,
            needed_variables=needed_variables,
        )

    eval_columns = {
        column: unit_data[column].to_numpy() for column in unit_data.columns
    }
    ipf_target_rows = []
    for row_idx, row in filtered_targets.reset_index(drop=True).iterrows():
        constraints = stratum_constraints.get(int(row["stratum_id"]), [])
        indicator_column = f"ipf_indicator_{row_idx:05d}"
        mask = _evaluate_constraints(constraints, eval_columns)
        unit_data[indicator_column] = mask.astype(np.int8)
        eval_columns[indicator_column] = unit_data[indicator_column].to_numpy()
        ipf_target_rows.append(
            {
                "scope": _target_scope(str(row["variable"])),
                "target_type": "numeric_total",
                "value_column": indicator_column,
                "variables": "benchmark_all",
                "cell": "benchmark_all=all",
                "target_value": float(row["value"]),
                "target_name": row.get("target_name", f"target_{row_idx}"),
                "source_variable": str(row["variable"]),
                "stratum_id": int(row["stratum_id"]),
            }
        )

    return unit_data, pd.DataFrame(ipf_target_rows)
