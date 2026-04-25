"""IPF-benchmark input conversion.

Turns a filtered slice of the calibration package into the unit-table +
categorical-margin representation `surveysd::ipf` consumes.

The converter is intentionally stricter than the shared benchmark harness:
it keeps only count-style targets that can be represented as one coherent
closed categorical margin system. When a requested target family is a binary
subset and an authored parent total exists on the exact reduced key, the
missing complement is derived from that authored total. Otherwise the family is
dropped with explicit diagnostics instead of being run as an open 1-cell
margin or sequentialized externally.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from benchmark_manifest import BenchmarkManifest


# ---------------------------------------------------------------------------
# Geography constants
# ---------------------------------------------------------------------------

_GEO_VARS = {"state_fips", "congressional_district_geoid"}


# ---------------------------------------------------------------------------
# Value coercion helpers
# ---------------------------------------------------------------------------


def _coerce_value(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s in ("-inf", "-infinity"):
        return -math.inf
    if s in ("inf", "infinity", "+inf", "+infinity"):
        return math.inf
    return float(s)


def _equality_value(v) -> object:
    """Canonicalize an `==` constraint value to a hashable label."""
    try:
        f = _coerce_value(v)
        if math.isfinite(f) and f.is_integer():
            return int(f)
        return f
    except ValueError:
        return str(v)


# ---------------------------------------------------------------------------
# Bucket schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeBucket:
    label: str
    constraints: frozenset  # frozenset of (op, float_value) pairs


def _bucket(label: str, lo_op: str, lo_val, hi_op: str, hi_val) -> RangeBucket:
    return RangeBucket(
        label=label,
        constraints=frozenset(
            [(lo_op, _coerce_value(lo_val)), (hi_op, _coerce_value(hi_val))]
        ),
    )


AGE_BUCKETS: Tuple[RangeBucket, ...] = tuple(
    _bucket(label, ">", lo, "<", hi)
    for label, lo, hi in [
        ("0-4", -1, 5),
        ("5-9", 4, 10),
        ("10-14", 9, 15),
        ("15-19", 14, 20),
        ("20-24", 19, 25),
        ("25-29", 24, 30),
        ("30-34", 29, 35),
        ("35-39", 34, 40),
        ("40-44", 39, 45),
        ("45-49", 44, 50),
        ("50-54", 49, 55),
        ("55-59", 54, 60),
        ("60-64", 59, 65),
        ("65-69", 64, 70),
        ("70-74", 69, 75),
        ("75-79", 74, 80),
        ("80-84", 79, 85),
        # The target DB uses two different upper bounds for the 85+ bucket
        # (`<999` and `<1000`). Both map to the same label so either encoding
        # matches cleanly.
        ("85+", 84, 999),
        ("85+", 84, 1000),
    ]
)


# `eitc_child_count` uses discrete values {0, 1, 2} plus an open-upper ">2"
# bucket. Declared explicitly so the matcher routes every authored target into
# the same derived column `eitc_child_count_bracket`.
EITC_CHILD_COUNT_BUCKETS: Tuple[RangeBucket, ...] = (
    RangeBucket("0", frozenset([("==", 0.0)])),
    RangeBucket("1", frozenset([("==", 1.0)])),
    RangeBucket("2", frozenset([("==", 2.0)])),
    RangeBucket(">2", frozenset([(">", 2.0)])),
)


AGI_BUCKETS_DISTRICT: Tuple[RangeBucket, ...] = tuple(
    _bucket(label, ">=", lo, "<", hi)
    for label, lo, hi in [
        ("(-inf,1)", -math.inf, 1.0),
        ("[1,10k)", 1.0, 10_000.0),
        ("[10k,25k)", 10_000.0, 25_000.0),
        ("[25k,50k)", 25_000.0, 50_000.0),
        ("[50k,75k)", 50_000.0, 75_000.0),
        ("[75k,100k)", 75_000.0, 100_000.0),
        ("[100k,200k)", 100_000.0, 200_000.0),
        ("[200k,500k)", 200_000.0, 500_000.0),
        ("[500k,inf)", 500_000.0, math.inf),
    ]
)

AGI_BUCKETS_STATE: Tuple[RangeBucket, ...] = AGI_BUCKETS_DISTRICT[:-1] + tuple(
    _bucket(label, ">=", lo, "<", hi)
    for label, lo, hi in [
        ("[500k,1M)", 500_000.0, 1_000_000.0),
        ("[1M,inf)", 1_000_000.0, math.inf),
    ]
)


@dataclass(frozen=True)
class BucketSchema:
    name: str  # derived-column name to emit on the unit table
    variable: str  # source variable the schema bucketizes
    buckets: Tuple[RangeBucket, ...]
    applies_to_geo_levels: Tuple[str, ...]

    def match_constraints(self, pairs: frozenset) -> Optional[str]:
        for b in self.buckets:
            if b.constraints == pairs:
                return b.label
        return None


AGE_SCHEMA = BucketSchema(
    name="age_bracket",
    variable="age",
    buckets=AGE_BUCKETS,
    applies_to_geo_levels=("national", "state", "district"),
)

AGI_DISTRICT_SCHEMA = BucketSchema(
    name="agi_bracket_district",
    variable="adjusted_gross_income",
    buckets=AGI_BUCKETS_DISTRICT,
    applies_to_geo_levels=("district",),
)

AGI_STATE_SCHEMA = BucketSchema(
    name="agi_bracket_state",
    variable="adjusted_gross_income",
    buckets=AGI_BUCKETS_STATE,
    applies_to_geo_levels=("state", "national"),
)

EITC_CHILD_COUNT_SCHEMA = BucketSchema(
    name="eitc_child_count_bracket",
    variable="eitc_child_count",
    buckets=EITC_CHILD_COUNT_BUCKETS,
    applies_to_geo_levels=("national", "state", "district"),
)

RANGE_SCHEMAS_BY_VARIABLE: Dict[str, Tuple[BucketSchema, ...]] = {
    "age": (AGE_SCHEMA,),
    "adjusted_gross_income": (AGI_DISTRICT_SCHEMA, AGI_STATE_SCHEMA),
    "eitc_child_count": (EITC_CHILD_COUNT_SCHEMA,),
}


# Cell label used for the synthetic positive/non-positive binary bucket.
POSITIVE_LABEL = "positive"
NON_POSITIVE_LABEL = "non_positive"


class IPFConversionError(ValueError):
    """Raised when the requested target slice cannot form one valid IPF run."""

    def __init__(self, message: str, diagnostics: Optional[Dict[str, object]] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _positive_column_name(variable: str) -> str:
    return f"{variable}_positive"


# ---------------------------------------------------------------------------
# Target normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeoKey:
    variable: Optional[str]
    value: Optional[object]

    def level(self) -> str:
        if self.variable == "state_fips":
            return "state"
        if self.variable == "congressional_district_geoid":
            return "district"
        return "national"


@dataclass(frozen=True)
class ResolvedTarget:
    target_id: int
    target_name: str
    target_value: float
    scope: str  # "person" | "household" | "tax_unit" | ...
    source_variable: str
    geo: GeoKey
    cell: Tuple[Tuple[str, str], ...]  # tuple of (derived_column, cell_label), sorted
    raw_constraints: Tuple[Tuple[str, str, str], ...]  # preserved for diagnostics


def _group_constraints_by_variable(
    records: Iterable[dict],
    exclude_vars: Iterable[str],
) -> Dict[str, frozenset]:
    exclude = set(exclude_vars)
    grouped: Dict[str, List[Tuple[str, object]]] = {}
    for r in records:
        var = str(r["variable"])
        if var in exclude:
            continue
        op = str(r["operation"])
        raw_val = r["value"]
        if op == "==":
            value: object = _equality_value(raw_val)
        else:
            value = _coerce_value(raw_val)
        grouped.setdefault(var, []).append((op, value))
    return {v: frozenset(pairs) for v, pairs in grouped.items()}


def _resolve_variable_cell(
    variable: str,
    pairs: frozenset,
    geo_level: str,
) -> Optional[Tuple[str, str]]:
    """Return (derived_column_name, cell_label) for this variable's constraints."""
    ops = {op for op, _ in pairs}

    # Declared range or discrete schemas take precedence: they pull equality
    # cases into the same derived column as the range cases (e.g. EITC).
    for schema in RANGE_SCHEMAS_BY_VARIABLE.get(variable, ()):
        if geo_level not in schema.applies_to_geo_levels:
            continue
        lbl = schema.match_constraints(pairs)
        if lbl is not None:
            return (schema.name, lbl)

    # Pure equality → raw categorical pass-through (value identifies the cell).
    if ops == {"=="}:
        if len(pairs) != 1:
            return None
        _, val = next(iter(pairs))
        return (variable, str(val))

    # Positive-dollar `>`-only single constraint at zero.
    if ops == {">"} and len(pairs) == 1:
        _, val = next(iter(pairs))
        if isinstance(val, float) and val == 0.0:
            return (_positive_column_name(variable), POSITIVE_LABEL)

    return None


def _resolve_target(
    target_row: pd.Series,
    constraints: List[dict],
) -> Optional[ResolvedTarget]:
    geo = _extract_geo(constraints)
    grouped = _group_constraints_by_variable(constraints, exclude_vars=_GEO_VARS)
    cell: List[Tuple[str, str]] = []
    for var, pairs in grouped.items():
        label = _resolve_variable_cell(var, pairs, geo.level())
        if label is None:
            return None
        cell.append(label)
    raw = tuple(
        sorted(
            (str(r["variable"]), str(r["operation"]), str(r["value"]))
            for r in constraints
        )
    )
    return ResolvedTarget(
        target_id=int(target_row["target_id"])
        if "target_id" in target_row.index
        else -1,
        target_name=str(
            target_row.get("target_name")
            or f"target_{target_row.get('stratum_id', 'unknown')}"
        ),
        target_value=float(target_row["value"]),
        scope=_target_scope(str(target_row["variable"])),
        source_variable=str(target_row["variable"]),
        geo=geo,
        cell=tuple(sorted(cell)),
        raw_constraints=raw,
    )


def _extract_geo(records: Iterable[dict]) -> GeoKey:
    for r in records:
        v = str(r["variable"])
        if v in _GEO_VARS:
            raw = r["value"]
            try:
                val: object = int(raw)
            except (ValueError, TypeError):
                val = str(raw)
            return GeoKey(variable=v, value=val)
    return GeoKey(variable=None, value=None)


_SCOPE_BY_VARIABLE = {
    "person_count": "person",
    "household_count": "household",
}


def _target_scope(target_variable: str) -> str:
    try:
        return _SCOPE_BY_VARIABLE[target_variable]
    except KeyError as exc:
        raise ValueError(
            f"IPF conversion does not support target variable "
            f"'{target_variable}'. Currently supported: "
            f"{sorted(_SCOPE_BY_VARIABLE)}. "
            "`tax_unit_count` and `spm_unit_count` remain outside the core "
            "household/person IPF path in this pass."
        ) from exc


# ---------------------------------------------------------------------------
# Margin assembly
# ---------------------------------------------------------------------------


@dataclass
class MarginBlock:
    margin_id: str
    scope: str
    source_variable: str
    cell_dims: Tuple[str, ...]  # sorted non-geo derived-column names
    cell_vars: Tuple[str, ...]  # sorted derived-column names used as cell dimensions
    geo_var: Optional[str]  # geo dimension included in the margin
    targets: List[ResolvedTarget]
    # cells that exist in the target package ( (geo_value, cell_tuple) )
    target_cells: set


@dataclass(frozen=True)
class MarginCell:
    geo_value: Optional[object]
    cell: Tuple[Tuple[str, str], ...]
    target_value: float
    target_name: str
    is_authored: bool
    authored_target_id: Optional[int]
    source_target_ids: Tuple[int, ...]
    derivation_reason: Optional[str] = None


@dataclass
class ClosedMarginBlock:
    margin_id: str
    scope: str
    source_variable: str
    cell_dims: Tuple[str, ...]
    cell_vars: Tuple[str, ...]
    geo_var: Optional[str]
    closure_status: str
    cells: List[MarginCell]


def _margin_key(t: ResolvedTarget) -> Tuple[str, Optional[str], Tuple[str, ...]]:
    return (
        t.source_variable,
        t.geo.variable,
        tuple(sorted({dc for dc, _ in t.cell})),
    )


def _assemble_margins(
    resolved: List[ResolvedTarget],
) -> List[MarginBlock]:
    blocks: Dict[Tuple, MarginBlock] = {}
    for t in resolved:
        key = _margin_key(t)
        if key not in blocks:
            source_variable, geo_var, cell_vars = key
            all_vars = (*cell_vars,) if geo_var is None else (geo_var, *cell_vars)
            blocks[key] = MarginBlock(
                margin_id=f"margin_{len(blocks):04d}",
                scope=t.scope,
                source_variable=source_variable,
                cell_dims=cell_vars,
                cell_vars=tuple(sorted(all_vars)),
                geo_var=geo_var,
                targets=[],
                target_cells=set(),
            )
        blocks[key].targets.append(t)
        blocks[key].target_cells.add((t.geo.value, t.cell))
    return list(blocks.values())


def _is_single_cell_per_geo(block: MarginBlock) -> bool:
    """True iff the block carries exactly one cell per geography.

    surveysd handles these as 1-cell `xtabs` arrays: the authored cell is
    raked to its target and units outside that cell are left unconstrained
    by the margin. Kept for diagnostics and reporting — no longer drives
    emission branching.
    """
    if block.geo_var is None:
        return len({cell for _, cell in block.target_cells}) == 1
    cells_per_geo: Dict[object, set] = {}
    for geo_value, cell in block.target_cells:
        cells_per_geo.setdefault(geo_value, set()).add(cell)
    return all(len(cells) == 1 for cells in cells_per_geo.values()) and (
        len({cell for _, cell in block.target_cells}) == 1
    )


def check_margin_consistency(
    blocks: List[object],
    tolerance: float = 1e-3,
) -> List[Dict[str, object]]:
    """Return per-geo population-total disagreements across closed blocks."""
    totals_by_scope_geo: Dict[Tuple[str, Optional[str], object], Dict[str, float]] = {}
    for block in blocks:
        by_geo: Dict[object, float] = {}
        if hasattr(block, "cells"):
            for cell in block.cells:
                by_geo[cell.geo_value] = by_geo.get(cell.geo_value, 0.0) + float(
                    cell.target_value
                )
            scope = block.scope
            geo_var = block.geo_var
        else:
            for t in block.targets:
                by_geo[t.geo.value] = by_geo.get(t.geo.value, 0.0) + float(
                    t.target_value
                )
            scope = block.scope
            geo_var = block.geo_var
        for geo_val, total in by_geo.items():
            key = (scope, geo_var, geo_val)
            totals_by_scope_geo.setdefault(key, {})[block.margin_id] = total

    issues: List[Dict[str, object]] = []
    for (scope, geo_var, geo_val), margin_totals in totals_by_scope_geo.items():
        if len(margin_totals) < 2:
            continue
        tmax = max(margin_totals.values())
        tmin = min(margin_totals.values())
        if tmax == 0 or (tmax - tmin) / tmax <= tolerance:
            continue
        issues.append(
            {
                "scope": scope,
                "geo_var": geo_var,
                "geo_value": geo_val,
                "margin_totals": dict(margin_totals),
                "relative_spread": (tmax - tmin) / tmax,
            }
        )
    return issues


# ---------------------------------------------------------------------------
# Derived-column materialization on the unit table
# ---------------------------------------------------------------------------


def _bracket_age(age_values: np.ndarray) -> np.ndarray:
    out = np.array(["unmatched"] * len(age_values), dtype=object)
    for b in AGE_BUCKETS:
        lo = next(v for op, v in b.constraints if op == ">")
        hi = next(v for op, v in b.constraints if op == "<")
        mask = (age_values > lo) & (age_values < hi)
        out[mask] = b.label
    return out


def _bracket_agi(agi_values: np.ndarray, schema: BucketSchema) -> np.ndarray:
    out = np.array(["unmatched"] * len(agi_values), dtype=object)
    for b in schema.buckets:
        lo = next(v for op, v in b.constraints if op == ">=")
        hi = next(v for op, v in b.constraints if op == "<")
        mask = (agi_values >= lo) & (agi_values < hi)
        out[mask] = b.label
    return out


def _bracket_eitc_child_count(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float).round().astype(int)
    out = np.where(v == 0, "0", np.where(v == 1, "1", np.where(v == 2, "2", ">2")))
    return out.astype(object)


def _materialize_derived_columns(
    unit_data: pd.DataFrame,
    derived_columns_needed: set,
    raw_variable_values: Dict[str, np.ndarray],
) -> pd.DataFrame:
    df = unit_data.copy()
    for col in derived_columns_needed:
        if col in df.columns:
            continue
        if col == AGE_SCHEMA.name:
            df[col] = _bracket_age(np.asarray(raw_variable_values["age"], dtype=float))
        elif col == AGI_DISTRICT_SCHEMA.name:
            df[col] = _bracket_agi(
                np.asarray(raw_variable_values["adjusted_gross_income"], dtype=float),
                AGI_DISTRICT_SCHEMA,
            )
        elif col == AGI_STATE_SCHEMA.name:
            df[col] = _bracket_agi(
                np.asarray(raw_variable_values["adjusted_gross_income"], dtype=float),
                AGI_STATE_SCHEMA,
            )
        elif col == EITC_CHILD_COUNT_SCHEMA.name:
            df[col] = _bracket_eitc_child_count(
                np.asarray(raw_variable_values["eitc_child_count"], dtype=float)
            )
        elif col.endswith("_positive"):
            src = col[: -len("_positive")]
            if src not in raw_variable_values:
                raise KeyError(
                    f"Cannot build derived column '{col}' — raw values for '{src}' not loaded"
                )
            df[col] = np.where(
                np.asarray(raw_variable_values[src], dtype=float) > 0.0,
                POSITIVE_LABEL,
                NON_POSITIVE_LABEL,
            ).astype(object)
        else:
            # Equality pass-through: derived column == source column, cast to str.
            if col not in raw_variable_values:
                raise KeyError(
                    f"Cannot build derived column '{col}' — raw values for '{col}' not loaded"
                )
            df[col] = np.asarray(raw_variable_values[col]).astype(str)
    return df


def _collect_required_source_variables(blocks: List[MarginBlock]) -> set:
    """Return the set of raw policyengine-us variables we need loaded."""
    needed = set()
    for block in blocks:
        for cell_var in block.cell_vars:
            if cell_var == AGE_SCHEMA.name:
                needed.add("age")
            elif cell_var in (AGI_DISTRICT_SCHEMA.name, AGI_STATE_SCHEMA.name):
                needed.add("adjusted_gross_income")
            elif cell_var == EITC_CHILD_COUNT_SCHEMA.name:
                needed.add("eitc_child_count")
            elif cell_var.endswith("_positive"):
                needed.add(cell_var[: -len("_positive")])
            elif cell_var in _GEO_VARS:
                continue
            else:
                # Equality pass-through uses the variable name directly.
                needed.add(cell_var)
    return needed


# ---------------------------------------------------------------------------
# Clone-unit table construction (carried over from the prior implementation)
# ---------------------------------------------------------------------------


def _detect_time_period(sim) -> int:
    raw_keys = sim.dataset.load_dataset()["household_id"]
    try:
        return int(next(iter(raw_keys)))
    except Exception:
        return 2024


def _load_stratum_constraints(
    db_path: str | Path, stratum_ids: Iterable[int]
) -> Dict[int, List[dict]]:
    ids = sorted({int(sid) for sid in stratum_ids})
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    query = f"""
        SELECT stratum_id, constraint_variable AS variable, operation, value
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


def _build_household_clone_arrays(
    package: Dict, sim
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
    sim, variables: Iterable[str], level: str
) -> Dict[str, np.ndarray]:
    columns: Dict[str, np.ndarray] = {}
    for variable in variables:
        values = sim.calculate(variable, map_to=level).values
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
    needed_variables: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
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
    stacked_raw: Dict[str, List[np.ndarray]] = {v: [] for v in person_columns}
    for clone_idx in range(n_clones):
        unit_index = person_hh_index + clone_idx * n_households
        frame = pd.DataFrame(
            {
                "unit_index": unit_index,
                "household_id": unit_index,
                "base_weight": household_df["base_weight"].to_numpy()[unit_index],
                "state_fips": household_df["state_fips"].to_numpy()[unit_index],
                "congressional_district_geoid": household_df[
                    "congressional_district_geoid"
                ].to_numpy()[unit_index],
            }
        )
        for variable, values in person_columns.items():
            frame[variable] = values
            stacked_raw[variable].append(values)
        person_frames.append(frame)
    raw_values = {v: np.concatenate(a) for v, a in stacked_raw.items()}
    return pd.concat(person_frames, ignore_index=True), raw_values


def _build_household_level_unit_data(
    household_df: pd.DataFrame,
    sim,
    needed_variables: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    frame = household_df.copy()
    household_columns = _load_sim_columns(sim, needed_variables, level="household")
    repeated = {
        name: np.tile(values, len(frame) // max(len(values), 1))
        for name, values in household_columns.items()
    }
    for name, values in repeated.items():
        frame[name] = values
    return frame, repeated


# ---------------------------------------------------------------------------
# Closed-system validation and exact closure
# ---------------------------------------------------------------------------


def _targeted_unit_slice(unit_data: pd.DataFrame, block: MarginBlock) -> pd.DataFrame:
    if block.geo_var is None:
        return unit_data
    geo_values = {geo_value for geo_value, _ in block.target_cells}
    return unit_data.loc[unit_data[block.geo_var].isin(list(geo_values))].copy()


def _row_to_cell(
    row: pd.Series | dict,
    *,
    geo_var: Optional[str],
    cell_dims: Tuple[str, ...],
) -> Tuple[Optional[object], Tuple[Tuple[str, str], ...]]:
    geo_value = row[geo_var] if geo_var is not None else None
    cell = tuple(sorted((col, str(row[col])) for col in cell_dims))
    return geo_value, cell


def _observed_cells_for_block(
    unit_data: pd.DataFrame, block: MarginBlock
) -> set[Tuple[Optional[object], Tuple[Tuple[str, str], ...]]]:
    targeted = _targeted_unit_slice(unit_data, block)
    if targeted.empty:
        return set()
    columns = [*block.cell_dims]
    if block.geo_var is not None:
        columns = [block.geo_var, *columns]
    observed = targeted[columns].drop_duplicates()
    return {
        _row_to_cell(row._asdict(), geo_var=block.geo_var, cell_dims=block.cell_dims)
        for row in observed.itertuples(index=False)
    }


def _parent_key_for_cell(
    geo_value: Optional[object],
    cell: Tuple[Tuple[str, str], ...],
    subset_dim: str,
) -> Tuple[Optional[object], Tuple[Tuple[str, str], ...]]:
    return (
        geo_value,
        tuple(sorted((col, label) for col, label in cell if col != subset_dim)),
    )


def _group_authored_cells_by_parent(
    block: MarginBlock, subset_dim: str
) -> Dict[Tuple[Optional[object], Tuple[Tuple[str, str], ...]], List[ResolvedTarget]]:
    grouped: Dict[Tuple[Optional[object], Tuple[Tuple[str, str], ...]], List[ResolvedTarget]] = {}
    for target in block.targets:
        grouped.setdefault(
            _parent_key_for_cell(target.geo.value, target.cell, subset_dim), []
        ).append(target)
    return grouped


def _group_observed_labels_by_parent(
    unit_data: pd.DataFrame, block: MarginBlock, subset_dim: str
) -> Dict[Tuple[Optional[object], Tuple[Tuple[str, str], ...]], set[str]]:
    targeted = _targeted_unit_slice(unit_data, block)
    if targeted.empty:
        return {}
    columns = [subset_dim, *(dim for dim in block.cell_dims if dim != subset_dim)]
    if block.geo_var is not None:
        columns = [block.geo_var, *columns]
    observed = targeted[columns].drop_duplicates()
    grouped: Dict[Tuple[Optional[object], Tuple[Tuple[str, str], ...]], set[str]] = {}
    for row in observed.to_dict("records"):
        geo_value = row[block.geo_var] if block.geo_var is not None else None
        parent_assignments = tuple(
            sorted(
                (dim, str(row[dim])) for dim in block.cell_dims if dim != subset_dim
            )
        )
        grouped.setdefault((geo_value, parent_assignments), set()).add(
            str(row[subset_dim])
        )
    return grouped


def _build_parent_total_lookup(
    block: MarginBlock,
) -> Tuple[
    Dict[Tuple[Optional[object], Tuple[Tuple[str, str], ...]], ResolvedTarget],
    Tuple[int, ...],
]:
    lookup: Dict[Tuple[Optional[object], Tuple[Tuple[str, str], ...]], ResolvedTarget] = {}
    ambiguous_target_ids: set[int] = set()
    for target in block.targets:
        key = (target.geo.value, tuple(sorted(target.cell)))
        if key in lookup:
            ambiguous_target_ids.add(int(target.target_id))
            ambiguous_target_ids.add(int(lookup[key].target_id))
            continue
        lookup[key] = target
    return lookup, tuple(sorted(ambiguous_target_ids))


def _full_partition_cells(block: MarginBlock) -> List[MarginCell]:
    return [
        MarginCell(
            geo_value=target.geo.value,
            cell=target.cell,
            target_value=float(target.target_value),
            target_name=target.target_name,
            is_authored=True,
            authored_target_id=target.target_id,
            source_target_ids=(target.target_id,),
        )
        for target in block.targets
    ]


def _try_close_binary_subset(
    block: MarginBlock,
    blocks_by_key: Dict[Tuple[str, Optional[str], Tuple[str, ...]], MarginBlock],
    unit_data: pd.DataFrame,
    tolerance: float,
) -> Tuple[Optional[ClosedMarginBlock], Optional[Dict[str, object]]]:
    missing_parent_reason: Optional[Dict[str, object]] = None
    ambiguous_parent_reason: Optional[Dict[str, object]] = None
    for subset_dim in block.cell_dims:
        observed_by_parent = _group_observed_labels_by_parent(unit_data, block, subset_dim)
        if not observed_by_parent:
            return None, {
                "reason": "missing_unit_support",
                "margin_id": block.margin_id,
                "target_ids": [int(target.target_id) for target in block.targets],
            }

        global_labels = set().union(*observed_by_parent.values())
        if len(global_labels) > 2:
            continue

        parent_key = (
            block.source_variable,
            block.geo_var,
            tuple(dim for dim in block.cell_dims if dim != subset_dim),
        )
        parent_block = blocks_by_key.get(parent_key)
        if parent_block is None:
            missing_parent_reason = {
                "reason": "missing_parent_total",
                "margin_id": block.margin_id,
                "subset_dimension": subset_dim,
                "parent_margin_key": {
                    "source_variable": block.source_variable,
                    "geo_var": block.geo_var,
                    "cell_dims": [dim for dim in block.cell_dims if dim != subset_dim],
                },
                "target_ids": [int(target.target_id) for target in block.targets],
            }
            continue

        parent_totals, ambiguous_target_ids = _build_parent_total_lookup(parent_block)
        if ambiguous_target_ids:
            ambiguous_parent_reason = {
                "reason": "ambiguous_parent_total",
                "margin_id": block.margin_id,
                "subset_dimension": subset_dim,
                "parent_margin_id": parent_block.margin_id,
                "parent_target_ids": list(ambiguous_target_ids),
                "target_ids": [int(target.target_id) for target in block.targets],
            }
            continue
        authored_by_parent = _group_authored_cells_by_parent(block, subset_dim)
        emitted_cells: List[MarginCell] = []
        valid = True
        invalid_reason: Optional[Dict[str, object]] = None

        for parent_lookup_key, observed_labels in observed_by_parent.items():
            authored_targets = authored_by_parent.get(parent_lookup_key, [])
            if not authored_targets:
                continue

            parent_target = parent_totals.get(parent_lookup_key)
            if parent_target is None:
                valid = False
                invalid_reason = {
                    "reason": "missing_parent_total",
                    "margin_id": block.margin_id,
                    "parent_margin_key": {
                        "source_variable": block.source_variable,
                        "geo_var": block.geo_var,
                        "cell_dims": [dim for dim in block.cell_dims if dim != subset_dim],
                    },
                    "target_ids": [int(target.target_id) for target in block.targets],
                }
                break

            authored_labels = {
                dict(target.cell)[subset_dim] for target in authored_targets
            }
            if len(observed_labels) > 2 or len(observed_labels - authored_labels) > 1:
                valid = False
                invalid_reason = {
                    "reason": "unsupported_partial_margin",
                    "margin_id": block.margin_id,
                    "subset_dimension": subset_dim,
                    "target_ids": [int(target.target_id) for target in block.targets],
                }
                break

            authored_sum = float(
                sum(float(target.target_value) for target in authored_targets)
            )
            complement_value = float(parent_target.target_value) - authored_sum
            if complement_value < -tolerance:
                valid = False
                invalid_reason = {
                    "reason": "negative_derived_complement",
                    "margin_id": block.margin_id,
                    "subset_dimension": subset_dim,
                    "parent_target_id": int(parent_target.target_id),
                    "target_ids": [int(target.target_id) for target in block.targets],
                    "derived_value": complement_value,
                }
                break

            for target in authored_targets:
                emitted_cells.append(
                    MarginCell(
                        geo_value=target.geo.value,
                        cell=target.cell,
                        target_value=float(target.target_value),
                        target_name=target.target_name,
                        is_authored=True,
                        authored_target_id=target.target_id,
                        source_target_ids=(target.target_id,),
                    )
                )

            missing_labels = observed_labels - authored_labels
            if len(missing_labels) == 1:
                missing_label = next(iter(missing_labels))
                if complement_value > tolerance and missing_label not in observed_labels:
                    valid = False
                    invalid_reason = {
                        "reason": "missing_unit_support",
                        "margin_id": block.margin_id,
                        "subset_dimension": subset_dim,
                        "target_ids": [int(target.target_id) for target in block.targets],
                    }
                    break
                parent_geo_value, parent_assignments = parent_lookup_key
                derived_cell = tuple(
                    sorted((*parent_assignments, (subset_dim, missing_label)))
                )
                emitted_cells.append(
                    MarginCell(
                        geo_value=parent_geo_value,
                        cell=derived_cell,
                        target_value=max(complement_value, 0.0),
                        target_name=(
                            f"derived::{block.margin_id}::{subset_dim}={missing_label}"
                        ),
                        is_authored=False,
                        authored_target_id=None,
                        source_target_ids=tuple(
                            sorted(
                                {
                                    int(parent_target.target_id),
                                    *[int(target.target_id) for target in authored_targets],
                                }
                            )
                        ),
                        derivation_reason="authored_parent_total",
                    )
                )

        if not valid:
            return None, invalid_reason

        unique_cells = {
            (cell.geo_value, cell.cell, cell.is_authored): cell for cell in emitted_cells
        }
        return (
            ClosedMarginBlock(
                margin_id=block.margin_id,
                scope=block.scope,
                source_variable=block.source_variable,
                cell_dims=block.cell_dims,
                cell_vars=block.cell_vars,
                geo_var=block.geo_var,
                closure_status="binary_subset_with_parent_total",
                cells=list(unique_cells.values()),
            ),
            None,
        )

    if ambiguous_parent_reason is not None:
        return None, ambiguous_parent_reason
    if missing_parent_reason is not None:
        return None, missing_parent_reason
    return None, {
        "reason": "unsupported_partial_margin",
        "margin_id": block.margin_id,
        "target_ids": [int(target.target_id) for target in block.targets],
    }


def _validate_household_margin_invariance(
    unit_data: pd.DataFrame,
    blocks: List[MarginBlock],
) -> Tuple[List[MarginBlock], List[Dict[str, object]]]:
    valid_blocks: List[MarginBlock] = []
    dropped: List[Dict[str, object]] = []
    for block in blocks:
        if block.scope != "household":
            valid_blocks.append(block)
            continue
        varying_columns = []
        for column in block.cell_vars:
            if column not in unit_data.columns:
                continue
            nunique = unit_data.groupby("household_id", sort=False)[column].nunique(
                dropna=False
            )
            if (nunique > 1).any():
                varying_columns.append(column)
        if varying_columns:
            dropped.append(
                {
                    "reason": "non_invariant_household_constraint_variable",
                    "margin_id": block.margin_id,
                    "columns": varying_columns,
                    "target_ids": [int(target.target_id) for target in block.targets],
                }
            )
            continue
        valid_blocks.append(block)
    return valid_blocks, dropped


def _close_margin_blocks(
    blocks: List[MarginBlock],
    unit_data: pd.DataFrame,
    tolerance: float = 1e-9,
) -> Tuple[List[ClosedMarginBlock], List[Dict[str, object]]]:
    blocks_by_key = {
        (block.source_variable, block.geo_var, block.cell_dims): block for block in blocks
    }
    closed_blocks: List[ClosedMarginBlock] = []
    dropped: List[Dict[str, object]] = []

    for block in blocks:
        observed_cells = _observed_cells_for_block(unit_data, block)
        authored_cells = {
            (target.geo.value, tuple(sorted(target.cell))) for target in block.targets
        }
        unsupported_authored = [
            target
            for target in block.targets
            if (target.geo.value, tuple(sorted(target.cell))) not in observed_cells
            and abs(float(target.target_value)) > tolerance
        ]
        if unsupported_authored:
            dropped.append(
                {
                    "reason": "missing_unit_support",
                    "margin_id": block.margin_id,
                    "target_ids": [int(target.target_id) for target in unsupported_authored],
                }
            )
            continue

        if authored_cells == observed_cells:
            closed_blocks.append(
                ClosedMarginBlock(
                    margin_id=block.margin_id,
                    scope=block.scope,
                    source_variable=block.source_variable,
                    cell_dims=block.cell_dims,
                    cell_vars=block.cell_vars,
                    geo_var=block.geo_var,
                    closure_status="full_partition",
                    cells=_full_partition_cells(block),
                )
            )
            continue

        closed_block, reason = _try_close_binary_subset(
            block=block,
            blocks_by_key=blocks_by_key,
            unit_data=unit_data,
            tolerance=tolerance,
        )
        if closed_block is None:
            dropped.append(reason or {"reason": "unsupported_partial_margin"})
            continue
        closed_blocks.append(closed_block)

    return closed_blocks, dropped


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_ipf_inputs(
    package: Dict,
    manifest: BenchmarkManifest,
    filtered_targets: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(unit_metadata, ipf_target_metadata)` ready for `ipf_runner.R`.

    Consumes the same `filtered_targets` slice that the GREG and L0 runners
    see. Internally keeps only IPF-eligible count targets, resolves their
    categorical cells, closes only exact binary subset systems backed by
    authored parent totals, and rejects any remaining open or incoherent
    system rather than sequentializing it.
    """
    if filtered_targets.empty:
        raise ValueError("filtered_targets is empty; nothing to convert.")

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

    # --- Count check: drop non-count-style targets ------------------------
    supported_mask = (
        filtered_targets["variable"].astype(str).isin(_SCOPE_BY_VARIABLE.keys())
    )
    dropped_non_count = filtered_targets.loc[~supported_mask].copy()
    targets = filtered_targets.loc[supported_mask].reset_index(drop=True)
    if targets.empty:
        raise ValueError(
            "No count-style targets in filtered_targets; IPF has nothing to run. "
            f"Supported variables: {sorted(_SCOPE_BY_VARIABLE)}."
        )
    dropped_target_details: List[Dict[str, object]] = [
        {
            "reason": "non_count_style",
            "target_id": int(row.get("target_id", -1)),
            "target_name": str(row.get("target_name", "?")),
        }
        for _, row in dropped_non_count.iterrows()
    ]

    stratum_constraints = _load_stratum_constraints(
        db_path=db_path,
        stratum_ids=targets["stratum_id"].astype(int).tolist(),
    )

    # --- Resolver check: keep only targets that map to a declared cell ----
    resolved: List[ResolvedTarget] = []
    dropped_unresolvable: List[Tuple[int, str]] = []
    for _, row in targets.iterrows():
        constraints = stratum_constraints.get(int(row["stratum_id"]), [])
        rt = _resolve_target(row, constraints)
        if rt is None:
            dropped_unresolvable.append(
                (int(row.get("target_id", -1)), str(row.get("target_name", "?")))
            )
            continue
        resolved.append(rt)
    if not resolved:
        raise ValueError(
            "No targets in filtered_targets resolved through the declared "
            "bucket schemas. Nothing for IPF to run."
        )
    dropped_target_details.extend(
        {
            "reason": "unresolvable_constraints",
            "target_id": target_id,
            "target_name": target_name,
        }
        for target_id, target_name in dropped_unresolvable
    )

    blocks = _assemble_margins(resolved)

    # --- Build the cloned-unit table with needed source variables ----------
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=str(dataset_path))
    _ = _detect_time_period(sim)
    household_df, _ = _build_household_clone_arrays(package, sim)

    needed_source_vars = _collect_required_source_variables(blocks)

    has_person_scoped = any(b.scope == "person" for b in blocks)
    if has_person_scoped:
        unit_data, raw_values = _build_person_level_unit_data(
            package=package,
            household_df=household_df,
            sim=sim,
            needed_variables=needed_source_vars,
        )
    else:
        unit_data, raw_values = _build_household_level_unit_data(
            household_df=household_df,
            sim=sim,
            needed_variables=needed_source_vars,
        )

    # --- Materialize derived columns on the unit table ---------------------
    derived_cols = {cv for b in blocks for cv in b.cell_vars if cv not in _GEO_VARS}
    unit_data = _materialize_derived_columns(unit_data, derived_cols, raw_values)

    if has_person_scoped and any(block.scope == "household" for block in blocks):
        blocks, invariance_drops = _validate_household_margin_invariance(
            unit_data=unit_data,
            blocks=blocks,
        )
        dropped_target_details.extend(
            {
                **detail,
                "target_name": None,
            }
            for detail in invariance_drops
        )

    closed_blocks, closure_drops = _close_margin_blocks(blocks=blocks, unit_data=unit_data)
    dropped_target_details.extend(closure_drops)
    dropped_counts: Dict[str, int] = {}
    for detail in dropped_target_details:
        reason = str(detail.get("reason", "unknown"))
        count = len(detail.get("target_ids", [])) or 1
        dropped_counts[reason] = dropped_counts.get(reason, 0) + int(count)
    if not closed_blocks:
        raise IPFConversionError(
            "No closed categorical IPF margins remain after validation.",
            diagnostics={
                "requested_target_count": int(len(filtered_targets)),
                "retained_authored_target_count": 0,
                "derived_complement_count": 0,
                "dropped_targets": dropped_counts,
                "dropped_target_details": dropped_target_details,
                "margin_consistency_issues": [],
                "derived_complement_rows": [],
            },
        )

    issues = check_margin_consistency(closed_blocks)
    if issues:
        incompatible_details = []
        for issue in issues:
            margin_ids = list(issue.get("margin_totals", {}).keys())
            target_ids = sorted(
                {
                    int(cell.authored_target_id)
                    for block in closed_blocks
                    if block.margin_id in margin_ids
                    for cell in block.cells
                    if cell.is_authored and cell.authored_target_id is not None
                }
            )
            incompatible_details.append(
                {
                    "reason": "incompatible_totals",
                    "scope": issue.get("scope"),
                    "geo_var": issue.get("geo_var"),
                    "geo_value": issue.get("geo_value"),
                    "margin_ids": margin_ids,
                    "target_ids": target_ids,
                }
            )
        dropped_target_details.extend(incompatible_details)
        dropped_counts["incompatible_totals"] = dropped_counts.get(
            "incompatible_totals", 0
        ) + sum(len(detail["target_ids"]) or 1 for detail in incompatible_details)
        raise IPFConversionError(
            "IPF-retained margins do not form one coherent IPF problem.",
            diagnostics={
                "requested_target_count": int(len(filtered_targets)),
                "retained_authored_target_count": int(
                    len(
                        {
                            int(cell.authored_target_id)
                            for block in closed_blocks
                            for cell in block.cells
                            if cell.is_authored
                            and cell.authored_target_id is not None
                        }
                    )
                ),
                "derived_complement_count": int(
                    sum(
                        1
                        for block in closed_blocks
                        for cell in block.cells
                        if not cell.is_authored
                    )
                ),
                "dropped_targets": dropped_counts,
                "dropped_target_details": dropped_target_details,
                "margin_consistency_issues": issues,
                "derived_complement_rows": [
                    {
                        "margin_id": block.margin_id,
                        "cell": "|".join(_cell_assignments_from_cell(block, cell)),
                        "target_value": float(cell.target_value),
                        "source_target_ids": list(cell.source_target_ids),
                        "derivation_reason": cell.derivation_reason,
                    }
                    for block in closed_blocks
                    for cell in block.cells
                    if not cell.is_authored
                ],
            },
        )

    target_metadata = emit_target_rows(closed_blocks)
    retained_authored_target_ids = sorted(
        {
            int(cell.authored_target_id)
            for block in closed_blocks
            for cell in block.cells
            if cell.is_authored and cell.authored_target_id is not None
        }
    )
    derived_complement_rows = [
        {
            "margin_id": block.margin_id,
            "cell": "|".join(_cell_assignments_from_cell(block, cell)),
            "target_value": float(cell.target_value),
            "source_target_ids": list(cell.source_target_ids),
            "derivation_reason": cell.derivation_reason,
        }
        for block in closed_blocks
        for cell in block.cells
        if not cell.is_authored
    ]
    target_metadata.attrs["margin_consistency_issues"] = issues
    target_metadata.attrs["retained_authored_target_ids"] = retained_authored_target_ids
    target_metadata.attrs["derived_complement_rows"] = derived_complement_rows
    target_metadata.attrs["dropped_targets"] = dropped_counts
    target_metadata.attrs["dropped_target_details"] = dropped_target_details
    target_metadata.attrs["requested_target_count"] = int(len(filtered_targets))
    target_metadata.attrs["retained_authored_target_count"] = int(
        len(retained_authored_target_ids)
    )
    target_metadata.attrs["derived_complement_count"] = int(len(derived_complement_rows))
    return unit_data, target_metadata


def split_target_metadata_by_margin(
    target_metadata: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Return one sub-frame per margin_id.

    Kept as a notebook/test helper. The benchmark no longer chains separate
    IPF calls across these blocks.
    """
    return {
        mid: sub.reset_index(drop=True)
        for mid, sub in target_metadata.groupby("margin_id", sort=False)
    }


def _cell_assignments(t: ResolvedTarget, block: MarginBlock) -> List[str]:
    assignments: Dict[str, str] = {}
    if block.geo_var is not None:
        assignments[block.geo_var] = str(t.geo.value)
    for col, lbl in t.cell:
        assignments[col] = str(lbl)
    return [f"{col}={assignments[col]}" for col in block.cell_vars]


def _cell_assignments_from_cell(block: ClosedMarginBlock, cell: MarginCell) -> List[str]:
    assignments: Dict[str, str] = {}
    if block.geo_var is not None:
        assignments[block.geo_var] = str(cell.geo_value)
    for col, lbl in cell.cell:
        assignments[col] = str(lbl)
    return [f"{col}={assignments[col]}" for col in block.cell_vars]


def emit_target_rows(blocks: List[ClosedMarginBlock]) -> pd.DataFrame:
    """Emit one `categorical_margin` row per authored or derived IPF cell."""
    out_rows = []
    for block in blocks:
        if not hasattr(block, "cells"):
            raise TypeError(
                "emit_target_rows expects closed margin blocks. "
                "Call close_margins_for_testing(...) or build_ipf_inputs(...) "
                "before emitting target rows."
            )
        margin_vars_joined = "|".join(block.cell_vars)
        for cell in block.cells:
            cell_assignments = _cell_assignments_from_cell(block, cell)
            out_rows.append(
                {
                    "margin_id": block.margin_id,
                    "scope": block.scope,
                    "target_type": "categorical_margin",
                    "variables": margin_vars_joined,
                    "cell": "|".join(cell_assignments),
                    "target_value": float(cell.target_value),
                    "target_name": cell.target_name,
                    "source_variable": block.source_variable,
                    "is_authored": bool(cell.is_authored),
                    "authored_target_id": (
                        int(cell.authored_target_id)
                        if cell.authored_target_id is not None
                        else np.nan
                    ),
                    "source_target_ids": "|".join(
                        str(target_id) for target_id in cell.source_target_ids
                    ),
                    "closure_status": block.closure_status,
                    "derivation_reason": cell.derivation_reason,
                }
            )
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Back-compat / diagnostics helpers exported for notebooks and tests
# ---------------------------------------------------------------------------


def resolve_targets_for_testing(
    targets_df: pd.DataFrame,
    stratum_constraints: Dict[int, List[dict]],
) -> Tuple[List[ResolvedTarget], List[Tuple[int, str]]]:
    """Pure-Python helper used by the notebook walkthrough and unit tests.

    Resolves each target against the declared bucket schemas without touching
    the saved calibration package or the microsimulation dataset. Returns a
    `(resolved, unresolved)` pair.
    """
    resolved: List[ResolvedTarget] = []
    unresolved: List[Tuple[int, str]] = []
    for _, row in targets_df.reset_index(drop=True).iterrows():
        constraints = stratum_constraints.get(int(row["stratum_id"]), [])
        rt = _resolve_target(row, constraints)
        if rt is None:
            unresolved.append(
                (int(row.get("target_id", -1)), str(row.get("target_name", "?")))
            )
        else:
            resolved.append(rt)
    return resolved, unresolved


def assemble_margins_for_testing(
    resolved: List[ResolvedTarget],
) -> List[MarginBlock]:
    """Expose margin assembly for the notebook walkthrough and tests."""
    return _assemble_margins(resolved)


def close_margins_for_testing(
    resolved: List[ResolvedTarget],
    unit_data: pd.DataFrame,
    tolerance: float = 1e-9,
) -> Tuple[List[ClosedMarginBlock], List[Dict[str, object]]]:
    """Pure-Python closure helper for unit tests and notebook walkthroughs."""
    blocks = _assemble_margins(resolved)
    return _close_margin_blocks(blocks=blocks, unit_data=unit_data, tolerance=tolerance)
