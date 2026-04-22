"""IPF-benchmark input conversion.

Turns a filtered slice of the calibration package into the unit-table +
categorical-margin representation surveysd::ipf consumes.

The conversion groups selected targets by their stratum-constraint signature
into margin blocks, maps each target's constraint tuples to a bucket label
using declared schemas (age 5-year buckets, AGI brackets at district and state
levels, positive-dollar indicators, raw categorical equality), materializes the
derived columns on the cloned-unit table, and emits one `categorical_margin`
row per (margin, geo, cell) triple.

Single-cell-per-geo margins (e.g. `(district × snap > 0)`) get their
complement cell synthesized from baseline microdata totals so the margin is
proper. Complement cells are labelled and scored separately so the paper's
reporting can distinguish authored targets from synthesized baseline pins.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from benchmark_manifest import BenchmarkManifest
from policyengine_us_data.calibration.calibration_utils import apply_op


# ---------------------------------------------------------------------------
# Geography and positive-dollar constants
# ---------------------------------------------------------------------------

_GEO_VARS = {"state_fips", "congressional_district_geoid"}

# Upper cap used to close `> 0` dollar constraints into a finite bucket.
# Larger than annual US GDP, so any real dollar amount falls inside [0, CAP].
POSITIVE_DOLLAR_CAP = 15e12


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
            "`tax_unit_count` and `spm_unit_count` would require a separate "
            "unit table keyed on the respective entity and a separate IPF run "
            "per entity; left out of this pass intentionally."
        ) from exc


# ---------------------------------------------------------------------------
# Margin assembly
# ---------------------------------------------------------------------------


@dataclass
class MarginBlock:
    margin_id: str
    scope: str
    cell_vars: Tuple[str, ...]  # sorted derived-column names used as cell dimensions
    geo_var: Optional[str]  # geo dimension included in the margin
    targets: List[ResolvedTarget]
    # cells that exist in the target package ( (geo_value, cell_tuple) )
    target_cells: set
    # cells synthesized from baseline to close the margin
    synthesized_cells: List[Tuple[object, Tuple[Tuple[str, str], ...], float]]


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
                cell_vars=tuple(sorted(all_vars)),
                geo_var=geo_var,
                targets=[],
                target_cells=set(),
                synthesized_cells=[],
            )
        blocks[key].targets.append(t)
        blocks[key].target_cells.add((t.geo.value, t.cell))
    return list(blocks.values())


def _should_synthesize_complement(block: MarginBlock) -> bool:
    """True iff the block has a 1-cell-per-geo pattern."""
    if block.geo_var is None:
        return len({cell for _, cell in block.target_cells}) == 1
    cells_per_geo: Dict[object, set] = {}
    for geo_value, cell in block.target_cells:
        cells_per_geo.setdefault(geo_value, set()).add(cell)
    return all(len(cells) == 1 for cells in cells_per_geo.values()) and (
        len({cell for _, cell in block.target_cells}) == 1
    )


def _synthesize_complement_cells(
    block: MarginBlock,
    unit_data: pd.DataFrame,
    weight_column: str,
) -> None:
    """Add a complement row per geo to single-cell-per-geo blocks.

    The complement's target value is pinned to the baseline weighted count of
    rows falling in the complement cell. That makes the margin feasible
    without requiring an external complement total.
    """
    cells = {cell for _, cell in block.target_cells}
    if len(cells) != 1:
        return
    (cell,) = cells
    complement_cell = _flip_cell(cell)
    if complement_cell is None:
        return

    # Evaluate the complement cell's weighted count from the unit table.
    mask = _mask_for_cell(unit_data, complement_cell)
    if block.geo_var is None:
        value = float(unit_data.loc[mask, weight_column].sum())
        block.synthesized_cells.append((None, complement_cell, value))
        return
    groups = unit_data.loc[mask].groupby(block.geo_var)[weight_column].sum()
    for geo_val in sorted({g for g, _ in block.target_cells}):
        value = float(groups.get(geo_val, 0.0))
        block.synthesized_cells.append((geo_val, complement_cell, value))


def _flip_cell(
    cell: Tuple[Tuple[str, str], ...],
) -> Optional[Tuple[Tuple[str, str], ...]]:
    """For a cell that contains exactly one `<var>_positive="positive"` entry,
    return the same cell with that entry flipped to `non_positive`. Otherwise
    return None (complement not defined)."""
    flipped: List[Tuple[str, str]] = []
    changed = False
    for col, label in cell:
        if col.endswith("_positive") and label == POSITIVE_LABEL:
            flipped.append((col, NON_POSITIVE_LABEL))
            changed = True
        else:
            flipped.append((col, label))
    if not changed:
        return None
    return tuple(sorted(flipped))


def _mask_for_cell(
    unit_data: pd.DataFrame, cell: Tuple[Tuple[str, str], ...]
) -> np.ndarray:
    mask = np.ones(len(unit_data), dtype=bool)
    for col, label in cell:
        if col not in unit_data.columns:
            raise KeyError(f"Unit table is missing derived column '{col}'")
        mask &= unit_data[col].astype(str).to_numpy() == str(label)
    return mask


def check_margin_consistency(
    blocks: List[MarginBlock],
    tolerance: float = 1e-3,
) -> List[Dict[str, object]]:
    """Return a list of consistency issues, one per (geo_value) that has
    mismatched totals across different margin blocks.

    `surveysd::ipf` refuses to run when two person-scope (or two household-
    scope) margins imply different population totals for the same geography.
    The check is `rel_err = |t1 - t2| / max(t1, t2) > tolerance`.
    """
    totals_by_scope_geo: Dict[Tuple[str, Optional[str], object], Dict[str, float]] = {}
    for block in blocks:
        # Include synthesized complement cells — they are part of the margin's total.
        by_geo: Dict[object, float] = {}
        for t in block.targets:
            by_geo[t.geo.value] = by_geo.get(t.geo.value, 0.0) + float(t.target_value)
        for geo_val, _, value in block.synthesized_cells:
            by_geo[geo_val] = by_geo.get(geo_val, 0.0) + float(value)
        for geo_val, total in by_geo.items():
            key = (block.scope, block.geo_var, geo_val)
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
# Main entry point
# ---------------------------------------------------------------------------


def build_ipf_inputs(
    package: Dict,
    manifest: BenchmarkManifest,
    filtered_targets: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(unit_metadata, ipf_target_metadata)` ready for `ipf_runner.R`.

    Consumes the same `filtered_targets` slice that the GREG and L0 runners
    see. Internally filters to the IPF-eligible subset with two checks:

    1. **Count check** — keep only targets whose `variable` is a supported
       count (`person_count`, `household_count`). Dollar-total targets stay
       in the shared matrix for GREG and L0 but are dropped here.
    2. **Resolver check** — keep only targets whose stratum constraints
       resolve through the declared bucket schemas (age, AGI district / AGI
       state, EITC child count, positive-dollar `>0`, raw equality). Targets
       whose constraints don't match any declared schema are dropped.

    Both checks are non-fatal; dropped targets are recorded on the returned
    metadata's `.attrs['dropped_targets']` so the caller can report them.

    The surviving subset is grouped by constraint signature into margin
    blocks; single-cell-per-geo margins get their complement cell
    synthesized from the baseline weighted count of the complement predicate
    on the cloned-unit table.
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

    # --- Synthesize complement cells for single-cell-per-geo margins -------
    for block in blocks:
        if _should_synthesize_complement(block):
            _synthesize_complement_cells(block, unit_data, weight_column="base_weight")

    # --- Emit categorical_margin target rows -------------------------------
    out_rows = []
    for block in blocks:
        margin_vars_joined = "|".join(block.cell_vars)
        # Authored target rows
        for t in block.targets:
            cell_assignments = _cell_assignments(t, block)
            out_rows.append(
                {
                    "margin_id": block.margin_id,
                    "scope": block.scope,
                    "target_type": "categorical_margin",
                    "variables": margin_vars_joined,
                    "cell": "|".join(cell_assignments),
                    "target_value": float(t.target_value),
                    "target_name": t.target_name,
                    "source_variable": t.source_variable,
                    "synthesized": False,
                }
            )
        # Synthesized complement rows
        for geo_val, cell, value in block.synthesized_cells:
            cell_assignments = _complement_cell_assignments(geo_val, cell, block)
            out_rows.append(
                {
                    "margin_id": block.margin_id,
                    "scope": block.scope,
                    "target_type": "categorical_margin",
                    "variables": margin_vars_joined,
                    "cell": "|".join(cell_assignments),
                    "target_value": float(value),
                    "target_name": f"{block.margin_id}_complement_{geo_val}",
                    "source_variable": "synthesized_baseline",
                    "synthesized": True,
                }
            )

    target_metadata = pd.DataFrame(out_rows)

    # --- Margin consistency check -----------------------------------------
    issues = check_margin_consistency(blocks)
    target_metadata.attrs["margin_consistency_issues"] = issues
    target_metadata.attrs["dropped_targets"] = {
        "non_count_style": int(len(dropped_non_count)),
        "unresolvable_constraints": int(len(dropped_unresolvable)),
        "unresolvable_examples": dropped_unresolvable[:10],
    }
    if issues:
        import warnings

        warnings.warn(
            f"{len(issues)} geography-scope combination(s) have mismatched "
            "totals across multiple IPF margin blocks. `surveysd::ipf` will "
            "refuse to run with these margins combined; run one margin block "
            "at a time or harmonize the authored totals. See "
            "target_metadata.attrs['margin_consistency_issues'] for details.",
            stacklevel=2,
        )

    return unit_data, target_metadata


def split_target_metadata_by_margin(
    target_metadata: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Return one sub-frame per margin_id, safe to pass to `ipf_runner.R`.

    Inconsistent authored totals across margin blocks break a combined
    `surveysd::ipf` call. Splitting by `margin_id` lets the benchmark run
    each block independently and score the weights separately.
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


def _complement_cell_assignments(
    geo_val: object,
    cell: Tuple[Tuple[str, str], ...],
    block: MarginBlock,
) -> List[str]:
    assignments: Dict[str, str] = {}
    if block.geo_var is not None:
        assignments[block.geo_var] = str(geo_val)
    for col, lbl in cell:
        assignments[col] = str(lbl)
    return [f"{col}={assignments[col]}" for col in block.cell_vars]


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
