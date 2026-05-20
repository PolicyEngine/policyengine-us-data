"""Target catalog and selection artifacts for Stage 2 package builds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.utils.target_variables import target_variable_components

from .specs import (
    CALIBRATION_TARGET_FACETS_FILENAME,
    CALIBRATION_TARGETS_FILENAME,
    TargetConfigIdentity,
)

TARGET_CATALOG_COLUMNS: tuple[str, ...] = (
    "target_id",
    "stratum_id",
    "variable",
    "reform_id",
    "value",
    "period",
    "geo_level",
    "geographic_id",
    "domain_variable",
    "source",
    "notes",
)
GEO_CONSTRAINT_VARIABLES = frozenset(
    {
        "state_fips",
        "congressional_district_geoid",
        "ucgid_str",
    }
)
TARGET_OVERVIEW_VIEW = """\
CREATE VIEW IF NOT EXISTS target_overview AS
SELECT
    t.target_id,
    t.stratum_id,
    t.variable,
    t.reform_id,
    t.value,
    t.period,
    t.active,
    CASE
        WHEN MAX(CASE
            WHEN sc.constraint_variable = 'congressional_district_geoid'
                THEN 1
            WHEN sc.constraint_variable = 'ucgid_str'
                AND length(sc.value) = 13 THEN 1
            ELSE 0 END) = 1 THEN 'district'
        WHEN MAX(CASE
            WHEN sc.constraint_variable = 'state_fips' THEN 1
            WHEN sc.constraint_variable = 'ucgid_str'
                AND length(sc.value) = 11 THEN 1
            ELSE 0 END) = 1 THEN 'state'
        ELSE 'national'
    END AS geo_level,
    COALESCE(
        MAX(CASE
            WHEN sc.constraint_variable
                = 'congressional_district_geoid'
            THEN sc.value END),
        MAX(CASE
            WHEN sc.constraint_variable = 'state_fips'
            THEN sc.value END),
        MAX(CASE
            WHEN sc.constraint_variable = 'ucgid_str'
            THEN sc.value END),
        'US'
    ) AS geographic_id,
    (
        SELECT GROUP_CONCAT(cv, ',')
        FROM (
            SELECT DISTINCT sc2.constraint_variable AS cv
            FROM stratum_constraints sc2
            WHERE sc2.stratum_id = t.stratum_id
              AND sc2.constraint_variable NOT IN (
                  'state_fips', 'congressional_district_geoid',
                  'tax_unit_is_filer', 'ucgid_str'
              )
            ORDER BY sc2.constraint_variable
        )
    ) AS domain_variable
FROM targets t
LEFT JOIN stratum_constraints sc ON t.stratum_id = sc.stratum_id
GROUP BY t.target_id, t.stratum_id, t.variable,
         t.reform_id, t.value, t.period, t.active;
"""


@pipeline_node(
    PipelineNode(
        id="stage2_target_catalog_reader",
        label="Stage 2 Target Catalog Reader",
        node_type="library",
        description="Read active and disabled calibration targets plus stratum constraints from policy_data.db.",
        source_file="policyengine_us_data/calibration_package/targets.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=["policy_data.db"],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_targets.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class TargetCatalogReader:
    """Read the Stage 2 target catalog from the calibration target database."""

    time_period: int
    db_uri: str | None = None
    engine: Any | None = None

    def load(self, target_filter: Mapping[str, Any] | None = None) -> "TargetCatalog":
        """Load selected active targets and disabled target rows."""

        if self.engine is None and self.db_uri is None:
            raise ValueError("TargetCatalogReader requires db_uri or engine")
        engine = self.engine if self.engine is not None else create_engine(self.db_uri)
        owns_engine = self.engine is None
        try:
            _ensure_target_overview(engine)
            target_columns = _table_columns(engine, "targets")
            view_columns = _table_columns(engine, "target_overview")
            targets = _query_targets(
                engine,
                time_period=self.time_period,
                target_filter=target_filter or {},
                active_only=True,
                target_columns=target_columns,
                view_columns=view_columns,
            )
            disabled_targets = _query_targets(
                engine,
                time_period=self.time_period,
                target_filter=target_filter or {},
                active_only=False,
                target_columns=target_columns,
                view_columns=view_columns,
            )
            constraints_by_stratum = _load_constraints_by_stratum(engine)
            return TargetCatalog(
                targets=targets,
                disabled_targets=disabled_targets,
                constraints_by_stratum=constraints_by_stratum,
            )
        finally:
            if owns_engine:
                engine.dispose()


@dataclass(frozen=True, kw_only=True)
class TargetCatalog:
    """Targets and stratum constraints available to Stage 2 selection."""

    targets: pd.DataFrame
    disabled_targets: pd.DataFrame = field(default_factory=pd.DataFrame)
    constraints_by_stratum: Mapping[int, tuple[Mapping[str, Any], ...]] = field(
        default_factory=dict
    )

    @classmethod
    def from_targets(
        cls,
        targets: pd.DataFrame,
        *,
        disabled_targets: pd.DataFrame | None = None,
        constraints_by_stratum: Mapping[int, Iterable[Mapping[str, Any]]] | None = None,
    ) -> "TargetCatalog":
        """Create a catalog from in-memory target rows."""

        return cls(
            targets=_normalize_target_frame(targets),
            disabled_targets=_normalize_target_frame(
                disabled_targets if disabled_targets is not None else pd.DataFrame()
            ),
            constraints_by_stratum={
                int(key): tuple(dict(item) for item in value)
                for key, value in (constraints_by_stratum or {}).items()
            },
        )

    def constraints_for(self, stratum_id: int) -> tuple[Mapping[str, Any], ...]:
        """Return deterministic constraints for a stratum."""

        return self.constraints_by_stratum.get(int(stratum_id), ())


@pipeline_node(
    PipelineNode(
        id="stage2_target_selection_policy",
        label="Stage 2 Target Selection Policy",
        node_type="library",
        description="Apply target config include/exclude rules and validate additive target expressions before matrix construction.",
        source_file="policyengine_us_data/calibration_package/targets.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=["policy_data.db"],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_targets.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class TargetSelectionPolicy:
    """Target include/exclude rules applied before matrix materialization."""

    include_rules: tuple[Mapping[str, Any], ...] = ()
    exclude_rules: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "TargetSelectionPolicy":
        """Create a policy from the Stage 2 target config mapping."""

        config = config or {}
        return cls(
            include_rules=tuple(dict(rule) for rule in config.get("include", ())),
            exclude_rules=tuple(dict(rule) for rule in config.get("exclude", ())),
        )

    def select(
        self,
        catalog: TargetCatalog,
        *,
        target_config_identity: TargetConfigIdentity | None = None,
        valid_variables: Iterable[str] | Mapping[str, Any] | None = None,
    ) -> "TargetSelectionResult":
        """Apply this policy to a target catalog."""

        targets = _normalize_target_frame(catalog.targets)
        _validate_target_expressions(targets, valid_variables)
        keep_mask = pd.Series(True, index=targets.index)
        if self.include_rules:
            keep_mask = _match_rules(targets, self.include_rules)
        if self.exclude_rules:
            keep_mask &= ~_match_rules(targets, self.exclude_rules)
        selected = targets.loc[keep_mask].reset_index(drop=True)
        disabled = _normalize_target_frame(catalog.disabled_targets)
        return TargetSelectionResult(
            targets_df=selected,
            disabled_targets_df=disabled,
            constraints_by_stratum=catalog.constraints_by_stratum,
            target_config_path=(
                target_config_identity.path
                if target_config_identity is not None
                else None
            ),
            target_config_sha256=(
                target_config_identity.sha256
                if target_config_identity is not None
                else None
            ),
            target_config_mode=(
                target_config_identity.mode
                if target_config_identity is not None
                else None
            ),
        )


@pipeline_node(
    PipelineNode(
        id="stage2_target_selection_result",
        label="Stage 2 Target Selection Result",
        node_type="library",
        description="Stable selected target metadata, checksum, JSONL rows, and facet counts consumed by Stage 2 matrix building and diagnostics.",
        source_file="policyengine_us_data/calibration_package/targets.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_out=[
            CALIBRATION_TARGETS_FILENAME,
            CALIBRATION_TARGET_FACETS_FILENAME,
        ],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_targets.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class TargetSelectionResult:
    """Selected target metadata in the order consumed by Stage 2."""

    targets_df: pd.DataFrame
    disabled_targets_df: pd.DataFrame
    constraints_by_stratum: Mapping[int, tuple[Mapping[str, Any], ...]]
    target_config_path: str | None
    target_config_sha256: str | None
    target_config_mode: str | None
    target_names: tuple[str, ...] = ()

    @property
    def target_ids(self) -> list[int]:
        """Return selected target IDs in stable order."""

        return [int(value) for value in self.targets_df["target_id"].tolist()]

    @property
    def n_selected_targets(self) -> int:
        """Return the number of selected package targets."""

        return int(len(self.targets_df))

    @property
    def checksum(self) -> str:
        """Return the stable target selection checksum."""

        digest = hashlib.sha256()
        for row in self.to_rows():
            digest.update(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            )
            digest.update(b"\n")
        return f"sha256:{digest.hexdigest()}"

    def with_matrix_order(
        self,
        targets_df: pd.DataFrame,
        target_names: Iterable[str],
    ) -> "TargetSelectionResult":
        """Return this result in the matrix/package target order."""

        ordered = _normalize_target_frame(targets_df)
        names = tuple(str(name) for name in target_names)
        if len(ordered) != len(names):
            raise ValueError("Target metadata row count must match target_names")
        return TargetSelectionResult(
            targets_df=ordered.reset_index(drop=True),
            disabled_targets_df=self.disabled_targets_df,
            constraints_by_stratum=self.constraints_by_stratum,
            target_config_path=self.target_config_path,
            target_config_sha256=self.target_config_sha256,
            target_config_mode=self.target_config_mode,
            target_names=names,
        )

    def to_rows(self) -> list[dict[str, Any]]:
        """Return JSONL-ready selected target rows."""

        rows: list[dict[str, Any]] = []
        target_names = self.target_names or tuple(
            _fallback_target_name(row)
            for _, row in self.targets_df.reset_index(drop=True).iterrows()
        )
        for target_index, (_, row) in enumerate(
            self.targets_df.reset_index(drop=True).iterrows()
        ):
            constraints = [
                _jsonable_constraint(constraint)
                for constraint in self.constraints_by_stratum.get(
                    int(row["stratum_id"]),
                    (),
                )
            ]
            components = target_variable_components(str(row["variable"]))
            target_expression = str(row["variable"]) if len(components) > 1 else None
            rows.append(
                {
                    "target_id": int(row["target_id"]),
                    "target_index": int(target_index),
                    "target_name": str(target_names[target_index]),
                    "variable": str(row["variable"]),
                    "target_expression": target_expression,
                    "target_components": components,
                    "target_value": _optional_float(row.get("value")),
                    "period": _optional_int(row.get("period")),
                    "geography_level": _optional_string(row.get("geo_level")),
                    "geography_id": _optional_string(row.get("geographic_id")),
                    "domain_variable": _optional_string(row.get("domain_variable")),
                    "source_table": "targets",
                    "source": _optional_string(row.get("source")),
                    "target_config_path": self.target_config_path,
                    "target_config_sha256": self.target_config_sha256,
                    "target_config_mode": self.target_config_mode,
                    "included_in_package": True,
                    "notes": _optional_string(row.get("notes")),
                    "constraint_key": _constraint_key(constraints),
                    "target_constraints": constraints,
                }
            )
        return rows

    def disabled_rows(self) -> list[dict[str, Any]]:
        """Return disabled target rows for reporting."""

        rows: list[dict[str, Any]] = []
        for _, row in self.disabled_targets_df.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "target_id": int(row["target_id"]),
                    "variable": str(row["variable"]),
                    "period": _optional_int(row.get("period")),
                    "geography_level": _optional_string(row.get("geo_level")),
                    "geography_id": _optional_string(row.get("geographic_id")),
                    "domain_variable": _optional_string(row.get("domain_variable")),
                    "included_in_package": False,
                    "notes": _optional_string(row.get("notes")),
                }
            )
        return rows

    def facets(self) -> dict[str, Any]:
        """Return compact counts derived from selected row-level metadata."""

        return target_facets_from_rows(self.to_rows())

    def summary(self) -> dict[str, Any]:
        """Return a compact selection summary for package metadata."""

        return {
            "target_count": self.n_selected_targets,
            "disabled_target_count": int(len(self.disabled_targets_df)),
            "target_selection_sha256": self.checksum,
            "target_config_path": self.target_config_path,
            "target_config_sha256": self.target_config_sha256,
            "target_config_mode": self.target_config_mode,
        }

    def write_artifacts(
        self,
        targets_path: str | Path,
        facets_path: str | Path,
    ) -> tuple[Path, Path]:
        """Write row-level target metadata and facet summary artifacts."""

        rows = self.to_rows()
        target_file = Path(targets_path)
        facet_file = Path(facets_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with target_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
        facet_file.write_text(
            json.dumps(self.facets(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target_file, facet_file


def target_facets_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive target facet counts from row-level target metadata."""

    material = [dict(row) for row in rows]
    return {
        "target_count": len(material),
        "by_variable": _counts(material, "variable"),
        "by_geography_level": _counts(material, "geography_level"),
        "by_target_name": _counts(material, "target_name"),
        "by_period": _counts(material, "period"),
        "by_constraint_key": _counts(material, "constraint_key"),
    }


def _query_targets(
    engine: Any,
    *,
    time_period: int,
    target_filter: Mapping[str, Any],
    active_only: bool,
    target_columns: set[str],
    view_columns: set[str],
) -> pd.DataFrame:
    where_clause, params = _target_filter_sql(target_filter, active_only=active_only)
    reform_expr = "tv.reform_id" if "reform_id" in view_columns else "0"
    reform_group = "reform_id" if "reform_id" in view_columns else "0"
    source_expr = "t.source" if "source" in target_columns else "NULL"
    notes_expr = "t.notes" if "notes" in target_columns else "NULL"
    active_condition = "tv.active = 1" if active_only else "tv.active != 1"
    if active_only:
        query = f"""
        WITH filtered_targets AS (
            SELECT tv.target_id, tv.stratum_id, tv.variable,
                   {reform_expr} AS reform_id, tv.value, tv.period,
                   tv.geo_level, tv.geographic_id, tv.domain_variable
            FROM target_overview tv
            WHERE {active_condition}
              AND ({where_clause})
        ),
        best_periods AS (
            SELECT stratum_id, variable, {reform_group} AS reform_id,
                CASE
                    WHEN MAX(CASE WHEN period <= :time_period THEN period END)
                         IS NOT NULL
                    THEN MAX(CASE WHEN period <= :time_period THEN period END)
                    ELSE MIN(period)
                END AS best_period
            FROM filtered_targets
            GROUP BY stratum_id, variable, reform_id
        )
        SELECT ft.*, {source_expr} AS source, {notes_expr} AS notes
        FROM filtered_targets ft
        JOIN best_periods bp
          ON ft.stratum_id = bp.stratum_id
         AND ft.variable = bp.variable
         AND ft.reform_id = bp.reform_id
         AND ft.period = bp.best_period
        LEFT JOIN targets t ON t.target_id = ft.target_id
        ORDER BY ft.target_id
        """
        params["time_period"] = int(time_period)
    else:
        query = f"""
        SELECT tv.target_id, tv.stratum_id, tv.variable,
               {reform_expr} AS reform_id, tv.value, tv.period,
               tv.geo_level, tv.geographic_id, tv.domain_variable,
               {source_expr} AS source, {notes_expr} AS notes
        FROM target_overview tv
        LEFT JOIN targets t ON t.target_id = tv.target_id
        WHERE {active_condition}
          AND ({where_clause})
        ORDER BY tv.target_id
        """
    with engine.connect() as conn:
        return _normalize_target_frame(pd.read_sql(text(query), conn, params=params))


def _ensure_target_overview(engine: Any) -> None:
    with engine.connect() as conn:
        conn.execute(text(TARGET_OVERVIEW_VIEW))
        conn.commit()


def _target_filter_sql(
    target_filter: Mapping[str, Any],
    *,
    active_only: bool,
) -> tuple[str, dict[str, Any]]:
    conditions: list[str] = []
    params: dict[str, Any] = {}
    filter_columns = {
        "domain_variables": "tv.domain_variable",
        "variables": "tv.variable",
        "target_ids": "tv.target_id",
        "stratum_ids": "tv.stratum_id",
    }
    for key, column in filter_columns.items():
        if key not in target_filter:
            continue
        values = list(target_filter[key])
        if not values:
            conditions.append("0 = 1")
            continue
        placeholders = []
        for index, value in enumerate(values):
            param = f"{key}_{index}_{'active' if active_only else 'disabled'}"
            placeholders.append(f":{param}")
            params[param] = value
        conditions.append(f"{column} IN ({', '.join(placeholders)})")
    return (" AND ".join(f"({condition})" for condition in conditions) or "1=1", params)


def _load_constraints_by_stratum(
    engine: Any,
) -> dict[int, tuple[Mapping[str, Any], ...]]:
    query = """
    SELECT stratum_id, constraint_variable AS variable, operation, value
    FROM stratum_constraints
    ORDER BY stratum_id, constraint_id
    """
    with engine.connect() as conn:
        frame = pd.read_sql(text(query), conn)
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for _, row in frame.iterrows():
        grouped.setdefault(int(row["stratum_id"]), []).append(
            {
                "variable": str(row["variable"]),
                "operation": str(row["operation"]),
                "value": str(row["value"]),
            }
        )
    return {key: tuple(value) for key, value in grouped.items()}


def _table_columns(engine: Any, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return {str(row[1]) for row in rows}


def _normalize_target_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in TARGET_CATALOG_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    if len(normalized):
        if normalized["target_id"].isna().all():
            normalized["target_id"] = list(range(len(normalized)))
        if normalized["stratum_id"].isna().all():
            normalized["stratum_id"] = list(range(len(normalized)))
        normalized = normalized.loc[:, list(TARGET_CATALOG_COLUMNS)]
    return normalized.reset_index(drop=True)


def _match_rules(
    targets_df: pd.DataFrame,
    rules: Iterable[Mapping[str, Any]],
) -> pd.Series:
    mask = pd.Series(False, index=targets_df.index)
    for rule in rules:
        if "variable" not in rule:
            raise ValueError("Target selection rules require a variable")
        rule_mask = targets_df["variable"].astype(str) == str(rule["variable"])
        if "geo_level" in rule:
            rule_mask &= targets_df["geo_level"].astype(str) == str(rule["geo_level"])
        if "domain_variable" in rule:
            domain_values = targets_df["domain_variable"].fillna("").astype(str)
            rule_mask &= domain_values == str(rule["domain_variable"])
        mask |= rule_mask
    return mask


def _validate_target_expressions(
    targets_df: pd.DataFrame,
    valid_variables: Iterable[str] | Mapping[str, Any] | None,
) -> None:
    if valid_variables is None:
        return
    valid = set(valid_variables)
    for variable in targets_df["variable"].astype(str):
        components = target_variable_components(variable)
        missing = [component for component in components if component not in valid]
        if missing:
            raise ValueError(
                "Target variable expression contains unknown component(s): "
                + ", ".join(missing)
            )


def _jsonable_constraint(constraint: Mapping[str, Any]) -> dict[str, str]:
    return {
        "variable": str(constraint.get("variable")),
        "operation": str(constraint.get("operation")),
        "value": str(constraint.get("value")),
    }


def _constraint_key(constraints: Iterable[Mapping[str, Any]]) -> str:
    material = [
        f"{item['variable']}{item['operation']}{item['value']}"
        for item in constraints
        if item.get("variable") not in GEO_CONSTRAINT_VARIABLES
    ]
    return "|".join(material) if material else "none"


def _fallback_target_name(row: pd.Series) -> str:
    geo = str(row.get("geographic_id") or "US")
    return f"{geo}/{row.get('variable')}"


def _counts(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key)) for row in rows)
    return dict(sorted(counter.items()))


def _optional_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


__all__ = [
    "GEO_CONSTRAINT_VARIABLES",
    "TARGET_CATALOG_COLUMNS",
    "TARGET_OVERVIEW_VIEW",
    "TargetCatalog",
    "TargetCatalogReader",
    "TargetSelectionPolicy",
    "TargetSelectionResult",
    "target_facets_from_rows",
]
