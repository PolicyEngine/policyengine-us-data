"""Diagnostic artifact writers for Stage 1 dataset-build outputs."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    DatasetArtifactSpec,
    stage_1_diagnostic_artifact_specs,
    stage_1_pipeline_artifact_specs,
)
from .context import DatasetBuildContext
from policyengine_us_data.utils.step_manifest import sha256_file


ARTIFACT_SCHEMA_VERSION = "1"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            default=_json_default,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".h5":
        return "application/x-hdf5"
    if suffix == ".db":
        return "application/vnd.sqlite3"
    if suffix == ".json":
        return "application/json"
    if suffix == ".npy":
        return "application/x-numpy-array"
    if suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


def _artifact_ref_for_path(
    *,
    logical_name: str,
    path: Path,
    metadata: Mapping[str, Any],
):
    from policyengine_us_data.stage_contracts import ArtifactRef

    return ArtifactRef(
        logical_name=logical_name,
        uri=path.resolve().as_uri(),
        sha256=f"sha256:{sha256_file(path)}",
        size_bytes=path.stat().st_size,
        media_type=_media_type_for_path(path),
        metadata=metadata,
    )


def _diagnostic_ref_for_path(
    *,
    spec: DatasetArtifactSpec,
    path: Path,
    summary: Mapping[str, Any],
):
    from policyengine_us_data.stage_contracts import DiagnosticRef

    return DiagnosticRef(
        name=spec.logical_name,
        kind=spec.diagnostic_kind or spec.artifact_family,
        artifact=_artifact_ref_for_path(
            logical_name=spec.logical_name,
            path=path,
            metadata={
                "artifact_family": spec.artifact_family,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "substage_id": spec.substage_id,
            },
        ),
        summary=summary,
        severity="info",
    )


def _diagnostic_spec(logical_name: str) -> DatasetArtifactSpec:
    for spec in stage_1_diagnostic_artifact_specs():
        if spec.logical_name == logical_name:
            return spec
    raise KeyError(f"Unknown Stage 1 diagnostic spec: {logical_name}")


def _cheap_h5_summary(path: Path) -> dict[str, Any]:
    import h5py

    datasets: list[dict[str, Any]] = []
    entities: dict[str, dict[str, Any]] = {}

    with h5py.File(path, "r") as h5_file:

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            parts = name.split("/")
            entity = parts[0] if parts else ""
            variable = parts[-2] if len(parts) > 1 else parts[-1]
            period = parts[-1] if parts[-1].isdigit() else None
            row_count = int(obj.shape[0]) if obj.shape else None
            datasets.append(
                {
                    "path": name,
                    "entity": entity,
                    "variable": variable,
                    "period": period,
                    "dtype": str(obj.dtype),
                    "shape": list(obj.shape),
                    "row_count": row_count,
                }
            )
            entity_summary = entities.setdefault(
                entity,
                {
                    "dataset_count": 0,
                    "variables": set(),
                    "periods": set(),
                    "row_counts": {},
                },
            )
            entity_summary["dataset_count"] += 1
            entity_summary["variables"].add(variable)
            if period is not None:
                entity_summary["periods"].add(period)
            if row_count is not None:
                entity_summary["row_counts"][name] = row_count

        h5_file.visititems(visit)

    return {
        "datasets": datasets,
        "entities": {
            entity: {
                "dataset_count": summary["dataset_count"],
                "variables": sorted(summary["variables"]),
                "periods": sorted(summary["periods"]),
                "row_counts": summary["row_counts"],
            }
            for entity, summary in sorted(entities.items())
        },
    }


def _sqlite_summary(path: Path) -> dict[str, Any]:
    tables = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        table_names = [
            row["name"]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]
        checksum_material = []
        for table_name in table_names:
            quoted_table_name = _quote_sql_identifier(table_name)
            columns = [
                {
                    "name": row["name"],
                    "type": row["type"],
                    "notnull": int(row["notnull"]),
                    "pk": int(row["pk"]),
                }
                for row in conn.execute(f"PRAGMA table_info({quoted_table_name})")
            ]
            row_count = conn.execute(
                f"SELECT COUNT(*) AS row_count FROM {quoted_table_name}"
            ).fetchone()["row_count"]
            table_summary = {
                "name": table_name,
                "columns": columns,
                "row_count": int(row_count),
            }
            tables.append(table_summary)
            checksum_material.append(table_summary)

    digest_payload = json.dumps(
        checksum_material,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    import hashlib

    return {
        "tables": tables,
        "known_target_tables": [
            table["name"]
            for table in tables
            if table["name"] in {"targets", "strata", "stratum_constraints"}
        ],
        "schema_checksum": hashlib.sha256(digest_payload).hexdigest(),
    }


def _quote_sql_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@dataclass(frozen=True, kw_only=True)
class DatasetInventoryWriter:
    """Write a compact inventory of Stage 1 artifacts staged for a run."""

    context: DatasetBuildContext

    def write(
        self,
        *,
        skip_enhanced_cps: bool = False,
        skip_stage_5: bool = False,
    ):
        spec = _diagnostic_spec("dataset_inventory")
        artifacts = []
        seen_logical_names: set[str] = set()
        for artifact_spec in stage_1_pipeline_artifact_specs():
            if artifact_spec.diagnostic_output:
                continue
            if skip_enhanced_cps and artifact_spec.skip_when_enhanced_cps_skipped:
                continue
            if skip_stage_5 and artifact_spec.skip_when_stage_5_skipped:
                continue
            path = self.context.artifact_path(artifact_spec.filename)
            if not path.exists():
                if artifact_spec.required:
                    raise FileNotFoundError(f"Missing staged artifact: {path}")
                continue
            if artifact_spec.logical_name in seen_logical_names:
                raise ValueError(
                    f"Duplicate Stage 1 artifact: {artifact_spec.logical_name}"
                )
            seen_logical_names.add(artifact_spec.logical_name)
            artifacts.append(_inventory_entry(artifact_spec, path))

        payload = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            **self.context.identity(),
            "artifacts": artifacts,
        }
        path = self.context.artifact_path(spec.filename)
        _write_json(path, payload)
        return _diagnostic_ref_for_path(
            spec=spec,
            path=path,
            summary={"artifact_count": len(artifacts)},
        )


@dataclass(frozen=True, kw_only=True)
class SourceDatasetSchemaSummaryWriter:
    """Write a metadata-only schema summary for the source-imputed H5 handoff."""

    context: DatasetBuildContext

    def write(self):
        spec = _diagnostic_spec("source_dataset_schema_summary")
        source_path = self.context.artifact_path(
            "source_imputed_stratified_extended_cps.h5"
        )
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source dataset artifact: {source_path}")
        h5_summary = _cheap_h5_summary(source_path)
        payload = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            **self.context.identity(),
            "logical_name": "source_imputed_stratified_extended_cps",
            "path": source_path.name,
            **h5_summary,
        }
        path = self.context.artifact_path(spec.filename)
        _write_json(path, payload)
        return _diagnostic_ref_for_path(
            spec=spec,
            path=path,
            summary={
                "entity_count": len(h5_summary["entities"]),
                "dataset_count": len(h5_summary["datasets"]),
            },
        )


@dataclass(frozen=True, kw_only=True)
class TargetDatabaseSchemaSummaryWriter:
    """Write a schema and row-count summary for the Stage 1 target database."""

    context: DatasetBuildContext

    def write(self):
        spec = _diagnostic_spec("target_database_schema_summary")
        db_path = self.context.artifact_path("policy_data.db")
        if not db_path.exists():
            raise FileNotFoundError(f"Missing target database artifact: {db_path}")
        db_summary = _sqlite_summary(db_path)
        payload = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            **self.context.identity(),
            "logical_name": "policy_data_db",
            "path": db_path.name,
            **db_summary,
        }
        path = self.context.artifact_path(spec.filename)
        _write_json(path, payload)
        return _diagnostic_ref_for_path(
            spec=spec,
            path=path,
            summary={
                "table_count": len(db_summary["tables"]),
                "known_target_tables": db_summary["known_target_tables"],
                "schema_checksum": db_summary["schema_checksum"],
            },
        )


def _inventory_entry(spec: DatasetArtifactSpec, path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "logical_name": spec.logical_name,
        "artifact_family": spec.artifact_family,
        "substage_id": spec.substage_id,
        "path": path.name,
        "sha256": f"sha256:{sha256_file(path)}",
        "size_bytes": path.stat().st_size,
        "media_type": _media_type_for_path(path),
    }
    if spec.period is not None:
        entry["period"] = spec.period
    if path.suffix == ".h5":
        entry["row_counts"] = {
            dataset["path"]: dataset["row_count"]
            for dataset in _cheap_h5_summary(path)["datasets"]
            if dataset["row_count"] is not None
        }
    elif path.suffix == ".db":
        db_summary = _sqlite_summary(path)
        entry["row_counts"] = {
            table["name"]: table["row_count"] for table in db_summary["tables"]
        }
        entry["schema_checksum"] = db_summary["schema_checksum"]
    return entry


def write_stage_1_diagnostics(
    *,
    context: DatasetBuildContext,
    skip_enhanced_cps: bool = False,
    skip_stage_5: bool = False,
) -> tuple[Any, ...]:
    """Write Stage 1 diagnostic artifacts and return their contract refs."""

    refs = [
        DatasetInventoryWriter(context=context).write(
            skip_enhanced_cps=skip_enhanced_cps,
            skip_stage_5=skip_stage_5,
        ),
        TargetDatabaseSchemaSummaryWriter(context=context).write(),
    ]
    if not skip_stage_5:
        refs.insert(1, SourceDatasetSchemaSummaryWriter(context=context).write())
    return tuple(refs)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DatasetInventoryWriter",
    "SourceDatasetSchemaSummaryWriter",
    "TargetDatabaseSchemaSummaryWriter",
    "write_stage_1_diagnostics",
]
