"""State and path helpers for Modal pipeline step manifests."""

from __future__ import annotations

import os
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

from policyengine_us_data.utils.run_context import RunContext
from policyengine_us_data.utils.step_manifest import (
    ArtifactReference,
    collect_artifacts,
    collect_directory_artifacts,
)

PIPELINE_MOUNT = "/pipeline"
STAGING_MOUNT = "/staging"
ARTIFACTS_BASE = f"{PIPELINE_MOUNT}/artifacts"
RUNS_DIR = f"{PIPELINE_MOUNT}/runs"


def artifacts_dir_for_run(run_id: str) -> str:
    """Return the run-scoped artifacts directory."""
    if run_id:
        return f"{ARTIFACTS_BASE}/{run_id}"
    return ARTIFACTS_BASE


@dataclass
class RunMetadata:
    """Metadata for a pipeline run."""

    run_id: str
    branch: str
    sha: str
    version: str
    start_time: str
    status: str
    error: Optional[str] = None
    resume_history: list = field(default_factory=list)
    fingerprint: Optional[str] = None
    regional_fingerprint: Optional[str] = None
    run_context: dict = field(default_factory=dict)
    modal_app_name: Optional[str] = None
    modal_environment: Optional[str] = None
    hf_staging_prefix: Optional[str] = None

    def __post_init__(self) -> None:
        if self.regional_fingerprint is None and self.fingerprint is not None:
            self.regional_fingerprint = self.fingerprint
        if self.fingerprint is None and self.regional_fingerprint is not None:
            self.fingerprint = self.regional_fingerprint

    def to_dict(self) -> dict:
        data = asdict(self)
        if (
            data.get("fingerprint") is None
            and data.get("regional_fingerprint") is not None
        ):
            data["fingerprint"] = data["regional_fingerprint"]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        data = dict(data)
        if "run_context" not in data and "publication_context" in data:
            data["run_context"] = data["publication_context"]
        if (
            data.get("regional_fingerprint") is None
            and data.get("fingerprint") is not None
        ):
            data["regional_fingerprint"] = data["fingerprint"]
        allowed_fields = {field.name for field in fields(cls)}
        return cls(
            **{key: value for key, value in data.items() if key in allowed_fields}
        )


def apply_run_context_env(context: RunContext) -> None:
    """Expose run context to subprocess upload helpers."""
    for key, value in context.export_env().items():
        os.environ[key] = value


def metadata_run_fields(context: RunContext) -> dict:
    return {
        "run_context": context.to_dict(),
        "modal_app_name": context.modal_app_name,
        "modal_environment": context.modal_environment,
        "hf_staging_prefix": context.hf_staging_prefix,
    }


def run_dir(run_id: str) -> Path:
    return Path(RUNS_DIR) / run_id


def artifacts_dir(run_id: str) -> Path:
    return Path(artifacts_dir_for_run(run_id))


def _quote_sql_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _canonical_sqlite_value(value):
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    return value


def _canonical_sqlite_sha256(path: Path) -> str:
    """Hash logical SQLite contents instead of mutable file metadata."""
    digest = hashlib.sha256()

    def update(payload) -> None:
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        schema_rows = conn.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        update(
            {
                "schema": [
                    {
                        "type": row["type"],
                        "name": row["name"],
                        "tbl_name": row["tbl_name"],
                        "sql": row["sql"],
                    }
                    for row in schema_rows
                ]
            }
        )

        table_names = [row["name"] for row in schema_rows if row["type"] == "table"]
        for table_name in table_names:
            columns = [
                row["name"]
                for row in conn.execute(
                    f"PRAGMA table_info({_quote_sql_identifier(table_name)})"
                )
            ]
            quoted_columns = [_quote_sql_identifier(column) for column in columns]
            select_columns = ", ".join(quoted_columns)
            order_columns = ", ".join(quoted_columns)
            for row in conn.execute(
                f"""
                SELECT {select_columns}
                FROM {_quote_sql_identifier(table_name)}
                ORDER BY {order_columns}
                """
            ):
                update(
                    {
                        "table": table_name,
                        "row": [
                            _canonical_sqlite_value(row[column]) for column in columns
                        ],
                    }
                )
    return digest.hexdigest()


def artifact_identity(path: str | Path) -> dict:
    artifact = ArtifactReference.from_path(path)
    identity = {
        "path": artifact.path,
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
    }
    if Path(path).suffix == ".db":
        identity["sha256"] = _canonical_sqlite_sha256(Path(path))
        identity.pop("size_bytes", None)
        identity["identity_kind"] = "sqlite_content"
    return identity


def artifact_identities(paths: dict[str, str | Path]) -> dict:
    identities = {}
    for label, path in paths.items():
        artifact_path = Path(path)
        identities[label] = (
            artifact_identity(artifact_path)
            if artifact_path.exists()
            else {"path": str(artifact_path), "missing": True}
        )
    return identities


def collect_diagnostics(run_id: str) -> list[ArtifactReference]:
    return collect_directory_artifacts(
        run_dir(run_id) / "diagnostics",
        patterns=("*.csv", "*.json", "*.txt"),
        role="diagnostic",
    )


def collect_staging_outputs(run_id: str, *, scope: str) -> list[ArtifactReference]:
    scoped_run_dir = Path(STAGING_MOUNT) / run_id
    paths: list[Path] = []
    if scope == "regional":
        for subdir in ("states", "districts", "cities"):
            paths.extend(sorted((scoped_run_dir / subdir).glob("*.h5")))
        manifest_path = scoped_run_dir / "manifest.json"
        if manifest_path.exists():
            paths.append(manifest_path)
    elif scope == "national":
        paths.extend(sorted((scoped_run_dir / "national").glob("*.h5")))
    else:
        raise ValueError(f"Unknown H5 output scope: {scope}")
    return collect_artifacts(paths, missing_ok=True)
