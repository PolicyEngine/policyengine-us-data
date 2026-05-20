"""Validation target catalog for Stage 1 dataset-build artifacts."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node

from .artifacts import stage_1_contract_artifact_specs


@dataclass(frozen=True, kw_only=True)
class ValidationTarget:
    """One logical artifact expectation for Stage 1 validation."""

    target_id: str
    substage_id: str
    logical_name: str
    required: bool = True
    warning_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.target_id, "target_id"),
            (self.substage_id, "substage_id"),
            (self.logical_name, "logical_name"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible target payload."""

        return {
            "target_id": self.target_id,
            "substage_id": self.substage_id,
            "logical_name": self.logical_name,
            "required": self.required,
            "warning_only": self.warning_only,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationTarget":
        """Build a validation target from a mapping."""

        return cls(
            target_id=str(data["target_id"]),
            substage_id=str(data["substage_id"]),
            logical_name=str(data["logical_name"]),
            required=bool(data.get("required", True)),
            warning_only=bool(data.get("warning_only", False)),
            metadata=dict(data.get("metadata", {})),
        )


@pipeline_node(
    id="stage_1_validation_target_catalog",
    label="Stage 1 Validation Target Catalog",
    node_type="library",
    description="Deterministic catalog of active Stage 1 validation artifact targets.",
    source_file="policyengine_us_data/build_datasets/validation_targets.py",
    status="current",
    stability="stable",
    pathways=["data_build", "stage_contracts", "cross_stage_validation"],
    validation_commands=["uv run pytest tests/unit/test_build_dataset_validation.py"],
)
@dataclass(frozen=True, kw_only=True)
class ValidationTargetCatalog:
    """Deterministic lookup for active Stage 1 validation targets."""

    targets: tuple[ValidationTarget, ...]

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        seen: set[str] = set()
        for target in targets:
            if not isinstance(target, ValidationTarget):
                raise TypeError("targets must contain ValidationTarget instances")
            if target.target_id in seen:
                raise ValueError(f"Duplicate validation target: {target.target_id}")
            seen.add(target.target_id)
        object.__setattr__(
            self,
            "targets",
            tuple(sorted(targets, key=lambda item: item.target_id)),
        )

    @classmethod
    def from_stage_1_specs(
        cls,
        *,
        skip_enhanced_cps: bool = False,
        skip_stage_5: bool = False,
    ) -> "ValidationTargetCatalog":
        """Build the active target catalog from Stage 1 artifact specs."""

        targets: list[ValidationTarget] = []
        for spec in stage_1_contract_artifact_specs():
            if skip_enhanced_cps and spec.skip_when_enhanced_cps_skipped:
                continue
            if skip_stage_5 and spec.skip_when_stage_5_skipped:
                continue
            targets.append(
                ValidationTarget(
                    target_id=f"{spec.substage_id}.{spec.logical_name}",
                    substage_id=spec.substage_id,
                    logical_name=spec.logical_name,
                    required=spec.required,
                    metadata={
                        "artifact_family": spec.artifact_family,
                        "filename": spec.filename,
                        "period": spec.period,
                    },
                )
            )
        return cls(targets=tuple(targets))

    @classmethod
    def load(cls, path: str | Path) -> "ValidationTargetCatalog":
        """Load a target catalog from JSON, CSV, or SQLite."""

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            rows = _load_sqlite_targets(path)
        else:
            raise ValueError(f"Unsupported validation target catalog: {path}")
        return cls.from_rows(rows)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
    ) -> "ValidationTargetCatalog":
        """Build a catalog from row dictionaries."""

        return cls(targets=tuple(ValidationTarget.from_dict(row) for row in rows))

    def active_for_substage(self, substage_id: str) -> tuple[ValidationTarget, ...]:
        """Return active targets for one Stage 1 substage."""

        return tuple(
            target for target in self.targets if target.substage_id == substage_id
        )

    def required_logical_names(self, substage_id: str) -> tuple[str, ...]:
        """Return required logical artifacts for one substage."""

        return tuple(
            target.logical_name
            for target in self.active_for_substage(substage_id)
            if target.required
        )


def _load_sqlite_targets(path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT target_id, substage_id, logical_name, required, warning_only
            FROM validation_targets
            ORDER BY target_id
            """
        ).fetchall()
    return [
        {
            "target_id": target_id,
            "substage_id": substage_id,
            "logical_name": logical_name,
            "required": bool(required),
            "warning_only": bool(warning_only),
        }
        for target_id, substage_id, logical_name, required, warning_only in rows
    ]


__all__ = [
    "ValidationTarget",
    "ValidationTargetCatalog",
]
