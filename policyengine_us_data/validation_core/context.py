"""Shared validation execution context and artifact resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import ArtifactRef
from policyengine_us_data.stage_contracts.stages import (
    is_canonical_stage_id,
    is_canonical_substage_id,
)


def _freeze_mapping(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(
        {str(key): _freeze_value(value) for key, value in mapping.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, tuple | list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _required_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _canonical_stage_id(value: str) -> str:
    stage_id = _required_string(value, "stage_id")
    if not is_canonical_stage_id(stage_id):
        raise ValueError(f"Invalid canonical stage_id: {stage_id!r}")
    return stage_id


def _canonical_substage_id(stage_id: str, value: str | None) -> str | None:
    if value is None:
        return None
    substage_id = _required_string(value, "substage_id")
    if not is_canonical_substage_id(stage_id, substage_id):
        raise ValueError(
            f"Invalid canonical substage_id {substage_id!r} for stage_id {stage_id!r}"
        )
    return substage_id


@pipeline_node(
    id="validation_core_artifact_resolver",
    label="ValidationArtifactResolver",
    node_type="library",
    description="Resolve logical validation artifact names to canonical stage-contract artifact references.",
    status="current",
    stability="stable",
    pathways=["cross_stage_validation"],
    validation_commands=["uv run pytest tests/unit/test_validation_core.py"],
)
@dataclass(frozen=True, kw_only=True)
class ValidationArtifactResolver:
    """Resolve logical validation artifact names to stage-contract references."""

    artifacts: Mapping[str, ArtifactRef]

    def __post_init__(self) -> None:
        artifacts = {}
        for logical_name, artifact in self.artifacts.items():
            name = _required_string(logical_name, "artifact logical name")
            if not isinstance(artifact, ArtifactRef):
                raise TypeError(
                    "artifacts must map logical names to ArtifactRef instances"
                )
            if artifact.logical_name != name:
                raise ValueError(
                    f"ArtifactRef logical_name {artifact.logical_name!r} does not "
                    f"match resolver key {name!r}"
                )
            artifacts[name] = artifact
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))

    def require(self, logical_name: str) -> ArtifactRef:
        """Return a required artifact reference or raise ``KeyError``."""

        name = _required_string(logical_name, "logical_name")
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise KeyError(f"Missing required validation artifact: {name}") from exc

    def optional(self, logical_name: str) -> ArtifactRef | None:
        """Return an artifact reference when present, otherwise ``None``."""

        name = _required_string(logical_name, "logical_name")
        return self.artifacts.get(name)


@pipeline_node(
    id="validation_core_context",
    label="ValidationContext",
    node_type="library",
    description="Read-only cross-stage validation context with canonical stage identity.",
    status="current",
    stability="stable",
    pathways=["cross_stage_validation"],
    validation_commands=["uv run pytest tests/unit/test_validation_core.py"],
)
@dataclass(frozen=True, kw_only=True)
class ValidationContext:
    """Read-only context passed to validation checks."""

    run_id: str
    stage_id: str
    substage_id: str | None = None
    resolver: ValidationArtifactResolver
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _required_string(self.run_id, "run_id"))
        stage_id = _canonical_stage_id(self.stage_id)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(
            self,
            "substage_id",
            _canonical_substage_id(stage_id, self.substage_id),
        )
        if not isinstance(self.resolver, ValidationArtifactResolver):
            raise TypeError("resolver must be a ValidationArtifactResolver")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
