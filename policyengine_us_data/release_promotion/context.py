"""Typed identity for Stage 5 release promotion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts._coercion import (
    freeze_mapping,
    jsonable_value,
    mapping_value,
    optional_string,
    optional_string_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION
from policyengine_us_data.utils.run_context import (
    normalize_release_bump,
    sanitize_run_id,
    sanitize_staging_version,
    stable_release_version,
    staging_prefix,
)


@pipeline_node(
    id="release_promotion_context",
    label="ReleasePromotionContext",
    node_type="library",
    description="Typed Stage 5 run, candidate, release, and destination identity.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class ReleasePromotionContext:
    """Canonical run, candidate, release, and destination identity for Stage 5."""

    run_id: str
    candidate_version: str
    release_version: str
    hf_repo_name: str
    gcs_bucket_name: str
    hf_repo_type: str = "model"
    base_release_version: str | None = None
    release_bump: str | None = None
    modal_app_name: str | None = None
    modal_environment: str | None = None
    hf_staging_prefix: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(self, "run_id", sanitize_run_id(self.run_id))
        object.__setattr__(
            self,
            "candidate_version",
            sanitize_staging_version(self.candidate_version),
        )
        object.__setattr__(
            self,
            "release_version",
            stable_release_version(self.release_version),
        )
        object.__setattr__(
            self,
            "hf_repo_name",
            require_non_empty(self.hf_repo_name, "hf_repo_name"),
        )
        object.__setattr__(
            self,
            "hf_repo_type",
            require_non_empty(self.hf_repo_type, "hf_repo_type"),
        )
        object.__setattr__(
            self,
            "gcs_bucket_name",
            require_non_empty(self.gcs_bucket_name, "gcs_bucket_name"),
        )
        object.__setattr__(
            self,
            "base_release_version",
            (
                stable_release_version(self.base_release_version)
                if self.base_release_version is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "release_bump",
            (
                normalize_release_bump(self.release_bump)
                if self.release_bump is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "modal_app_name",
            optional_string_value(self.modal_app_name, "modal_app_name"),
        )
        object.__setattr__(
            self,
            "modal_environment",
            optional_string_value(self.modal_environment, "modal_environment"),
        )
        derived_prefix = staging_prefix(
            self.run_id,
            candidate_version=self.candidate_version,
        )
        prefix = self.hf_staging_prefix or derived_prefix
        if prefix != derived_prefix:
            raise ValueError(
                "hf_staging_prefix must match run_id and candidate_version: "
                f"{derived_prefix!r}"
            )
        object.__setattr__(
            self,
            "hf_staging_prefix",
            require_non_empty(prefix, "hf_staging_prefix"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata or {}, "metadata"),
        )

    @property
    def candidate_scope(self) -> str:
        """Return the candidate staging scope used for run-scoped HF paths."""

        return self.candidate_version

    def to_dict(self) -> dict[str, Any]:
        """Serialize the context to JSON-compatible primitives."""

        return {
            "run_id": self.run_id,
            "candidate_version": self.candidate_version,
            "release_version": self.release_version,
            "hf_repo_name": self.hf_repo_name,
            "hf_repo_type": self.hf_repo_type,
            "gcs_bucket_name": self.gcs_bucket_name,
            "base_release_version": self.base_release_version,
            "release_bump": self.release_bump,
            "modal_app_name": self.modal_app_name,
            "modal_environment": self.modal_environment,
            "hf_staging_prefix": self.hf_staging_prefix,
            "metadata": jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleasePromotionContext":
        """Restore a release promotion context from serialized data."""

        return cls(
            run_id=required_string(data, "run_id"),
            candidate_version=required_string(data, "candidate_version"),
            release_version=required_string(data, "release_version"),
            hf_repo_name=required_string(data, "hf_repo_name"),
            hf_repo_type=data.get("hf_repo_type", "model"),
            gcs_bucket_name=required_string(data, "gcs_bucket_name"),
            base_release_version=optional_string(data, "base_release_version"),
            release_bump=optional_string(data, "release_bump"),
            modal_app_name=optional_string(data, "modal_app_name"),
            modal_environment=optional_string(data, "modal_environment"),
            hf_staging_prefix=optional_string(data, "hf_staging_prefix"),
            metadata=mapping_value(data, "metadata"),
            schema_version=schema_version(data),
        )
