"""Stage 5 release candidate bundle schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import DiagnosticRef
from policyengine_us_data.stage_contracts._coercion import (
    freeze_mapping,
    freeze_sequence,
    jsonable_value,
    optional_string,
    optional_string_value,
    required_string,
    schema_version,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION
from policyengine_us_data.stage_contracts.stages import (
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
)

from .artifacts import ReleaseArtifactSpec
from .context import ReleasePromotionContext
from .diagnostics import (
    normalize_run_contract_path,
    normalize_run_diagnostic_path,
    validation_report_ref_path,
)

RELEASE_CANDIDATE_BUNDLE_TYPE = "release_candidate_input_bundle"


@pipeline_node(
    id="release_candidate_input_bundle",
    label="ReleaseCandidateInputBundle",
    node_type="library",
    description="Typed Stage 5 input bundle describing artifacts eligible for release promotion.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class ReleaseCandidateInputBundle:
    """Typed Stage 5 input bundle describing a candidate ready for promotion."""

    context: ReleasePromotionContext
    artifacts: tuple[ReleaseArtifactSpec, ...]
    source_output_contract_path: str | None = None
    release_candidate_fingerprint: str | None = None
    validation_report_paths: tuple[str, ...] = ()
    validation_report_refs: tuple[DiagnosticRef, ...] = ()
    diagnostics_manifest_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    bundle_type: str = RELEASE_CANDIDATE_BUNDLE_TYPE
    stage_id: str = STAGE_5_VALIDATE_AND_PROMOTE_RELEASE
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        if self.bundle_type != RELEASE_CANDIDATE_BUNDLE_TYPE:
            raise ValueError(f"bundle_type must be {RELEASE_CANDIDATE_BUNDLE_TYPE!r}")
        if self.stage_id != STAGE_5_VALIDATE_AND_PROMOTE_RELEASE:
            raise ValueError(
                f"stage_id must be {STAGE_5_VALIDATE_AND_PROMOTE_RELEASE!r}"
            )
        if not isinstance(self.context, ReleasePromotionContext):
            raise ValueError("context must be ReleasePromotionContext")
        object.__setattr__(
            self,
            "artifacts",
            freeze_sequence(self.artifacts, "artifacts", ReleaseArtifactSpec),
        )
        if not self.artifacts:
            raise ValueError("artifacts must include at least one release artifact")
        object.__setattr__(
            self,
            "source_output_contract_path",
            (
                normalize_run_contract_path(
                    self.source_output_contract_path,
                    self.context,
                )
                if self.source_output_contract_path is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "release_candidate_fingerprint",
            optional_string_value(
                self.release_candidate_fingerprint,
                "release_candidate_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "validation_report_paths",
            tuple(
                normalize_run_diagnostic_path(path, self.context)
                for path in self.validation_report_paths
            ),
        )
        validation_report_refs = freeze_sequence(
            self.validation_report_refs,
            "validation_report_refs",
            DiagnosticRef,
        )
        for ref in validation_report_refs:
            validation_report_ref_path(ref, self.context)
        object.__setattr__(
            self,
            "validation_report_refs",
            validation_report_refs,
        )
        object.__setattr__(
            self,
            "diagnostics_manifest_path",
            (
                normalize_run_diagnostic_path(
                    self.diagnostics_manifest_path,
                    self.context,
                )
                if self.diagnostics_manifest_path is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate bundle to JSON-compatible primitives."""

        return {
            "bundle_type": self.bundle_type,
            "stage_id": self.stage_id,
            "schema_version": self.schema_version,
            "context": self.context.to_dict(),
            "source_output_contract_path": self.source_output_contract_path,
            "release_candidate_fingerprint": self.release_candidate_fingerprint,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "validation_report_paths": list(self.validation_report_paths),
            "validation_report_refs": [
                ref.to_dict() for ref in self.validation_report_refs
            ],
            "diagnostics_manifest_path": self.diagnostics_manifest_path,
            "metadata": jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseCandidateInputBundle":
        """Restore a release candidate bundle from serialized data."""

        return cls(
            context=ReleasePromotionContext.from_dict(data["context"]),
            source_output_contract_path=optional_string(
                data,
                "source_output_contract_path",
            ),
            release_candidate_fingerprint=optional_string(
                data,
                "release_candidate_fingerprint",
            ),
            artifacts=tuple(
                ReleaseArtifactSpec.from_dict(item)
                for item in data.get("artifacts", ())
            ),
            validation_report_paths=tuple(
                required_string({"path": item}, "path")
                for item in data.get("validation_report_paths", ())
            ),
            validation_report_refs=tuple(
                DiagnosticRef.from_dict(item)
                for item in data.get("validation_report_refs", ())
            ),
            diagnostics_manifest_path=optional_string(
                data,
                "diagnostics_manifest_path",
            ),
            metadata=data.get("metadata", {}),
            bundle_type=data.get("bundle_type", RELEASE_CANDIDATE_BUNDLE_TYPE),
            stage_id=data.get("stage_id", STAGE_5_VALIDATE_AND_PROMOTE_RELEASE),
            schema_version=schema_version(data),
        )


from .candidate_builders import build_legacy_release_candidate_bundle  # noqa: E402
from .stage4_reader import (  # noqa: E402
    build_release_candidate_bundle_from_stage4_contract,
    read_stage4_release_candidate_bundle,
)

__all__ = [
    "RELEASE_CANDIDATE_BUNDLE_TYPE",
    "ReleaseCandidateInputBundle",
    "build_legacy_release_candidate_bundle",
    "build_release_candidate_bundle_from_stage4_contract",
    "read_stage4_release_candidate_bundle",
]
