"""Published artifact index rows for Stage 5 release promotion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import ArtifactRef
from policyengine_us_data.stage_contracts._coercion import (
    freeze_mapping,
    jsonable_value,
    mapping_value,
    optional_int_value,
    optional_string,
    optional_string_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_optional_int,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION
from policyengine_us_data.stage_contracts.stages import (
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
)
from policyengine_us_data.utils.canonical_json import canonical_json_dumps

from .candidate import ReleaseCandidateInputBundle
from .context import ReleasePromotionContext
from .results import FullPromotionResult

PUBLISHED_ARTIFACT_INDEX_FILENAME = "published_artifact_index.jsonl"
PUBLISHED_ARTIFACT_INDEX_MEDIA_TYPE = "application/jsonl"


def published_artifact_index_repo_path(run_id: str) -> str:
    """Return the run-scoped repository path for the published artifact index."""

    return f"calibration/runs/{run_id}/diagnostics/{PUBLISHED_ARTIFACT_INDEX_FILENAME}"


def published_artifact_index_path(run_dir: str | Path) -> Path:
    """Return the run-local diagnostics path for the published artifact index."""

    return Path(run_dir) / "diagnostics" / PUBLISHED_ARTIFACT_INDEX_FILENAME


def published_artifact_index_artifact_ref(
    context: ReleasePromotionContext,
    *,
    row_count: int | None = None,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> ArtifactRef:
    """Return a stage-contract reference to the published artifact index."""

    metadata: dict[str, Any] = {
        "artifact_family": "published_artifact_index",
        "source_stage_id": STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
        "relative_path": published_artifact_index_repo_path(context.run_id),
    }
    if row_count is not None:
        metadata["row_count"] = row_count
    return ArtifactRef(
        logical_name="published_artifact_index",
        uri=(
            f"hf://{context.hf_repo_name}/"
            f"{published_artifact_index_repo_path(context.run_id)}"
        ),
        sha256=sha256,
        size_bytes=size_bytes,
        media_type=PUBLISHED_ARTIFACT_INDEX_MEDIA_TYPE,
        metadata=metadata,
    )


@pipeline_node(
    id="published_artifact_index_row",
    label="PublishedArtifactIndexRow",
    node_type="library",
    description="One published HF/GCS artifact row emitted by Stage 5.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    artifacts_in=["release candidate bundle", "release manifest"],
    artifacts_out=["published_artifact_index.jsonl"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_published_index.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class PublishedArtifactIndexRow:
    """One row in the Stage 5 published artifact JSONL index."""

    run_id: str
    candidate_version: str
    release_version: str
    logical_name: str
    relative_path: str
    artifact_role: str
    artifact_family: str
    source_stage_id: str
    hf_uri: str
    gcs_uri: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    area_type: str | None = None
    area_id: str | None = None
    release_manifest_key: str | None = None
    release_manifest_revision: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        for field_name in (
            "run_id",
            "candidate_version",
            "release_version",
            "logical_name",
            "relative_path",
            "artifact_role",
            "artifact_family",
            "source_stage_id",
            "hf_uri",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "gcs_uri",
            optional_string_value(self.gcs_uri, "gcs_uri"),
        )
        object.__setattr__(
            self,
            "sha256",
            optional_string_value(self.sha256, "sha256"),
        )
        validate_optional_int(self.size_bytes, "size_bytes")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        object.__setattr__(
            self,
            "area_type",
            optional_string_value(self.area_type, "area_type"),
        )
        object.__setattr__(
            self,
            "area_id",
            optional_string_value(self.area_id, "area_id"),
        )
        object.__setattr__(
            self,
            "release_manifest_key",
            optional_string_value(
                self.release_manifest_key,
                "release_manifest_key",
            ),
        )
        object.__setattr__(
            self,
            "release_manifest_revision",
            optional_string_value(
                self.release_manifest_revision,
                "release_manifest_revision",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the row to JSON-compatible primitives."""

        return {
            "run_id": self.run_id,
            "candidate_version": self.candidate_version,
            "release_version": self.release_version,
            "logical_name": self.logical_name,
            "relative_path": self.relative_path,
            "artifact_role": self.artifact_role,
            "artifact_family": self.artifact_family,
            "source_stage_id": self.source_stage_id,
            "hf_uri": self.hf_uri,
            "gcs_uri": self.gcs_uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "area_type": self.area_type,
            "area_id": self.area_id,
            "release_manifest_key": self.release_manifest_key,
            "release_manifest_revision": self.release_manifest_revision,
            "metadata": jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PublishedArtifactIndexRow":
        """Restore an index row from serialized data."""

        return cls(
            run_id=required_string(data, "run_id"),
            candidate_version=required_string(data, "candidate_version"),
            release_version=required_string(data, "release_version"),
            logical_name=required_string(data, "logical_name"),
            relative_path=required_string(data, "relative_path"),
            artifact_role=required_string(data, "artifact_role"),
            artifact_family=required_string(data, "artifact_family"),
            source_stage_id=required_string(data, "source_stage_id"),
            hf_uri=required_string(data, "hf_uri"),
            gcs_uri=optional_string(data, "gcs_uri"),
            sha256=optional_string(data, "sha256"),
            size_bytes=optional_int_value(data, "size_bytes"),
            area_type=optional_string(data, "area_type"),
            area_id=optional_string(data, "area_id"),
            release_manifest_key=optional_string(data, "release_manifest_key"),
            release_manifest_revision=optional_string(
                data,
                "release_manifest_revision",
            ),
            metadata=mapping_value(data, "metadata"),
            schema_version=schema_version(data),
        )


@pipeline_node(
    id="published_artifact_index_builder",
    label="Published Artifact Index Builder",
    node_type="library",
    description="Build the Stage 5 published artifact JSONL index.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    artifacts_in=["release candidate bundle", "typed promotion result"],
    artifacts_out=["published_artifact_index.jsonl"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_published_index.py"
    ],
)
def build_published_artifact_index(
    *,
    candidate_bundle: ReleaseCandidateInputBundle,
    promotion_result: FullPromotionResult,
    release_manifest: Mapping[str, Any] | None = None,
    diagnostic_artifacts: Sequence[ArtifactRef] = (),
) -> tuple[PublishedArtifactIndexRow, ...]:
    """Build deterministic published artifact rows for a promoted release."""

    _validate_result_matches_candidate(promotion_result, candidate_bundle)
    context = candidate_bundle.context
    manifest_artifacts = _release_manifest_artifacts(release_manifest)
    rows = [
        *(
            _candidate_artifact_row(
                context=context,
                artifact=artifact,
                manifest_artifacts=manifest_artifacts,
            )
            for artifact in candidate_bundle.artifacts
        ),
        *_release_metadata_rows(context, promotion_result),
        *(
            _diagnostic_artifact_row(context=context, artifact=artifact)
            for artifact in sorted(
                diagnostic_artifacts,
                key=lambda item: item.logical_name,
            )
        ),
    ]
    return tuple(sorted(rows, key=lambda row: (row.artifact_role, row.relative_path)))


def published_artifact_index_to_jsonl(
    rows: Sequence[PublishedArtifactIndexRow],
) -> str:
    """Serialize published artifact rows to deterministic JSONL."""

    return "".join(
        canonical_json_dumps(
            row.to_dict(),
            compact=True,
            trailing_newline=False,
        )
        + "\n"
        for row in rows
    )


def published_artifact_index_from_jsonl(
    payload: str,
) -> tuple[PublishedArtifactIndexRow, ...]:
    """Read published artifact rows from JSONL text."""

    import json

    return tuple(
        PublishedArtifactIndexRow.from_dict(json.loads(line))
        for line in payload.splitlines()
        if line.strip()
    )


def write_published_artifact_index(
    rows: Sequence[PublishedArtifactIndexRow],
    path: str | Path,
) -> tuple[PublishedArtifactIndexRow, ...]:
    """Write published artifact rows to an explicit JSONL path."""

    frozen_rows = tuple(rows)
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        published_artifact_index_to_jsonl(frozen_rows),
        encoding="utf-8",
    )
    return frozen_rows


def read_published_artifact_index(
    path: str | Path,
) -> tuple[PublishedArtifactIndexRow, ...]:
    """Read published artifact rows from a JSONL path."""

    return published_artifact_index_from_jsonl(Path(path).read_text(encoding="utf-8"))


def _validate_result_matches_candidate(
    result: FullPromotionResult,
    candidate_bundle: ReleaseCandidateInputBundle,
) -> None:
    context = candidate_bundle.context
    if result.run_id != context.run_id:
        raise ValueError("promotion_result.run_id must match context.run_id")
    if result.candidate_version != context.candidate_version:
        raise ValueError(
            "promotion_result.candidate_version must match context.candidate_version"
        )
    if result.release_version != context.release_version:
        raise ValueError(
            "promotion_result.release_version must match context.release_version"
        )
    if result.artifact_count != len(candidate_bundle.artifacts):
        raise ValueError(
            "promotion_result.artifact_count must match candidate artifacts"
        )


def _release_manifest_artifacts(
    release_manifest: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if not release_manifest:
        return {}
    artifacts = release_manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return {}
    return {
        str(key): artifact
        for key, artifact in artifacts.items()
        if isinstance(artifact, Mapping)
    }


def _candidate_artifact_row(
    *,
    context: ReleasePromotionContext,
    artifact,
    manifest_artifacts: Mapping[str, Mapping[str, Any]],
) -> PublishedArtifactIndexRow:
    manifest_key = _release_manifest_key(artifact.relative_path)
    manifest_entry = manifest_artifacts.get(manifest_key, {})
    relative_path = str(manifest_entry.get("path") or artifact.relative_path)
    manifest_size_bytes = _optional_manifest_int(manifest_entry, "size_bytes")
    return PublishedArtifactIndexRow(
        run_id=context.run_id,
        candidate_version=context.candidate_version,
        release_version=context.release_version,
        logical_name=artifact.logical_name,
        relative_path=relative_path,
        artifact_role="release_artifact",
        artifact_family=artifact.artifact_family,
        source_stage_id=artifact.source_stage_id,
        hf_uri=f"hf://{context.hf_repo_name}/{relative_path}",
        gcs_uri=f"gs://{context.gcs_bucket_name}/{relative_path}",
        sha256=_optional_manifest_string(manifest_entry, "sha256") or artifact.sha256,
        size_bytes=manifest_size_bytes
        if manifest_size_bytes is not None
        else artifact.size_bytes,
        area_type=artifact.area_type,
        area_id=artifact.area_id,
        release_manifest_key=manifest_key,
        release_manifest_revision=_optional_manifest_string(
            manifest_entry,
            "revision",
        ),
        metadata={
            "release_manifest_kind": _optional_manifest_string(
                manifest_entry,
                "kind",
            ),
            "candidate_metadata": jsonable_value(artifact.metadata),
        },
    )


def _release_metadata_rows(
    context: ReleasePromotionContext,
    result: FullPromotionResult,
) -> tuple[PublishedArtifactIndexRow, ...]:
    artifacts = (
        (
            "release_manifest",
            result.release_manifest.root_path,
            "release_manifest",
            "application/json",
            result.release_manifest.manifest_sha256,
        ),
        (
            "versioned_release_manifest",
            result.release_manifest.versioned_path,
            "release_manifest",
            "application/json",
            result.release_manifest.manifest_sha256,
        ),
        (
            "trace_tro",
            result.release_manifest.trace_tro_path,
            "trace_tro",
            "application/ld+json",
            None,
        ),
        (
            "versioned_trace_tro",
            result.release_manifest.versioned_trace_tro_path,
            "trace_tro",
            "application/ld+json",
            None,
        ),
        (
            "version_manifest",
            result.version_manifest.path,
            "version_manifest",
            "application/json",
            None,
        ),
        (
            "release_completion_marker",
            result.completion_marker.marker_path,
            "release_completion_marker",
            "application/json",
            None,
        ),
    )
    return tuple(
        PublishedArtifactIndexRow(
            run_id=context.run_id,
            candidate_version=context.candidate_version,
            release_version=context.release_version,
            logical_name=logical_name,
            relative_path=relative_path,
            artifact_role="release_metadata",
            artifact_family=artifact_family,
            source_stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
            hf_uri=f"hf://{context.hf_repo_name}/{relative_path}",
            sha256=sha256,
            metadata={"media_type": media_type},
        )
        for logical_name, relative_path, artifact_family, media_type, sha256 in artifacts
    )


def _diagnostic_artifact_row(
    *,
    context: ReleasePromotionContext,
    artifact: ArtifactRef,
) -> PublishedArtifactIndexRow:
    relative_path = _artifact_relative_path(artifact, context)
    return PublishedArtifactIndexRow(
        run_id=context.run_id,
        candidate_version=context.candidate_version,
        release_version=context.release_version,
        logical_name=artifact.logical_name,
        relative_path=relative_path,
        artifact_role="diagnostic",
        artifact_family=str(artifact.metadata.get("artifact_family") or "diagnostic"),
        source_stage_id=str(
            artifact.metadata.get("source_stage_id")
            or STAGE_5_VALIDATE_AND_PROMOTE_RELEASE
        ),
        hf_uri=artifact.uri,
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
        metadata={
            key: value
            for key, value in jsonable_value(artifact.metadata).items()
            if key not in {"artifact_family", "source_stage_id"}
        },
    )


def _artifact_relative_path(
    artifact: ArtifactRef,
    context: ReleasePromotionContext,
) -> str:
    metadata_path = artifact.metadata.get("relative_path")
    if isinstance(metadata_path, str) and metadata_path:
        return metadata_path
    hf_prefix = f"hf://{context.hf_repo_name}/"
    if artifact.uri.startswith(hf_prefix):
        return artifact.uri[len(hf_prefix) :]
    return artifact.uri


def _release_manifest_key(relative_path: str) -> str:
    return PurePosixPath(relative_path).with_suffix("").as_posix()


def _optional_manifest_string(
    manifest_entry: Mapping[str, Any],
    key: str,
) -> str | None:
    value = manifest_entry.get(key)
    return value if isinstance(value, str) and value else None


def _optional_manifest_int(
    manifest_entry: Mapping[str, Any],
    key: str,
) -> int | None:
    value = manifest_entry.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None
