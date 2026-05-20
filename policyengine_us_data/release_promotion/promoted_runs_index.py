"""Run-oriented promoted release discovery index for Stage 5."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import ArtifactRef
from policyengine_us_data.stage_contracts._coercion import (
    freeze_mapping,
    jsonable_value,
    mapping_value,
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
from policyengine_us_data.utils.canonical_json import (
    canonical_json_dumps,
    canonical_json_loads,
)

from .context import ReleasePromotionContext
from .results import FullPromotionResult

PROMOTED_RUNS_INDEX_FILENAME = "index.json"
PROMOTED_RUNS_INDEX_MEDIA_TYPE = "application/json"


def promoted_runs_index_repo_path() -> str:
    """Return the repository path for the promoted runs discovery index."""

    return f"calibration/runs/{PROMOTED_RUNS_INDEX_FILENAME}"


def promoted_runs_index_path(runs_dir: str | Path) -> Path:
    """Return the pipeline-volume path for the promoted runs discovery index."""

    return Path(runs_dir) / PROMOTED_RUNS_INDEX_FILENAME


def promoted_runs_index_artifact_ref(
    context: ReleasePromotionContext,
    update: "PromotedRunsIndexUpdate",
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> ArtifactRef:
    """Return a stage-contract reference to the promoted runs index."""

    return ArtifactRef(
        logical_name="promoted_runs_index",
        uri=f"hf://{context.hf_repo_name}/{promoted_runs_index_repo_path()}",
        sha256=sha256,
        size_bytes=size_bytes,
        media_type=PROMOTED_RUNS_INDEX_MEDIA_TYPE,
        metadata={
            "artifact_family": "promoted_runs_index",
            "source_stage_id": STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
            "relative_path": promoted_runs_index_repo_path(),
            "run_id": update.run_id,
            "release_version": update.release_version,
            "update_status": update.status,
            "run_count": update.run_count,
            "release_version_run_count": update.release_version_run_count,
            "already_finalized": update.already_finalized,
        },
    )


@pipeline_node(
    id="promoted_run_index_entry",
    label="PromotedRunIndexEntry",
    node_type="library",
    description="One run-oriented discovery entry emitted by Stage 5 promotion.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    artifacts_in=["typed promotion result", "release promotion contract"],
    artifacts_out=["promoted_runs_index.json"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_promoted_runs_index.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class PromotedRunIndexEntry:
    """One promoted run entry keyed by canonical run ID."""

    run_id: str
    candidate_version: str
    release_version: str
    status: str
    promoted_at: str
    updated_at: str
    artifact_count: int
    hf_promoted_count: int
    gcs_uploaded_count: int
    release_manifest_artifacts: int
    version_manifest_updated: bool
    completion_marker_path: str
    already_finalized: bool = False
    release_promotion_contract_path: str | None = None
    published_artifact_index_path: str | None = None
    run_manifest_path: str | None = None
    step_manifest_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        for field_name in (
            "run_id",
            "candidate_version",
            "release_version",
            "status",
            "promoted_at",
            "updated_at",
            "completion_marker_path",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        for field_name in (
            "artifact_count",
            "hf_promoted_count",
            "gcs_uploaded_count",
            "release_manifest_artifacts",
        ):
            _nonnegative_int_value(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "version_manifest_updated",
            _bool_value(self.version_manifest_updated, "version_manifest_updated"),
        )
        object.__setattr__(
            self,
            "already_finalized",
            _bool_value(self.already_finalized, "already_finalized"),
        )
        for field_name in (
            "release_promotion_contract_path",
            "published_artifact_index_path",
            "run_manifest_path",
            "step_manifest_path",
        ):
            object.__setattr__(
                self,
                field_name,
                optional_string_value(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the promoted run entry to JSON-compatible primitives."""

        return {
            "run_id": self.run_id,
            "candidate_version": self.candidate_version,
            "release_version": self.release_version,
            "status": self.status,
            "promoted_at": self.promoted_at,
            "updated_at": self.updated_at,
            "artifact_count": self.artifact_count,
            "hf_promoted_count": self.hf_promoted_count,
            "gcs_uploaded_count": self.gcs_uploaded_count,
            "release_manifest_artifacts": self.release_manifest_artifacts,
            "version_manifest_updated": self.version_manifest_updated,
            "completion_marker_path": self.completion_marker_path,
            "already_finalized": self.already_finalized,
            "release_promotion_contract_path": self.release_promotion_contract_path,
            "published_artifact_index_path": self.published_artifact_index_path,
            "run_manifest_path": self.run_manifest_path,
            "step_manifest_path": self.step_manifest_path,
            "metadata": jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromotedRunIndexEntry":
        """Restore a promoted run entry from serialized data."""

        return cls(
            run_id=required_string(data, "run_id"),
            candidate_version=required_string(data, "candidate_version"),
            release_version=required_string(data, "release_version"),
            status=required_string(data, "status"),
            promoted_at=required_string(data, "promoted_at"),
            updated_at=required_string(data, "updated_at"),
            artifact_count=_nonnegative_int(data, "artifact_count"),
            hf_promoted_count=_nonnegative_int(data, "hf_promoted_count"),
            gcs_uploaded_count=_nonnegative_int(data, "gcs_uploaded_count"),
            release_manifest_artifacts=_nonnegative_int(
                data,
                "release_manifest_artifacts",
            ),
            version_manifest_updated=_bool_field(data, "version_manifest_updated"),
            completion_marker_path=required_string(data, "completion_marker_path"),
            already_finalized=_bool_field(data, "already_finalized", default=False),
            release_promotion_contract_path=optional_string(
                data,
                "release_promotion_contract_path",
            ),
            published_artifact_index_path=optional_string(
                data,
                "published_artifact_index_path",
            ),
            run_manifest_path=optional_string(data, "run_manifest_path"),
            step_manifest_path=optional_string(data, "step_manifest_path"),
            metadata=mapping_value(data, "metadata"),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class PromotedReleaseVersionEntry:
    """Release-version lookup entry inside the promoted runs index."""

    release_version: str
    latest_run_id: str
    run_ids: tuple[str, ...]
    updated_at: str
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "release_version",
            require_non_empty(self.release_version, "release_version"),
        )
        object.__setattr__(
            self,
            "latest_run_id",
            require_non_empty(self.latest_run_id, "latest_run_id"),
        )
        run_ids = _string_tuple(self.run_ids, "run_ids")
        if not run_ids:
            raise ValueError("run_ids must not be empty")
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("run_ids must not contain duplicates")
        if self.latest_run_id not in run_ids:
            raise ValueError("latest_run_id must be present in run_ids")
        object.__setattr__(self, "run_ids", run_ids)
        object.__setattr__(
            self,
            "updated_at",
            require_non_empty(self.updated_at, "updated_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the release-version lookup entry."""

        return {
            "release_version": self.release_version,
            "latest_run_id": self.latest_run_id,
            "run_ids": list(self.run_ids),
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "PromotedReleaseVersionEntry":
        """Restore a release-version lookup entry."""

        return cls(
            release_version=required_string(data, "release_version"),
            latest_run_id=required_string(data, "latest_run_id"),
            run_ids=_string_tuple(data.get("run_ids", ()), "run_ids"),
            updated_at=required_string(data, "updated_at"),
            schema_version=schema_version(data),
        )


@pipeline_node(
    id="promoted_runs_index",
    label="PromotedRunsIndex",
    node_type="library",
    description="Run-oriented discovery index for promoted Stage 5 releases.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    artifacts_in=["typed promotion result", "release promotion contract"],
    artifacts_out=["calibration/runs/index.json"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_promoted_runs_index.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class PromotedRunsIndex:
    """Promoted run discovery index keyed by canonical run ID."""

    updated_at: str
    runs: Mapping[str, PromotedRunIndexEntry] = field(default_factory=dict)
    release_versions: Mapping[str, PromotedReleaseVersionEntry] = field(
        default_factory=dict
    )
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "updated_at",
            require_non_empty(self.updated_at, "updated_at"),
        )
        runs = _coerce_runs(self.runs)
        release_versions = _coerce_release_versions(self.release_versions)
        _validate_release_version_entries(runs, release_versions)
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "release_versions", release_versions)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the promoted runs index."""

        return {
            "updated_at": self.updated_at,
            "runs": {
                run_id: self.runs[run_id].to_dict() for run_id in sorted(self.runs)
            },
            "release_versions": {
                release_version: self.release_versions[release_version].to_dict()
                for release_version in sorted(self.release_versions)
            },
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromotedRunsIndex":
        """Restore the promoted runs index from serialized data."""

        return cls(
            updated_at=required_string(data, "updated_at"),
            runs=mapping_value(data, "runs"),
            release_versions=mapping_value(data, "release_versions"),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class PromotedRunsIndexUpdate:
    """Status for one promoted runs index upsert."""

    status: str
    run_id: str
    release_version: str
    run_count: int
    release_version_run_count: int
    already_finalized: bool
    updated_at: str
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        if self.status not in {"created", "updated"}:
            raise ValueError("status must be 'created' or 'updated'")
        for field_name in ("run_id", "release_version", "updated_at"):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(getattr(self, field_name), field_name),
            )
        for field_name in ("run_count", "release_version_run_count"):
            _nonnegative_int_value(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "already_finalized",
            _bool_value(self.already_finalized, "already_finalized"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the index update status."""

        return {
            "status": self.status,
            "run_id": self.run_id,
            "release_version": self.release_version,
            "run_count": self.run_count,
            "release_version_run_count": self.release_version_run_count,
            "already_finalized": self.already_finalized,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }


def empty_promoted_runs_index(updated_at: str) -> PromotedRunsIndex:
    """Return an empty promoted runs index."""

    return PromotedRunsIndex(updated_at=updated_at)


def build_promoted_run_index_entry(
    *,
    context: ReleasePromotionContext,
    promotion_result: FullPromotionResult,
    promoted_at: str,
    release_promotion_contract_path: str | None = None,
    published_artifact_index_path: str | None = None,
    run_manifest_path: str | None = None,
    step_manifest_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PromotedRunIndexEntry:
    """Build the run discovery entry for a successful Stage 5 promotion."""

    _validate_result_matches_context(promotion_result, context)
    completion_marker = (
        promotion_result.completion_marker.marker_path
        or f"releases/{context.release_version}/release-complete.json"
    )
    return PromotedRunIndexEntry(
        run_id=context.run_id,
        candidate_version=context.candidate_version,
        release_version=context.release_version,
        status="promoted",
        promoted_at=promoted_at,
        updated_at=promoted_at,
        artifact_count=promotion_result.artifact_count,
        hf_promoted_count=promotion_result.hf.promoted_count,
        gcs_uploaded_count=promotion_result.gcs.uploaded_count,
        release_manifest_artifacts=promotion_result.release_manifest.artifact_count,
        version_manifest_updated=promotion_result.version_manifest.updated,
        completion_marker_path=completion_marker,
        already_finalized=promotion_result.already_finalized,
        release_promotion_contract_path=release_promotion_contract_path,
        published_artifact_index_path=published_artifact_index_path,
        run_manifest_path=run_manifest_path,
        step_manifest_path=step_manifest_path,
        metadata=metadata or {},
    )


def upsert_promoted_run(
    index: PromotedRunsIndex,
    entry: PromotedRunIndexEntry,
    *,
    updated_at: str,
) -> tuple[PromotedRunsIndex, PromotedRunsIndexUpdate]:
    """Upsert one promoted run without duplicating run or version entries."""

    existed = entry.run_id in index.runs
    runs = dict(index.runs)
    runs[entry.run_id] = entry

    release_versions = _release_versions_without_run(index, entry.run_id)
    current_version = release_versions.get(entry.release_version)
    if current_version is None:
        run_ids = (entry.run_id,)
    elif entry.run_id in current_version.run_ids:
        run_ids = current_version.run_ids
    else:
        run_ids = (*current_version.run_ids, entry.run_id)
    release_versions[entry.release_version] = PromotedReleaseVersionEntry(
        release_version=entry.release_version,
        latest_run_id=entry.run_id,
        run_ids=run_ids,
        updated_at=updated_at,
    )
    updated = PromotedRunsIndex(
        updated_at=updated_at,
        runs=runs,
        release_versions=release_versions,
    )
    update = PromotedRunsIndexUpdate(
        status="updated" if existed else "created",
        run_id=entry.run_id,
        release_version=entry.release_version,
        run_count=len(updated.runs),
        release_version_run_count=len(
            updated.release_versions[entry.release_version].run_ids
        ),
        already_finalized=entry.already_finalized,
        updated_at=updated_at,
    )
    return updated, update


def promoted_runs_index_to_json(index: PromotedRunsIndex) -> str:
    """Serialize the promoted runs index deterministically."""

    return canonical_json_dumps(index.to_dict())


def promoted_runs_index_from_json(payload: str) -> PromotedRunsIndex:
    """Restore the promoted runs index from JSON text."""

    return PromotedRunsIndex.from_dict(canonical_json_loads(payload))


def read_promoted_runs_index(path: str | Path) -> PromotedRunsIndex:
    """Read a promoted runs index from disk."""

    return promoted_runs_index_from_json(Path(path).read_text(encoding="utf-8"))


def load_promoted_runs_index(
    path: str | Path,
    *,
    updated_at: str,
) -> PromotedRunsIndex:
    """Read a promoted runs index, returning an empty index when absent."""

    index_path = Path(path)
    if not index_path.exists():
        return empty_promoted_runs_index(updated_at)
    return read_promoted_runs_index(index_path)


def write_promoted_runs_index(
    index: PromotedRunsIndex,
    path: str | Path,
) -> PromotedRunsIndex:
    """Write the promoted runs index to disk."""

    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(promoted_runs_index_to_json(index), encoding="utf-8")
    return index


def update_promoted_runs_index(
    *,
    path: str | Path,
    entry: PromotedRunIndexEntry,
    updated_at: str,
) -> tuple[PromotedRunsIndex, PromotedRunsIndexUpdate]:
    """Load, upsert, and persist one promoted run entry."""

    index = load_promoted_runs_index(path, updated_at=updated_at)
    updated, update = upsert_promoted_run(index, entry, updated_at=updated_at)
    write_promoted_runs_index(updated, path)
    return updated, update


def _validate_result_matches_context(
    result: FullPromotionResult,
    context: ReleasePromotionContext,
) -> None:
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


def _release_versions_without_run(
    index: PromotedRunsIndex,
    run_id: str,
) -> dict[str, PromotedReleaseVersionEntry]:
    release_versions: dict[str, PromotedReleaseVersionEntry] = {}
    for release_version, entry in index.release_versions.items():
        run_ids = tuple(item for item in entry.run_ids if item != run_id)
        if not run_ids:
            continue
        latest_run_id = (
            entry.latest_run_id if entry.latest_run_id in run_ids else run_ids[-1]
        )
        release_versions[release_version] = PromotedReleaseVersionEntry(
            release_version=release_version,
            latest_run_id=latest_run_id,
            run_ids=run_ids,
            updated_at=entry.updated_at,
        )
    return release_versions


def _coerce_runs(
    value: Mapping[str, PromotedRunIndexEntry | Mapping[str, Any]],
) -> Mapping[str, PromotedRunIndexEntry]:
    if not isinstance(value, Mapping):
        raise ValueError("runs must be a mapping")
    runs: dict[str, PromotedRunIndexEntry] = {}
    for run_id, entry in value.items():
        if isinstance(entry, PromotedRunIndexEntry):
            runs[str(run_id)] = entry
        elif isinstance(entry, Mapping):
            runs[str(run_id)] = PromotedRunIndexEntry.from_dict(entry)
        else:
            raise ValueError("runs entries must be PromotedRunIndexEntry mappings")
    for run_id, entry in runs.items():
        if run_id != entry.run_id:
            raise ValueError("runs keys must match entry.run_id")
    return freeze_mapping(runs, "runs")


def _coerce_release_versions(
    value: Mapping[str, PromotedReleaseVersionEntry | Mapping[str, Any]],
) -> Mapping[str, PromotedReleaseVersionEntry]:
    if not isinstance(value, Mapping):
        raise ValueError("release_versions must be a mapping")
    release_versions: dict[str, PromotedReleaseVersionEntry] = {}
    for release_version, entry in value.items():
        if isinstance(entry, PromotedReleaseVersionEntry):
            release_versions[str(release_version)] = entry
        elif isinstance(entry, Mapping):
            release_versions[str(release_version)] = (
                PromotedReleaseVersionEntry.from_dict(entry)
            )
        else:
            raise ValueError(
                "release_versions entries must be PromotedReleaseVersionEntry mappings"
            )
    for release_version, entry in release_versions.items():
        if release_version != entry.release_version:
            raise ValueError("release_versions keys must match entry.release_version")
    return freeze_mapping(release_versions, "release_versions")


def _validate_release_version_entries(
    runs: Mapping[str, PromotedRunIndexEntry],
    release_versions: Mapping[str, PromotedReleaseVersionEntry],
) -> None:
    for release_version, entry in release_versions.items():
        for run_id in entry.run_ids:
            run = runs.get(run_id)
            if run is None:
                raise ValueError("release_versions run_ids must exist in runs")
            if run.release_version != release_version:
                raise ValueError(
                    "release_versions run_ids must match their release_version"
                )


def _nonnegative_int(data: Mapping[str, Any], field_name: str) -> int:
    value = data.get(field_name)
    return _nonnegative_int_value(value, field_name)


def _nonnegative_int_value(value: Any, field_name: str) -> int:
    validate_optional_int(value, field_name)
    if value is None:
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _bool_value(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _bool_field(
    data: Mapping[str, Any],
    field_name: str,
    *,
    default: bool | None = None,
) -> bool:
    value = data.get(field_name, default)
    return _bool_value(value, field_name)


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise ValueError(f"{field_name} must be a tuple or list")
    return tuple(require_non_empty(item, field_name) for item in value)
