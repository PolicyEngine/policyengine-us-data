"""Run-scoped execution manifests for pipeline steps.

Step manifests are execution records: they describe what a pipeline step
read, wrote, reused, invalidated, and failed for one run ID. They are kept
separate from release manifests, which remain the publication contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from policyengine_us_data.utils.canonical_json import (
    canonical_json_dumps as _canonical_json_dumps,
)
from policyengine_us_data.utils.error_redaction import redact_error_text


STEP_MANIFEST_SCHEMA_VERSION = "1"

COMPLETED_STATUSES = frozenset({"completed", "reused", "partially_reused"})
VALID_STATUSES = frozenset(
    {
        "pending",
        "running",
        "completed",
        "failed",
        "reused",
        "partially_reused",
    }
)
VALID_REUSE_DECISIONS = frozenset(
    {
        "computed",
        "reused",
        "partially_reused",
        "invalidated",
        "failed",
        "not_applicable",
    }
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def canonical_json_dumps(payload: Mapping[str, Any]) -> str:
    """Serialize manifest JSON deterministically."""
    return _canonical_json_dumps(payload)


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


def sha256_file(path: Path) -> str:
    """Compute a file SHA-256 digest as lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path, *, base_dir: Path | None = None) -> str:
    if base_dir is None:
        return str(path)
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class ArtifactReference:
    """Durable artifact reference recorded in a step manifest."""

    path: str
    size_bytes: int
    sha256: str
    role: str = "output"
    media_type: str | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        role: str = "output",
        base_dir: str | Path | None = None,
        manifest_path: str | None = None,
        media_type: str | None = None,
    ) -> "ArtifactReference":
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Cannot record missing artifact: {artifact_path}")
        if not artifact_path.is_file():
            raise ValueError(f"Step manifest artifacts must be files: {artifact_path}")
        base = Path(base_dir) if base_dir is not None else None
        return cls(
            path=manifest_path or _manifest_path(artifact_path, base_dir=base),
            size_bytes=artifact_path.stat().st_size,
            sha256=sha256_file(artifact_path),
            role=role,
            media_type=media_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactReference":
        return cls(
            path=str(data["path"]),
            size_bytes=int(data["size_bytes"]),
            sha256=str(data["sha256"]),
            role=str(data.get("role", "output")),
            media_type=data.get("media_type"),
        )


@dataclass(frozen=True)
class ReuseMeasurement:
    """Measured reuse/recompute counts for one step."""

    expected_outputs: int = 0
    valid_reused_outputs: int = 0
    recomputed_outputs: int = 0
    invalid_outputs: int = 0
    saved_duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ReuseMeasurement":
        data = data or {}
        return cls(
            expected_outputs=int(data.get("expected_outputs", 0)),
            valid_reused_outputs=int(data.get("valid_reused_outputs", 0)),
            recomputed_outputs=int(data.get("recomputed_outputs", 0)),
            invalid_outputs=int(data.get("invalid_outputs", 0)),
            saved_duration_s=data.get("saved_duration_s"),
        )


@dataclass(frozen=True)
class OutputValidation:
    """Result of validating manifest-declared outputs."""

    valid: bool
    reason: str
    missing_outputs: tuple[str, ...] = ()
    checksum_mismatches: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepReuseDecision:
    """Manifest-backed decision about whether a step can be reused."""

    reusable: bool
    reason: str
    manifest: "StepManifest | None" = None
    validation: OutputValidation | None = None


@dataclass
class StepManifest:
    """Execution manifest for one meaningful pipeline step or sub-step."""

    run_id: str
    step_id: str
    status: str
    attempt: int
    started_at: str
    completed_at: str | None = None
    duration_s: float | None = None
    branch: str | None = None
    sha: str | None = None
    version: str | None = None
    modal_app_name: str | None = None
    modal_environment: str | None = None
    hf_staging_prefix: str | None = None
    parent_step_id: str | None = None
    scope: str | None = None
    modal_app_id: str | None = None
    modal_call_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    input_identities: dict[str, Any] = field(default_factory=dict)
    outputs: list[ArtifactReference] = field(default_factory=list)
    diagnostics: list[ArtifactReference] = field(default_factory=list)
    reuse_decision: str = "not_applicable"
    reuse_reason: str | None = None
    reuse_measurement: ReuseMeasurement = field(default_factory=ReuseMeasurement)
    error: dict[str, Any] | None = None
    schema_version: str = STEP_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid step manifest status: {self.status}")
        if self.reuse_decision not in VALID_REUSE_DECISIONS:
            raise ValueError(
                f"Invalid step manifest reuse decision: {self.reuse_decision}"
            )

    def complete(
        self,
        *,
        completed_at: str | None = None,
        status: str = "completed",
        outputs: Sequence[ArtifactReference] | None = None,
        diagnostics: Sequence[ArtifactReference] | None = None,
        reuse_decision: str = "computed",
        reuse_reason: str | None = None,
        reuse_measurement: ReuseMeasurement | None = None,
    ) -> "StepManifest":
        completed = completed_at or utc_now()
        started = datetime.fromisoformat(self.started_at)
        ended = datetime.fromisoformat(completed)
        return StepManifest(
            run_id=self.run_id,
            step_id=self.step_id,
            status=status,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=completed,
            duration_s=round((ended - started).total_seconds(), 1),
            branch=self.branch,
            sha=self.sha,
            version=self.version,
            modal_app_name=self.modal_app_name,
            modal_environment=self.modal_environment,
            hf_staging_prefix=self.hf_staging_prefix,
            parent_step_id=self.parent_step_id,
            scope=self.scope,
            modal_app_id=self.modal_app_id,
            modal_call_id=self.modal_call_id,
            parameters=self.parameters,
            input_identities=self.input_identities,
            outputs=list(outputs if outputs is not None else self.outputs),
            diagnostics=list(
                diagnostics if diagnostics is not None else self.diagnostics
            ),
            reuse_decision=reuse_decision,
            reuse_reason=reuse_reason,
            reuse_measurement=reuse_measurement or self.reuse_measurement,
            schema_version=self.schema_version,
        )

    def fail(
        self,
        exc: BaseException,
        *,
        completed_at: str | None = None,
        error_details: Mapping[str, Any] | None = None,
    ) -> "StepManifest":
        completed = completed_at or utc_now()
        started = datetime.fromisoformat(self.started_at)
        ended = datetime.fromisoformat(completed)
        error = {
            "type": type(exc).__name__,
            "message": redact_error_text(str(exc)),
        }
        if error_details:
            error.update(dict(error_details))
        return StepManifest(
            run_id=self.run_id,
            step_id=self.step_id,
            status="failed",
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=completed,
            duration_s=round((ended - started).total_seconds(), 1),
            branch=self.branch,
            sha=self.sha,
            version=self.version,
            modal_app_name=self.modal_app_name,
            modal_environment=self.modal_environment,
            hf_staging_prefix=self.hf_staging_prefix,
            parent_step_id=self.parent_step_id,
            scope=self.scope,
            modal_app_id=self.modal_app_id,
            modal_call_id=self.modal_call_id,
            parameters=self.parameters,
            input_identities=self.input_identities,
            outputs=self.outputs,
            diagnostics=self.diagnostics,
            reuse_decision="failed",
            reuse_reason="step_failed",
            reuse_measurement=self.reuse_measurement,
            error=error,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "scope": self.scope,
            "status": self.status,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "branch": self.branch,
            "sha": self.sha,
            "version": self.version,
            "modal_app_name": self.modal_app_name,
            "modal_environment": self.modal_environment,
            "hf_staging_prefix": self.hf_staging_prefix,
            "parent_step_id": self.parent_step_id,
            "modal_app_id": self.modal_app_id,
            "modal_call_id": self.modal_call_id,
            "parameters": self.parameters,
            "input_identities": self.input_identities,
            "outputs": [artifact.to_dict() for artifact in self.outputs],
            "diagnostics": [artifact.to_dict() for artifact in self.diagnostics],
            "reuse_decision": self.reuse_decision,
            "reuse_reason": self.reuse_reason,
            "reuse_measurement": self.reuse_measurement.to_dict(),
            "error": self.error,
        }
        return _drop_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepManifest":
        return cls(
            schema_version=str(
                data.get("schema_version", STEP_MANIFEST_SCHEMA_VERSION)
            ),
            run_id=str(data["run_id"]),
            step_id=str(data["step_id"]),
            scope=data.get("scope"),
            status=str(data["status"]),
            attempt=int(data["attempt"]),
            started_at=str(data["started_at"]),
            completed_at=data.get("completed_at"),
            duration_s=data.get("duration_s"),
            branch=data.get("branch"),
            sha=data.get("sha"),
            version=data.get("version"),
            modal_app_name=data.get("modal_app_name"),
            modal_environment=data.get("modal_environment"),
            hf_staging_prefix=data.get("hf_staging_prefix"),
            parent_step_id=data.get("parent_step_id"),
            modal_app_id=data.get("modal_app_id"),
            modal_call_id=data.get("modal_call_id"),
            parameters=dict(data.get("parameters", {})),
            input_identities=dict(data.get("input_identities", {})),
            outputs=[
                ArtifactReference.from_dict(item) for item in data.get("outputs", [])
            ],
            diagnostics=[
                ArtifactReference.from_dict(item)
                for item in data.get("diagnostics", [])
            ],
            reuse_decision=str(data.get("reuse_decision", "not_applicable")),
            reuse_reason=data.get("reuse_reason"),
            reuse_measurement=ReuseMeasurement.from_dict(data.get("reuse_measurement")),
            error=data.get("error"),
        )

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())


@dataclass
class RunManifest:
    """Run-level execution ledger for step manifests."""

    run_id: str
    branch: str
    sha: str
    version: str
    status: str
    started_at: str
    known_step_ids: list[str]
    candidate_version: str | None = None
    release_version: str | None = None
    run_context: dict[str, Any] = field(default_factory=dict)
    modal_app_name: str | None = None
    modal_environment: str | None = None
    hf_staging_prefix: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    resume_history: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    schema_version: str = STEP_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        return cls(
            schema_version=str(
                data.get("schema_version", STEP_MANIFEST_SCHEMA_VERSION)
            ),
            run_id=str(data["run_id"]),
            branch=str(data["branch"]),
            sha=str(data["sha"]),
            version=str(data["version"]),
            candidate_version=data.get("candidate_version") or data.get("version"),
            release_version=data.get("release_version") or data.get("version"),
            status=str(data["status"]),
            started_at=str(data["started_at"]),
            run_context=dict(
                data.get("run_context") or data.get("publication_context", {})
            ),
            modal_app_name=data.get("modal_app_name"),
            modal_environment=data.get("modal_environment"),
            hf_staging_prefix=data.get("hf_staging_prefix"),
            updated_at=data.get("updated_at"),
            completed_at=data.get("completed_at"),
            known_step_ids=list(data.get("known_step_ids", [])),
            resume_history=list(data.get("resume_history", [])),
            error=data.get("error"),
        )

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())


def run_manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "run_manifest.json"


def step_manifest_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "steps"


def step_manifest_path(run_dir: str | Path, step_id: str) -> Path:
    return step_manifest_dir(run_dir) / f"{step_id}.json"


def write_run_manifest(path: str | Path, manifest: RunManifest) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json())


def read_run_manifest(path: str | Path) -> RunManifest:
    return RunManifest.from_dict(json.loads(Path(path).read_text()))


def write_step_manifest(path: str | Path, manifest: StepManifest) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json())


def read_step_manifest(path: str | Path) -> StepManifest:
    return StepManifest.from_dict(json.loads(Path(path).read_text()))


def collect_artifacts(
    paths: Iterable[str | Path],
    *,
    role: str = "output",
    base_dir: str | Path | None = None,
    missing_ok: bool = False,
) -> list[ArtifactReference]:
    artifacts: list[ArtifactReference] = []
    for path in paths:
        artifact_path = Path(path)
        if not artifact_path.exists():
            if missing_ok:
                continue
            raise FileNotFoundError(f"Expected artifact does not exist: {path}")
        artifacts.append(
            ArtifactReference.from_path(
                artifact_path,
                role=role,
                base_dir=base_dir,
            )
        )
    return artifacts


def collect_directory_artifacts(
    root: str | Path,
    *,
    patterns: Sequence[str] = ("*",),
    role: str = "output",
    base_dir: str | Path | None = None,
) -> list[ArtifactReference]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(path for path in root_path.glob(pattern) if path.is_file())
    return [
        ArtifactReference.from_path(path, role=role, base_dir=base_dir)
        for path in sorted(set(paths))
    ]


def _resolve_artifact_path(path: str, *, root: str | Path | None = None) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute() or root is None:
        return artifact_path
    return Path(root) / artifact_path


def _contains_expected_values(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Return True when every expected key/value is present in actual."""
    return all(actual.get(key) == value for key, value in expected.items())


def validate_step_outputs(
    manifest: StepManifest,
    *,
    root: str | Path | None = None,
) -> OutputValidation:
    missing: list[str] = []
    mismatches: list[str] = []

    for artifact in manifest.outputs:
        path = _resolve_artifact_path(artifact.path, root=root)
        if not path.exists():
            missing.append(artifact.path)
            continue
        actual_sha = sha256_file(path)
        if actual_sha != artifact.sha256:
            mismatches.append(artifact.path)

    if missing:
        return OutputValidation(False, "missing_output", tuple(missing), ())
    if mismatches:
        return OutputValidation(False, "checksum_mismatch", (), tuple(mismatches))
    return OutputValidation(True, "valid")


def evaluate_step_reuse(
    manifest_path_value: str | Path,
    *,
    expected_input_identities: Mapping[str, Any] | None = None,
    expected_parameters: Mapping[str, Any] | None = None,
    output_root: str | Path | None = None,
) -> StepReuseDecision:
    path = Path(manifest_path_value)
    if not path.exists():
        return StepReuseDecision(False, "missing_manifest")

    manifest = read_step_manifest(path)
    if manifest.status not in COMPLETED_STATUSES:
        return StepReuseDecision(False, "incomplete_status", manifest=manifest)

    if expected_input_identities is not None and not _contains_expected_values(
        manifest.input_identities, dict(expected_input_identities)
    ):
        return StepReuseDecision(False, "input_changed", manifest=manifest)

    if expected_parameters is not None and manifest.parameters != dict(
        expected_parameters
    ):
        return StepReuseDecision(False, "parameter_changed", manifest=manifest)

    validation = validate_step_outputs(manifest, root=output_root)
    if not validation.valid:
        return StepReuseDecision(
            False,
            validation.reason,
            manifest=manifest,
            validation=validation,
        )

    return StepReuseDecision(
        True, "prior_success", manifest=manifest, validation=validation
    )


def completed_validated_outputs(
    run_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    step_ids: Iterable[str] | None = None,
) -> list[ArtifactReference]:
    """Return validated outputs from completed step manifests."""
    root = Path(run_dir)
    wanted = set(step_ids) if step_ids is not None else None
    outputs: list[ArtifactReference] = []
    for manifest_file in sorted(step_manifest_dir(root).glob("*.json")):
        manifest = read_step_manifest(manifest_file)
        if wanted is not None and manifest.step_id not in wanted:
            continue
        if manifest.status not in COMPLETED_STATUSES:
            continue
        validation = validate_step_outputs(manifest, root=output_root)
        if not validation.valid:
            continue
        outputs.extend(manifest.outputs)
    return outputs
