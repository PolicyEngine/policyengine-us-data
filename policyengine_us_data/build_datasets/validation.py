"""Stage 1 validation adapters over the shared validation core."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias
from urllib.parse import unquote, urlparse

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    StageContract,
    ValidationFinding,
    ValidationReport,
)
from policyengine_us_data.utils.step_manifest import sha256_file
from policyengine_us_data.validation_core import (
    ValidationArtifactResolver,
    ValidationCheck,
    ValidationContext,
    ValidationRunner,
    ValidationSuite,
)

from .artifacts import DatasetArtifactSpec, stage_1_artifact_specs
from .results import DatasetSubstepResult
from .specs import STAGE_1_BUILD_DATASETS, stage_1_step_specs
from .validation_targets import ValidationTargetCatalog


Stage1Validator: TypeAlias = Callable[[ValidationContext], ValidationFinding | None]


class Stage1ValidationError(RuntimeError):
    """Raised when an error-level Stage 1 validation report fails."""


@dataclass(frozen=True, kw_only=True)
class Stage1ValidationContext:
    """Stage 1-specific context adapted into validation_core objects."""

    run_id: str
    substage_id: str
    artifact_refs: Mapping[str, ArtifactRef] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    output_contract: StageContract | None = None

    @classmethod
    def from_contract(
        cls,
        *,
        contract: StageContract,
        substage_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Stage1ValidationContext":
        """Build a context from a Stage 1 output contract."""

        artifacts = {
            artifact.logical_name: artifact
            for artifact in contract.outputs
            if artifact.metadata.get("substage_id") == substage_id
        }
        return cls(
            run_id=contract.run_id or "unknown",
            substage_id=substage_id,
            artifact_refs=artifacts,
            metadata=dict(metadata or {}),
            output_contract=contract,
        )

    @classmethod
    def from_substep_result(
        cls,
        *,
        run_id: str,
        result: DatasetSubstepResult,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Stage1ValidationContext":
        """Build a context from one completed Stage 1 substep result."""

        artifacts = {
            artifact.logical_name: artifact
            for artifact in (
                _artifact_ref_for_path(Path(path), result.substep_id)
                for path in result.artifact_paths
            )
            if artifact is not None
        }
        context_metadata = {
            "substep_status": result.status,
            "command_names": list(result.command_names),
            "artifact_paths": list(result.artifact_paths),
        }
        context_metadata.update(dict(metadata or {}))
        return cls(
            run_id=run_id,
            substage_id=result.substep_id,
            artifact_refs=artifacts,
            metadata=context_metadata,
        )

    def to_core_context(self) -> ValidationContext:
        """Return the shared validation_core context."""

        return ValidationContext(
            run_id=self.run_id,
            stage_id=STAGE_1_BUILD_DATASETS,
            substage_id=self.substage_id,
            resolver=ValidationArtifactResolver(artifacts=self.artifact_refs),
            metadata=self.metadata,
        )


@dataclass(frozen=True, kw_only=True)
class Stage1ValidatorSpec:
    """Stage 1 wrapper around a shared ValidationCheck."""

    validator_id: str
    substage_id: str
    description: str
    run: Stage1Validator
    severity: str = "error"

    def to_check(self, *, required_artifacts: tuple[str, ...]) -> ValidationCheck:
        """Return the validation_core check represented by this spec."""

        return ValidationCheck(
            check_id=self.validator_id,
            stage_id=STAGE_1_BUILD_DATASETS,
            substage_id=self.substage_id,
            description=self.description,
            severity="warning" if self.severity == "warning" else "error",
            required_artifacts=required_artifacts,
            run=self.run,
        )


@pipeline_node(
    id="stage_1_validation_runner",
    label="Stage 1 Validation Runner",
    node_type="library",
    description="Stage 1 adapter that runs ordered validators through validation_core.",
    source_file="policyengine_us_data/build_datasets/validation.py",
    status="current",
    stability="stable",
    pathways=["data_build", "stage_contracts", "cross_stage_validation"],
    artifacts_in=["Stage 1 artifacts", "dataset_build_output.json"],
    artifacts_out=["ValidationReport"],
    validation_commands=["uv run pytest tests/unit/test_build_dataset_validation.py"],
)
@dataclass(frozen=True, kw_only=True)
class Stage1ValidationRunner:
    """Run Stage 1 validators and aggregate canonical validation reports."""

    run_id: str
    catalog: ValidationTargetCatalog = field(
        default_factory=ValidationTargetCatalog.from_stage_1_specs
    )
    runner: ValidationRunner = field(default_factory=ValidationRunner)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def run_for_context(
        self,
        context: Stage1ValidationContext,
        *,
        required_artifacts: tuple[str, ...] | None = None,
    ) -> ValidationReport:
        """Run validators for a prepared Stage 1 context."""

        specs = validators_for_substage(context.substage_id)
        if not specs:
            return ValidationReport(
                status="not_run",
                metadata=self._report_metadata(context, skip_reason="no_validators"),
            )
        required = (
            tuple(required_artifacts)
            if required_artifacts is not None
            else tuple(context.artifact_refs)
        )
        if not required:
            return ValidationReport(
                status="not_run",
                metadata=self._report_metadata(
                    context,
                    skip_reason="no_artifacts_for_substep",
                ),
            )
        suite = ValidationSuite(
            suite_id=f"stage_1.{context.substage_id}",
            stage_id=STAGE_1_BUILD_DATASETS,
            substage_id=context.substage_id,
            checks=tuple(spec.to_check(required_artifacts=required) for spec in specs),
        )
        return self.runner.run(suite, context.to_core_context())

    def run_for_substep_result(
        self,
        result: DatasetSubstepResult,
    ) -> ValidationReport:
        """Run validators for one completed coordinator substep result."""

        context = Stage1ValidationContext.from_substep_result(
            run_id=self.run_id,
            result=result,
            metadata=self.metadata,
        )
        return self.run_for_context(context)

    def run_for_contract(
        self,
        contract: StageContract,
        substage_id: str,
    ) -> ValidationReport:
        """Run validators for one substage using a Stage 1 output contract."""

        context = Stage1ValidationContext.from_contract(
            contract=contract,
            substage_id=substage_id,
            metadata=self.metadata,
        )
        required = self.catalog.required_logical_names(substage_id)
        return self.run_for_context(context, required_artifacts=required)

    def should_stop(self, report: ValidationReport) -> bool:
        """Return whether a report should stop downstream Stage 1 execution."""

        return report.status == "fail"

    def _report_metadata(
        self,
        context: Stage1ValidationContext,
        **extra: Any,
    ) -> dict[str, Any]:
        metadata = {
            "stage_id": STAGE_1_BUILD_DATASETS,
            "substage_id": context.substage_id,
            "run_id": context.run_id,
            "context_metadata": dict(context.metadata),
        }
        metadata.update(extra)
        return metadata


def iter_stage_1_validators() -> tuple[Stage1ValidatorSpec, ...]:
    """Return all registered Stage 1 validator specs."""

    return _STAGE_1_VALIDATORS


def validators_for_substage(substage_id: str) -> tuple[Stage1ValidatorSpec, ...]:
    """Return validator specs wired to one Stage 1 substage."""

    validation_ids = ()
    for spec in stage_1_step_specs():
        if spec.id == substage_id:
            validation_ids = spec.validation_ids
            break
    validators_by_id = {spec.validator_id: spec for spec in _STAGE_1_VALIDATORS}
    return tuple(validators_by_id[validator_id] for validator_id in validation_ids)


def run_stage_1_validators(
    context: Stage1ValidationContext,
    *,
    runner: Stage1ValidationRunner | None = None,
) -> ValidationReport:
    """Run registered validators for a Stage 1 context."""

    active_runner = runner or Stage1ValidationRunner(run_id=context.run_id)
    return active_runner.run_for_context(context)


def _validate_artifact_refs(
    context: ValidationContext,
    *,
    check_id: str,
) -> ValidationFinding | None:
    for logical_name, artifact in sorted(context.resolver.artifacts.items()):
        path = _file_uri_to_path(artifact.uri)
        if path is None:
            continue
        if not path.exists():
            return ValidationFinding(
                check_id=check_id,
                status="fail",
                message=f"Validation artifact does not exist: {logical_name}",
                metric="artifact_exists",
                value=str(path),
                metadata={
                    "logical_name": logical_name,
                    "uri": artifact.uri,
                    "substage_id": context.substage_id,
                },
            )
        if path.is_file() and path.stat().st_size == 0:
            return ValidationFinding(
                check_id=check_id,
                status="fail",
                message=f"Validation artifact is empty: {logical_name}",
                metric="artifact_size_bytes",
                value=0,
                metadata={
                    "logical_name": logical_name,
                    "uri": artifact.uri,
                    "substage_id": context.substage_id,
                },
            )
    return None


def _artifact_ref_for_path(path: Path, substage_id: str) -> ArtifactRef | None:
    spec = _spec_for_path(path, substage_id)
    if spec is None and not path.exists():
        return None
    logical_name = spec.logical_name if spec else path.stem
    metadata: dict[str, Any] = {"substage_id": substage_id}
    if spec is not None:
        metadata.update(
            {
                "artifact_family": spec.artifact_family,
                "filename": spec.filename,
                "period": spec.period,
            }
        )
    return ArtifactRef(
        logical_name=logical_name,
        uri=path.resolve().as_uri(),
        sha256=f"sha256:{sha256_file(path)}"
        if path.exists() and path.is_file()
        else None,
        size_bytes=path.stat().st_size if path.exists() and path.is_file() else None,
        media_type=_media_type_for_path(path),
        metadata=metadata,
    )


def _spec_for_path(path: Path, substage_id: str) -> DatasetArtifactSpec | None:
    path_name = path.name
    path_text = str(path)
    fallback: DatasetArtifactSpec | None = None
    for spec in stage_1_artifact_specs():
        candidates = {spec.filename}
        if spec.storage_path is not None:
            candidates.add(spec.storage_path)
            candidates.add(Path(spec.storage_path).name)
        if path_text in candidates or path_name in candidates:
            if spec.substage_id == substage_id:
                return spec
            fallback = fallback or spec
    return fallback


def _file_uri_to_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path))


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".h5":
        return "application/x-hdf5"
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return "application/vnd.sqlite3"
    if suffix == ".json":
        return "application/json"
    if suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


def _artifact_contract_validator(substage_id: str) -> Stage1ValidatorSpec:
    validator_id = f"stage_1.{substage_id}.artifact_contract"

    def run(context: ValidationContext) -> ValidationFinding | None:
        return _validate_artifact_refs(context, check_id=validator_id)

    return Stage1ValidatorSpec(
        validator_id=validator_id,
        substage_id=substage_id,
        description="Validate Stage 1 artifact references for this substage.",
        run=run,
    )


_STAGE_1_VALIDATORS: tuple[Stage1ValidatorSpec, ...] = tuple(
    _artifact_contract_validator(spec.id)
    for spec in stage_1_step_specs()
    if spec.validation_ids
)


__all__ = [
    "Stage1ValidationContext",
    "Stage1ValidationError",
    "Stage1ValidationRunner",
    "Stage1Validator",
    "Stage1ValidatorSpec",
    "iter_stage_1_validators",
    "run_stage_1_validators",
    "validators_for_substage",
]
