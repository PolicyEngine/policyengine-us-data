"""Shared Stage 2 calibration-package identity and artifact specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import unquote, urlparse

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.utils.manifest import compute_file_checksum

if TYPE_CHECKING:
    from policyengine_us_data.stage_contracts import StageContract, ValidationReport

DEFAULT_TARGET_CONFIG_PATH = "policyengine_us_data/calibration/target_config.yaml"
SOURCE_DATASET_FILENAME = "source_imputed_stratified_extended_cps.h5"
TARGET_DATABASE_FILENAME = "policy_data.db"
DATASET_BUILD_OUTPUT_CONTRACT_FILENAME = "dataset_build_output.json"
CALIBRATION_PACKAGE_FILENAME = "calibration_package.pkl"
CALIBRATION_PACKAGE_METADATA_FILENAME = "calibration_package_meta.json"
CALIBRATION_PACKAGE_CONTRACT_FILENAME = "calibration_package_contract.json"
CALIBRATION_TARGETS_FILENAME = "calibration_targets.jsonl"
CALIBRATION_TARGET_FACETS_FILENAME = "calibration_target_facets.json"
GEOGRAPHY_ASSIGNMENT_SUMMARY_FILENAME = "geography_assignment_summary.json"
CALIBRATION_REPORTS_DIRNAME = "calibration_reports"
MATRIX_BUILD_DIRNAME = "matrix_build"
CALIBRATION_PACKAGE_SUBSTAGE_ID = "2a_matrix_build_calibration_target_construction"

TargetConfigMode = Literal["default", "explicit", "all_active_targets"]
Stage2InputSource = Literal["stage1_contract", "artifacts_dir_fallback"]
TARGET_CONFIG_IDENTITY_MODES: frozenset[str] = frozenset(
    {"default", "explicit", "all_active_targets"}
)
_SOURCE_DATASET_LOGICAL_NAMES = (
    "source_imputed_stratified_extended_cps",
    "source_imputed_stratified_extended_cps_2024",
)
_TARGET_DATABASE_LOGICAL_NAMES = ("policy_data_db",)


@dataclass(frozen=True, kw_only=True)
class TargetConfigIdentity:
    """Checksum-backed identity for the Stage 2 target selection config."""

    path: str | None
    sha256: str | None
    mode: TargetConfigMode
    resolved_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in TARGET_CONFIG_IDENTITY_MODES:
            raise ValueError(f"Unknown target config identity mode: {self.mode!r}")
        if self.mode == "all_active_targets":
            if self.path is not None or self.sha256 is not None:
                raise ValueError(
                    "all_active_targets target config identity cannot include "
                    "a path or checksum"
                )
            return
        if not self.path:
            raise ValueError(f"{self.mode} target config identity requires a path")
        if not self.sha256:
            raise ValueError(f"{self.mode} target config identity requires a checksum")

    def to_parameters(self) -> dict[str, str | None]:
        """Return the identity fields used in Stage 2 reuse parameters."""

        return {
            "target_config": self.path,
            "target_config_sha256": self.sha256,
            "target_config_mode": self.mode,
        }


@dataclass(frozen=True, kw_only=True)
class Stage2InputBundle:
    """Canonical Stage 2 input artifacts resolved for one run."""

    artifacts_dir: Path
    source_dataset: Path
    target_database: Path
    source: Stage2InputSource
    stage1_contract_path: Path | None = None
    stage1_contract_run_id: str | None = None

    @property
    def manifest_inputs(self) -> dict[str, Path]:
        """Return input paths recorded in Stage 2 step manifests."""

        return {
            "dataset": self.source_dataset,
            "database": self.target_database,
        }

    @property
    def compatibility_only(self) -> bool:
        """Return whether the bundle came from legacy filename discovery."""

        return self.source == "artifacts_dir_fallback"

    def missing_required_artifacts(self) -> tuple[tuple[str, Path], ...]:
        """Return missing required Stage 2 input labels and paths."""

        missing: list[tuple[str, Path]] = []
        for label, path in self.manifest_inputs.items():
            if not path.exists():
                missing.append((label, path))
        return tuple(missing)

    def validation_report(self) -> "ValidationReport":
        """Return a canonical validation report for Stage 2 input readiness."""

        from policyengine_us_data.stage_contracts.validation import (
            ValidationFinding,
            ValidationReport,
        )

        missing = self.missing_required_artifacts()
        if missing:
            findings = tuple(
                ValidationFinding(
                    check_id=f"stage2_input_exists:{label}",
                    status="fail",
                    message=f"Missing Stage 2 {label} artifact: {path}",
                    metadata={
                        "artifact_label": label,
                        "path": str(path),
                        "source": self.source,
                    },
                )
                for label, path in missing
            )
            return ValidationReport(
                status="fail",
                findings=findings,
                metadata=self._validation_metadata(),
            )
        return ValidationReport(
            status="pass",
            findings=(),
            metadata=self._validation_metadata(),
        )

    def require_existing(self) -> "Stage2InputBundle":
        """Raise a structured error when required Stage 2 inputs are missing."""

        report = self.validation_report()
        if report.status != "pass":
            raise Stage2InputBundleError(self, report)
        return self

    def _validation_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": self.source,
            "artifacts_dir": str(self.artifacts_dir),
            "compatibility_only": self.compatibility_only,
        }
        if self.stage1_contract_path is not None:
            metadata["stage1_contract_path"] = str(self.stage1_contract_path)
        if self.stage1_contract_run_id is not None:
            metadata["stage1_contract_run_id"] = self.stage1_contract_run_id
        return metadata


class Stage2InputBundleError(FileNotFoundError):
    """Input validation failure raised before Stage 2 package work starts."""

    def __init__(
        self,
        bundle: Stage2InputBundle,
        validation_report: "ValidationReport",
    ) -> None:
        missing = ", ".join(
            f"{label}: {path}" for label, path in bundle.missing_required_artifacts()
        )
        super().__init__(f"Missing Stage 2 input artifact(s): {missing}")
        self.bundle = bundle
        self.validation_report = validation_report


@dataclass(frozen=True, kw_only=True)
class CalibrationPackageOutputBundle:
    """Canonical run-scoped Stage 2 output artifact paths."""

    artifacts_dir: Path
    package: Path
    metadata: Path
    contract: Path
    targets: Path
    target_facets: Path
    geography_summary: Path
    reports_dir: Path
    matrix_build_dir: Path

    @property
    def manifest_outputs(self) -> tuple[Path, Path, Path, Path, Path]:
        """Return the durable Stage 2 outputs recorded in step manifests."""

        return (
            self.package,
            self.contract,
            self.targets,
            self.target_facets,
            self.geography_summary,
        )


CalibrationPackageArtifactPaths = CalibrationPackageOutputBundle


@dataclass(frozen=True, kw_only=True)
class Stage2BuildContext:
    """Run-scoped Stage 2 input and output bundles."""

    artifacts_dir: Path
    input_bundle: Stage2InputBundle
    output_bundle: CalibrationPackageOutputBundle
    run_id: str | None = None

    def require_inputs(self) -> "Stage2BuildContext":
        """Validate inputs and return this context when Stage 2 may start."""

        self.input_bundle.require_existing()
        return self


@pipeline_node(
    PipelineNode(
        id="stage2_input_bundle",
        label="Stage 2 Input Bundle",
        node_type="library",
        description="Resolve the source-imputed dataset and policy target database from a Stage 1 contract or compatibility filename fallback.",
        source_file="policyengine_us_data/calibration_package/specs.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=[
            DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
            SOURCE_DATASET_FILENAME,
            TARGET_DATABASE_FILENAME,
        ],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_specs.py"
        ],
    )
)
def stage2_input_bundle_from_artifacts_dir(
    artifacts_dir: str | Path,
) -> Stage2InputBundle:
    """Return a compatibility Stage 2 input bundle from canonical filenames."""

    root = Path(artifacts_dir)
    return Stage2InputBundle(
        artifacts_dir=root,
        source_dataset=root / SOURCE_DATASET_FILENAME,
        target_database=root / TARGET_DATABASE_FILENAME,
        source="artifacts_dir_fallback",
    )


def stage2_input_bundle_from_stage1_contract(
    contract: "StageContract",
    *,
    artifacts_dir: str | Path | None = None,
    contract_path: str | Path | None = None,
) -> Stage2InputBundle:
    """Return a Stage 2 input bundle from a Stage 1 handoff contract."""

    if getattr(contract, "stage_id", None) != "1_build_datasets":
        raise ValueError("Stage 2 inputs require a Stage 1 dataset-build contract")
    source_dataset = _contract_artifact_path(
        contract,
        logical_names=_SOURCE_DATASET_LOGICAL_NAMES,
        label="source dataset",
    )
    target_database = _contract_artifact_path(
        contract,
        logical_names=_TARGET_DATABASE_LOGICAL_NAMES,
        label="target database",
    )
    root = Path(artifacts_dir) if artifacts_dir is not None else source_dataset.parent
    return Stage2InputBundle(
        artifacts_dir=root,
        source_dataset=source_dataset,
        target_database=target_database,
        source="stage1_contract",
        stage1_contract_path=Path(contract_path) if contract_path is not None else None,
        stage1_contract_run_id=getattr(contract, "run_id", None),
    )


def stage2_input_bundle_from_stage1_contract_path(
    contract_path: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
) -> Stage2InputBundle:
    """Read a Stage 1 handoff contract and return the Stage 2 input bundle."""

    from policyengine_us_data.stage_contracts.io import read_contract

    contract_file = Path(contract_path)
    return stage2_input_bundle_from_stage1_contract(
        read_contract(contract_file),
        artifacts_dir=artifacts_dir,
        contract_path=contract_file,
    )


@pipeline_node(
    PipelineNode(
        id="stage2_build_context",
        label="Stage 2 Build Context",
        node_type="library",
        description="Bind one run_id to canonical Stage 2 input and output bundles before remote package construction starts.",
        source_file="policyengine_us_data/calibration_package/specs.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=[
            DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
            SOURCE_DATASET_FILENAME,
            TARGET_DATABASE_FILENAME,
        ],
        artifacts_out=[
            CALIBRATION_PACKAGE_FILENAME,
            CALIBRATION_PACKAGE_CONTRACT_FILENAME,
            GEOGRAPHY_ASSIGNMENT_SUMMARY_FILENAME,
        ],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_specs.py"
        ],
    )
)
def stage2_build_context_for_run(
    pipeline_mount: str | Path,
    run_id: str | None = "",
    *,
    stage1_contract_path: str | Path | None = None,
) -> Stage2BuildContext:
    """Return Stage 2 run context, preferring the Stage 1 handoff contract."""

    artifacts_dir = Path(pipeline_mount) / "artifacts"
    if run_id:
        artifacts_dir = artifacts_dir / run_id
    contract_path = (
        Path(stage1_contract_path)
        if stage1_contract_path is not None
        else artifacts_dir / DATASET_BUILD_OUTPUT_CONTRACT_FILENAME
    )
    if contract_path.exists():
        input_bundle = stage2_input_bundle_from_stage1_contract_path(
            contract_path,
            artifacts_dir=artifacts_dir,
        )
    else:
        input_bundle = stage2_input_bundle_from_artifacts_dir(artifacts_dir)
    return Stage2BuildContext(
        artifacts_dir=artifacts_dir,
        input_bundle=input_bundle,
        output_bundle=calibration_package_artifact_paths(artifacts_dir),
        run_id=run_id or None,
    )


@pipeline_node(
    PipelineNode(
        id="stage2_artifact_specs",
        label="Stage 2 Artifact Specs",
        node_type="library",
        description="Centralize Stage 2 input, package, contract, metadata, report, and matrix-build artifact names.",
        source_file="policyengine_us_data/calibration_package/specs.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=[
            SOURCE_DATASET_FILENAME,
            TARGET_DATABASE_FILENAME,
        ],
        artifacts_out=[
            CALIBRATION_PACKAGE_FILENAME,
            CALIBRATION_PACKAGE_METADATA_FILENAME,
            CALIBRATION_PACKAGE_CONTRACT_FILENAME,
            CALIBRATION_TARGETS_FILENAME,
            CALIBRATION_TARGET_FACETS_FILENAME,
            GEOGRAPHY_ASSIGNMENT_SUMMARY_FILENAME,
        ],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_specs.py"
        ],
    )
)
def calibration_package_artifact_paths(
    artifacts_dir: str | Path,
) -> CalibrationPackageOutputBundle:
    """Return canonical Stage 2 paths rooted in an artifacts directory."""

    root = Path(artifacts_dir)
    return CalibrationPackageOutputBundle(
        artifacts_dir=root,
        package=root / CALIBRATION_PACKAGE_FILENAME,
        metadata=root / CALIBRATION_PACKAGE_METADATA_FILENAME,
        contract=root / CALIBRATION_PACKAGE_CONTRACT_FILENAME,
        targets=root / CALIBRATION_TARGETS_FILENAME,
        target_facets=root / CALIBRATION_TARGET_FACETS_FILENAME,
        geography_summary=root / GEOGRAPHY_ASSIGNMENT_SUMMARY_FILENAME,
        reports_dir=root / CALIBRATION_REPORTS_DIRNAME,
        matrix_build_dir=root / MATRIX_BUILD_DIRNAME,
    )


@pipeline_node(
    PipelineNode(
        id="stage2_target_config_identity",
        label="Stage 2 Target Config Identity",
        node_type="library",
        description="Resolve the effective Stage 2 target config path and checksum before package reuse or rebuild.",
        source_file="policyengine_us_data/calibration_package/specs.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=[DEFAULT_TARGET_CONFIG_PATH],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_specs.py"
        ],
    )
)
def resolve_target_config_identity(
    target_config_path: str | Path | None = None,
    *,
    all_active_targets: bool = False,
    repo_root: str | Path | None = None,
) -> TargetConfigIdentity:
    """Resolve the target config identity used by Stage 2 package construction."""

    if all_active_targets:
        if target_config_path is not None:
            raise ValueError(
                "--all-active-targets cannot be combined with a target config path"
            )
        return TargetConfigIdentity(
            path=None,
            sha256=None,
            mode="all_active_targets",
            resolved_path=None,
        )

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    mode: TargetConfigMode = "explicit" if target_config_path is not None else "default"
    identity_path = Path(target_config_path or DEFAULT_TARGET_CONFIG_PATH)
    resolved_path = _resolve_existing_config_path(identity_path, root)
    logical_path = (
        DEFAULT_TARGET_CONFIG_PATH
        if mode == "default"
        else _logical_identity_path(identity_path, resolved_path, root)
    )
    return TargetConfigIdentity(
        path=logical_path,
        sha256=compute_file_checksum(resolved_path),
        mode=mode,
        resolved_path=str(resolved_path),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_existing_config_path(path: Path, repo_root: Path) -> Path:
    candidates = [path] if path.is_absolute() else [repo_root / path, Path.cwd() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Target config not found: {path}")


def _logical_identity_path(path: Path, resolved_path: Path, repo_root: Path) -> str:
    try:
        return resolved_path.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved_path.as_posix() if path.is_absolute() else path.as_posix()


def _contract_artifact_path(
    contract: "StageContract",
    *,
    logical_names: tuple[str, ...],
    label: str,
) -> Path:
    for logical_name in logical_names:
        for artifact in getattr(contract, "outputs", ()):
            if getattr(artifact, "logical_name", None) == logical_name:
                return _artifact_uri_to_path(getattr(artifact, "uri"))
    raise ValueError(
        f"Stage 1 contract is missing required Stage 2 {label}: "
        + " or ".join(logical_names)
    )


def _artifact_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri)
    raise ValueError(f"Unsupported artifact URI scheme for Stage 2 input: {uri}")


__all__ = [
    "CALIBRATION_PACKAGE_CONTRACT_FILENAME",
    "CALIBRATION_PACKAGE_FILENAME",
    "CALIBRATION_PACKAGE_METADATA_FILENAME",
    "CALIBRATION_PACKAGE_SUBSTAGE_ID",
    "CALIBRATION_TARGET_FACETS_FILENAME",
    "CALIBRATION_TARGETS_FILENAME",
    "CALIBRATION_REPORTS_DIRNAME",
    "DATASET_BUILD_OUTPUT_CONTRACT_FILENAME",
    "DEFAULT_TARGET_CONFIG_PATH",
    "GEOGRAPHY_ASSIGNMENT_SUMMARY_FILENAME",
    "MATRIX_BUILD_DIRNAME",
    "SOURCE_DATASET_FILENAME",
    "TARGET_CONFIG_IDENTITY_MODES",
    "TARGET_DATABASE_FILENAME",
    "CalibrationPackageArtifactPaths",
    "CalibrationPackageOutputBundle",
    "Stage2BuildContext",
    "Stage2InputBundle",
    "Stage2InputBundleError",
    "Stage2InputSource",
    "TargetConfigIdentity",
    "TargetConfigMode",
    "calibration_package_artifact_paths",
    "resolve_target_config_identity",
    "stage2_build_context_for_run",
    "stage2_input_bundle_from_artifacts_dir",
    "stage2_input_bundle_from_stage1_contract",
    "stage2_input_bundle_from_stage1_contract_path",
]
