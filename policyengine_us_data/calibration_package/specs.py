"""Shared Stage 2 calibration-package identity and artifact specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.utils.manifest import compute_file_checksum

DEFAULT_TARGET_CONFIG_PATH = "policyengine_us_data/calibration/target_config.yaml"
CALIBRATION_PACKAGE_FILENAME = "calibration_package.pkl"
CALIBRATION_PACKAGE_METADATA_FILENAME = "calibration_package_meta.json"
CALIBRATION_PACKAGE_CONTRACT_FILENAME = "calibration_package_contract.json"
MATRIX_BUILD_DIRNAME = "matrix_build"
CALIBRATION_PACKAGE_SUBSTAGE_ID = "2a_matrix_build_calibration_target_construction"

TargetConfigMode = Literal["default", "explicit", "all_active_targets"]
TARGET_CONFIG_IDENTITY_MODES: frozenset[str] = frozenset(
    {"default", "explicit", "all_active_targets"}
)


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
class CalibrationPackageArtifactPaths:
    """Canonical run-scoped Stage 2 artifact paths."""

    artifacts_dir: Path
    package: Path
    metadata: Path
    contract: Path
    matrix_build_dir: Path

    @property
    def manifest_outputs(self) -> tuple[Path, Path]:
        """Return the durable Stage 2 outputs recorded in step manifests."""

        return (self.package, self.contract)


@pipeline_node(
    PipelineNode(
        id="stage2_artifact_specs",
        label="Stage 2 Artifact Specs",
        node_type="library",
        description="Centralize calibration package, contract, metadata, and matrix-build artifact paths.",
        source_file="policyengine_us_data/calibration_package/specs.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_out=[
            CALIBRATION_PACKAGE_FILENAME,
            CALIBRATION_PACKAGE_CONTRACT_FILENAME,
        ],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_specs.py"
        ],
    )
)
def calibration_package_artifact_paths(
    artifacts_dir: str | Path,
) -> CalibrationPackageArtifactPaths:
    """Return canonical Stage 2 paths rooted in an artifacts directory."""

    root = Path(artifacts_dir)
    return CalibrationPackageArtifactPaths(
        artifacts_dir=root,
        package=root / CALIBRATION_PACKAGE_FILENAME,
        metadata=root / CALIBRATION_PACKAGE_METADATA_FILENAME,
        contract=root / CALIBRATION_PACKAGE_CONTRACT_FILENAME,
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


__all__ = [
    "CALIBRATION_PACKAGE_CONTRACT_FILENAME",
    "CALIBRATION_PACKAGE_FILENAME",
    "CALIBRATION_PACKAGE_METADATA_FILENAME",
    "CALIBRATION_PACKAGE_SUBSTAGE_ID",
    "DEFAULT_TARGET_CONFIG_PATH",
    "MATRIX_BUILD_DIRNAME",
    "TARGET_CONFIG_IDENTITY_MODES",
    "CalibrationPackageArtifactPaths",
    "TargetConfigIdentity",
    "TargetConfigMode",
    "calibration_package_artifact_paths",
    "resolve_target_config_identity",
]
