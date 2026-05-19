"""Run context for the Stage 1 dataset-build handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .specs import STAGE_1_BUILD_DATASETS


@dataclass(frozen=True, kw_only=True)
class DatasetBuildContext:
    """Identity and filesystem context for one Stage 1 dataset-build run."""

    run_id: str
    branch: str
    code_sha: str
    package_version: str
    artifacts_dir: Path
    storage_dir: Path = Path("policyengine_us_data/storage")
    work_dir: Path = Path(".")
    stage_id: str = STAGE_1_BUILD_DATASETS

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.branch:
            raise ValueError("branch is required")
        if not self.code_sha:
            raise ValueError("code_sha is required")
        if not self.package_version:
            raise ValueError("package_version is required")
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir))
        object.__setattr__(self, "storage_dir", Path(self.storage_dir))
        object.__setattr__(self, "work_dir", Path(self.work_dir))

    def source_path(self, storage_path: str) -> Path:
        """Resolve a declared storage or working-directory source path."""

        path = Path(storage_path)
        if path.is_absolute():
            return path
        storage_prefix = Path("policyengine_us_data/storage")
        try:
            return self.storage_dir / path.relative_to(storage_prefix)
        except ValueError:
            return self.work_dir / path

    def artifact_path(self, filename: str) -> Path:
        """Return the run-scoped destination path for a staged artifact."""

        return self.artifacts_dir / filename

    def identity(self) -> dict[str, str]:
        """Return stable identity fields for Stage 1 diagnostic payloads."""

        return {
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "branch": self.branch,
            "code_sha": self.code_sha,
            "package_version": self.package_version,
        }


__all__ = ["DatasetBuildContext"]
