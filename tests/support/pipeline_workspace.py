"""Shared fixture-scale pipeline workspace helpers.

This module is intentionally test-only. Production code must not import it.
It defines the canonical tiny pipeline directory and artifact names that local
integration tests and Modal test harnesses can share as we move coverage
upstream from H5-only seams toward dataset build stages 1-5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

__test__ = False


STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "stage_1": (
        "uprating_factors.csv",
        "acs_2022.h5",
        "irs_puf_2015.h5",
    ),
    "stage_2": (
        "cps_2024.h5",
        "puf_2024.h5",
    ),
    "stage_3": ("extended_cps_2024.h5",),
    "stage_4": (
        "enhanced_cps_2024.h5",
        "stratified_extended_cps_2024.h5",
    ),
    "stage_5": (
        "source_imputed_stratified_extended_cps_2024.h5",
        "source_imputed_stratified_extended_cps.h5",
        "small_enhanced_cps_2024.h5",
        "sparse_enhanced_cps_2024.h5",
    ),
    "calibration": (
        "policy_data.db",
        "calibration_package.pkl",
        "calibration_weights.npy",
        "geography_assignment.npz",
        "unified_run_config.json",
    ),
    "h5_outputs": (
        "states/AL.h5",
        "districts/NC-01.h5",
        "national/US.h5",
    ),
}


@dataclass(frozen=True)
class TinyPipelineWorkspace:
    """Canonical on-disk layout for fixture-scale pipeline tests."""

    root: Path

    STAGE_NAMES: ClassVar[tuple[str, ...]] = tuple(STAGE_ARTIFACTS)
    TOP_LEVEL_DIRS: ClassVar[tuple[str, ...]] = (
        "inputs",
        "stage_1",
        "stage_2",
        "stage_3",
        "stage_4",
        "stage_5",
        "calibration",
        "h5",
    )
    H5_DIRS: ClassVar[tuple[str, ...]] = (
        "outputs",
        "staging",
        "diagnostics",
        "manifests",
    )

    @classmethod
    def create(cls, root: Path) -> "TinyPipelineWorkspace":
        """Create an empty canonical workspace under ``root``."""

        workspace = cls(root=root)
        workspace.materialize()
        return workspace

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def stage_1(self) -> Path:
        return self.root / "stage_1"

    @property
    def stage_2(self) -> Path:
        return self.root / "stage_2"

    @property
    def stage_3(self) -> Path:
        return self.root / "stage_3"

    @property
    def stage_4(self) -> Path:
        return self.root / "stage_4"

    @property
    def stage_5(self) -> Path:
        return self.root / "stage_5"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration"

    @property
    def h5(self) -> Path:
        return self.root / "h5"

    @property
    def h5_outputs(self) -> Path:
        return self.h5 / "outputs"

    @property
    def h5_staging(self) -> Path:
        return self.h5 / "staging"

    @property
    def h5_diagnostics(self) -> Path:
        return self.h5 / "diagnostics"

    @property
    def h5_manifests(self) -> Path:
        return self.h5 / "manifests"

    def materialize(self) -> None:
        """Create all canonical directories without writing artifacts."""

        for dirname in self.TOP_LEVEL_DIRS:
            (self.root / dirname).mkdir(parents=True, exist_ok=True)
        for dirname in self.H5_DIRS:
            (self.h5 / dirname).mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str) -> Path:
        """Return the directory for a known stage name."""

        if stage == "h5_outputs":
            return self.h5_outputs
        if stage not in STAGE_ARTIFACTS:
            raise KeyError(f"Unknown tiny pipeline stage: {stage}")
        return self.root / stage

    def artifact_path(self, stage: str, relative_path: str) -> Path:
        """Return an artifact path and ensure nested parent dirs exist."""

        path = self.stage_dir(stage) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def expected_artifacts(self, stage: str) -> tuple[Path, ...]:
        """Return expected artifact paths for a known stage."""

        if stage not in STAGE_ARTIFACTS:
            raise KeyError(f"Unknown tiny pipeline stage: {stage}")
        return tuple(
            self.artifact_path(stage, relative_path)
            for relative_path in STAGE_ARTIFACTS[stage]
        )

    def all_expected_artifacts(self) -> dict[str, tuple[Path, ...]]:
        """Return every currently defined expected artifact path by stage."""

        return {stage: self.expected_artifacts(stage) for stage in STAGE_ARTIFACTS}
