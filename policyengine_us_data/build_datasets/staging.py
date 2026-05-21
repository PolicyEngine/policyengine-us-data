"""Artifact staging helpers for Stage 1 dataset-build outputs."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    DatasetArtifactSpec,
    stage_1_pipeline_artifact_specs,
)
from .context import DatasetBuildContext


@dataclass(frozen=True, kw_only=True)
class PipelineArtifactStager:
    """Stage declared Stage 1 artifacts into a run-scoped pipeline directory."""

    context: DatasetBuildContext

    def stage_declared_artifacts(
        self,
        *,
        skip_enhanced_cps: bool = False,
        skip_stage_5: bool = False,
    ) -> tuple[Path, ...]:
        self.context.artifacts_dir.mkdir(parents=True, exist_ok=True)
        staged: list[Path] = []
        missing_required: list[str] = []

        for spec in stage_1_pipeline_artifact_specs():
            if skip_enhanced_cps and spec.skip_when_enhanced_cps_skipped:
                continue
            if skip_stage_5 and spec.skip_when_stage_5_skipped:
                continue
            if spec.yearless_alias:
                alias = self._stage_yearless_alias(spec)
                if alias is not None:
                    staged.append(alias)
                continue
            if spec.storage_path is None:
                continue

            source = self.context.source_path(spec.storage_path)
            destination = self.context.artifact_path(spec.filename)
            if not source.exists():
                if spec.required:
                    missing_required.append(spec.filename)
                continue
            shutil.copy2(source, destination)
            staged.append(destination)

        if missing_required:
            raise FileNotFoundError(
                "Missing Stage 1 pipeline artifact(s): "
                + ", ".join(sorted(missing_required))
            )
        return tuple(staged)

    def write_checkpoint_stats(self, checkpoint_stats: Mapping[str, int]) -> Path:
        """Write checkpoint reuse metadata as an explicit Stage 1 artifact."""

        path = self.context.artifact_path("data_build_checkpoint_stats.json")
        path.write_text(
            json.dumps(dict(checkpoint_stats), indent=2, sort_keys=True) + "\n"
        )
        return path

    def _stage_yearless_alias(self, spec: DatasetArtifactSpec) -> Path | None:
        source = self.context.artifact_path(
            "source_imputed_stratified_extended_cps_2024.h5"
        )
        if not source.exists():
            return None
        destination = self.context.artifact_path(spec.filename)
        shutil.copy2(source, destination)
        return destination


__all__ = ["PipelineArtifactStager"]
