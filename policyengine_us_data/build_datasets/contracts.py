"""Contract builder facade for Stage 1 dataset-build outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .context import DatasetBuildContext


@dataclass(frozen=True, kw_only=True)
class DatasetBuildOutputContractBuilder:
    """Build and persist the Stage 1 dataset-build handoff contract."""

    context: DatasetBuildContext

    def build(
        self,
        *,
        checkpoint_stats: Mapping[str, int],
        started_at: str | None,
        completed_at: str,
        duration_s: float | None,
        upload_requested: bool,
        stage_only: bool,
        skip_enhanced_cps: bool,
        skip_stage_5: bool = False,
        diagnostics: Sequence[object] = (),
        validation: object | None = None,
        substage_validation: Mapping[str, object] | None = None,
        stage_1_status_metadata: Mapping[str, object] | None = None,
    ):
        """Build the Stage 1 handoff contract from staged artifacts."""

        from policyengine_us_data.stage_contracts import (
            build_dataset_build_output_contract,
        )

        return build_dataset_build_output_contract(
            artifacts_dir=self.context.artifacts_dir,
            run_id=self.context.run_id,
            code_sha=self.context.code_sha,
            package_version=self.context.package_version,
            checkpoint_stats=checkpoint_stats,
            started_at=started_at,
            completed_at=completed_at,
            duration_s=duration_s,
            upload_requested=upload_requested,
            stage_only=stage_only,
            skip_enhanced_cps=skip_enhanced_cps,
            skip_stage_5=skip_stage_5,
            diagnostics=tuple(diagnostics),
            validation=validation,
            substage_validation=substage_validation,
            stage_1_status_metadata=stage_1_status_metadata,
        )

    def write(self, **kwargs):
        """Build and write the Stage 1 handoff contract next to artifacts."""

        from policyengine_us_data.stage_contracts import (
            DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
            write_contract,
        )

        contract = self.build(**kwargs)
        write_contract(
            contract,
            self.context.artifacts_dir / DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
        )
        return contract


__all__ = ["DatasetBuildOutputContractBuilder"]
