"""Rerun and semantic reuse planning for Stage 1 dataset builds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from policyengine_us_data.stage_contracts import fingerprint_material


Stage1ReuseAction = Literal["reuse", "recompute", "blocked"]


@dataclass(frozen=True, kw_only=True)
class Stage1IdentityMaterial:
    """Semantic identity material for a Stage 1 substep attempt."""

    substep_id: str
    code_sha: str
    schema_version: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    artifact_specs: Sequence[Mapping[str, Any]] = ()
    upstream_contract_fingerprints: Sequence[str] = ()
    randomness: Mapping[str, Any] = field(default_factory=dict)
    blocked_reason: str | None = None

    def fingerprint(self) -> str:
        """Return the deterministic semantic identity fingerprint."""

        return fingerprint_material(
            {
                "substep_id": self.substep_id,
                "inputs": dict(self.inputs),
                "parameters": dict(self.parameters),
                "artifact_specs": list(self.artifact_specs),
                "code_sha": self.code_sha,
                "schema_version": self.schema_version,
                "upstream_contract_fingerprints": list(
                    self.upstream_contract_fingerprints
                ),
                "randomness": dict(self.randomness),
            }
        ).value


@dataclass(frozen=True, kw_only=True)
class Stage1ReuseDecision:
    """Semantic rerun/reuse decision for a Stage 1 substep."""

    run_id: str
    rerun_id: str | None
    artifact_namespace: str
    substep_id: str
    action: Stage1ReuseAction
    reason: str
    identity_fingerprint: str

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-compatible reuse decision."""

        return {
            "run_id": self.run_id,
            "rerun_id": self.rerun_id,
            "artifact_namespace": self.artifact_namespace,
            "substep_id": self.substep_id,
            "action": self.action,
            "reason": self.reason,
            "identity_fingerprint": self.identity_fingerprint,
        }


@dataclass(frozen=True, kw_only=True)
class Stage1RerunPlanner:
    """Decide whether Stage 1 substeps may reuse semantic work."""

    previous_identities: Mapping[str, str] = field(default_factory=dict)

    def decide(
        self,
        material: Stage1IdentityMaterial,
        *,
        run_id: str,
        rerun_id: str | None = None,
    ) -> Stage1ReuseDecision:
        """Return a semantic reuse, recompute, or blocked decision."""

        fingerprint = material.fingerprint()
        if material.blocked_reason:
            return Stage1ReuseDecision(
                run_id=run_id,
                rerun_id=rerun_id,
                artifact_namespace=run_id,
                substep_id=material.substep_id,
                action="blocked",
                reason=material.blocked_reason,
                identity_fingerprint=fingerprint,
            )

        previous = self.previous_identities.get(material.substep_id)
        if previous == fingerprint:
            action = "reuse"
            reason = "identity_match"
        elif previous is None:
            action = "recompute"
            reason = "no_previous_identity"
        else:
            action = "recompute"
            reason = "identity_mismatch"

        return Stage1ReuseDecision(
            run_id=run_id,
            rerun_id=rerun_id,
            artifact_namespace=run_id,
            substep_id=material.substep_id,
            action=action,
            reason=reason,
            identity_fingerprint=fingerprint,
        )


__all__ = [
    "Stage1IdentityMaterial",
    "Stage1RerunPlanner",
    "Stage1ReuseAction",
    "Stage1ReuseDecision",
]
