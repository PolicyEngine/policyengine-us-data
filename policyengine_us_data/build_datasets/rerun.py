"""Rerun and semantic reuse planning for Stage 1 dataset builds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from policyengine_us_data.stage_contracts.fingerprints import fingerprint_material


Stage1ReuseAction = Literal["reuse", "recompute", "blocked"]
STAGE_1_REUSE_MANIFEST_FILENAME = "stage_1_reuse_manifest.json"
STAGE_1_REUSE_MANIFEST_SCHEMA_VERSION = "stage-1-reuse-manifest-v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


@dataclass(frozen=True, kw_only=True)
class Stage1IdentityMaterial:
    """Semantic identity material for one Stage 1 execution unit."""

    substep_id: str
    identity_key: str
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

        return fingerprint_material(self.to_dict()).value

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe semantic identity material."""

        return {
            "substep_id": self.substep_id,
            "identity_key": self.identity_key,
            "inputs": _json_safe(self.inputs),
            "parameters": _json_safe(self.parameters),
            "artifact_specs": _json_safe(self.artifact_specs),
            "code_sha": self.code_sha,
            "schema_version": self.schema_version,
            "upstream_contract_fingerprints": _json_safe(
                self.upstream_contract_fingerprints
            ),
            "randomness": _json_safe(self.randomness),
        }


@dataclass(frozen=True, kw_only=True)
class Stage1ReuseDecision:
    """Semantic rerun/reuse decision for a Stage 1 substep."""

    run_id: str
    rerun_id: str | None
    artifact_namespace: str
    substep_id: str
    identity_key: str
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
            "identity_key": self.identity_key,
            "action": self.action,
            "reason": self.reason,
            "identity_fingerprint": self.identity_fingerprint,
        }


@dataclass(frozen=True, kw_only=True)
class Stage1ReuseManifestRecord:
    """Persisted semantic identity for one checkpointed Stage 1 execution unit."""

    substep_id: str
    identity_key: str
    identity_fingerprint: str
    identity_material: Mapping[str, Any]
    reuse_decision: Mapping[str, Any] | None = None
    checkpoint_summary: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest record."""

        return {
            "substep_id": self.substep_id,
            "identity_key": self.identity_key,
            "identity_fingerprint": self.identity_fingerprint,
            "identity_material": _json_safe(self.identity_material),
            "reuse_decision": _json_safe(self.reuse_decision or {}),
            "checkpoint_summary": _json_safe(self.checkpoint_summary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage1ReuseManifestRecord":
        """Load one manifest record or raise ValueError for invalid payloads."""

        substep_id = payload.get("substep_id")
        identity_key = payload.get("identity_key")
        identity_fingerprint = payload.get("identity_fingerprint")
        identity_material = payload.get("identity_material")
        if not isinstance(substep_id, str) or not substep_id:
            raise ValueError("reuse manifest record missing substep_id")
        if not isinstance(identity_key, str) or not identity_key:
            raise ValueError("reuse manifest record missing identity_key")
        if not isinstance(identity_fingerprint, str) or not identity_fingerprint:
            raise ValueError("reuse manifest record missing identity_fingerprint")
        if not isinstance(identity_material, Mapping):
            raise ValueError("reuse manifest record missing identity_material")
        reuse_decision = payload.get("reuse_decision")
        checkpoint_summary = payload.get("checkpoint_summary")
        return cls(
            substep_id=substep_id,
            identity_key=identity_key,
            identity_fingerprint=identity_fingerprint,
            identity_material=dict(identity_material),
            reuse_decision=dict(reuse_decision)
            if isinstance(reuse_decision, Mapping)
            else None,
            checkpoint_summary={
                str(key): int(value)
                for key, value in dict(checkpoint_summary or {}).items()
            }
            if isinstance(checkpoint_summary, Mapping)
            else {},
        )


@dataclass(frozen=True, kw_only=True)
class Stage1ReuseManifest:
    """Checkpoint-scoped semantic identity manifest for Stage 1 reruns."""

    branch: str
    commit_sha: str
    records: Mapping[str, Stage1ReuseManifestRecord] = field(default_factory=dict)
    schema_version: str = STAGE_1_REUSE_MANIFEST_SCHEMA_VERSION

    @classmethod
    def empty(cls, *, branch: str, commit_sha: str) -> "Stage1ReuseManifest":
        """Return an empty manifest for a checkpoint scope."""

        return cls(branch=branch, commit_sha=commit_sha)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        branch: str,
        commit_sha: str,
    ) -> "Stage1ReuseManifest":
        """Load a manifest or raise ValueError for invalid payloads."""

        if payload.get("schema_version") != STAGE_1_REUSE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 1 reuse manifest schema")
        records_payload = payload.get("records", ())
        if not isinstance(records_payload, list):
            raise ValueError("Stage 1 reuse manifest records must be a list")
        records: dict[str, Stage1ReuseManifestRecord] = {}
        for record_payload in records_payload:
            if not isinstance(record_payload, Mapping):
                raise ValueError("Stage 1 reuse manifest contains invalid records")
            record = Stage1ReuseManifestRecord.from_dict(record_payload)
            if record.identity_key in records:
                raise ValueError("Stage 1 reuse manifest contains duplicate keys")
            records[record.identity_key] = record
        return cls(
            branch=str(payload.get("branch") or branch),
            commit_sha=str(payload.get("commit_sha") or commit_sha),
            records=records,
        )

    def previous_identities(self) -> dict[str, str]:
        """Return prior semantic fingerprints keyed by execution identity key."""

        return {
            identity_key: record.identity_fingerprint
            for identity_key, record in self.records.items()
        }

    def with_record(
        self,
        record: Stage1ReuseManifestRecord,
    ) -> "Stage1ReuseManifest":
        """Return a manifest with one record added or replaced."""

        records = dict(self.records)
        records[record.identity_key] = record
        return Stage1ReuseManifest(
            branch=self.branch,
            commit_sha=self.commit_sha,
            records=records,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible manifest payload."""

        return {
            "schema_version": self.schema_version,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "records": [
                self.records[identity_key].to_dict()
                for identity_key in sorted(self.records)
            ],
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
                identity_key=material.identity_key,
                action="blocked",
                reason=material.blocked_reason,
                identity_fingerprint=fingerprint,
            )

        previous = self.previous_identities.get(material.identity_key)
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
            identity_key=material.identity_key,
            action=action,
            reason=reason,
            identity_fingerprint=fingerprint,
        )


__all__ = [
    "STAGE_1_REUSE_MANIFEST_FILENAME",
    "STAGE_1_REUSE_MANIFEST_SCHEMA_VERSION",
    "Stage1IdentityMaterial",
    "Stage1RerunPlanner",
    "Stage1ReuseAction",
    "Stage1ReuseDecision",
    "Stage1ReuseManifest",
    "Stage1ReuseManifestRecord",
]
