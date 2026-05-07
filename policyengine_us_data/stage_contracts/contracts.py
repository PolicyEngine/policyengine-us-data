"""Top-level stage contract records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._coercion import (
    freeze_mapping,
    freeze_sequence,
    jsonable_value,
    mapping_value,
    optional_string,
    optional_string_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_optional_instance,
    validate_schema_version,
)
from .artifacts import ArtifactRef
from .constants import CONTRACT_SCHEMA_VERSION, ContractPayload
from .diagnostics import DiagnosticRef
from .execution import ExecutionRecord
from .fingerprints import Fingerprint
from .stages import (
    contract_type_for_stage,
    is_canonical_stage_id,
    is_canonical_substage_id,
)
from .substages import SubstageRecord
from .validation import ValidationReport


@dataclass(frozen=True, kw_only=True)
class StageContract:
    """Durable semantic contract at a canonical stage boundary."""

    contract_type: str
    stage_id: str
    created_at: str
    fingerprint: Fingerprint
    execution: ExecutionRecord
    run_id: str | None = None
    code_sha: str | None = None
    package_version: str | None = None
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    substages: tuple[SubstageRecord, ...] = ()
    validation: ValidationReport | None = None
    diagnostics: tuple[DiagnosticRef, ...] = ()
    schema_version: str = CONTRACT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "contract_type",
            require_non_empty(self.contract_type, "contract_type"),
        )
        object.__setattr__(
            self,
            "stage_id",
            require_non_empty(self.stage_id, "stage_id"),
        )
        object.__setattr__(
            self,
            "created_at",
            require_non_empty(self.created_at, "created_at"),
        )
        object.__setattr__(
            self,
            "run_id",
            optional_string_value(self.run_id, "run_id"),
        )
        object.__setattr__(
            self,
            "code_sha",
            optional_string_value(self.code_sha, "code_sha"),
        )
        object.__setattr__(
            self,
            "package_version",
            optional_string_value(self.package_version, "package_version"),
        )
        if not is_canonical_stage_id(self.stage_id):
            raise ValueError(f"Invalid canonical stage_id: {self.stage_id!r}")
        expected_contract_type = contract_type_for_stage(self.stage_id)
        if self.contract_type != expected_contract_type:
            raise ValueError(
                "contract_type must match canonical stage "
                f"{self.stage_id!r}: {expected_contract_type!r}"
            )
        if not isinstance(self.fingerprint, Fingerprint):
            raise ValueError("fingerprint must be Fingerprint")
        if not isinstance(self.execution, ExecutionRecord):
            raise ValueError("execution must be ExecutionRecord")
        object.__setattr__(
            self,
            "inputs",
            freeze_sequence(self.inputs, "inputs", ArtifactRef),
        )
        object.__setattr__(
            self,
            "outputs",
            freeze_sequence(self.outputs, "outputs", ArtifactRef),
        )
        object.__setattr__(
            self,
            "substages",
            freeze_sequence(self.substages, "substages", SubstageRecord),
        )
        for substage in self.substages:
            if not is_canonical_substage_id(self.stage_id, substage.substage_id):
                raise ValueError(
                    "substage_id must belong to canonical stage "
                    f"{self.stage_id!r}: {substage.substage_id!r}"
                )
        validate_optional_instance(self.validation, "validation", ValidationReport)
        object.__setattr__(
            self,
            "diagnostics",
            freeze_sequence(self.diagnostics, "diagnostics", DiagnosticRef),
        )
        object.__setattr__(
            self,
            "parameters",
            freeze_mapping(self.parameters, "parameters"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "code_sha": self.code_sha,
            "package_version": self.package_version,
            "inputs": [jsonable_value(item) for item in self.inputs],
            "outputs": [jsonable_value(item) for item in self.outputs],
            "parameters": jsonable_value(self.parameters),
            "fingerprint": self.fingerprint.to_dict(),
            "substages": [jsonable_value(item) for item in self.substages],
            "execution": self.execution.to_dict(),
            "validation": (
                self.validation.to_dict() if self.validation is not None else None
            ),
            "diagnostics": [jsonable_value(item) for item in self.diagnostics],
            "metadata": jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageContract":
        return cls(
            contract_type=required_string(data, "contract_type"),
            schema_version=schema_version(data),
            stage_id=required_string(data, "stage_id"),
            run_id=optional_string(data, "run_id"),
            created_at=required_string(data, "created_at"),
            code_sha=optional_string(data, "code_sha"),
            package_version=optional_string(data, "package_version"),
            inputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("inputs", ())
            ),
            outputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("outputs", ())
            ),
            parameters=mapping_value(data, "parameters"),
            fingerprint=Fingerprint.from_dict(data["fingerprint"]),
            substages=tuple(
                SubstageRecord.from_dict(item) for item in data.get("substages", ())
            ),
            execution=ExecutionRecord.from_dict(data["execution"]),
            validation=(
                ValidationReport.from_dict(data["validation"])
                if data.get("validation") is not None
                else None
            ),
            diagnostics=tuple(
                DiagnosticRef.from_dict(item) for item in data.get("diagnostics", ())
            ),
            metadata=mapping_value(data, "metadata"),
        )
