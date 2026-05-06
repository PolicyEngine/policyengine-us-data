"""Shared semantic contract types for stage handoffs.

Stage contracts are semantic records for canonical stage boundaries. They are
intended to become the durable handoff shape for stage inputs, outputs,
fingerprints, substage records, and execution bookkeeping.

Legacy step manifests are operational execution records to replace during
runtime wiring. Substages remain embedded records inside a parent stage
contract, not independent persisted contracts.

This package is intentionally dependency-light and has no Modal dependency.
"""

from .core import (
    CONTRACT_FINGERPRINT_ALGORITHM,
    CONTRACT_SCHEMA_VERSION,
    DIAGNOSTIC_SEVERITIES,
    EXECUTION_STATUSES,
    REUSE_DECISIONS,
    SUBSTAGE_REUSE_MODES,
    SUBSTAGE_STATUSES,
    VALIDATION_FINDING_STATUSES,
    VALIDATION_REPORT_STATUSES,
    ArtifactRef,
    DiagnosticRef,
    DiagnosticSeverity,
    ExecutionRecord,
    ExecutionStatus,
    Fingerprint,
    ReuseDecision,
    ReuseSummary,
    StageContract,
    SubstageRecord,
    SubstageReuseMode,
    SubstageStatus,
    ValidationFinding,
    ValidationFindingStatus,
    ValidationReport,
    ValidationReportStatus,
)
from .fingerprints import canonicalize_for_fingerprint, fingerprint_material
from .io import contract_from_json, contract_to_json, read_contract, write_contract
from .stages import (
    CANONICAL_STAGE_IDS,
    CONTRACT_TYPE_BY_STAGE_ID,
    STAGE_1_BUILD_DATASETS,
    STAGE_2_BUILD_CALIBRATION_PACKAGE,
    STAGE_3_FIT_WEIGHTS,
    STAGE_4_BUILD_OUTPUTS,
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
    SUBSTAGE_IDS_BY_STAGE_ID,
    contract_type_for_stage,
    is_canonical_stage_id,
    is_canonical_substage_id,
    substage_ids_for_stage,
)

__all__ = [
    "CONTRACT_FINGERPRINT_ALGORITHM",
    "CONTRACT_SCHEMA_VERSION",
    "DIAGNOSTIC_SEVERITIES",
    "EXECUTION_STATUSES",
    "REUSE_DECISIONS",
    "SUBSTAGE_REUSE_MODES",
    "SUBSTAGE_STATUSES",
    "VALIDATION_FINDING_STATUSES",
    "VALIDATION_REPORT_STATUSES",
    "ArtifactRef",
    "CANONICAL_STAGE_IDS",
    "CONTRACT_TYPE_BY_STAGE_ID",
    "DiagnosticRef",
    "DiagnosticSeverity",
    "ExecutionRecord",
    "ExecutionStatus",
    "Fingerprint",
    "ReuseDecision",
    "ReuseSummary",
    "STAGE_1_BUILD_DATASETS",
    "STAGE_2_BUILD_CALIBRATION_PACKAGE",
    "STAGE_3_FIT_WEIGHTS",
    "STAGE_4_BUILD_OUTPUTS",
    "STAGE_5_VALIDATE_AND_PROMOTE_RELEASE",
    "SUBSTAGE_IDS_BY_STAGE_ID",
    "StageContract",
    "SubstageRecord",
    "SubstageReuseMode",
    "SubstageStatus",
    "ValidationFinding",
    "ValidationFindingStatus",
    "ValidationReport",
    "ValidationReportStatus",
    "canonicalize_for_fingerprint",
    "contract_from_json",
    "contract_to_json",
    "contract_type_for_stage",
    "fingerprint_material",
    "is_canonical_stage_id",
    "is_canonical_substage_id",
    "read_contract",
    "substage_ids_for_stage",
    "write_contract",
]
