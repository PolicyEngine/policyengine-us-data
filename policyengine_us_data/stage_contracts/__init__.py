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
    "DiagnosticRef",
    "DiagnosticSeverity",
    "ExecutionRecord",
    "ExecutionStatus",
    "Fingerprint",
    "ReuseDecision",
    "ReuseSummary",
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
    "fingerprint_material",
    "read_contract",
    "write_contract",
]
