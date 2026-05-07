"""Compatibility exports for semantic stage contract types."""

from __future__ import annotations

from .artifacts import ArtifactRef
from .constants import (
    CONTRACT_FINGERPRINT_ALGORITHM,
    CONTRACT_SCHEMA_VERSION,
    DIAGNOSTIC_SEVERITIES,
    EXECUTION_STATUSES,
    REUSE_DECISIONS,
    SUBSTAGE_REUSE_MODES,
    SUBSTAGE_STATUSES,
    VALIDATION_FINDING_STATUSES,
    VALIDATION_REPORT_STATUSES,
    ContractPayload,
    DiagnosticSeverity,
    ExecutionStatus,
    ReuseDecision,
    SubstageReuseMode,
    SubstageStatus,
    ValidationFindingStatus,
    ValidationReportStatus,
)
from .contracts import StageContract
from .diagnostics import DiagnosticRef
from .execution import ExecutionRecord, ReuseSummary
from .fingerprints import Fingerprint
from .substages import SubstageRecord
from .validation import ValidationFinding, ValidationReport

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
    "ContractPayload",
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
]
