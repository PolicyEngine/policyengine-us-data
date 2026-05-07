"""Shared constants and type aliases for stage contracts."""

from __future__ import annotations

from typing import Any, Literal, get_args

CONTRACT_SCHEMA_VERSION = "1"
CONTRACT_FINGERPRINT_ALGORITHM = "sha256-canonical-json-v1"

SubstageStatus = Literal[
    "planned",
    "running",
    "completed",
    "skipped",
    "failed",
    "not_run",
]
SubstageReuseMode = Literal[
    "observed_only",
    "checkpointable",
    "reusable",
    "handoff",
]
ExecutionStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "reused",
    "partially_reused",
    "skipped",
]
ReuseDecision = Literal[
    "computed",
    "reused",
    "partially_reused",
    "invalidated",
    "failed",
    "not_applicable",
]
DiagnosticSeverity = Literal["info", "warning", "error"]
ValidationFindingStatus = Literal["pass", "warn", "fail"]
ValidationReportStatus = Literal["pass", "warn", "fail", "not_run"]

SUBSTAGE_STATUSES = frozenset(get_args(SubstageStatus))
SUBSTAGE_REUSE_MODES = frozenset(get_args(SubstageReuseMode))
EXECUTION_STATUSES = frozenset(get_args(ExecutionStatus))
REUSE_DECISIONS = frozenset(get_args(ReuseDecision))
DIAGNOSTIC_SEVERITIES = frozenset(get_args(DiagnosticSeverity))
VALIDATION_FINDING_STATUSES = frozenset(get_args(ValidationFindingStatus))
VALIDATION_REPORT_STATUSES = frozenset(get_args(ValidationReportStatus))

ContractPayload = dict[str, Any]
