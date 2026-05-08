"""Shared semantic contract types for stage handoffs.

Stage contracts are semantic records for canonical stage boundaries. They are
intended to become the durable handoff shape for stage inputs, outputs,
fingerprints, substage records, and execution bookkeeping.

Legacy step manifests are operational execution records to replace during
runtime wiring. Substages remain embedded records inside a parent stage
contract, not independent persisted contracts.

This package is intentionally dependency-light and has no Modal dependency.
"""

from .artifacts import ArtifactRef
from .calibration_package import (
    CALIBRATION_PACKAGE_CONTRACT_FILENAME,
    CALIBRATION_PACKAGE_CONTRACT_TYPE,
    build_calibration_package_contract,
    load_calibration_package_payload,
    summarize_calibration_package,
    validate_calibration_package_contract,
    write_calibration_package_contract,
)
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
    DiagnosticSeverity,
    ExecutionStatus,
    ReuseDecision,
    SubstageReuseMode,
    SubstageStatus,
    ValidationFindingStatus,
    ValidationReportStatus,
)
from .contracts import StageContract
from .dataset_build import (
    DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
    DATASET_BUILD_OUTPUT_CONTRACT_TYPE,
    build_dataset_build_output_contract,
)
from .diagnostics import DiagnosticRef
from .execution import ExecutionRecord, ReuseSummary
from .fingerprints import (
    Fingerprint,
    canonicalize_for_fingerprint,
    fingerprint_material,
)
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
    "CANONICAL_STAGE_IDS",
    "CALIBRATION_PACKAGE_CONTRACT_FILENAME",
    "CALIBRATION_PACKAGE_CONTRACT_TYPE",
    "CONTRACT_TYPE_BY_STAGE_ID",
    "DATASET_BUILD_OUTPUT_CONTRACT_FILENAME",
    "DATASET_BUILD_OUTPUT_CONTRACT_TYPE",
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
    "build_calibration_package_contract",
    "build_dataset_build_output_contract",
    "canonicalize_for_fingerprint",
    "contract_from_json",
    "contract_to_json",
    "contract_type_for_stage",
    "fingerprint_material",
    "is_canonical_stage_id",
    "is_canonical_substage_id",
    "load_calibration_package_payload",
    "read_contract",
    "summarize_calibration_package",
    "substage_ids_for_stage",
    "validate_calibration_package_contract",
    "write_calibration_package_contract",
    "write_contract",
]
