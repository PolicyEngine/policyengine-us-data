"""Internal contracts and components for the local H5 refactor."""

from .contracts import (
    AreaBuildRequest,
    AreaBuildResult,
    AreaFilter,
    BuildStatus,
    FilterOp,
    PublishingInputBundle,
    ValidationIssue,
    ValidationPolicy,
    ValidationResult,
    ValidationStatus,
    WorkerResult,
)
from .builder import LocalAreaBuildArtifacts, LocalAreaDatasetBuilder
from .entity_graph import EntityGraph, EntityGraphExtractor
from .fingerprinting import (
    FingerprintComponents,
    FingerprintInputs,
    FingerprintRecord,
    FingerprintService,
)
from .reindexing import EntityReindexer, ReindexedEntities
from .selection import AreaSelector, CloneSelection
from .partitioning import partition_weighted_work_items, work_item_key
from .package_geography import (
    CalibrationPackageGeographyLoader,
    LoadedPackageGeography,
    require_calibration_package_path,
)
from .source_dataset import (
    PolicyEngineDatasetReader,
    PolicyEngineVariableArrayProvider,
    SourceDatasetSnapshot,
)
from .us_augmentations import USAugmentationService
from .validation import (
    make_validation_error,
    summarize_validation_rows,
    tag_validation_errors,
    validation_geo_level_for_area_type,
)
from .variables import H5Payload, VariableCloner, VariableExportPolicy
from .weights import CloneWeightMatrix, infer_clone_count_from_weight_length
from .worker_service import (
    LocalH5WorkerService,
    ValidationContext,
    WorkerSession,
    build_request_from_work_item,
    build_requests_from_work_items,
    load_validation_context,
    validate_in_subprocess,
    validate_output_subprocess,
    worker_result_to_legacy_dict,
)
from .writer import H5Writer

__all__ = [
    "AreaBuildRequest",
    "AreaBuildResult",
    "AreaFilter",
    "AreaSelector",
    "BuildStatus",
    "CalibrationPackageGeographyLoader",
    "CloneSelection",
    "CloneWeightMatrix",
    "EntityGraph",
    "EntityGraphExtractor",
    "EntityReindexer",
    "FingerprintComponents",
    "FingerprintInputs",
    "FingerprintRecord",
    "FingerprintService",
    "FilterOp",
    "LoadedPackageGeography",
    "H5Payload",
    "H5Writer",
    "LocalAreaBuildArtifacts",
    "LocalAreaDatasetBuilder",
    "PolicyEngineDatasetReader",
    "PolicyEngineVariableArrayProvider",
    "PublishingInputBundle",
    "ReindexedEntities",
    "SourceDatasetSnapshot",
    "USAugmentationService",
    "ValidationIssue",
    "ValidationContext",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationStatus",
    "WorkerResult",
    "WorkerSession",
    "LocalH5WorkerService",
    "build_request_from_work_item",
    "build_requests_from_work_items",
    "infer_clone_count_from_weight_length",
    "load_validation_context",
    "make_validation_error",
    "partition_weighted_work_items",
    "require_calibration_package_path",
    "summarize_validation_rows",
    "tag_validation_errors",
    "validate_in_subprocess",
    "validate_output_subprocess",
    "validation_geo_level_for_area_type",
    "VariableCloner",
    "VariableExportPolicy",
    "worker_result_to_legacy_dict",
    "work_item_key",
]
