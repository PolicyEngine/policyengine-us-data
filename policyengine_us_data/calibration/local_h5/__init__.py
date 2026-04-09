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
    "PolicyEngineDatasetReader",
    "PolicyEngineVariableArrayProvider",
    "PublishingInputBundle",
    "ReindexedEntities",
    "SourceDatasetSnapshot",
    "USAugmentationService",
    "ValidationIssue",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationStatus",
    "WorkerResult",
    "infer_clone_count_from_weight_length",
    "make_validation_error",
    "partition_weighted_work_items",
    "require_calibration_package_path",
    "summarize_validation_rows",
    "tag_validation_errors",
    "validation_geo_level_for_area_type",
    "VariableCloner",
    "VariableExportPolicy",
    "work_item_key",
]
