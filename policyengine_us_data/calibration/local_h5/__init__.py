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
from .validation import (
    make_validation_error,
    summarize_validation_rows,
    tag_validation_errors,
    validation_geo_level_for_area_type,
)
from .weights import CloneWeightMatrix

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
    "PolicyEngineDatasetReader",
    "PolicyEngineVariableArrayProvider",
    "PublishingInputBundle",
    "ReindexedEntities",
    "SourceDatasetSnapshot",
    "ValidationIssue",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationStatus",
    "WorkerResult",
    "make_validation_error",
    "partition_weighted_work_items",
    "require_calibration_package_path",
    "summarize_validation_rows",
    "tag_validation_errors",
    "validation_geo_level_for_area_type",
    "work_item_key",
]
