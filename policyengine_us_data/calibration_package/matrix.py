"""Typed Stage 2 matrix build boundary and summary writer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.utils.manifest import compute_file_checksum

from .specs import MATRIX_SUMMARY_FILENAME

if TYPE_CHECKING:
    from policyengine_us_data.stage_contracts.calibration_package_schema import (
        MatrixBuildSummary,
    )

MATRIX_BUILD_SCHEMA_VERSION = 1
CHUNK_CACHE_MANIFEST_SCHEMA_VERSION = 2
CHUNK_EXECUTION_SCHEMA_VERSION = 1
MatrixBuilderMode = Literal["precompute", "chunked"]
MATRIX_BUILDER_MODES = frozenset({"precompute", "chunked"})
ChunkExecutionStatus = Literal["completed", "cached", "failed"]
CHUNK_EXECUTION_STATUSES = frozenset({"completed", "cached", "failed"})


@dataclass(frozen=True, kw_only=True)
class ChunkCacheManifest:
    """Structured lineage manifest for resumable chunked matrix shards."""

    schema_version: int
    lineage_signature: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_positive_int(self.schema_version, "schema_version")
        if not isinstance(self.lineage_signature, Mapping):
            raise ValueError("lineage_signature must be a mapping")

    @classmethod
    def from_signature(cls, signature: Mapping[str, Any]) -> "ChunkCacheManifest":
        """Create a cache manifest from a computed chunk lineage signature."""

        return cls(
            schema_version=CHUNK_CACHE_MANIFEST_SCHEMA_VERSION,
            lineage_signature=dict(signature),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChunkCacheManifest":
        """Parse a chunk cache manifest from JSON-compatible material."""

        if not isinstance(data, Mapping):
            raise ValueError("chunk cache manifest must be a mapping")
        if "signature" in data and "lineage_signature" not in data:
            # PR-2f caches wrote {"signature": ...}; preserve compatibility
            # while the schema name becomes explicit.
            lineage = data["signature"]
        else:
            lineage = data.get("lineage_signature")
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            lineage_signature=_mapping_value(lineage, "lineage_signature"),
        )

    @classmethod
    def read(cls, path: str | Path) -> "ChunkCacheManifest":
        """Read a structured chunk cache manifest from disk."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate_lineage(
        self,
        expected_signature: Mapping[str, Any],
        *,
        cache_root: str | Path | None = None,
    ) -> None:
        """Reject resume when stored lineage differs from expected lineage."""

        fatal = _lineage_mismatches(self.lineage_signature, expected_signature)
        if fatal:
            root = Path(cache_root) if cache_root is not None else "chunk cache"
            joined = "; ".join(fatal)
            raise ValueError(f"Chunk cache lineage mismatch for {root}: {joined}")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible cache manifest material."""

        return {
            "lineage_signature": dict(self.lineage_signature),
            "schema_version": self.schema_version,
        }

    def write(self, path: str | Path) -> Path:
        """Write the cache manifest and return the path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path


@dataclass(frozen=True, kw_only=True)
class ChunkBuildRequest:
    """Typed request consumed by a chunked matrix worker."""

    schema_version: int
    run_id: str
    chunk_ids: tuple[int, ...]
    chunk_root: str
    state_path: str
    resume_chunks: bool
    lineage_signature: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_positive_int(self.schema_version, "schema_version")
        _validate_string(self.run_id, "run_id")
        _validate_string(self.chunk_root, "chunk_root")
        _validate_string(self.state_path, "state_path")
        _validate_bool(self.resume_chunks, "resume_chunks")
        if not isinstance(self.lineage_signature, Mapping):
            raise ValueError("lineage_signature must be a mapping")
        if not isinstance(self.chunk_ids, tuple) or not self.chunk_ids:
            raise ValueError("chunk_ids must be a non-empty tuple")
        for chunk_id in self.chunk_ids:
            _validate_non_negative_int(chunk_id, "chunk_ids")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChunkBuildRequest":
        """Parse a chunk worker request from JSON-compatible material."""

        if not isinstance(data, Mapping):
            raise ValueError("chunk build request must be a mapping")
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            run_id=_string_value(data.get("run_id"), "run_id"),
            chunk_ids=tuple(
                _non_negative_int_value(chunk_id, "chunk_ids")
                for chunk_id in data.get("chunk_ids", ())
            ),
            chunk_root=_string_value(data.get("chunk_root"), "chunk_root"),
            state_path=_string_value(data.get("state_path"), "state_path"),
            resume_chunks=_bool_value(data.get("resume_chunks"), "resume_chunks"),
            lineage_signature=_mapping_value(
                data.get("lineage_signature"),
                "lineage_signature",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible worker request material."""

        return {
            "chunk_ids": list(self.chunk_ids),
            "chunk_root": self.chunk_root,
            "lineage_signature": dict(self.lineage_signature),
            "resume_chunks": self.resume_chunks,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "state_path": self.state_path,
        }


@dataclass(frozen=True, kw_only=True)
class ChunkExecutionResult:
    """Structured per-chunk progress or error metadata."""

    schema_version: int
    run_id: str
    chunk_id: int
    status: ChunkExecutionStatus
    nnz: int | None = None
    cached: bool = False
    error: str | None = None
    traceback: str | None = None
    n_households: int | None = None
    n_persons: int | None = None
    unique_states: int | None = None
    unique_counties: int | None = None
    unique_cds: int | None = None

    def __post_init__(self) -> None:
        _validate_positive_int(self.schema_version, "schema_version")
        _validate_string(self.run_id, "run_id")
        _validate_non_negative_int(self.chunk_id, "chunk_id")
        _validate_bool(self.cached, "cached")
        if self.status not in CHUNK_EXECUTION_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(CHUNK_EXECUTION_STATUSES)}"
            )
        _validate_optional_non_negative_int(self.nnz, "nnz")
        for key in (
            "n_households",
            "n_persons",
            "unique_states",
            "unique_counties",
            "unique_cds",
        ):
            _validate_optional_non_negative_int(getattr(self, key), key)
        for key in ("error", "traceback"):
            value = getattr(self, key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{key} must be a string or None")
        if self.status == "failed":
            if not self.error:
                raise ValueError("failed chunk results require error")
            if self.nnz is not None:
                raise ValueError("failed chunk results must not report nnz")
        else:
            _validate_non_negative_int(self.nnz, "nnz")
            if self.error is not None or self.traceback is not None:
                raise ValueError("successful chunk results must not report errors")
        if self.status == "cached" and not self.cached:
            raise ValueError("cached chunk results require cached=True")
        if self.status == "completed" and self.cached:
            raise ValueError("completed chunk results require cached=False")

    @classmethod
    def from_chunk_result(
        cls,
        *,
        run_id: str,
        result: Any,
    ) -> "ChunkExecutionResult":
        """Normalize an assembler `ChunkResult` into manifest metadata."""

        cached = bool(getattr(result, "cached"))
        return cls(
            schema_version=CHUNK_EXECUTION_SCHEMA_VERSION,
            run_id=run_id,
            chunk_id=int(getattr(result, "chunk_id")),
            status="cached" if cached else "completed",
            nnz=int(getattr(result, "nnz")),
            cached=cached,
            n_households=getattr(result, "n_households", None),
            n_persons=getattr(result, "n_persons", None),
            unique_states=getattr(result, "unique_states", None),
            unique_counties=getattr(result, "unique_counties", None),
            unique_cds=getattr(result, "unique_cds", None),
        )

    @classmethod
    def failure(
        cls,
        *,
        run_id: str,
        chunk_id: int,
        error: str,
        traceback: str | None = None,
    ) -> "ChunkExecutionResult":
        """Create failure metadata for a chunk worker error."""

        return cls(
            schema_version=CHUNK_EXECUTION_SCHEMA_VERSION,
            run_id=run_id,
            chunk_id=chunk_id,
            status="failed",
            cached=False,
            error=error,
            traceback=traceback,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChunkExecutionResult":
        """Parse per-chunk result metadata."""

        if not isinstance(data, Mapping):
            raise ValueError("chunk execution result must be a mapping")
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            run_id=_string_value(data.get("run_id"), "run_id"),
            chunk_id=_non_negative_int_value(data.get("chunk_id"), "chunk_id"),
            status=_string_value(data.get("status"), "status"),
            nnz=_optional_non_negative_int_value(data.get("nnz"), "nnz"),
            cached=_bool_value(data.get("cached", False), "cached"),
            error=data.get("error"),
            traceback=data.get("traceback"),
            n_households=_optional_non_negative_int_value(
                data.get("n_households"),
                "n_households",
            ),
            n_persons=_optional_non_negative_int_value(
                data.get("n_persons"),
                "n_persons",
            ),
            unique_states=_optional_non_negative_int_value(
                data.get("unique_states"),
                "unique_states",
            ),
            unique_counties=_optional_non_negative_int_value(
                data.get("unique_counties"),
                "unique_counties",
            ),
            unique_cds=_optional_non_negative_int_value(
                data.get("unique_cds"),
                "unique_cds",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible chunk result material."""

        return {
            "cached": self.cached,
            "chunk_id": self.chunk_id,
            "error": self.error,
            "n_households": self.n_households,
            "n_persons": self.n_persons,
            "nnz": self.nnz,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "status": self.status,
            "traceback": self.traceback,
            "unique_cds": self.unique_cds,
            "unique_counties": self.unique_counties,
            "unique_states": self.unique_states,
        }


@dataclass(frozen=True, kw_only=True)
class ChunkWorkerResult:
    """Structured result returned by one chunked matrix worker."""

    schema_version: int
    run_id: str
    chunk_ids: tuple[int, ...]
    chunk_results: tuple[ChunkExecutionResult, ...]

    def __post_init__(self) -> None:
        _validate_positive_int(self.schema_version, "schema_version")
        _validate_string(self.run_id, "run_id")
        if not isinstance(self.chunk_ids, tuple):
            raise ValueError("chunk_ids must be a tuple")
        for chunk_id in self.chunk_ids:
            _validate_non_negative_int(chunk_id, "chunk_ids")
        if not isinstance(self.chunk_results, tuple) or not all(
            isinstance(result, ChunkExecutionResult) for result in self.chunk_results
        ):
            raise ValueError("chunk_results must contain ChunkExecutionResult entries")
        result_ids = tuple(result.chunk_id for result in self.chunk_results)
        if sorted(result_ids) != sorted(self.chunk_ids):
            raise ValueError(
                "chunk_results must include one result per requested chunk"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChunkWorkerResult":
        """Parse worker result material."""

        if not isinstance(data, Mapping):
            raise ValueError("chunk worker result must be a mapping")
        chunk_results = tuple(
            ChunkExecutionResult.from_dict(item)
            for item in data.get("chunk_results", ())
        )
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            run_id=_string_value(data.get("run_id"), "run_id"),
            chunk_ids=tuple(
                _non_negative_int_value(chunk_id, "chunk_ids")
                for chunk_id in data.get("chunk_ids", ())
            ),
            chunk_results=chunk_results,
        )

    @property
    def errors(self) -> tuple[ChunkExecutionResult, ...]:
        """Return failed per-chunk results."""

        return tuple(
            result for result in self.chunk_results if result.status == "failed"
        )

    @property
    def completed_count(self) -> int:
        """Return the number of completed or cached chunks."""

        return len(self.chunk_results) - len(self.errors)

    @property
    def cached_count(self) -> int:
        """Return the number of cached chunks."""

        return sum(1 for result in self.chunk_results if result.status == "cached")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible worker result material."""

        errors = [
            {
                "chunk_id": result.chunk_id,
                "error": result.error,
                "traceback": result.traceback,
            }
            for result in self.errors
        ]
        return {
            "cached_count": self.cached_count,
            "chunk_ids": list(self.chunk_ids),
            "chunk_results": [result.to_dict() for result in self.chunk_results],
            "completed_count": self.completed_count,
            "error_count": len(errors),
            "errors": errors,
            "nnz_per_chunk": [
                result.nnz for result in self.chunk_results if result.status != "failed"
            ],
            "run_id": self.run_id,
            "schema_version": self.schema_version,
        }


def chunk_result_manifest_path(chunk_root: str | Path, chunk_id: int) -> Path:
    """Return the per-chunk result manifest path for a chunk root."""

    _validate_non_negative_int(chunk_id, "chunk_id")
    return Path(chunk_root) / "chunk_results" / f"chunk_{chunk_id:06d}.json"


def write_chunk_result_manifest(
    chunk_root: str | Path,
    result: ChunkExecutionResult,
) -> Path:
    """Write one structured chunk result manifest and return the path."""

    output_path = chunk_result_manifest_path(chunk_root, result.chunk_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


@pipeline_node(
    PipelineNode(
        id="stage2_matrix_build_spec",
        label="Stage 2 Matrix Build Spec",
        node_type="library",
        description="Capture the runtime choices that determine Stage 2 calibration matrix materialization.",
        source_file="policyengine_us_data/calibration_package/matrix.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_matrix.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class MatrixBuildSpec:
    """Runtime choices that control Stage 2 matrix materialization."""

    matrix_builder: MatrixBuilderMode
    base_n_records: int
    n_clones: int
    chunk_size: int | None = None
    chunk_dir: str | None = None
    keep_chunks: bool = False
    resume_chunks: bool = False
    rerandomize_takeup: bool = True
    parallel_matrix: bool = False
    num_matrix_workers: int | None = None
    county_level: bool = True
    workers: int | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.matrix_builder not in MATRIX_BUILDER_MODES:
            raise ValueError(
                f"matrix_builder must be one of {sorted(MATRIX_BUILDER_MODES)}"
            )
        _validate_positive_int(self.base_n_records, "base_n_records")
        _validate_positive_int(self.n_clones, "n_clones")
        _validate_bool(self.keep_chunks, "keep_chunks")
        _validate_bool(self.resume_chunks, "resume_chunks")
        _validate_bool(self.rerandomize_takeup, "rerandomize_takeup")
        _validate_bool(self.parallel_matrix, "parallel_matrix")
        _validate_bool(self.county_level, "county_level")
        _validate_optional_positive_int(self.chunk_size, "chunk_size")
        _validate_optional_positive_int(self.num_matrix_workers, "num_matrix_workers")
        _validate_optional_positive_int(self.workers, "workers")
        if self.chunk_dir is not None and not isinstance(self.chunk_dir, str):
            raise ValueError("chunk_dir must be a string or None")
        if self.run_id is not None and not isinstance(self.run_id, str):
            raise ValueError("run_id must be a string or None")
        if self.matrix_builder == "chunked":
            if self.chunk_size is None:
                raise ValueError("chunk_size is required for chunked matrix builds")
            if self.workers is not None:
                raise ValueError("workers must be None for chunked matrix builds")
        else:
            if self.chunk_size is not None:
                raise ValueError("chunk_size must be None for precompute matrix builds")
            if self.chunk_dir is not None:
                raise ValueError("chunk_dir must be None for precompute matrix builds")
            if self.keep_chunks or self.resume_chunks:
                raise ValueError("chunk cache flags require chunked matrix builds")
            if self.parallel_matrix:
                raise ValueError("parallel_matrix requires chunked matrix builds")
            if self.num_matrix_workers is not None:
                raise ValueError(
                    "num_matrix_workers must be None for precompute matrix builds"
                )
            if self.workers is None:
                raise ValueError("workers is required for precompute matrix builds")
        if self.parallel_matrix:
            if self.num_matrix_workers is None:
                raise ValueError(
                    "num_matrix_workers is required when parallel_matrix is true"
                )
            if not self.run_id:
                raise ValueError("run_id is required when parallel_matrix is true")
        elif self.num_matrix_workers is not None:
            raise ValueError(
                "num_matrix_workers must be None when parallel_matrix is false"
            )

    @classmethod
    def from_runtime_args(
        cls,
        *,
        chunked_matrix: bool,
        base_n_records: int,
        n_clones: int,
        chunk_size: int,
        chunk_dir: str | None,
        keep_chunks: bool,
        resume_chunks: bool,
        rerandomize_takeup: bool,
        parallel: bool,
        num_matrix_workers: int,
        county_level: bool,
        workers: int,
        run_id: str | None,
    ) -> "MatrixBuildSpec":
        """Build a matrix spec from `run_calibration` runtime arguments."""

        parallel_matrix = bool(chunked_matrix and parallel)
        return cls(
            matrix_builder="chunked" if chunked_matrix else "precompute",
            base_n_records=base_n_records,
            n_clones=n_clones,
            chunk_size=chunk_size if chunked_matrix else None,
            chunk_dir=chunk_dir if chunked_matrix else None,
            keep_chunks=keep_chunks if chunked_matrix else False,
            resume_chunks=resume_chunks if chunked_matrix else False,
            rerandomize_takeup=rerandomize_takeup,
            parallel_matrix=parallel_matrix,
            num_matrix_workers=num_matrix_workers if parallel_matrix else None,
            county_level=county_level if not chunked_matrix else True,
            workers=workers if not chunked_matrix else None,
            run_id=run_id or None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible spec material."""

        return {
            "base_n_records": self.base_n_records,
            "chunk_dir": self.chunk_dir,
            "chunk_size": self.chunk_size,
            "county_level": self.county_level,
            "keep_chunks": self.keep_chunks,
            "matrix_builder": self.matrix_builder,
            "n_clones": self.n_clones,
            "num_matrix_workers": self.num_matrix_workers,
            "parallel_matrix": self.parallel_matrix,
            "rerandomize_takeup": self.rerandomize_takeup,
            "resume_chunks": self.resume_chunks,
            "run_id": self.run_id,
            "workers": self.workers,
        }


@pipeline_node(
    PipelineNode(
        id="stage2_matrix_build_result",
        label="Stage 2 Matrix Build Result",
        node_type="library",
        description="Normalize standard and chunked matrix outputs with target order, chunk lineage, and a compact summary artifact.",
        source_file="policyengine_us_data/calibration_package/matrix.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_out=[MATRIX_SUMMARY_FILENAME],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_matrix.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class MatrixBuildResult:
    """Normalized output from standard or chunked matrix construction."""

    spec: MatrixBuildSpec
    targets_df: Any
    X_sparse: Any
    target_names: tuple[str, ...]
    chunk_manifest_path: Path | None = None
    chunk_shard_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        matrix_shape = _matrix_shape(self.X_sparse)
        if not hasattr(self.X_sparse, "nnz"):
            raise ValueError("X_sparse must expose nnz")
        n_targets, n_columns = matrix_shape
        target_row_count = int(len(self.targets_df))
        target_name_count = int(len(self.target_names))
        if target_row_count != n_targets:
            raise ValueError("targets_df row count must match matrix target rows")
        if target_name_count != n_targets:
            raise ValueError("target_names count must match matrix target rows")
        expected_columns = self.spec.base_n_records * self.spec.n_clones
        if n_columns != expected_columns:
            raise ValueError("matrix column count must equal base_n_records * n_clones")
        if self.chunk_manifest_path is not None and not isinstance(
            self.chunk_manifest_path,
            Path,
        ):
            raise ValueError("chunk_manifest_path must be a Path or None")
        if not all(isinstance(path, Path) for path in self.chunk_shard_paths):
            raise ValueError("chunk_shard_paths must contain Path entries")

    @classmethod
    def from_builder_output(
        cls,
        *,
        spec: MatrixBuildSpec,
        targets_df: Any,
        X_sparse: Any,
        target_names: list[str] | tuple[str, ...],
        chunk_manifest_path: Path | None = None,
        chunk_shard_paths: tuple[Path, ...] = (),
    ) -> "MatrixBuildResult":
        """Return a normalized result from legacy builder tuple output."""

        return cls(
            spec=spec,
            targets_df=targets_df,
            X_sparse=X_sparse,
            target_names=tuple(str(name) for name in target_names),
            chunk_manifest_path=chunk_manifest_path,
            chunk_shard_paths=tuple(chunk_shard_paths),
        )

    def summary(self) -> MatrixBuildSummary:
        """Return the compact, contract-safe matrix summary."""

        from policyengine_us_data.stage_contracts.calibration_package_schema import (
            MatrixBuildSummary,
        )

        n_targets, n_columns = _matrix_shape(self.X_sparse)
        nnz = int(self.X_sparse.nnz)
        density = nnz / (n_targets * n_columns) if n_targets * n_columns else 0.0
        manifest_path = self.chunk_manifest_path
        manifest_sha = (
            f"sha256:{compute_file_checksum(manifest_path)}"
            if manifest_path is not None and manifest_path.exists()
            else None
        )
        return MatrixBuildSummary(
            schema_version=MATRIX_BUILD_SCHEMA_VERSION,
            matrix_shape=(n_targets, n_columns),
            matrix_nnz=nnz,
            matrix_density=float(density),
            n_targets=int(len(self.targets_df)),
            n_columns=n_columns,
            target_name_count=int(len(self.target_names)),
            target_order_sha256=_hash_target_order(self.target_names),
            base_n_records=self.spec.base_n_records,
            n_clones=self.spec.n_clones,
            matrix_builder=self.spec.matrix_builder,
            chunk_size=self.spec.chunk_size,
            chunk_dir=self.spec.chunk_dir,
            keep_chunks=self.spec.keep_chunks,
            resume_chunks=self.spec.resume_chunks,
            rerandomize_takeup=self.spec.rerandomize_takeup,
            parallel_matrix=self.spec.parallel_matrix,
            num_matrix_workers=self.spec.num_matrix_workers,
            county_level=self.spec.county_level,
            workers=self.spec.workers,
            chunk_manifest_path=(
                str(manifest_path) if manifest_path is not None else None
            ),
            chunk_manifest_sha256=manifest_sha,
            chunk_shard_count=len(self.chunk_shard_paths),
            chunk_shard_paths=tuple(str(path) for path in self.chunk_shard_paths),
        )

    def write_summary(self, path: str | Path) -> Path:
        """Write `matrix_summary.json` and return its path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.summary().to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path


@pipeline_node(
    PipelineNode(
        id="stage2_matrix_build_service",
        label="Stage 2 Matrix Build Service",
        node_type="library",
        description="Invoke the existing matrix builder engine through one Stage 2 service boundary.",
        source_file="policyengine_us_data/calibration_package/matrix.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_out=[MATRIX_SUMMARY_FILENAME],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_matrix.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class MatrixBuildService:
    """Adapter service around the current `UnifiedMatrixBuilder` engine."""

    builder: Any

    def build(
        self,
        *,
        spec: MatrixBuildSpec,
        geography: Any,
        sim: Any,
        target_filter: dict | None = None,
        hierarchical_domains: list[str] | None = None,
        sim_modifier: Any | None = None,
    ) -> MatrixBuildResult:
        """Build a matrix through the configured engine and normalize output."""

        if spec.matrix_builder == "chunked":
            targets_df, X_sparse, target_names = self.builder.build_matrix_chunked(
                geography=geography,
                sim=sim,
                target_filter=target_filter,
                hierarchical_domains=hierarchical_domains,
                chunk_size=spec.chunk_size,
                chunk_dir=spec.chunk_dir,
                keep_chunks=spec.keep_chunks,
                resume_chunks=spec.resume_chunks,
                rerandomize_takeup=spec.rerandomize_takeup,
                parallel=spec.parallel_matrix,
                num_matrix_workers=spec.num_matrix_workers or 1,
                run_id=spec.run_id or "",
            )
            return MatrixBuildResult.from_builder_output(
                spec=spec,
                targets_df=targets_df,
                X_sparse=X_sparse,
                target_names=target_names,
                chunk_manifest_path=_chunk_manifest_path(spec),
                chunk_shard_paths=_chunk_shard_paths(spec),
            )

        targets_df, X_sparse, target_names = self.builder.build_matrix(
            geography=geography,
            sim=sim,
            target_filter=target_filter,
            hierarchical_domains=hierarchical_domains,
            sim_modifier=sim_modifier,
            rerandomize_takeup=spec.rerandomize_takeup,
            county_level=spec.county_level,
            workers=spec.workers or 1,
        )
        return MatrixBuildResult.from_builder_output(
            spec=spec,
            targets_df=targets_df,
            X_sparse=X_sparse,
            target_names=target_names,
        )


def _chunk_manifest_path(spec: MatrixBuildSpec) -> Path | None:
    if spec.matrix_builder != "chunked" or spec.chunk_dir is None:
        return None
    return Path(spec.chunk_dir) / "chunk_manifest.json"


def _chunk_shard_paths(spec: MatrixBuildSpec) -> tuple[Path, ...]:
    if spec.matrix_builder != "chunked" or spec.chunk_dir is None:
        return ()
    coo_dir = Path(spec.chunk_dir) / "coo"
    if not coo_dir.exists():
        return ()
    return tuple(sorted(coo_dir.glob("chunk_*.npz")))


def _hash_target_order(target_names: tuple[str, ...]) -> str:
    payload = json.dumps(list(target_names), separators=(",", ":"), sort_keys=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matrix_shape(X_sparse: Any) -> tuple[int, int]:
    try:
        n_targets, n_columns = X_sparse.shape
    except (AttributeError, ValueError) as exc:
        raise ValueError("X_sparse must expose a two-dimensional shape") from exc
    return (int(n_targets), int(n_columns))


def _validate_bool(value: Any, key: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")


def _validate_string(value: Any, key: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")


def _validate_positive_int(value: Any, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def _validate_non_negative_int(value: Any, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")


def _validate_optional_positive_int(value: Any, key: str) -> None:
    if value is not None:
        _validate_positive_int(value, key)


def _validate_optional_non_negative_int(value: Any, key: str) -> None:
    if value is not None:
        _validate_non_negative_int(value, key)


def _string_value(value: Any, key: str) -> str:
    _validate_string(value, key)
    return value


def _bool_value(value: Any, key: str) -> bool:
    _validate_bool(value, key)
    return value


def _non_negative_int_value(value: Any, key: str) -> int:
    _validate_non_negative_int(value, key)
    return value


def _optional_non_negative_int_value(value: Any, key: str) -> int | None:
    if value is None:
        return None
    _validate_non_negative_int(value, key)
    return value


def _mapping_value(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _lineage_mismatches(
    stored_signature: Mapping[str, Any],
    expected_signature: Mapping[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in expected_signature.items():
        stored_value = stored_signature.get(key)
        if stored_value is None:
            mismatches.append(f"{key} missing from stored signature")
        elif stored_value != expected_value:
            mismatches.append(f"{key} expected {stored_value}, got {expected_value}")
    return mismatches


__all__ = [
    "CHUNK_CACHE_MANIFEST_SCHEMA_VERSION",
    "CHUNK_EXECUTION_SCHEMA_VERSION",
    "CHUNK_EXECUTION_STATUSES",
    "MATRIX_BUILD_SCHEMA_VERSION",
    "MATRIX_BUILDER_MODES",
    "ChunkBuildRequest",
    "ChunkCacheManifest",
    "ChunkExecutionResult",
    "ChunkExecutionStatus",
    "ChunkWorkerResult",
    "MatrixBuilderMode",
    "MatrixBuildResult",
    "MatrixBuildService",
    "MatrixBuildSpec",
    "chunk_result_manifest_path",
    "write_chunk_result_manifest",
]
