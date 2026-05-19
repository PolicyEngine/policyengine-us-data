"""Typed schemas for Stage 2 calibration-package contract payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from policyengine_us_data.calibration_package.specs import (
    TARGET_CONFIG_IDENTITY_MODES,
)

GEOGRAPHY_ASSIGNMENT_SOURCE_KINDS = frozenset(
    {
        "calibration_package",
        "unavailable",
    }
)
GEOGRAPHY_ASSIGNMENT_SUMMARY_KEYS = frozenset(
    {
        "block_geoid_length",
        "block_geoid_sha256",
        "canonical_geography_sha256",
        "cd_geoid_length",
        "cd_geoid_sha256",
        "has_block_geoid",
        "has_cd_geoid",
        "n_clones",
        "n_records",
        "n_rows",
        "source_kind",
    }
)
CALIBRATION_PACKAGE_PARAMETER_KEYS = frozenset(
    {
        "chunk_size",
        "chunked_matrix",
        "n_clones",
        "num_matrix_workers",
        "parallel_matrix",
        "skip_county",
        "skip_source_impute",
        "skip_takeup_rerandomize",
        "target_config",
        "target_config_mode",
        "target_config_sha256",
        "workers",
    }
)
CALIBRATION_PACKAGE_SUMMARY_KEYS = frozenset(
    {
        "base_n_records",
        "block_geoid_length",
        "cd_geoid_length",
        "chunk_dir",
        "chunk_size",
        "dataset_sha256",
        "db_sha256",
        "has_block_geoid",
        "has_cd_geoid",
        "has_initial_weights",
        "matrix_builder",
        "matrix_density",
        "matrix_nnz",
        "matrix_shape",
        "n_clones",
        "n_columns",
        "n_targets",
        "package_scope",
        "seed",
        "target_config_path",
        "target_config_sha256",
        "target_name_count",
    }
)


@dataclass(frozen=True, kw_only=True)
class GeographyAssignmentSummary:
    """Canonical summary of package-backed Stage 2 geography assignment."""

    source_kind: str
    n_records: int | None
    n_clones: int | None
    n_rows: int | None
    has_block_geoid: bool
    has_cd_geoid: bool
    block_geoid_length: int | None
    cd_geoid_length: int | None
    block_geoid_sha256: str | None
    cd_geoid_sha256: str | None
    canonical_geography_sha256: str | None

    def __post_init__(self) -> None:
        if self.source_kind not in GEOGRAPHY_ASSIGNMENT_SOURCE_KINDS:
            raise ValueError(
                "source_kind must be one of "
                f"{sorted(GEOGRAPHY_ASSIGNMENT_SOURCE_KINDS)}"
            )
        _validate_optional_non_negative_int(self.n_records, "n_records")
        _validate_optional_non_negative_int(self.n_clones, "n_clones")
        _validate_optional_non_negative_int(self.n_rows, "n_rows")
        _validate_optional_non_negative_int(
            self.block_geoid_length,
            "block_geoid_length",
        )
        _validate_optional_non_negative_int(
            self.cd_geoid_length,
            "cd_geoid_length",
        )
        _validate_bool(self.has_block_geoid, "has_block_geoid")
        _validate_bool(self.has_cd_geoid, "has_cd_geoid")
        _validate_optional_sha256(self.block_geoid_sha256, "block_geoid_sha256")
        _validate_optional_sha256(self.cd_geoid_sha256, "cd_geoid_sha256")
        _validate_optional_sha256(
            self.canonical_geography_sha256,
            "canonical_geography_sha256",
        )
        if self.source_kind == "calibration_package":
            for key in (
                "n_records",
                "n_clones",
                "n_rows",
                "block_geoid_length",
                "cd_geoid_length",
                "block_geoid_sha256",
                "cd_geoid_sha256",
                "canonical_geography_sha256",
            ):
                if getattr(self, key) is None:
                    raise ValueError(
                        f"{key} is required for calibration_package geography"
                    )
            if not self.has_block_geoid or not self.has_cd_geoid:
                raise ValueError(
                    "calibration_package geography requires block and CD arrays"
                )
            if self.n_records * self.n_clones != self.n_rows:
                raise ValueError(
                    "n_records * n_clones must equal n_rows for geography summary"
                )
            if self.block_geoid_length != self.n_rows:
                raise ValueError("block_geoid_length must equal n_rows")
            if self.cd_geoid_length != self.n_rows:
                raise ValueError("cd_geoid_length must equal n_rows")
        else:
            if self.has_block_geoid or self.has_cd_geoid:
                raise ValueError("unavailable geography cannot report present arrays")
            for key in (
                "n_rows",
                "block_geoid_length",
                "cd_geoid_length",
                "block_geoid_sha256",
                "cd_geoid_sha256",
                "canonical_geography_sha256",
            ):
                if getattr(self, key) is not None:
                    raise ValueError(
                        f"{key} must be None when geography is unavailable"
                    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeographyAssignmentSummary":
        """Parse geography assignment summary JSON into a typed schema object."""

        if not isinstance(data, Mapping):
            raise ValueError("geography assignment summary must be a mapping")
        _require_exact_keys(
            data,
            "geography assignment summary",
            GEOGRAPHY_ASSIGNMENT_SUMMARY_KEYS,
        )
        return cls(
            source_kind=_required_string_field(data, "source_kind"),
            n_records=_optional_int_field(data, "n_records"),
            n_clones=_optional_int_field(data, "n_clones"),
            n_rows=_optional_int_field(data, "n_rows"),
            has_block_geoid=_required_bool_field(data, "has_block_geoid"),
            has_cd_geoid=_required_bool_field(data, "has_cd_geoid"),
            block_geoid_length=_optional_int_field(data, "block_geoid_length"),
            cd_geoid_length=_optional_int_field(data, "cd_geoid_length"),
            block_geoid_sha256=_optional_string_field(data, "block_geoid_sha256"),
            cd_geoid_sha256=_optional_string_field(data, "cd_geoid_sha256"),
            canonical_geography_sha256=_optional_string_field(
                data,
                "canonical_geography_sha256",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible geography assignment summary."""

        return {
            "block_geoid_length": self.block_geoid_length,
            "block_geoid_sha256": self.block_geoid_sha256,
            "canonical_geography_sha256": self.canonical_geography_sha256,
            "cd_geoid_length": self.cd_geoid_length,
            "cd_geoid_sha256": self.cd_geoid_sha256,
            "has_block_geoid": self.has_block_geoid,
            "has_cd_geoid": self.has_cd_geoid,
            "n_clones": self.n_clones,
            "n_records": self.n_records,
            "n_rows": self.n_rows,
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True, kw_only=True)
class CalibrationPackageParameters:
    """Canonical parameters that affect Stage 2 package construction."""

    workers: int | None
    n_clones: int
    target_config: str | None
    target_config_sha256: str | None
    target_config_mode: str | None
    skip_county: bool
    skip_source_impute: bool
    skip_takeup_rerandomize: bool
    chunked_matrix: bool
    chunk_size: int | None
    parallel_matrix: bool
    num_matrix_workers: int | None

    def __post_init__(self) -> None:
        _validate_positive_int(self.n_clones, "n_clones")
        _validate_optional_positive_int(self.workers, "workers")
        _validate_optional_positive_int(self.chunk_size, "chunk_size")
        _validate_optional_positive_int(
            self.num_matrix_workers,
            "num_matrix_workers",
        )
        _validate_bool(self.skip_county, "skip_county")
        _validate_bool(self.skip_source_impute, "skip_source_impute")
        _validate_bool(self.skip_takeup_rerandomize, "skip_takeup_rerandomize")
        _validate_bool(self.chunked_matrix, "chunked_matrix")
        _validate_bool(self.parallel_matrix, "parallel_matrix")
        if self.target_config is not None and not isinstance(self.target_config, str):
            raise ValueError("target_config must be a string or None")
        if self.target_config_sha256 is not None and not isinstance(
            self.target_config_sha256,
            str,
        ):
            raise ValueError("target_config_sha256 must be a string or None")
        if self.target_config_mode is not None:
            if not isinstance(self.target_config_mode, str):
                raise ValueError("target_config_mode must be a string or None")
            if self.target_config_mode not in TARGET_CONFIG_IDENTITY_MODES:
                raise ValueError(
                    "target_config_mode must be one of "
                    f"{sorted(TARGET_CONFIG_IDENTITY_MODES)}"
                )
        if self.target_config_mode == "all_active_targets":
            if self.target_config is not None or self.target_config_sha256 is not None:
                raise ValueError(
                    "all_active_targets target config parameters cannot include "
                    "a path or checksum"
                )
        if self.chunked_matrix:
            if self.workers is not None:
                raise ValueError("workers must be None when chunked_matrix is true")
            if self.chunk_size is None:
                raise ValueError("chunk_size is required when chunked_matrix is true")
        else:
            if self.workers is None:
                raise ValueError("workers is required when chunked_matrix is false")
            if self.chunk_size is not None:
                raise ValueError("chunk_size must be None when chunked_matrix is false")
            if self.parallel_matrix:
                raise ValueError("parallel_matrix requires chunked_matrix")
        if self.parallel_matrix and self.num_matrix_workers is None:
            raise ValueError(
                "num_matrix_workers is required when parallel_matrix is true"
            )
        if not self.parallel_matrix and self.num_matrix_workers is not None:
            raise ValueError(
                "num_matrix_workers must be None when parallel_matrix is false"
            )

    @classmethod
    def from_runtime_args(
        cls,
        *,
        workers: int,
        n_clones: int,
        target_config_path: str | None,
        skip_county: bool,
        skip_source_impute: bool,
        skip_takeup_rerandomize: bool,
        chunked_matrix: bool,
        chunk_size: int,
        parallel: bool,
        num_matrix_workers: int,
        target_config_sha256: str | None = None,
        target_config_mode: str | None = None,
    ) -> "CalibrationPackageParameters":
        """Build canonical Stage 2 parameters from runtime CLI arguments."""

        parallel_matrix = bool(chunked_matrix and parallel)
        resolved_mode = target_config_mode or (
            "all_active_targets" if target_config_path is None else "explicit"
        )
        return cls(
            workers=workers if not chunked_matrix else None,
            n_clones=n_clones,
            target_config=target_config_path,
            target_config_sha256=target_config_sha256,
            target_config_mode=resolved_mode,
            skip_county=skip_county,
            skip_source_impute=skip_source_impute,
            skip_takeup_rerandomize=skip_takeup_rerandomize,
            chunked_matrix=chunked_matrix,
            chunk_size=chunk_size if chunked_matrix else None,
            parallel_matrix=parallel_matrix,
            num_matrix_workers=num_matrix_workers if parallel_matrix else None,
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "CalibrationPackageParameters":
        """Parse Stage 2 parameter JSON into a typed schema object."""

        if not isinstance(data, Mapping):
            raise ValueError("calibration package parameters must be a mapping")
        _require_compatible_keys(
            data,
            "calibration package parameters",
            CALIBRATION_PACKAGE_PARAMETER_KEYS,
            legacy_optional_keys=frozenset(
                {"target_config_mode", "target_config_sha256"}
            ),
        )
        target_config = _optional_string_field(data, "target_config")
        target_config_mode = _optional_string_field(data, "target_config_mode")
        return cls(
            workers=_optional_int_field(data, "workers"),
            n_clones=_required_int_field(data, "n_clones"),
            target_config=target_config,
            target_config_sha256=_optional_string_field(
                data,
                "target_config_sha256",
            ),
            target_config_mode=target_config_mode
            or ("all_active_targets" if target_config is None else "explicit"),
            skip_county=_required_bool_field(data, "skip_county"),
            skip_source_impute=_required_bool_field(data, "skip_source_impute"),
            skip_takeup_rerandomize=_required_bool_field(
                data,
                "skip_takeup_rerandomize",
            ),
            chunked_matrix=_required_bool_field(data, "chunked_matrix"),
            chunk_size=_optional_int_field(data, "chunk_size"),
            parallel_matrix=_required_bool_field(data, "parallel_matrix"),
            num_matrix_workers=_optional_int_field(data, "num_matrix_workers"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible Stage 2 parameters."""

        return {
            "chunk_size": self.chunk_size,
            "chunked_matrix": self.chunked_matrix,
            "n_clones": self.n_clones,
            "num_matrix_workers": self.num_matrix_workers,
            "parallel_matrix": self.parallel_matrix,
            "skip_county": self.skip_county,
            "skip_source_impute": self.skip_source_impute,
            "skip_takeup_rerandomize": self.skip_takeup_rerandomize,
            "target_config": self.target_config,
            "target_config_mode": self.target_config_mode,
            "target_config_sha256": self.target_config_sha256,
            "workers": self.workers,
        }


@dataclass(frozen=True, kw_only=True)
class CalibrationPackageSummary:
    """Canonical summary of the persisted Stage 2 calibration package."""

    matrix_shape: tuple[int, int]
    matrix_nnz: int
    matrix_density: float
    n_targets: int
    n_columns: int
    target_name_count: int
    dataset_sha256: str | None
    db_sha256: str | None
    target_config_path: str | None
    target_config_sha256: str | None
    n_clones: int | None
    seed: int | None
    base_n_records: int | None
    package_scope: str | None
    matrix_builder: str | None
    chunk_size: int | None
    chunk_dir: str | None
    has_initial_weights: bool
    has_cd_geoid: bool
    has_block_geoid: bool
    cd_geoid_length: int | None
    block_geoid_length: int | None

    def __post_init__(self) -> None:
        if len(self.matrix_shape) != 2:
            raise ValueError("matrix_shape must have two entries")
        for index, value in enumerate(self.matrix_shape):
            _validate_non_negative_int(value, f"matrix_shape[{index}]")
        _validate_non_negative_int(self.matrix_nnz, "matrix_nnz")
        _validate_non_negative_float(self.matrix_density, "matrix_density")
        _validate_non_negative_int(self.n_targets, "n_targets")
        _validate_non_negative_int(self.n_columns, "n_columns")
        _validate_non_negative_int(self.target_name_count, "target_name_count")
        _validate_optional_non_negative_int(self.n_clones, "n_clones")
        _validate_optional_non_negative_int(self.seed, "seed")
        _validate_optional_non_negative_int(self.base_n_records, "base_n_records")
        _validate_optional_non_negative_int(self.chunk_size, "chunk_size")
        _validate_optional_non_negative_int(self.cd_geoid_length, "cd_geoid_length")
        _validate_optional_non_negative_int(
            self.block_geoid_length,
            "block_geoid_length",
        )
        _validate_bool(self.has_initial_weights, "has_initial_weights")
        _validate_bool(self.has_cd_geoid, "has_cd_geoid")
        _validate_bool(self.has_block_geoid, "has_block_geoid")
        for key in (
            "dataset_sha256",
            "db_sha256",
            "target_config_path",
            "target_config_sha256",
            "package_scope",
            "matrix_builder",
            "chunk_dir",
        ):
            value = getattr(self, key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{key} must be a string or None")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationPackageSummary":
        """Parse package summary JSON into a typed schema object."""

        if not isinstance(data, Mapping):
            raise ValueError("calibration package summary must be a mapping")
        _require_exact_keys(
            data,
            "calibration package summary",
            CALIBRATION_PACKAGE_SUMMARY_KEYS,
        )
        return cls(
            matrix_shape=_matrix_shape_field(data, "matrix_shape"),
            matrix_nnz=_required_int_field(data, "matrix_nnz"),
            matrix_density=_required_float_field(data, "matrix_density"),
            n_targets=_required_int_field(data, "n_targets"),
            n_columns=_required_int_field(data, "n_columns"),
            target_name_count=_required_int_field(data, "target_name_count"),
            dataset_sha256=_optional_string_field(data, "dataset_sha256"),
            db_sha256=_optional_string_field(data, "db_sha256"),
            target_config_path=_optional_string_field(data, "target_config_path"),
            target_config_sha256=_optional_string_field(
                data,
                "target_config_sha256",
            ),
            n_clones=_optional_int_field(data, "n_clones"),
            seed=_optional_int_field(data, "seed"),
            base_n_records=_optional_int_field(data, "base_n_records"),
            package_scope=_optional_string_field(data, "package_scope"),
            matrix_builder=_optional_string_field(data, "matrix_builder"),
            chunk_size=_optional_int_field(data, "chunk_size"),
            chunk_dir=_optional_string_field(data, "chunk_dir"),
            has_initial_weights=_required_bool_field(data, "has_initial_weights"),
            has_cd_geoid=_required_bool_field(data, "has_cd_geoid"),
            has_block_geoid=_required_bool_field(data, "has_block_geoid"),
            cd_geoid_length=_optional_int_field(data, "cd_geoid_length"),
            block_geoid_length=_optional_int_field(data, "block_geoid_length"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible package summary."""

        return {
            "base_n_records": self.base_n_records,
            "block_geoid_length": self.block_geoid_length,
            "cd_geoid_length": self.cd_geoid_length,
            "chunk_dir": self.chunk_dir,
            "chunk_size": self.chunk_size,
            "dataset_sha256": self.dataset_sha256,
            "db_sha256": self.db_sha256,
            "has_block_geoid": self.has_block_geoid,
            "has_cd_geoid": self.has_cd_geoid,
            "has_initial_weights": self.has_initial_weights,
            "matrix_builder": self.matrix_builder,
            "matrix_density": self.matrix_density,
            "matrix_nnz": self.matrix_nnz,
            "matrix_shape": self.matrix_shape,
            "n_clones": self.n_clones,
            "n_columns": self.n_columns,
            "n_targets": self.n_targets,
            "package_scope": self.package_scope,
            "seed": self.seed,
            "target_config_path": self.target_config_path,
            "target_config_sha256": self.target_config_sha256,
            "target_name_count": self.target_name_count,
        }


def _require_exact_keys(
    data: Mapping[str, Any],
    label: str,
    expected_keys: frozenset[str],
) -> None:
    _require_compatible_keys(
        data,
        label,
        expected_keys,
        legacy_optional_keys=frozenset(),
    )


def _require_compatible_keys(
    data: Mapping[str, Any],
    label: str,
    expected_keys: frozenset[str],
    *,
    legacy_optional_keys: frozenset[str],
) -> None:
    keys = {str(key) for key in data}
    missing = sorted((expected_keys - legacy_optional_keys) - keys)
    unexpected = sorted(keys - expected_keys)
    if missing:
        raise ValueError(f"{label} missing required key: {missing[0]}")
    if unexpected:
        raise ValueError(f"{label} has unexpected key: {unexpected[0]}")


def _required_int_field(data: Mapping[str, Any], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Calibration package field {key!r} must be an integer")
    return value


def _optional_int_field(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Calibration package field {key!r} must be an integer or None"
        )
    return value


def _optional_string_field(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Calibration package field {key!r} must be a string or None")
    return value


def _required_string_field(data: Mapping[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Calibration package field {key!r} must be a non-empty string"
        )
    return value


def _required_float_field(data: Mapping[str, Any], key: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Calibration package field {key!r} must be numeric")
    return float(value)


def _matrix_shape_field(data: Mapping[str, Any], key: str) -> tuple[int, int]:
    value = data[key]
    if not isinstance(value, tuple | list) or len(value) != 2:
        raise ValueError("matrix_shape must have two entries")
    first, second = value
    if isinstance(first, bool) or not isinstance(first, int):
        raise ValueError("matrix_shape entries must be integers")
    if isinstance(second, bool) or not isinstance(second, int):
        raise ValueError("matrix_shape entries must be integers")
    return (first, second)


def _required_bool_field(data: Mapping[str, Any], key: str) -> bool:
    value = data[key]
    _validate_bool(value, key)
    return value


def _validate_bool(value: Any, key: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"Calibration package field {key!r} must be a boolean")


def _validate_positive_int(value: Any, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Calibration package field {key!r} must be a positive integer"
        )


def _validate_optional_positive_int(value: Any, key: str) -> None:
    if value is not None:
        _validate_positive_int(value, key)


def _validate_non_negative_int(value: Any, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Calibration package field {key!r} must be a non-negative integer"
        )


def _validate_optional_non_negative_int(value: Any, key: str) -> None:
    if value is not None:
        _validate_non_negative_int(value, key)


def _validate_non_negative_float(value: Any, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Calibration package field {key!r} must be numeric")
    if not isfinite(float(value)) or value < 0:
        raise ValueError(
            f"Calibration package field {key!r} must be a finite non-negative number"
        )


def _validate_optional_sha256(value: Any, key: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"Calibration package field {key!r} must be a SHA-256 digest")
    if len(value) != len("sha256:") + 64:
        raise ValueError(f"Calibration package field {key!r} must be a SHA-256 digest")
