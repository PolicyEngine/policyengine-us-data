"""Typed Stage 2 geography assignment boundary and summary writer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.utils.geography_checksum import (
    canonical_geography_checksum,
    hash_string_array,
)

from .specs import GEOGRAPHY_ASSIGNMENT_SUMMARY_FILENAME

GEOGRAPHY_ASSIGNMENT_SCHEMA_VERSION = 1
GEOGRAPHY_ASSIGNMENT_ORDERING = "clone_major"


@pipeline_node(
    PipelineNode(
        id="stage2_geography_assignment_spec",
        label="Stage 2 Geography Assignment Spec",
        node_type="library",
        description="Capture deterministic inputs for Stage 2 geography assignment before clone-level sampling runs.",
        source_file="policyengine_us_data/calibration_package/geography.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=["source_imputed_stratified_extended_cps.h5", "policy_data.db"],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_geography.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class GeographyAssignmentSpec:
    """Deterministic inputs that control Stage 2 geography assignment."""

    n_records: int
    n_clones: int
    seed: int | None
    agi_threshold_pctile: float = 90.0
    household_agi_sha256: str | None = None
    household_agi_length: int | None = None
    cd_agi_targets_sha256: str | None = None
    cd_agi_target_count: int = 0
    fixed_state_fips_sha256: str | None = None
    fixed_state_fips_length: int | None = None
    fixed_state_fips_present_count: int = 0

    def __post_init__(self) -> None:
        _validate_positive_int(self.n_records, "n_records")
        _validate_positive_int(self.n_clones, "n_clones")
        if self.seed is not None:
            _validate_non_negative_int(self.seed, "seed")
        if self.agi_threshold_pctile < 0 or self.agi_threshold_pctile > 100:
            raise ValueError("agi_threshold_pctile must be between 0 and 100")
        _validate_optional_non_negative_int(
            self.household_agi_length,
            "household_agi_length",
        )
        if (
            self.household_agi_length is not None
            and self.household_agi_length != self.n_records
        ):
            raise ValueError("household_agi_length must equal n_records")
        _validate_non_negative_int(self.cd_agi_target_count, "cd_agi_target_count")
        _validate_optional_non_negative_int(
            self.fixed_state_fips_length,
            "fixed_state_fips_length",
        )
        if (
            self.fixed_state_fips_length is not None
            and self.fixed_state_fips_length != self.n_records
        ):
            raise ValueError("fixed_state_fips_length must equal n_records")
        _validate_non_negative_int(
            self.fixed_state_fips_present_count,
            "fixed_state_fips_present_count",
        )
        if self.fixed_state_fips_length is None:
            if self.fixed_state_fips_present_count != 0:
                raise ValueError(
                    "fixed_state_fips_present_count requires fixed_state_fips_length"
                )
        elif self.fixed_state_fips_present_count > self.fixed_state_fips_length:
            raise ValueError(
                "fixed_state_fips_present_count cannot exceed fixed_state_fips_length"
            )
        for key in (
            "household_agi_sha256",
            "cd_agi_targets_sha256",
            "fixed_state_fips_sha256",
        ):
            _validate_optional_sha256(getattr(self, key), key)

    @classmethod
    def from_runtime_inputs(
        cls,
        *,
        n_records: int,
        n_clones: int,
        seed: int,
        household_agi: Any | None,
        cd_agi_targets: Mapping[str, Any] | None,
        fixed_state_fips: Any | None,
        agi_threshold_pctile: float = 90.0,
    ) -> "GeographyAssignmentSpec":
        """Build a spec from the runtime inputs that affect assignment."""

        household_agi_array = _optional_float_array(household_agi)
        fixed_state_array = _optional_int_array(fixed_state_fips)
        target_payload = _normalise_cd_agi_targets(cd_agi_targets)
        return cls(
            n_records=n_records,
            n_clones=n_clones,
            seed=seed,
            agi_threshold_pctile=agi_threshold_pctile,
            household_agi_sha256=(
                _hash_float_array(household_agi_array)
                if household_agi_array is not None
                else None
            ),
            household_agi_length=(
                int(len(household_agi_array))
                if household_agi_array is not None
                else None
            ),
            cd_agi_targets_sha256=(
                _hash_json_payload(target_payload) if target_payload else None
            ),
            cd_agi_target_count=len(target_payload),
            fixed_state_fips_sha256=(
                _hash_int_array(fixed_state_array)
                if fixed_state_array is not None
                else None
            ),
            fixed_state_fips_length=(
                int(len(fixed_state_array)) if fixed_state_array is not None else None
            ),
            fixed_state_fips_present_count=(
                int(np.count_nonzero(fixed_state_array > 0))
                if fixed_state_array is not None
                else 0
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeographyAssignmentSpec":
        """Parse a JSON-compatible assignment spec."""

        if not isinstance(data, Mapping):
            raise ValueError("geography assignment spec must be a mapping")
        return cls(
            n_records=_required_int(data, "n_records"),
            n_clones=_required_int(data, "n_clones"),
            seed=_optional_int(data, "seed"),
            agi_threshold_pctile=float(data.get("agi_threshold_pctile", 90.0)),
            household_agi_sha256=_optional_string(data, "household_agi_sha256"),
            household_agi_length=_optional_int(data, "household_agi_length"),
            cd_agi_targets_sha256=_optional_string(data, "cd_agi_targets_sha256"),
            cd_agi_target_count=_required_int(data, "cd_agi_target_count"),
            fixed_state_fips_sha256=_optional_string(
                data,
                "fixed_state_fips_sha256",
            ),
            fixed_state_fips_length=_optional_int(data, "fixed_state_fips_length"),
            fixed_state_fips_present_count=_required_int(
                data,
                "fixed_state_fips_present_count",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible spec material."""

        return {
            "agi_threshold_pctile": self.agi_threshold_pctile,
            "cd_agi_target_count": self.cd_agi_target_count,
            "cd_agi_targets_sha256": self.cd_agi_targets_sha256,
            "fixed_state_fips_length": self.fixed_state_fips_length,
            "fixed_state_fips_present_count": self.fixed_state_fips_present_count,
            "fixed_state_fips_sha256": self.fixed_state_fips_sha256,
            "household_agi_length": self.household_agi_length,
            "household_agi_sha256": self.household_agi_sha256,
            "n_clones": self.n_clones,
            "n_records": self.n_records,
            "seed": self.seed,
        }

    def assign(
        self,
        *,
        household_agi: Any | None,
        cd_agi_targets: Mapping[str, Any] | None,
        fixed_state_fips: Any | None,
        assigner: Callable[..., Any] | None = None,
    ) -> "GeographyAssignmentResult":
        """Run geography assignment through the current production assigner."""

        if assigner is None:
            from policyengine_us_data.calibration.clone_and_assign import (
                assign_random_geography,
            )

            assigner = assign_random_geography
        runtime_spec = GeographyAssignmentSpec.from_runtime_inputs(
            n_records=self.n_records,
            n_clones=self.n_clones,
            seed=self.seed if self.seed is not None else 42,
            household_agi=household_agi,
            cd_agi_targets=cd_agi_targets,
            fixed_state_fips=fixed_state_fips,
            agi_threshold_pctile=self.agi_threshold_pctile,
        )
        if runtime_spec != self:
            raise ValueError("Geography assignment runtime inputs do not match spec")
        assignment = assigner(
            n_records=self.n_records,
            n_clones=self.n_clones,
            seed=self.seed if self.seed is not None else 42,
            household_agi=household_agi,
            cd_agi_targets=cd_agi_targets,
            agi_threshold_pctile=self.agi_threshold_pctile,
            fixed_state_fips=fixed_state_fips,
        )
        return GeographyAssignmentResult.from_assignment(assignment, spec=self)


@pipeline_node(
    PipelineNode(
        id="stage2_geography_assignment_result",
        label="Stage 2 Geography Assignment Result",
        node_type="library",
        description="Summarize assigned block, county, state, and congressional district arrays without requiring consumers to load the package pickle.",
        source_file="policyengine_us_data/calibration_package/geography.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_out=[GEOGRAPHY_ASSIGNMENT_SUMMARY_FILENAME],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_geography.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class GeographyAssignmentResult:
    """Clone-major Stage 2 geography arrays and compact identity summary."""

    spec: GeographyAssignmentSpec
    block_geoid: Any
    cd_geoid: Any
    county_fips: Any
    state_fips: Any
    status: str = "completed"
    validation_errors: tuple[str, ...] = ()

    @classmethod
    def from_assignment(
        cls,
        assignment: Any,
        *,
        spec: GeographyAssignmentSpec,
    ) -> "GeographyAssignmentResult":
        """Wrap an existing `GeographyAssignment` object."""

        return cls.from_arrays(
            spec=spec,
            block_geoid=getattr(assignment, "block_geoid"),
            cd_geoid=getattr(assignment, "cd_geoid"),
            county_fips=getattr(assignment, "county_fips"),
            state_fips=getattr(assignment, "state_fips"),
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        spec: GeographyAssignmentSpec,
        block_geoid: Any,
        cd_geoid: Any,
        county_fips: Any | None = None,
        state_fips: Any | None = None,
        fail_on_validation: bool = True,
    ) -> "GeographyAssignmentResult":
        """Build a result from arrays, deriving county/state when needed."""

        block_array = _string_array(block_geoid, "block_geoid")
        cd_array = _string_array(cd_geoid, "cd_geoid")
        county_array = (
            _string_array(county_fips, "county_fips")
            if county_fips is not None
            else np.fromiter(
                (str(block)[:5] for block in block_array),
                dtype="U5",
                count=len(block_array),
            )
        )
        state_array = (
            _int_array(state_fips, "state_fips")
            if state_fips is not None
            else np.fromiter(
                (int(str(block)[:2]) for block in block_array),
                dtype=np.int32,
                count=len(block_array),
            )
        )
        errors = _geography_validation_errors(
            spec=spec,
            block_geoid=block_array,
            cd_geoid=cd_array,
            county_fips=county_array,
            state_fips=state_array,
        )
        if errors and fail_on_validation:
            raise ValueError("; ".join(errors))
        return cls(
            spec=spec,
            block_geoid=block_array,
            cd_geoid=cd_array,
            county_fips=county_array,
            state_fips=state_array,
            status="failed" if errors else "completed",
            validation_errors=tuple(errors),
        )

    @classmethod
    def from_package(
        cls,
        *,
        metadata: Mapping[str, Any],
        block_geoid: Any,
        cd_geoid: Any,
    ) -> "GeographyAssignmentResult":
        """Reconstruct summary material from package geography keys."""

        spec = geography_spec_from_metadata(metadata)
        return cls.from_arrays(
            spec=spec,
            block_geoid=block_geoid,
            cd_geoid=cd_geoid,
        )

    @property
    def n_rows(self) -> int:
        """Return the clone-level row count."""

        return int(len(self.block_geoid))

    @property
    def canonical_geography_sha256(self) -> str:
        """Return the canonical cross-stage geography checksum."""

        return canonical_geography_checksum(
            block_geoid=self.block_geoid,
            cd_geoid=self.cd_geoid,
            county_fips=self.county_fips,
            state_fips=self.state_fips,
            n_records=self.spec.n_records,
            n_clones=self.spec.n_clones,
        )

    def summary(self) -> dict[str, Any]:
        """Return compact JSON-compatible geography identity material."""

        return {
            "block_geoid_length": self.n_rows,
            "block_geoid_sha256": hash_string_array(self.block_geoid),
            "block_geoid_unique_count": _unique_count(self.block_geoid),
            "canonical_geography_sha256": self.canonical_geography_sha256,
            "cd_geoid_length": int(len(self.cd_geoid)),
            "cd_geoid_sha256": hash_string_array(self.cd_geoid),
            "cd_geoid_unique_count": _unique_count(self.cd_geoid),
            "county_fips_length": int(len(self.county_fips)),
            "county_fips_sha256": hash_string_array(self.county_fips),
            "county_fips_unique_count": _unique_count(self.county_fips),
            "has_block_geoid": True,
            "has_cd_geoid": True,
            "has_county_fips": True,
            "has_state_fips": True,
            "n_clones": self.spec.n_clones,
            "n_records": self.spec.n_records,
            "n_rows": self.n_rows,
            "ordering": GEOGRAPHY_ASSIGNMENT_ORDERING,
            "schema_version": GEOGRAPHY_ASSIGNMENT_SCHEMA_VERSION,
            "source_kind": "calibration_package",
            "spec": self.spec.to_dict(),
            "state_fips_length": int(len(self.state_fips)),
            "state_fips_sha256": hash_string_array(
                np.asarray(self.state_fips, dtype=str)
            ),
            "state_fips_unique_count": _unique_count(self.state_fips),
            "status": self.status,
            "validation_errors": list(self.validation_errors),
        }

    def to_contract_summary(self) -> Any:
        """Return the typed contract geography summary."""

        from policyengine_us_data.stage_contracts.calibration_package_schema import (
            GeographyAssignmentSummary,
        )

        return GeographyAssignmentSummary.from_dict(self.summary())

    def write_summary(self, path: str | Path) -> Path:
        """Write `geography_assignment_summary.json` and return its path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path


def geography_spec_from_metadata(
    metadata: Mapping[str, Any],
) -> GeographyAssignmentSpec:
    """Return a geography spec from package metadata, using legacy fallbacks."""

    spec = metadata.get("geography_assignment_spec")
    if isinstance(spec, Mapping):
        return GeographyAssignmentSpec.from_dict(spec)
    return GeographyAssignmentSpec(
        n_records=_metadata_int(metadata, "base_n_records"),
        n_clones=_metadata_int(metadata, "n_clones"),
        seed=_metadata_optional_int(metadata, "seed"),
    )


def geography_summary_from_package(
    *,
    metadata: Mapping[str, Any],
    block_geoid: Any | None,
    cd_geoid: Any | None,
) -> Any:
    """Return the typed geography summary for package-backed arrays."""

    from policyengine_us_data.stage_contracts.calibration_package_schema import (
        GeographyAssignmentSummary,
    )

    spec = geography_spec_from_metadata(metadata)
    if block_geoid is None and cd_geoid is None:
        return GeographyAssignmentSummary.from_dict(
            unavailable_geography_summary(spec=spec)
        )
    if block_geoid is None or cd_geoid is None:
        raise ValueError(
            "Calibration package geography requires both block_geoid and cd_geoid"
        )
    return GeographyAssignmentResult.from_arrays(
        spec=spec,
        block_geoid=block_geoid,
        cd_geoid=cd_geoid,
    ).to_contract_summary()


def unavailable_geography_summary(
    *,
    spec: GeographyAssignmentSpec,
) -> dict[str, Any]:
    """Return JSON-compatible material for legacy packages without geography."""

    return {
        "block_geoid_length": None,
        "block_geoid_sha256": None,
        "block_geoid_unique_count": None,
        "canonical_geography_sha256": None,
        "cd_geoid_length": None,
        "cd_geoid_sha256": None,
        "cd_geoid_unique_count": None,
        "county_fips_length": None,
        "county_fips_sha256": None,
        "county_fips_unique_count": None,
        "has_block_geoid": False,
        "has_cd_geoid": False,
        "has_county_fips": False,
        "has_state_fips": False,
        "n_clones": spec.n_clones,
        "n_records": spec.n_records,
        "n_rows": None,
        "ordering": None,
        "schema_version": GEOGRAPHY_ASSIGNMENT_SCHEMA_VERSION,
        "source_kind": "unavailable",
        "spec": spec.to_dict(),
        "state_fips_length": None,
        "state_fips_sha256": None,
        "state_fips_unique_count": None,
        "status": "unavailable",
        "validation_errors": [],
    }


def _geography_validation_errors(
    *,
    spec: GeographyAssignmentSpec,
    block_geoid: np.ndarray,
    cd_geoid: np.ndarray,
    county_fips: np.ndarray,
    state_fips: np.ndarray,
) -> list[str]:
    errors: list[str] = []
    expected_rows = spec.n_records * spec.n_clones
    lengths = {
        "block_geoid": len(block_geoid),
        "cd_geoid": len(cd_geoid),
        "county_fips": len(county_fips),
        "state_fips": len(state_fips),
    }
    for key, length in lengths.items():
        if length != expected_rows:
            errors.append(f"{key} length {length} does not match {expected_rows}")
    if len(set(lengths.values())) != 1:
        errors.append("geography arrays have mismatched lengths")
    if len(block_geoid) == len(county_fips):
        block_counties = np.fromiter(
            (str(block)[:5] for block in block_geoid),
            dtype="U5",
            count=len(block_geoid),
        )
        if np.any(block_counties != county_fips.astype(str)):
            errors.append("county_fips must match block_geoid prefixes")
    if len(block_geoid) == len(state_fips):
        block_states = np.fromiter(
            (int(str(block)[:2]) for block in block_geoid),
            dtype=np.int32,
            count=len(block_geoid),
        )
        if np.any(block_states != state_fips.astype(np.int32)):
            errors.append("state_fips must match block_geoid prefixes")
    return errors


def _string_array(value: Any, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=str)
    if array.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional")
    if len(array) and np.any(array == ""):
        raise ValueError(f"{key} contains empty values")
    return array


def _int_array(value: Any, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int32)
    if array.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional")
    return array


def _optional_float_array(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("household_agi must be one-dimensional")
    return array


def _optional_int_array(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    return _int_array(value, "fixed_state_fips")


def _hash_float_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"policyengine-us-data:float-array:v1")
    digest.update(len(values).to_bytes(8, byteorder="big", signed=False))
    digest.update(np.ascontiguousarray(values.astype(np.float64)).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _hash_int_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"policyengine-us-data:int-array:v1")
    digest.update(len(values).to_bytes(8, byteorder="big", signed=False))
    digest.update(np.ascontiguousarray(values.astype(np.int32)).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _normalise_cd_agi_targets(
    targets: Mapping[str, Any] | None,
) -> dict[str, float]:
    if not targets:
        return {}
    return {
        str(key): float(targets[key])
        for key in sorted(targets, key=lambda item: str(item))
    }


def _unique_count(values: Any) -> int:
    return int(len(np.unique(np.asarray(values))))


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"Calibration package metadata {key!r} is required")
    return int(value)


def _metadata_optional_int(metadata: Mapping[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    return int(value)


def _required_int(data: Mapping[str, Any], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"geography assignment spec {key!r} must be an integer")
    return value


def _optional_int(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"geography assignment spec {key!r} must be an integer or None"
        )
    return value


def _optional_string(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"geography assignment spec {key!r} must be a string or None")
    return value


def _validate_positive_int(value: Any, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def _validate_non_negative_int(value: Any, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")


def _validate_optional_non_negative_int(value: Any, key: str) -> None:
    if value is not None:
        _validate_non_negative_int(value, key)


def _validate_optional_sha256(value: Any, key: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{key} must be a SHA-256 digest")
    if len(value) != len("sha256:") + 64:
        raise ValueError(f"{key} must be a SHA-256 digest")


__all__ = [
    "GEOGRAPHY_ASSIGNMENT_ORDERING",
    "GEOGRAPHY_ASSIGNMENT_SCHEMA_VERSION",
    "GeographyAssignmentResult",
    "GeographyAssignmentSpec",
    "geography_spec_from_metadata",
    "geography_summary_from_package",
    "unavailable_geography_summary",
]
