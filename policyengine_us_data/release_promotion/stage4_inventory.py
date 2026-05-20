"""Stage 4 inventory record helpers for release candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

from policyengine_us_data.stage_contracts.stages import STAGE_4_BUILD_OUTPUTS

from .artifacts import strip_staging_prefix
from .context import ReleasePromotionContext

_INVENTORY_PATH_KEYS = (
    "expected_release_path",
    "relative_path",
    "output_relative_path",
    "repo_path",
    "path",
    "destination_path",
    "staging_path",
)


def validate_inventory_record_context(
    record: Mapping[str, Any],
    context: ReleasePromotionContext,
) -> None:
    """Validate that an inventory record belongs to the release context."""

    run_id = optional_nested_record_string(record, "run_id")
    if run_id is not None and run_id != context.run_id:
        raise ValueError("inventory record run_id must match context.run_id")
    stage_id = optional_nested_record_string(record, "stage_id")
    if stage_id is not None and stage_id != STAGE_4_BUILD_OUTPUTS:
        raise ValueError("inventory record stage_id must be 4_build_outputs")


def inventory_record_path(
    record: Mapping[str, Any],
    *,
    context: ReleasePromotionContext,
) -> str:
    """Return the single agreed release path from an inventory record."""

    paths = inventory_record_paths(record)
    if not paths:
        raise ValueError("inventory record must include a release path")
    normalized_paths = tuple(
        strip_staging_prefix(path, context.hf_staging_prefix) for path in paths
    )
    if len(set(normalized_paths)) != 1:
        raise ValueError("inventory record path fields must agree")
    return normalized_paths[0]


def inventory_record_paths(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect supported path fields from an inventory record."""

    paths: list[str] = []
    for key in _INVENTORY_PATH_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    artifact = record.get("artifact")
    if isinstance(artifact, Mapping):
        for key in _INVENTORY_PATH_KEYS:
            value = artifact.get(key)
            if isinstance(value, str) and value:
                paths.append(value)
    return tuple(paths)


def optional_record_string(record: Mapping[str, Any], key: str) -> str | None:
    """Return an optional top-level string from an inventory record."""

    value = record.get(key)
    return value if isinstance(value, str) and value else None


def optional_nested_record_string(
    record: Mapping[str, Any],
    key: str,
) -> str | None:
    """Return an optional string from a record or nested artifact mapping."""

    value = record_value(record, key)
    return value if isinstance(value, str) and value else None


def optional_record_int(record: Mapping[str, Any], key: str) -> int | None:
    """Return an optional integer from a record or nested artifact mapping."""

    value = record_value(record, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"inventory record {key} must be an integer")
    return value


def record_value(
    record: Mapping[str, Any],
    key: str,
    *,
    default: Any = None,
) -> Any:
    """Return a value from a record or nested artifact mapping."""

    if key in record:
        return record[key]
    artifact = record.get("artifact")
    if isinstance(artifact, Mapping) and key in artifact:
        return artifact[key]
    return default


def read_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    """Read mapping-shaped records from JSONL."""

    with Path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError("output inventory JSONL rows must be mappings")
            yield payload
