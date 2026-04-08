"""Pure helpers for H5 validation semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def validation_geo_level_for_area_type(area_type: str) -> str:
    """Map current worker/validator area types onto sanity-check geo levels."""

    if area_type == "states":
        return "state"
    if area_type == "national":
        return "national"
    return "district"


def summarize_validation_rows(
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    """Summarize per-target validation rows for worker reporting."""

    n_fail = sum(1 for row in validation_rows if row.get("sanity_check") == "FAIL")
    rel_abs_errors = [
        row["rel_abs_error"]
        for row in validation_rows
        if isinstance(row.get("rel_abs_error"), (int, float))
        and row["rel_abs_error"] != float("inf")
    ]
    mean_rae = sum(rel_abs_errors) / len(rel_abs_errors) if rel_abs_errors else 0.0
    return {
        "n_targets": len(validation_rows),
        "n_sanity_fail": n_fail,
        "mean_rel_abs_error": round(mean_rae, 4),
    }


def make_validation_error(
    item_key: str,
    error: Exception | str,
    traceback_text: str | None = None,
) -> dict[str, str]:
    """Build a structured validation error record for worker JSON output."""

    return {
        "item": item_key,
        "error": str(error),
        "traceback": traceback_text or "",
    }


def tag_validation_errors(
    validation_errors: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    """Attach a diagnostics source label to structured validation errors."""

    tagged = []
    for error in validation_errors:
        item = dict(error)
        item["source"] = source
        tagged.append(item)
    return tagged
