"""Canonical JSON helpers shared by repo metadata formats."""

from __future__ import annotations

import json
from typing import Any


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def canonical_json_dumps(
    value: Any,
    *,
    indent: int | None = 2,
    trailing_newline: bool = True,
    compact: bool = False,
) -> str:
    """Serialize JSON deterministically and reject non-standard float values."""

    kwargs: dict[str, Any] = {
        "allow_nan": False,
        "sort_keys": True,
    }
    if compact:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = indent

    rendered = json.dumps(value, **kwargs)
    if trailing_newline:
        return f"{rendered}\n"
    return rendered


def canonical_json_bytes(
    value: Any,
    *,
    indent: int | None = 2,
    trailing_newline: bool = True,
    compact: bool = False,
) -> bytes:
    """Serialize JSON deterministically as UTF-8 bytes."""

    return canonical_json_dumps(
        value,
        indent=indent,
        trailing_newline=trailing_newline,
        compact=compact,
    ).encode("utf-8")


def canonical_json_loads(payload: str | bytes | bytearray) -> Any:
    """Parse JSON payloads serialized by `canonical_json_dumps`."""

    return json.loads(payload, parse_constant=_reject_json_constant)
