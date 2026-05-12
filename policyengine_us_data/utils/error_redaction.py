"""Helpers for redacting and bounding error text."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

DEFAULT_ERROR_TEXT_MAX_CHARS = 24_000

_SECRET_KEY_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "API_KEY",
    "ACCESS_KEY",
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE_KEY|"
    r"API_KEY|ACCESS_KEY)[A-Z0-9_]*)\s*=\s*([^\s,;]+)"
)


@dataclass(frozen=True)
class BoundedErrorText:
    """Error text bounded for status payloads and logs."""

    text: str
    truncated: bool
    omitted_chars: int


def _is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in _SECRET_KEY_MARKERS)


def redact_error_text(text: str | None, *, env: Mapping[str, str] | None = None) -> str:
    """Redact obvious secret values from error text."""

    redacted = text or ""
    source_env = env or os.environ
    for key, value in source_env.items():
        if not value or len(value) < 8 or not _is_secret_key(key):
            continue
        redacted = redacted.replace(value, f"<redacted:{key}>")
    return _SECRET_ASSIGNMENT_RE.sub(r"\1=<redacted>", redacted)


def bound_error_text(
    text: str | None,
    *,
    max_chars: int | None = DEFAULT_ERROR_TEXT_MAX_CHARS,
) -> BoundedErrorText:
    """Keep the newest error text when a traceback is too long."""

    value = text or ""
    if max_chars is None or len(value) <= max_chars:
        return BoundedErrorText(text=value, truncated=False, omitted_chars=0)
    if max_chars <= 0:
        return BoundedErrorText(text="", truncated=True, omitted_chars=len(value))

    marker = "\n[truncated older error text; omitted {omitted} chars]\n"
    omitted = len(value) - max_chars
    rendered_marker = marker.format(omitted=omitted)
    if len(rendered_marker) >= max_chars:
        return BoundedErrorText(
            text=value[-max_chars:],
            truncated=True,
            omitted_chars=omitted,
        )

    tail_chars = max_chars - len(rendered_marker)
    omitted = len(value) - tail_chars
    rendered_marker = marker.format(omitted=omitted)
    return BoundedErrorText(
        text=f"{rendered_marker}{value[-tail_chars:]}",
        truncated=True,
        omitted_chars=omitted,
    )


def redacted_bounded_error_text(
    text: str | None,
    *,
    env: Mapping[str, str] | None = None,
    max_chars: int | None = DEFAULT_ERROR_TEXT_MAX_CHARS,
) -> BoundedErrorText:
    """Redact error text, then bound it by keeping the newest content."""

    return bound_error_text(redact_error_text(text, env=env), max_chars=max_chars)
