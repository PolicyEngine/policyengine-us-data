"""HTTP helpers for brittle public data-source requests."""

import logging
import time
from collections.abc import Callable
from typing import Any

import requests

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def get_with_exponential_backoff(
    url: str,
    *,
    timeout: int | float = 30,
    max_attempts: int = 5,
    initial_wait_seconds: int | float = 30,
    max_wait_seconds: int | float = 240,
    retry_statuses: set[int] | frozenset[int] = RETRYABLE_STATUS_CODES,
    sleep: Callable[[int | float], None] = time.sleep,
    session: Any = requests,
    **kwargs: Any,
) -> requests.Response:
    """GET a URL, retrying transient failures with capped exponential backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    wait_seconds = initial_wait_seconds
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else None
            if status_code not in retry_statuses or attempt == max_attempts:
                raise
        except requests.RequestException:
            if attempt == max_attempts:
                raise

        logger.warning(
            "Request failed for %s (attempt %s/%s); retrying in %.0f seconds",
            url,
            attempt,
            max_attempts,
            wait_seconds,
        )
        sleep(wait_seconds)
        wait_seconds = min(wait_seconds * 2, max_wait_seconds)

    raise RuntimeError(f"Request failed after {max_attempts} attempts: {url}")
