"""Subprocess retry wrapper for network-dependent operations."""

import subprocess
import time
from typing import Optional


def run_with_retry(
    cmd: list[str],
    max_retries: int = 3,
    backoff: float = 5.0,
    env: Optional[dict] = None,
    label: str = "",
) -> subprocess.CompletedProcess:
    """Run a subprocess command with retries on failure.

    Args:
        cmd: Command and arguments.
        max_retries: Maximum number of retry attempts.
        backoff: Base delay between retries (doubled each attempt).
        env: Environment variables.
        label: Label for log messages.

    Returns:
        CompletedProcess on success.

    Raises:
        subprocess.CalledProcessError: If all retries exhausted.
    """
    tag = f"[{label}] " if label else ""
    for attempt in range(max_retries + 1):
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            return result
        if attempt < max_retries:
            delay = backoff * (2**attempt)
            print(
                f"{tag}Attempt {attempt + 1} failed "
                f"(rc={result.returncode}), "
                f"retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
        else:
            raise subprocess.CalledProcessError(result.returncode, cmd)
