"""Helpers for retry and resume safety in Modal workflows."""

import json
import shutil
import subprocess
import time
from pathlib import Path
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


def ensure_resume_sha_compatible(
    branch: str,
    run_sha: str,
    current_sha: str,
    force: bool = False,
) -> bool:
    """Require resumed runs to use the same pinned commit.

    Modal workers execute the code baked into the current image,
    so resuming across branch movement would mix new code with
    artifacts and metadata from the old run.

    Args:
        branch: Branch name (for error messages).
        run_sha: SHA recorded in the run metadata.
        current_sha: Current branch tip SHA.
        force: If True, warn instead of raising on mismatch
            (enables explicit cross-SHA resume).

    Returns:
        True if SHAs match, False if mismatched but force=True.

    Raises:
        RuntimeError: If SHAs differ and force is False.
    """
    if run_sha == current_sha:
        return True
    if force:
        print(
            f"WARNING: SHA mismatch on branch {branch} "
            f"(run={run_sha[:12]}, current={current_sha[:12]}). "
            f"Proceeding with mixed provenance (force=True)."
        )
        return False
    raise RuntimeError(
        f"Branch {branch} has moved since run started.\n"
        f"  Run SHA:     {run_sha[:12]}\n"
        f"  Current SHA: {current_sha[:12]}\n"
        f"Start a fresh run instead."
    )


def reconcile_run_dir_fingerprint(
    run_dir: Path,
    fingerprint: str,
) -> str:
    """Prepare a staging run directory for a specific fingerprint.

    Safe behavior:
    - same fingerprint: resume in place
    - changed or missing fingerprint with existing H5s: stop and preserve
    - changed or missing fingerprint without H5s: clear stale directory
    """
    fingerprint_file = run_dir / "fingerprint.json"

    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        fingerprint_file.write_text(json.dumps({"fingerprint": fingerprint}))
        return "initialized"

    h5_count = len(list(run_dir.rglob("*.h5")))
    if fingerprint_file.exists():
        stored = json.loads(fingerprint_file.read_text())
        stored_fingerprint = stored.get("fingerprint")
        if stored_fingerprint == fingerprint:
            return "resume"
        if h5_count > 0:
            raise RuntimeError(
                "Fingerprint mismatch with existing staged H5 files.\n"
                f"  Stored:  {stored_fingerprint}\n"
                f"  Current: {fingerprint}\n"
                f"  H5 files preserved: {h5_count}\n"
                "Start a fresh version or clear the stale outputs explicitly."
            )
        shutil.rmtree(run_dir)
    else:
        if h5_count > 0:
            raise RuntimeError(
                "Missing fingerprint metadata with existing staged H5 files.\n"
                f"  H5 files preserved: {h5_count}\n"
                "Start a fresh version or clear the stale outputs explicitly."
            )
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_file.write_text(json.dumps({"fingerprint": fingerprint}))
    return "initialized"
