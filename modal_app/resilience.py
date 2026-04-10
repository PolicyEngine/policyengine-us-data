"""Helpers for retry and resume safety in Modal workflows."""

from dataclasses import dataclass
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PublishScope:
    name: str
    owned_dirs: tuple[str, ...]


_PUBLISH_SCOPES = {
    "all": PublishScope(
        name="all",
        owned_dirs=("states", "districts", "cities", "national"),
    ),
    "regional": PublishScope(
        name="regional",
        owned_dirs=("states", "districts", "cities"),
    ),
    "national": PublishScope(
        name="national",
        owned_dirs=("national",),
    ),
}


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
    fingerprint,
    *,
    scope: str = "all",
) -> str:
    """Prepare a staging run directory for a specific fingerprint.

    Safe behavior:
    - same fingerprint: resume in place
    - changed or missing fingerprint with existing H5s: stop and preserve
    - changed or missing fingerprint without H5s: clear stale directory
    """
    from policyengine_us_data.calibration.local_h5.fingerprinting import (
        FingerprintRecord,
        FingerprintService,
    )

    service = FingerprintService()
    if isinstance(fingerprint, FingerprintRecord):
        current = fingerprint
    else:
        current = service.legacy_record(str(fingerprint))
    publish_scope = _resolve_publish_scope(scope)
    fingerprint_file = _fingerprint_file_for_scope(run_dir, publish_scope)
    legacy_fingerprint_file = run_dir / "fingerprint.json"

    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        fingerprint_file.parent.mkdir(parents=True, exist_ok=True)
        service.write_record(fingerprint_file, current)
        return "initialized"

    h5_count = _count_owned_h5_files(run_dir, publish_scope)
    stored_file = _stored_fingerprint_file(
        fingerprint_file=fingerprint_file,
        legacy_fingerprint_file=legacy_fingerprint_file,
        publish_scope=publish_scope,
    )
    if stored_file is not None:
        stored = service.read_record(stored_file)
        stored_fingerprint = stored.digest
        if service.matches(stored, current):
            if stored_file != fingerprint_file:
                fingerprint_file.parent.mkdir(parents=True, exist_ok=True)
                service.write_record(fingerprint_file, current)
            return "resume"
        if h5_count > 0:
            raise RuntimeError(
                f"Fingerprint mismatch with existing staged {publish_scope.name} H5 files.\n"
                f"  Stored:  {stored_fingerprint}\n"
                f"  Current: {current.digest}\n"
                f"  H5 files preserved: {h5_count}\n"
                "Start a fresh version or clear the stale outputs explicitly."
            )
        _clear_scope_outputs(run_dir, publish_scope)
    else:
        if h5_count > 0:
            raise RuntimeError(
                f"Missing fingerprint metadata with existing staged {publish_scope.name} H5 files.\n"
                f"  H5 files preserved: {h5_count}\n"
                "Start a fresh version or clear the stale outputs explicitly."
            )
        _clear_scope_outputs(run_dir, publish_scope)

    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_file.parent.mkdir(parents=True, exist_ok=True)
    service.write_record(fingerprint_file, current)
    return "initialized"


def _resolve_publish_scope(scope: str) -> PublishScope:
    try:
        return _PUBLISH_SCOPES[scope]
    except KeyError as error:
        raise ValueError(f"Unknown publish scope: {scope!r}") from error


def _fingerprint_file_for_scope(run_dir: Path, publish_scope: PublishScope) -> Path:
    if publish_scope.name == "all":
        return run_dir / "fingerprint.json"
    return run_dir / ".publish_scopes" / publish_scope.name / "fingerprint.json"


def _stored_fingerprint_file(
    *,
    fingerprint_file: Path,
    legacy_fingerprint_file: Path,
    publish_scope: PublishScope,
) -> Path | None:
    if fingerprint_file.exists():
        return fingerprint_file
    if publish_scope.name != "all" and legacy_fingerprint_file.exists():
        return legacy_fingerprint_file
    return None


def _count_owned_h5_files(run_dir: Path, publish_scope: PublishScope) -> int:
    return sum(
        len(list((run_dir / owned_dir).rglob("*.h5")))
        for owned_dir in publish_scope.owned_dirs
        if (run_dir / owned_dir).exists()
    )


def _clear_scope_outputs(run_dir: Path, publish_scope: PublishScope) -> None:
    if publish_scope.name == "all":
        if run_dir.exists():
            shutil.rmtree(run_dir)
        return

    for owned_dir in publish_scope.owned_dirs:
        target = run_dir / owned_dir
        if target.exists():
            shutil.rmtree(target)

    scope_meta_dir = run_dir / ".publish_scopes" / publish_scope.name
    if scope_meta_dir.exists():
        shutil.rmtree(scope_meta_dir)

    publish_scopes_root = run_dir / ".publish_scopes"
    if publish_scopes_root.exists() and not any(publish_scopes_root.iterdir()):
        publish_scopes_root.rmdir()
