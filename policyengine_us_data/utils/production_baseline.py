"""Resolve the canonical *production* enhanced CPS baseline.

The "production eCPS" is the ``enhanced_cps_2024.h5`` published to Hugging
Face under :data:`DEFAULT_REPO`, tagged with the released
``policyengine-us-data`` version. Uploads create that tag
(:func:`policyengine_us_data.utils.data_upload.upload_files_to_hf`), so a
package version pins a reproducible, byte-identical dataset.

This module exists so that anything comparing against "the eCPS" (microplex,
audits, replacement diagnostics) can never silently use a stale or broken
*local* rebuild. It fetches the pinned release into the local Hugging Face
cache, records provenance (repo, revision, sha256), and runs an integrity
gate that fails loudly if a required column is missing or all-zero -- the
recurring failure mode where a build step quietly drops a column (e.g. the
extended-CPS step dropping ``social_security_retirement``).

The recorded ``sha256`` is provenance only: byte-level integrity of the
download is already guaranteed by ``hf_hub_download`` against the Hub's content
hash, so the hash here is for audit/traceability, not an independent integrity
check. The gate this module adds is about *content* -- the required columns
being present and non-zero, which the Hub hash cannot tell you.

Typical use::

    from policyengine_us_data.utils.production_baseline import (
        resolve_production_ecps,
    )

    baseline = resolve_production_ecps()  # pinned, verified
    run_comparison(baseline.path)
    record_provenance(baseline.to_dict())
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_REPO = "policyengine/policyengine-us-data"
DEFAULT_FILENAME = "enhanced_cps_2024.h5"
PACKAGE_NAME = "policyengine-us-data"

#: Environment override for the pinned revision (commit, tag, or branch).
VERSION_ENV_VAR = "POLICYENGINE_US_DATA_ECPS_VERSION"

#: Columns that MUST be present and non-zero in a healthy production eCPS.
#: This is the guardrail against the recurring "a build step silently dropped
#: a column" failure -- most recently ``social_security_retirement`` vanishing
#: in the extended-CPS step, which left the published comparison baseline 64%
#: short on total Social Security. Extend as new must-have inputs are added.
#:
#: Only the *robustly-populated* Social Security components are gated.
#: ``social_security_retirement`` and ``social_security_disability`` are always
#: populated -- the imputation fallback ``_age_heuristic_ss_shares`` assigns
#: both even when the QRF share model is unavailable -- so an all-zero value
#: reliably means the column was dropped. ``social_security_survivors`` and
#: ``social_security_dependents`` are sparse and stay zero under that fallback,
#: so gating them would reject builds the pipeline intentionally allows; they
#: are deliberately excluded.
REQUIRED_NONZERO_COLUMNS: tuple[str, ...] = (
    "social_security_retirement",
    "social_security_disability",
    "employment_income_before_lsr",
)


class BaselineIntegrityError(RuntimeError):
    """Raised when a resolved baseline fails the integrity gate."""


@dataclass
class ProductionBaseline:
    """A resolved, verified production baseline and its provenance."""

    path: str
    repo: str
    revision: str
    sha256: str
    checks: dict
    fetched_at: str

    def to_dict(self) -> dict:
        """Return a JSON-serialisable provenance record."""
        return asdict(self)


def production_pin(version: Optional[str] = None) -> str:
    """Return the revision to pin the production eCPS to.

    Args:
        version: Explicit revision (commit, tag, or branch). When ``None``,
            falls back to the ``POLICYENGINE_US_DATA_ECPS_VERSION`` environment
            variable and then to the installed ``policyengine-us-data``
            package version.

    Returns:
        The revision string to request from Hugging Face.
    """
    if version:
        return version
    env_version = os.environ.get(VERSION_ENV_VAR)
    if env_version:
        return env_version
    return importlib_metadata.version(PACKAGE_NAME)


def _sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_nonzero_count(group_or_dataset) -> int:
    """Return the number of non-zero entries in an h5py node.

    Handles both a bare dataset and a group keyed by period (the latest
    period is used).
    """
    import h5py
    import numpy as np

    node = group_or_dataset
    if isinstance(node, h5py.Group):
        node = node[sorted(node.keys())[-1]]
    values = np.asarray(node[()], dtype=float)
    return int((values != 0).sum())


def assert_baseline_intact(
    path: str | Path,
    required_nonzero_columns: Sequence[str] = REQUIRED_NONZERO_COLUMNS,
) -> dict:
    """Assert required columns are present and non-zero in a dataset.

    Args:
        path: Path to an HDF5 dataset (PolicyEngine ``Dataset`` layout).
        required_nonzero_columns: Columns that must exist and have at least
            one non-zero value.

    Returns:
        A checks dict describing what was inspected.

    Raises:
        BaselineIntegrityError: If any required column is missing or all-zero.
    """
    import h5py

    path = Path(path)
    missing: list[str] = []
    all_zero: list[str] = []
    nonzero_counts: dict[str, int] = {}

    with h5py.File(path, "r") as handle:
        present_keys = set(handle.keys())
        for column in required_nonzero_columns:
            if column not in present_keys:
                missing.append(column)
                continue
            count = _column_nonzero_count(handle[column])
            nonzero_counts[column] = count
            if count == 0:
                all_zero.append(column)

    passed = not missing and not all_zero
    checks = {
        "required_nonzero_columns": list(required_nonzero_columns),
        "missing": missing,
        "all_zero": all_zero,
        "nonzero_counts": nonzero_counts,
        "passed": passed,
    }
    if not passed:
        raise BaselineIntegrityError(
            f"Production eCPS baseline at {path} failed the integrity gate: "
            f"missing={missing or 'none'}, all_zero={all_zero or 'none'}. "
            "This usually means a build step silently dropped a column (e.g. "
            "the extended-CPS step dropping social_security_retirement). "
            "Refusing to use a broken baseline."
        )
    return checks


def resolve_production_ecps(
    version: Optional[str] = None,
    *,
    repo: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILENAME,
    token: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    verify: bool = True,
    required_nonzero_columns: Sequence[str] = REQUIRED_NONZERO_COLUMNS,
) -> ProductionBaseline:
    """Download (or reuse from cache) and verify the production eCPS.

    The file is fetched from Hugging Face pinned to a specific revision (the
    installed package version by default) into the local Hugging Face cache,
    so repeat calls reuse the byte-identical file. Integrity is checked unless
    ``verify`` is ``False``.

    Args:
        version: Revision to pin to. Defaults to
            :func:`production_pin`.
        repo: Hugging Face repository id.
        filename: File to fetch from the repository.
        token: Hugging Face token. Defaults to ``HUGGING_FACE_TOKEN``.
        cache_dir: Optional override for the Hugging Face cache directory.
        verify: Whether to run the integrity gate.
        required_nonzero_columns: Columns the integrity gate requires.

    Returns:
        A :class:`ProductionBaseline` with the local path and provenance.

    Raises:
        BaselineIntegrityError: If ``verify`` is set and the gate fails.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    revision = production_pin(version)
    token = token or os.environ.get("HUGGING_FACE_TOKEN")
    try:
        local_path = hf_hub_download(
            repo_id=repo,
            repo_type="model",
            filename=filename,
            revision=revision,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except (HfHubHTTPError, FileNotFoundError) as exc:
        # RepositoryNotFoundError / RevisionNotFoundError /
        # RemoteEntryNotFoundError / GatedRepoError all subclass HfHubHTTPError;
        # LocalEntryNotFoundError (offline, uncached) is a FileNotFoundError.
        # Turn any of them into a clear, actionable failure rather than an
        # opaque Hub traceback.
        raise BaselineIntegrityError(
            f"Could not fetch the production eCPS '{filename}' at revision "
            f"'{revision}' from '{repo}': {exc}. The pin defaults to the "
            f"installed {PACKAGE_NAME} version, which may never have been "
            f"published to Hugging Face (e.g. a local/dev/editable install). "
            f"Set {VERSION_ENV_VAR} to a published revision (tag or commit), "
            f"or install a released {PACKAGE_NAME}."
        ) from exc
    checks: dict
    if verify:
        checks = assert_baseline_intact(local_path, required_nonzero_columns)
    else:
        checks = {"passed": None, "skipped": True}
    return ProductionBaseline(
        path=str(local_path),
        repo=repo,
        revision=revision,
        sha256=_sha256(local_path),
        checks=checks,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: print the resolved production eCPS path (or JSON provenance)."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Resolve and verify the production eCPS baseline "
            "(HF-published, pinned to the package version)."
        )
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Revision/pin to fetch (default: installed policyengine-us-data version)."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the integrity gate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full provenance as JSON instead of just the path.",
    )
    args = parser.parse_args(argv)
    baseline = resolve_production_ecps(version=args.version, verify=not args.no_verify)
    if args.json:
        print(json.dumps(baseline.to_dict(), indent=2))
    else:
        print(baseline.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
