"""Zenodo deposit client for preservation-grade mirroring of releases.

See issue #810 for background. The PolicyEngine-calibrated microdata
artifacts live on HuggingFace, which is fast and Python-friendly but
does not publish a preservation commitment. Mirroring each certified
release to Zenodo gives us a DOI-minted, CERN / OpenAIRE-operated
long-term archive — the canonical preservation target referenced in
the 2026-04-21 meeting with Lars Vilhuber.

This module wraps the Zenodo REST API (documented at
https://developers.zenodo.org/) behind a minimal typed interface
suitable for the Modal build pipeline. The actual wiring into the
Modal upload flow is a follow-up commit; this commit ships the client
and tests it with mocked HTTP so the data contract is settled before
real credentials get exercised.

Env vars consulted:
  ZENODO_ACCESS_TOKEN — API token with deposit:write scope. When
      unset, every public function raises ``ZenodoNotConfigured``
      and the caller is expected to no-op (mirror uploads are
      optional; the release still ships to HuggingFace).
  ZENODO_BASE_URL — defaults to ``https://zenodo.org/api``. Set to
      ``https://sandbox.zenodo.org/api`` for testing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests

DEFAULT_BASE_URL = "https://zenodo.org/api"
SANDBOX_BASE_URL = "https://sandbox.zenodo.org/api"
DEFAULT_TIMEOUT_SECONDS = 60


class ZenodoNotConfigured(RuntimeError):
    """Raised when the Zenodo access token is not set.

    Callers should treat this as a signal that preservation mirroring
    is disabled, not as a pipeline failure. Letting the HF-only path
    continue is the correct default when no Zenodo token is available.
    """


class ZenodoDepositError(RuntimeError):
    """Raised when the Zenodo API rejects a deposit operation."""


@dataclass(frozen=True)
class DepositedFile:
    """A single file in a published Zenodo deposit."""

    path_in_deposit: str
    download_url: str
    size_bytes: int
    checksum: str
    """Zenodo reports the MD5 checksum in ``md5:<hex>`` form."""


@dataclass(frozen=True)
class ZenodoDeposit:
    """A published Zenodo deposit."""

    deposit_id: int
    concept_doi: str
    """Version-stable DOI that always resolves to the latest version."""
    version_doi: str
    """Per-version DOI pinned to this specific release."""
    landing_page: str
    """Human-facing Zenodo landing page URL."""
    files: tuple[DepositedFile, ...]


@dataclass(frozen=True)
class ZenodoMetadata:
    """Metadata attached to a Zenodo deposit."""

    title: str
    description: str
    creators: tuple[dict, ...]
    keywords: tuple[str, ...] = ()
    version: Optional[str] = None
    upload_type: str = "dataset"
    access_right: str = "open"
    license: str = "cc-by-4.0"
    related_identifiers: tuple[dict, ...] = ()

    def as_zenodo_payload(self) -> dict:
        payload: dict = {
            "title": self.title,
            "description": self.description,
            "upload_type": self.upload_type,
            "access_right": self.access_right,
            "license": self.license,
            "creators": list(self.creators),
        }
        if self.keywords:
            payload["keywords"] = list(self.keywords)
        if self.version is not None:
            payload["version"] = self.version
        if self.related_identifiers:
            payload["related_identifiers"] = list(self.related_identifiers)
        return payload


def resolve_base_url() -> str:
    return os.environ.get("ZENODO_BASE_URL", DEFAULT_BASE_URL)


def _require_token() -> str:
    token = os.environ.get("ZENODO_ACCESS_TOKEN")
    if not token:
        raise ZenodoNotConfigured(
            "ZENODO_ACCESS_TOKEN is not set; Zenodo preservation mirroring is disabled."
        )
    return token


def _raise_for_status(response: requests.Response, context: str) -> None:
    if response.status_code >= 400:
        raise ZenodoDepositError(
            f"Zenodo {context} failed ({response.status_code}): {response.text}"
        )


def create_and_publish_deposit(
    *,
    files: Iterable[tuple[Path, str]],
    metadata: ZenodoMetadata,
    session: Optional[requests.Session] = None,
    base_url: Optional[str] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ZenodoDeposit:
    """Create a new Zenodo deposit, upload files, and publish.

    Args:
        files: Iterable of ``(local_path, filename_in_deposit)`` pairs.
            ``local_path`` must exist at call time.
        metadata: Title, creators, license, etc.
        session: Optional pre-configured ``requests.Session``. Useful
            for injecting retries or custom adapters in production.
            When ``None`` a fresh session is created.
        base_url: Override the Zenodo API base URL (defaults to the
            ``ZENODO_BASE_URL`` env var, or the production URL).
        timeout_seconds: Per-request timeout.

    Returns:
        The published deposit's identifiers and file metadata.

    Raises:
        ZenodoNotConfigured: ``ZENODO_ACCESS_TOKEN`` env var is unset.
        ZenodoDepositError: Any Zenodo API call returned >= 400.
        FileNotFoundError: One of the local paths does not exist.
    """
    token = _require_token()
    resolved_base = base_url or resolve_base_url()
    client = session or requests.Session()
    params = {"access_token": token}

    create_response = client.post(
        f"{resolved_base}/deposit/depositions",
        params=params,
        json={},
        timeout=timeout_seconds,
    )
    _raise_for_status(create_response, "create-deposit")
    created = create_response.json()
    deposit_id = int(created["id"])
    bucket_url: str = created["links"]["bucket"]

    uploaded_files: list[DepositedFile] = []
    for local_path, deposit_filename in files:
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Zenodo upload source missing: {local_path}")
        with local_path.open("rb") as payload:
            put_response = client.put(
                f"{bucket_url}/{deposit_filename}",
                data=payload,
                params=params,
                timeout=timeout_seconds,
            )
        _raise_for_status(put_response, f"upload-file ({deposit_filename})")
        put_body = put_response.json()
        uploaded_files.append(
            DepositedFile(
                path_in_deposit=deposit_filename,
                download_url=put_body["links"]["self"],
                size_bytes=int(put_body["size"]),
                checksum=str(put_body["checksum"]),
            )
        )

    metadata_response = client.put(
        f"{resolved_base}/deposit/depositions/{deposit_id}",
        params=params,
        json={"metadata": metadata.as_zenodo_payload()},
        timeout=timeout_seconds,
    )
    _raise_for_status(metadata_response, "set-metadata")

    publish_response = client.post(
        f"{resolved_base}/deposit/depositions/{deposit_id}/actions/publish",
        params=params,
        timeout=timeout_seconds,
    )
    _raise_for_status(publish_response, "publish")
    published = publish_response.json()

    return ZenodoDeposit(
        deposit_id=deposit_id,
        concept_doi=str(published.get("conceptdoi", "")),
        version_doi=str(published.get("doi", "")),
        landing_page=str(published["links"]["html"]),
        files=tuple(uploaded_files),
    )


def zenodo_md5_to_hex(checksum: str) -> str:
    """Normalize Zenodo's ``md5:<hex>`` checksum representation to bare hex.

    Zenodo reports file checksums as ``md5:<32-hex>``; we want the
    hex for comparison against content hashes recorded elsewhere.
    Rejects any non-md5 prefix explicitly rather than silently
    accepting sha256-prefixed strings in the future.
    """
    if checksum.startswith("md5:"):
        return checksum[4:]
    if ":" in checksum:
        raise ValueError(f"Unsupported Zenodo checksum algorithm: {checksum}")
    return checksum
