"""Publication identity helpers for US data runs.

The publication ID is the cross-system correlation key for one candidate
publication attempt. GitHub creates it first, Modal records it while running,
and Hugging Face staging uses it as the staging namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Mapping


PUBLICATION_ID_ENV = "US_DATA_PUBLICATION_ID"
LEGACY_PUBLICATION_ID_ENV = "PUBLICATION_ID"
MODAL_APP_NAME_ENV = "US_DATA_MODAL_APP_NAME"
MODAL_ENVIRONMENT_ENV = "US_DATA_MODAL_ENVIRONMENT"
DEFAULT_MODAL_APP_PREFIX = "policyengine-us-data-pub"
DEFAULT_MODAL_ENVIRONMENT = "main"
DEFAULT_MAX_RESOURCE_NAME_LENGTH = 64


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _truncate_with_digest(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    head_length = max_length - len(digest) - 1
    return f"{value[:head_length].rstrip('-')}-{digest}"


def sanitize_publication_id(value: str) -> str:
    """Return a Modal/HF-path-safe publication ID."""
    slug = _slugify(value)
    if not slug:
        raise ValueError("Publication ID cannot be empty")
    return _truncate_with_digest(slug, DEFAULT_MAX_RESOURCE_NAME_LENGTH)


def build_publication_id(
    *,
    github_run_id: str,
    github_run_attempt: str,
    github_sha: str,
) -> str:
    """Build a deterministic publication ID from GitHub Actions identity."""
    if not github_run_id:
        raise ValueError("github_run_id is required")
    attempt = github_run_attempt or "1"
    sha = (github_sha or "unknown")[:8]
    return sanitize_publication_id(f"usdata-gha{github_run_id}-a{attempt}-{sha}")


def build_modal_resource_name(
    publication_id: str,
    *,
    prefix: str = DEFAULT_MODAL_APP_PREFIX,
    max_length: int = DEFAULT_MAX_RESOURCE_NAME_LENGTH,
) -> str:
    """Build a safe Modal app or volume name from a publication ID."""
    return _truncate_with_digest(
        _slugify(f"{prefix}-{sanitize_publication_id(publication_id)}"),
        max_length,
    )


def staging_prefix(publication_id: str = "") -> str:
    return f"staging/{publication_id}" if publication_id else "staging"


def github_run_url(env: Mapping[str, str]) -> str:
    repository = env.get("GITHUB_REPOSITORY", "")
    run_id = env.get("GITHUB_RUN_ID", "")
    if not repository or not run_id:
        return ""
    server_url = env.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server_url}/{repository}/actions/runs/{run_id}"


def resolve_publication_id(
    explicit: str = "",
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve a publication ID from an explicit value or environment."""
    env = env or os.environ
    candidate = (
        explicit
        or env.get(PUBLICATION_ID_ENV, "")
        or env.get(LEGACY_PUBLICATION_ID_ENV, "")
    )
    if candidate:
        return sanitize_publication_id(candidate)
    if env.get("GITHUB_RUN_ID"):
        return build_publication_id(
            github_run_id=env.get("GITHUB_RUN_ID", ""),
            github_run_attempt=env.get("GITHUB_RUN_ATTEMPT", "1"),
            github_sha=env.get("GITHUB_SHA", ""),
        )
    return ""


@dataclass(frozen=True)
class PublicationContext:
    """Cross-system context for one publication attempt."""

    publication_id: str
    modal_app_name: str
    modal_environment: str
    hf_staging_prefix: str
    github_run_url: str = ""
    github_repository: str = ""
    github_workflow: str = ""
    github_ref: str = ""
    github_ref_name: str = ""
    github_sha: str = ""
    github_run_id: str = ""
    github_run_attempt: str = ""
    pipeline_volume_name: str = ""
    staging_volume_name: str = ""
    checkpoint_volume_name: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        publication_id: str = "",
        modal_app_name: str = "",
        modal_environment: str = "",
        env: Mapping[str, str] | None = None,
        modal_app_prefix: str = DEFAULT_MODAL_APP_PREFIX,
    ) -> "PublicationContext":
        env = env or os.environ
        resolved_publication_id = resolve_publication_id(publication_id, env=env)
        resolved_modal_environment = (
            modal_environment
            or env.get(MODAL_ENVIRONMENT_ENV, "")
            or env.get("MODAL_ENVIRONMENT", "")
            or DEFAULT_MODAL_ENVIRONMENT
        )
        resolved_modal_app_name = (
            modal_app_name
            or env.get(MODAL_APP_NAME_ENV, "")
            or env.get("MODAL_APP_NAME", "")
            or (
                build_modal_resource_name(
                    resolved_publication_id,
                    prefix=modal_app_prefix,
                )
                if resolved_publication_id
                else ""
            )
        )
        return cls(
            publication_id=resolved_publication_id,
            modal_app_name=resolved_modal_app_name,
            modal_environment=resolved_modal_environment,
            hf_staging_prefix=staging_prefix(resolved_publication_id),
            github_run_url=env.get("US_DATA_GITHUB_RUN_URL", "")
            or github_run_url(env),
            github_repository=env.get("GITHUB_REPOSITORY", ""),
            github_workflow=env.get("GITHUB_WORKFLOW", ""),
            github_ref=env.get("GITHUB_REF", ""),
            github_ref_name=env.get("GITHUB_REF_NAME", ""),
            github_sha=env.get("GITHUB_SHA", ""),
            github_run_id=env.get("GITHUB_RUN_ID", ""),
            github_run_attempt=env.get("GITHUB_RUN_ATTEMPT", ""),
            pipeline_volume_name=env.get("US_DATA_PIPELINE_VOLUME_NAME", ""),
            staging_volume_name=env.get("US_DATA_STAGING_VOLUME_NAME", ""),
            checkpoint_volume_name=env.get("US_DATA_CHECKPOINT_VOLUME_NAME", ""),
        )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        env: Mapping[str, str] | None = None,
        publication_id: str = "",
        modal_app_name: str = "",
        modal_environment: str = "",
    ) -> "PublicationContext":
        base = cls.from_env(
            publication_id=publication_id,
            modal_app_name=modal_app_name,
            modal_environment=modal_environment,
            env=env,
        )
        if not data:
            return base
        merged = asdict(base)
        for key, value in data.items():
            if key in merged and value:
                merged[key] = str(value)
        if merged.get("publication_id"):
            merged["publication_id"] = sanitize_publication_id(
                str(merged["publication_id"])
            )
            merged["hf_staging_prefix"] = staging_prefix(merged["publication_id"])
        return cls(**merged)

    def to_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value not in ("", None)
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def export_env(self) -> dict[str, str]:
        """Return environment variables representing this context."""
        values = {
            PUBLICATION_ID_ENV: self.publication_id,
            LEGACY_PUBLICATION_ID_ENV: self.publication_id,
            MODAL_APP_NAME_ENV: self.modal_app_name,
            "MODAL_APP_NAME": self.modal_app_name,
            MODAL_ENVIRONMENT_ENV: self.modal_environment,
            "MODAL_ENVIRONMENT": self.modal_environment,
            "US_DATA_HF_STAGING_PREFIX": self.hf_staging_prefix,
            "US_DATA_GITHUB_RUN_URL": self.github_run_url,
        }
        if self.pipeline_volume_name:
            values["US_DATA_PIPELINE_VOLUME_NAME"] = self.pipeline_volume_name
        if self.staging_volume_name:
            values["US_DATA_STAGING_VOLUME_NAME"] = self.staging_volume_name
        if self.checkpoint_volume_name:
            values["US_DATA_CHECKPOINT_VOLUME_NAME"] = self.checkpoint_volume_name
        return {key: value for key, value in values.items() if value}
