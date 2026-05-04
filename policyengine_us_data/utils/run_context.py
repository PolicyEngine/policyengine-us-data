"""Run identity helpers for US data publication runs.

The run ID is the cross-system correlation key for one candidate publication
attempt. GitHub creates it first, Modal records it while running, and Hugging
Face staging uses it as the staging namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Mapping


RUN_ID_ENV = "US_DATA_RUN_ID"
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


def sanitize_run_id(value: str) -> str:
    """Return a Modal/HF-path-safe run ID."""
    slug = _slugify(value)
    if not slug:
        raise ValueError("Run ID cannot be empty")
    return _truncate_with_digest(slug, DEFAULT_MAX_RESOURCE_NAME_LENGTH)


def build_run_id(
    *,
    github_run_id: str,
    github_run_attempt: str,
    github_sha: str,
) -> str:
    """Build a deterministic run ID from GitHub Actions identity."""
    if not github_run_id:
        raise ValueError("github_run_id is required")
    attempt = github_run_attempt or "1"
    sha = (github_sha or "unknown")[:8]
    return sanitize_run_id(f"usdata-gha{github_run_id}-a{attempt}-{sha}")


def build_modal_resource_name(
    run_id: str,
    *,
    prefix: str = DEFAULT_MODAL_APP_PREFIX,
    max_length: int = DEFAULT_MAX_RESOURCE_NAME_LENGTH,
) -> str:
    """Build a safe Modal app or volume name from a run ID."""
    return _truncate_with_digest(
        _slugify(f"{prefix}-{sanitize_run_id(run_id)}"),
        max_length,
    )


def staging_prefix(run_id: str = "") -> str:
    return f"staging/{run_id}" if run_id else "staging"


def github_run_url(env: Mapping[str, str]) -> str:
    repository = env.get("GITHUB_REPOSITORY", "")
    run_id = env.get("GITHUB_RUN_ID", "")
    if not repository or not run_id:
        return ""
    server_url = env.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server_url}/{repository}/actions/runs/{run_id}"


def resolve_run_id(
    explicit: str = "",
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the canonical run ID from an explicit value or publication env.

    Raw GitHub Actions IDs are intentionally not publication run IDs. GitHub
    workflow scripts translate them once, then export US_DATA_RUN_ID for
    library code and Modal entrypoints.
    """
    env = env or os.environ
    candidate = explicit or env.get(RUN_ID_ENV, "")
    if candidate:
        return sanitize_run_id(candidate)
    return ""


@dataclass(frozen=True)
class RunContext:
    """Cross-system context for one publication run."""

    run_id: str
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
        run_id: str = "",
        modal_app_name: str = "",
        modal_environment: str = "",
        env: Mapping[str, str] | None = None,
        modal_app_prefix: str = DEFAULT_MODAL_APP_PREFIX,
    ) -> "RunContext":
        env = env or os.environ
        resolved_run_id = resolve_run_id(run_id, env=env)
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
                    resolved_run_id,
                    prefix=modal_app_prefix,
                )
                if resolved_run_id
                else ""
            )
        )
        return cls(
            run_id=resolved_run_id,
            modal_app_name=resolved_modal_app_name,
            modal_environment=resolved_modal_environment,
            hf_staging_prefix=staging_prefix(resolved_run_id),
            github_run_url=env.get("US_DATA_GITHUB_RUN_URL", "") or github_run_url(env),
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
        run_id: str = "",
        modal_app_name: str = "",
        modal_environment: str = "",
    ) -> "RunContext":
        base = cls.from_env(
            run_id=run_id,
            modal_app_name=modal_app_name,
            modal_environment=modal_environment,
            env=env,
        )
        if not data:
            return base
        merged = asdict(base)
        for key, value in data.items():
            if key == "publication_id":
                key = "run_id"
            if key in merged and value:
                merged[key] = str(value)
        if merged.get("run_id"):
            merged["run_id"] = sanitize_run_id(str(merged["run_id"]))
            merged["hf_staging_prefix"] = staging_prefix(merged["run_id"])
        return cls(**merged)

    def to_dict(self) -> dict[str, str]:
        return {
            key: value for key, value in asdict(self).items() if value not in ("", None)
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def export_env(self) -> dict[str, str]:
        """Return environment variables representing this context."""
        values = {
            RUN_ID_ENV: self.run_id,
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
