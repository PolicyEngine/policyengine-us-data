"""Run identity helpers for US data publication runs.

The run ID is the cross-system correlation key for one candidate publication
attempt. GitHub creates it first, Modal records it while running, and Hugging
Face staging uses a candidate scope plus run ID as the staging namespace.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from typing import Mapping

from policyengine_us_data.utils.canonical_json import canonical_json_dumps


RUN_ID_ENV = "US_DATA_RUN_ID"
CANDIDATE_VERSION_ENV = "US_DATA_CANDIDATE_VERSION"
CANDIDATE_SCOPE_ENV = "US_DATA_CANDIDATE_SCOPE"
RELEASE_VERSION_ENV = "US_DATA_RELEASE_VERSION"
BASE_RELEASE_VERSION_ENV = "US_DATA_BASE_RELEASE_VERSION"
RELEASE_BUMP_ENV = "US_DATA_RELEASE_BUMP"
DATA_PACKAGE_VERSION_ENV = "US_DATA_PACKAGE_VERSION"
MODAL_APP_NAME_ENV = "US_DATA_MODAL_APP_NAME"
MODAL_ENVIRONMENT_ENV = "US_DATA_MODAL_ENVIRONMENT"
DEFAULT_MODAL_APP_PREFIX = "policyengine-us-data-pub"
DEFAULT_MODAL_ENVIRONMENT = "main"
DEFAULT_MAX_RESOURCE_NAME_LENGTH = 64
VALID_RELEASE_BUMPS = frozenset({"major", "minor", "patch"})
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:rc\d+)?$")


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


def sanitize_staging_version(value: str) -> str:
    """Return a Hugging Face path-safe candidate scope segment."""
    sanitized = re.sub(r"[^A-Za-z0-9._+-]+", "-", value).strip("-")
    sanitized = re.sub(r"-+", "-", sanitized)
    if not sanitized:
        raise ValueError("Staging version cannot be empty")
    return sanitized


def normalize_release_bump(value: str) -> str:
    """Return a supported SemVer bump label."""
    bump = value.strip().lower()
    if bump not in VALID_RELEASE_BUMPS:
        raise ValueError(
            f"release_bump must be one of {sorted(VALID_RELEASE_BUMPS)}; got {value!r}"
        )
    return bump


def stable_release_version(value: str) -> str:
    """Return the stable SemVer core for a final or rc package version."""
    match = SEMVER_RE.match(value)
    if not match:
        raise ValueError(f"Unsupported release version: {value}")
    major, minor, patch = match.groups()
    return f"{major}.{minor}.{patch}"


def release_version_from_bump(base_release_version: str, release_bump: str) -> str:
    """Apply a SemVer bump to a stable base release version."""
    base = stable_release_version(base_release_version)
    bump = normalize_release_bump(release_bump)
    major, minor, patch = (int(part) for part in base.split("."))
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def build_candidate_scope(base_release_version: str, release_bump: str) -> str:
    """Build the HF staging scope for a candidate release line.

    The run ID remains the candidate number in the next path segment, so the
    scope only records the deployed base release and intended SemVer bump.
    """
    base = stable_release_version(base_release_version)
    bump = normalize_release_bump(release_bump)
    return sanitize_staging_version(f"{base}-{bump}")


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


def staging_prefix(
    run_id: str = "",
    candidate_version: str = "",
    *,
    version: str = "",
) -> str:
    if not run_id:
        return "staging"
    resolved_run_id = sanitize_run_id(run_id)
    resolved_candidate_version = candidate_version or version
    if not resolved_candidate_version:
        return f"staging/{resolved_run_id}"
    staging_scope = sanitize_staging_version(
        f"{sanitize_staging_version(resolved_candidate_version)}-{resolved_run_id}"
    )
    return f"staging/{staging_scope}"


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


def resolve_candidate_version(
    explicit: str = "",
    *,
    base_release_version: str = "",
    release_bump: str = "",
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the candidate staging scope used for HF staging."""
    env = env or os.environ
    candidate = (
        explicit
        or env.get(CANDIDATE_SCOPE_ENV, "")
        or env.get(CANDIDATE_VERSION_ENV, "")
        or env.get("CANDIDATE_SCOPE", "")
        or env.get("CANDIDATE_VERSION", "")
        or env.get(DATA_PACKAGE_VERSION_ENV, "")
    )
    if candidate:
        return sanitize_staging_version(candidate)
    base = base_release_version or env.get(BASE_RELEASE_VERSION_ENV, "")
    bump = release_bump or env.get(RELEASE_BUMP_ENV, "")
    if base and bump:
        return build_candidate_scope(base, bump)
    return ""


def resolve_release_version(
    explicit: str = "",
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the final stable release version for promotion."""
    env = env or os.environ
    value = (
        explicit or env.get(RELEASE_VERSION_ENV, "") or env.get("RELEASE_VERSION", "")
    )
    return stable_release_version(value) if value else ""


def resolve_base_release_version(
    explicit: str = "",
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the deployed base release version used to label a candidate."""
    env = env or os.environ
    base = explicit or env.get(BASE_RELEASE_VERSION_ENV, "")
    return stable_release_version(base) if base else ""


def resolve_release_bump(
    explicit: str = "",
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the intended SemVer bump for a candidate run."""
    env = env or os.environ
    bump = explicit or env.get(RELEASE_BUMP_ENV, "")
    return normalize_release_bump(bump) if bump else ""


@dataclass(frozen=True)
class PublicationVersions:
    """Version identity for one candidate publication attempt."""

    candidate_version: str
    release_version: str
    run_id: str
    base_release_version: str = ""
    release_bump: str = ""
    source_sha: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        candidate_version: str = "",
        release_version: str = "",
        base_release_version: str = "",
        release_bump: str = "",
        run_id: str = "",
        source_sha: str = "",
        env: Mapping[str, str] | None = None,
    ) -> "PublicationVersions":
        env = env or os.environ
        resolved_base_release_version = resolve_base_release_version(
            base_release_version,
            env=env,
        )
        resolved_release_bump = resolve_release_bump(
            release_bump,
            env=env,
        )
        resolved_candidate_version = resolve_candidate_version(
            candidate_version,
            base_release_version=resolved_base_release_version,
            release_bump=resolved_release_bump,
            env=env,
        )
        resolved_release_version = resolve_release_version(
            release_version,
            env=env,
        )
        resolved_run_id = resolve_run_id(run_id, env=env)
        if not resolved_candidate_version:
            raise ValueError("candidate_version is required")
        if not resolved_run_id:
            raise ValueError("run_id is required")
        return cls(
            candidate_version=sanitize_staging_version(resolved_candidate_version),
            release_version=resolved_release_version,
            run_id=resolved_run_id,
            base_release_version=resolved_base_release_version,
            release_bump=resolved_release_bump,
            source_sha=source_sha
            or env.get("SOURCE_SHA", "")
            or env.get("GITHUB_SHA", ""),
        )


@dataclass(frozen=True)
class RunContext:
    """Cross-system context for one publication run."""

    run_id: str
    modal_app_name: str
    modal_environment: str
    hf_staging_prefix: str
    candidate_version: str = ""
    release_version: str = ""
    base_release_version: str = ""
    release_bump: str = ""
    data_package_version: str = ""
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
        data_package_version: str = "",
        candidate_version: str = "",
        release_version: str = "",
        base_release_version: str = "",
        release_bump: str = "",
        env: Mapping[str, str] | None = None,
        modal_app_prefix: str = DEFAULT_MODAL_APP_PREFIX,
    ) -> "RunContext":
        env = env or os.environ
        resolved_run_id = resolve_run_id(run_id, env=env)
        resolved_base_release_version = resolve_base_release_version(
            base_release_version,
            env=env,
        )
        resolved_release_bump = resolve_release_bump(
            release_bump,
            env=env,
        )
        resolved_candidate_version = resolve_candidate_version(
            candidate_version or data_package_version,
            base_release_version=resolved_base_release_version,
            release_bump=resolved_release_bump,
            env=env,
        )
        resolved_release_version = resolve_release_version(
            release_version,
            env=env,
        )
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
            hf_staging_prefix=staging_prefix(
                resolved_run_id,
                candidate_version=resolved_candidate_version,
            ),
            candidate_version=resolved_candidate_version,
            release_version=resolved_release_version,
            base_release_version=resolved_base_release_version,
            release_bump=resolved_release_bump,
            data_package_version=resolved_candidate_version,
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
        data_package_version: str = "",
        candidate_version: str = "",
        release_version: str = "",
        base_release_version: str = "",
        release_bump: str = "",
    ) -> "RunContext":
        base = cls.from_env(
            run_id=run_id,
            modal_app_name=modal_app_name,
            modal_environment=modal_environment,
            data_package_version=data_package_version,
            candidate_version=candidate_version,
            release_version=release_version,
            base_release_version=base_release_version,
            release_bump=release_bump,
            env=env,
        )
        if not data:
            return base
        merged = asdict(base)
        for key, value in data.items():
            if key == "publication_id":
                key = "run_id"
            if key == "version":
                key = "candidate_version"
            if key in merged and value:
                merged[key] = str(value)
        if merged.get("base_release_version"):
            merged["base_release_version"] = stable_release_version(
                str(merged["base_release_version"])
            )
        if merged.get("release_bump"):
            merged["release_bump"] = normalize_release_bump(str(merged["release_bump"]))
        if (
            not merged.get("candidate_version")
            and merged.get("base_release_version")
            and merged.get("release_bump")
        ):
            merged["candidate_version"] = build_candidate_scope(
                str(merged["base_release_version"]),
                str(merged["release_bump"]),
            )
        if merged.get("data_package_version") and not merged.get("candidate_version"):
            merged["candidate_version"] = str(merged["data_package_version"])
        if merged.get("candidate_version"):
            merged["candidate_version"] = sanitize_staging_version(
                str(merged["candidate_version"])
            )
            merged["data_package_version"] = str(merged["candidate_version"])
        if merged.get("release_version"):
            merged["release_version"] = stable_release_version(
                str(merged["release_version"])
            )
        if merged.get("run_id"):
            merged["run_id"] = sanitize_run_id(str(merged["run_id"]))
            merged["hf_staging_prefix"] = staging_prefix(
                merged["run_id"],
                candidate_version=str(merged.get("candidate_version") or ""),
            )
        return cls(**merged)

    def to_dict(self) -> dict[str, str]:
        return {
            key: value for key, value in asdict(self).items() if value not in ("", None)
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    def export_env(self) -> dict[str, str]:
        """Return environment variables representing this context."""
        values = {
            RUN_ID_ENV: self.run_id,
            MODAL_APP_NAME_ENV: self.modal_app_name,
            "MODAL_APP_NAME": self.modal_app_name,
            MODAL_ENVIRONMENT_ENV: self.modal_environment,
            "MODAL_ENVIRONMENT": self.modal_environment,
            CANDIDATE_VERSION_ENV: self.candidate_version,
            CANDIDATE_SCOPE_ENV: self.candidate_version,
            RELEASE_VERSION_ENV: self.release_version,
            BASE_RELEASE_VERSION_ENV: self.base_release_version,
            RELEASE_BUMP_ENV: self.release_bump,
            DATA_PACKAGE_VERSION_ENV: self.data_package_version,
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
