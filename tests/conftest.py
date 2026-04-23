"""Shared fixtures and helpers for version manifest tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

# -- Fixtures ------------------------------------------------------


@pytest.fixture
def sample_generations() -> dict[str, int]:
    return {
        "enhanced_cps_2024.h5": 1710203948123456,
        "cps_2024.h5": 1710203948234567,
        "states/AL.h5": 1710203948345678,
    }


@pytest.fixture
def sample_hf_info() -> HFVersionInfo:
    from policyengine_us_data.utils.version_manifest import HFVersionInfo

    return HFVersionInfo(
        repo="policyengine/policyengine-us-data",
        commit="abc123def456",
    )


@pytest.fixture
def sample_policyengine_us_info() -> PolicyEngineUSBuildInfo:
    from policyengine_us_data.utils.policyengine import PolicyEngineUSBuildInfo

    return PolicyEngineUSBuildInfo(
        version="1.587.0",
        locked_version="1.587.0",
        git_commit="deadbeef1234",
        source_path="/tmp/policyengine-us",
    )


@pytest.fixture
def sample_manifest(
    sample_generations: dict[str, int],
    sample_hf_info: HFVersionInfo,
    sample_policyengine_us_info: PolicyEngineUSBuildInfo,
) -> VersionManifest:
    from policyengine_us_data.utils.version_manifest import (
        GCSVersionInfo,
        VersionManifest,
    )

    return VersionManifest(
        version="1.72.3",
        created_at="2026-03-10T14:30:00Z",
        hf=sample_hf_info,
        gcs=GCSVersionInfo(
            bucket="policyengine-us-data",
            generations=sample_generations,
        ),
        policyengine_us=sample_policyengine_us_info,
    )


@pytest.fixture
def sample_registry(
    sample_manifest: VersionManifest,
) -> VersionRegistry:
    """A registry with one version entry."""
    from policyengine_us_data.utils.version_manifest import VersionRegistry

    return VersionRegistry(
        current="1.72.3",
        versions=[sample_manifest],
    )


@pytest.fixture
def mock_bucket() -> MagicMock:
    bucket = MagicMock()
    bucket.name = "policyengine-us-data"
    return bucket


# -- Helpers -------------------------------------------------------


def make_mock_blob(generation: int) -> MagicMock:
    blob = MagicMock()
    blob.generation = generation
    return blob


def setup_bucket_with_registry(
    bucket: MagicMock,
    registry: VersionRegistry,
) -> None:
    """Configure a mock bucket to serve a registry."""
    registry_json = json.dumps(registry.to_dict())
    blob = MagicMock()
    blob.download_as_text.return_value = registry_json
    bucket.blob.return_value = blob
