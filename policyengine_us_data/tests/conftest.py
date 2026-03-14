"""Shared fixtures for version manifest tests."""

from unittest.mock import MagicMock

import pytest

from policyengine_us_data.utils.version_manifest import (
    HFVersionInfo,
    GCSVersionInfo,
    VersionManifest,
    VersionRegistry,
)


@pytest.fixture
def sample_generations() -> dict[str, int]:
    return {
        "enhanced_cps_2024.h5": 1710203948123456,
        "cps_2024.h5": 1710203948234567,
        "states/AL.h5": 1710203948345678,
    }


@pytest.fixture
def sample_hf_info() -> HFVersionInfo:
    return HFVersionInfo(
        repo="policyengine/policyengine-us-data",
        commit="abc123def456",
    )


@pytest.fixture
def sample_manifest(
    sample_generations: dict[str, int],
    sample_hf_info: HFVersionInfo,
) -> VersionManifest:
    return VersionManifest(
        version="1.72.3",
        created_at="2026-03-10T14:30:00Z",
        hf=sample_hf_info,
        gcs=GCSVersionInfo(
            bucket="policyengine-us-data",
            generations=sample_generations,
        ),
    )


@pytest.fixture
def sample_registry(
    sample_manifest: VersionManifest,
) -> VersionRegistry:
    """A registry with one version entry."""
    return VersionRegistry(
        current="1.72.3",
        versions=[sample_manifest],
    )


@pytest.fixture
def mock_bucket() -> MagicMock:
    bucket = MagicMock()
    bucket.name = "policyengine-us-data"
    return bucket
