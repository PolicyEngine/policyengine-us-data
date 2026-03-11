"""Helper functions for version manifest tests."""

import json
from unittest.mock import MagicMock

from policyengine_us_data.utils.version_manifest import (
    VersionRegistry,
)


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
