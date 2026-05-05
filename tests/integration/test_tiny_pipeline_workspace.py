from pathlib import Path

import pytest

from tests.integration.support.pipeline_workspace import (
    STAGE_ARTIFACTS,
    TinyPipelineWorkspace,
)


def test_tiny_pipeline_workspace_creates_canonical_directories(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")

    expected_dirs = [
        workspace.inputs,
        workspace.stage_1,
        workspace.stage_2,
        workspace.stage_3,
        workspace.stage_4,
        workspace.stage_5,
        workspace.calibration,
        workspace.h5_outputs,
        workspace.h5_staging,
        workspace.h5_diagnostics,
        workspace.h5_manifests,
    ]

    assert all(path.is_dir() for path in expected_dirs)


def test_tiny_pipeline_workspace_resolves_expected_artifacts(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")

    assert workspace.expected_artifacts("stage_1") == (
        workspace.stage_1 / "uprating_factors.csv",
        workspace.stage_1 / "acs_2022.h5",
        workspace.stage_1 / "irs_puf_2015.h5",
    )
    assert workspace.expected_artifacts("h5_outputs") == (
        workspace.h5_outputs / "states" / "NC.h5",
        workspace.h5_outputs / "districts" / "NC-01.h5",
        workspace.h5_outputs / "national" / "US.h5",
    )

    # Nested expected artifact paths should be immediately writable by later
    # fixture builders.
    for path in workspace.expected_artifacts("h5_outputs"):
        assert path.parent.is_dir()


def test_tiny_pipeline_workspace_rejects_unknown_stage(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")

    with pytest.raises(KeyError, match="Unknown tiny pipeline stage"):
        workspace.stage_dir("not-a-stage")


def test_tiny_pipeline_workspace_exposes_all_declared_artifacts(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")

    artifacts = workspace.all_expected_artifacts()

    assert set(artifacts) == set(STAGE_ARTIFACTS)
    assert all(isinstance(path, Path) for paths in artifacts.values() for path in paths)
