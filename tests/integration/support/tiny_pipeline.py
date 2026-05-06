"""Fixture-scale pipeline composition helpers for integration tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_stage_1 import (
    Stage1Artifacts,
    create_stage_1_artifacts,
)
from tests.integration.support.tiny_stage_2 import (
    Stage2Artifacts,
    create_stage_2_artifacts,
)
from tests.integration.support.tiny_stage_3 import (
    Stage3Artifacts,
    create_stage_3_artifacts,
)
from tests.integration.support.tiny_stage_4 import (
    Stage4Artifacts,
    create_stage_4_artifacts,
)
from tests.integration.support.tiny_stage_5 import (
    Stage5Artifacts,
    create_stage_5_artifacts,
)

__test__ = False


@dataclass(frozen=True)
class TinyPipelineArtifacts:
    """Artifacts emitted by one fixture-scale Stage 1-5 pipeline run."""

    stage_1: Stage1Artifacts
    stage_2: Stage2Artifacts
    stage_3: Stage3Artifacts
    stage_4: Stage4Artifacts
    stage_5: Stage5Artifacts

    def by_stage(self) -> dict[str, tuple[Path, ...]]:
        return {
            "stage_1": self.stage_1.as_tuple(),
            "stage_2": self.stage_2.as_tuple(),
            "stage_3": self.stage_3.as_tuple(),
            "stage_4": self.stage_4.as_tuple(),
            "stage_5": self.stage_5.as_tuple(),
        }


def create_tiny_pipeline_artifacts(
    workspace: TinyPipelineWorkspace,
) -> TinyPipelineArtifacts:
    """Run the fixture-backed Stage 1-5 pipeline into one workspace."""

    stage_1 = create_stage_1_artifacts(workspace)
    stage_2 = create_stage_2_artifacts(workspace)
    stage_3 = create_stage_3_artifacts(workspace)
    stage_4 = create_stage_4_artifacts(workspace)
    stage_5 = create_stage_5_artifacts(workspace)
    return TinyPipelineArtifacts(
        stage_1=stage_1,
        stage_2=stage_2,
        stage_3=stage_3,
        stage_4=stage_4,
        stage_5=stage_5,
    )


def artifact_content_digest(path: Path) -> str:
    """Return a stable digest for a tiny pipeline artifact."""

    digest = hashlib.sha256()
    if h5py.is_hdf5(path):
        with h5py.File(path, mode="r") as h5:
            _digest_h5_object(digest, "/", h5)
        return digest.hexdigest()

    digest.update(path.read_bytes())
    return digest.hexdigest()


def stage_content_digests(
    artifacts: TinyPipelineArtifacts,
    *,
    stages: tuple[str, ...] = ("stage_3", "stage_4", "stage_5"),
) -> dict[str, dict[str, str]]:
    """Return stable content digests by stage and artifact filename."""

    by_stage = artifacts.by_stage()
    return {
        stage: {path.name: artifact_content_digest(path) for path in by_stage[stage]}
        for stage in stages
    }


def _digest_h5_object(
    digest: hashlib._Hash,
    name: str,
    obj: h5py.Dataset | h5py.Group | h5py.File,
) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(type(obj).__name__.encode("utf-8"))
    for attr_name in sorted(obj.attrs):
        digest.update(attr_name.encode("utf-8"))
        digest.update(_normalise_h5_value(obj.attrs[attr_name]))

    if isinstance(obj, h5py.Dataset):
        values = obj[()]
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(values).tobytes())
        return

    for child_name in sorted(obj.keys()):
        _digest_h5_object(digest, f"{name}/{child_name}", obj[child_name])


def _normalise_h5_value(value: object) -> bytes:
    array = np.asarray(value)
    if array.shape == ():
        return repr(array.item()).encode("utf-8")
    return repr(array.tolist()).encode("utf-8")
