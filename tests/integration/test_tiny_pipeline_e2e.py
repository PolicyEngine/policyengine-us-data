import socket
from pathlib import Path

import h5py
import pytest

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_pipeline import (
    create_tiny_pipeline_artifacts,
    stage_content_digests,
)
from tests.integration.support.tiny_stage_2 import (
    CPS_REQUIRED_VARIABLES,
    PERIOD_KEY,
    PUF_REQUIRED_VARIABLES,
)
from tests.integration.support.tiny_stage_3 import EXTENDED_CPS_REQUIRED_VARIABLES
from tests.integration.support.tiny_stage_4 import (
    ENHANCED_CPS_REQUIRED_VARIABLES,
    STRATIFIED_CPS_REQUIRED_VARIABLES,
)
from tests.integration.support.tiny_stage_5 import (
    SMALL_ENHANCED_REQUIRED_VARIABLES,
    SOURCE_IMPUTED_REQUIRED_VARIABLES,
    SPARSE_ENHANCED_REQUIRED_VARIABLES,
)


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("fixture-scale pipeline must not open network sockets")

    monkeypatch.setattr(socket, "socket", fail_network)


def _assert_declared_artifacts_exist(
    workspace: TinyPipelineWorkspace,
    artifacts_by_stage: dict[str, tuple[Path, ...]],
) -> None:
    for stage, paths in artifacts_by_stage.items():
        assert paths == workspace.expected_artifacts(stage)
        assert all(path.exists() for path in paths)


def _assert_period_grouped_h5(path: Path, required_variables: tuple[str, ...]) -> None:
    with h5py.File(path, mode="r") as h5:
        assert bool(h5.attrs["fixture_scale"]) is True
        assert set(required_variables).issubset(h5.keys())
        for variable in required_variables:
            assert PERIOD_KEY in h5[variable]


def _assert_consumed_source(
    path: Path,
    expected_attrs: dict[str, str],
) -> None:
    with h5py.File(path, mode="r") as h5:
        for attr, value in expected_attrs.items():
            assert h5.attrs[attr] == value


def _build_pipeline(root: Path):
    workspace = TinyPipelineWorkspace.create(root / "tiny-pipeline")
    artifacts = create_tiny_pipeline_artifacts(workspace)
    return workspace, artifacts


def test_fixture_scale_pipeline_builds_stage_1_through_5_without_network(
    tmp_path,
    monkeypatch,
):
    _block_network(monkeypatch)

    workspace, artifacts = _build_pipeline(tmp_path)

    _assert_declared_artifacts_exist(workspace, artifacts.by_stage())


def test_fixture_scale_pipeline_outputs_required_handoff_schemas(tmp_path):
    _workspace, artifacts = _build_pipeline(tmp_path)

    _assert_period_grouped_h5(artifacts.stage_2.cps_path, CPS_REQUIRED_VARIABLES)
    _assert_period_grouped_h5(artifacts.stage_2.puf_path, PUF_REQUIRED_VARIABLES)
    _assert_period_grouped_h5(
        artifacts.stage_3.extended_cps_path,
        EXTENDED_CPS_REQUIRED_VARIABLES,
    )
    _assert_period_grouped_h5(
        artifacts.stage_4.enhanced_cps_path,
        ENHANCED_CPS_REQUIRED_VARIABLES,
    )
    _assert_period_grouped_h5(
        artifacts.stage_4.stratified_extended_cps_path,
        STRATIFIED_CPS_REQUIRED_VARIABLES,
    )
    _assert_period_grouped_h5(
        artifacts.stage_5.source_imputed_path,
        SOURCE_IMPUTED_REQUIRED_VARIABLES,
    )
    _assert_period_grouped_h5(
        artifacts.stage_5.small_enhanced_cps_path,
        SMALL_ENHANCED_REQUIRED_VARIABLES,
    )
    _assert_period_grouped_h5(
        artifacts.stage_5.sparse_enhanced_cps_path,
        SPARSE_ENHANCED_REQUIRED_VARIABLES,
    )


def test_fixture_scale_pipeline_records_stage_handoffs(tmp_path):
    _workspace, artifacts = _build_pipeline(tmp_path)

    _assert_consumed_source(
        artifacts.stage_2.cps_path,
        {"source_stage_1_acs": "acs_2022.h5"},
    )
    _assert_consumed_source(
        artifacts.stage_2.puf_path,
        {"source_stage_1_irs_puf": "irs_puf_2015.h5"},
    )
    _assert_consumed_source(
        artifacts.stage_3.extended_cps_path,
        {
            "source_stage_2_cps": "cps_2024.h5",
            "source_stage_2_puf": "puf_2024.h5",
        },
    )
    for path in artifacts.stage_4.as_tuple():
        _assert_consumed_source(
            path, {"source_stage_3_extended_cps": "extended_cps_2024.h5"}
        )
    _assert_consumed_source(
        artifacts.stage_5.source_imputed_path,
        {"source_stage_4_stratified": "stratified_extended_cps_2024.h5"},
    )
    _assert_consumed_source(
        artifacts.stage_5.small_enhanced_cps_path,
        {"source_stage_4_enhanced": "enhanced_cps_2024.h5"},
    )
    _assert_consumed_source(
        artifacts.stage_5.sparse_enhanced_cps_path,
        {"source_stage_4_enhanced": "enhanced_cps_2024.h5"},
    )


def test_fixture_scale_pipeline_stage_digests_are_stable(tmp_path):
    _workspace_a, artifacts_a = _build_pipeline(tmp_path / "a")
    _workspace_b, artifacts_b = _build_pipeline(tmp_path / "b")

    assert stage_content_digests(artifacts_a) == stage_content_digests(artifacts_b)


def test_fixture_scale_pipeline_source_imputed_alias_matches_versioned_output(tmp_path):
    _workspace, artifacts = _build_pipeline(tmp_path)

    assert (
        artifacts.stage_5.source_imputed_alias_path.read_bytes()
        == artifacts.stage_5.source_imputed_path.read_bytes()
    )
