import h5py
import numpy as np
import pytest

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_stage_1 import create_stage_1_artifacts
from tests.integration.support.tiny_stage_2 import PERIOD_KEY, create_stage_2_artifacts
from tests.integration.support.tiny_stage_3 import (
    EXTENDED_CPS_REQUIRED_VARIABLES,
    STAGE_3_PERIOD,
    STAGE_4_INPUT_VARIABLES,
    create_stage_3_artifacts,
    stage_3_artifact_digest,
)


def _load_period_arrays(path):
    with h5py.File(path, mode="r") as h5:
        return {name: h5[name][PERIOD_KEY][:] for name in h5.keys()}


def _create_stage_3_workspace(root):
    workspace = TinyPipelineWorkspace.create(root / "tiny-pipeline")
    create_stage_1_artifacts(workspace)
    create_stage_2_artifacts(workspace)
    return workspace


def test_create_stage_3_artifacts_requires_stage_2_outputs(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")

    with pytest.raises(FileNotFoundError, match="Missing Stage 2 artifact"):
        create_stage_3_artifacts(workspace)


def test_create_stage_3_artifacts_writes_declared_workspace_output(tmp_path):
    workspace = _create_stage_3_workspace(tmp_path)

    artifacts = create_stage_3_artifacts(workspace)

    assert artifacts.as_tuple() == workspace.expected_artifacts("stage_3")
    assert artifacts.extended_cps_path.exists()


def test_tiny_extended_cps_has_required_period_grouped_variables(tmp_path):
    workspace = _create_stage_3_workspace(tmp_path)
    artifacts = create_stage_3_artifacts(workspace)

    with h5py.File(artifacts.extended_cps_path, mode="r") as extended:
        assert bool(extended.attrs["fixture_scale"]) is True
        assert extended.attrs["time_period"] == STAGE_3_PERIOD
        assert set(EXTENDED_CPS_REQUIRED_VARIABLES).issubset(extended.keys())
        for variable in EXTENDED_CPS_REQUIRED_VARIABLES:
            assert PERIOD_KEY in extended[variable]


def test_tiny_extended_cps_combines_cps_and_puf_rows(tmp_path):
    workspace = _create_stage_3_workspace(tmp_path)
    artifacts = create_stage_3_artifacts(workspace)

    arrays = _load_period_arrays(artifacts.extended_cps_path)

    assert len(arrays["person_id"]) == 6
    assert len(arrays["household_id"]) == 5
    assert len(np.unique(arrays["person_id"])) == 6
    assert len(np.unique(arrays["household_id"])) == 5
    assert arrays["is_puf_clone"].tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert arrays["household_is_puf_clone"].tolist() == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_tiny_extended_cps_derives_stage_4_contract_variables(tmp_path):
    workspace = _create_stage_3_workspace(tmp_path)
    artifacts = create_stage_3_artifacts(workspace)

    arrays = _load_period_arrays(artifacts.extended_cps_path)

    assert set(STAGE_4_INPUT_VARIABLES).issubset(arrays)
    np.testing.assert_allclose(
        arrays["employment_income_before_lsr"],
        arrays["employment_income"],
    )
    assert (arrays["pre_tax_contributions"] >= 0).all()
    assert (arrays["spm_unit_total_income_reported"] >= 0).all()
    assert (arrays["spm_unit_net_income_reported"] >= 0).all()


def test_tiny_extended_cps_digest_is_stable_for_same_inputs(tmp_path):
    workspace_a = _create_stage_3_workspace(tmp_path / "a")
    workspace_b = _create_stage_3_workspace(tmp_path / "b")

    artifact_a = create_stage_3_artifacts(workspace_a)
    artifact_b = create_stage_3_artifacts(workspace_b)

    assert stage_3_artifact_digest(
        artifact_a.extended_cps_path
    ) == stage_3_artifact_digest(artifact_b.extended_cps_path)
