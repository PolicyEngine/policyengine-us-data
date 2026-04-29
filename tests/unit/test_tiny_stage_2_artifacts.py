import h5py
import numpy as np
import pandas as pd
import pytest

from tests.support.pipeline_workspace import TinyPipelineWorkspace
from tests.support.tiny_stage_1 import create_stage_1_artifacts
from tests.support.tiny_stage_2 import (
    CPS_REQUIRED_VARIABLES,
    PERIOD_KEY,
    PUF_REQUIRED_VARIABLES,
    STAGE_2_PERIOD,
    create_stage_2_artifacts,
)


def _load_period_arrays(path):
    with h5py.File(path, mode="r") as h5:
        return {name: h5[name][PERIOD_KEY][:] for name in h5.keys()}


def _assert_period_grouped_contract(path, required_variables):
    with h5py.File(path, mode="r") as h5:
        assert bool(h5.attrs["fixture_scale"]) is True
        assert h5.attrs["time_period"] == STAGE_2_PERIOD
        assert set(required_variables).issubset(h5.keys())
        for variable in required_variables:
            assert PERIOD_KEY in h5[variable]


def _assert_identity_contract(arrays):
    assert len(np.unique(arrays["person_id"])) == len(arrays["person_id"])
    assert set(arrays["person_household_id"]).issubset(set(arrays["household_id"]))
    assert set(arrays["person_tax_unit_id"]).issubset(set(arrays["tax_unit_id"]))
    assert set(arrays["person_spm_unit_id"]).issubset(set(arrays["spm_unit_id"]))
    assert (arrays["household_weight"] > 0).all()


def test_create_stage_2_artifacts_requires_stage_1_outputs(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")

    with pytest.raises(FileNotFoundError, match="Missing Stage 1 artifact"):
        create_stage_2_artifacts(workspace)


def test_create_stage_2_artifacts_writes_declared_workspace_outputs(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")
    create_stage_1_artifacts(workspace)

    artifacts = create_stage_2_artifacts(workspace)

    assert artifacts.as_tuple() == workspace.expected_artifacts("stage_2")
    assert all(path.exists() for path in artifacts.as_tuple())


def test_tiny_cps_artifact_has_period_grouped_array_contract(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")
    create_stage_1_artifacts(workspace)
    artifacts = create_stage_2_artifacts(workspace)

    _assert_period_grouped_contract(artifacts.cps_path, CPS_REQUIRED_VARIABLES)
    arrays = _load_period_arrays(artifacts.cps_path)

    assert len(arrays["person_id"]) == 3
    assert len(arrays["household_id"]) == 2
    assert len(arrays["household_weight"]) == 2
    assert arrays["filing_status"].dtype.kind == "S"
    _assert_identity_contract(arrays)


def test_tiny_puf_artifact_has_period_grouped_array_contract(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")
    create_stage_1_artifacts(workspace)
    artifacts = create_stage_2_artifacts(workspace)

    _assert_period_grouped_contract(artifacts.puf_path, PUF_REQUIRED_VARIABLES)
    arrays = _load_period_arrays(artifacts.puf_path)

    assert len(arrays["person_id"]) == 3
    assert len(arrays["household_id"]) == 3
    assert len(arrays["household_weight"]) == 3
    assert arrays["filing_status"].dtype.kind == "S"
    _assert_identity_contract(arrays)


def test_stage_2_cps_uses_stage_1_acs_and_uprating_inputs(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")
    stage_1 = create_stage_1_artifacts(workspace)
    artifacts = create_stage_2_artifacts(workspace)

    factors = pd.read_csv(stage_1.uprating_factors_path, index_col="Variable")
    expected_growth = factors.loc["employment_income", PERIOD_KEY]
    with h5py.File(stage_1.acs_path, mode="r") as acs:
        expected = acs["employment_income"][:] * expected_growth

    arrays = _load_period_arrays(artifacts.cps_path)
    np.testing.assert_allclose(arrays["employment_income"], expected)


def test_stage_2_puf_uses_stage_1_raw_puf_and_uprating_inputs(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")
    stage_1 = create_stage_1_artifacts(workspace)
    artifacts = create_stage_2_artifacts(workspace)

    factors = pd.read_csv(stage_1.uprating_factors_path, index_col="Variable")
    expected_growth = factors.loc["employment_income", PERIOD_KEY]
    with pd.HDFStore(stage_1.irs_puf_path, mode="r") as store:
        raw_puf = store["puf"]
    expected = raw_puf["E00200"].to_numpy(dtype=np.float32) * expected_growth

    arrays = _load_period_arrays(artifacts.puf_path)
    np.testing.assert_allclose(arrays["employment_income"], expected)
