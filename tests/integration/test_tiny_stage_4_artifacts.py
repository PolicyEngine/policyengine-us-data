import h5py
import numpy as np
import pytest

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_stage_1 import create_stage_1_artifacts
from tests.integration.support.tiny_stage_2 import PERIOD_KEY, create_stage_2_artifacts
from tests.integration.support.tiny_stage_3 import create_stage_3_artifacts
from tests.integration.support.tiny_stage_4 import (
    ENHANCED_CPS_REQUIRED_VARIABLES,
    STAGE_4_PERIOD,
    STRATIFIED_CPS_REQUIRED_VARIABLES,
    create_stage_4_artifacts,
)


def _load_period_arrays(path):
    with h5py.File(path, mode="r") as h5:
        return {name: h5[name][PERIOD_KEY][:] for name in h5.keys()}


def _create_stage_4_workspace(root):
    workspace = TinyPipelineWorkspace.create(root / "tiny-pipeline")
    create_stage_1_artifacts(workspace)
    create_stage_2_artifacts(workspace)
    create_stage_3_artifacts(workspace)
    return workspace


def _assert_period_grouped_contract(path, required_variables, artifact_name):
    with h5py.File(path, mode="r") as h5:
        assert bool(h5.attrs["fixture_scale"]) is True
        assert h5.attrs["time_period"] == STAGE_4_PERIOD
        assert h5.attrs["stage_4_artifact"] == artifact_name
        assert set(required_variables).issubset(h5.keys())
        for variable in required_variables:
            assert PERIOD_KEY in h5[variable]


def test_create_stage_4_artifacts_requires_stage_3_output(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")

    with pytest.raises(FileNotFoundError, match="Missing Stage 3 artifact"):
        create_stage_4_artifacts(workspace)


def test_create_stage_4_artifacts_writes_declared_workspace_outputs(tmp_path):
    workspace = _create_stage_4_workspace(tmp_path)

    artifacts = create_stage_4_artifacts(workspace)

    assert artifacts.as_tuple() == workspace.expected_artifacts("stage_4")
    assert all(path.exists() for path in artifacts.as_tuple())


def test_tiny_enhanced_cps_has_required_schema_and_weights(tmp_path):
    workspace = _create_stage_4_workspace(tmp_path)
    artifacts = create_stage_4_artifacts(workspace)

    _assert_period_grouped_contract(
        artifacts.enhanced_cps_path,
        ENHANCED_CPS_REQUIRED_VARIABLES,
        "enhanced_cps",
    )
    enhanced = _load_period_arrays(artifacts.enhanced_cps_path)
    extended = _load_period_arrays(workspace.stage_3 / "extended_cps_2024.h5")

    assert len(enhanced["household_weight"]) == len(extended["household_weight"])
    assert (enhanced["household_weight"] > 0).all()
    assert not np.array_equal(
        enhanced["household_weight"],
        extended["household_weight"],
    )


def test_tiny_enhanced_cps_carries_identification_and_tip_contract(tmp_path):
    workspace = _create_stage_4_workspace(tmp_path)
    artifacts = create_stage_4_artifacts(workspace)

    arrays = _load_period_arrays(artifacts.enhanced_cps_path)
    ssn_card_type = arrays["ssn_card_type"].astype(str)
    taxpayer_id_type = arrays["taxpayer_id_type"].astype(str)

    assert arrays["tip_income"].shape == arrays["person_id"].shape
    assert arrays["tip_income"].sum() > 0
    np.testing.assert_array_equal(arrays["has_itin"], arrays["has_tin"])
    np.testing.assert_array_equal(
        arrays["has_valid_ssn"],
        taxpayer_id_type == "VALID_SSN",
    )
    np.testing.assert_array_equal(
        arrays["has_tin"],
        taxpayer_id_type != "NONE",
    )
    np.testing.assert_array_equal(
        arrays["has_valid_ssn"][ssn_card_type == "NONE"], False
    )


def test_tiny_stratified_cps_has_required_schema_and_representative_rows(tmp_path):
    workspace = _create_stage_4_workspace(tmp_path)
    artifacts = create_stage_4_artifacts(workspace)

    _assert_period_grouped_contract(
        artifacts.stratified_extended_cps_path,
        STRATIFIED_CPS_REQUIRED_VARIABLES,
        "stratified_extended_cps",
    )
    arrays = _load_period_arrays(artifacts.stratified_extended_cps_path)

    assert len(arrays["household_id"]) == 3
    assert len(arrays["person_id"]) == 4
    assert set(arrays["person_household_id"]).issubset(set(arrays["household_id"]))
    assert arrays["household_is_puf_clone"].any()
    assert (~arrays["household_is_puf_clone"]).any()
    assert arrays["is_puf_clone"].any()
    assert (~arrays["is_puf_clone"]).any()


def test_tiny_stratified_cps_preserves_low_middle_and_high_income_rows(tmp_path):
    workspace = _create_stage_4_workspace(tmp_path)
    artifacts = create_stage_4_artifacts(workspace)

    arrays = _load_period_arrays(artifacts.stratified_extended_cps_path)
    income = (
        arrays["employment_income"].astype(np.float32)
        + arrays["self_employment_income"].astype(np.float32)
        + arrays["social_security"].astype(np.float32)
    )

    assert income.min() == 0
    assert income.max() >= 50_000
    assert len(np.unique(income)) >= 3
