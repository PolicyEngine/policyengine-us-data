import h5py
import numpy as np
import pytest

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_stage_1 import create_stage_1_artifacts
from tests.integration.support.tiny_stage_2 import PERIOD_KEY, create_stage_2_artifacts
from tests.integration.support.tiny_stage_3 import create_stage_3_artifacts
from tests.integration.support.tiny_stage_4 import create_stage_4_artifacts
from tests.integration.support.tiny_stage_5 import (
    SMALL_ENHANCED_REQUIRED_VARIABLES,
    SOURCE_IMPUTED_REQUIRED_VARIABLES,
    SPARSE_ENHANCED_REQUIRED_VARIABLES,
    STAGE_5_PERIOD,
    create_stage_5_artifacts,
)


def _load_period_arrays(path):
    with h5py.File(path, mode="r") as h5:
        return {name: h5[name][PERIOD_KEY][:] for name in h5.keys()}


def _create_stage_5_workspace(root):
    workspace = TinyPipelineWorkspace.create(root / "tiny-pipeline")
    create_stage_1_artifacts(workspace)
    create_stage_2_artifacts(workspace)
    create_stage_3_artifacts(workspace)
    create_stage_4_artifacts(workspace)
    return workspace


def _assert_period_grouped_contract(path, required_variables, artifact_name):
    with h5py.File(path, mode="r") as h5:
        assert bool(h5.attrs["fixture_scale"]) is True
        assert h5.attrs["time_period"] == STAGE_5_PERIOD
        assert h5.attrs["stage_5_artifact"] == artifact_name
        assert set(required_variables).issubset(h5.keys())
        for variable in required_variables:
            assert PERIOD_KEY in h5[variable]


def test_create_stage_5_artifacts_requires_stage_4_outputs(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")

    with pytest.raises(FileNotFoundError, match="Missing Stage 4 artifact"):
        create_stage_5_artifacts(workspace)


def test_create_stage_5_artifacts_writes_declared_workspace_outputs(tmp_path):
    workspace = _create_stage_5_workspace(tmp_path)

    artifacts = create_stage_5_artifacts(workspace)

    assert artifacts.as_tuple() == workspace.expected_artifacts("stage_5")
    assert all(path.exists() for path in artifacts.as_tuple())


def test_source_imputed_stratified_cps_has_alias_and_source_contract(tmp_path):
    workspace = _create_stage_5_workspace(tmp_path)
    artifacts = create_stage_5_artifacts(workspace)

    _assert_period_grouped_contract(
        artifacts.source_imputed_path,
        SOURCE_IMPUTED_REQUIRED_VARIABLES,
        "source_imputed_stratified_extended_cps",
    )
    assert (
        artifacts.source_imputed_alias_path.read_bytes()
        == artifacts.source_imputed_path.read_bytes()
    )


def test_source_imputed_stratified_cps_adds_expected_imputations(tmp_path):
    workspace = _create_stage_5_workspace(tmp_path)
    artifacts = create_stage_5_artifacts(workspace)

    arrays = _load_period_arrays(artifacts.source_imputed_path)

    assert arrays["tip_income"].shape == arrays["person_id"].shape
    assert arrays["hourly_wage"].shape == arrays["person_id"].shape
    assert arrays["is_paid_hourly"].dtype == np.bool_
    assert arrays["is_union_member_or_covered"].dtype == np.bool_
    assert arrays["tip_income"].sum() > 0
    assert arrays["bank_account_assets"].shape == arrays["household_id"].shape
    assert arrays["net_worth"].shape == arrays["household_id"].shape
    assert (arrays["bank_account_assets"] >= 0).all()
    assert (arrays["net_worth"] >= 0).all()
    np.testing.assert_allclose(arrays["pre_subsidy_rent"], arrays["rent"])


def test_small_enhanced_cps_is_subset_with_enhanced_contract(tmp_path):
    workspace = _create_stage_5_workspace(tmp_path)
    artifacts = create_stage_5_artifacts(workspace)

    _assert_period_grouped_contract(
        artifacts.small_enhanced_cps_path,
        SMALL_ENHANCED_REQUIRED_VARIABLES,
        "small_enhanced_cps",
    )
    small = _load_period_arrays(artifacts.small_enhanced_cps_path)
    enhanced = _load_period_arrays(workspace.stage_4 / "enhanced_cps_2024.h5")

    assert len(small["household_id"]) == 2
    assert len(small["person_id"]) < len(enhanced["person_id"])
    assert set(small["person_household_id"]).issubset(set(small["household_id"]))
    assert (small["household_weight"] > 0).all()
    assert set(small["taxpayer_id_type"].astype(str)).issubset(
        {"VALID_SSN", "OTHER_TIN", "NONE"}
    )


def test_sparse_enhanced_cps_keeps_only_positive_weight_subset(tmp_path):
    workspace = _create_stage_5_workspace(tmp_path)
    artifacts = create_stage_5_artifacts(workspace)

    _assert_period_grouped_contract(
        artifacts.sparse_enhanced_cps_path,
        SPARSE_ENHANCED_REQUIRED_VARIABLES,
        "sparse_enhanced_cps",
    )
    sparse = _load_period_arrays(artifacts.sparse_enhanced_cps_path)
    enhanced = _load_period_arrays(workspace.stage_4 / "enhanced_cps_2024.h5")

    assert 0 < len(sparse["household_id"]) < len(enhanced["household_id"])
    assert len(sparse["person_id"]) < len(enhanced["person_id"])
    assert (sparse["household_weight"] > 0).all()
    assert set(sparse["person_household_id"]).issubset(set(sparse["household_id"]))
    assert set(sparse["household_id"]).issubset(set(enhanced["household_id"]))
