import h5py
import pandas as pd

from tests.support.pipeline_workspace import TinyPipelineWorkspace
from tests.support.tiny_stage_1 import (
    ACS_HOUSEHOLD_ARRAYS,
    ACS_PERSON_ARRAYS,
    PUF_CORE_COLUMNS,
    PUF_DEMOGRAPHIC_COLUMNS,
    UPRATING_VARIABLES,
    UPRATING_YEARS,
    create_stage_1_artifacts,
)


def test_create_stage_1_artifacts_writes_declared_workspace_outputs(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")

    artifacts = create_stage_1_artifacts(workspace)

    assert artifacts.as_tuple() == workspace.expected_artifacts("stage_1")
    assert all(path.exists() for path in artifacts.as_tuple())


def test_tiny_uprating_factors_have_production_table_shape(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")
    artifacts = create_stage_1_artifacts(workspace)

    factors = pd.read_csv(artifacts.uprating_factors_path, index_col="Variable")

    assert tuple(factors.index) == UPRATING_VARIABLES
    assert tuple(map(int, factors.columns)) == UPRATING_YEARS
    assert factors[["2020", "2024", "2034"]].notna().all().all()
    assert (factors["2020"] == 1.0).all()
    assert (
        factors.loc["employment_income", "2034"]
        > factors.loc["employment_income", "2020"]
    )


def test_tiny_acs_artifact_has_minimal_array_contract(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")
    artifacts = create_stage_1_artifacts(workspace)

    with h5py.File(artifacts.acs_path, mode="r") as acs:
        assert set(ACS_PERSON_ARRAYS).issubset(acs.keys())
        assert set(ACS_HOUSEHOLD_ARRAYS).issubset(acs.keys())
        assert bool(acs.attrs["fixture_scale"]) is True
        assert len(acs["person_id"]) == 3
        assert len(acs["person_household_id"]) == 3
        assert len(acs["household_id"]) == 2
        assert len(acs["household_weight"]) == 2
        assert acs["tenure_type"].dtype.kind == "S"


def test_tiny_irs_puf_artifact_has_raw_table_contract(tmp_path):
    workspace = TinyPipelineWorkspace.create(tmp_path / "tiny-pipeline")
    artifacts = create_stage_1_artifacts(workspace)

    with pd.HDFStore(artifacts.irs_puf_path, mode="r") as store:
        assert set(store.keys()) == {"/puf", "/puf_demographics"}
        puf = store["puf"]
        demographics = store["puf_demographics"]

    assert set(PUF_CORE_COLUMNS).issubset(puf.columns)
    assert set(PUF_DEMOGRAPHIC_COLUMNS).issubset(demographics.columns)
    assert len(puf) == 3
    assert len(demographics) == 3
    assert set(puf["RECID"]) == set(demographics["RECID"])
