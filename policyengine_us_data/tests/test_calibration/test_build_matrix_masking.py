"""Integration test for build_matrix geographic masking.

Traces one household through the matrix with 2 clones, verifying:
- National targets: both clones can contribute (non-zero)
- State targets: only the clone assigned to that state contributes
- CD targets: only the clone assigned to that CD contributes;
  a different CD in the same state gets zero
"""

import os

import numpy as np
import pytest
from scipy import sparse

from policyengine_us_data.storage import STORAGE_FOLDER

DATASET_PATH = str(STORAGE_FOLDER / "stratified_extended_cps_2024.h5")
DB_PATH = str(STORAGE_FOLDER / "calibration" / "policy_data.db")
DB_URI = f"sqlite:///{DB_PATH}"

N_CLONES = 2
SEED = 42
RECORD_IDX = 8629  # High SNAP ($18k), lands in TX/PA with seed=42


def _data_available():
    return os.path.exists(DATASET_PATH) and os.path.exists(DB_PATH)


@pytest.fixture(scope="module")
def matrix_result():
    if not _data_available():
        pytest.skip("Calibration data not available")

    from policyengine_us import Microsimulation
    from policyengine_us_data.calibration.clone_and_assign import (
        assign_random_geography,
    )
    from policyengine_us_data.calibration.unified_matrix_builder import (
        UnifiedMatrixBuilder,
    )

    sim = Microsimulation(dataset=DATASET_PATH)
    n_records = sim.calculate("household_id").values.shape[0]
    geography = assign_random_geography(
        n_records, n_clones=N_CLONES, seed=SEED
    )
    builder = UnifiedMatrixBuilder(
        db_uri=DB_URI,
        time_period=2024,
        dataset_path=DATASET_PATH,
    )
    targets_df, X_sparse, target_names = builder.build_matrix(
        geography=geography,
        sim=sim,
        target_filter={"domain_variables": ["snap", "medicaid"]},
    )
    return {
        "geography": geography,
        "targets_df": targets_df,
        "X": X_sparse,
        "target_names": target_names,
        "n_records": n_records,
    }


def _clone_col(n_records, clone_idx, record_idx):
    return clone_idx * n_records + record_idx


class TestMatrixShape:
    def test_columns_equal_clones_times_records(self, matrix_result):
        X = matrix_result["X"]
        n_records = matrix_result["n_records"]
        assert X.shape[1] == N_CLONES * n_records

    def test_rows_equal_targets(self, matrix_result):
        X = matrix_result["X"]
        assert X.shape[0] == len(matrix_result["targets_df"])

    def test_matrix_is_sparse(self, matrix_result):
        X = matrix_result["X"]
        density = X.nnz / (X.shape[0] * X.shape[1])
        assert density < 0.1


class TestNationalMasking:
    def test_both_clones_visible_to_national_target(self, matrix_result):
        X = matrix_result["X"]
        targets_df = matrix_result["targets_df"]
        n_records = matrix_result["n_records"]

        national_rows = targets_df[targets_df["geo_level"] == "national"].index
        assert len(national_rows) > 0

        col_0 = _clone_col(n_records, 0, RECORD_IDX)
        col_1 = _clone_col(n_records, 1, RECORD_IDX)
        X_csc = X.tocsc()

        visible_0 = X_csc[:, col_0].toarray().ravel()
        visible_1 = X_csc[:, col_1].toarray().ravel()

        for row_idx in national_rows:
            if visible_0[row_idx] != 0 or visible_1[row_idx] != 0:
                return
        pytest.fail(
            "Household has zero value for all national targets "
            "in both clones — cannot verify masking"
        )


class TestStateMasking:
    def test_clone_visible_only_to_own_state(self, matrix_result):
        X = matrix_result["X"]
        targets_df = matrix_result["targets_df"]
        geography = matrix_result["geography"]
        n_records = matrix_result["n_records"]

        col_0 = _clone_col(n_records, 0, RECORD_IDX)
        col_1 = _clone_col(n_records, 1, RECORD_IDX)
        state_0 = str(int(geography.state_fips[col_0]))
        state_1 = str(int(geography.state_fips[col_1]))

        if state_0 == state_1:
            pytest.skip(
                "Both clones landed in the same state — "
                "cannot test cross-state masking"
            )

        state_targets = targets_df[targets_df["geo_level"] == "state"]
        X_csc = X.tocsc()
        vals_0 = X_csc[:, col_0].toarray().ravel()
        vals_1 = X_csc[:, col_1].toarray().ravel()

        for _, row in state_targets.iterrows():
            row_idx = row.name
            geo_id = str(row["geographic_id"])
            if geo_id == state_0:
                assert vals_1[row_idx] == 0, (
                    f"Clone 1 (state {state_1}) should be zero "
                    f"for state {state_0} target row {row_idx}"
                )
            elif geo_id == state_1:
                assert vals_0[row_idx] == 0, (
                    f"Clone 0 (state {state_0}) should be zero "
                    f"for state {state_1} target row {row_idx}"
                )


class TestDistrictMasking:
    def test_clone_visible_only_to_own_cd(self, matrix_result):
        X = matrix_result["X"]
        targets_df = matrix_result["targets_df"]
        geography = matrix_result["geography"]
        n_records = matrix_result["n_records"]

        col_0 = _clone_col(n_records, 0, RECORD_IDX)
        cd_0 = str(geography.cd_geoid[col_0])
        state_0 = str(int(geography.state_fips[col_0]))

        district_targets = targets_df[targets_df["geo_level"] == "district"]
        X_csc = X.tocsc()
        vals_0 = X_csc[:, col_0].toarray().ravel()

        same_state_other_cd = district_targets[
            (
                district_targets["geographic_id"].apply(
                    lambda g: g.startswith(state_0)
                )
            )
            & (district_targets["geographic_id"] != cd_0)
        ]

        for _, row in same_state_other_cd.iterrows():
            row_idx = row.name
            assert vals_0[row_idx] == 0, (
                f"Clone 0 (CD {cd_0}) should be zero for "
                f"CD {row['geographic_id']} target row {row_idx}"
            )

    def test_clone_nonzero_for_own_cd(self, matrix_result):
        X = matrix_result["X"]
        targets_df = matrix_result["targets_df"]
        geography = matrix_result["geography"]
        n_records = matrix_result["n_records"]

        col_0 = _clone_col(n_records, 0, RECORD_IDX)
        cd_0 = str(geography.cd_geoid[col_0])

        own_cd_targets = targets_df[
            (targets_df["geo_level"] == "district")
            & (targets_df["geographic_id"] == cd_0)
        ]
        if len(own_cd_targets) == 0:
            pytest.skip(f"No district targets for CD {cd_0}")

        X_csc = X.tocsc()
        vals_0 = X_csc[:, col_0].toarray().ravel()

        any_nonzero = any(
            vals_0[row.name] != 0 for _, row in own_cd_targets.iterrows()
        )
        assert any_nonzero, (
            f"Clone 0 should have at least one non-zero entry "
            f"for its own CD {cd_0}"
        )
