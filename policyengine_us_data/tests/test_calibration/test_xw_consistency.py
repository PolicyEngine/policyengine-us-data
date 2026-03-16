"""
End-to-end test: X @ w from matrix builder must equal
sim.calculate() from stacked builder.

Uses uniform weights to isolate the consistency invariant
from any optimizer behavior.

Usage:
    pytest policyengine_us_data/tests/test_calibration/test_xw_consistency.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from policyengine_us_data.storage import STORAGE_FOLDER

DATASET_PATH = str(
    STORAGE_FOLDER / "source_imputed_stratified_extended_cps_2024.h5"
)
DB_PATH = str(STORAGE_FOLDER / "calibration" / "policy_data.db")
DB_URI = f"sqlite:///{DB_PATH}"

SEED = 42
N_CLONES = 3
N_CDS_TO_CHECK = 3


def _dataset_available():
    return Path(DATASET_PATH).exists() and Path(DB_PATH).exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not _dataset_available(),
    reason="Base dataset or DB not available",
)
def test_xw_matches_stacked_sim():
    from policyengine_us import Microsimulation
    from policyengine_us_data.calibration.clone_and_assign import (
        assign_random_geography,
    )
    from policyengine_us_data.calibration.unified_matrix_builder import (
        UnifiedMatrixBuilder,
    )
    from policyengine_us_data.calibration.publish_local_area import (
        build_h5,
    )
    from policyengine_us_data.utils.takeup import (
        TAKEUP_AFFECTED_TARGETS,
    )

    sim = Microsimulation(dataset=DATASET_PATH)
    n_records = len(sim.calculate("household_id", map_to="household").values)

    geography = assign_random_geography(
        n_records=n_records, n_clones=N_CLONES, seed=SEED
    )
    n_total = n_records * N_CLONES

    builder = UnifiedMatrixBuilder(
        db_uri=DB_URI,
        time_period=2024,
        dataset_path=DATASET_PATH,
    )

    target_filter = {
        "variables": [
            "aca_ptc",
            "snap",
            "household_count",
            "tax_unit_count",
        ]
    }
    targets_df, X, target_names = builder.build_matrix(
        geography=geography,
        sim=sim,
        target_filter=target_filter,
        hierarchical_domains=["aca_ptc", "snap"],
        rerandomize_takeup=True,
        county_level=True,
        workers=2,
    )

    target_vars = set(target_filter["variables"])
    takeup_filter = [
        info["takeup_var"]
        for key, info in TAKEUP_AFFECTED_TARGETS.items()
        if key in target_vars
    ]

    w = np.ones(n_total, dtype=np.float64)
    xw = X @ w

    cds_ordered = sorted(set(geography.cd_geoid.astype(str)))

    # Per-CD weight sums to find top CDs
    cd_weights = {}
    for i, cd in enumerate(cds_ordered):
        mask = geography.cd_geoid.astype(str) == cd
        cd_weights[cd] = w[mask].sum()
    top_cds = sorted(cd_weights, key=cd_weights.get, reverse=True)[
        :N_CDS_TO_CHECK
    ]

    check_vars = ["aca_ptc", "snap"]
    tmpdir = tempfile.mkdtemp()

    for cd in top_cds:
        h5_path = f"{tmpdir}/{cd}.h5"
        build_h5(
            weights=w,
            geography=geography,
            dataset_path=Path(DATASET_PATH),
            output_path=Path(h5_path),
            cd_subset=[cd],
            takeup_filter=takeup_filter,
        )

        stacked_sim = Microsimulation(dataset=h5_path)
        hh_weight = stacked_sim.calculate(
            "household_weight", 2024, map_to="household"
        ).values

        for var in check_vars:
            vals = stacked_sim.calculate(var, 2024, map_to="household").values
            stacked_sum = (vals * hh_weight).sum()

            cd_row = targets_df[
                (targets_df["variable"] == var)
                & (targets_df["geographic_id"] == cd)
            ]
            if len(cd_row) == 0:
                continue

            row_num = targets_df.index.get_loc(cd_row.index[0])
            xw_val = float(xw[row_num])

            if stacked_sum == 0 and xw_val == 0:
                continue

            ratio = xw_val / stacked_sum if stacked_sum != 0 else 0
            assert abs(ratio - 1.0) < 0.01, (
                f"CD {cd}, {var}: X@w={xw_val:.0f} vs "
                f"stacked={stacked_sum:.0f}, ratio={ratio:.4f}"
            )
