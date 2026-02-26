"""
Verify that (X @ w)[i] matches the stacked h5 weighted sum.

Single procedural flow:
  1. Load base dataset, create geography assignment
  2. Build X with county-aware matrix builder
  3. Pick uniform weights, convert to stacked format
  4. Build stacked h5 for a few CDs
  5. Compare X @ w vs stacked sim weighted sums

Usage:
    python scripts/verify_county_fix.py
"""

import tempfile
import numpy as np

from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.calibration.clone_and_assign import (
    assign_random_geography,
)
from policyengine_us_data.calibration.unified_matrix_builder import (
    UnifiedMatrixBuilder,
)
from policyengine_us_data.calibration.unified_calibration import (
    convert_weights_to_stacked_format,
    convert_blocks_to_stacked_format,
)
from policyengine_us_data.datasets.cps.local_area_calibration.stacked_dataset_builder import (
    create_sparse_cd_stacked_dataset,
)
from policyengine_us_data.utils.takeup import TAKEUP_AFFECTED_TARGETS

DATASET_PATH = str(STORAGE_FOLDER / "stratified_extended_cps_2024.h5")
DB_PATH = str(STORAGE_FOLDER / "calibration" / "policy_data.db")
DB_URI = f"sqlite:///{DB_PATH}"

SEED = 42
N_CLONES = 3
N_CDS_TO_CHECK = 5


def main():
    # --- Step 1: Base dataset and geography ---
    print("=" * 60)
    print("Step 1: Load base dataset, create geography")
    print("=" * 60)

    sim = Microsimulation(dataset=DATASET_PATH)
    n_records = len(sim.calculate("household_id", map_to="household").values)
    print(f"  Base households: {n_records:,}")
    print(f"  Clones: {N_CLONES}")

    geography = assign_random_geography(
        n_records=n_records, n_clones=N_CLONES, seed=SEED
    )
    n_total = n_records * N_CLONES

    # --- Step 2: Build X ---
    print("\n" + "=" * 60)
    print("Step 2: Build X with county-aware matrix builder")
    print("=" * 60)

    builder = UnifiedMatrixBuilder(
        db_uri=DB_URI,
        time_period=2024,
        dataset_path=DATASET_PATH,
    )

    # tax_unit_count is not strictly necessary for this example,
    # gets crossed with every stjatum constraint in the database,
    # so you get rows like "tax_unit_count where age < 18 in
    # CD 4821", "tax_unit_count where income > 50k in state 37", etc.
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
    print(f"  Matrix shape: {X.shape}")
    print(f"  Targets: {len(targets_df)}")

    # Compute which takeup vars the matrix builder re-randomized
    target_vars = set(target_filter["variables"])
    takeup_filter = [
        info["takeup_var"]
        for key, info in TAKEUP_AFFECTED_TARGETS.items()
        if key in target_vars
    ]
    print(f"  Takeup filter: {takeup_filter}")

    # --- Step 3: Uniform weights, convert to stacked format ---
    print("\n" + "=" * 60)
    print("Step 3: Uniform weights -> stacked format")
    print("=" * 60)

    w = np.ones(n_total, dtype=np.float64)
    xw = X @ w

    geo_cd_strs = np.array([str(g) for g in geography.cd_geoid])
    cds_ordered = sorted(set(geo_cd_strs))
    w_stacked = convert_weights_to_stacked_format(
        weights=w,
        cd_geoid=geography.cd_geoid,
        base_n_records=n_records,
        cds_ordered=cds_ordered,
    )
    blocks_stacked = convert_blocks_to_stacked_format(
        block_geoid=geography.block_geoid,
        cd_geoid=geography.cd_geoid,
        base_n_records=n_records,
        cds_ordered=cds_ordered,
    )
    print(f"  CDs in geography: {len(cds_ordered)}")
    print(f"  Stacked weight vector length: {len(w_stacked):,}")

    # Pick CDs with the most weight (most clones assigned)
    cd_weights = {}
    for i, cd in enumerate(cds_ordered):
        start = i * n_records
        end = start + n_records
        cd_weights[cd] = w_stacked[start:end].sum()
    top_cds = sorted(cd_weights, key=cd_weights.get, reverse=True)[
        :N_CDS_TO_CHECK
    ]
    print(f"  Checking CDs: {top_cds}")

    # --- Step 4: Build stacked h5 and compare ---
    print("\n" + "=" * 60)
    print("Step 4: Build stacked h5, compare X @ w vs sim sums")
    print("=" * 60)

    check_vars = ["aca_ptc", "snap"]
    tmpdir = tempfile.mkdtemp()

    for cd in top_cds:
        h5_path = f"{tmpdir}/{cd}.h5"
        create_sparse_cd_stacked_dataset(
            w=w_stacked,
            cds_to_calibrate=cds_ordered,
            cd_subset=[cd],
            output_path=h5_path,
            dataset_path=DATASET_PATH,
            rerandomize_takeup=True,
            calibration_blocks=blocks_stacked,
            takeup_filter=takeup_filter,
        )

        stacked_sim = Microsimulation(dataset=h5_path)
        hh_weight = stacked_sim.calculate(
            "household_weight", 2024, map_to="household"
        ).values

        print(f"\n  CD {cd}:")
        for var in check_vars:
            vals = stacked_sim.calculate(var, 2024, map_to="household").values
            stacked_sum = (vals * hh_weight).sum()

            cd_row = targets_df[
                (targets_df["variable"] == var)
                & (targets_df["geographic_id"] == cd)
            ]
            if len(cd_row) == 0:
                print(f"    {var}: no target row — skipped")
                continue

            row_num = targets_df.index.get_loc(cd_row.index[0])
            xw_val = float(xw[row_num])

            ratio = xw_val / stacked_sum if stacked_sum != 0 else 0
            status = "PASS" if abs(ratio - 1.0) < 0.01 else "GAP"
            print(f"    {var}:")
            print(f"      X @ w:       ${xw_val:>12,.0f}")
            print(f"      Stacked sum: ${stacked_sum:>12,.0f}")
            print(f"      Ratio:       {ratio:.4f}  [{status}]")

    # --- Step 5: State-level snap for NC (FIPS 37) ---
    print("\n" + "=" * 60)
    print("Step 5: State-level snap for NC (FIPS 37)")
    print("=" * 60)

    nc_cds = [cd for cd in cds_ordered if cd.startswith("37")]
    print(f"  NC CDs: {len(nc_cds)}")

    nc_h5_path = f"{tmpdir}/nc_all.h5"
    create_sparse_cd_stacked_dataset(
        w=w_stacked,
        cds_to_calibrate=cds_ordered,
        cd_subset=nc_cds,
        output_path=nc_h5_path,
        dataset_path=DATASET_PATH,
        rerandomize_takeup=True,
        calibration_blocks=blocks_stacked,
        takeup_filter=takeup_filter,
    )

    stacked_sim = Microsimulation(dataset=nc_h5_path)
    hh_weight = stacked_sim.calculate(
        "household_weight", 2024, map_to="household"
    ).values
    snap_vals = stacked_sim.calculate("snap", 2024, map_to="household").values
    stacked_sum = (snap_vals * hh_weight).sum()

    snap_nc_row = targets_df[
        (targets_df["variable"] == "snap")
        & (targets_df["geographic_id"] == "37")
    ]
    if len(snap_nc_row) == 0:
        print("  snap NC: no target row — skipped")
    else:
        row_num = targets_df.index.get_loc(snap_nc_row.index[0])
        xw_val = float(xw[row_num])
        ratio = xw_val / stacked_sum if stacked_sum != 0 else 0
        status = "PASS" if abs(ratio - 1.0) < 0.01 else "GAP"
        print(f"  snap (NC state):")
        print(f"    X @ w:       ${xw_val:>12,.0f}")
        print(f"    Stacked sum: ${stacked_sum:>12,.0f}")
        print(f"    Ratio:       {ratio:.4f}  [{status}]")

    # --- Step 5b: Diagnose eligible amounts (no takeup re-randomization) ---
    print("\n  Diagnostic: stacked with rerandomize_takeup=False...")
    nc_norand_path = f"{tmpdir}/nc_norand.h5"
    create_sparse_cd_stacked_dataset(
        w=w_stacked,
        cds_to_calibrate=cds_ordered,
        cd_subset=nc_cds,
        output_path=nc_norand_path,
        dataset_path=DATASET_PATH,
        rerandomize_takeup=False,
        calibration_blocks=blocks_stacked,
    )
    norand_sim = Microsimulation(dataset=nc_norand_path)
    nr_weight = norand_sim.calculate(
        "household_weight", 2024, map_to="household"
    ).values
    nr_snap = norand_sim.calculate("snap", 2024, map_to="household").values
    nr_sum = (nr_snap * nr_weight).sum()
    print(f"    Stacked snap (default takeup): ${nr_sum:>12,.0f}")
    print(f"    With re-randomized takeup:     ${stacked_sum:>12,.0f}")
    print(
        f"    Ratio (default/rerand):        {nr_sum / stacked_sum:.4f}"
        if stacked_sum != 0
        else "    Ratio: N/A"
    )

    # --- Step 6: CD-level household_count for OH-02 (3902) ---
    print("\n" + "=" * 60)
    print("Step 6: CD-level household_count for OH-02 (3902)")
    print("=" * 60)

    oh02_h5_path = f"{tmpdir}/oh02.h5"
    create_sparse_cd_stacked_dataset(
        w=w_stacked,
        cds_to_calibrate=cds_ordered,
        cd_subset=["3902"],
        output_path=oh02_h5_path,
        dataset_path=DATASET_PATH,
        rerandomize_takeup=True,
        calibration_blocks=blocks_stacked,
        takeup_filter=takeup_filter,
    )

    stacked_sim = Microsimulation(dataset=oh02_h5_path)
    hh_weight = stacked_sim.calculate(
        "household_weight", 2024, map_to="household"
    ).values
    hh_snap = stacked_sim.calculate("snap", 2024, map_to="household").values
    stacked_sum = ((hh_snap > 0).astype(float) * hh_weight).sum()

    hc_row = targets_df[
        (targets_df["variable"] == "household_count")
        & (targets_df["geographic_id"] == "3902")
    ]
    if len(hc_row) == 0:
        print("  household_count OH-02: no target row — skipped")
    else:
        row_num = targets_df.index.get_loc(hc_row.index[0])
        xw_val = float(xw[row_num])
        ratio = xw_val / stacked_sum if stacked_sum != 0 else 0
        status = "PASS" if abs(ratio - 1.0) < 0.01 else "GAP"
        print(f"  household_count (OH-02, snap > 0):")
        print(f"    X @ w:       {xw_val:>12,.0f}")
        print(f"    Stacked sum: {stacked_sum:>12,.0f}")
        print(f"    Ratio:       {ratio:.4f}  [{status}]")


if __name__ == "__main__":
    main()
