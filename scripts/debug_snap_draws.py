"""
Debug SNAP ~4% gap: raw draw comparison between matrix and stacked builders.

Picks one NC CD and ~10 households with SNAP-eligible SPM units,
then prints every detail of the takeup draw from both sides.

What to look for in the output:
  - Step 2 prints the actual X matrix value X[snap_NC, col] next to
    our manually computed eligible * takeup.  If these differ for any
    household, the matrix builder's state precomputation produced
    different eligible amounts than a fresh sim.  This is the
    signature of state-loop pollution (see debug_state_precomp.py
    and docs/snap_state_loop_pollution.md).
  - Steps 1 & 3 confirm that blocks, salts, seeds, raw draws, and
    takeup booleans are byte-identical between the two builders.
    The draws themselves are NOT the problem.
  - Step 4 shows the aggregate X @ w vs stacked sim weighted sum
    at the CD and state level.

Usage:
    python scripts/debug_snap_draws.py
"""

import tempfile
import numpy as np
import pandas as pd

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
from policyengine_us_data.utils.takeup import (
    TAKEUP_AFFECTED_TARGETS,
    _resolve_rate,
    _build_entity_to_hh_index,
    SIMPLE_TAKEUP_VARS,
)
from policyengine_us_data.utils.randomness import (
    seeded_rng,
    _stable_string_hash,
)
from policyengine_us_data.parameters import load_take_up_rate

DATASET_PATH = str(STORAGE_FOLDER / "stratified_extended_cps_2024.h5")
DB_PATH = str(STORAGE_FOLDER / "calibration" / "policy_data.db")
DB_URI = f"sqlite:///{DB_PATH}"

SEED = 42
N_CLONES = 3
N_SAMPLE = 10
TARGET_CD = "3701"  # NC CD-01
TIME_PERIOD = 2024
TAKEUP_VAR = "takes_up_snap_if_eligible"
TARGET_VAR = "snap"
RATE_KEY = "snap"
ENTITY_LEVEL = "spm_unit"


def main():
    # ================================================================
    # Setup: Load dataset, create geography, build matrix
    # ================================================================
    print("=" * 70)
    print("SETUP: Load dataset, create geography, build matrix")
    print("=" * 70)

    sim = Microsimulation(dataset=DATASET_PATH)
    n_records = len(sim.calculate("household_id", map_to="household").values)
    print(f"  Base households: {n_records:,}")

    geography = assign_random_geography(
        n_records=n_records, n_clones=N_CLONES, seed=SEED
    )
    n_total = n_records * N_CLONES

    builder = UnifiedMatrixBuilder(
        db_uri=DB_URI,
        time_period=TIME_PERIOD,
        dataset_path=DATASET_PATH,
    )

    target_filter = {"variables": ["aca_ptc", "snap", "household_count"]}
    targets_df, X, target_names = builder.build_matrix(
        geography=geography,
        sim=sim,
        target_filter=target_filter,
        hierarchical_domains=["aca_ptc", "snap"],
        rerandomize_takeup=True,
    )
    print(f"  Matrix shape: {X.shape}")

    target_vars = set(target_filter["variables"])
    takeup_filter = [
        info["takeup_var"]
        for key, info in TAKEUP_AFFECTED_TARGETS.items()
        if key in target_vars
    ]
    print(f"  Takeup filter: {takeup_filter}")

    # Uniform weights and stacked format
    w = np.ones(n_total, dtype=np.float64)
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

    # ================================================================
    # Step 1: Pick target households
    # ================================================================
    print("\n" + "=" * 70)
    print(f"STEP 1: Pick {N_SAMPLE} households in CD {TARGET_CD}")
    print("=" * 70)

    # Find records assigned to this CD
    cd_mask_cols = geo_cd_strs == TARGET_CD
    cd_col_indices = np.where(cd_mask_cols)[0]
    print(f"  Columns in CD {TARGET_CD}: {len(cd_col_indices)}")

    # Get record indices (within base dataset) for these columns
    cd_record_indices = cd_col_indices % n_records
    cd_clone_indices = cd_col_indices // n_records
    print(f"  Clones present: " f"{sorted(set(cd_clone_indices.tolist()))}")

    # Use the base sim to find SNAP-eligible SPM units
    # Force takeup=True to get eligible amounts
    base_sim = Microsimulation(dataset=DATASET_PATH)
    for spec in SIMPLE_TAKEUP_VARS:
        var_name = spec["variable"]
        entity = spec["entity"]
        n_ent = len(base_sim.calculate(f"{entity}_id", map_to=entity).values)
        base_sim.set_input(
            var_name,
            TIME_PERIOD,
            np.ones(n_ent, dtype=bool),
        )
    # Set state_fips to NC for all
    base_sim.set_input(
        "state_fips",
        TIME_PERIOD,
        np.full(n_records, 37, dtype=np.int32),
    )
    from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
        get_calculated_variables,
    )

    for var in get_calculated_variables(base_sim):
        base_sim.delete_arrays(var)

    # Get SPM unit level SNAP eligible amounts
    spm_snap = base_sim.calculate(
        "snap", TIME_PERIOD, map_to="spm_unit"
    ).values
    spm_ids = base_sim.calculate("spm_unit_id", map_to="spm_unit").values
    household_ids = base_sim.calculate(
        "household_id", map_to="household"
    ).values
    hh_id_to_idx = {int(hid): idx for idx, hid in enumerate(household_ids)}

    # Build entity-to-household mapping
    entity_rel = pd.DataFrame(
        {
            "person_id": base_sim.calculate(
                "person_id", map_to="person"
            ).values,
            "household_id": base_sim.calculate(
                "household_id", map_to="person"
            ).values,
            "spm_unit_id": base_sim.calculate(
                "spm_unit_id", map_to="person"
            ).values,
        }
    )
    spm_to_hh = (
        entity_rel.groupby("spm_unit_id")["household_id"].first().to_dict()
    )
    spm_hh_idx = np.array(
        [hh_id_to_idx[int(spm_to_hh[int(sid)])] for sid in spm_ids]
    )

    # Find households in our CD with nonzero SNAP eligible
    # (at least one SPM unit with snap > 0)
    cd_unique_records = sorted(set(cd_record_indices.tolist()))
    eligible_records = []
    for rec_idx in cd_unique_records:
        hh_id = int(household_ids[rec_idx])
        # SPM units belonging to this household
        spm_mask = spm_hh_idx == rec_idx
        spm_eligible = spm_snap[spm_mask]
        n_spm = int(spm_mask.sum())
        if n_spm > 0 and spm_eligible.sum() > 0:
            eligible_records.append(
                {
                    "record_idx": rec_idx,
                    "household_id": hh_id,
                    "n_spm_units": n_spm,
                    "snap_eligible_per_spm": spm_eligible.tolist(),
                    "total_snap_eligible": float(spm_eligible.sum()),
                }
            )

    print(
        f"  Records in CD with SNAP-eligible SPM units: "
        f"{len(eligible_records)}"
    )

    # Pick up to N_SAMPLE
    sample = eligible_records[:N_SAMPLE]
    print(f"  Sampled: {len(sample)} households\n")
    print(
        f"  {'rec_idx':>8s}  {'hh_id':>8s}  "
        f"{'n_spm':>5s}  {'total_eligible':>14s}"
    )
    print("  " + "-" * 42)
    for s in sample:
        print(
            f"  {s['record_idx']:8d}  {s['household_id']:8d}  "
            f"{s['n_spm_units']:5d}  "
            f"${s['total_snap_eligible']:>12,.0f}"
        )

    # ================================================================
    # Step 2: Matrix builder side
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Matrix builder draw details")
    print("=" * 70)

    rate_or_dict = load_take_up_rate(RATE_KEY, TIME_PERIOD)
    nc_rate = _resolve_rate(rate_or_dict, 37)
    print(f"  SNAP takeup rate for NC (FIPS 37): {nc_rate}")

    # For each sampled household, trace the matrix builder's draws
    # The matrix builder iterates clone by clone
    matrix_results = []

    for s in sample:
        rec_idx = s["record_idx"]
        hh_id = s["household_id"]
        spm_mask = spm_hh_idx == rec_idx
        n_spm = int(spm_mask.sum())
        spm_eligible = spm_snap[spm_mask]

        print(
            f"\n  --- HH {hh_id} (rec_idx={rec_idx}, "
            f"{n_spm} SPM units) ---"
        )

        hh_clones = []
        for clone_idx in range(N_CLONES):
            col = clone_idx * n_records + rec_idx
            if geo_cd_strs[col] != TARGET_CD:
                continue

            block = str(geography.block_geoid[col])
            salt = f"{block}:{hh_id}"
            seed_val = int(_stable_string_hash(f"{TAKEUP_VAR}:{salt}")) % (
                2**63
            )

            rng = seeded_rng(TAKEUP_VAR, salt=salt)
            draws = rng.random(n_spm)
            takeup = draws < nc_rate
            final_vals = spm_eligible * takeup
            hh_snap = float(final_vals.sum())

            # Get the actual X matrix value for this column
            # Find the state-level SNAP row for NC
            snap_nc_row = targets_df[
                (targets_df["variable"] == "snap")
                & (targets_df["geographic_id"] == "37")
            ]
            x_val = None
            if len(snap_nc_row) > 0:
                row_num = targets_df.index.get_loc(snap_nc_row.index[0])
                x_val = float(X[row_num, col])

            print(f"    Clone {clone_idx}: " f"block={block[:15]}...")
            print(f'      salt  = "{salt[:40]}..."')
            print(f"      seed  = {seed_val}")
            print(f"      draws = {draws}")
            print(f"      rate  = {nc_rate}")
            print(f"      takeup= {takeup}")
            print(f"      eligible = {spm_eligible}")
            print(f"      final    = {final_vals}")
            print(f"      hh_snap  = ${hh_snap:,.0f}")
            if x_val is not None:
                print(f"      X[snap_NC, col={col}] = " f"${x_val:,.0f}")

            hh_clones.append(
                {
                    "clone_idx": clone_idx,
                    "col": col,
                    "block": block,
                    "salt": salt,
                    "seed": seed_val,
                    "draws": draws.copy(),
                    "takeup": takeup.copy(),
                    "eligible": spm_eligible.copy(),
                    "final": final_vals.copy(),
                    "hh_snap": hh_snap,
                    "x_val": x_val,
                }
            )

        matrix_results.append(
            {
                "record_idx": rec_idx,
                "household_id": hh_id,
                "n_spm": n_spm,
                "clones": hh_clones,
            }
        )

    # ================================================================
    # Step 3: Stacked builder side
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Stacked builder draw details")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()
    h5_path = f"{tmpdir}/{TARGET_CD}.h5"

    print(f"  Building stacked h5 for CD {TARGET_CD}...")
    create_sparse_cd_stacked_dataset(
        w=w_stacked,
        cds_to_calibrate=cds_ordered,
        cd_subset=[TARGET_CD],
        output_path=h5_path,
        dataset_path=DATASET_PATH,
        rerandomize_takeup=True,
        calibration_blocks=blocks_stacked,
        takeup_filter=takeup_filter,
    )

    print("  Loading stacked sim...")
    stacked_sim = Microsimulation(dataset=h5_path)

    # Get household-level SNAP from stacked sim
    stacked_snap_hh = stacked_sim.calculate(
        "snap", TIME_PERIOD, map_to="household"
    ).values
    stacked_hh_weight = stacked_sim.calculate(
        "household_weight", TIME_PERIOD, map_to="household"
    ).values
    stacked_hh_ids = stacked_sim.calculate(
        "household_id", map_to="household"
    ).values

    # Get SPM-level details from stacked sim
    stacked_spm_snap = stacked_sim.calculate(
        "snap", TIME_PERIOD, map_to="spm_unit"
    ).values
    stacked_spm_takeup = stacked_sim.calculate(
        TAKEUP_VAR, TIME_PERIOD, map_to="spm_unit"
    ).values
    stacked_spm_ids = stacked_sim.calculate(
        "spm_unit_id", map_to="spm_unit"
    ).values

    # Build stacked entity-to-household mapping
    stacked_entity_idx = _build_entity_to_hh_index(stacked_sim)
    stacked_spm_hh_idx = stacked_entity_idx["spm_unit"]

    # Get blocks from the stacked sim's inputs
    # (these were set during stacked dataset building)
    stacked_block_geoid = stacked_sim.calculate(
        "block_geoid", TIME_PERIOD, map_to="household"
    ).values

    # Also manually reproduce the draws on the stacked sim
    # to see what apply_block_takeup_draws_to_sim would produce
    print("\n  Tracing stacked builder draws for sampled HHs:")

    # The stacked sim has reindexed IDs. We need to map back
    # to original household IDs via the household mapping CSV.
    # But the mapping CSV might not be saved in this case.
    # Instead, reconstruct from the stacked format.

    # The stacked builder uses cd_blocks which are from
    # blocks_stacked for this CD. Let's get those directly.
    cal_idx = cds_ordered.index(TARGET_CD)
    cd_blocks_raw = blocks_stacked[
        cal_idx * n_records : (cal_idx + 1) * n_records
    ]

    # Also get the stacked weights for this CD to know
    # which records are active
    cd_weights_raw = w_stacked[cal_idx * n_records : (cal_idx + 1) * n_records]
    active_mask = cd_weights_raw > 0
    active_indices = np.where(active_mask)[0]
    print(f"  Active records in CD: {len(active_indices)}")

    # Now manually reproduce what the stacked builder does:
    # It creates a fresh sim, sets state_fips, sets blocks,
    # then calls apply_block_takeup_draws_to_sim with cd_blocks_raw.
    #
    # apply_block_takeup_draws_to_sim:
    # 1. Gets hh_ids from sim (original IDs)
    # 2. Builds entity_hh_idx via _build_entity_to_hh_index
    # 3. For each SPM unit: block = hh_blocks[hh_idx],
    #    hh_id = hh_ids[hh_idx]
    # 4. Calls compute_block_takeup_for_entities which loops
    #    per (block, hh_id) and uses
    #    seeded_rng(var, salt=f"{block}:{hh_id}")

    # Create a fresh sim to reproduce the stacked builder's
    # exact draw path
    repro_sim = Microsimulation(dataset=DATASET_PATH)
    repro_hh_ids = repro_sim.calculate(
        "household_id", map_to="household"
    ).values
    repro_spm_ids = repro_sim.calculate(
        "spm_unit_id", map_to="spm_unit"
    ).values

    # Build entity-to-hh index on the repro sim
    repro_entity_idx = _build_entity_to_hh_index(repro_sim)
    repro_spm_hh_idx = repro_entity_idx["spm_unit"]

    stacked_results = []

    for s in sample:
        rec_idx = s["record_idx"]
        hh_id = s["household_id"]
        n_spm = s["n_spm_units"]

        print(
            f"\n  --- HH {hh_id} (rec_idx={rec_idx}, "
            f"{n_spm} SPM units) ---"
        )

        # What the stacked builder sees for this record:
        block_for_record = str(cd_blocks_raw[rec_idx])
        weight_for_record = cd_weights_raw[rec_idx]
        print(f"    block (from calibration): " f"{block_for_record[:15]}...")
        print(f"    weight: {weight_for_record}")
        print(f"    active: {weight_for_record > 0}")

        # SPM units for this household in the repro sim
        repro_spm_mask = repro_spm_hh_idx == rec_idx
        repro_spm_for_hh = np.where(repro_spm_mask)[0]
        print(f"    SPM unit indices: {repro_spm_for_hh}")

        # Reproduce the draws exactly as the stacked builder would
        for spm_local_idx, spm_global_idx in enumerate(repro_spm_for_hh):
            repro_hh_idx = repro_spm_hh_idx[spm_global_idx]
            repro_block = str(cd_blocks_raw[repro_hh_idx])
            repro_hh_id = int(repro_hh_ids[repro_hh_idx])
            print(
                f"    SPM[{spm_global_idx}]: "
                f"hh_idx={repro_hh_idx}, "
                f"hh_id={repro_hh_id}, "
                f"block={repro_block[:15]}..."
            )

        # Now do the actual draw computation as
        # compute_block_takeup_for_entities would
        # Entity-level blocks and hh_ids
        ent_blocks = np.array(
            [str(cd_blocks_raw[repro_spm_hh_idx[i]]) for i in repro_spm_for_hh]
        )
        ent_hh_ids_arr = repro_hh_ids[repro_spm_hh_idx[repro_spm_for_hh]]
        ent_states = np.full(len(repro_spm_for_hh), 37)

        # Reproduce the per-(block, hh) draw loop
        print(f"    Reproducing draws (stacked path):")
        for blk in np.unique(ent_blocks):
            bm = ent_blocks == blk
            sf = int(blk[:2]) if blk else 0
            rate = _resolve_rate(rate_or_dict, sf)
            for hh_id_val in np.unique(ent_hh_ids_arr[bm]):
                hh_mask = bm & (ent_hh_ids_arr == hh_id_val)
                n_draws = int(hh_mask.sum())
                salt = f"{blk}:{int(hh_id_val)}"
                seed_val = int(_stable_string_hash(f"{TAKEUP_VAR}:{salt}")) % (
                    2**63
                )
                rng = seeded_rng(TAKEUP_VAR, salt=salt)
                draws = rng.random(n_draws)
                takeup = draws < rate
                print(f"      block={blk[:15]}..., " f"hh_id={int(hh_id_val)}")
                print(f'        salt  = "{salt[:40]}..."')
                print(f"        seed  = {seed_val}")
                print(f"        draws = {draws}")
                print(f"        rate  = {rate}")
                print(f"        takeup= {takeup}")

        # Now check what the ACTUAL stacked sim computed
        # We need to find this household in the stacked sim
        # The stacked sim has reindexed IDs, so we need
        # to find the new ID for this original household.
        # The stacked builder assigns new IDs based on
        # cd_to_index and a counter.
        # Since we only have 1 CD in this subset,
        # the new IDs start at cd_idx * 25000.
        # We can't directly map, so let's use the stacked sim's
        # block_geoid to match.

        # Actually, a simpler approach: match on block + weight
        # Or we can look at the household mapping approach.
        # Let's try to find by matching snap values.

        # For now, get aggregate from stacked sim
        stacked_hh_info = {
            "snap_hh_values": stacked_snap_hh.tolist(),
            "hh_ids": stacked_hh_ids.tolist(),
        }

        stacked_results.append(
            {
                "record_idx": rec_idx,
                "household_id": hh_id,
                "block": block_for_record,
                "weight": weight_for_record,
            }
        )

    # ================================================================
    # Step 4: Side-by-side comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Side-by-side comparison")
    print("=" * 70)

    # Also do a full aggregate comparison for this CD
    # Matrix builder: X @ w for snap/CD row
    xw = X @ w
    snap_cd_row = targets_df[
        (targets_df["variable"] == "snap")
        & (targets_df["geographic_id"] == TARGET_CD)
    ]
    if len(snap_cd_row) > 0:
        row_num = targets_df.index.get_loc(snap_cd_row.index[0])
        matrix_cd_snap = float(xw[row_num])
    else:
        matrix_cd_snap = None

    stacked_cd_snap = float((stacked_snap_hh * stacked_hh_weight).sum())

    print(f"\n  CD-level SNAP for {TARGET_CD}:")
    if matrix_cd_snap is not None:
        print(f"    Matrix (X @ w): ${matrix_cd_snap:>12,.0f}")
    print(f"    Stacked sum:    ${stacked_cd_snap:>12,.0f}")
    if matrix_cd_snap is not None and stacked_cd_snap != 0:
        ratio = matrix_cd_snap / stacked_cd_snap
        print(f"    Ratio:          {ratio:.6f}")

    # State-level NC check
    snap_nc_row = targets_df[
        (targets_df["variable"] == "snap")
        & (targets_df["geographic_id"] == "37")
    ]
    if len(snap_nc_row) > 0:
        row_num = targets_df.index.get_loc(snap_nc_row.index[0])
        matrix_nc_snap = float(xw[row_num])
        print(f"\n  State-level SNAP for NC (FIPS 37):")
        print(f"    Matrix (X @ w): ${matrix_nc_snap:>12,.0f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
