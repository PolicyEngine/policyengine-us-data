"""
Test whether the state precomputation loop produces different SNAP
eligible amounts than a fresh sim.

Hypothesis: cycling 51 states on one sim object leaves stale
intermediate state that pollutes SNAP values for some households.

Three comparisons:
  A) Fresh sim, state=37, takeup=True → baseline
  B) Same sim after cycling states 1..51 → extract state 37
  C) Fresh sim, set state=36, delete, set state=37 → minimal cycle

If B != A, we've found the pollution.
If C != A but B == A, the issue is multi-state accumulation.

Usage:
    python scripts/debug_state_precomp.py
"""

import numpy as np

from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.takeup import SIMPLE_TAKEUP_VARS
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_calculated_variables,
)

DATASET_PATH = str(STORAGE_FOLDER / "stratified_extended_cps_2024.h5")
TIME_PERIOD = 2024
NC_FIPS = 37


def force_takeup_true(sim):
    """Set all simple takeup variables to True."""
    for spec in SIMPLE_TAKEUP_VARS:
        var_name = spec["variable"]
        entity = spec["entity"]
        n_ent = len(sim.calculate(f"{entity}_id", map_to=entity).values)
        sim.set_input(var_name, TIME_PERIOD, np.ones(n_ent, dtype=bool))


def set_state(sim, fips, n_hh):
    """Set state_fips and delete calculated caches."""
    sim.set_input(
        "state_fips",
        TIME_PERIOD,
        np.full(n_hh, fips, dtype=np.int32),
    )
    for var in get_calculated_variables(sim):
        sim.delete_arrays(var)


def get_snap_spm(sim):
    """Get SNAP at spm_unit level."""
    return sim.calculate("snap", TIME_PERIOD, map_to="spm_unit").values.astype(
        np.float32
    )


def get_snap_hh(sim):
    """Get SNAP at household level."""
    return sim.calculate(
        "snap", TIME_PERIOD, map_to="household"
    ).values.astype(np.float32)


def main():
    # ================================================================
    # A) Fresh sim baseline: state=37, takeup=True
    # ================================================================
    print("=" * 70)
    print("A) FRESH SIM BASELINE: state=37, takeup=True")
    print("=" * 70)

    sim_a = Microsimulation(dataset=DATASET_PATH)
    n_hh = len(sim_a.calculate("household_id", map_to="household").values)
    print(f"  Households: {n_hh:,}")

    force_takeup_true(sim_a)
    set_state(sim_a, NC_FIPS, n_hh)

    snap_spm_a = get_snap_spm(sim_a)
    snap_hh_a = get_snap_hh(sim_a)
    print(f"  SPM units: {len(snap_spm_a):,}")
    print(f"  SNAP total (hh): ${snap_hh_a.sum():,.0f}")
    print(f"  SNAP total (spm): ${snap_spm_a.sum():,.0f}")
    print(f"  Nonzero SPM units: {(snap_spm_a > 0).sum()}")

    # ================================================================
    # B) Loop sim: cycle all 51 states, extract state 37
    # ================================================================
    print("\n" + "=" * 70)
    print("B) LOOP SIM: cycle states 1..56, extract state 37")
    print("=" * 70)

    sim_b = Microsimulation(dataset=DATASET_PATH)
    force_takeup_true(sim_b)

    # All unique state FIPS codes
    all_states = sorted(
        set(
            int(s)
            for s in [
                1,
                2,
                4,
                5,
                6,
                8,
                9,
                10,
                11,
                12,
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                53,
                54,
                55,
                56,
            ]
        )
    )
    print(f"  Cycling through {len(all_states)} states...")

    snap_spm_b = None
    snap_hh_b = None
    for i, state in enumerate(all_states):
        set_state(sim_b, state, n_hh)

        # Calculate snap for every state (mimics builder)
        spm_vals = get_snap_spm(sim_b)
        hh_vals = get_snap_hh(sim_b)

        if state == NC_FIPS:
            snap_spm_b = spm_vals.copy()
            snap_hh_b = hh_vals.copy()
            nc_position = i
            print(
                f"  State {state} (NC) at position {i}: "
                f"spm_total=${spm_vals.sum():,.0f}, "
                f"hh_total=${hh_vals.sum():,.0f}"
            )

        if (i + 1) % 10 == 0:
            print(f"  ...processed {i + 1}/{len(all_states)}")

    print(f"  Done. NC was at position {nc_position}.")

    # ================================================================
    # C) Minimal cycle: state=36 → state=37
    # ================================================================
    print("\n" + "=" * 70)
    print("C) MINIMAL CYCLE: state=36 → state=37")
    print("=" * 70)

    sim_c = Microsimulation(dataset=DATASET_PATH)
    force_takeup_true(sim_c)

    # First compute for NY (state 36)
    set_state(sim_c, 36, n_hh)
    snap_ny = get_snap_spm(sim_c)
    _ = get_snap_hh(sim_c)
    print(f"  After state=36 (NY): spm_total=${snap_ny.sum():,.0f}")

    # Now switch to NC
    set_state(sim_c, NC_FIPS, n_hh)
    snap_spm_c = get_snap_spm(sim_c)
    snap_hh_c = get_snap_hh(sim_c)
    print(
        f"  After state=37 (NC): spm_total=${snap_spm_c.sum():,.0f}, "
        f"hh_total=${snap_hh_c.sum():,.0f}"
    )

    # ================================================================
    # D) Extra: state=37 computed TWICE on same sim (no other state)
    # ================================================================
    print("\n" + "=" * 70)
    print("D) SAME SIM, state=37 TWICE")
    print("=" * 70)

    sim_d = Microsimulation(dataset=DATASET_PATH)
    force_takeup_true(sim_d)

    set_state(sim_d, NC_FIPS, n_hh)
    snap_spm_d1 = get_snap_spm(sim_d)
    snap_hh_d1 = get_snap_hh(sim_d)
    print(
        f"  First:  spm_total=${snap_spm_d1.sum():,.0f}, "
        f"hh_total=${snap_hh_d1.sum():,.0f}"
    )

    set_state(sim_d, NC_FIPS, n_hh)
    snap_spm_d2 = get_snap_spm(sim_d)
    snap_hh_d2 = get_snap_hh(sim_d)
    print(
        f"  Second: spm_total=${snap_spm_d2.sum():,.0f}, "
        f"hh_total=${snap_hh_d2.sum():,.0f}"
    )

    # ================================================================
    # Compare
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    def compare(label, spm_test, hh_test, spm_base, hh_base):
        spm_diff = spm_test - spm_base
        hh_diff = hh_test - hh_base
        n_spm_diff = (np.abs(spm_diff) > 0.01).sum()
        n_hh_diff = (np.abs(hh_diff) > 0.01).sum()
        spm_total_diff = spm_diff.sum()
        hh_total_diff = hh_diff.sum()

        status = "MATCH" if n_spm_diff == 0 else "DIVERGE"
        print(f"\n  {label}: [{status}]")
        print(f"    SPM units differ: {n_spm_diff} / {len(spm_diff)}")
        print(f"    Households differ: {n_hh_diff} / {len(hh_diff)}")
        print(
            f"    SPM total: baseline=${spm_base.sum():,.0f}, "
            f"test=${spm_test.sum():,.0f}, "
            f"diff=${spm_total_diff:,.0f}"
        )
        print(
            f"    HH total:  baseline=${hh_base.sum():,.0f}, "
            f"test=${hh_test.sum():,.0f}, "
            f"diff=${hh_total_diff:,.0f}"
        )

        if n_spm_diff > 0:
            ratio = spm_test.sum() / spm_base.sum()
            print(f"    Ratio: {ratio:.6f}")

            # Show the top divergent SPM units
            abs_diff = np.abs(spm_diff)
            top_idx = np.argsort(abs_diff)[-10:][::-1]
            print(f"\n    Top {min(10, n_spm_diff)} divergent " f"SPM units:")
            print(
                f"    {'idx':>6s}  {'baseline':>10s}  "
                f"{'test':>10s}  {'diff':>10s}  {'pct':>8s}"
            )
            print("    " + "-" * 50)
            for idx in top_idx:
                if abs_diff[idx] < 0.01:
                    break
                pct = (
                    spm_diff[idx] / spm_base[idx] * 100
                    if spm_base[idx] != 0
                    else float("inf")
                )
                print(
                    f"    {idx:6d}  "
                    f"${spm_base[idx]:>9,.0f}  "
                    f"${spm_test[idx]:>9,.0f}  "
                    f"${spm_diff[idx]:>9,.0f}  "
                    f"{pct:>7.1f}%"
                )

        if n_hh_diff > 0:
            abs_hh_diff = np.abs(hh_diff)
            top_hh = np.argsort(abs_hh_diff)[-5:][::-1]
            print(f"\n    Top divergent households:")
            print(
                f"    {'idx':>6s}  {'baseline':>10s}  "
                f"{'test':>10s}  {'diff':>10s}"
            )
            print("    " + "-" * 42)
            for idx in top_hh:
                if abs_hh_diff[idx] < 0.01:
                    break
                print(
                    f"    {idx:6d}  "
                    f"${hh_base[idx]:>9,.0f}  "
                    f"${hh_test[idx]:>9,.0f}  "
                    f"${hh_diff[idx]:>9,.0f}"
                )

        return n_spm_diff

    n1 = compare(
        "B vs A (loop vs fresh)",
        snap_spm_b,
        snap_hh_b,
        snap_spm_a,
        snap_hh_a,
    )
    n2 = compare(
        "C vs A (36→37 vs fresh)",
        snap_spm_c,
        snap_hh_c,
        snap_spm_a,
        snap_hh_a,
    )
    n3 = compare(
        "D vs A (37 twice vs fresh)",
        snap_spm_d2,
        snap_hh_d2,
        snap_spm_a,
        snap_hh_a,
    )
    n4 = compare(
        "D1 vs A (37 first vs fresh)",
        snap_spm_d1,
        snap_hh_d1,
        snap_spm_a,
        snap_hh_a,
    )

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if n1 > 0:
        print(
            "  >>> STATE LOOP POLLUTION CONFIRMED: "
            "cycling states changes SNAP eligible amounts"
        )
    elif n2 > 0:
        print(
            "  >>> MINIMAL POLLUTION: even one state " "switch changes values"
        )
    elif n3 > 0 or n4 > 0:
        print(
            "  >>> SELF-POLLUTION: even recalculating "
            "the same state changes values"
        )
    else:
        print(
            "  >>> NO POLLUTION FOUND: all computations "
            "match the fresh baseline"
        )
        print(
            "      The X matrix discrepancy must come " "from somewhere else."
        )


if __name__ == "__main__":
    main()
