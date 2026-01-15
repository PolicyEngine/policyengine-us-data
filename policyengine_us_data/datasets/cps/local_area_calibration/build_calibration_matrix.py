"""
Build calibration matrix for geo-stacking reweighting.
Generates X_sparse and target vector, prints diagnostics using MatrixTracer.
"""

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from sparse_matrix_builder import SparseMatrixBuilder
from matrix_tracer import MatrixTracer
from calibration_utils import create_target_groups

# ============================================================================
# CONFIGURATION
# ============================================================================
db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
db_uri = f"sqlite:///{db_path}"
time_period = 2023

# Base dataset for geo-stacking: stratified extended CPS
dataset_path = STORAGE_FOLDER / "stratified_extended_cps_2023.h5"

cds_to_calibrate = [
    "101",  # Alabama CD-1
    "601",  # California CD-1
    "602",  # California CD-2
    "3601",  # New York CD-1
    "4801",  # Texas CD-1
]

print(f"Testing with {len(cds_to_calibrate)} congressional districts")

# ============================================================================
# STEP 1: LOAD SIMULATION FROM EXTENDED CPS
# ============================================================================
print(f"\nLoading simulation from {dataset_path}...")
sim = Microsimulation(dataset=str(dataset_path))
n_households = len(sim.calculate("household_id", map_to="household").values)
print(f"Loaded {n_households:,} households")

# ============================================================================
# STEP 2: BUILD SPARSE MATRIX WITH COMBINED TARGETS
# ============================================================================
print("\nBuilding sparse matrix...")
builder = SparseMatrixBuilder(
    db_uri=db_uri,
    time_period=time_period,
    cds_to_calibrate=cds_to_calibrate,
    dataset_path=None,
)

# SNAP targets (stratum_group_id=4) + specific health insurance variable
# Uses OR logic: gets all SNAP targets OR the health insurance target
targets_df, X_sparse, household_id_mapping = builder.build_matrix(
    sim,
    target_filter={
        "stratum_group_ids": [4],
        "variables": ["health_insurance_premiums_without_medicare_part_b"],
    },
)

print(f"\nMatrix built successfully:")
print(f"  Shape: {X_sparse.shape}")
print(f"  Targets: {len(targets_df)}")
nnz = X_sparse.nnz
total = X_sparse.shape[0] * X_sparse.shape[1]
print(f"  Sparsity: {1 - nnz / total:.4%}")

# ============================================================================
# STEP 3: EXTRACT TARGET VECTOR
# ============================================================================
target_vector = targets_df["value"].values
print(f"\nTarget vector shape: {target_vector.shape}")
print(f"Target total: ${target_vector.sum():,.0f}")

# ============================================================================
# STEP 4: HEALTH INSURANCE PREMIUM VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("HEALTH INSURANCE PREMIUM TARGET ANALYSIS")
print("=" * 80)

health_ins_targets = targets_df[
    targets_df["variable"]
    == "health_insurance_premiums_without_medicare_part_b"
]

if len(health_ins_targets) > 0:
    print(f"\nFound {len(health_ins_targets)} health insurance target(s):")
    print(
        health_ins_targets[
            [
                "target_id",
                "variable",
                "value",
                "geographic_id",
                "stratum_group_id",
            ]
        ]
    )

    health_ins_idx = health_ins_targets.index[0]
    health_ins_row = X_sparse[health_ins_idx, :]

    print(f"\nMatrix row {health_ins_idx} (health insurance):")
    print(f"  Non-zero entries: {health_ins_row.nnz:,}")
    print(f"  Row sum: ${health_ins_row.sum():,.0f}")
    print(f"  Target value: ${health_ins_targets.iloc[0]['value']:,.0f}")

    person_total = sim.calculate(
        "health_insurance_premiums_without_medicare_part_b",
        time_period,
        map_to="person",
    ).values.sum()

    household_total = sim.calculate(
        "health_insurance_premiums_without_medicare_part_b",
        time_period,
        map_to="household",
    ).values.sum()

    print(f"\nEntity aggregation verification:")
    print(f"  Person-level total:    ${person_total:,.0f}")
    print(f"  Household-level total: ${household_total:,.0f}")
    print(f"  Match: {np.isclose(person_total, household_total, rtol=1e-6)}")

else:
    print("\nWARNING: No health insurance targets found!")

# ============================================================================
# STEP 5: SNAP TARGET SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SNAP TARGET SUMMARY")
print("=" * 80)

snap_targets = targets_df[targets_df["variable"] == "snap"]
household_count_targets = targets_df[
    targets_df["variable"] == "household_count"
]

print(f"\nSNAP benefit targets: {len(snap_targets)}")
print(f"Household count targets: {len(household_count_targets)}")

if len(snap_targets) > 0:
    print(f"\nSNAP total (all states): ${snap_targets['value'].sum():,.0f}")
    print(f"\nSample SNAP targets:")
    print(
        snap_targets[["target_id", "variable", "value", "geographic_id"]].head(
            10
        )
    )

# ============================================================================
# STEP 6: USE MATRIX TRACER FOR DETAILED DIAGNOSTICS
# ============================================================================
print("\n" + "=" * 80)
print("MATRIX TRACER DIAGNOSTICS")
print("=" * 80)

tracer = MatrixTracer(
    targets_df=targets_df,
    matrix=X_sparse,
    household_id_mapping=household_id_mapping,
    geographic_ids=cds_to_calibrate,
    sim=sim,
)

tracer.print_matrix_structure(show_groups=True)

# ============================================================================
# STEP 7: TARGET GROUP ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("TARGET GROUP ANALYSIS")
print("=" * 80)

target_groups, group_info = create_target_groups(targets_df)

print(f"\nTotal target groups: {len(group_info)}")
for group_id, info in enumerate(group_info):
    group_mask = target_groups == group_id
    n_targets_in_group = group_mask.sum()
    print(f"  Group {group_id}: {info} ({n_targets_in_group} targets)")

print("\n" + "=" * 80)
print("RUNNER COMPLETED SUCCESSFULLY")
print("=" * 80)
