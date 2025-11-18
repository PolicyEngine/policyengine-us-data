#!/usr/bin/env python
"""
Weight diagnostics for geo-stacked calibration (states or congressional districts).
Analyzes calibration weights to understand sparsity patterns and accuracy.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import sparse as sp
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import (
    create_target_groups,
)


def load_calibration_data(geo_level="state"):
    """Load calibration matrix, weights, and targets for the specified geo level."""

    if geo_level == "state":
        export_dir = os.path.expanduser("~/Downloads/state_calibration_data")
        weight_file = "/home/baogorek/Downloads/w_array_20250908_185748.npy"
        matrix_file = "X_sparse.npz"
        targets_file = "targets_df.pkl"
        dataset_uri = "hf://policyengine/test/extended_cps_2023.h5"
    else:  # congressional_district
        export_dir = os.path.expanduser("~/Downloads/cd_calibration_data")
        weight_file = "w_cd_20250911_102023.npy"
        matrix_file = "cd_matrix_sparse.npz"
        targets_file = "cd_targets_df.pkl"
        dataset_uri = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/stratified_extended_cps_2023.h5"

    print(f"Loading {geo_level} calibration data...")

    # Check for weight file in multiple locations
    if os.path.exists(weight_file):
        w = np.load(weight_file)
    elif os.path.exists(
        os.path.join(export_dir, os.path.basename(weight_file))
    ):
        w = np.load(os.path.join(export_dir, os.path.basename(weight_file)))
    else:
        print(f"Error: Weight file not found at {weight_file}")
        sys.exit(1)

    # Load matrix
    matrix_path = os.path.join(export_dir, matrix_file)
    if os.path.exists(matrix_path):
        X_sparse = sp.load_npz(matrix_path)
    else:
        # Try downloading from huggingface for states
        if geo_level == "state":
            from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import (
                download_from_huggingface,
            )

            X_sparse = sp.load_npz(download_from_huggingface(matrix_file))
        else:
            print(f"Error: Matrix file not found at {matrix_path}")
            sys.exit(1)

    # Load targets
    targets_path = os.path.join(export_dir, targets_file)
    if os.path.exists(targets_path):
        targets_df = pd.read_pickle(targets_path)
    else:
        # Try downloading from huggingface for states
        if geo_level == "state":
            from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import (
                download_from_huggingface,
            )

            targets_df = pd.read_pickle(
                download_from_huggingface(targets_file)
            )
        else:
            print(f"Error: Targets file not found at {targets_path}")
            sys.exit(1)

    # Load simulation
    print(f"Loading simulation from {dataset_uri}...")
    sim = Microsimulation(dataset=dataset_uri)
    sim.build_from_dataset()

    return w, X_sparse, targets_df, sim


def analyze_weight_statistics(w):
    """Analyze basic weight statistics."""
    print("\n" + "=" * 70)
    print("WEIGHT STATISTICS")
    print("=" * 70)

    n_active = sum(w != 0)
    print(f"Total weights: {len(w):,}")
    print(f"Active weights (non-zero): {n_active:,}")
    print(f"Sparsity: {100*n_active/len(w):.2f}%")

    if n_active > 0:
        active_weights = w[w != 0]
        print(f"\nActive weight statistics:")
        print(f"  Min: {active_weights.min():.2f}")
        print(f"  Max: {active_weights.max():.2f}")
        print(f"  Mean: {active_weights.mean():.2f}")
        print(f"  Median: {np.median(active_weights):.2f}")
        print(f"  Std: {active_weights.std():.2f}")

    return n_active


def analyze_prediction_errors(w, X_sparse, targets_df):
    """Analyze prediction errors."""
    print("\n" + "=" * 70)
    print("PREDICTION ERROR ANALYSIS")
    print("=" * 70)

    # Calculate predictions
    y_pred = X_sparse @ w
    y_actual = targets_df["value"].values

    correlation = np.corrcoef(y_pred, y_actual)[0, 1]
    print(f"Correlation between predicted and actual: {correlation:.4f}")

    # Calculate errors
    abs_errors = np.abs(y_actual - y_pred)
    rel_errors = np.abs((y_actual - y_pred) / (y_actual + 1))

    targets_df["y_pred"] = y_pred
    targets_df["abs_error"] = abs_errors
    targets_df["rel_error"] = rel_errors

    # Overall statistics
    print(f"\nOverall error statistics:")
    print(f"  Mean relative error: {np.mean(rel_errors):.2%}")
    print(f"  Median relative error: {np.median(rel_errors):.2%}")
    print(f"  Max relative error: {np.max(rel_errors):.2%}")
    print(f"  95th percentile: {np.percentile(rel_errors, 95):.2%}")
    print(f"  99th percentile: {np.percentile(rel_errors, 99):.2%}")

    return targets_df


def analyze_geographic_errors(targets_df, geo_level="state"):
    """Analyze errors by geographic region."""
    print("\n" + "=" * 70)
    print(f"ERROR ANALYSIS BY {geo_level.upper()}")
    print("=" * 70)

    # Filter for geographic targets
    geo_targets = targets_df[targets_df["geographic_id"] != "US"]

    if geo_targets.empty:
        print("No geographic targets found")
        return

    geo_errors = (
        geo_targets.groupby("geographic_id")
        .agg({"rel_error": ["mean", "median", "max", "count"]})
        .round(4)
    )

    geo_errors = geo_errors.sort_values(("rel_error", "mean"), ascending=False)

    print(f"\nTop 10 {geo_level}s with highest mean relative error:")
    for geo_id in geo_errors.head(10).index:
        geo_data = geo_errors.loc[geo_id]
        n_targets = geo_data[("rel_error", "count")]
        mean_err = geo_data[("rel_error", "mean")]
        max_err = geo_data[("rel_error", "max")]
        median_err = geo_data[("rel_error", "median")]

        if geo_level == "congressional_district":
            state_fips = geo_id[:-2] if len(geo_id) > 2 else geo_id
            district = geo_id[-2:]
            label = f"CD {geo_id} (State {state_fips}, District {district})"
        else:
            label = f"State {geo_id}"

        print(
            f"{label}: Mean={mean_err:.1%}, Median={median_err:.1%}, Max={max_err:.1%} ({n_targets:.0f} targets)"
        )


def analyze_target_type_errors(targets_df):
    """Analyze errors by target type."""
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS BY TARGET TYPE")
    print("=" * 70)

    type_errors = (
        targets_df.groupby("stratum_group_id")
        .agg({"rel_error": ["mean", "median", "max", "count"]})
        .round(4)
    )

    type_errors = type_errors.sort_values(
        ("rel_error", "mean"), ascending=False
    )

    group_name_map = {
        2: "Age histogram",
        3: "AGI distribution",
        4: "SNAP",
        5: "Medicaid",
        6: "EITC",
    }

    print("\nError by target type (sorted by mean error):")
    for type_id in type_errors.index:
        type_data = type_errors.loc[type_id]
        n_targets = type_data[("rel_error", "count")]
        mean_err = type_data[("rel_error", "mean")]
        max_err = type_data[("rel_error", "max")]
        median_err = type_data[("rel_error", "median")]

        type_label = group_name_map.get(type_id, f"Type {type_id}")
        print(
            f"{type_label:30}: Mean={mean_err:.1%}, Median={median_err:.1%}, Max={max_err:.1%} ({n_targets:.0f} targets)"
        )


def analyze_worst_targets(targets_df, n=10):
    """Show worst performing individual targets."""
    print("\n" + "=" * 70)
    print(f"WORST PERFORMING TARGETS (Top {n})")
    print("=" * 70)

    worst_targets = targets_df.nlargest(n, "rel_error")
    for idx, row in worst_targets.iterrows():
        if row["geographic_id"] == "US":
            geo_label = "National"
        elif (
            "congressional_district" in targets_df.columns
            or len(row["geographic_id"]) > 2
        ):
            geo_label = f"CD {row['geographic_id']}"
        else:
            geo_label = f"State {row['geographic_id']}"

        print(
            f"\n{geo_label} - {row['variable']} (Group {row['stratum_group_id']})"
        )
        print(f"  Description: {row['description']}")
        print(
            f"  Target: {row['value']:,.0f}, Predicted: {row['y_pred']:,.0f}"
        )
        print(f"  Relative Error: {row['rel_error']:.1%}")


def analyze_weight_distribution(w, sim, geo_level="state"):
    """Analyze how weights are distributed across geographic regions."""
    print("\n" + "=" * 70)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 70)

    household_ids = sim.calculate("household_id", map_to="household").values
    n_households_total = len(household_ids)

    if geo_level == "state":
        geos = [
            "1",
            "2",
            "4",
            "5",
            "6",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
            "34",
            "35",
            "36",
            "37",
            "38",
            "39",
            "40",
            "41",
            "42",
            "44",
            "45",
            "46",
            "47",
            "48",
            "49",
            "50",
            "51",
            "53",
            "54",
            "55",
            "56",
        ]
    else:
        # For CDs, need to get list from weights length
        n_geos = len(w) // n_households_total
        print(f"Detected {n_geos} geographic units")
        return

    n_households_per_geo = n_households_total

    # Map weights to geographic regions
    weight_to_geo = {}
    for geo_idx, geo_id in enumerate(geos):
        start_idx = geo_idx * n_households_per_geo
        for hh_idx in range(n_households_per_geo):
            weight_idx = start_idx + hh_idx
            if weight_idx < len(w):
                weight_to_geo[weight_idx] = geo_id

    # Count active weights per geo
    active_weights_by_geo = {}
    for idx, weight_val in enumerate(w):
        if weight_val != 0:
            geo = weight_to_geo.get(idx, "unknown")
            if geo not in active_weights_by_geo:
                active_weights_by_geo[geo] = []
            active_weights_by_geo[geo].append(weight_val)

    # Calculate activation rates
    activation_rates = []
    for geo in geos:
        if geo in active_weights_by_geo:
            n_active = len(active_weights_by_geo[geo])
            rate = n_active / n_households_per_geo
            total_weight = sum(active_weights_by_geo[geo])
            activation_rates.append((geo, rate, n_active, total_weight))
        else:
            activation_rates.append((geo, 0, 0, 0))

    activation_rates.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 {geo_level}s by activation rate:")
    for geo, rate, n_active, total_weight in activation_rates[:5]:
        print(
            f"  {geo_level.title()} {geo}: {100*rate:.1f}% active ({n_active}/{n_households_per_geo}), Sum={total_weight:,.0f}"
        )

    print(f"\nBottom 5 {geo_level}s by activation rate:")
    for geo, rate, n_active, total_weight in activation_rates[-5:]:
        print(
            f"  {geo_level.title()} {geo}: {100*rate:.1f}% active ({n_active}/{n_households_per_geo}), Sum={total_weight:,.0f}"
        )


def export_calibration_log(targets_df, output_file, geo_level="state"):
    """Export results to calibration log CSV format."""
    print("\n" + "=" * 70)
    print("EXPORTING CALIBRATION LOG")
    print("=" * 70)

    log_rows = []
    for idx, row in targets_df.iterrows():
        # Create hierarchical target name
        if row["geographic_id"] == "US":
            target_name = f"nation/{row['variable']}/{row['description']}"
        elif geo_level == "congressional_district":
            target_name = f"CD{row['geographic_id']}/{row['variable']}/{row['description']}"
        else:
            target_name = f"US{row['geographic_id']}/{row['variable']}/{row['description']}"

        # Calculate metrics
        estimate = row["y_pred"]
        target = row["value"]
        error = estimate - target
        rel_error = error / target if target != 0 else 0

        log_rows.append(
            {
                "target_name": target_name,
                "estimate": estimate,
                "target": target,
                "epoch": 0,
                "error": error,
                "rel_error": rel_error,
                "abs_error": abs(error),
                "rel_abs_error": abs(rel_error),
                "loss": rel_error**2,
            }
        )

    calibration_log_df = pd.DataFrame(log_rows)
    calibration_log_df.to_csv(output_file, index=False)
    print(f"Saved calibration log to: {output_file}")
    print(f"Total rows: {len(calibration_log_df):,}")

    return calibration_log_df


def main():
    """Run weight diagnostics based on command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze calibration weights")
    parser.add_argument(
        "--geo",
        choices=["state", "congressional_district", "cd"],
        default="state",
        help="Geographic level (default: state)",
    )
    parser.add_argument(
        "--weight-file", type=str, help="Path to weight file (optional)"
    )
    parser.add_argument(
        "--export-csv", type=str, help="Export calibration log to CSV file"
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=10,
        help="Number of worst targets to show (default: 10)",
    )

    args = parser.parse_args()

    # Normalize geo level
    geo_level = "congressional_district" if args.geo == "cd" else args.geo

    print("\n" + "=" * 70)
    print(f"{geo_level.upper()} CALIBRATION WEIGHT DIAGNOSTICS")
    print("=" * 70)

    # Load data
    w, X_sparse, targets_df, sim = load_calibration_data(geo_level)

    # Override weight file if specified
    if args.weight_file:
        print(f"Loading weights from: {args.weight_file}")
        w = np.load(args.weight_file)

    # Basic weight statistics
    n_active = analyze_weight_statistics(w)

    if n_active == 0:
        print("\n❌ No active weights found! Check weight file.")
        sys.exit(1)

    # Analyze prediction errors
    targets_df = analyze_prediction_errors(w, X_sparse, targets_df)

    # Geographic error analysis
    analyze_geographic_errors(targets_df, geo_level)

    # Target type error analysis
    analyze_target_type_errors(targets_df)

    # Worst performing targets
    analyze_worst_targets(targets_df, args.worst_n)

    # Weight distribution analysis
    analyze_weight_distribution(w, sim, geo_level)

    # Export to CSV if requested
    if args.export_csv:
        export_calibration_log(targets_df, args.export_csv, geo_level)

    # Group-wise performance
    print("\n" + "=" * 70)
    print("GROUP-WISE PERFORMANCE")
    print("=" * 70)

    target_groups, group_info = create_target_groups(targets_df)
    rel_errors = targets_df["rel_error"].values

    group_means = []
    for group_id in np.unique(target_groups):
        group_mask = target_groups == group_id
        group_errors = rel_errors[group_mask]
        group_means.append(np.mean(group_errors))

    print(f"Mean of group means: {np.mean(group_means):.2%}")
    print(f"Max group mean: {np.max(group_means):.2%}")

    print("\n" + "=" * 70)
    print("WEIGHT DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
