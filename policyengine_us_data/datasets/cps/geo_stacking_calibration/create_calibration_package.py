#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from datetime import datetime
import pickle
import json
from sqlalchemy import create_engine, text
import logging

import numpy as np
from scipy import sparse as sp

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking_sparse import (
    SparseGeoStackingMatrixBuilder,
)
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import (
    create_target_groups,
    filter_target_groups,
)
from policyengine_us_data.utils.data_upload import upload_files_to_gcs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_calibration_package(
    db_path: str,
    dataset_uri: str,
    mode: str = "Stratified",
    groups_to_exclude: list = None,
    local_output_dir: str = None,
    gcs_bucket: str = None,
    gcs_date_prefix: str = None,
):
    """
    Create a calibration package from database and dataset.

    Args:
        db_path: Path to policy_data.db
        dataset_uri: URI for the CPS dataset (local path or hf://)
        mode: "Test", "Stratified", or "Full"
        groups_to_exclude: List of target group IDs to exclude
        local_output_dir: Local directory to save package (optional)
        gcs_bucket: GCS bucket name (optional)
        gcs_date_prefix: Date prefix for GCS (e.g., "2025-10-15-1430", auto-generated if None)

    Returns:
        dict with 'local_path' and/or 'gcs_path' keys
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if groups_to_exclude is None:
        groups_to_exclude = []

    # Step 1: Load data and get CD list
    db_uri = f"sqlite:///{db_path}"
    builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)

    engine = create_engine(db_uri)
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM strata s
    JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
    WHERE s.stratum_group_id = 1
      AND sc.constraint_variable = 'congressional_district_geoid'
    ORDER BY sc.value
    """

    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        all_cd_geoids = [row[0] for row in result]

    logging.info(
        f"Found {len(all_cd_geoids)} congressional districts in database"
    )

    # Select CDs based on mode
    if mode == "Test":
        cds_to_calibrate = [
            "601",
            "652",
            "3601",
            "3626",
            "4801",
            "4838",
            "1201",
            "1228",
            "1701",
            "1101",
        ]
        logging.info(f"TEST MODE: Using {len(cds_to_calibrate)} CDs")
    else:
        cds_to_calibrate = all_cd_geoids
        logging.info(f"Using all {len(cds_to_calibrate)} CDs")

    sim = Microsimulation(dataset=dataset_uri)

    # Step 2: Build sparse matrix
    logging.info("Building sparse matrix...")
    targets_df, X_sparse, household_id_mapping = (
        builder.build_stacked_matrix_sparse(
            "congressional_district", cds_to_calibrate, sim
        )
    )
    logging.info(f"Matrix shape: {X_sparse.shape}")
    logging.info(f"Total targets: {len(targets_df)}")

    # Step 3: Create and filter target groups
    target_groups, group_info = create_target_groups(targets_df)

    logging.info(f"Total groups: {len(np.unique(target_groups))}")
    for info in group_info[:5]:
        logging.info(f"  {info}")

    if groups_to_exclude:
        logging.info(f"Excluding {len(groups_to_exclude)} target groups")
        targets_df, X_sparse, target_groups = filter_target_groups(
            targets_df, X_sparse, target_groups, groups_to_exclude
        )

    targets = targets_df.value.values

    # Step 4: Calculate initial weights
    cd_populations = {}
    for cd_geoid in cds_to_calibrate:
        cd_age_targets = targets_df[
            (targets_df["geographic_id"] == cd_geoid)
            & (targets_df["variable"] == "person_count")
            & (targets_df["variable_desc"].str.contains("age", na=False))
        ]
        if not cd_age_targets.empty:
            unique_ages = cd_age_targets.drop_duplicates(
                subset=["variable_desc"]
            )
            cd_populations[cd_geoid] = unique_ages["value"].sum()

    if cd_populations:
        min_pop = min(cd_populations.values())
        max_pop = max(cd_populations.values())
        logging.info(f"CD population range: {min_pop:,.0f} to {max_pop:,.0f}")
    else:
        logging.warning("Could not calculate CD populations, using default")
        min_pop = 700000

    keep_probs = np.zeros(X_sparse.shape[1])
    init_weights = np.zeros(X_sparse.shape[1])
    cumulative_idx = 0
    cd_household_indices = {}

    for cd_key, household_list in household_id_mapping.items():
        cd_geoid = cd_key.replace("cd", "")
        n_households = len(household_list)

        if cd_geoid in cd_populations:
            cd_pop = cd_populations[cd_geoid]
        else:
            cd_pop = min_pop

        pop_ratio = cd_pop / min_pop
        adjusted_keep_prob = min(0.15, 0.02 * np.sqrt(pop_ratio))
        keep_probs[cumulative_idx : cumulative_idx + n_households] = (
            adjusted_keep_prob
        )

        base_weight = cd_pop / n_households
        sparsity_adjustment = 1.0 / np.sqrt(adjusted_keep_prob)
        initial_weight = base_weight * sparsity_adjustment

        init_weights[cumulative_idx : cumulative_idx + n_households] = (
            initial_weight
        )
        cd_household_indices[cd_geoid] = (
            cumulative_idx,
            cumulative_idx + n_households,
        )
        cumulative_idx += n_households

    logging.info(
        f"Initial weight range: {init_weights.min():.0f} to {init_weights.max():.0f}"
    )
    logging.info(f"Mean initial weight: {init_weights.mean():.0f}")

    # Step 5: Create calibration package
    calibration_package = {
        "X_sparse": X_sparse,
        "targets_df": targets_df,
        "household_id_mapping": household_id_mapping,
        "cd_household_indices": cd_household_indices,
        "dataset_uri": dataset_uri,
        "cds_to_calibrate": cds_to_calibrate,
        "initial_weights": init_weights,
        "keep_probs": keep_probs,
        "target_groups": target_groups,
    }

    # Create metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "mode": mode,
        "dataset_uri": dataset_uri,
        "n_cds": len(cds_to_calibrate),
        "n_targets": len(targets_df),
        "n_households": X_sparse.shape[1],
        "matrix_shape": X_sparse.shape,
        "groups_excluded": groups_to_exclude,
    }

    results = {}

    # Save locally if requested
    if local_output_dir:
        local_dir = Path(local_output_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        pkg_path = local_dir / "calibration_package.pkl"
        with open(pkg_path, "wb") as f:
            pickle.dump(calibration_package, f)

        meta_path = local_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"✅ Saved locally to {pkg_path}")
        logging.info(
            f"   Size: {pkg_path.stat().st_size / 1024 / 1024:.1f} MB"
        )
        results["local_path"] = str(pkg_path)

    # Upload to GCS if requested
    if gcs_bucket:
        if not gcs_date_prefix:
            gcs_date_prefix = datetime.now().strftime("%Y-%m-%d-%H%M")

        gcs_prefix = f"{gcs_date_prefix}/inputs"

        # Save to temp location for upload
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pkg = Path(tmpdir) / "calibration_package.pkl"
            tmp_meta = Path(tmpdir) / "metadata.json"

            with open(tmp_pkg, "wb") as f:
                pickle.dump(calibration_package, f)
            with open(tmp_meta, "w") as f:
                json.dump(metadata, f, indent=2)

            # Upload to GCS with prefix
            from google.cloud import storage
            import google.auth

            credentials, project_id = google.auth.default()
            storage_client = storage.Client(
                credentials=credentials, project=project_id
            )
            bucket = storage_client.bucket(gcs_bucket)

            for local_file, blob_name in [
                (tmp_pkg, "calibration_package.pkl"),
                (tmp_meta, "metadata.json"),
            ]:
                blob_path = f"{gcs_prefix}/{blob_name}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_file)
                logging.info(f"✅ Uploaded to gs://{gcs_bucket}/{blob_path}")

        gcs_path = f"gs://{gcs_bucket}/{gcs_prefix}"
        results["gcs_path"] = gcs_path
        results["gcs_prefix"] = gcs_prefix

    return results


def main():
    parser = argparse.ArgumentParser(description="Create calibration package")
    parser.add_argument(
        "--db-path", required=True, help="Path to policy_data.db"
    )
    parser.add_argument(
        "--dataset-uri",
        required=True,
        help="Dataset URI (local path or hf://)",
    )
    parser.add_argument(
        "--mode", default="Stratified", choices=["Test", "Stratified", "Full"]
    )
    parser.add_argument("--local-output", help="Local output directory")
    parser.add_argument(
        "--gcs-bucket", help="GCS bucket name (e.g., policyengine-calibration)"
    )
    parser.add_argument(
        "--gcs-date", help="GCS date prefix (default: YYYY-MM-DD-HHMM)"
    )

    args = parser.parse_args()

    # Default groups to exclude (from original script)
    groups_to_exclude = [
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        12,
        10,
        15,
        17,
        18,
        21,
        34,
        35,
        36,
        37,
        31,
        56,
        42,
        64,
        46,
        68,
        47,
        69,
    ]

    results = create_calibration_package(
        db_path=args.db_path,
        dataset_uri=args.dataset_uri,
        mode=args.mode,
        groups_to_exclude=groups_to_exclude,
        local_output_dir=args.local_output,
        gcs_bucket=args.gcs_bucket,
        gcs_date_prefix=args.gcs_date,
    )

    print("\n" + "=" * 70)
    print("CALIBRATION PACKAGE CREATED")
    print("=" * 70)
    if "local_path" in results:
        print(f"Local: {results['local_path']}")
    if "gcs_path" in results:
        print(f"GCS: {results['gcs_path']}")
        print(f"\nTo use with optimize_weights.py:")
        print(f"  --gcs-input gs://{args.gcs_bucket}/{results['gcs_prefix']}")


if __name__ == "__main__":
    main()
