#!/usr/bin/env python
"""
Worker script for building local area H5 files.

Called by Modal workers via subprocess to avoid import conflicts.
"""

import argparse
import json
import sys
import traceback
import numpy as np
from pathlib import Path


def _validate_in_subprocess(
    h5_path,
    area_type,
    area_id,
    display_id,
    area_targets,
    area_training,
    constraints_map,
    db_path,
    period,
):
    """Run validation for one area inside a subprocess.

    All Microsimulation memory is reclaimed when the
    subprocess exits.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from policyengine_us import Microsimulation
    from sqlalchemy import create_engine as _ce
    from policyengine_us_data.calibration.validate_staging import (
        validate_area,
        _build_variable_entity_map,
    )

    engine = _ce(f"sqlite:///{db_path}")
    sim = Microsimulation(dataset=h5_path)
    variable_entity_map = _build_variable_entity_map(sim)

    results = validate_area(
        sim=sim,
        targets_df=area_targets,
        engine=engine,
        area_type=area_type,
        area_id=area_id,
        display_id=display_id,
        dataset_path=h5_path,
        period=period,
        training_mask=area_training,
        variable_entity_map=variable_entity_map,
        constraints_map=constraints_map,
    )
    return results


def _record_validation_success(results, item_key, validation_rows):
    """Record validation rows and derived summary for one built H5."""
    from policyengine_us_data.calibration.local_h5.validation import (
        summarize_validation_rows,
    )

    results["validation_rows"].extend(validation_rows)
    results["validation_summary"][item_key] = summarize_validation_rows(validation_rows)


def _record_validation_error(results, item_key, error):
    """Record a structured validation error without converting it into a build failure."""
    from policyengine_us_data.calibration.local_h5.validation import (
        make_validation_error,
    )

    error_entry = make_validation_error(
        item_key=item_key,
        error=error,
        traceback_text=traceback.format_exc(),
    )
    results["validation_errors"].append(error_entry)
    return error_entry


def _validate_h5_subprocess(
    h5_path,
    item_type,
    item_id,
    state_fips,
    candidate,
    cd_subset,
    validation_targets,
    training_mask_full,
    constraints_map,
    db_path,
    period,
):
    """Spawn a subprocess to validate one H5 file.

    Uses multiprocessing spawn to isolate memory.
    """
    import multiprocessing as _mp

    # Determine geo_level and geographic_id for filtering targets
    if item_type == "state":
        geo_level = "state"
        geographic_id = str(state_fips)
        area_type = "states"
        display_id = item_id
    elif item_type == "district":
        geo_level = "district"
        geographic_id = str(candidate)
        area_type = "districts"
        display_id = item_id
    elif item_type == "city":
        # NYC: aggregate targets for NYC CDs
        geo_level = "district"
        area_type = "cities"
        display_id = item_id
    elif item_type == "national":
        geo_level = "national"
        geographic_id = "US"
        area_type = "national"
        display_id = "US"
    else:
        return []

    # Filter targets to matching area
    if item_type == "city":
        # Match targets for any NYC CD
        nyc_cd_set = set(str(cd) for cd in cd_subset)
        mask = (validation_targets["geo_level"] == geo_level) & validation_targets[
            "geographic_id"
        ].astype(str).isin(nyc_cd_set)
    elif item_type == "national":
        mask = validation_targets["geo_level"] == geo_level
    else:
        mask = (validation_targets["geo_level"] == geo_level) & (
            validation_targets["geographic_id"].astype(str) == geographic_id
        )

    area_targets = validation_targets[mask].reset_index(drop=True)
    area_training = training_mask_full[mask.values]

    if len(area_targets) == 0:
        return []

    # Filter constraints_map to relevant strata
    area_strata = area_targets["stratum_id"].unique().tolist()
    area_constraints = {int(s): constraints_map.get(int(s), []) for s in area_strata}

    ctx = _mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        results = pool.apply(
            _validate_in_subprocess,
            (
                h5_path,
                area_type,
                item_id,
                display_id,
                area_targets,
                area_training,
                area_constraints,
                db_path,
                period,
            ),
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-items", required=True, help="JSON work items")
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--calibration-package-path",
        default=None,
        help="Optional calibration package path for exact geography reuse",
    )
    parser.add_argument(
        "--n-clones",
        type=int,
        default=430,
        help="Number of clones used in calibration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in calibration",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip per-item validation after each H5 build",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=2024,
        help="Tax year for validation targets",
    )
    parser.add_argument(
        "--target-config",
        default=None,
        help="Path to training target_config.yaml",
    )
    parser.add_argument(
        "--validation-config",
        default=None,
        help="Path to target_config_full.yaml for validation",
    )
    args = parser.parse_args()

    work_items = json.loads(args.work_items)
    weights_path = Path(args.weights_path)
    dataset_path = Path(args.dataset_path)
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)
    calibration_package_path = (
        Path(args.calibration_package_path)
        if args.calibration_package_path
        else None
    )
    if calibration_package_path is not None:
        from policyengine_us_data.calibration.local_h5.package_geography import (
            require_calibration_package_path,
        )

        calibration_package_path = require_calibration_package_path(
            calibration_package_path
        )

    from policyengine_us_data.utils.takeup import (
        SIMPLE_TAKEUP_VARS,
    )

    takeup_filter = [spec["variable"] for spec in SIMPLE_TAKEUP_VARS]

    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    from policyengine_us_data.calibration.publish_local_area import (
        build_h5,
        NYC_COUNTY_FIPS,
        AT_LARGE_DISTRICTS,
        SUB_ENTITIES,
    )
    from policyengine_us_data.calibration.calibration_utils import (
        STATE_CODES,
    )
    from policyengine_us_data.calibration.local_h5.package_geography import (
        CalibrationPackageGeographyLoader,
    )
    from policyengine_us_data.calibration.local_h5.source_dataset import (
        PolicyEngineDatasetReader,
    )

    weights = np.load(weights_path)
    source_snapshot = PolicyEngineDatasetReader(tuple(SUB_ENTITIES)).load(dataset_path)
    n_records = source_snapshot.n_households

    if weights.shape[0] % n_records != 0:
        raise ValueError(
            f"Weight vector length {weights.shape[0]} is not divisible by n_records={n_records}"
        )
    n_clones_from_weights = weights.shape[0] // n_records
    if n_clones_from_weights != args.n_clones:
        print(
            f"WARNING: weights imply {n_clones_from_weights} clones "
            f"but --n-clones={args.n_clones}; using weights-derived value",
            file=sys.stderr,
        )

    geography_loader = CalibrationPackageGeographyLoader()
    geography_resolution = geography_loader.resolve_for_weights(
        package_path=calibration_package_path,
        weights_length=weights.shape[0],
        n_records=n_records,
        n_clones=n_clones_from_weights,
        seed=args.seed,
    )
    geography = geography_resolution.geography
    cds_to_calibrate = sorted(set(geography.cd_geoid.astype(str)))
    geo_labels = cds_to_calibrate
    print(
        f"Loaded geography from {geography_resolution.source}: "
        f"{geography.n_clones} clones x "
        f"{geography.n_records} records",
        file=sys.stderr,
    )
    print(
        f"Loaded source snapshot once for worker: "
        f"{source_snapshot.n_households} households",
        file=sys.stderr,
    )
    for warning in geography_resolution.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    # ── Validation setup (once per worker) ──
    validation_targets = None
    training_mask_full = None
    constraints_map = None
    if not args.no_validate:
        from sqlalchemy import create_engine
        from policyengine_us_data.calibration.validate_staging import (
            _query_all_active_targets,
            _batch_stratum_constraints,
        )
        from policyengine_us_data.calibration.unified_calibration import (
            load_target_config,
            _match_rules,
        )

        engine = create_engine(f"sqlite:///{db_path}")
        validation_targets = _query_all_active_targets(engine, args.period)
        print(
            f"Loaded {len(validation_targets)} validation targets",
            file=sys.stderr,
        )

        # Apply exclude/include from validation config
        if args.validation_config:
            val_cfg = load_target_config(args.validation_config)
            exc_rules = val_cfg.get("exclude", [])
            if exc_rules:
                exc_mask = _match_rules(validation_targets, exc_rules)
                validation_targets = validation_targets[~exc_mask].reset_index(
                    drop=True
                )
            inc_rules = val_cfg.get("include", [])
            if inc_rules:
                inc_mask = _match_rules(validation_targets, inc_rules)
                validation_targets = validation_targets[inc_mask].reset_index(drop=True)

        # Compute training mask from training config
        if args.target_config:
            tr_cfg = load_target_config(args.target_config)
            tr_inc = tr_cfg.get("include", [])
            if tr_inc:
                training_mask_full = np.asarray(
                    _match_rules(validation_targets, tr_inc),
                    dtype=bool,
                )
            else:
                training_mask_full = np.ones(len(validation_targets), dtype=bool)
        else:
            training_mask_full = np.ones(len(validation_targets), dtype=bool)

        # Batch-load constraints
        stratum_ids = validation_targets["stratum_id"].unique().tolist()
        constraints_map = _batch_stratum_constraints(engine, stratum_ids)
        print(
            f"Validation ready: {len(validation_targets)} targets, "
            f"{len(stratum_ids)} strata",
            file=sys.stderr,
        )

    results = {
        "completed": [],
        "failed": [],
        "errors": [],
        "validation_errors": [],
        "validation_rows": [],
        "validation_summary": {},
    }

    for item in work_items:
        item_type = item["type"]
        item_id = item["id"]
        state_fips = None
        candidate = None
        cd_subset = None

        try:
            if item_type == "state":
                state_fips = None
                for fips, code in STATE_CODES.items():
                    if code == item_id:
                        state_fips = fips
                        break
                if state_fips is None:
                    raise ValueError(f"Unknown state code: {item_id}")
                cd_subset = [
                    cd for cd in cds_to_calibrate if int(cd) // 100 == state_fips
                ]
                if not cd_subset:
                    print(
                        f"No CDs for {item_id}, skipping",
                        file=sys.stderr,
                    )
                    continue
                states_dir = output_dir / "states"
                states_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=states_dir / f"{item_id}.h5",
                    cd_subset=cd_subset,
                    takeup_filter=takeup_filter,
                    source_snapshot=source_snapshot,
                )

            elif item_type == "district":
                state_code, dist_num = item_id.split("-")
                state_fips = None
                for fips, code in STATE_CODES.items():
                    if code == state_code:
                        state_fips = fips
                        break
                if state_fips is None:
                    raise ValueError(f"Unknown state in district: {item_id}")

                candidate = f"{state_fips}{int(dist_num):02d}"
                if candidate in geo_labels:
                    geoid = candidate
                else:
                    state_cds = [
                        cd for cd in geo_labels if int(cd) // 100 == state_fips
                    ]
                    if len(state_cds) == 1:
                        geoid = state_cds[0]
                    else:
                        raise ValueError(
                            f"CD {candidate} not found and "
                            f"state {state_code} has "
                            f"{len(state_cds)} CDs"
                        )

                cd_int = int(geoid)
                district_num = cd_int % 100
                if district_num in AT_LARGE_DISTRICTS:
                    district_num = 1
                friendly_name = f"{state_code}-{district_num:02d}"

                districts_dir = output_dir / "districts"
                districts_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=districts_dir / f"{friendly_name}.h5",
                    cd_subset=[geoid],
                    takeup_filter=takeup_filter,
                    source_snapshot=source_snapshot,
                )

            elif item_type == "city":
                cities_dir = output_dir / "cities"
                cities_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=cities_dir / "NYC.h5",
                    county_fips_filter=NYC_COUNTY_FIPS,
                    takeup_filter=takeup_filter,
                    source_snapshot=source_snapshot,
                )

            elif item_type == "national":
                national_dir = output_dir / "national"
                national_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=national_dir / "US.h5",
                    source_snapshot=source_snapshot,
                )
            else:
                raise ValueError(f"Unknown item type: {item_type}")

            if path:
                results["completed"].append(f"{item_type}:{item_id}")
                print(
                    f"Completed {item_type}:{item_id}",
                    file=sys.stderr,
                )

                # ── Per-item validation ──
                if not args.no_validate and validation_targets is not None:
                    key = f"{item_type}:{item_id}"
                    try:
                        v_rows = _validate_h5_subprocess(
                            h5_path=str(path),
                            item_type=item_type,
                            item_id=item_id,
                            state_fips=(
                                state_fips
                                if item_type in ("state", "district")
                                else None
                            ),
                            candidate=(candidate if item_type == "district" else None),
                            cd_subset=(cd_subset if item_type == "city" else None),
                            validation_targets=validation_targets,
                            training_mask_full=training_mask_full,
                            constraints_map=constraints_map,
                            db_path=str(db_path),
                            period=args.period,
                        )
                        _record_validation_success(results, key, v_rows)
                        summary = results["validation_summary"][key]
                        print(
                            f"  Validated {key}: "
                            f"{summary['n_targets']} targets, "
                            f"{summary['n_sanity_fail']} sanity fails, "
                            f"mean RAE={summary['mean_rel_abs_error']:.4f}",
                            file=sys.stderr,
                        )
                    except Exception as ve:
                        error_entry = _record_validation_error(results, key, ve)
                        print(
                            f"  Validation failed for {item_type}:{item_id}: "
                            f"{error_entry['error']}",
                            file=sys.stderr,
                        )

        except Exception as e:
            results["failed"].append(f"{item_type}:{item_id}")
            results["errors"].append(
                {
                    "item": f"{item_type}:{item_id}",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"FAILED {item_type}:{item_id}: {e}",
                file=sys.stderr,
            )

    sys.stdout = original_stdout
    print(json.dumps(results))


if __name__ == "__main__":
    main()
