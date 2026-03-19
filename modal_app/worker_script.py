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
        period=period,
        training_mask=area_training,
        variable_entity_map=variable_entity_map,
        constraints_map=constraints_map,
    )
    return results


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
        mask = (
            validation_targets["geo_level"] == geo_level
        ) & validation_targets["geographic_id"].astype(str).isin(nyc_cd_set)
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
    area_constraints = {
        int(s): constraints_map.get(int(s), []) for s in area_strata
    }

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

    from policyengine_us_data.utils.takeup import (
        SIMPLE_TAKEUP_VARS,
    )

    takeup_filter = [spec["variable"] for spec in SIMPLE_TAKEUP_VARS]

    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    from policyengine_us_data.calibration.publish_local_area import (
        build_h5,
        NYC_COUNTIES,
        NYC_CDS,
        AT_LARGE_DISTRICTS,
    )
    from policyengine_us_data.calibration.calibration_utils import (
        STATE_CODES,
    )
    from policyengine_us_data.calibration.clone_and_assign import (
        assign_random_geography,
    )
    from policyengine_us import Microsimulation

    weights = np.load(weights_path)

    sim = Microsimulation(dataset=str(dataset_path))
    n_records = sim.calculate("household_id", map_to="household").shape[0]
    del sim

    geography = assign_random_geography(
        n_records=n_records,
        n_clones=args.n_clones,
        seed=args.seed,
    )
    cds_to_calibrate = sorted(set(geography.cd_geoid.astype(str)))
    geo_labels = cds_to_calibrate
    print(
        f"Generated geography: "
        f"{geography.n_clones} clones x "
        f"{geography.n_records} records",
        file=sys.stderr,
    )

    # ── Validation setup (once per worker) ──
    validation_targets = None
    training_mask_full = None
    constraints_map = None
    if not args.no_validate:
        from sqlalchemy import create_engine
        from policyengine_us_data.calibration.validate_staging import (
            _query_all_active_targets,
            _batch_stratum_constraints,
            CSV_COLUMNS,
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
                validation_targets = validation_targets[inc_mask].reset_index(
                    drop=True
                )

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
                training_mask_full = np.ones(
                    len(validation_targets), dtype=bool
                )
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
                    cd
                    for cd in cds_to_calibrate
                    if int(cd) // 100 == state_fips
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
                )

            elif item_type == "city":
                cd_subset = [cd for cd in cds_to_calibrate if cd in NYC_CDS]
                if not cd_subset:
                    print(
                        "No NYC CDs found, skipping",
                        file=sys.stderr,
                    )
                    continue
                cities_dir = output_dir / "cities"
                cities_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=cities_dir / "NYC.h5",
                    cd_subset=cd_subset,
                    county_filter=NYC_COUNTIES,
                    takeup_filter=takeup_filter,
                )

            elif item_type == "national":
                national_dir = output_dir / "national"
                national_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=national_dir / "US.h5",
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
                            candidate=(
                                candidate if item_type == "district" else None
                            ),
                            cd_subset=(
                                cd_subset if item_type == "city" else None
                            ),
                            validation_targets=validation_targets,
                            training_mask_full=training_mask_full,
                            constraints_map=constraints_map,
                            db_path=str(db_path),
                            period=args.period,
                        )
                        results["validation_rows"].extend(v_rows)
                        key = f"{item_type}:{item_id}"
                        n_fail = sum(
                            1
                            for r in v_rows
                            if r.get("sanity_check") == "FAIL"
                        )
                        rae_vals = [
                            r["rel_abs_error"]
                            for r in v_rows
                            if isinstance(
                                r.get("rel_abs_error"),
                                (int, float),
                            )
                            and r["rel_abs_error"] != float("inf")
                        ]
                        mean_rae = (
                            sum(rae_vals) / len(rae_vals) if rae_vals else 0.0
                        )
                        results["validation_summary"][key] = {
                            "n_targets": len(v_rows),
                            "n_sanity_fail": n_fail,
                            "mean_rel_abs_error": round(mean_rae, 4),
                        }
                        print(
                            f"  Validated {key}: "
                            f"{len(v_rows)} targets, "
                            f"{n_fail} sanity fails, "
                            f"mean RAE={mean_rae:.4f}",
                            file=sys.stderr,
                        )
                    except Exception as ve:
                        print(
                            f"  Validation failed for "
                            f"{item_type}:{item_id}: {ve}",
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
