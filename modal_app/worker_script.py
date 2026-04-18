#!/usr/bin/env python
"""
Worker script for building local area H5 files.

Called by Modal workers via subprocess to avoid import conflicts.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


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
    request,
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

    geo_level = request.validation_geo_level
    geographic_ids = tuple(str(item) for item in request.validation_geographic_ids)
    if geo_level is None:
        return []
    area_type = {
        "state": "states",
        "district": "districts",
        "city": "cities",
        "national": "national",
    }.get(request.area_type)
    if area_type is None:
        return []
    display_id = request.display_name

    # Filter targets to matching area
    if request.area_type == "national":
        mask = validation_targets["geo_level"] == geo_level
    else:
        mask = (validation_targets["geo_level"] == geo_level) & (
            validation_targets["geographic_id"].astype(str).isin(geographic_ids)
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
                request.area_id,
                display_id,
                area_targets,
                area_training,
                area_constraints,
                db_path,
                period,
            ),
        )

    return results


def parse_args(argv: list[str] | None = None):
    """Parse worker arguments for legacy and typed request inputs."""

    parser = argparse.ArgumentParser()
    request_inputs = parser.add_mutually_exclusive_group(required=True)
    request_inputs.add_argument(
        "--work-items",
        help="JSON work items kept for backwards compatibility; new callers "
        "should use --requests-json",
    )
    request_inputs.add_argument(
        "--requests-json",
        help="JSON-serialized AreaBuildRequest payloads",
    )
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--geography-path",
        default=None,
        help="Optional explicit path to geography_assignment.npz",
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
    return parser.parse_args(argv)


def _load_request_inputs_from_args(
    *,
    args,
    area_build_request_cls,
):
    """Load either typed requests or raw legacy work items from CLI args."""

    if args.requests_json:
        request_payloads = json.loads(args.requests_json)
        return "requests", tuple(
            area_build_request_cls.from_dict(item) for item in request_payloads
        )

    return "work_items", tuple(json.loads(args.work_items))


def _build_kwargs_from_request(request) -> dict[str, Any]:
    """Translate a typed request into `build_h5(...)` keyword arguments."""

    if request.area_type == "national":
        return {}

    if len(request.filters) != 1:
        raise ValueError(
            f"{request.area_type} requests must carry exactly one build filter"
        )

    build_filter = request.filters[0]
    if (
        request.area_type in {"state", "district"}
        and build_filter.geography_field == "cd_geoid"
        and build_filter.op == "in"
    ):
        return {"cd_subset": [str(item) for item in build_filter.value]}

    if (
        request.area_type == "city"
        and build_filter.geography_field == "county_fips"
        and build_filter.op == "in"
    ):
        return {"county_fips_filter": {str(item) for item in build_filter.value}}

    raise ValueError(
        f"Unsupported build filter for {request.area_type}: "
        f"{build_filter.geography_field}:{build_filter.op}"
    )


def _request_key(request) -> str:
    """Return the stable completion key used by worker/coordinator flows."""

    return f"{request.area_type}:{request.area_id}"


def _work_item_key(work_item) -> str:
    """Return a stable key for legacy work items, even if malformed."""

    if not isinstance(work_item, dict):
        return "unknown:<invalid-work-item>"
    item_type = work_item.get("type", "<missing-type>")
    item_id = work_item.get("id", "<missing-id>")
    return f"{item_type}:{item_id}"


def _resolve_output_path(*, output_dir: Path, output_relative_path: str) -> Path:
    """Resolve one request output path and reject attempts to escape the run dir."""

    candidate_path = (output_dir / output_relative_path).resolve(strict=False)
    output_dir_path = output_dir.resolve(strict=False)
    try:
        candidate_path.relative_to(output_dir_path)
    except ValueError as exc:
        raise ValueError(
            "output_relative_path must stay within the worker output_dir"
        ) from exc
    return candidate_path


def _resolve_request_input(
    *,
    request_input_mode,
    request_input,
    area_catalog,
    geography,
):
    """Resolve one queued worker input into a typed request and stable key."""

    if request_input_mode == "requests":
        request = request_input
        return _request_key(request), request

    request = area_catalog.build_request_from_work_item(
        request_input,
        geography=geography,
    )
    if request is None:
        return _work_item_key(request_input), None
    return _request_key(request), request


def main(argv: list[str] | None = None):
    args = parse_args(argv)

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
        load_calibration_geography,
    )
    from policyengine_us_data.calibration.local_h5.area_catalog import USAreaCatalog
    from policyengine_us_data.calibration.local_h5.requests import AreaBuildRequest

    weights = np.load(weights_path)

    from policyengine_us import Microsimulation

    _sim = Microsimulation(dataset=str(dataset_path))
    n_records = len(_sim.calculate("household_id", map_to="household").values)
    del _sim

    geography = load_calibration_geography(
        weights_path=weights_path,
        n_records=n_records,
        n_clones=args.n_clones,
        geography_path=(
            Path(args.geography_path) if args.geography_path is not None else None
        ),
    )
    print(
        f"Loaded geography: "
        f"{geography.n_clones} clones x "
        f"{geography.n_records} records",
        file=sys.stderr,
    )
    area_catalog = USAreaCatalog.default()
    request_input_mode, request_inputs = _load_request_inputs_from_args(
        args=args,
        area_build_request_cls=AreaBuildRequest,
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
        "validation_rows": [],
        "validation_summary": {},
    }

    for request_input in request_inputs:
        try:
            request_key = (
                _work_item_key(request_input)
                if request_input_mode == "work_items"
                else None
            )
            request_key, request = _resolve_request_input(
                request_input_mode=request_input_mode,
                request_input=request_input,
                area_catalog=area_catalog,
                geography=geography,
            )
            if request is None:
                print(
                    f"Skipping {request_key}: no matching geography in legacy work item",
                    file=sys.stderr,
                )
                continue

            output_path = _resolve_output_path(
                output_dir=output_dir,
                output_relative_path=request.output_relative_path,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            build_kwargs = _build_kwargs_from_request(request)
            if request.area_type == "national":
                n_clones_from_weights = weights.shape[0] // n_records
                if n_clones_from_weights != geography.n_clones:
                    raise ValueError(
                        f"National weights have {n_clones_from_weights} clones "
                        f"but geography has {geography.n_clones}. "
                        "Use the matching saved geography artifact."
                    )
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=output_path,
                )
            else:
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=output_path,
                    takeup_filter=takeup_filter,
                    **build_kwargs,
                )

            if path:
                results["completed"].append(request_key)
                print(
                    f"Completed {request_key}",
                    file=sys.stderr,
                )

                # ── Per-item validation ──
                if not args.no_validate and validation_targets is not None:
                    try:
                        v_rows = _validate_h5_subprocess(
                            h5_path=str(path),
                            request=request,
                            validation_targets=validation_targets,
                            training_mask_full=training_mask_full,
                            constraints_map=constraints_map,
                            db_path=str(db_path),
                            period=args.period,
                        )
                        results["validation_rows"].extend(v_rows)
                        n_fail = sum(
                            1 for r in v_rows if r.get("sanity_check") == "FAIL"
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
                        mean_rae = sum(rae_vals) / len(rae_vals) if rae_vals else 0.0
                        results["validation_summary"][request_key] = {
                            "n_targets": len(v_rows),
                            "n_sanity_fail": n_fail,
                            "mean_rel_abs_error": round(mean_rae, 4),
                        }
                        print(
                            f"  Validated {request_key}: "
                            f"{len(v_rows)} targets, "
                            f"{n_fail} sanity fails, "
                            f"mean RAE={mean_rae:.4f}",
                            file=sys.stderr,
                        )
                    except Exception as ve:
                        print(
                            f"  Validation failed for {request_key}: {ve}",
                            file=sys.stderr,
                        )

        except Exception as e:
            results["failed"].append(request_key)
            results["errors"].append(
                {
                    "item": request_key,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"FAILED {request_key}: {e}",
                file=sys.stderr,
            )

    sys.stdout = original_stdout
    print(json.dumps(results))


if __name__ == "__main__":
    main()
