#!/usr/bin/env python
"""
Thin CLI adapter for the local H5 worker service.

Modal launches this script in a subprocess so the worker runtime stays isolated
from the coordinator process. The actual chunk-level build behavior now lives
in `policyengine_us_data.calibration.local_h5.worker_service`.
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    # Kept for backward compatibility with older worker launchers.
    # New callers should pass fully resolved AreaBuildRequests via
    # --requests-json instead.
    parser.add_argument("--work-items", default=None, help="JSON work items")
    parser.add_argument(
        "--requests-json",
        default=None,
        help="JSON serialized AreaBuildRequest list",
    )
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

    if not args.requests_json and not args.work_items:
        raise ValueError("Either --requests-json or --work-items is required")

    work_items = json.loads(args.work_items) if args.work_items else None
    request_payloads = json.loads(args.requests_json) if args.requests_json else None
    weights_path = Path(args.weights_path)
    dataset_path = Path(args.dataset_path)
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)
    calibration_package_path = (
        Path(args.calibration_package_path)
        if args.calibration_package_path
        else None
    )

    from policyengine_us_data.utils.takeup import SIMPLE_TAKEUP_VARS
    from policyengine_us_data.calibration.local_h5.contracts import (
        AreaBuildRequest,
        ValidationPolicy,
    )
    from policyengine_us_data.calibration.local_h5.package_geography import (
        require_calibration_package_path,
    )
    from policyengine_us_data.calibration.local_h5.worker_service import (
        LocalH5WorkerService,
        WorkerSession,
        build_requests_from_work_items,
        load_validation_context,
    )
    from policyengine_us_data.calibration.publish_local_area import (
        AT_LARGE_DISTRICTS,
        NYC_COUNTY_FIPS,
        SUB_ENTITIES,
    )
    from policyengine_us_data.calibration.calibration_utils import STATE_CODES
    from policyengine_us_data.calibration.local_h5.source_dataset import (
        PolicyEngineDatasetReader,
    )

    takeup_filter = tuple(spec["variable"] for spec in SIMPLE_TAKEUP_VARS)

    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    if calibration_package_path is not None:
        calibration_package_path = require_calibration_package_path(
            calibration_package_path
        )

    validation_policy = ValidationPolicy(enabled=not args.no_validate)
    validation_context = load_validation_context(
        db_path=db_path,
        period=args.period,
        target_config_path=args.target_config,
        validation_config_path=args.validation_config,
        policy=validation_policy,
    )
    if validation_context is not None:
        print(
            f"Validation ready: {len(validation_context.validation_targets)} targets, "
            f"{len(validation_context.constraints_map)} strata",
            file=sys.stderr,
        )

    session = WorkerSession.load(
        weights_path=weights_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        calibration_package_path=calibration_package_path,
        requested_n_clones=args.n_clones,
        seed=args.seed,
        takeup_filter=takeup_filter,
        validation_policy=validation_policy,
        validation_context=validation_context,
        source_reader=PolicyEngineDatasetReader(tuple(SUB_ENTITIES)),
        allow_seed_fallback=False,
    )
    if session.requested_n_clones is not None and session.requested_n_clones != session.n_clones:
        print(
            f"WARNING: weights imply {session.n_clones} clones "
            f"but --n-clones={session.requested_n_clones}; using weights-derived value",
            file=sys.stderr,
        )
    print(
        f"Loaded geography from {session.geography_source}: "
        f"{session.geography.n_clones} clones x {session.geography.n_records} records",
        file=sys.stderr,
    )
    print(
        f"Loaded source snapshot once for worker: "
        f"{session.source_snapshot.n_households} households",
        file=sys.stderr,
    )
    for warning in session.geography_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if request_payloads is not None:
        requests = tuple(
            AreaBuildRequest.from_dict(payload) for payload in request_payloads
        )
        initial_failures = ()
    else:
        requests, initial_failures = build_requests_from_work_items(
            work_items,
            geography=session.geography,
            state_codes=STATE_CODES,
            at_large_districts=AT_LARGE_DISTRICTS,
            nyc_county_fips=NYC_COUNTY_FIPS,
        )
    service = LocalH5WorkerService()
    worker_result = service.run(
        session,
        requests,
        initial_failures=initial_failures,
    )

    sys.stdout = original_stdout
    print(json.dumps(worker_result.to_dict()))


if __name__ == "__main__":
    main()
