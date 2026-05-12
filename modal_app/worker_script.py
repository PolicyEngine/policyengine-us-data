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
        "--scope",
        choices=("regional", "national"),
        required=True,
        help="Worker bootstrap scope to use for this request batch",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Pipeline run ID used for traceability and bootstrap lookup",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional run-scoped pipeline artifacts directory containing bootstrap artifacts",
    )
    parser.add_argument(
        "--run-config-path",
        default=None,
        help="Optional unified run configuration JSON used for traceability",
    )
    parser.add_argument(
        "--scope-fingerprint",
        default=None,
        help="Coordinator-resolved scope fingerprint expected by bootstrap artifacts",
    )
    parser.add_argument(
        "--geography-path",
        default=None,
        help="Optional explicit path to geography_assignment.npz",
    )
    parser.add_argument(
        "--calibration-package-path",
        default=None,
        help="Optional calibration_package.pkl used as a geography fallback",
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


def _build_publishing_inputs(*, args, run_id: str):
    """Build the traceability input bundle consumed by worker setup services."""

    from policyengine_us_data.build_outputs.worker_inputs import (
        WorkerCalibrationInputs,
    )

    worker_inputs = WorkerCalibrationInputs(
        weights_path=Path(args.weights_path),
        dataset_path=Path(args.dataset_path),
        database_path=Path(args.db_path),
        geography_path=(
            Path(args.geography_path) if args.geography_path is not None else None
        ),
        calibration_package_path=(
            Path(args.calibration_package_path)
            if args.calibration_package_path is not None
            else None
        ),
        run_config_path=(
            Path(args.run_config_path) if args.run_config_path is not None else None
        ),
        n_clones=args.n_clones,
        seed=args.seed,
    )
    return worker_inputs.to_publishing_input_bundle(run_id=run_id)


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


def _log_worker_session_ready(*, scope: str, session, geography) -> None:
    """Write worker-session setup details to stderr for Modal diagnostics."""

    print(
        "Worker session ready: "
        f"scope={scope}, bootstrap={session.bootstrap_status}, "
        f"{geography.n_clones} clones x {geography.n_records} records",
        file=sys.stderr,
    )
    bootstrap_error = session.caches.get("bootstrap_error")
    if bootstrap_error:
        print(
            f"Worker bootstrap fallback reason: {bootstrap_error}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or output_dir.name or "local-worker"

    from policyengine_us_data.utils.takeup import (
        SIMPLE_TAKEUP_VARS,
    )

    takeup_filter = [spec["variable"] for spec in SIMPLE_TAKEUP_VARS]

    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    from policyengine_us_data.calibration.publish_local_area import (
        build_h5,
    )
    from policyengine_us_data.build_outputs.area_catalog import USAreaCatalog
    from policyengine_us_data.build_outputs.requests import AreaBuildRequest
    from policyengine_us_data.build_outputs.validation import (
        AreaValidationService,
        ValidationPolicy,
    )
    from policyengine_us_data.build_outputs.worker_session import WorkerSessionFactory

    area_catalog = USAreaCatalog.default()
    request_input_mode, request_inputs = _load_request_inputs_from_args(
        args=args,
        area_build_request_cls=AreaBuildRequest,
    )
    scope = args.scope
    inputs = _build_publishing_inputs(args=args, run_id=run_id)
    validation_service = AreaValidationService()

    session = WorkerSessionFactory(validation_service=validation_service).create(
        inputs=inputs,
        scope=scope,
        validation_policy=ValidationPolicy(enabled=not args.no_validate),
        period=args.period,
        target_config_path=Path(args.target_config) if args.target_config else None,
        validation_config_path=(
            Path(args.validation_config) if args.validation_config else None
        ),
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
        expected_scope_fingerprint=args.scope_fingerprint,
    )
    weights = session.weights.values
    n_records = session.weights.n_records
    geography = session.geography
    validation_context = session.validation_context
    _log_worker_session_ready(scope=scope, session=session, geography=geography)
    if (
        validation_context is not None
        and validation_context.validation_targets is not None
    ):
        print(
            f"Validation ready: {len(validation_context.validation_targets)} targets, "
            f"{len(validation_context.constraints_map or {})} strata",
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

                if not args.no_validate and validation_context is not None:
                    try:
                        validation_result = validation_service.validate_request(
                            context=validation_context,
                            h5_path=str(path),
                            request=request,
                        )
                        v_rows = list(validation_result.rows)
                        results["validation_rows"].extend(v_rows)
                        summary = dict(validation_result.summary)
                        results["validation_summary"][request_key] = summary
                        print(
                            f"  Validated {request_key}: "
                            f"{summary['n_targets']} targets, "
                            f"{summary['n_sanity_fail']} sanity fails, "
                            f"mean RAE={summary['mean_rel_abs_error']:.4f}",
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
