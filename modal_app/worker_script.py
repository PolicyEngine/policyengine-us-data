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


def _resolve_worker_requests(
    *,
    request_input_mode,
    request_inputs,
    area_catalog,
    geography,
) -> tuple[tuple, tuple]:
    """Resolve queued CLI inputs into typed requests plus conversion issues."""

    from policyengine_us_data.build_outputs.worker_service import WorkerIssue

    if request_input_mode == "requests":
        return tuple(request_inputs), ()

    requests = []
    issues = []
    for request_input in request_inputs:
        request_key = _work_item_key(request_input)
        try:
            request_key, request = _resolve_request_input(
                request_input_mode=request_input_mode,
                request_input=request_input,
                area_catalog=area_catalog,
                geography=geography,
            )
        except Exception as exc:
            issues.append(
                WorkerIssue(
                    item=request_key,
                    phase="request",
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )
            continue
        if request is None:
            print(
                f"Skipping {request_key}: no matching geography in legacy work item",
                file=sys.stderr,
            )
            continue
        requests.append(request)
    return tuple(requests), tuple(issues)


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

    output_dir = Path(args.output_dir)
    run_id = args.run_id or output_dir.name or "local-worker"

    from policyengine_us_data.utils.takeup import (
        SIMPLE_TAKEUP_VARS,
    )

    takeup_filter = [spec["variable"] for spec in SIMPLE_TAKEUP_VARS]

    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    from policyengine_us_data.build_outputs.area_catalog import USAreaCatalog
    from policyengine_us_data.build_outputs.requests import AreaBuildRequest
    from policyengine_us_data.build_outputs.validation import (
        AreaValidationService,
        ValidationPolicy,
    )
    from policyengine_us_data.build_outputs.worker_service import (
        LocalH5WorkerService,
        WorkerExecutionConfig,
        WorkerResult,
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

    requests, request_issues = _resolve_worker_requests(
        request_input_mode=request_input_mode,
        request_inputs=request_inputs,
        area_catalog=area_catalog,
        geography=geography,
    )
    worker_result = LocalH5WorkerService(
        validation_service=validation_service,
    ).execute(
        session=session,
        requests=requests,
        config=WorkerExecutionConfig(
            output_dir=output_dir,
            takeup_filter=tuple(takeup_filter),
            validate=not args.no_validate,
        ),
    )
    if request_issues:
        worker_result = WorkerResult(
            area_results=worker_result.area_results,
            issues=(*request_issues, *worker_result.issues),
        )

    for area_result in worker_result.area_results:
        if area_result.status == "completed":
            print(f"Completed {area_result.key}", file=sys.stderr)
        else:
            message = (
                area_result.issues[0].message if area_result.issues else "unknown error"
            )
            print(f"FAILED {area_result.key}: {message}", file=sys.stderr)
        if area_result.validation_status == "passed" and area_result.validation_summary:
            summary = area_result.validation_summary
            print(
                f"  Validated {area_result.key}: "
                f"{summary['n_targets']} targets, "
                f"{summary['n_sanity_fail']} sanity fails, "
                f"mean RAE={summary['mean_rel_abs_error']:.4f}",
                file=sys.stderr,
            )
        elif area_result.validation_status == "error" and area_result.issues:
            print(
                f"  Validation failed for {area_result.key}: "
                f"{area_result.issues[-1].message}",
                file=sys.stderr,
            )

    sys.stdout = original_stdout
    print(json.dumps(worker_result.to_legacy_dict()))


if __name__ == "__main__":
    main()
