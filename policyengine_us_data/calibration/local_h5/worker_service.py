"""Worker-scoped session loading and per-request H5 execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import traceback

import numpy as np

from .builder import LocalAreaDatasetBuilder
from .contracts import (
    AreaBuildRequest,
    AreaBuildResult,
    AreaFilter,
    ValidationIssue,
    ValidationPolicy,
    ValidationResult,
    WorkerResult,
)
from .package_geography import CalibrationPackageGeographyLoader
from .source_dataset import PolicyEngineDatasetReader, SourceDatasetSnapshot
from .validation import summarize_validation_rows
from .weights import infer_clone_count_from_weight_length
from .writer import H5Writer


@dataclass(frozen=True)
class ValidationContext:
    validation_targets: Any
    training_mask_full: np.ndarray
    constraints_map: Mapping[int, Sequence[Any]]
    db_path: Path
    period: int


@dataclass(frozen=True)
class WorkerSession:
    source_snapshot: SourceDatasetSnapshot
    weights: np.ndarray
    geography: Any
    output_dir: Path
    takeup_filter: tuple[str, ...] = ()
    validation_policy: ValidationPolicy = field(default_factory=ValidationPolicy)
    validation_context: ValidationContext | None = None
    dataset_path: Path | None = None
    weights_path: Path | None = None
    db_path: Path | None = None
    calibration_package_path: Path | None = None
    seed: int = 42
    requested_n_clones: int | None = None
    geography_source: str = ""
    geography_warnings: tuple[str, ...] = ()

    @property
    def n_records(self) -> int:
        return self.source_snapshot.n_households

    @property
    def n_clones(self) -> int:
        return infer_clone_count_from_weight_length(len(self.weights), self.n_records)

    @classmethod
    def load(
        cls,
        *,
        weights_path: Path,
        dataset_path: Path,
        output_dir: Path,
        calibration_package_path: Path | None = None,
        requested_n_clones: int | None = None,
        seed: int = 42,
        takeup_filter: Sequence[str] = (),
        validation_policy: ValidationPolicy | None = None,
        validation_context: ValidationContext | None = None,
        source_reader: PolicyEngineDatasetReader | None = None,
        geography_loader: CalibrationPackageGeographyLoader | None = None,
    ) -> "WorkerSession":
        weights = np.load(weights_path)
        source_reader = source_reader or PolicyEngineDatasetReader(())
        source_snapshot = source_reader.load(dataset_path)
        n_records = source_snapshot.n_households
        n_clones = infer_clone_count_from_weight_length(len(weights), n_records)

        geography_loader = geography_loader or CalibrationPackageGeographyLoader()
        geography_resolution = geography_loader.resolve_for_weights(
            package_path=calibration_package_path,
            weights_length=len(weights),
            n_records=n_records,
            n_clones=n_clones,
            seed=seed,
        )

        return cls(
            source_snapshot=source_snapshot,
            weights=weights,
            geography=geography_resolution.geography,
            output_dir=Path(output_dir),
            takeup_filter=tuple(takeup_filter),
            validation_policy=validation_policy or ValidationPolicy(),
            validation_context=validation_context,
            dataset_path=Path(dataset_path),
            weights_path=Path(weights_path),
            db_path=(validation_context.db_path if validation_context else None),
            calibration_package_path=(
                Path(calibration_package_path)
                if calibration_package_path is not None
                else None
            ),
            seed=seed,
            requested_n_clones=requested_n_clones,
            geography_source=geography_resolution.source,
            geography_warnings=tuple(geography_resolution.warnings),
        )


def load_validation_context(
    *,
    db_path: Path,
    period: int,
    target_config_path: str | Path | None = None,
    validation_config_path: str | Path | None = None,
    policy: ValidationPolicy | None = None,
) -> ValidationContext | None:
    policy = policy or ValidationPolicy()
    if not policy.enabled:
        return None

    from sqlalchemy import create_engine
    from policyengine_us_data.calibration.validate_staging import (
        _batch_stratum_constraints,
        _query_all_active_targets,
    )
    from policyengine_us_data.calibration.unified_calibration import (
        _match_rules,
        load_target_config,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    validation_targets = _query_all_active_targets(engine, period)

    if validation_config_path:
        val_cfg = load_target_config(str(validation_config_path))
        exclude_rules = val_cfg.get("exclude", [])
        if exclude_rules:
            exclude_mask = _match_rules(validation_targets, exclude_rules)
            validation_targets = validation_targets[~exclude_mask].reset_index(drop=True)
        include_rules = val_cfg.get("include", [])
        if include_rules:
            include_mask = _match_rules(validation_targets, include_rules)
            validation_targets = validation_targets[include_mask].reset_index(drop=True)

    if target_config_path:
        target_cfg = load_target_config(str(target_config_path))
        include_rules = target_cfg.get("include", [])
        if include_rules:
            training_mask_full = np.asarray(
                _match_rules(validation_targets, include_rules),
                dtype=bool,
            )
        else:
            training_mask_full = np.ones(len(validation_targets), dtype=bool)
    else:
        training_mask_full = np.ones(len(validation_targets), dtype=bool)

    stratum_ids = validation_targets["stratum_id"].unique().tolist()
    constraints_map = _batch_stratum_constraints(engine, stratum_ids)

    return ValidationContext(
        validation_targets=validation_targets,
        training_mask_full=training_mask_full,
        constraints_map=constraints_map,
        db_path=Path(db_path),
        period=period,
    )


def validate_in_subprocess(
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
    """Run validation for one area inside a subprocess."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from policyengine_us import Microsimulation
    from sqlalchemy import create_engine as _ce
    from policyengine_us_data.calibration.validate_staging import (
        _build_variable_entity_map,
        validate_area,
    )

    engine = _ce(f"sqlite:///{db_path}")
    sim = Microsimulation(dataset=h5_path)
    variable_entity_map = _build_variable_entity_map(sim)

    return validate_area(
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


def validate_output_subprocess(
    output_path: Path,
    request: AreaBuildRequest,
    session: WorkerSession,
) -> ValidationResult:
    if session.validation_context is None:
        return ValidationResult(status="not_run")

    validation_targets = session.validation_context.validation_targets
    geographic_ids = tuple(str(item) for item in request.validation_geographic_ids)

    if request.validation_geo_level is None:
        return ValidationResult(status="passed", summary=summarize_validation_rows(()))

    mask = validation_targets["geo_level"] == request.validation_geo_level
    if geographic_ids:
        mask &= validation_targets["geographic_id"].astype(str).isin(geographic_ids)

    area_targets = validation_targets[mask].reset_index(drop=True)
    area_training = session.validation_context.training_mask_full[mask.values]

    if len(area_targets) == 0:
        summary = summarize_validation_rows(())
        return ValidationResult(status="passed", rows=(), summary=summary)

    area_strata = area_targets["stratum_id"].unique().tolist()
    area_constraints = {
        int(stratum_id): session.validation_context.constraints_map.get(
            int(stratum_id), []
        )
        for stratum_id in area_strata
    }

    import multiprocessing as _mp

    with _mp.get_context("spawn").Pool(1) as pool:
        rows = pool.apply(
            validate_in_subprocess,
            (
                str(output_path),
                _validation_area_type(request),
                request.area_id,
                request.display_name,
                area_targets,
                area_training,
                area_constraints,
                str(session.validation_context.db_path),
                session.validation_context.period,
            ),
        )

    summary = summarize_validation_rows(rows)
    status = "failed" if summary["n_sanity_fail"] > 0 else "passed"
    return ValidationResult(
        status=status,
        rows=tuple(dict(row) for row in rows),
        summary=summary,
    )


def build_request_from_work_item(
    item: Mapping[str, Any],
    *,
    geography,
    state_codes: Mapping[int, str],
    at_large_districts: set[int],
    nyc_county_fips: set[str],
) -> AreaBuildRequest | None:
    item_type = item["type"]
    item_id = item["id"]
    geo_labels = sorted(set(np.asarray(geography.cd_geoid).astype(str)))

    if item_type == "state":
        state_fips = _state_fips_for_code(item_id, state_codes)
        cd_subset = [cd for cd in geo_labels if int(cd) // 100 == state_fips]
        if not cd_subset:
            return None
        return AreaBuildRequest(
            area_type="state",
            area_id=item_id,
            display_name=item_id,
            output_relative_path=f"states/{item_id}.h5",
            filters=(
                AreaFilter(
                    geography_field="cd_geoid",
                    op="in",
                    value=tuple(cd_subset),
                ),
            ),
            validation_geo_level="state",
            validation_geographic_ids=(str(state_fips),),
        )

    if item_type == "district":
        state_code, dist_num = item_id.split("-")
        state_fips = _state_fips_for_code(state_code, state_codes)
        candidate = f"{state_fips}{int(dist_num):02d}"
        if candidate in geo_labels:
            geoid = candidate
        else:
            state_cds = [cd for cd in geo_labels if int(cd) // 100 == state_fips]
            if len(state_cds) == 1:
                geoid = state_cds[0]
            else:
                raise ValueError(
                    f"CD {candidate} not found and state {state_code} has "
                    f"{len(state_cds)} CDs"
                )

        cd_int = int(geoid)
        district_num = cd_int % 100
        if district_num in at_large_districts:
            district_num = 1
        friendly_name = f"{state_code}-{district_num:02d}"
        return AreaBuildRequest(
            area_type="district",
            area_id=friendly_name,
            display_name=friendly_name,
            output_relative_path=f"districts/{friendly_name}.h5",
            filters=(
                AreaFilter(
                    geography_field="cd_geoid",
                    op="in",
                    value=(geoid,),
                ),
            ),
            validation_geo_level="district",
            validation_geographic_ids=(geoid,),
        )

    if item_type == "city":
        county_values = np.asarray(geography.county_fips).astype(str)
        city_mask = np.isin(county_values, sorted(nyc_county_fips))
        city_cds = tuple(sorted(set(np.asarray(geography.cd_geoid).astype(str)[city_mask])))
        return AreaBuildRequest(
            area_type="city",
            area_id=item_id,
            display_name=item_id,
            output_relative_path=f"cities/{item_id}.h5",
            filters=(
                AreaFilter(
                    geography_field="county_fips",
                    op="in",
                    value=tuple(sorted(nyc_county_fips)),
                ),
            ),
            validation_geo_level="district",
            validation_geographic_ids=city_cds,
        )

    if item_type == "national":
        return AreaBuildRequest.national()

    raise ValueError(f"Unknown item type: {item_type}")


def build_requests_from_work_items(
    work_items: Sequence[Mapping[str, Any]],
    *,
    geography,
    state_codes: Mapping[int, str],
    at_large_districts: set[int],
    nyc_county_fips: set[str],
) -> tuple[tuple[AreaBuildRequest, ...], tuple[AreaBuildResult, ...]]:
    requests: list[AreaBuildRequest] = []
    failures: list[AreaBuildResult] = []

    for item in work_items:
        try:
            request = build_request_from_work_item(
                item,
                geography=geography,
                state_codes=state_codes,
                at_large_districts=at_large_districts,
                nyc_county_fips=nyc_county_fips,
            )
        except Exception as error:
            failures.append(
                AreaBuildResult(
                    request=_fallback_request(item),
                    build_status="failed",
                    build_error=str(error),
                )
            )
            continue

        if request is not None:
            requests.append(request)

    return tuple(requests), tuple(failures)


class LocalH5WorkerService:
    """Execute one worker chunk against a shared worker session."""

    def __init__(
        self,
        *,
        builder: LocalAreaDatasetBuilder | None = None,
        writer: H5Writer | None = None,
        validator: Callable[[Path, AreaBuildRequest, WorkerSession], ValidationResult]
        | None = None,
    ) -> None:
        self.builder = builder or LocalAreaDatasetBuilder()
        self.writer = writer or H5Writer()
        self.validator = validator or validate_output_subprocess

    def run(
        self,
        session: WorkerSession,
        requests: Sequence[AreaBuildRequest],
        *,
        initial_failures: Sequence[AreaBuildResult] = (),
    ) -> WorkerResult:
        completed: list[AreaBuildResult] = []
        failed: list[AreaBuildResult] = list(initial_failures)

        for request in requests:
            result = self.build_one(session, request)
            if result.build_status == "completed":
                completed.append(result)
            else:
                failed.append(result)

        return WorkerResult(
            completed=tuple(completed),
            failed=tuple(failed),
        )

    def build_one(
        self,
        session: WorkerSession,
        request: AreaBuildRequest,
    ) -> AreaBuildResult:
        output_path = session.output_dir / request.output_relative_path

        try:
            built = self.builder.build(
                weights=session.weights,
                geography=session.geography,
                source=session.source_snapshot,
                filters=request.filters,
                takeup_filter=session.takeup_filter,
            )
            written_path = self.writer.write_payload(built.payload, output_path)
            self.writer.verify_output(written_path, time_period=built.time_period)
        except Exception as error:
            return AreaBuildResult(
                request=request,
                build_status="failed",
                build_error=str(error),
            )

        validation = self._validate_output(written_path, request, session)
        return AreaBuildResult(
            request=request,
            build_status="completed",
            output_path=written_path,
            validation=validation,
        )

    def _validate_output(
        self,
        output_path: Path,
        request: AreaBuildRequest,
        session: WorkerSession,
    ) -> ValidationResult:
        if not session.validation_policy.enabled or session.validation_context is None:
            return ValidationResult(status="not_run")

        try:
            return self.validator(output_path, request, session)
        except Exception as error:
            return ValidationResult(
                status="error",
                issues=(
                    ValidationIssue(
                        code="validation_exception",
                        message=str(error),
                        severity="error",
                        details={"traceback": traceback.format_exc()},
                    ),
                ),
            )


def worker_result_to_legacy_dict(worker_result: WorkerResult) -> dict[str, Any]:
    completed = []
    failed = []
    errors = []
    validation_rows: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    validation_summary: dict[str, Any] = {}

    for result in worker_result.completed:
        item_key = _result_item_key(result.request)
        completed.append(item_key)
        if result.validation.status in ("passed", "failed"):
            validation_rows.extend(dict(row) for row in result.validation.rows)
            if result.validation.summary:
                validation_summary[item_key] = dict(result.validation.summary)
        elif result.validation.status == "error":
            for issue in result.validation.issues:
                validation_errors.append(
                    {
                        "item": item_key,
                        "error": issue.message,
                        "code": issue.code,
                        "details": dict(issue.details),
                    }
                )

    for result in worker_result.failed:
        item_key = _result_item_key(result.request)
        failed.append(item_key)
        errors.append({"item": item_key, "error": result.build_error})

    for issue in worker_result.worker_issues:
        errors.append(
            {
                "item": "worker",
                "error": issue.message,
                "code": issue.code,
                "details": dict(issue.details),
            }
        )

    return {
        "completed": completed,
        "failed": failed,
        "errors": errors,
        "validation_errors": validation_errors,
        "validation_rows": validation_rows,
        "validation_summary": validation_summary,
    }


def _validation_area_type(request: AreaBuildRequest) -> str:
    if request.area_type == "state":
        return "states"
    if request.area_type == "district":
        return "districts"
    if request.area_type == "city":
        return "cities"
    return "national"


def _result_item_key(request: AreaBuildRequest) -> str:
    return f"{request.area_type}:{request.area_id}"


def _fallback_request(item: Mapping[str, Any]) -> AreaBuildRequest:
    area_type = item.get("type", "custom")
    if area_type not in {"national", "state", "district", "city", "custom"}:
        area_type = "custom"
    area_id = str(item.get("id", "unknown"))
    return AreaBuildRequest(
        area_type=area_type,
        area_id=area_id,
        display_name=area_id,
        output_relative_path=f"invalid/{area_id}.h5",
    )


def _state_fips_for_code(state_code: str, state_codes: Mapping[int, str]) -> int:
    for fips, code in state_codes.items():
        if code == state_code:
            return int(fips)
    raise ValueError(f"Unknown state code: {state_code}")
