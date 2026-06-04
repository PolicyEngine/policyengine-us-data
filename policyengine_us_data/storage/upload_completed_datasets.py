import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
from huggingface_hub import HfApi, hf_hub_download
from policyengine_core.data import Dataset

from policyengine_us_data.__version__ import __version__ as DATA_PACKAGE_VERSION
from policyengine_us_data.db.etl_national_targets import (
    BEA_NIPA_WAGES_AND_SALARIES_2024,
)
from policyengine_us_data.datasets import EnhancedCPS_2024
from policyengine_us_data.datasets.cps.cps import CPS_2024
from policyengine_us_data.datasets.cps.enhanced_cps import clone_diagnostics_path
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.data_upload import (
    cleanup_staging_hf,
    preflight_release_manifest_publish,
    promote_staging_to_production_hf,
    publish_release_manifest_to_hf,
    upload_from_hf_staging_to_gcs,
    upload_to_staging_hf,
)
from policyengine_us_data.utils.run_context import (
    resolve_run_id,
    staging_prefix as build_staging_prefix,
)
from policyengine_us_data.utils.dataset_validation import (
    DatasetContractError,
    load_dataset_for_validation,
    validate_dataset_contract,
)
from policyengine_us_data.utils.version_manifest import (
    HFVersionInfo,
    build_manifest,
    upload_manifest,
)

HF_REPO_NAME = "policyengine/policyengine-us-data"
HF_REPO_TYPE = "model"
GCS_BUCKET_NAME = "policyengine-us-data"

# Datasets that require full validation before upload.
# These are the main datasets used in production simulations.
VALIDATED_FILENAMES = {
    "enhanced_cps_2024.h5",
    "cps_2024.h5",
}

# Raw CPS is a source artifact used to construct final simulation datasets.
# It intentionally carries some Census source measures whose names now map to
# formula variables in policyengine-us. The leaf-input export contract is
# enforced on final simulation datasets instead.
ENFORCE_LEAF_INPUT_EXPORT_BY_FILENAME = {
    "enhanced_cps_2024.h5": True,
    "cps_2024.h5": False,
}

# Minimum file sizes in bytes for validated datasets.
MIN_FILE_SIZES = {
    "enhanced_cps_2024.h5": 95 * 1024 * 1024,  # 95 MB
    "cps_2024.h5": 50 * 1024 * 1024,  # 50 MB
}


@dataclass(frozen=True)
class H5SumCheck:
    variable: str
    label: str
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class MicrosimulationAggregateCheck:
    variable: str
    label: str
    statistic: str
    min_value: float | None = None
    max_value: float | None = None
    map_to: str | None = None
    filter_variable: str | None = None
    filter_map_to: str | None = None
    filter_min_value: float | None = None


# H5 groups that must exist and contain data in every validated dataset.
REQUIRED_GROUPS = ("household_weight",)

# Extra file-specific leaf inputs that should survive publication. These are
# not formula outputs; they are source or imputed inputs used by model formulas.
REQUIRED_VARIABLES_BY_FILENAME = {
    "enhanced_cps_2024.h5": (
        # All four Social Security components must be present and non-zero.
        # Only `retirement` was guarded before; a build that dropped one of
        # the others (as the extended-CPS step once dropped `retirement`)
        # would otherwise publish an eCPS that under-counts Social Security.
        "social_security_retirement",
        "social_security_disability",
        "social_security_survivors",
        "social_security_dependents",
        "takes_up_snap_if_eligible",
        "takes_up_ssi_if_eligible",
        "takes_up_tanf_if_eligible",
        "takes_up_housing_assistance_if_eligible",
        "would_claim_wic",
        "is_wic_at_nutritional_risk",
    ),
}

PROHIBITED_DATASET_VARIABLES = {
    "social_security_retirement_reported",
    "snap_reported",
    "ssi_reported",
    "tanf_reported",
    "wic_reported",
    "free_school_meals_reported",
    "reduced_price_school_meals_reported",
    "spm_unit_wic_reported",
    "spm_unit_total_income_reported",
    "spm_unit_net_income_reported",
    "spm_unit_broadband_subsidy",
    "spm_unit_broadband_subsidy_reported",
    "spm_unit_payroll_tax_reported",
    "spm_unit_federal_tax_reported",
    "spm_unit_state_tax_reported",
}

# At least one of these income groups must exist with data.
INCOME_GROUPS = [
    "employment_income_before_lsr",
    "employment_income",
]

# Aggregate thresholds for broad sanity checks (year 2024).
MIN_PLAUSIBLE_EMPLOYMENT_INCOME_SUM = 5e12  # $5 trillion
# This is a publication gate, not a broad plausibility check: enhanced CPS
# calibration should hit the BEA NIPA wages target closely enough that missing
# target wiring fails before local-area outputs are built.
NIPA_EMPLOYMENT_INCOME_TOLERANCE = 0.01
MIN_ENHANCED_CPS_EMPLOYMENT_INCOME_SUM = BEA_NIPA_WAGES_AND_SALARIES_2024 * (
    1 - NIPA_EMPLOYMENT_INCOME_TOLERANCE
)
MAX_ENHANCED_CPS_EMPLOYMENT_INCOME_SUM = BEA_NIPA_WAGES_AND_SALARIES_2024 * (
    1 + NIPA_EMPLOYMENT_INCOME_TOLERANCE
)
MIN_HOUSEHOLD_WEIGHT_SUM = 100e6  # 100 million
MAX_HOUSEHOLD_WEIGHT_SUM = 200e6  # 200 million

H5_SUM_CHECKS_BY_FILENAME = {
    "enhanced_cps_2024.h5": (
        H5SumCheck(
            variable="household_weight",
            label="household_weight raw sum",
            min_value=MIN_HOUSEHOLD_WEIGHT_SUM,
            max_value=MAX_HOUSEHOLD_WEIGHT_SUM,
        ),
    ),
    "cps_2024.h5": (
        H5SumCheck(
            variable="household_weight",
            label="household_weight raw sum",
            min_value=MIN_HOUSEHOLD_WEIGHT_SUM,
            max_value=MAX_HOUSEHOLD_WEIGHT_SUM,
        ),
    ),
}

MICROSIMULATION_AGGREGATE_CHECKS_BY_FILENAME = {
    "enhanced_cps_2024.h5": (
        MicrosimulationAggregateCheck(
            variable="employment_income",
            label="employment_income sum vs NIPA wages target",
            statistic="sum",
            min_value=MIN_ENHANCED_CPS_EMPLOYMENT_INCOME_SUM,
            max_value=MAX_ENHANCED_CPS_EMPLOYMENT_INCOME_SUM,
        ),
        MicrosimulationAggregateCheck(
            variable="social_security_retirement",
            label="social_security_retirement sum",
            statistic="sum",
            min_value=8e11,
            max_value=1.6e12,
        ),
        MicrosimulationAggregateCheck(
            variable="person_in_poverty",
            label="SPM poverty rate",
            statistic="mean",
            min_value=0.08,
            max_value=0.35,
            map_to="person",
        ),
        MicrosimulationAggregateCheck(
            variable="person_in_poverty",
            label="senior SPM poverty rate",
            statistic="mean",
            min_value=0.05,
            max_value=0.35,
            map_to="person",
            filter_variable="age",
            filter_map_to="person",
            filter_min_value=65,
        ),
    ),
    "cps_2024.h5": (
        MicrosimulationAggregateCheck(
            variable="employment_income",
            label="employment_income sum",
            statistic="sum",
            min_value=MIN_PLAUSIBLE_EMPLOYMENT_INCOME_SUM,
        ),
    ),
}

CLONE_DIAGNOSTICS_FILENAME = "enhanced_cps_2024.clone_diagnostics.json"
CLONE_DIAGNOSTICS_METRICS = {
    "clone_household_weight_share_pct",
    "clone_person_weight_share_pct",
    "clone_poor_person_weight_share_pct",
    "clone_childcare_exceeds_pre_subsidy_share_pct",
    "clone_childcare_above_5000_share_pct",
    "clone_taxes_exceed_market_income_share_pct",
}


def _resolve_run_id(run_id: str = "") -> str:
    return run_id or resolve_run_id()


class DatasetValidationError(Exception):
    """Raised when a dataset fails pre-upload validation."""

    pass


def _dataset_upload_specs(
    require_enhanced_cps: bool = True,
    require_small_enhanced_cps: bool = True,
) -> list[tuple[Path, str, bool]]:
    specs = [
        (CPS_2024.file_path, CPS_2024.file_path.name, True),
        (
            STORAGE_FOLDER / "calibration" / "policy_data.db",
            "policy_data.db",
            True,
        ),
        (
            EnhancedCPS_2024.file_path,
            EnhancedCPS_2024.file_path.name,
            require_enhanced_cps,
        ),
        (
            clone_diagnostics_path(EnhancedCPS_2024.file_path),
            clone_diagnostics_path(EnhancedCPS_2024.file_path).name,
            require_enhanced_cps,
        ),
    ]
    if require_small_enhanced_cps:
        specs.append(
            (
                STORAGE_FOLDER / "small_enhanced_cps_2024.h5",
                "small_enhanced_cps_2024.h5",
                require_enhanced_cps,
            )
        )
    return specs


def _enhanced_dataset_files(require_small_enhanced_cps: bool = True) -> list[Path]:
    files = [
        EnhancedCPS_2024.file_path,
        clone_diagnostics_path(EnhancedCPS_2024.file_path),
    ]
    if require_small_enhanced_cps:
        files.append(STORAGE_FOLDER / "small_enhanced_cps_2024.h5")
    return files


def _collect_existing_dataset_artifacts(
    require_enhanced_cps: bool = True,
    require_small_enhanced_cps: bool = True,
) -> list[tuple[Path, str]]:
    existing_files = []
    for file_path, repo_path, required in _dataset_upload_specs(
        require_enhanced_cps,
        require_small_enhanced_cps,
    ):
        if file_path.exists():
            existing_files.append((file_path, repo_path))
            print(f"✓ Found{' (optional)' if not required else ''}: {file_path}")
            continue
        if required:
            raise FileNotFoundError(f"Required file not found: {file_path}")
        print(f"⚠ Skipping (not built): {file_path}")

    if not existing_files:
        raise ValueError("No dataset files found to upload!")

    return existing_files


def _collect_staged_dataset_repo_paths(
    require_enhanced_cps: bool = True,
    require_small_enhanced_cps: bool = True,
    run_id: str = "",
    candidate_version: str | None = None,
) -> list[str]:
    api = HfApi()
    run_id = _resolve_run_id(run_id)
    prefix = build_staging_prefix(
        run_id,
        candidate_version=candidate_version or DATA_PACKAGE_VERSION,
    )
    repo_files = set(
        api.list_repo_files(
            repo_id=HF_REPO_NAME,
            repo_type=HF_REPO_TYPE,
        )
    )

    rel_paths = []
    missing_required = []
    for _, repo_path, required in _dataset_upload_specs(
        require_enhanced_cps,
        require_small_enhanced_cps,
    ):
        staged_path = f"{prefix}/{repo_path}"
        if staged_path in repo_files:
            rel_paths.append(repo_path)
        elif required:
            missing_required.append(staged_path)

    if missing_required:
        raise FileNotFoundError(
            "Missing staged dataset artifacts: " + ", ".join(sorted(missing_required))
        )
    if not rel_paths:
        raise ValueError("No staged dataset files found to promote.")

    return rel_paths


def _download_staged_dataset_artifacts(
    rel_paths: list[str],
    run_id: str = "",
    candidate_version: str | None = None,
) -> list[tuple[Path, str]]:
    run_id = _resolve_run_id(run_id)
    staging_prefix = build_staging_prefix(
        run_id,
        candidate_version=candidate_version or DATA_PACKAGE_VERSION,
    )
    downloaded_files = []
    for rel_path in rel_paths:
        local_path = Path(
            hf_hub_download(
                repo_id=HF_REPO_NAME,
                filename=f"{staging_prefix}/{rel_path}",
                repo_type=HF_REPO_TYPE,
            )
        )
        downloaded_files.append((local_path, rel_path))
    return downloaded_files


def _validate_dataset_artifacts(files_with_repo_paths: list[tuple[Path, str]]) -> None:
    print("\nValidating datasets...")
    for file_path, _ in files_with_repo_paths:
        validate_dataset(file_path)


def validate_dataset(file_path: Path) -> None:
    """Validate a dataset file before upload.

    Checks file size, H5 structure, and aggregate statistics
    to prevent uploading corrupted data.

    Args:
        file_path: Path to the H5 dataset file.

    Raises:
        DatasetValidationError: If any validation check fails.
    """
    file_path = Path(file_path)
    filename = file_path.name

    if filename == CLONE_DIAGNOSTICS_FILENAME:
        validate_clone_diagnostics(file_path)
        return

    if filename not in VALIDATED_FILENAMES:
        return  # Skip validation for auxiliary files

    errors = []

    # 1. File size check
    file_size = file_path.stat().st_size
    min_size = MIN_FILE_SIZES.get(filename, 0)
    if file_size < min_size:
        errors.append(
            f"File size {file_size / 1024 / 1024:.1f} MB is below "
            f"minimum {min_size / 1024 / 1024:.0f} MB. "
            f"This likely indicates corrupted/incomplete data."
        )

    try:
        with h5py.File(file_path, "r") as f:
            prohibited_variables = sorted(
                PROHIBITED_DATASET_VARIABLES.intersection(f.keys())
            )
            if prohibited_variables:
                errors.append(
                    "Dataset contains temporary or retired variables: "
                    + ", ".join(prohibited_variables)
                )

            required_groups = (
                *REQUIRED_GROUPS,
                *REQUIRED_VARIABLES_BY_FILENAME.get(filename, ()),
            )
            for group_name in required_groups:
                if not _check_group_has_data(f, group_name):
                    errors.append(
                        f"Required group '{group_name}' missing or empty in H5 file."
                    )

            for check in H5_SUM_CHECKS_BY_FILENAME.get(filename, ()):
                if not _check_group_has_data(f, check.variable):
                    continue
                _validate_range(
                    errors,
                    label=check.label,
                    value=_h5_sum(f, check.variable),
                    min_value=check.min_value,
                    max_value=check.max_value,
                )

            # At least one income group must have data
            has_income = any(_check_group_has_data(f, g) for g in INCOME_GROUPS)
            if not has_income:
                errors.append(
                    f"No income data found. Need at least one of "
                    f"{INCOME_GROUPS} with data in H5 file."
                )
    except Exception as e:
        errors.append(f"Failed to read H5 file: {e}")

    if errors:
        raise DatasetValidationError(
            f"Validation failed for {filename}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    try:
        contract_summary = validate_dataset_contract(
            file_path,
            enforce_no_computed_policyengine_us_variables=ENFORCE_LEAF_INPUT_EXPORT_BY_FILENAME.get(
                filename,
                True,
            ),
        )
    except DatasetContractError as e:
        errors.append(f"Dataset contract validation failed: {e}")
        raise DatasetValidationError(
            f"Validation failed for {filename}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        ) from e

    # 3. Aggregate statistics check via Microsimulation
    # Import here to avoid heavy import at module level.
    from policyengine_us import Microsimulation

    try:
        sim = Microsimulation(
            dataset=load_dataset_for_validation(file_path, Dataset.from_file)
        )
        aggregate_results = _run_microsimulation_aggregate_checks(
            sim,
            filename=filename,
            period=2024,
            errors=errors,
        )
    except Exception as e:
        errors.append(f"Microsimulation validation failed: {e}")

    if errors:
        raise DatasetValidationError(
            f"Validation failed for {filename}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    print(f"  ✓ Validation passed for {filename}")
    print(f"    File size: {file_size / 1024 / 1024:.1f} MB")
    print(
        "    policyengine-us: "
        f"{contract_summary.policyengine_us.version}"
        + (
            f" (locked {contract_summary.policyengine_us.locked_version})"
            if contract_summary.policyengine_us.locked_version
            else ""
        )
    )
    for label, value in aggregate_results:
        print(f"    {label}: {_format_validation_value(value)}")


def _check_group_has_data(f, name):
    """Return True if the H5 group/dataset has non-empty data."""
    if name not in f:
        return False
    group = f[name]
    if isinstance(group, h5py.Group):
        if len(group.keys()) == 0:
            return False
        first_key = list(group.keys())[0]
        return len(group[first_key][:]) > 0
    elif isinstance(group, h5py.Dataset):
        return group.size > 0
    return False


def _h5_sum(f, name: str) -> float:
    group = f[name]
    if isinstance(group, h5py.Group):
        first_key = list(group.keys())[0]
        return float(group[first_key][:].sum())
    return float(group[:].sum())


def _format_validation_value(value: float) -> str:
    if abs(value) >= 1e6:
        return f"{value:,.0f}"
    return f"{value:.4f}"


def _validate_range(
    errors: list[str],
    *,
    label: str,
    value: float,
    min_value: float | None,
    max_value: float | None,
) -> None:
    if min_value is not None and value < min_value:
        errors.append(
            f"{label} = {_format_validation_value(value)}, "
            f"expected >= {_format_validation_value(min_value)}."
        )
    if max_value is not None and value > max_value:
        errors.append(
            f"{label} = {_format_validation_value(value)}, "
            f"expected <= {_format_validation_value(max_value)}."
        )


def _calculate_for_check(sim, variable: str, period: int, map_to: str | None):
    kwargs = {"period": period}
    if map_to is not None:
        kwargs["map_to"] = map_to
    return sim.calculate(variable, **kwargs)


def _run_microsimulation_aggregate_checks(
    sim,
    *,
    filename: str,
    period: int,
    errors: list[str],
) -> list[tuple[str, float]]:
    results = []
    for check in MICROSIMULATION_AGGREGATE_CHECKS_BY_FILENAME.get(filename, ()):
        values = _calculate_for_check(
            sim,
            check.variable,
            period,
            check.map_to,
        )
        if check.filter_variable is not None:
            filter_values = _calculate_for_check(
                sim,
                check.filter_variable,
                period,
                check.filter_map_to,
            )
            if check.filter_min_value is not None:
                values = values[filter_values >= check.filter_min_value]
        if check.statistic == "sum":
            result = float(values.sum())
        elif check.statistic == "mean":
            result = float(values.mean())
        else:
            raise ValueError(f"Unsupported aggregate statistic: {check.statistic}")
        _validate_range(
            errors,
            label=check.label,
            value=result,
            min_value=check.min_value,
            max_value=check.max_value,
        )
        results.append((check.label, result))
    return results


def validate_clone_diagnostics(file_path: Path) -> None:
    file_path = Path(file_path)
    errors = []

    try:
        payload = json.loads(file_path.read_text())
    except Exception as exc:
        raise DatasetValidationError(
            f"Validation failed for {file_path.name}:\n"
            f"  - Failed to read clone diagnostics JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        errors.append("Expected clone diagnostics JSON object.")
    elif "periods" in payload:
        periods = payload["periods"]
        if not isinstance(periods, dict) or not periods:
            errors.append("Expected non-empty 'periods' object.")
        else:
            for period, diagnostics in periods.items():
                try:
                    int(period)
                except (TypeError, ValueError):
                    errors.append(f"Period key {period!r} is not an integer year.")
                    continue
                errors.extend(
                    _clone_diagnostics_errors(
                        diagnostics,
                        context=f"period {period}",
                    )
                )
    else:
        period = payload.get("period")
        if isinstance(period, bool):
            errors.append("'period' must be an integer year.")
        else:
            try:
                int(period)
            except (TypeError, ValueError):
                errors.append("'period' must be an integer year.")
        errors.extend(_clone_diagnostics_errors(payload, context="top-level"))

    if errors:
        raise DatasetValidationError(
            f"Validation failed for {file_path.name}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    print(f"  ✓ Validation passed for {file_path.name}")


def _clone_diagnostics_errors(diagnostics, *, context: str) -> list[str]:
    errors = []
    if not isinstance(diagnostics, dict):
        return [f"Expected clone diagnostics object for {context}."]

    missing = sorted(CLONE_DIAGNOSTICS_METRICS.difference(diagnostics))
    if missing:
        errors.append(f"Missing clone diagnostics metric(s) in {context}: {missing}")

    for metric in sorted(CLONE_DIAGNOSTICS_METRICS.intersection(diagnostics)):
        value = diagnostics[metric]
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append(f"{context} metric {metric} must be numeric.")
            continue
        if not math.isfinite(float(value)):
            errors.append(f"{context} metric {metric} must be finite.")
            continue
        if not 0 <= float(value) <= 100:
            errors.append(f"{context} metric {metric} must be between 0 and 100.")

    return errors


def stage_datasets(
    require_enhanced_cps: bool = True,
    require_small_enhanced_cps: bool = True,
    version: str | None = None,
    candidate_version: str | None = None,
    run_id: str = "",
) -> list[tuple[Path, str]]:
    run_id = _resolve_run_id(run_id)
    candidate_version = candidate_version or version or DATA_PACKAGE_VERSION
    files_with_repo_paths = _collect_existing_dataset_artifacts(
        require_enhanced_cps=require_enhanced_cps,
        require_small_enhanced_cps=require_small_enhanced_cps,
    )
    _validate_dataset_artifacts(files_with_repo_paths)

    print(f"\nStaging {len(files_with_repo_paths)} files on Hugging Face...")
    upload_to_staging_hf(
        files_with_repo_paths,
        candidate_version=candidate_version,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        run_id=run_id,
    )
    return files_with_repo_paths


def promote_datasets(
    require_enhanced_cps: bool = True,
    require_small_enhanced_cps: bool = True,
    version: str | None = None,
    candidate_version: str | None = None,
    release_version: str | None = None,
    run_id: str = "",
    files_with_repo_paths: list[tuple[Path, str]] | None = None,
    cleanup_staging: bool = True,
) -> list[str]:
    run_id = _resolve_run_id(run_id)
    candidate_version = candidate_version or version or DATA_PACKAGE_VERSION
    release_version = release_version or version or candidate_version
    rel_paths = (
        [repo_path for _, repo_path in files_with_repo_paths]
        if files_with_repo_paths
        else _collect_staged_dataset_repo_paths(
            require_enhanced_cps=require_enhanced_cps,
            require_small_enhanced_cps=require_small_enhanced_cps,
            run_id=run_id,
            candidate_version=candidate_version,
        )
    )
    manifest_files = (
        files_with_repo_paths
        if files_with_repo_paths
        else _download_staged_dataset_artifacts(
            rel_paths,
            run_id=run_id,
            candidate_version=candidate_version,
        )
    )
    if files_with_repo_paths is None:
        _validate_dataset_artifacts(manifest_files)
    should_finalize, missing_prefixes = preflight_release_manifest_publish(
        manifest_files,
        version=release_version,
        new_repo_paths=rel_paths,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        pipeline_run_id=run_id,
    )

    print(f"\nPromoting {len(rel_paths)} staged files to production...")
    promote_staging_to_production_hf(
        rel_paths,
        candidate_version=candidate_version,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        run_id=run_id,
    )
    upload_from_hf_staging_to_gcs(
        rel_paths,
        candidate_version=candidate_version,
        release_version=release_version,
        gcs_bucket_name=GCS_BUCKET_NAME,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        run_id=run_id,
    )
    manifest = publish_release_manifest_to_hf(
        manifest_files,
        version=release_version,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        create_tag=should_finalize,
        pipeline_run_id=run_id,
    )
    if not should_finalize:
        print(
            "Release manifest updated without final tag; missing local-area prefixes: "
            + ", ".join(missing_prefixes)
        )

    # Legacy consumers still resolve versions through version_manifest.json,
    # but only once the release has been finalized at a stable HF revision.
    if should_finalize:
        upload_manifest(
            build_manifest(
                version=release_version,
                blob_names=sorted(
                    artifact["path"] for artifact in manifest["artifacts"].values()
                ),
                hf_info=HFVersionInfo(repo=HF_REPO_NAME, commit=release_version),
                run_id=run_id or None,
            )
        )
    else:
        print("Deferring version_manifest.json update until the release is finalized.")
    if cleanup_staging:
        cleanup_staging_hf(
            rel_paths,
            candidate_version=candidate_version,
            hf_repo_name=HF_REPO_NAME,
            hf_repo_type=HF_REPO_TYPE,
            run_id=run_id,
        )
    else:
        print("Deferring staged dataset cleanup until full release promotion succeeds.")
    return rel_paths


def upload_datasets(
    require_enhanced_cps: bool = True,
    require_small_enhanced_cps: bool = True,
    *,
    stage_only: bool = False,
    promote_only: bool = False,
    run_id: str = "",
    version: str | None = None,
    candidate_version: str | None = None,
    release_version: str | None = None,
    cleanup_staging: bool = True,
):
    run_id = _resolve_run_id(run_id)
    if stage_only and promote_only:
        raise ValueError("Choose either stage_only or promote_only, not both.")

    candidate_version = candidate_version or version or DATA_PACKAGE_VERSION
    release_version = release_version or version or candidate_version

    if promote_only:
        return promote_datasets(
            require_enhanced_cps=require_enhanced_cps,
            require_small_enhanced_cps=require_small_enhanced_cps,
            candidate_version=candidate_version,
            release_version=release_version,
            run_id=run_id,
            cleanup_staging=cleanup_staging,
        )

    files_with_repo_paths = stage_datasets(
        require_enhanced_cps=require_enhanced_cps,
        require_small_enhanced_cps=require_small_enhanced_cps,
        candidate_version=candidate_version,
        run_id=run_id,
    )
    if stage_only:
        return [repo_path for _, repo_path in files_with_repo_paths]

    return promote_datasets(
        require_enhanced_cps=require_enhanced_cps,
        require_small_enhanced_cps=require_small_enhanced_cps,
        candidate_version=candidate_version,
        release_version=release_version,
        run_id=run_id,
        files_with_repo_paths=files_with_repo_paths,
        cleanup_staging=cleanup_staging,
    )


def validate_all_datasets():
    """Validate all main datasets in storage. Called by `make validate-data`."""
    validate_built_datasets(require_enhanced_cps=True)


def validate_built_datasets(
    require_enhanced_cps: bool = True,
    require_small_enhanced_cps: bool = True,
):
    required_files = [CPS_2024.file_path]
    if require_enhanced_cps:
        required_files.extend(_enhanced_dataset_files(require_small_enhanced_cps))

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Expected dataset not found at {file_path}")
        validate_dataset(file_path)
    print("\nAll dataset validations passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-require-enhanced-cps",
        action="store_true",
        help="Treat enhanced_cps and small_enhanced_cps as optional.",
    )
    parser.add_argument(
        "--no-require-small-enhanced-cps",
        action="store_true",
        help="Treat small_enhanced_cps as optional while still requiring enhanced_cps.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate built datasets without uploading them.",
    )
    upload_mode = parser.add_mutually_exclusive_group()
    upload_mode.add_argument(
        "--stage-only",
        action="store_true",
        help="Validate and upload built datasets only to HF staging.",
    )
    upload_mode.add_argument(
        "--promote-only",
        action="store_true",
        help="Promote already-staged datasets into the immutable release.",
    )
    parser.add_argument(
        "--run-id",
        default=resolve_run_id(),
        help="GitHub-created staging run ID.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Override the policyengine-us-data version used for staging/promote.",
    )
    args = parser.parse_args()
    require_enhanced_cps = not args.no_require_enhanced_cps
    require_small_enhanced_cps = (
        require_enhanced_cps and not args.no_require_small_enhanced_cps
    )
    if args.validate_only:
        validate_built_datasets(
            require_enhanced_cps=require_enhanced_cps,
            require_small_enhanced_cps=require_small_enhanced_cps,
        )
    else:
        upload_datasets(
            require_enhanced_cps=require_enhanced_cps,
            require_small_enhanced_cps=require_small_enhanced_cps,
            stage_only=args.stage_only,
            promote_only=args.promote_only,
            run_id=args.run_id,
            version=args.version,
        )
