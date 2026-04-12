from pathlib import Path
from importlib import metadata

import h5py
from huggingface_hub import HfApi, hf_hub_download
from policyengine_core.data import Dataset

from policyengine_us_data.datasets import EnhancedCPS_2024
from policyengine_us_data.datasets.cps.cps import CPS_2024
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.data_upload import (
    cleanup_staging_hf,
    preflight_release_manifest_publish,
    promote_staging_to_production_hf,
    publish_release_manifest_to_hf,
    upload_from_hf_staging_to_gcs,
    upload_to_staging_hf,
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

# Minimum file sizes in bytes for validated datasets.
MIN_FILE_SIZES = {
    "enhanced_cps_2024.h5": 95 * 1024 * 1024,  # 95 MB
    "cps_2024.h5": 50 * 1024 * 1024,  # 50 MB
}

# H5 groups that must exist and contain data.
REQUIRED_GROUPS = [
    "household_weight",
]

# At least one of these income groups must exist with data.
INCOME_GROUPS = [
    "employment_income_before_lsr",
    "employment_income",
]

# Aggregate thresholds for sanity checks (year 2024).
MIN_EMPLOYMENT_INCOME_SUM = 5e12  # $5 trillion
MIN_HOUSEHOLD_WEIGHT_SUM = 100e6  # 100 million
MAX_HOUSEHOLD_WEIGHT_SUM = 200e6  # 200 million


class DatasetValidationError(Exception):
    """Raised when a dataset fails pre-upload validation."""

    pass


def _dataset_upload_specs(
    require_enhanced_cps: bool = True,
) -> list[tuple[Path, str, bool]]:
    return [
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
            STORAGE_FOLDER / "small_enhanced_cps_2024.h5",
            "small_enhanced_cps_2024.h5",
            require_enhanced_cps,
        ),
    ]


def _collect_existing_dataset_artifacts(
    require_enhanced_cps: bool = True,
) -> list[tuple[Path, str]]:
    existing_files = []
    for file_path, repo_path, required in _dataset_upload_specs(require_enhanced_cps):
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
    run_id: str = "",
) -> list[str]:
    api = HfApi()
    prefix = f"staging/{run_id}" if run_id else "staging"
    repo_files = set(
        api.list_repo_files(
            repo_id=HF_REPO_NAME,
            repo_type=HF_REPO_TYPE,
        )
    )

    rel_paths = []
    missing_required = []
    for _, repo_path, required in _dataset_upload_specs(require_enhanced_cps):
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
) -> list[tuple[Path, str]]:
    staging_prefix = f"staging/{run_id}" if run_id else "staging"
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

    # 2. H5 structure check - verify critical groups exist with data
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

    try:
        with h5py.File(file_path, "r") as f:
            for group_name in REQUIRED_GROUPS:
                if not _check_group_has_data(f, group_name):
                    errors.append(
                        f"Required group '{group_name}' missing or empty in H5 file."
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
        contract_summary = validate_dataset_contract(file_path)
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
        year = 2024

        emp_income = sim.calculate("employment_income", year).sum()
        if emp_income < MIN_EMPLOYMENT_INCOME_SUM:
            errors.append(
                f"employment_income sum = ${emp_income:,.0f}, "
                f"expected > ${MIN_EMPLOYMENT_INCOME_SUM:,.0f}. "
                f"Data may have dropped employment income."
            )

        hh_weight = sim.calculate("household_weight", year).values.sum()
        if hh_weight < MIN_HOUSEHOLD_WEIGHT_SUM:
            errors.append(
                f"household_weight sum = {hh_weight:,.0f}, "
                f"expected > {MIN_HOUSEHOLD_WEIGHT_SUM:,.0f}."
            )
        if hh_weight > MAX_HOUSEHOLD_WEIGHT_SUM:
            errors.append(
                f"household_weight sum = {hh_weight:,.0f}, "
                f"expected < {MAX_HOUSEHOLD_WEIGHT_SUM:,.0f}."
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
    print(f"    employment_income sum: ${emp_income:,.0f}")
    print(f"    Household weight sum: {hh_weight:,.0f}")


def stage_datasets(
    require_enhanced_cps: bool = True,
    version: str | None = None,
    run_id: str = "",
) -> list[tuple[Path, str]]:
    version = version or metadata.version("policyengine-us-data")
    files_with_repo_paths = _collect_existing_dataset_artifacts(
        require_enhanced_cps=require_enhanced_cps
    )
    _validate_dataset_artifacts(files_with_repo_paths)

    print(f"\nStaging {len(files_with_repo_paths)} files on Hugging Face...")
    upload_to_staging_hf(
        files_with_repo_paths,
        version=version,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        run_id=run_id,
    )
    return files_with_repo_paths


def promote_datasets(
    require_enhanced_cps: bool = True,
    version: str | None = None,
    run_id: str = "",
    files_with_repo_paths: list[tuple[Path, str]] | None = None,
) -> list[str]:
    version = version or metadata.version("policyengine-us-data")
    rel_paths = (
        [repo_path for _, repo_path in files_with_repo_paths]
        if files_with_repo_paths
        else _collect_staged_dataset_repo_paths(
            require_enhanced_cps=require_enhanced_cps,
            run_id=run_id,
        )
    )
    manifest_files = (
        files_with_repo_paths
        if files_with_repo_paths
        else _download_staged_dataset_artifacts(rel_paths, run_id=run_id)
    )
    should_finalize, missing_prefixes = preflight_release_manifest_publish(
        manifest_files,
        version=version,
        new_repo_paths=rel_paths,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
    )

    print(f"\nPromoting {len(rel_paths)} staged files to production...")
    promote_staging_to_production_hf(
        rel_paths,
        version=version,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        run_id=run_id,
    )
    upload_from_hf_staging_to_gcs(
        rel_paths,
        version=version,
        gcs_bucket_name=GCS_BUCKET_NAME,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        run_id=run_id,
    )
    manifest = publish_release_manifest_to_hf(
        manifest_files,
        version=version,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        create_tag=should_finalize,
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
                version=version,
                blob_names=sorted(
                    artifact["path"] for artifact in manifest["artifacts"].values()
                ),
                hf_info=HFVersionInfo(repo=HF_REPO_NAME, commit=version),
            )
        )
    else:
        print("Deferring version_manifest.json update until the release is finalized.")
    cleanup_staging_hf(
        rel_paths,
        version=version,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_type=HF_REPO_TYPE,
        run_id=run_id,
    )
    return rel_paths


def upload_datasets(
    require_enhanced_cps: bool = True,
    *,
    stage_only: bool = False,
    promote_only: bool = False,
    run_id: str = "",
    version: str | None = None,
):
    if stage_only and promote_only:
        raise ValueError("Choose either stage_only or promote_only, not both.")

    version = version or metadata.version("policyengine-us-data")

    if promote_only:
        promote_datasets(
            require_enhanced_cps=require_enhanced_cps,
            version=version,
            run_id=run_id,
        )
        return

    files_with_repo_paths = stage_datasets(
        require_enhanced_cps=require_enhanced_cps,
        version=version,
        run_id=run_id,
    )
    if stage_only:
        return

    promote_datasets(
        require_enhanced_cps=require_enhanced_cps,
        version=version,
        run_id=run_id,
        files_with_repo_paths=files_with_repo_paths,
    )


def validate_all_datasets():
    """Validate all main datasets in storage. Called by `make validate-data`."""
    validate_built_datasets(require_enhanced_cps=True)


def validate_built_datasets(require_enhanced_cps: bool = True):
    required_files = [CPS_2024.file_path]
    if require_enhanced_cps:
        required_files.append(EnhancedCPS_2024.file_path)

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
        default="",
        help="Optional staging run ID, for example a CI commit SHA.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Override the policyengine-us-data version used for staging/promote.",
    )
    args = parser.parse_args()
    if args.validate_only:
        validate_built_datasets(require_enhanced_cps=not args.no_require_enhanced_cps)
    else:
        upload_datasets(
            require_enhanced_cps=not args.no_require_enhanced_cps,
            stage_only=args.stage_only,
            promote_only=args.promote_only,
            run_id=args.run_id,
            version=args.version,
        )
