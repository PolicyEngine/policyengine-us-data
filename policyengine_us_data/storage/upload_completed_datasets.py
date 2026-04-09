from pathlib import Path

import h5py
from policyengine_core.data import Dataset

from policyengine_us_data.datasets import EnhancedCPS_2024
from policyengine_us_data.datasets.cps.cps import CPS_2024
from policyengine_us_data.datasets.cps.enhanced_cps import clone_diagnostics_path
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.data_upload import upload_data_files
from policyengine_us_data.utils.dataset_validation import (
    DatasetContractError,
    load_dataset_for_validation,
    validate_dataset_contract,
)

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


def upload_datasets(require_enhanced_cps: bool = True):
    required_files = [
        CPS_2024.file_path,
        STORAGE_FOLDER / "calibration" / "policy_data.db",
    ]
    enhanced_files = [
        EnhancedCPS_2024.file_path,
        clone_diagnostics_path(EnhancedCPS_2024.file_path),
        STORAGE_FOLDER / "small_enhanced_cps_2024.h5",
    ]
    if require_enhanced_cps:
        required_files.extend(enhanced_files)

    existing_files = []
    for file_path in required_files:
        if file_path.exists():
            existing_files.append(file_path)
            print(f"✓ Found: {file_path}")
        else:
            raise FileNotFoundError(f"Required file not found: {file_path}")

    if not require_enhanced_cps:
        for file_path in enhanced_files:
            if file_path.exists():
                existing_files.append(file_path)
                print(f"✓ Found (optional): {file_path}")
            else:
                print(f"⚠ Skipping (not built): {file_path}")

    if not existing_files:
        raise ValueError("No dataset files found to upload!")

    # Validate datasets before uploading
    print("\nValidating datasets...")
    for file_path in existing_files:
        validate_dataset(file_path)

    print(f"\nUploading {len(existing_files)} files...")
    upload_data_files(
        files=existing_files,
        hf_repo_name="policyengine/policyengine-us-data",
        hf_repo_type="model",
        gcs_bucket_name="policyengine-us-data",
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
    args = parser.parse_args()
    if args.validate_only:
        validate_built_datasets(require_enhanced_cps=not args.no_require_enhanced_cps)
    else:
        upload_datasets(require_enhanced_cps=not args.no_require_enhanced_cps)
