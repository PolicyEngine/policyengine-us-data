"""Fixture-backed Stage 5 artifacts for tiny pipeline integration tests."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_stage_2 import PERIOD_KEY
from tests.integration.support.tiny_stage_4 import (
    ENHANCED_CPS_REQUIRED_VARIABLES,
    STAGE_4_PERIOD,
    STRATIFIED_CPS_REQUIRED_VARIABLES,
)

__test__ = False


STAGE_5_PERIOD = STAGE_4_PERIOD

SOURCE_IMPUTED_PERSON_VARIABLES = (
    "tip_income",
    "pre_subsidy_rent",
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
    "is_paid_hourly",
)
SOURCE_IMPUTED_HOUSEHOLD_VARIABLES = (
    "household_vehicles_value",
    "net_worth",
    "auto_loan_balance",
    "auto_loan_interest",
)
SOURCE_IMPUTED_REQUIRED_VARIABLES = tuple(
    dict.fromkeys(
        (
            *STRATIFIED_CPS_REQUIRED_VARIABLES,
            *SOURCE_IMPUTED_PERSON_VARIABLES,
            *SOURCE_IMPUTED_HOUSEHOLD_VARIABLES,
        )
    )
)
SMALL_ENHANCED_REQUIRED_VARIABLES = ENHANCED_CPS_REQUIRED_VARIABLES
SPARSE_ENHANCED_REQUIRED_VARIABLES = ENHANCED_CPS_REQUIRED_VARIABLES


@dataclass(frozen=True)
class Stage5Artifacts:
    """Paths written by the fixture-backed Stage 5 builder."""

    source_imputed_path: Path
    source_imputed_alias_path: Path
    small_enhanced_cps_path: Path
    sparse_enhanced_cps_path: Path

    def as_tuple(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.source_imputed_path,
            self.source_imputed_alias_path,
            self.small_enhanced_cps_path,
            self.sparse_enhanced_cps_path,
        )


def create_stage_5_artifacts(workspace: TinyPipelineWorkspace) -> Stage5Artifacts:
    """Write deterministic Stage 5 artifacts from Stage 4 outputs."""

    stage_4_paths = _stage_4_paths(workspace)
    _require_paths(stage_4_paths.values())

    artifacts = Stage5Artifacts(
        source_imputed_path=workspace.artifact_path(
            "stage_5",
            "source_imputed_stratified_extended_cps_2024.h5",
        ),
        source_imputed_alias_path=workspace.artifact_path(
            "stage_5",
            "source_imputed_stratified_extended_cps.h5",
        ),
        small_enhanced_cps_path=workspace.artifact_path(
            "stage_5",
            "small_enhanced_cps_2024.h5",
        ),
        sparse_enhanced_cps_path=workspace.artifact_path(
            "stage_5",
            "sparse_enhanced_cps_2024.h5",
        ),
    )

    write_tiny_source_imputed_stratified_cps(
        artifacts.source_imputed_path,
        stratified_extended_cps_path=stage_4_paths["stratified"],
    )
    shutil.copy2(artifacts.source_imputed_path, artifacts.source_imputed_alias_path)
    write_tiny_small_enhanced_cps(
        artifacts.small_enhanced_cps_path,
        enhanced_cps_path=stage_4_paths["enhanced"],
    )
    write_tiny_sparse_enhanced_cps(
        artifacts.sparse_enhanced_cps_path,
        enhanced_cps_path=stage_4_paths["enhanced"],
    )

    return artifacts


def write_tiny_source_imputed_stratified_cps(
    path: Path,
    *,
    stratified_extended_cps_path: Path,
) -> None:
    """Create a tiny source-imputed stratified CPS artifact."""

    arrays = _load_period_arrays(stratified_extended_cps_path)
    arrays.update(_source_imputed_person_arrays(arrays))
    arrays.update(_source_imputed_household_arrays(arrays))

    _assert_source_imputed_lengths(arrays)
    _write_period_h5(
        path,
        arrays,
        attrs={
            "fixture_scale": 1,
            "source": "tests.integration.support.tiny_stage_5",
            "source_stage_4_stratified": stratified_extended_cps_path.name,
            "time_period": STAGE_5_PERIOD,
            "stage_5_artifact": "source_imputed_stratified_extended_cps",
        },
    )


def write_tiny_small_enhanced_cps(path: Path, *, enhanced_cps_path: Path) -> None:
    """Create a deterministic tiny subsample of the enhanced CPS artifact."""

    arrays = _load_period_arrays(enhanced_cps_path)
    selected_household_ids = arrays["household_id"].astype(np.int64)[:2]
    subset = _subset_by_household_ids(arrays, selected_household_ids)

    _assert_enhanced_lengths(
        subset, required_variables=SMALL_ENHANCED_REQUIRED_VARIABLES
    )
    _write_period_h5(
        path,
        subset,
        attrs={
            "fixture_scale": 1,
            "source": "tests.integration.support.tiny_stage_5",
            "source_stage_4_enhanced": enhanced_cps_path.name,
            "time_period": STAGE_5_PERIOD,
            "stage_5_artifact": "small_enhanced_cps",
            "selected_household_ids": selected_household_ids,
        },
    )


def write_tiny_sparse_enhanced_cps(path: Path, *, enhanced_cps_path: Path) -> None:
    """Create a sparse enhanced CPS artifact by retaining non-zero-weight rows."""

    arrays = _load_period_arrays(enhanced_cps_path)
    household_ids = arrays["household_id"].astype(np.int64)
    household_weight = arrays["household_weight"].astype(np.float32)
    selected_household_ids = household_ids[
        household_weight >= np.median(household_weight)
    ]
    if len(selected_household_ids) == len(household_ids):
        selected_household_ids = household_ids[np.argsort(household_weight)[-3:]]

    subset = _subset_by_household_ids(arrays, selected_household_ids)
    positive_mask = subset["household_weight"] > 0
    subset = _subset_by_household_ids(subset, subset["household_id"][positive_mask])

    _assert_enhanced_lengths(
        subset,
        required_variables=SPARSE_ENHANCED_REQUIRED_VARIABLES,
    )
    _write_period_h5(
        path,
        subset,
        attrs={
            "fixture_scale": 1,
            "source": "tests.integration.support.tiny_stage_5",
            "source_stage_4_enhanced": enhanced_cps_path.name,
            "time_period": STAGE_5_PERIOD,
            "stage_5_artifact": "sparse_enhanced_cps",
            "selected_household_ids": selected_household_ids,
        },
    )


def _stage_4_paths(workspace: TinyPipelineWorkspace) -> dict[str, Path]:
    return {
        "enhanced": workspace.stage_4 / "enhanced_cps_2024.h5",
        "stratified": workspace.stage_4 / "stratified_extended_cps_2024.h5",
    }


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Stage 4 artifact(s): {missing_list}")


def _load_period_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, mode="r") as h5:
        return {variable: h5[variable][PERIOD_KEY][:] for variable in h5.keys()}


def _source_imputed_person_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    employment_income = arrays["employment_income"].astype(np.float32)
    hours = arrays["weekly_hours_worked"].astype(np.float32)
    household_assets = _source_imputed_household_asset_inputs(arrays)

    return {
        "tip_income": np.where(
            arrays["treasury_tipped_occupation_code"].astype(np.int16) > 0,
            np.round(employment_income * 0.08, 2),
            0,
        ).astype(np.float32),
        "pre_subsidy_rent": _household_values_to_person(
            arrays,
            arrays["rent"].astype(np.float32),
        ),
        "bank_account_assets": _household_values_to_person(
            arrays,
            household_assets["bank_account_assets"],
        ),
        "stock_assets": _household_values_to_person(
            arrays,
            household_assets["stock_assets"],
        ),
        "bond_assets": _household_values_to_person(
            arrays,
            household_assets["bond_assets"],
        ),
        "is_paid_hourly": hours > 0,
    }


def _source_imputed_household_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    household_assets = _source_imputed_household_asset_inputs(arrays)
    bank_account_assets = household_assets["bank_account_assets"]
    stock_assets = household_assets["stock_assets"]
    bond_assets = household_assets["bond_assets"]
    vehicles_owned = arrays["household_vehicles_owned"].astype(np.float32)
    vehicle_value = np.round(vehicles_owned * 8_000, 2)
    auto_loan_balance = np.round(vehicles_owned * 2_500, 2)

    return {
        "household_vehicles_value": vehicle_value.astype(np.float32),
        "net_worth": (
            bank_account_assets
            + stock_assets
            + bond_assets
            + vehicle_value
            - auto_loan_balance
        ).astype(np.float32),
        "auto_loan_balance": auto_loan_balance.astype(np.float32),
        "auto_loan_interest": np.round(auto_loan_balance * 0.07, 2).astype(np.float32),
    }


def _source_imputed_household_asset_inputs(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    income = arrays["spm_unit_total_income_reported"].astype(np.float32)
    return {
        "bank_account_assets": np.round(np.maximum(income * 0.06, 250), 2).astype(
            np.float32
        ),
        "stock_assets": np.round(
            np.where(income > 80_000, income * 0.35, income * 0.05),
            2,
        ).astype(np.float32),
        "bond_assets": np.round(np.where(income > 50_000, income * 0.03, 0), 2).astype(
            np.float32
        ),
    }


def _household_values_to_person(
    arrays: dict[str, np.ndarray],
    household_values: np.ndarray,
) -> np.ndarray:
    household_id_to_value = dict(zip(arrays["household_id"], household_values))
    return np.array(
        [
            household_id_to_value[household_id]
            for household_id in arrays["person_household_id"]
        ],
        dtype=np.asarray(household_values).dtype,
    )


def _subset_by_household_ids(
    arrays: dict[str, np.ndarray],
    household_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    source_household_ids = arrays["household_id"]
    household_mask = np.isin(source_household_ids, household_ids)
    person_mask = np.isin(arrays["person_household_id"], household_ids)
    person_count = len(arrays["person_id"])
    household_count = len(source_household_ids)

    subset = {}
    for variable, values in arrays.items():
        if len(values) == person_count:
            subset[variable] = values[person_mask]
        elif len(values) == household_count:
            subset[variable] = values[household_mask]
        else:
            raise ValueError(f"Cannot infer entity level for {variable}")
    return subset


def _resize_pattern(values: list[object], length: int, *, dtype) -> np.ndarray:
    repeats = int(np.ceil(length / len(values)))
    return np.resize(np.array(values * repeats, dtype=dtype), length)


def _write_period_h5(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    attrs: dict[str, object],
) -> None:
    with h5py.File(path, mode="w") as h5:
        for key, value in attrs.items():
            h5.attrs[key] = value
        for variable in sorted(arrays):
            h5.create_group(variable).create_dataset(PERIOD_KEY, data=arrays[variable])


def _assert_source_imputed_lengths(arrays: dict[str, np.ndarray]) -> None:
    person_count = len(arrays["person_id"])
    household_count = len(arrays["household_id"])
    for variable in SOURCE_IMPUTED_REQUIRED_VARIABLES:
        length = len(arrays[variable])
        assert length in {person_count, household_count}, variable


def _assert_enhanced_lengths(
    arrays: dict[str, np.ndarray],
    *,
    required_variables: tuple[str, ...],
) -> None:
    person_count = len(arrays["person_id"])
    household_count = len(arrays["household_id"])
    for variable in required_variables:
        length = len(arrays[variable])
        assert length in {person_count, household_count}, variable
