"""Fixture-backed Stage 4 artifacts for tiny pipeline integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_stage_2 import PERIOD_KEY
from tests.integration.support.tiny_stage_3 import (
    EXTENDED_CPS_REQUIRED_VARIABLES,
    STAGE_3_GROUP_VARIABLES,
    STAGE_3_PERSON_VARIABLES,
    STAGE_3_PERIOD,
)

__test__ = False


STAGE_4_PERIOD = STAGE_3_PERIOD

ENHANCED_PERSON_VARIABLES = tuple(
    dict.fromkeys(
        (
            *STAGE_3_PERSON_VARIABLES,
            "tip_income",
            "ssn_card_type",
            "immigration_status_str",
            "taxpayer_id_type",
            "has_tin",
            "has_itin",
            "has_valid_ssn",
        )
    )
)

ENHANCED_GROUP_VARIABLES = tuple(
    dict.fromkeys((*STAGE_3_GROUP_VARIABLES, "takes_up_aca_if_eligible"))
)

ENHANCED_CPS_REQUIRED_VARIABLES = tuple(
    dict.fromkeys((*ENHANCED_PERSON_VARIABLES, *ENHANCED_GROUP_VARIABLES))
)
STRATIFIED_CPS_REQUIRED_VARIABLES = EXTENDED_CPS_REQUIRED_VARIABLES


@dataclass(frozen=True)
class Stage4Artifacts:
    """Paths written by the fixture-backed Stage 4 builder."""

    enhanced_cps_path: Path
    stratified_extended_cps_path: Path

    def as_tuple(self) -> tuple[Path, Path]:
        return (self.enhanced_cps_path, self.stratified_extended_cps_path)


def create_stage_4_artifacts(workspace: TinyPipelineWorkspace) -> Stage4Artifacts:
    """Write deterministic enhanced and stratified CPS artifacts."""

    extended_cps_path = workspace.stage_3 / "extended_cps_2024.h5"
    _require_paths((extended_cps_path,))

    artifacts = Stage4Artifacts(
        enhanced_cps_path=workspace.artifact_path("stage_4", "enhanced_cps_2024.h5"),
        stratified_extended_cps_path=workspace.artifact_path(
            "stage_4",
            "stratified_extended_cps_2024.h5",
        ),
    )

    write_tiny_enhanced_cps(
        artifacts.enhanced_cps_path,
        extended_cps_path=extended_cps_path,
    )
    write_tiny_stratified_extended_cps(
        artifacts.stratified_extended_cps_path,
        extended_cps_path=extended_cps_path,
    )

    return artifacts


def write_tiny_enhanced_cps(path: Path, *, extended_cps_path: Path) -> None:
    """Create a tiny enhanced CPS artifact without running calibration."""

    arrays = _load_period_arrays(extended_cps_path)
    arrays["household_weight"] = _calibrated_household_weights(arrays)
    arrays.update(_enhanced_person_arrays(arrays))
    arrays.update(_enhanced_group_arrays(arrays))

    _assert_enhanced_lengths(arrays)
    _write_period_h5(
        path,
        arrays,
        attrs={
            "fixture_scale": 1,
            "source": "tests.integration.support.tiny_stage_4",
            "source_stage_3_extended_cps": extended_cps_path.name,
            "time_period": STAGE_4_PERIOD,
            "stage_4_artifact": "enhanced_cps",
        },
    )


def write_tiny_stratified_extended_cps(
    path: Path,
    *,
    extended_cps_path: Path,
) -> None:
    """Create a tiny stratified extended CPS subset from Stage 3 output."""

    arrays = _load_period_arrays(extended_cps_path)
    selected_household_ids = _select_representative_household_ids(arrays)
    stratified = _subset_by_household_ids(arrays, selected_household_ids)

    _assert_stratified_lengths(stratified)
    _write_period_h5(
        path,
        stratified,
        attrs={
            "fixture_scale": 1,
            "source": "tests.integration.support.tiny_stage_4",
            "source_stage_3_extended_cps": extended_cps_path.name,
            "time_period": STAGE_4_PERIOD,
            "stage_4_artifact": "stratified_extended_cps",
            "selected_household_ids": selected_household_ids,
        },
    )


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Stage 3 artifact(s): {missing_list}")


def _load_period_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, mode="r") as h5:
        return {variable: h5[variable][PERIOD_KEY][:] for variable in h5.keys()}


def _calibrated_household_weights(arrays: dict[str, np.ndarray]) -> np.ndarray:
    weights = arrays["household_weight"].astype(np.float32)
    income = arrays["spm_unit_total_income_reported"].astype(np.float32)
    income_rank = np.argsort(np.argsort(income)).astype(np.float32)
    center = income_rank.mean()
    scale = 1.0 + (income_rank - center) * 0.04
    return np.round(weights * scale, 2).astype(np.float32)


def _enhanced_person_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    person_count = len(arrays["person_id"])
    ssn_card_type = _resize_pattern(
        [
            b"CITIZEN",
            b"CITIZEN",
            b"NONE",
            b"NON_CITIZEN_VALID_EAD",
            b"OTHER_NON_CITIZEN",
            b"CITIZEN",
        ],
        person_count,
        dtype="S32",
    )
    has_valid_ssn = ssn_card_type == b"CITIZEN"
    has_tin = has_valid_ssn | (ssn_card_type == b"OTHER_NON_CITIZEN")

    return {
        "tip_income": np.where(
            arrays["treasury_tipped_occupation_code"].astype(np.int16) > 0,
            arrays["employment_income"].astype(np.float32) * 0.08,
            0,
        ).astype(np.float32),
        "ssn_card_type": ssn_card_type,
        "immigration_status_str": np.where(
            ssn_card_type == b"NONE",
            b"UNDOCUMENTED",
            b"CITIZEN",
        ).astype("S32"),
        "taxpayer_id_type": np.where(
            has_valid_ssn,
            b"VALID_SSN",
            np.where(has_tin, b"OTHER_TIN", b"NONE"),
        ).astype("S16"),
        "has_tin": has_tin.astype(np.bool_),
        "has_itin": has_tin.astype(np.bool_),
        "has_valid_ssn": has_valid_ssn.astype(np.bool_),
    }


def _enhanced_group_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    group_count = len(arrays["household_id"])
    return {
        "takes_up_aca_if_eligible": _resize_pattern(
            [True, False, True, True, False],
            group_count,
            dtype=np.bool_,
        )
    }


def _select_representative_household_ids(
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    household_ids = arrays["household_id"].astype(np.int64)
    income = arrays["spm_unit_total_income_reported"].astype(np.float32)
    ordered = household_ids[np.argsort(income)]
    candidates = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]

    is_puf = arrays["household_is_puf_clone"].astype(bool)
    if not np.isin(household_ids[is_puf], candidates).any():
        candidates.append(household_ids[is_puf][0])
    if not np.isin(household_ids[~is_puf], candidates).any():
        candidates.append(household_ids[~is_puf][0])

    selected = np.array(list(dict.fromkeys(int(value) for value in candidates)))
    return selected.astype(np.int64)


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


def _assert_enhanced_lengths(arrays: dict[str, np.ndarray]) -> None:
    person_count = len(arrays["person_id"])
    household_count = len(arrays["household_id"])
    for variable in ENHANCED_PERSON_VARIABLES:
        assert len(arrays[variable]) == person_count, variable
    for variable in ENHANCED_GROUP_VARIABLES:
        assert len(arrays[variable]) == household_count, variable


def _assert_stratified_lengths(arrays: dict[str, np.ndarray]) -> None:
    person_count = len(arrays["person_id"])
    household_count = len(arrays["household_id"])
    for variable in STAGE_3_PERSON_VARIABLES:
        assert len(arrays[variable]) == person_count, variable
    for variable in STAGE_3_GROUP_VARIABLES:
        assert len(arrays[variable]) == household_count, variable
