"""Fixture-backed Stage 3 artifacts for tiny pipeline integration tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace
from tests.integration.support.tiny_stage_2 import (
    GROUP_LEVEL_VARIABLES,
    PERIOD_KEY,
    PERSON_LEVEL_VARIABLES,
    STAGE_2_PERIOD,
)

__test__ = False


STAGE_3_PERIOD = STAGE_2_PERIOD

STAGE_3_PERSON_VARIABLES = tuple(
    dict.fromkeys(
        (
            *PERSON_LEVEL_VARIABLES,
            "employment_income_before_lsr",
            "pre_tax_contributions",
            "weekly_hours_worked",
            "hours_worked_last_week",
            "is_hispanic",
            "cps_race",
            "detailed_occupation_recode",
            "treasury_tipped_occupation_code",
            "tanf_reported",
            "ssi_reported",
            "is_puf_clone",
        )
    )
)

STAGE_3_GROUP_VARIABLES = tuple(
    dict.fromkeys(
        (
            *GROUP_LEVEL_VARIABLES,
            "tax_unit_count_dependents",
            "tax_unit_is_joint",
            "spm_unit_total_income_reported",
            "spm_unit_net_income_reported",
            "spm_unit_capped_housing_subsidy_reported",
            "snap_reported",
            "household_is_puf_clone",
        )
    )
)

EXTENDED_CPS_REQUIRED_VARIABLES = tuple(
    dict.fromkeys((*STAGE_3_PERSON_VARIABLES, *STAGE_3_GROUP_VARIABLES))
)

STAGE_4_INPUT_VARIABLES = (
    "person_id",
    "household_id",
    "person_household_id",
    "household_weight",
    "employment_income",
    "employment_income_before_lsr",
    "self_employment_income",
    "social_security",
    "taxable_private_pension_income",
    "state_fips",
    "tax_unit_count_dependents",
    "tax_unit_is_joint",
    "spm_unit_total_income_reported",
    "spm_unit_net_income_reported",
    "is_puf_clone",
)


@dataclass(frozen=True)
class Stage3Artifacts:
    """Paths written by the fixture-backed Stage 3 builder."""

    extended_cps_path: Path

    def as_tuple(self) -> tuple[Path]:
        return (self.extended_cps_path,)


def create_stage_3_artifacts(workspace: TinyPipelineWorkspace) -> Stage3Artifacts:
    """Write deterministic extended CPS artifact from Stage 2 outputs."""

    stage_2 = _stage_2_paths(workspace)
    _require_paths(stage_2.values())

    artifacts = Stage3Artifacts(
        extended_cps_path=workspace.artifact_path("stage_3", "extended_cps_2024.h5")
    )
    write_tiny_extended_cps(
        artifacts.extended_cps_path,
        cps_path=stage_2["cps"],
        puf_path=stage_2["puf"],
    )

    return artifacts


def write_tiny_extended_cps(
    path: Path,
    *,
    cps_path: Path,
    puf_path: Path,
) -> None:
    """Create a tiny extended CPS by appending PUF clone rows to CPS rows."""

    cps = _load_period_arrays(cps_path)
    puf = _load_period_arrays(puf_path)

    arrays = _concatenate_common_arrays(cps, puf)
    arrays.update(
        _extended_person_arrays(arrays, cps_person_count=len(cps["person_id"]))
    )
    arrays.update(
        _extended_group_arrays(
            arrays,
            cps_household_count=len(cps["household_id"]),
        )
    )

    _assert_lengths(arrays)
    _write_period_h5(
        path,
        arrays,
        attrs={
            "fixture_scale": 1,
            "source": "tests.integration.support.tiny_stage_3",
            "source_stage_2_cps": cps_path.name,
            "source_stage_2_puf": puf_path.name,
            "time_period": STAGE_3_PERIOD,
        },
    )


def stage_3_artifact_digest(path: Path) -> str:
    """Return a deterministic content digest for a Stage 3 H5 artifact."""

    digest = hashlib.sha256()
    with h5py.File(path, mode="r") as h5:
        for key in sorted(h5.attrs):
            digest.update(str(key).encode("utf-8"))
            digest.update(str(h5.attrs[key]).encode("utf-8"))
        for variable in sorted(h5.keys()):
            values = h5[variable][PERIOD_KEY][:]
            digest.update(variable.encode("utf-8"))
            digest.update(str(values.dtype).encode("utf-8"))
            digest.update(str(values.shape).encode("utf-8"))
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _stage_2_paths(workspace: TinyPipelineWorkspace) -> dict[str, Path]:
    return {
        "cps": workspace.stage_2 / "cps_2024.h5",
        "puf": workspace.stage_2 / "puf_2024.h5",
    }


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Stage 2 artifact(s): {missing_list}")


def _load_period_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, mode="r") as h5:
        return {variable: h5[variable][PERIOD_KEY][:] for variable in h5.keys()}


def _concatenate_common_arrays(
    cps: dict[str, np.ndarray],
    puf: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        variable: np.concatenate([cps[variable], puf[variable]])
        for variable in sorted(set(cps) & set(puf))
    }


def _extended_person_arrays(
    arrays: dict[str, np.ndarray],
    *,
    cps_person_count: int,
) -> dict[str, np.ndarray]:
    person_count = len(arrays["person_id"])
    puf_person_count = person_count - cps_person_count
    employment_income = arrays["employment_income"].astype(np.float32)

    return {
        "employment_income_before_lsr": employment_income.copy(),
        "pre_tax_contributions": np.round(employment_income * 0.04, 2).astype(
            np.float32
        ),
        "weekly_hours_worked": np.where(employment_income > 0, 40, 0).astype(np.int16),
        "hours_worked_last_week": np.where(employment_income > 0, 40, 0).astype(
            np.int16
        ),
        "is_hispanic": _resize_pattern(
            [False, True, False, False, False, True],
            person_count,
            dtype=np.bool_,
        ),
        "cps_race": _resize_pattern([1, 2, 1, 1, 1, 2], person_count, dtype=np.int16),
        "detailed_occupation_recode": _resize_pattern(
            [10, 20, 0, 30, 20, 10],
            person_count,
            dtype=np.int16,
        ),
        "treasury_tipped_occupation_code": _resize_pattern(
            [0, 1, 0, 0, 0, 1],
            person_count,
            dtype=np.int16,
        ),
        "tanf_reported": np.zeros(person_count, dtype=np.float32),
        "ssi_reported": np.zeros(person_count, dtype=np.float32),
        "is_puf_clone": np.concatenate(
            [
                np.zeros(cps_person_count, dtype=np.bool_),
                np.ones(puf_person_count, dtype=np.bool_),
            ]
        ),
    }


def _extended_group_arrays(
    arrays: dict[str, np.ndarray],
    *,
    cps_household_count: int,
) -> dict[str, np.ndarray]:
    household_count = len(arrays["household_id"])
    puf_household_count = household_count - cps_household_count
    tax_unit_count_dependents = _count_dependents_by_tax_unit(arrays)
    total_income = _sum_person_values_by_group(
        group_ids=arrays["spm_unit_id"],
        person_group_ids=arrays["person_spm_unit_id"],
        person_values=(
            arrays["employment_income"].astype(np.float32)
            + arrays["self_employment_income"].astype(np.float32)
            + arrays["social_security"].astype(np.float32)
        ),
    )

    return {
        "tax_unit_count_dependents": tax_unit_count_dependents,
        "tax_unit_is_joint": arrays["filing_status"] == b"JOINT",
        "spm_unit_total_income_reported": total_income.astype(np.float32),
        "spm_unit_net_income_reported": np.round(total_income * 0.85, 2).astype(
            np.float32
        ),
        "spm_unit_capped_housing_subsidy_reported": np.where(
            arrays["tenure_type"] == b"RENTED",
            1_200,
            0,
        ).astype(np.float32),
        "snap_reported": np.where(total_income < 50_000, 1_000, 0).astype(np.float32),
        "household_is_puf_clone": np.concatenate(
            [
                np.zeros(cps_household_count, dtype=np.bool_),
                np.ones(puf_household_count, dtype=np.bool_),
            ]
        ),
    }


def _count_dependents_by_tax_unit(arrays: dict[str, np.ndarray]) -> np.ndarray:
    dependents = arrays["is_tax_unit_dependent"].astype(bool)
    return np.array(
        [
            dependents[arrays["person_tax_unit_id"] == tax_unit_id].sum()
            for tax_unit_id in arrays["tax_unit_id"]
        ],
        dtype=np.int16,
    )


def _sum_person_values_by_group(
    *,
    group_ids: np.ndarray,
    person_group_ids: np.ndarray,
    person_values: np.ndarray,
) -> np.ndarray:
    return np.array(
        [person_values[person_group_ids == group_id].sum() for group_id in group_ids],
        dtype=np.float32,
    )


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


def _assert_lengths(arrays: dict[str, np.ndarray]) -> None:
    person_count = len(arrays["person_id"])
    household_count = len(arrays["household_id"])
    for variable in STAGE_3_PERSON_VARIABLES:
        assert len(arrays[variable]) == person_count, variable
    for variable in STAGE_3_GROUP_VARIABLES:
        assert len(arrays[variable]) == household_count, variable
