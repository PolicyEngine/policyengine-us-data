"""Fixture-backed Stage 2 artifacts for tiny pipeline integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace

__test__ = False


STAGE_2_PERIOD = 2024
PERIOD_KEY = str(STAGE_2_PERIOD)

ID_VARIABLES = (
    "person_id",
    "tax_unit_id",
    "marital_unit_id",
    "spm_unit_id",
    "family_id",
    "household_id",
    "person_tax_unit_id",
    "person_marital_unit_id",
    "person_spm_unit_id",
    "person_family_id",
    "person_household_id",
)

PERSON_LEVEL_VARIABLES = (
    "person_id",
    "person_tax_unit_id",
    "person_marital_unit_id",
    "person_spm_unit_id",
    "person_family_id",
    "person_household_id",
    "age",
    "is_male",
    "employment_income",
    "self_employment_income",
    "social_security",
    "taxable_private_pension_income",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "rent",
    "real_estate_taxes",
    "primary_residence_value",
    "deductible_mortgage_interest",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
)

GROUP_LEVEL_VARIABLES = (
    "tax_unit_id",
    "marital_unit_id",
    "spm_unit_id",
    "family_id",
    "household_id",
    "household_weight",
    "filing_status",
    "state_fips",
    "household_state_fips",
    "tenure_type",
    "household_vehicles_owned",
)

CPS_REQUIRED_VARIABLES = tuple(
    dict.fromkeys((*ID_VARIABLES, *PERSON_LEVEL_VARIABLES, *GROUP_LEVEL_VARIABLES))
)
PUF_REQUIRED_VARIABLES = CPS_REQUIRED_VARIABLES


@dataclass(frozen=True)
class Stage2Artifacts:
    """Paths written by the fixture-backed Stage 2 builder."""

    cps_path: Path
    puf_path: Path

    def as_tuple(self) -> tuple[Path, Path]:
        return (self.cps_path, self.puf_path)


def create_stage_2_artifacts(workspace: TinyPipelineWorkspace) -> Stage2Artifacts:
    """Write deterministic CPS and PUF artifacts from Stage 1 inputs."""

    stage_1 = _stage_1_paths(workspace)
    _require_paths(stage_1.values())

    artifacts = Stage2Artifacts(
        cps_path=workspace.artifact_path("stage_2", "cps_2024.h5"),
        puf_path=workspace.artifact_path("stage_2", "puf_2024.h5"),
    )

    uprating = pd.read_csv(stage_1["uprating"], index_col="Variable")
    write_tiny_cps(
        artifacts.cps_path,
        acs_path=stage_1["acs"],
        uprating=uprating,
    )
    write_tiny_puf(
        artifacts.puf_path,
        irs_puf_path=stage_1["irs_puf"],
        uprating=uprating,
    )

    return artifacts


def write_tiny_cps(
    path: Path,
    *,
    acs_path: Path,
    uprating: pd.DataFrame,
) -> None:
    """Create a tiny CPS-like array dataset from the tiny ACS artifact."""

    with h5py.File(acs_path, mode="r") as acs:
        employment_growth = _uprating_factor(uprating, "employment_income")
        self_employment_growth = _uprating_factor(uprating, "self_employment_income")
        social_security_growth = _uprating_factor(uprating, "social_security")
        weight_growth = _uprating_factor(uprating, "household_weight")

        person_household_id = acs["person_household_id"][:].astype(np.int64)
        household_id = acs["household_id"][:].astype(np.int64)
        person_count = len(person_household_id)
        household_count = len(household_id)

        arrays = {
            "person_id": acs["person_id"][:].astype(np.int64),
            "tax_unit_id": household_id,
            "marital_unit_id": household_id,
            "spm_unit_id": household_id,
            "family_id": household_id,
            "household_id": household_id,
            "person_tax_unit_id": person_household_id,
            "person_marital_unit_id": person_household_id,
            "person_spm_unit_id": person_household_id,
            "person_family_id": person_household_id,
            "person_household_id": person_household_id,
            "household_weight": acs["household_weight"][:] * weight_growth,
            "age": acs["age"][:],
            "is_male": acs["is_male"][:],
            "employment_income": acs["employment_income"][:] * employment_growth,
            "self_employment_income": (
                acs["self_employment_income"][:] * self_employment_growth
            ),
            "social_security": acs["social_security"][:] * social_security_growth,
            "taxable_private_pension_income": acs["taxable_private_pension_income"][:],
            "taxable_interest_income": np.array([100, 50, 0], dtype=np.float32),
            "tax_exempt_interest_income": np.array([25, 0, 0], dtype=np.float32),
            "qualified_dividend_income": np.array([40, 10, 0], dtype=np.float32),
            "non_qualified_dividend_income": np.array([10, 5, 0], dtype=np.float32),
            "rent": acs["rent"][:],
            "real_estate_taxes": acs["real_estate_taxes"][:],
            "primary_residence_value": acs["primary_residence_value"][:],
            "deductible_mortgage_interest": np.array([1_800, 0, 0], dtype=np.float32),
            "is_tax_unit_head": np.array([True, False, True], dtype=np.bool_),
            "is_tax_unit_spouse": np.array([False, True, False], dtype=np.bool_),
            "is_tax_unit_dependent": np.array([False, False, True], dtype=np.bool_),
            "filing_status": np.array([b"JOINT", b"HEAD_OF_HOUSEHOLD"]),
            "state_fips": acs["state_fips"][:].astype(np.int32),
            "household_state_fips": acs["household_state_fips"][:].astype(np.int32),
            "tenure_type": acs["tenure_type"][:],
            "household_vehicles_owned": acs["household_vehicles_owned"][:],
        }

    _assert_lengths(
        arrays,
        person_count=person_count,
        household_count=household_count,
    )
    _write_period_h5(
        path,
        arrays,
        attrs={
            "fixture_scale": 1,
            "source": "tests.integration.support.tiny_stage_2",
            "source_stage_1_acs": acs_path.name,
            "time_period": STAGE_2_PERIOD,
        },
    )


def write_tiny_puf(
    path: Path,
    *,
    irs_puf_path: Path,
    uprating: pd.DataFrame,
) -> None:
    """Create a tiny PUF-like array dataset from the tiny raw IRS PUF."""

    with pd.HDFStore(irs_puf_path, mode="r") as store:
        puf = store["puf"]
        demographics = store["puf_demographics"]

    raw = puf.merge(demographics, on="RECID", validate="one_to_one")
    record_id = raw["RECID"].to_numpy(dtype=np.int64)
    person_id = record_id * 100 + 1
    employment_growth = _uprating_factor(uprating, "employment_income")
    weight_growth = _uprating_factor(uprating, "household_weight")
    person_count = len(raw)
    household_count = len(raw)

    arrays = {
        "person_id": person_id,
        "tax_unit_id": record_id,
        "marital_unit_id": person_id,
        "spm_unit_id": record_id,
        "family_id": record_id,
        "household_id": record_id,
        "person_tax_unit_id": record_id,
        "person_marital_unit_id": person_id,
        "person_spm_unit_id": record_id,
        "person_family_id": record_id,
        "person_household_id": record_id,
        "household_weight": raw["S006"].to_numpy(dtype=np.float32)
        / 100
        * weight_growth,
        "age": _decode_age_range(raw["AGERANGE"].to_numpy(dtype=np.int16)),
        "is_male": raw["GENDER"].to_numpy(dtype=np.int16) == 1,
        "employment_income": raw["E00200"].to_numpy(dtype=np.float32)
        * employment_growth,
        "self_employment_income": raw["E00900"].to_numpy(dtype=np.float32),
        "social_security": raw["E02400"].to_numpy(dtype=np.float32),
        "taxable_private_pension_income": raw["E01700"].to_numpy(dtype=np.float32),
        "taxable_interest_income": raw["E00300"].to_numpy(dtype=np.float32),
        "tax_exempt_interest_income": raw["E00400"].to_numpy(dtype=np.float32),
        "qualified_dividend_income": raw["E00650"].to_numpy(dtype=np.float32),
        "non_qualified_dividend_income": (
            raw["E00600"].to_numpy(dtype=np.float32)
            - raw["E00650"].to_numpy(dtype=np.float32)
        ),
        "rent": np.zeros(person_count, dtype=np.float32),
        "real_estate_taxes": raw["E18500"].to_numpy(dtype=np.float32),
        "primary_residence_value": np.zeros(person_count, dtype=np.float32),
        "deductible_mortgage_interest": raw["E19200"].to_numpy(dtype=np.float32),
        "is_tax_unit_head": np.ones(person_count, dtype=np.bool_),
        "is_tax_unit_spouse": np.zeros(person_count, dtype=np.bool_),
        "is_tax_unit_dependent": np.zeros(person_count, dtype=np.bool_),
        "filing_status": _filing_status(raw["MARS"].to_numpy(dtype=np.int16)),
        "state_fips": np.array([37, 6, 48], dtype=np.int32),
        "household_state_fips": np.array([37, 6, 48], dtype=np.int32),
        "tenure_type": np.array([b"OWNED_WITH_MORTGAGE", b"RENTED", b"NONE"]),
        "household_vehicles_owned": np.array([2, 1, 0], dtype=np.int16),
    }

    _assert_lengths(
        arrays,
        person_count=person_count,
        household_count=household_count,
    )
    _write_period_h5(
        path,
        arrays,
        attrs={
            "fixture_scale": 1,
            "source": "tests.integration.support.tiny_stage_2",
            "source_stage_1_irs_puf": irs_puf_path.name,
            "time_period": STAGE_2_PERIOD,
        },
    )


def _stage_1_paths(workspace: TinyPipelineWorkspace) -> dict[str, Path]:
    return {
        "uprating": workspace.stage_1 / "uprating_factors.csv",
        "acs": workspace.stage_1 / "acs_2022.h5",
        "irs_puf": workspace.stage_1 / "irs_puf_2015.h5",
    }


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Stage 1 artifact(s): {missing_list}")


def _uprating_factor(uprating: pd.DataFrame, variable: str) -> float:
    return float(uprating.loc[variable, PERIOD_KEY])


def _write_period_h5(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    attrs: dict[str, object],
) -> None:
    with h5py.File(path, mode="w") as h5:
        for key, value in attrs.items():
            h5.attrs[key] = value
        for variable, values in arrays.items():
            h5.create_group(variable).create_dataset(PERIOD_KEY, data=values)


def _assert_lengths(
    arrays: dict[str, np.ndarray],
    *,
    person_count: int,
    household_count: int,
) -> None:
    for variable in PERSON_LEVEL_VARIABLES:
        assert len(arrays[variable]) == person_count, variable
    for variable in GROUP_LEVEL_VARIABLES:
        assert len(arrays[variable]) == household_count, variable


def _decode_age_range(age_range: np.ndarray) -> np.ndarray:
    age_by_range = {
        1: 18,
        2: 26,
        3: 35,
        4: 45,
        5: 55,
        6: 65,
        7: 80,
    }
    return np.array([age_by_range.get(int(value), 40) for value in age_range])


def _filing_status(mars: np.ndarray) -> np.ndarray:
    status_by_mars = {
        1: b"SINGLE",
        2: b"JOINT",
        3: b"SEPARATE",
        4: b"HEAD_OF_HOUSEHOLD",
    }
    return np.array([status_by_mars[int(value)] for value in mars])
