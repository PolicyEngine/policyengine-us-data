"""Fixture-backed Stage 1 artifacts for tiny pipeline integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tests.integration.support.pipeline_workspace import TinyPipelineWorkspace

__test__ = False


UPRATING_YEARS = tuple(range(2020, 2035))
UPRATING_VARIABLES = (
    "employment_income",
    "self_employment_income",
    "social_security",
    "household_weight",
    "population",
)

ACS_PERSON_ARRAYS = (
    "person_id",
    "person_household_id",
    "person_spm_unit_id",
    "person_tax_unit_id",
    "person_family_id",
    "person_marital_unit_id",
    "age",
    "is_male",
    "employment_income",
    "self_employment_income",
    "social_security",
    "taxable_private_pension_income",
    "is_household_head",
    "rent",
    "real_estate_taxes",
)

ACS_HOUSEHOLD_ARRAYS = (
    "household_id",
    "spm_unit_id",
    "tax_unit_id",
    "family_id",
    "marital_unit_id",
    "household_weight",
    "tenure_type",
    "household_vehicles_owned",
    "state_fips",
    "household_state_fips",
)

PUF_CORE_COLUMNS = (
    "RECID",
    "S006",
    "MARS",
    "DSI",
    "EIC",
    "XTOT",
    "E00200",
)

PUF_DEMOGRAPHIC_COLUMNS = (
    "RECID",
    "AGEDP1",
    "AGEDP2",
    "AGEDP3",
    "AGERANGE",
    "EARNSPLIT",
    "GENDER",
)

_PUF_ZERO_COLUMNS = (
    "E00300",
    "E00400",
    "E00600",
    "E00650",
    "E00700",
    "E00800",
    "E00900",
    "E01100",
    "E01200",
    "E01400",
    "E01500",
    "E01700",
    "E02100",
    "E02300",
    "E02400",
    "E03150",
    "E03210",
    "E03220",
    "E03230",
    "E03240",
    "E03270",
    "E03290",
    "E03300",
    "E03400",
    "E03500",
    "E07240",
    "E07260",
    "E07300",
    "E07400",
    "E07600",
    "E09700",
    "E09800",
    "E09900",
    "E11200",
    "E17500",
    "E18400",
    "E18500",
    "E19200",
    "E19800",
    "E20100",
    "E20400",
    "E20500",
    "E24515",
    "E24518",
    "E25850",
    "E25860",
    "E25920",
    "E25940",
    "E25960",
    "E25980",
    "E26180",
    "E26190",
    "E26390",
    "E26400",
    "E27200",
    "E30400",
    "E30500",
    "E32800",
    "E58990",
    "E62900",
    "E87521",
    "P08000",
    "P22250",
    "P23250",
    "T27800",
)


@dataclass(frozen=True)
class Stage1Artifacts:
    """Paths written by the fixture-backed Stage 1 builder."""

    uprating_factors_path: Path
    acs_path: Path
    irs_puf_path: Path

    def as_tuple(self) -> tuple[Path, Path, Path]:
        return (
            self.uprating_factors_path,
            self.acs_path,
            self.irs_puf_path,
        )


def create_stage_1_artifacts(workspace: TinyPipelineWorkspace) -> Stage1Artifacts:
    """Write deterministic Stage 1 artifacts into ``workspace``."""

    artifacts = Stage1Artifacts(
        uprating_factors_path=workspace.artifact_path(
            "stage_1", "uprating_factors.csv"
        ),
        acs_path=workspace.artifact_path("stage_1", "acs_2022.h5"),
        irs_puf_path=workspace.artifact_path("stage_1", "irs_puf_2015.h5"),
    )

    write_tiny_uprating_factors(artifacts.uprating_factors_path)
    write_tiny_acs(artifacts.acs_path)
    write_tiny_irs_puf(artifacts.irs_puf_path)

    return artifacts


def write_tiny_uprating_factors(path: Path) -> None:
    """Write a production-shaped uprating factor table with tiny values."""

    year_offsets = np.array(UPRATING_YEARS) - UPRATING_YEARS[0]
    growth_rates = {
        "employment_income": 0.030,
        "self_employment_income": 0.025,
        "social_security": 0.020,
        "household_weight": 0.010,
        "population": 0.005,
    }
    table = pd.DataFrame(
        {
            year: {
                variable: round(1 + growth_rates[variable] * offset, 3)
                for variable in UPRATING_VARIABLES
            }
            for year, offset in zip(UPRATING_YEARS, year_offsets)
        }
    )
    table.index.name = "Variable"
    table.to_csv(path)


def write_tiny_acs(path: Path) -> None:
    """Write a minimal ACS array H5 compatible with Stage 1 contracts."""

    person_household_id = np.array([1, 1, 2], dtype=np.int64)
    household_id = np.array([1, 2], dtype=np.int64)

    arrays = {
        "person_id": np.array([1, 2, 3], dtype=np.int64),
        "household_id": household_id,
        "spm_unit_id": household_id,
        "tax_unit_id": household_id,
        "family_id": household_id,
        "marital_unit_id": household_id,
        "person_household_id": person_household_id,
        "person_spm_unit_id": person_household_id,
        "person_tax_unit_id": person_household_id,
        "person_family_id": person_household_id,
        "person_marital_unit_id": person_household_id,
        "household_weight": np.array([120.0, 80.0], dtype=np.float32),
        "age": np.array([40, 38, 10], dtype=np.int16),
        "is_male": np.array([True, False, True], dtype=np.bool_),
        "employment_income": np.array([55_000, 35_000, 0], dtype=np.float32),
        "self_employment_income": np.array([0, 5_000, 0], dtype=np.float32),
        "social_security": np.array([0, 0, 0], dtype=np.float32),
        "taxable_private_pension_income": np.array([0, 0, 0], dtype=np.float32),
        "is_household_head": np.array([True, False, True], dtype=np.bool_),
        "rent": np.array([0, 0, 14_400], dtype=np.float32),
        "real_estate_taxes": np.array([2_400, 0, 0], dtype=np.float32),
        "tenure_type": np.array([b"OWNED_WITH_MORTGAGE", b"RENTED"]),
        "household_vehicles_owned": np.array([2, 1], dtype=np.int16),
        "state_fips": np.array([37, 37], dtype=np.int16),
        "household_state_fips": np.array([37, 37], dtype=np.int16),
    }

    with h5py.File(path, mode="w") as h5:
        h5.attrs["fixture_scale"] = 1
        h5.attrs["source"] = "tests.integration.support.tiny_stage_1"
        for name, values in arrays.items():
            h5.create_dataset(name, data=values)


def write_tiny_irs_puf(path: Path) -> None:
    """Write minimal raw IRS PUF tables with production table names."""

    puf = pd.DataFrame(
        {
            "RECID": [1001, 1002, 1003],
            "S006": [12_000, 8_000, 5_000],
            "MARS": [2, 1, 4],
            "DSI": [0, 0, 0],
            "EIC": [0, 1, 0],
            "XTOT": [2, 1, 3],
            "E00200": [90_000, 45_000, 30_000],
            **{column: [0, 0, 0] for column in _PUF_ZERO_COLUMNS},
        }
    )
    demographics = pd.DataFrame(
        {
            "RECID": [1001, 1002, 1003],
            "AGEDP1": [0, 0, 1],
            "AGEDP2": [0, 0, 0],
            "AGEDP3": [0, 0, 0],
            "AGERANGE": [4, 3, 5],
            "EARNSPLIT": [2, 0, 0],
            "GENDER": [1, 2, 2],
        }
    )

    with pd.HDFStore(path, mode="w") as store:
        store.put("puf", puf, format="table")
        store.put("puf_demographics", demographics, format="table")
