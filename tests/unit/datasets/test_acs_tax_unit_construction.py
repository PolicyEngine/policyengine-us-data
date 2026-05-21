import h5py
import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.datasets.acs.acs import ACS, ACS_2022
from policyengine_us_data.datasets.acs.census_acs import CensusACS
from policyengine_us_data.datasets.acs.acs_to_cps_columns import (
    acs_person_to_cps_tax_unit_columns,
)
from policyengine_us_data.datasets.acs.tax_unit_construction import (
    construct_tax_units_acs,
)


def _acs_person_fixture(**overrides):
    n = max((len(value) for value in overrides.values()), default=1)
    defaults = {
        "SERIALNO": np.repeat("1", n),
        "SPORDER": np.arange(1, n + 1, dtype=int),
        "household_id": np.ones(n, dtype=int),
        "AGEP": np.zeros(n, dtype=int),
        "MAR": np.full(n, 5, dtype=int),
        "RELSHIPP": np.full(n, 36, dtype=int),
        "SEX": np.ones(n, dtype=int),
        "WAGP": np.zeros(n, dtype=float),
        "SEMP": np.zeros(n, dtype=float),
        "INTP": np.zeros(n, dtype=float),
        "RETP": np.zeros(n, dtype=float),
        "OIP": np.zeros(n, dtype=float),
        "PAP": np.zeros(n, dtype=float),
        "SSP": np.zeros(n, dtype=float),
        "SSIP": np.zeros(n, dtype=float),
        "PINCP": np.zeros(n, dtype=float),
        "SCH": np.ones(n, dtype=int),
        "SCHG": np.zeros(n, dtype=int),
        "DDRS": np.full(n, 2, dtype=int),
        "DEAR": np.full(n, 2, dtype=int),
        "DEYE": np.full(n, 2, dtype=int),
        "DOUT": np.full(n, 2, dtype=int),
        "DPHY": np.full(n, 2, dtype=int),
        "DREM": np.full(n, 2, dtype=int),
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


def _decoded_roles(assignments: pd.DataFrame) -> list[str]:
    return [value.decode() for value in assignments["tax_unit_role_input"].tolist()]


def _decoded_statuses(tax_unit: pd.DataFrame) -> list[str]:
    return [value.decode() for value in tax_unit["filing_status_input"].tolist()]


def test_acs_mapper_links_reference_spouse_and_child():
    person = _acs_person_fixture(
        AGEP=[40, 38, 8],
        MAR=[1, 1, 5],
        RELSHIPP=[20, 21, 25],
        SEX=[1, 2, 1],
        WAGP=[60_000, 20_000, 0],
    )

    cps_like = acs_person_to_cps_tax_unit_columns(person)
    assignments, tax_unit = construct_tax_units_acs(person, year=2022)

    assert cps_like["A_SPOUSE"].tolist() == [2, 1, 0]
    assert cps_like["PEPAR1"].tolist() == [0, 0, 1]
    assert cps_like["PEPAR2"].tolist() == [0, 0, 2]
    assert cps_like["A_MARITL"].tolist() == [1, 1, 7]
    assert assignments["TAX_ID"].nunique() == 1
    assert _decoded_roles(assignments) == ["HEAD", "SPOUSE", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["JOINT"]


def test_construct_tax_units_acs_splits_unrelated_adult_roommates():
    person = _acs_person_fixture(
        AGEP=[32, 29],
        MAR=[5, 5],
        RELSHIPP=[20, 34],
        WAGP=[50_000, 45_000],
    )

    assignments, tax_unit = construct_tax_units_acs(person, year=2022)

    assert assignments["TAX_ID"].tolist() == [1, 2]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["SINGLE", "SINGLE"]


def test_construct_tax_units_acs_pairs_married_child_and_child_in_law():
    person = _acs_person_fixture(
        AGEP=[68, 35, 34, 5],
        MAR=[2, 1, 1, 5],
        RELSHIPP=[20, 25, 32, 30],
        SEX=[2, 2, 1, 1],
        WAGP=[20_000, 60_000, 55_000, 0],
    )

    cps_like = acs_person_to_cps_tax_unit_columns(person)
    assignments, tax_unit = construct_tax_units_acs(person, year=2022)

    assert cps_like["A_SPOUSE"].tolist() == [0, 3, 2, 0]
    assert cps_like["acs_spouse_link_imputed"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert cps_like["PEPAR1"].tolist() == [0, 1, 0, 2]
    assert cps_like["PEPAR2"].tolist() == [0, 0, 0, 3]
    assert cps_like["acs_parent_link_imputed"].tolist() == [
        False,
        False,
        False,
        True,
    ]
    assert assignments["TAX_ID"].tolist() == [1, 2, 2, 2]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD", "SPOUSE", "DEPENDENT"]
    assert sorted(_decoded_statuses(tax_unit)) == ["JOINT", "SINGLE"]


def test_construct_tax_units_acs_uses_three_generation_parent_inference():
    person = _acs_person_fixture(
        AGEP=[70, 30, 4],
        MAR=[2, 5, 5],
        RELSHIPP=[20, 25, 30],
        SEX=[2, 2, 1],
        WAGP=[35_000, 22_000, 0],
    )

    cps_like = acs_person_to_cps_tax_unit_columns(person)
    assignments, tax_unit = construct_tax_units_acs(person, year=2022)

    assert cps_like["PEPAR1"].tolist() == [0, 1, 2]
    assert cps_like["PEPAR2"].tolist() == [0, 0, 0]
    assert assignments["TAX_ID"].tolist() == [1, 2, 2]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD", "DEPENDENT"]
    assert sorted(_decoded_statuses(tax_unit)) == ["HEAD_OF_HOUSEHOLD", "SINGLE"]


def test_construct_tax_units_acs_splits_group_quarters_adults():
    person = _acs_person_fixture(
        AGEP=[22, 23],
        MAR=[5, 5],
        RELSHIPP=[37, 38],
        WAGP=[15_000, 18_000],
    )

    assignments, tax_unit = construct_tax_units_acs(person, year=2022)

    assert assignments["TAX_ID"].tolist() == [1, 2]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["SINGLE", "SINGLE"]


def test_construct_tax_units_acs_handles_child_only_household():
    person = _acs_person_fixture(
        AGEP=[12, 10],
        MAR=[5, 5],
        RELSHIPP=[20, 28],
        WAGP=[0, 0],
    )

    assignments, tax_unit = construct_tax_units_acs(person, year=2022)

    assert assignments["TAX_ID"].tolist() == [1, 2]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["SINGLE", "SINGLE"]


def test_acs_mapper_rejects_stale_raw_person_table_missing_tax_unit_columns():
    person = _acs_person_fixture().drop(columns=["RELSHIPP", "INTP", "SCH", "SCHG"])

    with pytest.raises(KeyError, match="Regenerate the raw Census ACS dataset"):
        acs_person_to_cps_tax_unit_columns(person)


def test_acs_add_id_variables_writes_tax_unit_ids():
    person = _acs_person_fixture(
        SERIALNO=["1", "1"],
        household_id=[0, 0],
        AGEP=[32, 29],
        MAR=[5, 5],
        RELSHIPP=[20, 34],
        WAGP=[50_000, 45_000],
    )
    household = pd.DataFrame({"SERIALNO": ["1"], "WGTP": [100]})

    with h5py.File("memory", mode="w", driver="core", backing_store=False) as acs:
        ACS.add_id_variables(acs, person, household, year=2022)
        person_tax_unit_id = acs["person_tax_unit_id"][:]
        tax_unit_id = acs["tax_unit_id"][:]

    assert person_tax_unit_id.tolist() == [1, 2]
    assert tax_unit_id.tolist() == [1, 2]


def test_acs_add_id_variables_handles_duplicate_person_index_labels():
    person = _acs_person_fixture(
        SERIALNO=["1", "2"],
        AGEP=[32, 29],
        MAR=[5, 5],
        RELSHIPP=[20, 20],
        WAGP=[50_000, 45_000],
    )
    person.index = [0, 0]
    household = pd.DataFrame({"SERIALNO": ["1", "2"], "WGTP": [100, 90]})

    with h5py.File("memory", mode="w", driver="core", backing_store=False) as acs:
        ACS.add_id_variables(acs, person, household, year=2022)
        person_id = acs["person_id"][:]
        person_tax_unit_id = acs["person_tax_unit_id"][:]
        tax_unit_id = acs["tax_unit_id"][:]

    assert person_id.tolist() == [1, 2]
    assert person_tax_unit_id.tolist() == [1, 2]
    assert tax_unit_id.tolist() == [1, 2]


def test_acs_add_id_variables_writes_related_to_head_or_spouse():
    person = _acs_person_fixture(
        SERIALNO=["1", "1"],
        AGEP=[40, 12],
        MAR=[5, 5],
        RELSHIPP=[20, 36],
        WAGP=[50_000, 0],
    )
    household = pd.DataFrame({"SERIALNO": ["1"], "WGTP": [100]})

    with h5py.File("memory", mode="w", driver="core", backing_store=False) as acs:
        ACS.add_id_variables(acs, person, household, year=2022)
        person_tax_unit_id = acs["person_tax_unit_id"][:]
        is_related_to_head_or_spouse = acs["is_related_to_head_or_spouse"][:]

    assert person_tax_unit_id.tolist() == [1, 1]
    assert is_related_to_head_or_spouse.tolist() == [True, False]


def test_acs_add_person_variables_requires_raw_allocation_flags():
    person = _acs_person_fixture(household_id=[0])
    household = pd.DataFrame(
        {
            "household_id": [0],
            "RNTP": [1_200],
            "TAXAMT": [500],
            "TEN": [3],
        }
    )

    with h5py.File("memory", mode="w", driver="core", backing_store=False) as acs:
        with pytest.raises(KeyError, match="FRNTP"):
            ACS.add_person_variables(acs, person, household)


def test_acs_add_person_variables_writes_allocation_flags_for_heads_only():
    person = _acs_person_fixture(
        household_id=[0, 0],
        SPORDER=[1, 2],
        AGEP=[40, 10],
    )
    household = pd.DataFrame(
        {
            "household_id": [0],
            "RNTP": [1_200],
            "TAXAMT": [500],
            "FRNTP": [1],
            "FTAXP": [0],
            "TEN": [3],
        }
    )

    with h5py.File("memory", mode="w", driver="core", backing_store=False) as acs:
        ACS.add_person_variables(acs, person, household)
        rent_is_allocated = acs["rent_is_allocated"][:]
        real_estate_taxes_is_allocated = acs["real_estate_taxes_is_allocated"][:]

    assert rent_is_allocated.tolist() == [True, False]
    assert real_estate_taxes_is_allocated.tolist() == [False, False]


def test_acs_2022_does_not_download_stale_release_artifact():
    assert ACS_2022.url is None


def test_acs_exists_rejects_stale_artifact_missing_allocation_flags(tmp_path):
    path = tmp_path / "acs_2022.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("is_household_head", data=np.array([True]))

    class TempACS(ACS):
        name = "temp_acs"
        label = "Temp ACS"
        file_path = path

    assert TempACS().exists is False

    with h5py.File(path, "a") as h5:
        h5.create_dataset("rent_is_allocated", data=np.array([False]))
        h5.create_dataset("real_estate_taxes_is_allocated", data=np.array([False]))

    assert TempACS().exists is True


def test_raw_census_acs_exists_rejects_stale_cache_missing_allocation_flags(tmp_path):
    path = tmp_path / "census_acs_2022.h5"

    class TempCensusACS(CensusACS):
        name = "temp_census_acs"
        label = "Temp Census ACS"
        file_path = path

    with pd.HDFStore(path, mode="w") as storage:
        storage["household"] = pd.DataFrame({"SERIALNO": ["1"], "RNTP": [1_200]})
        storage["person"] = pd.DataFrame({"SERIALNO": ["1"]})

    assert TempCensusACS().exists is False

    with pd.HDFStore(path, mode="w") as storage:
        storage["household"] = pd.DataFrame(
            {
                "SERIALNO": ["1"],
                "RNTP": [1_200],
                "FRNTP": [0],
                "FTAXP": [0],
            }
        )
        storage["person"] = pd.DataFrame({"SERIALNO": ["1"]})

    assert TempCensusACS().exists is True
