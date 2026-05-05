import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.tax_unit_construction import construct_tax_units


def _person_fixture(**overrides):
    n = max((len(value) for value in overrides.values()), default=1)
    defaults = {
        "PH_SEQ": np.ones(n, dtype=int),
        "A_LINENO": np.arange(1, n + 1, dtype=int),
        "A_AGE": np.zeros(n, dtype=int),
        "A_MARITL": np.full(n, 7, dtype=int),
        "A_SPOUSE": np.zeros(n, dtype=int),
        "PECOHAB": np.full(n, -1, dtype=int),
        "PEPAR1": np.full(n, -1, dtype=int),
        "PEPAR2": np.full(n, -1, dtype=int),
        "A_EXPRRP": np.full(n, 14, dtype=int),
        "A_ENRLW": np.zeros(n, dtype=int),
        "A_FTPT": np.zeros(n, dtype=int),
        "A_HSCOL": np.zeros(n, dtype=int),
        "WSAL_VAL": np.zeros(n, dtype=float),
        "SEMP_VAL": np.zeros(n, dtype=float),
        "FRSE_VAL": np.zeros(n, dtype=float),
        "INT_VAL": np.zeros(n, dtype=float),
        "DIV_VAL": np.zeros(n, dtype=float),
        "RNT_VAL": np.zeros(n, dtype=float),
        "CAP_VAL": np.zeros(n, dtype=float),
        "UC_VAL": np.zeros(n, dtype=float),
        "OI_VAL": np.zeros(n, dtype=float),
        "ANN_VAL": np.zeros(n, dtype=float),
        "PNSN_VAL": np.zeros(n, dtype=float),
        "PTOTVAL": np.zeros(n, dtype=float),
        "SS_VAL": np.zeros(n, dtype=float),
        "PEDISDRS": np.zeros(n, dtype=int),
        "PEDISEAR": np.zeros(n, dtype=int),
        "PEDISEYE": np.zeros(n, dtype=int),
        "PEDISOUT": np.zeros(n, dtype=int),
        "PEDISPHY": np.zeros(n, dtype=int),
        "PEDISREM": np.zeros(n, dtype=int),
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


def _decoded_roles(assignments: pd.DataFrame) -> list[str]:
    return [value.decode() for value in assignments["tax_unit_role_input"].tolist()]


def _decoded_statuses(tax_unit: pd.DataFrame) -> list[str]:
    return [value.decode() for value in tax_unit["filing_status_input"].tolist()]


def test_construct_tax_units_keeps_married_couple_and_child_together():
    person = _person_fixture(
        A_AGE=[40, 38, 8],
        A_MARITL=[1, 1, 7],
        A_SPOUSE=[2, 1, 0],
        A_EXPRRP=[1, 4, 5],
        PEPAR1=[-1, -1, 1],
        PEPAR2=[-1, -1, 2],
        WSAL_VAL=[60_000, 20_000, 0],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].nunique() == 1
    assert _decoded_roles(assignments) == ["HEAD", "SPOUSE", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["JOINT"]


def test_construct_tax_units_claims_low_income_full_time_student():
    person = _person_fixture(
        A_AGE=[45, 20],
        A_EXPRRP=[1, 5],
        PEPAR1=[-1, 1],
        A_ENRLW=[0, 1],
        A_FTPT=[0, 1],
        WSAL_VAL=[70_000, 3_000],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].nunique() == 1
    assert _decoded_roles(assignments) == ["HEAD", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["HEAD_OF_HOUSEHOLD"]


def test_construct_tax_units_claims_enrolled_young_adult_student():
    person = _person_fixture(
        A_AGE=[45, 21],
        A_EXPRRP=[1, 5],
        PEPAR1=[-1, 1],
        A_ENRLW=[0, 1],
        A_FTPT=[0, 2],
        A_HSCOL=[0, 2],
        WSAL_VAL=[70_000, 12_000],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].nunique() == 1
    assert _decoded_roles(assignments) == ["HEAD", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["HEAD_OF_HOUSEHOLD"]


def test_construct_tax_units_leaves_low_income_nonstudent_adult_child_independent():
    person = _person_fixture(
        A_AGE=[45, 22],
        A_EXPRRP=[1, 5],
        PEPAR1=[-1, 1],
        A_ENRLW=[0, 0],
        A_FTPT=[0, 0],
        WSAL_VAL=[70_000, 2_000],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].nunique() == 2
    assert _decoded_roles(assignments) == ["HEAD", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["SINGLE", "SINGLE"]


def test_construct_tax_units_leaves_zero_income_nonstudent_young_adult_child_independent():
    person = _person_fixture(
        A_AGE=[45, 22],
        A_EXPRRP=[1, 5],
        PEPAR1=[-1, 1],
        A_ENRLW=[0, 0],
        A_FTPT=[0, 0],
        WSAL_VAL=[70_000, 0],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].nunique() == 2
    assert _decoded_roles(assignments) == ["HEAD", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["SINGLE", "SINGLE"]


def test_construct_tax_units_leaves_high_income_adult_child_independent():
    person = _person_fixture(
        A_AGE=[45, 22],
        A_EXPRRP=[1, 5],
        PEPAR1=[-1, 1],
        WSAL_VAL=[70_000, 10_000],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].nunique() == 2
    assert _decoded_roles(assignments) == ["HEAD", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["SINGLE", "SINGLE"]


def test_construct_tax_units_assigns_child_to_higher_income_separated_parent():
    person = _person_fixture(
        A_AGE=[40, 38, 10],
        A_MARITL=[6, 6, 7],
        A_EXPRRP=[1, 13, 5],
        PEPAR1=[-1, -1, 1],
        PEPAR2=[-1, -1, 2],
        WSAL_VAL=[50_000, 20_000, 0],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].nunique() == 2
    assert _decoded_roles(assignments) == ["HEAD", "HEAD", "DEPENDENT"]
    child_unit = assignments.loc[2, "TAX_ID"]
    assert child_unit == assignments.loc[0, "TAX_ID"]
    assert sorted(_decoded_statuses(tax_unit)) == ["HEAD_OF_HOUSEHOLD", "SEPARATE"]


def test_construct_tax_units_can_roll_child_of_claimed_adult_up_to_grandparent():
    person = _person_fixture(
        A_AGE=[70, 22, 4],
        A_EXPRRP=[1, 5, 7],
        PEPAR1=[-1, 1, 2],
        A_ENRLW=[0, 1, 0],
        A_FTPT=[0, 1, 0],
        WSAL_VAL=[40_000, 2_000, 0],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].nunique() == 1
    assert _decoded_roles(assignments) == ["HEAD", "DEPENDENT", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["HEAD_OF_HOUSEHOLD"]


def test_construct_tax_units_handles_nonconsecutive_person_index():
    person = _person_fixture(
        A_AGE=[40, 10],
        A_EXPRRP=[1, 5],
        PEPAR1=[-1, 1],
        WSAL_VAL=[50_000, 0],
    )
    person.index = [10, 20]

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments.index.tolist() == [10, 20]
    assert assignments["TAX_ID"].tolist() == [1, 1]
    assert _decoded_roles(assignments) == ["HEAD", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["HEAD_OF_HOUSEHOLD"]


def test_construct_tax_units_allows_missing_optional_evidence_columns():
    person = _person_fixture(
        A_AGE=[40, 10],
        A_EXPRRP=[1, 5],
        PEPAR1=[-1, 1],
    ).drop(
        columns=[
            "A_ENRLW",
            "A_FTPT",
            "A_HSCOL",
            "PTOTVAL",
            "PEDISDRS",
            "PEDISEAR",
            "PEDISEYE",
            "PEDISOUT",
            "PEDISPHY",
            "PEDISREM",
        ]
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].tolist() == [1, 1]
    assert _decoded_roles(assignments) == ["HEAD", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["HEAD_OF_HOUSEHOLD"]


def test_construct_tax_units_collapses_transitive_adult_claim_chains():
    person = _person_fixture(
        A_AGE=[46, 69, 43],
        A_MARITL=[5, 5, 7],
        A_EXPRRP=[1, 10, 12],
        PEPAR1=[-1, -1, 2],
        WSAL_VAL=[0, 0, 0],
        SEMP_VAL=[120_000, 0, 0],
        A_ENRLW=[0, 0, 0],
        A_FTPT=[0, 0, 0],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].tolist() == [1, 2, 3]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["SINGLE", "SINGLE", "SINGLE"]


def test_construct_tax_units_prevents_mutual_adult_claim_cycles():
    person = _person_fixture(
        A_AGE=[39, 75, 42],
        A_MARITL=[7, 5, 7],
        A_EXPRRP=[1, 8, 13],
        PEPAR1=[2, -1, -1],
        PECOHAB=[3, -1, 1],
        WSAL_VAL=[0, 0, 40_000],
        INT_VAL=[13, 3, 3],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].tolist() == [1, 2, 3]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["SINGLE", "SINGLE", "SINGLE"]


def test_construct_tax_units_does_not_claim_adult_child_with_children():
    person = _person_fixture(
        A_AGE=[70, 42, 11],
        A_EXPRRP=[1, 5, 7],
        PEPAR1=[-1, 1, 2],
        WSAL_VAL=[23_000, 0, 0],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].tolist() == [1, 2, 2]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD", "DEPENDENT"]
    assert sorted(_decoded_statuses(tax_unit)) == ["HEAD_OF_HOUSEHOLD", "SINGLE"]


def test_construct_tax_units_keeps_older_grandchild_without_parent_pointer_separate():
    person = _person_fixture(
        A_AGE=[64, 58, 16],
        A_MARITL=[1, 1, 7],
        A_SPOUSE=[2, 1, 0],
        A_EXPRRP=[1, 4, 7],
        WSAL_VAL=[0, 9_000, 0],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].tolist() == [1, 1, 2]
    assert _decoded_roles(assignments) == ["HEAD", "SPOUSE", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["JOINT", "SINGLE"]


def test_construct_tax_units_claims_younger_grandchild_without_parent_pointer():
    person = _person_fixture(
        A_AGE=[64, 58, 12],
        A_MARITL=[1, 1, 7],
        A_SPOUSE=[2, 1, 0],
        A_EXPRRP=[1, 4, 7],
        WSAL_VAL=[0, 9_000, 0],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].tolist() == [1, 1, 1]
    assert _decoded_roles(assignments) == ["HEAD", "SPOUSE", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["JOINT"]


def test_construct_tax_units_claims_under15_nonrelative_without_parent_pointer():
    person = _person_fixture(
        A_AGE=[40, 12],
        A_EXPRRP=[1, 14],
        WSAL_VAL=[50_000, 0],
    )

    assignments, tax_unit = construct_tax_units(person, year=2024)

    assert assignments["TAX_ID"].tolist() == [1, 1]
    assert _decoded_roles(assignments) == ["HEAD", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["SINGLE"]


def test_census_documented_claims_under15_without_parent_pointer_to_main_unit():
    person = _person_fixture(
        A_AGE=[40, 12],
        A_EXPRRP=[1, 14],
        WSAL_VAL=[50_000, 0],
        PTOTVAL=[50_000, 0],
    )

    assignments, tax_unit = construct_tax_units(
        person,
        year=2024,
        mode="census_documented",
    )

    assert assignments["TAX_ID"].tolist() == [1, 1]
    assert _decoded_roles(assignments) == ["HEAD", "DEPENDENT"]
    assert _decoded_statuses(tax_unit) == ["HEAD_OF_HOUSEHOLD"]


def test_census_documented_leaves_age15_without_parent_pointer_independent():
    person = _person_fixture(
        A_AGE=[40, 15],
        A_EXPRRP=[1, 14],
        WSAL_VAL=[50_000, 0],
        PTOTVAL=[50_000, 0],
    )

    assignments, tax_unit = construct_tax_units(
        person,
        year=2024,
        mode="census_documented",
    )

    assert assignments["TAX_ID"].tolist() == [1, 2]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD"]
    assert sorted(_decoded_statuses(tax_unit)) == ["SINGLE", "SINGLE"]


def test_census_documented_uses_total_money_income_for_split_parents():
    person = _person_fixture(
        A_AGE=[40, 38, 10],
        A_MARITL=[7, 7, 7],
        A_EXPRRP=[1, 13, 5],
        PEPAR1=[-1, -1, 1],
        PEPAR2=[-1, -1, 2],
        WSAL_VAL=[0, 50_000, 0],
        PTOTVAL=[30_000, 20_000, 0],
    )

    assignments, tax_unit = construct_tax_units(
        person,
        year=2024,
        mode="census_documented",
    )

    assert assignments["TAX_ID"].tolist() == [1, 2, 1]
    assert _decoded_roles(assignments) == ["HEAD", "HEAD", "DEPENDENT"]
    assert sorted(_decoded_statuses(tax_unit)) == ["HEAD_OF_HOUSEHOLD", "SINGLE"]


def test_construct_tax_units_rejects_unknown_mode():
    person = _person_fixture(A_AGE=[40], A_EXPRRP=[1])

    try:
        construct_tax_units(person, year=2024, mode="unknown")
    except ValueError as error:
        assert "Unsupported tax-unit construction mode" in str(error)
    else:
        raise AssertionError("Expected construct_tax_units to reject unknown modes")
