from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.tax_unit_rule_helpers import (
    REFERENCE_PERSON_CODES,
    dependent_gross_income_limit,
    qualifying_child_age_test,
    reference_relationship_allows_qualifying_child,
    reference_relationship_allows_qualifying_relative,
    related_to_head_or_spouse as reference_related_to_head_or_spouse,
)


HEAD = "HEAD"
SPOUSE = "SPOUSE"
DEPENDENT = "DEPENDENT"

POLICYENGINE_MODE = "policyengine"
CENSUS_DOCUMENTED_MODE = "census_documented"
SUPPORTED_TAX_UNIT_CONSTRUCTION_MODES = frozenset(
    {
        POLICYENGINE_MODE,
        CENSUS_DOCUMENTED_MODE,
    }
)


@dataclass(frozen=True)
class _HouseholdPerson:
    index: int
    household_id: int
    line_no: int
    age: int
    relationship_code: int | None
    marital_status: int
    spouse_line: int | None
    parent_lines: tuple[int, ...]
    gross_income: float
    claimant_income: float
    total_money_income: float
    is_full_time_student: bool
    is_permanently_disabled: bool

    @property
    def starts_base_unit(self) -> bool:
        return self.age >= 18 or self.marital_status in {1, 2, 3, 4, 5, 6}

    @property
    def married_spouse_present(self) -> bool:
        return self.marital_status in {1, 2} and self.spouse_line is not None


@dataclass
class _BaseTaxUnit:
    key: tuple
    household_id: int
    head_index: int
    spouse_index: int | None = None
    claimant_lines: tuple[int, ...] = ()
    claimant_income: float = 0.0
    total_money_income: float = 0.0
    head_age: int = 0


@dataclass(frozen=True)
class _ClaimCandidate:
    unit_key: tuple
    priority: int
    score: tuple[Any, ...]


def _to_optional_positive_int(value) -> int | None:
    if pd.isna(value):
        return None
    value = int(value)
    return value if value > 0 else None


def _to_optional_parent_line(value) -> int | None:
    if pd.isna(value):
        return None
    value = int(value)
    return value if value > 0 else None


def _positive_series(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person:
        return np.zeros(len(person), dtype=float)
    values = (
        pd.to_numeric(person[column], errors="coerce")
        .fillna(0)
        .to_numpy(
            dtype=float,
            copy=False,
        )
    )
    return np.maximum(values, 0)


def estimate_dependent_gross_income(person: pd.DataFrame) -> np.ndarray:
    return (
        _positive_series(person, "WSAL_VAL")
        + _positive_series(person, "SEMP_VAL")
        + _positive_series(person, "FRSE_VAL")
        + _positive_series(person, "INT_VAL")
        + _positive_series(person, "DIV_VAL")
        + _positive_series(person, "RNT_VAL")
        + _positive_series(person, "CAP_VAL")
        + _positive_series(person, "UC_VAL")
        + _positive_series(person, "OI_VAL")
        + _positive_series(person, "ANN_VAL")
        + _positive_series(person, "PNSN_VAL")
    )


def _estimate_claimant_income(person: pd.DataFrame) -> np.ndarray:
    return estimate_dependent_gross_income(person) + _positive_series(person, "SS_VAL")


def _prepare_household_people(
    household: pd.DataFrame,
    household_id: int,
) -> list[_HouseholdPerson]:
    disability_flags = [
        "PEDISDRS",
        "PEDISEAR",
        "PEDISEYE",
        "PEDISOUT",
        "PEDISPHY",
        "PEDISREM",
    ]
    gross_income = estimate_dependent_gross_income(household)
    claimant_income = _estimate_claimant_income(household)
    total_money_income = (
        pd.to_numeric(household["PTOTVAL"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float, copy=False)
        if "PTOTVAL" in household
        else claimant_income.copy()
    )
    has_disability = (
        pd.DataFrame(
            {
                flag: household[flag] if flag in household else 0
                for flag in disability_flags
            },
            index=household.index,
        )
        .eq(1)
        .any(axis=1)
        .to_numpy()
    )
    enrolled = (
        household["A_ENRLW"]
        if "A_ENRLW" in household
        else pd.Series(0, index=household.index)
    )
    full_time = (
        household["A_FTPT"]
        if "A_FTPT" in household
        else pd.Series(0, index=household.index)
    )
    school_level = (
        household["A_HSCOL"]
        if "A_HSCOL" in household
        else pd.Series(0, index=household.index)
    )
    enrolled_values = pd.to_numeric(enrolled, errors="coerce").fillna(0)
    full_time_values = pd.to_numeric(full_time, errors="coerce").fillna(0)
    school_level_values = pd.to_numeric(school_level, errors="coerce").fillna(0)
    # Limit this to tax-unit construction: CPS TAX_ID behavior treats current
    # high-school or college enrollment as strong student evidence for young
    # adults even when the full-time flag is absent or part-time.
    is_full_time_student = (
        ((enrolled_values == 1) & (full_time_values == 1))
        | ((enrolled_values == 1) & school_level_values.isin([1, 2]))
    ).to_numpy()
    people = []
    for row_number, (index, row) in enumerate(household.iterrows()):
        line_no = int(row["A_LINENO"])
        parent_lines = tuple(
            parent
            for parent in (
                _to_optional_parent_line(row.get("PEPAR1", 0)),
                _to_optional_parent_line(row.get("PEPAR2", 0)),
            )
            if parent is not None
        )
        relationship_code = row.get("A_EXPRRP")
        if pd.isna(relationship_code):
            relationship_code = None
        else:
            relationship_code = int(relationship_code)
        people.append(
            _HouseholdPerson(
                index=index,
                household_id=household_id,
                line_no=line_no,
                age=int(row["A_AGE"]),
                relationship_code=relationship_code,
                marital_status=int(row.get("A_MARITL", 7)),
                spouse_line=_to_optional_positive_int(row.get("A_SPOUSE", 0)),
                parent_lines=parent_lines,
                gross_income=float(gross_income[row_number]),
                claimant_income=float(claimant_income[row_number]),
                total_money_income=float(total_money_income[row_number]),
                is_full_time_student=bool(is_full_time_student[row_number]),
                is_permanently_disabled=bool(has_disability[row_number]),
            )
        )
    return people


def _choose_pair_head(
    person_a: _HouseholdPerson,
    person_b: _HouseholdPerson,
) -> tuple[_HouseholdPerson, _HouseholdPerson]:
    if person_a.relationship_code in {code.value for code in REFERENCE_PERSON_CODES}:
        return person_a, person_b
    if person_b.relationship_code in {code.value for code in REFERENCE_PERSON_CODES}:
        return person_b, person_a
    if person_a.age != person_b.age:
        return (
            (person_a, person_b)
            if person_a.age > person_b.age
            else (person_b, person_a)
        )
    return (
        (person_a, person_b)
        if person_a.line_no < person_b.line_no
        else (person_b, person_a)
    )


def _build_base_tax_units(
    people: list[_HouseholdPerson],
) -> tuple[dict[tuple, _BaseTaxUnit], dict[int, tuple], tuple | None]:
    by_line = {person.line_no: person for person in people}
    paired_indices: set[int] = set()
    units: dict[tuple, _BaseTaxUnit] = {}
    base_unit_by_person: dict[int, tuple] = {}
    reference_unit_key: tuple | None = None

    married_pairs: set[tuple[int, int]] = set()
    for person in people:
        if not person.married_spouse_present:
            continue
        spouse = by_line.get(person.spouse_line)
        if (
            spouse is None
            or spouse.index == person.index
            or not spouse.married_spouse_present
        ):
            continue
        married_pairs.add(tuple(sorted((person.line_no, spouse.line_no))))

    for line_a, line_b in sorted(married_pairs):
        person_a = by_line[line_a]
        person_b = by_line[line_b]
        head, spouse = _choose_pair_head(person_a, person_b)
        key = ("pair", min(line_a, line_b), max(line_a, line_b))
        unit = _BaseTaxUnit(
            key=key,
            household_id=head.household_id,
            head_index=head.index,
            spouse_index=spouse.index,
            claimant_lines=(head.line_no, spouse.line_no),
            claimant_income=head.claimant_income + spouse.claimant_income,
            total_money_income=head.total_money_income + spouse.total_money_income,
            head_age=head.age,
        )
        units[key] = unit
        paired_indices.update({head.index, spouse.index})
        base_unit_by_person[head.index] = key
        base_unit_by_person[spouse.index] = key
        if head.relationship_code in {
            code.value for code in REFERENCE_PERSON_CODES
        } or spouse.relationship_code in {
            code.value for code in REFERENCE_PERSON_CODES
        }:
            reference_unit_key = key

    for person in people:
        if person.index in paired_indices or not person.starts_base_unit:
            continue
        key = ("single", person.line_no)
        units[key] = _BaseTaxUnit(
            key=key,
            household_id=person.household_id,
            head_index=person.index,
            claimant_lines=(person.line_no,),
            claimant_income=person.claimant_income,
            total_money_income=person.total_money_income,
            head_age=person.age,
        )
        base_unit_by_person[person.index] = key
        if person.relationship_code in {code.value for code in REFERENCE_PERSON_CODES}:
            reference_unit_key = key

    return units, base_unit_by_person, reference_unit_key


def _parent_candidate_units(
    person: _HouseholdPerson,
    base_units: dict[tuple, _BaseTaxUnit],
    eligible_units: set[tuple],
) -> list[tuple]:
    candidates = []
    for unit_key in eligible_units:
        unit = base_units[unit_key]
        if any(
            parent_line in unit.claimant_lines for parent_line in person.parent_lines
        ):
            candidates.append(unit_key)
    return candidates


def _reference_candidate_unit(
    person: _HouseholdPerson,
    reference_unit_key: tuple | None,
    base_unit_key: tuple | None,
    eligible_units: set[tuple],
) -> tuple | None:
    if (
        reference_unit_key is None
        or reference_unit_key == base_unit_key
        or reference_unit_key not in eligible_units
    ):
        return None
    return reference_unit_key


def _unit_income_score(
    unit_key: tuple,
    base_units: dict[tuple, _BaseTaxUnit],
) -> tuple[float, int, int]:
    unit = base_units[unit_key]
    return (
        unit.claimant_income,
        unit.head_age,
        -unit.claimant_lines[0],
    )


def _choose_best_candidate(candidates: list[_ClaimCandidate]) -> tuple | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (candidate.priority, candidate.score),
    ).unit_key


def _choose_best_parent_unit_by_total_money_income(
    candidate_units: list[tuple],
    base_units: dict[tuple, _BaseTaxUnit],
) -> tuple | None:
    if not candidate_units:
        return None
    return max(
        candidate_units,
        key=lambda key: (
            base_units[key].total_money_income,
            base_units[key].claimant_income,
            base_units[key].head_age,
            -base_units[key].claimant_lines[0],
        ),
    )


def _choose_main_filing_unit(
    base_units: dict[tuple, _BaseTaxUnit],
    reference_unit_key: tuple | None,
) -> tuple | None:
    if reference_unit_key in base_units:
        return reference_unit_key
    if not base_units:
        return None
    return max(
        base_units,
        key=lambda key: (
            base_units[key].total_money_income,
            base_units[key].claimant_income,
            base_units[key].head_age,
            -base_units[key].claimant_lines[0],
        ),
    )


def _select_claimant_unit(
    person: _HouseholdPerson,
    year: int,
    base_units: dict[tuple, _BaseTaxUnit],
    base_unit_key: tuple | None,
    reference_unit_key: tuple | None,
    eligible_units: set[tuple],
) -> tuple | None:
    parent_units = _parent_candidate_units(person, base_units, eligible_units)
    age_eligible = qualifying_child_age_test(
        age=person.age,
        is_full_time_student=person.is_full_time_student,
        is_permanently_disabled=person.is_permanently_disabled,
    )

    reference_unit = _reference_candidate_unit(
        person,
        reference_unit_key,
        base_unit_key,
        eligible_units,
    )
    candidates: list[_ClaimCandidate] = []

    if age_eligible:
        candidates.extend(
            _ClaimCandidate(
                unit_key=unit_key,
                priority=100,
                score=_unit_income_score(unit_key, base_units),
            )
            for unit_key in parent_units
        )
        if (
            reference_unit is not None
            and not person.starts_base_unit
            and not person.parent_lines
            and person.age < 15
        ):
            candidates.append(
                _ClaimCandidate(
                    unit_key=reference_unit,
                    priority=80,
                    score=_unit_income_score(reference_unit, base_units),
                )
            )
        selected = _choose_best_candidate(candidates)
        if selected is not None:
            return selected

    if person.gross_income >= dependent_gross_income_limit(year):
        return None

    if person.starts_base_unit:
        return None

    candidates.extend(
        _ClaimCandidate(
            unit_key=unit_key,
            priority=60,
            score=_unit_income_score(unit_key, base_units),
        )
        for unit_key in parent_units
    )

    if (
        reference_unit is not None
        and (
            reference_relationship_allows_qualifying_relative(person.relationship_code)
            or (not person.parent_lines and person.age < 15)
        )
        and person.age < 15
    ):
        candidates.append(
            _ClaimCandidate(
                unit_key=reference_unit,
                priority=50,
                score=_unit_income_score(reference_unit, base_units),
            )
        )

    return _choose_best_candidate(candidates)


def _determine_final_assignments_for_household_policyengine(
    people: list[_HouseholdPerson],
    year: int,
) -> tuple[dict[int, tuple], dict[int, str], dict[tuple, str], dict[int, bool]]:
    base_units, base_unit_by_person, reference_unit_key = _build_base_tax_units(people)
    person_by_index = {person.index: person for person in people}

    adult_claims: dict[int, tuple] = {}
    adult_candidates = [
        person
        for person in people
        if person.starts_base_unit
        and base_unit_by_person.get(person.index) in base_units
        and base_units[base_unit_by_person[person.index]].spouse_index is None
    ]
    eligible_units = set(base_units)
    for person in sorted(adult_candidates, key=lambda item: (item.age, item.line_no)):
        unit_key = _select_claimant_unit(
            person=person,
            year=year,
            base_units=base_units,
            base_unit_key=base_unit_by_person.get(person.index),
            reference_unit_key=reference_unit_key,
            eligible_units=eligible_units,
        )
        if unit_key is not None:
            adult_claims[person.index] = unit_key
            claimed_person_unit_key = base_unit_by_person.get(person.index)
            if claimed_person_unit_key is not None:
                eligible_units.discard(claimed_person_unit_key)

    def _resolve_surviving_unit(unit_key: tuple) -> tuple:
        seen: set[tuple] = set()
        current_unit_key = unit_key
        while current_unit_key not in seen:
            seen.add(current_unit_key)
            unit = base_units[current_unit_key]
            if unit.spouse_index is not None:
                return current_unit_key
            next_unit_key = adult_claims.get(unit.head_index)
            if next_unit_key is None:
                return current_unit_key
            current_unit_key = next_unit_key
        return current_unit_key

    adult_claims = {
        person_index: _resolve_surviving_unit(unit_key)
        for person_index, unit_key in adult_claims.items()
    }

    surviving_units = {
        unit_key
        for unit_key, unit in base_units.items()
        if unit.spouse_index is not None or unit.head_index not in adult_claims
    }

    child_claims: dict[int, tuple] = {}
    child_candidates = [
        person
        for person in people
        if not person.starts_base_unit and person.index not in adult_claims
    ]
    for person in sorted(child_candidates, key=lambda item: (item.age, item.line_no)):
        unit_key = _select_claimant_unit(
            person=person,
            year=year,
            base_units=base_units,
            base_unit_key=base_unit_by_person.get(person.index),
            reference_unit_key=reference_unit_key,
            eligible_units=surviving_units,
        )
        if unit_key is not None:
            child_claims[person.index] = unit_key

    final_unit_key_by_person: dict[int, tuple] = {}
    roles_by_person: dict[int, str] = {}
    for unit_key, unit in base_units.items():
        if unit.spouse_index is not None:
            final_unit_key_by_person[unit.head_index] = unit_key
            final_unit_key_by_person[unit.spouse_index] = unit_key
            roles_by_person[unit.head_index] = HEAD
            roles_by_person[unit.spouse_index] = SPOUSE
            continue
        if unit.head_index in adult_claims:
            continue
        final_unit_key_by_person[unit.head_index] = unit_key
        roles_by_person[unit.head_index] = HEAD

    for person_index, unit_key in adult_claims.items():
        final_unit_key_by_person[person_index] = unit_key
        roles_by_person[person_index] = DEPENDENT

    for person_index, unit_key in child_claims.items():
        final_unit_key_by_person[person_index] = unit_key
        roles_by_person[person_index] = DEPENDENT

    for person in people:
        if person.index in final_unit_key_by_person:
            continue
        unit_key = ("single", person.line_no)
        final_unit_key_by_person[person.index] = unit_key
        roles_by_person[person.index] = HEAD

    related_to_head_or_spouse: dict[int, bool] = {}
    head_spouse_lines_by_unit: dict[tuple, set[int]] = {}
    for person_index, unit_key in final_unit_key_by_person.items():
        role = roles_by_person[person_index]
        if role in {HEAD, SPOUSE}:
            head_spouse_lines_by_unit.setdefault(unit_key, set()).add(
                person_by_index[person_index].line_no
            )

    filing_status_by_unit: dict[tuple, str] = {}
    unit_members: dict[tuple, list[_HouseholdPerson]] = {}
    for person_index, unit_key in final_unit_key_by_person.items():
        unit_members.setdefault(unit_key, []).append(person_by_index[person_index])

    for unit_key, members in unit_members.items():
        roles = {person.index: roles_by_person[person.index] for person in members}
        has_spouse = any(role == SPOUSE for role in roles.values())
        head = next(person for person in members if roles[person.index] == HEAD)
        claimant_lines = head_spouse_lines_by_unit.get(unit_key, {head.line_no})

        for person in members:
            if roles[person.index] in {HEAD, SPOUSE}:
                related_to_head_or_spouse[person.index] = True
                continue
            related_to_head_or_spouse[person.index] = any(
                parent_line in claimant_lines for parent_line in person.parent_lines
            ) or reference_related_to_head_or_spouse(person.relationship_code)

        if has_spouse:
            filing_status_by_unit[unit_key] = "JOINT"
            continue

        has_qualifying_child = any(
            roles[person.index] == DEPENDENT
            and (
                any(
                    parent_line in claimant_lines for parent_line in person.parent_lines
                )
                or reference_relationship_allows_qualifying_child(
                    person.relationship_code
                )
            )
            and qualifying_child_age_test(
                age=person.age,
                is_full_time_student=person.is_full_time_student,
                is_permanently_disabled=person.is_permanently_disabled,
            )
            for person in members
        )
        has_qualifying_relative = any(
            roles[person.index] == DEPENDENT
            and related_to_head_or_spouse[person.index]
            and person.gross_income < dependent_gross_income_limit(year)
            for person in members
        )
        has_head_of_household_person = has_qualifying_child or has_qualifying_relative

        if head.marital_status == 4 and has_qualifying_child:
            filing_status_by_unit[unit_key] = "SURVIVING_SPOUSE"
        elif has_head_of_household_person and head.marital_status != 6:
            filing_status_by_unit[unit_key] = "HEAD_OF_HOUSEHOLD"
        elif has_head_of_household_person and head.marital_status == 6:
            filing_status_by_unit[unit_key] = "HEAD_OF_HOUSEHOLD"
        elif head.marital_status == 6:
            filing_status_by_unit[unit_key] = "SEPARATE"
        else:
            filing_status_by_unit[unit_key] = "SINGLE"

    return (
        final_unit_key_by_person,
        roles_by_person,
        filing_status_by_unit,
        related_to_head_or_spouse,
    )


def _determine_final_assignments_for_household_census_documented(
    people: list[_HouseholdPerson],
    year: int,
) -> tuple[dict[int, tuple], dict[int, str], dict[tuple, str], dict[int, bool]]:
    del year
    # Follow the publicly documented Census tax-model flow: married + dependents
    # + others, qualifying-child-only parent-pointer claims, and under-15
    # no-parent fallback to the household's main filing unit.
    base_units, _, reference_unit_key = _build_base_tax_units(people)
    person_by_index = {person.index: person for person in people}
    main_unit_key = _choose_main_filing_unit(base_units, reference_unit_key)

    final_unit_key_by_person: dict[int, tuple] = {}
    roles_by_person: dict[int, str] = {}

    for unit_key, unit in base_units.items():
        final_unit_key_by_person[unit.head_index] = unit_key
        roles_by_person[unit.head_index] = HEAD
        if unit.spouse_index is not None:
            final_unit_key_by_person[unit.spouse_index] = unit_key
            roles_by_person[unit.spouse_index] = SPOUSE

    dependent_claims: dict[int, tuple] = {}
    for person in sorted(people, key=lambda item: (item.age, item.line_no)):
        if person.index in final_unit_key_by_person or person.married_spouse_present:
            continue

        age_eligible = qualifying_child_age_test(
            age=person.age,
            is_full_time_student=person.is_full_time_student,
            is_permanently_disabled=person.is_permanently_disabled,
        )
        if person.parent_lines and age_eligible:
            parent_units = [
                unit_key
                for unit_key, unit in base_units.items()
                if any(
                    parent_line in unit.claimant_lines
                    for parent_line in person.parent_lines
                )
            ]
            unit_key = _choose_best_parent_unit_by_total_money_income(
                parent_units,
                base_units,
            )
            if unit_key is not None:
                dependent_claims[person.index] = unit_key
                continue

        if not person.parent_lines and person.age < 15 and main_unit_key is not None:
            dependent_claims[person.index] = main_unit_key

    for person_index, unit_key in dependent_claims.items():
        final_unit_key_by_person[person_index] = unit_key
        roles_by_person[person_index] = DEPENDENT

    for person in people:
        if person.index in final_unit_key_by_person:
            continue
        unit_key = ("single", person.line_no)
        final_unit_key_by_person[person.index] = unit_key
        roles_by_person[person.index] = HEAD

    related_to_head_or_spouse: dict[int, bool] = {}
    unit_members: dict[tuple, list[_HouseholdPerson]] = {}
    head_spouse_lines_by_unit: dict[tuple, set[int]] = {}
    for person_index, unit_key in final_unit_key_by_person.items():
        unit_members.setdefault(unit_key, []).append(person_by_index[person_index])
        if roles_by_person[person_index] in {HEAD, SPOUSE}:
            head_spouse_lines_by_unit.setdefault(unit_key, set()).add(
                person_by_index[person_index].line_no
            )

    filing_status_by_unit: dict[tuple, str] = {}
    for unit_key, members in unit_members.items():
        roles = {person.index: roles_by_person[person.index] for person in members}
        has_spouse = any(role == SPOUSE for role in roles.values())
        has_dependents = any(role == DEPENDENT for role in roles.values())
        claimant_lines = head_spouse_lines_by_unit.get(unit_key, set())

        for person in members:
            if roles[person.index] in {HEAD, SPOUSE}:
                related_to_head_or_spouse[person.index] = True
                continue
            related_to_head_or_spouse[person.index] = any(
                parent_line in claimant_lines for parent_line in person.parent_lines
            ) or reference_related_to_head_or_spouse(person.relationship_code)

        if has_spouse:
            filing_status_by_unit[unit_key] = "JOINT"
        elif has_dependents:
            filing_status_by_unit[unit_key] = "HEAD_OF_HOUSEHOLD"
        else:
            filing_status_by_unit[unit_key] = "SINGLE"

    return (
        final_unit_key_by_person,
        roles_by_person,
        filing_status_by_unit,
        related_to_head_or_spouse,
    )


def construct_tax_units(
    person: pd.DataFrame,
    year: int,
    mode: str = POLICYENGINE_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {
        "PH_SEQ",
        "A_LINENO",
        "A_AGE",
        "A_MARITL",
        "A_SPOUSE",
        "PEPAR1",
        "PEPAR2",
        "A_EXPRRP",
    }
    missing = sorted(
        column for column in required_columns if column not in person.columns
    )
    if missing:
        raise KeyError(
            "Missing required CPS columns for tax-unit construction: "
            + ", ".join(missing)
        )
    if mode not in SUPPORTED_TAX_UNIT_CONSTRUCTION_MODES:
        raise ValueError(
            "Unsupported tax-unit construction mode "
            f"{mode!r}. Expected one of: "
            + ", ".join(sorted(SUPPORTED_TAX_UNIT_CONSTRUCTION_MODES))
        )

    person_assignments = pd.DataFrame(index=person.index)
    unit_key_records: list[tuple] = []
    unit_filing_records: list[str] = []

    household_unit_keys: list[tuple] = []
    household_roles: list[str] = []
    household_related_flags: list[bool] = []

    assignment_fn = (
        _determine_final_assignments_for_household_policyengine
        if mode == POLICYENGINE_MODE
        else _determine_final_assignments_for_household_census_documented
    )

    for household_id, household in person.groupby("PH_SEQ", sort=False):
        household_people = _prepare_household_people(household, int(household_id))
        (
            unit_key_by_person,
            roles_by_person,
            filing_status_by_unit,
            related_to_head_or_spouse,
        ) = assignment_fn(household_people, year)

        for row_index in household.index:
            unit_key = (int(household_id),) + tuple(unit_key_by_person[row_index])
            household_unit_keys.append(unit_key)
            household_roles.append(roles_by_person[row_index])
            household_related_flags.append(related_to_head_or_spouse[row_index])

        for unit_key, filing_status in filing_status_by_unit.items():
            unit_key_records.append((int(household_id),) + tuple(unit_key))
            unit_filing_records.append(filing_status)

    dense_unit_ids = {
        unit_key: unit_id
        for unit_id, unit_key in enumerate(dict.fromkeys(household_unit_keys), start=1)
    }
    person_assignments["TAX_ID"] = np.array(
        [dense_unit_ids[unit_key] for unit_key in household_unit_keys],
        dtype=np.int64,
    )
    person_assignments["tax_unit_role_input"] = np.array(household_roles).astype("S")
    person_assignments["is_related_to_head_or_spouse"] = np.array(
        household_related_flags,
        dtype=bool,
    )

    tax_unit = pd.DataFrame(
        {
            "TAX_ID": np.array(
                [dense_unit_ids[unit_key] for unit_key in unit_key_records],
                dtype=np.int64,
            ),
            "filing_status_input": np.array(unit_filing_records).astype("S"),
        }
    ).drop_duplicates("TAX_ID")
    tax_unit = tax_unit.sort_values("TAX_ID").reset_index(drop=True)

    return person_assignments, tax_unit
