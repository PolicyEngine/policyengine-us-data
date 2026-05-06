"""
Map ACS PUMS person records onto the CPS-like columns consumed by
``policyengine_us_data.datasets.cps.tax_unit_construction``.

Column contract:

ACS ``household_id`` or dense ``SERIALNO`` -> CPS ``PH_SEQ``
ACS ``SPORDER`` -> CPS ``A_LINENO``
ACS ``AGEP`` -> CPS ``A_AGE``
ACS ``MAR`` -> CPS ``A_MARITL`` using CPS-like codes:
    1 married spouse present, 3 married spouse absent, 4 widowed,
    5 divorced, 6 separated, 7 never married or under age 15
ACS ``RELSHIPP`` (2019+) or ``RELP`` (pre-2019) -> CPS ``A_EXPRRP``
ACS ``WAGP`` -> CPS ``WSAL_VAL``
ACS ``SEMP`` -> CPS ``SEMP_VAL``
ACS ``INTP`` -> CPS ``INT_VAL``
ACS ``OIP`` plus ``PAP`` -> CPS ``OI_VAL``
ACS ``RETP`` -> CPS ``PNSN_VAL``
ACS ``SSP`` plus ``SSIP`` -> CPS ``SS_VAL``
ACS ``PINCP`` -> CPS ``PTOTVAL``
ACS ``DDRS``, ``DEAR``, ``DEYE``, ``DOUT``, ``DPHY``, ``DREM`` -> CPS
    ``PEDISDRS``, ``PEDISEAR``, ``PEDISEYE``, ``PEDISOUT``,
    ``PEDISPHY``, ``PEDISREM``
ACS ``SCH`` and ``SCHG`` -> CPS ``A_ENRLW``, ``A_FTPT``, ``A_HSCOL``

ACS does not provide universal spouse or parent pointers. This module links
the reference person's spouse directly from RELSHIPP/RELP, pairs common
non-reference in-law spouse patterns by age and marital status, and assigns
parent pointers for own children, foster children, and grandchildren when the
relationship-to-reference-person evidence supports it. The derived boolean
``acs_spouse_link_imputed`` and ``acs_parent_link_imputed`` columns identify
links that are heuristic rather than direct ACS relationship codes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ACS_REFERENCE_CODES = {20}
ACS_SPOUSE_CODES = {21, 23}
ACS_UNMARRIED_PARTNER_CODES = {22, 24}
ACS_CHILD_OF_REFERENCE_CODES = {25, 26, 27}
ACS_SIBLING_CODES = {28}
ACS_PARENT_CODES = {29}
ACS_GRANDCHILD_CODES = {30}
ACS_PARENT_IN_LAW_CODES = {31}
ACS_CHILD_IN_LAW_CODES = {32}
ACS_OTHER_RELATIVE_CODES = {33}
ACS_ROOMMATE_CODES = {34}
ACS_FOSTER_CHILD_CODES = {35}
ACS_OTHER_NONRELATIVE_CODES = {36, 37, 38}

OLD_REFERENCE_CODES = {0}
OLD_SPOUSE_CODES = {1}
OLD_CHILD_OF_REFERENCE_CODES = {2, 3, 4}
OLD_SIBLING_CODES = {5}
OLD_PARENT_CODES = {6}
OLD_GRANDCHILD_CODES = {7}
OLD_PARENT_IN_LAW_CODES = {8}
OLD_CHILD_IN_LAW_CODES = {9}
OLD_OTHER_RELATIVE_CODES = {10}
OLD_ROOMMATE_CODES = {11, 12}
OLD_UNMARRIED_PARTNER_CODES = {13}
OLD_FOSTER_CHILD_CODES = {14}
OLD_OTHER_NONRELATIVE_CODES = {15, 16, 17}

REQUIRED_ACS_TAX_UNIT_COLUMNS = {
    "SPORDER",
    "AGEP",
    "MAR",
    "SEX",
    "WAGP",
    "SEMP",
    "INTP",
    "RETP",
    "OIP",
    "PAP",
    "SSP",
    "SSIP",
    "PINCP",
    "SCH",
    "SCHG",
    "DDRS",
    "DEAR",
    "DEYE",
    "DOUT",
    "DPHY",
    "DREM",
}


def acs_person_to_cps_tax_unit_columns(person: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(person)
    original_index = person.index
    person = person.reset_index(drop=True)
    cps = pd.DataFrame(index=person.index)
    rel, relationship_system = _relationship_codes(person)
    household_id = _household_id(person)
    line_no = _numeric(person, "SPORDER").astype(int)
    age = _numeric(person, "AGEP").astype(int)

    spouse_line, spouse_link_imputed = _infer_spouse_lines(
        person=person,
        rel=rel,
        relationship_system=relationship_system,
        household_id=household_id,
        line_no=line_no,
        age=age,
    )
    parent1, parent2, parent_link_imputed = _infer_parent_lines(
        person=person,
        rel=rel,
        relationship_system=relationship_system,
        household_id=household_id,
        line_no=line_no,
        age=age,
        spouse_line=spouse_line,
    )

    cps["PH_SEQ"] = household_id.astype(int)
    cps["A_LINENO"] = line_no.astype(int)
    cps["A_AGE"] = age.astype(int)
    cps["A_SPOUSE"] = spouse_line.astype(int)
    cps["PEPAR1"] = parent1.astype(int)
    cps["PEPAR2"] = parent2.astype(int)
    cps["A_MARITL"] = _map_marital_status(person, spouse_line).astype(int)
    cps["A_EXPRRP"] = _map_relationship_to_cps(rel, relationship_system, person)

    cps["WSAL_VAL"] = _numeric(person, "WAGP")
    cps["SEMP_VAL"] = _numeric(person, "SEMP")
    cps["FRSE_VAL"] = 0.0
    cps["INT_VAL"] = _numeric(person, "INTP")
    cps["DIV_VAL"] = 0.0
    cps["RNT_VAL"] = 0.0
    cps["CAP_VAL"] = 0.0
    cps["UC_VAL"] = 0.0
    cps["OI_VAL"] = _numeric(person, "OIP") + _numeric(person, "PAP")
    cps["ANN_VAL"] = 0.0
    cps["PNSN_VAL"] = _numeric(person, "RETP")
    cps["SS_VAL"] = _numeric(person, "SSP") + _numeric(person, "SSIP")
    cps["PTOTVAL"] = _numeric(person, "PINCP")

    disability_map = {
        "DDRS": "PEDISDRS",
        "DEAR": "PEDISEAR",
        "DEYE": "PEDISEYE",
        "DOUT": "PEDISOUT",
        "DPHY": "PEDISPHY",
        "DREM": "PEDISREM",
    }
    for acs_column, cps_column in disability_map.items():
        cps[cps_column] = (_numeric(person, acs_column) == 1).astype(int)

    school = _numeric(person, "SCH").astype(int)
    grade = _numeric(person, "SCHG").astype(int)
    enrolled = school.isin([2, 3])
    cps["A_ENRLW"] = enrolled.astype(int)
    cps["A_FTPT"] = enrolled.astype(int)
    cps["A_HSCOL"] = np.select(
        [grade.between(1, 14), grade.between(15, 16)],
        [1, 2],
        default=0,
    ).astype(int)

    cps["acs_spouse_link_imputed"] = spouse_link_imputed.astype(bool)
    cps["acs_parent_link_imputed"] = parent_link_imputed.astype(bool)
    cps.index = original_index
    return cps


def _validate_required_columns(person: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_ACS_TAX_UNIT_COLUMNS.difference(person.columns))
    if "household_id" not in person.columns and "SERIALNO" not in person.columns:
        missing.append("household_id or SERIALNO")
    if "RELSHIPP" not in person.columns and "RELP" not in person.columns:
        missing.append("RELSHIPP or RELP")
    if missing:
        raise KeyError(
            "Missing required ACS columns for tax-unit construction: "
            + ", ".join(missing)
            + ". Regenerate the raw Census ACS dataset if this came from a "
            "cached census_acs file."
        )


def _numeric(person: pd.DataFrame, column: str, default=0) -> pd.Series:
    if column not in person.columns:
        return pd.Series(default, index=person.index)
    return pd.to_numeric(person[column], errors="coerce").fillna(default)


def _household_id(person: pd.DataFrame) -> pd.Series:
    if "household_id" in person.columns:
        return _numeric(person, "household_id").astype(int)
    codes, _ = pd.factorize(person["SERIALNO"], sort=False)
    return pd.Series(codes + 1, index=person.index)


def _relationship_codes(person: pd.DataFrame) -> tuple[pd.Series, str]:
    if "RELSHIPP" in person.columns:
        rel = _numeric(person, "RELSHIPP", default=-1).astype(int)
        if (rel > 0).any():
            return rel, "RELSHIPP"
    if "RELP" in person.columns:
        rel = _numeric(person, "RELP", default=-1).astype(int)
        if (rel >= 0).any():
            return rel, "RELP"

    raise ValueError(
        "ACS relationship columns contain no valid RELSHIPP or RELP codes."
    )


def _codes_for(system: str, relshipp_codes: set[int], relp_codes: set[int]) -> set[int]:
    return relshipp_codes if system == "RELSHIPP" else relp_codes


def _reference_codes(system: str) -> set[int]:
    return _codes_for(system, ACS_REFERENCE_CODES, OLD_REFERENCE_CODES)


def _spouse_codes(system: str) -> set[int]:
    return _codes_for(system, ACS_SPOUSE_CODES, OLD_SPOUSE_CODES)


def _child_codes(system: str) -> set[int]:
    return _codes_for(
        system, ACS_CHILD_OF_REFERENCE_CODES, OLD_CHILD_OF_REFERENCE_CODES
    )


def _foster_child_codes(system: str) -> set[int]:
    return _codes_for(system, ACS_FOSTER_CHILD_CODES, OLD_FOSTER_CHILD_CODES)


def _grandchild_codes(system: str) -> set[int]:
    return _codes_for(system, ACS_GRANDCHILD_CODES, OLD_GRANDCHILD_CODES)


def _child_in_law_codes(system: str) -> set[int]:
    return _codes_for(system, ACS_CHILD_IN_LAW_CODES, OLD_CHILD_IN_LAW_CODES)


def _parent_codes(system: str) -> set[int]:
    return _codes_for(system, ACS_PARENT_CODES, OLD_PARENT_CODES)


def _parent_in_law_codes(system: str) -> set[int]:
    return _codes_for(system, ACS_PARENT_IN_LAW_CODES, OLD_PARENT_IN_LAW_CODES)


def _map_marital_status(person: pd.DataFrame, spouse_line: pd.Series) -> pd.Series:
    mar = _numeric(person, "MAR").astype(int)
    result = pd.Series(7, index=person.index)
    result[mar == 2] = 4
    result[mar == 3] = 5
    result[mar == 4] = 6
    result[(mar == 1) & (spouse_line > 0)] = 1
    result[(mar == 1) & (spouse_line <= 0)] = 3
    return result


def _map_relationship_to_cps(
    rel: pd.Series,
    system: str,
    person: pd.DataFrame,
) -> pd.Series:
    result = pd.Series(14, index=rel.index, dtype=int)
    household_size = rel.groupby(_household_id(person)).transform("size")
    result[rel.isin(_reference_codes(system))] = np.where(
        household_size[rel.isin(_reference_codes(system))] > 1,
        1,
        2,
    )
    spouse_mask = rel.isin(_spouse_codes(system))
    sex = _numeric(person, "SEX").astype(int)
    result[spouse_mask] = np.where(sex[spouse_mask] == 1, 3, 4)
    result[rel.isin(_child_codes(system))] = 5
    result[rel.isin(_foster_child_codes(system))] = 11
    result[rel.isin(_grandchild_codes(system))] = 7
    result[rel.isin(_codes_for(system, ACS_SIBLING_CODES, OLD_SIBLING_CODES))] = 9
    result[rel.isin(_parent_codes(system))] = 8
    result[
        rel.isin(
            _codes_for(
                system,
                ACS_PARENT_IN_LAW_CODES
                | ACS_CHILD_IN_LAW_CODES
                | ACS_OTHER_RELATIVE_CODES,
                OLD_PARENT_IN_LAW_CODES
                | OLD_CHILD_IN_LAW_CODES
                | OLD_OTHER_RELATIVE_CODES,
            )
        )
    ] = 10
    result[
        rel.isin(
            _codes_for(
                system,
                ACS_UNMARRIED_PARTNER_CODES | ACS_ROOMMATE_CODES,
                OLD_UNMARRIED_PARTNER_CODES | OLD_ROOMMATE_CODES,
            )
        )
    ] = 13
    result[
        rel.isin(
            _codes_for(
                system,
                ACS_OTHER_NONRELATIVE_CODES,
                OLD_OTHER_NONRELATIVE_CODES,
            )
        )
    ] = 14
    return result


def _infer_spouse_lines(
    person: pd.DataFrame,
    rel: pd.Series,
    relationship_system: str,
    household_id: pd.Series,
    line_no: pd.Series,
    age: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    spouse_line = pd.Series(0, index=person.index, dtype=int)
    imputed = pd.Series(False, index=person.index, dtype=bool)
    mar = _numeric(person, "MAR").astype(int)

    frame = pd.DataFrame(
        {
            "household_id": household_id,
            "line_no": line_no,
            "age": age,
            "rel": rel,
            "mar": mar,
            "sex": _numeric(person, "SEX").astype(int),
        },
        index=person.index,
    )

    for _, household in frame.groupby("household_id", sort=False):
        reference = household[
            household["rel"].isin(_reference_codes(relationship_system))
        ]
        if reference.empty:
            reference = household[household["line_no"] == household["line_no"].min()]
        reference_index = reference.index[0]
        reference_line = int(frame.loc[reference_index, "line_no"])

        direct_spouses = household[
            household["rel"].isin(_spouse_codes(relationship_system))
            & (household["mar"] == 1)
        ]
        if not direct_spouses.empty and frame.loc[reference_index, "mar"] == 1:
            spouse_index = direct_spouses.sort_values("line_no").index[0]
            spouse_line.loc[reference_index] = int(frame.loc[spouse_index, "line_no"])
            spouse_line.loc[spouse_index] = reference_line

        unlinked = household[
            (household["mar"] == 1)
            & (household["age"] >= 18)
            & (spouse_line.loc[household.index] <= 0)
        ].copy()
        remaining = set(unlinked.index)
        for index in sorted(remaining, key=lambda item: frame.loc[item, "line_no"]):
            if index not in remaining:
                continue
            candidate_indexes = [
                candidate for candidate in remaining if candidate != index
            ]
            scored_candidates = []
            for candidate in candidate_indexes:
                score = _spouse_pair_score(
                    frame.loc[index],
                    frame.loc[candidate],
                    relationship_system,
                )
                if score is not None:
                    scored_candidates.append((score, candidate))
            if not scored_candidates:
                continue
            _, spouse_index = max(scored_candidates)
            spouse_line.loc[index] = int(frame.loc[spouse_index, "line_no"])
            spouse_line.loc[spouse_index] = int(frame.loc[index, "line_no"])
            imputed.loc[[index, spouse_index]] = True
            remaining.discard(index)
            remaining.discard(spouse_index)

    return spouse_line, imputed


def _spouse_pair_score(
    person_a: pd.Series,
    person_b: pd.Series,
    relationship_system: str,
) -> tuple[int, int, int] | None:
    rel_a = int(person_a["rel"])
    rel_b = int(person_b["rel"])
    age_gap = abs(int(person_a["age"]) - int(person_b["age"]))
    if age_gap > 20:
        return None

    child_codes = _child_codes(relationship_system)
    child_in_law_codes = _child_in_law_codes(relationship_system)
    parent_codes = _parent_codes(relationship_system)
    parent_in_law_codes = _parent_in_law_codes(relationship_system)
    pair = {rel_a, rel_b}
    if pair & child_codes and pair & child_in_law_codes:
        return (100, -age_gap, -min(int(person_a["line_no"]), int(person_b["line_no"])))
    if pair & parent_codes and pair & parent_in_law_codes:
        return (90, -age_gap, -min(int(person_a["line_no"]), int(person_b["line_no"])))
    return None


def _infer_parent_lines(
    person: pd.DataFrame,
    rel: pd.Series,
    relationship_system: str,
    household_id: pd.Series,
    line_no: pd.Series,
    age: pd.Series,
    spouse_line: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    parent1 = pd.Series(0, index=person.index, dtype=int)
    parent2 = pd.Series(0, index=person.index, dtype=int)
    imputed = pd.Series(False, index=person.index, dtype=bool)
    frame = pd.DataFrame(
        {
            "household_id": household_id,
            "line_no": line_no,
            "age": age,
            "rel": rel,
            "spouse_line": spouse_line,
        },
        index=person.index,
    )

    for _, household in frame.groupby("household_id", sort=False):
        reference = household[
            household["rel"].isin(_reference_codes(relationship_system))
        ]
        if reference.empty:
            reference = household[household["line_no"] == household["line_no"].min()]
        reference_index = reference.index[0]
        reference_line = int(frame.loc[reference_index, "line_no"])
        reference_spouse_line = int(frame.loc[reference_index, "spouse_line"])

        own_child_mask = household["rel"].isin(
            _child_codes(relationship_system) | _foster_child_codes(relationship_system)
        )
        for index in household[own_child_mask].index:
            parent1.loc[index] = reference_line
            if reference_spouse_line > 0:
                parent2.loc[index] = reference_spouse_line

        grandchild_indexes = household[
            household["rel"].isin(_grandchild_codes(relationship_system))
        ].index
        parent_candidates = household[
            household["rel"].isin(
                _child_codes(relationship_system)
                | _child_in_law_codes(relationship_system)
            )
        ]
        for index in grandchild_indexes:
            possible = parent_candidates[
                (parent_candidates["age"] - frame.loc[index, "age"]).between(15, 55)
            ].copy()
            if possible.empty:
                continue
            possible["score"] = -(possible["age"] - frame.loc[index, "age"] - 30).abs()
            selected_index = possible.sort_values(
                ["score", "age", "line_no"],
                ascending=[False, False, True],
            ).index[0]
            selected_line = int(frame.loc[selected_index, "line_no"])
            parent1.loc[index] = selected_line
            selected_spouse_line = int(frame.loc[selected_index, "spouse_line"])
            if selected_spouse_line > 0:
                parent2.loc[index] = selected_spouse_line
            imputed.loc[index] = True

    return parent1, parent2, imputed
