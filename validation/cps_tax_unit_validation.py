"""
Compare constructed CPS tax units against Census TAX_ID partitions.

The comparison is done within households using tax-unit member partitions rather
than raw TAX_ID values, since constructed tax-unit identifiers are renumbered.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from policyengine_us_data.datasets.cps.tax_unit_construction import (
    POLICYENGINE_MODE,
    SUPPORTED_TAX_UNIT_CONSTRUCTION_MODES,
    construct_tax_units,
)
from policyengine_us_data.datasets.cps.tax_unit_rule_helpers import (
    CPSRelationshipCode,
    qualifying_child_age_test,
)


DEFAULT_USECOLS = [
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    "A_MARITL",
    "A_SPOUSE",
    "PECOHAB",
    "PEPAR1",
    "PEPAR2",
    "A_EXPRRP",
    "A_ENRLW",
    "A_FTPT",
    "A_HSCOL",
    "WSAL_VAL",
    "SEMP_VAL",
    "FRSE_VAL",
    "INT_VAL",
    "DIV_VAL",
    "RNT_VAL",
    "CAP_VAL",
    "UC_VAL",
    "OI_VAL",
    "ANN_VAL",
    "PNSN_VAL",
    "PTOTVAL",
    "SS_VAL",
    "PEDISDRS",
    "PEDISEAR",
    "PEDISEYE",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
    "TAX_ID",
]


def load_person_file(
    input_path: Path,
    csv_name: str | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    usecols = usecols or DEFAULT_USECOLS
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            selected_name = csv_name
            if selected_name is None:
                matches = [
                    name
                    for name in zf.namelist()
                    if name.lower().startswith("pppub")
                    and name.lower().endswith(".csv")
                ]
                if not matches:
                    raise FileNotFoundError(
                        f"No pppub*.csv person file found in {input_path}."
                    )
                selected_name = sorted(matches)[0]
            with zf.open(selected_name) as f:
                return pd.read_csv(f, usecols=usecols, low_memory=False)
    return pd.read_csv(input_path, usecols=usecols, low_memory=False)


def _has_tax_unit_student_evidence(row) -> bool:
    return int(row.A_ENRLW) == 1 and (
        int(row.A_FTPT) == 1 or int(row.A_HSCOL) in {1, 2}
    )


def compute_tax_unit_comparison(
    person: pd.DataFrame,
    year: int,
    mode: str = POLICYENGINE_MODE,
) -> dict:
    person = person.copy()
    person["CENSUS_TAX_ID"] = person["TAX_ID"].astype(int)
    assignments, _ = construct_tax_units(person, year=year, mode=mode)
    person = person.join(assignments.rename(columns={"TAX_ID": "CONSTRUCTED_TAX_ID"}))

    rel_own_child = CPSRelationshipCode.OWN_CHILD.value
    rel_grandchild = CPSRelationshipCode.GRANDCHILD.value
    other_adult_rels = {
        CPSRelationshipCode.PARENT.value,
        CPSRelationshipCode.SIBLING.value,
        CPSRelationshipCode.OTHER_RELATIVE.value,
    }

    exact_match = 0
    same_unit_count = 0
    constructed_lt = 0
    constructed_gt = 0
    person_same_unit = 0
    persons_total = len(person)
    mismatched_households = 0
    mismatched_person_rows = 0
    mismatch_buckets = {
        "construct_fewer": 0,
        "own_child_adult": 0,
        "other_adult_relative": 0,
        "grandchild_present": 0,
        "no_parent_pointer_minor": 0,
        "reciprocal_married_pair_split_by_census": 0,
    }
    reciprocal_spouse_split_households = 0
    exact_match_excluding_census_spouse_split = 0
    non_spouse_split_households = 0

    reciprocal_spouse_people = 0
    reciprocal_spouse_split_people_census = 0
    reciprocal_spouse_split_people_constructed = 0
    qual_child_pointer_people = 0
    qual_child_pointer_split_census = 0
    qual_child_pointer_split_constructed = 0
    minor_people = 0
    minor_singleton_census = 0
    minor_singleton_constructed = 0
    young_parent_pointer_people = 0
    young_parent_pointer_split_census = 0
    young_parent_pointer_split_constructed = 0

    for household_id, household in person.groupby("PH_SEQ", sort=False):
        household = household.copy()
        household["line_no"] = household["A_LINENO"].astype(int)

        census_groups = [
            frozenset(group["line_no"].tolist())
            for _, group in household.groupby("CENSUS_TAX_ID", sort=False)
        ]
        constructed_groups = [
            frozenset(group["line_no"].tolist())
            for _, group in household.groupby("CONSTRUCTED_TAX_ID", sort=False)
        ]
        census_partition = frozenset(census_groups)
        constructed_partition = frozenset(constructed_groups)

        census_unit_for_line = {
            line_no: group for group in census_groups for line_no in group
        }
        constructed_unit_for_line = {
            line_no: group for group in constructed_groups for line_no in group
        }

        household_exact = census_partition == constructed_partition
        if household_exact:
            exact_match += 1
        else:
            mismatched_households += 1
            mismatched_person_rows += sum(
                census_unit_for_line[line_no]
                != constructed_unit_for_line[line_no]
                for line_no in household["line_no"]
            )

        if len(census_groups) == len(constructed_groups):
            same_unit_count += 1
        elif len(constructed_groups) < len(census_groups):
            constructed_lt += 1
        else:
            constructed_gt += 1

        person_same_unit += sum(
            census_unit_for_line[line_no] == constructed_unit_for_line[line_no]
            for line_no in household["line_no"]
        )

        line_to_row = {
            int(row.A_LINENO): row for row in household.itertuples(index=False)
        }
        line_to_census_tax = {
            int(row.A_LINENO): int(row.CENSUS_TAX_ID)
            for row in household.itertuples(index=False)
        }
        line_to_constructed_tax = {
            int(row.A_LINENO): int(row.CONSTRUCTED_TAX_ID)
            for row in household.itertuples(index=False)
        }

        household_has_census_spouse_split = False
        for row in household.itertuples(index=False):
            spouse_line = int(row.A_SPOUSE) if pd.notna(row.A_SPOUSE) else 0
            line_no = int(row.A_LINENO)
            if spouse_line <= 0:
                continue
            spouse = line_to_row.get(spouse_line)
            if spouse is None:
                continue
            spouse_spouse_line = (
                int(spouse.A_SPOUSE) if pd.notna(spouse.A_SPOUSE) else 0
            )
            if spouse_spouse_line != line_no:
                continue
            reciprocal_spouse_people += 1
            if line_to_census_tax[line_no] != line_to_census_tax[spouse_line]:
                reciprocal_spouse_split_people_census += 1
                household_has_census_spouse_split = True
            if (
                line_to_constructed_tax[line_no]
                != line_to_constructed_tax[spouse_line]
            ):
                reciprocal_spouse_split_people_constructed += 1

        if household_has_census_spouse_split:
            reciprocal_spouse_split_households += 1
        else:
            non_spouse_split_households += 1
            if household_exact:
                exact_match_excluding_census_spouse_split += 1

        for row in household.itertuples(index=False):
            line_no = int(row.A_LINENO)
            age = int(row.A_AGE)
            parent_lines = [
                int(value)
                for value in (row.PEPAR1, row.PEPAR2)
                if pd.notna(value)
                and int(value) > 0
                and int(value) in line_to_row
            ]
            is_student = _has_tax_unit_student_evidence(row)
            disability = any(
                int(getattr(row, col)) == 1
                for col in [
                    "PEDISDRS",
                    "PEDISEAR",
                    "PEDISEYE",
                    "PEDISOUT",
                    "PEDISPHY",
                    "PEDISREM",
                ]
            )

            if age < 18:
                minor_people += 1
                if len(census_unit_for_line[line_no]) == 1:
                    minor_singleton_census += 1
                if len(constructed_unit_for_line[line_no]) == 1:
                    minor_singleton_constructed += 1

            if parent_lines and qualifying_child_age_test(
                age, is_student, disability
            ):
                qual_child_pointer_people += 1
                if not any(
                    line_to_census_tax[line_no] == line_to_census_tax[parent_line]
                    for parent_line in parent_lines
                ):
                    qual_child_pointer_split_census += 1
                if not any(
                    line_to_constructed_tax[line_no]
                    == line_to_constructed_tax[parent_line]
                    for parent_line in parent_lines
                ):
                    qual_child_pointer_split_constructed += 1

            if parent_lines and 18 <= age <= 23:
                young_parent_pointer_people += 1
                if not any(
                    line_to_census_tax[line_no] == line_to_census_tax[parent_line]
                    for parent_line in parent_lines
                ):
                    young_parent_pointer_split_census += 1
                if not any(
                    line_to_constructed_tax[line_no]
                    == line_to_constructed_tax[parent_line]
                    for parent_line in parent_lines
                ):
                    young_parent_pointer_split_constructed += 1

        if not household_exact:
            if len(constructed_groups) < len(census_groups):
                mismatch_buckets["construct_fewer"] += 1
            if any(
                int(row.A_EXPRRP) == rel_own_child and int(row.A_AGE) >= 18
                for row in household.itertuples(index=False)
            ):
                mismatch_buckets["own_child_adult"] += 1
            if any(
                int(row.A_EXPRRP) in other_adult_rels and int(row.A_AGE) >= 18
                for row in household.itertuples(index=False)
            ):
                mismatch_buckets["other_adult_relative"] += 1
            if any(
                int(row.A_EXPRRP) == rel_grandchild
                for row in household.itertuples(index=False)
            ):
                mismatch_buckets["grandchild_present"] += 1
            if any(
                int(row.A_AGE) < 18
                and all(
                    (not pd.notna(value)) or int(value) <= 0
                    for value in (row.PEPAR1, row.PEPAR2)
                )
                for row in household.itertuples(index=False)
            ):
                mismatch_buckets["no_parent_pointer_minor"] += 1
            if household_has_census_spouse_split:
                mismatch_buckets["reciprocal_married_pair_split_by_census"] += 1

    households_total = int(person["PH_SEQ"].nunique())
    return {
        "mode": mode,
        "summary": {
            "households_total": households_total,
            "persons_total": int(persons_total),
            "household_exact_match_pct": round(
                100 * exact_match / households_total, 2
            ),
            "household_exact_match_excluding_census_spouse_split_pct": round(
                100
                * exact_match_excluding_census_spouse_split
                / non_spouse_split_households,
                2,
            )
            if non_spouse_split_households
            else None,
            "census_spouse_split_households_pct": round(
                100 * reciprocal_spouse_split_households / households_total, 2
            ),
            "person_same_unit_pct": round(100 * person_same_unit / persons_total, 2),
            "households_same_unit_count_pct": round(
                100 * same_unit_count / households_total, 2
            ),
            "constructed_lt_census_pct": round(
                100 * constructed_lt / households_total, 2
            ),
            "constructed_gt_census_pct": round(
                100 * constructed_gt / households_total, 2
            ),
            "census_tax_units": int(person["CENSUS_TAX_ID"].nunique()),
            "constructed_tax_units": int(person["CONSTRUCTED_TAX_ID"].nunique()),
            "mismatched_households": int(mismatched_households),
            "mismatched_persons": int(mismatched_person_rows),
        },
        "mismatch_buckets_pct": {
            key: (
                round(100 * value / mismatched_households, 2)
                if mismatched_households
                else 0.0
            )
            for key, value in mismatch_buckets.items()
        },
        "legal_consistency": {
            "reciprocal_spouse_people_split_pct": {
                "census": round(
                    100 * reciprocal_spouse_split_people_census / reciprocal_spouse_people,
                    2,
                )
                if reciprocal_spouse_people
                else None,
                "constructed": round(
                    100
                    * reciprocal_spouse_split_people_constructed
                    / reciprocal_spouse_people,
                    2,
                )
                if reciprocal_spouse_people
                else None,
            },
            "qualifying_child_with_parent_pointer_split_pct": {
                "census": round(
                    100 * qual_child_pointer_split_census / qual_child_pointer_people,
                    2,
                )
                if qual_child_pointer_people
                else None,
                "constructed": round(
                    100
                    * qual_child_pointer_split_constructed
                    / qual_child_pointer_people,
                    2,
                )
                if qual_child_pointer_people
                else None,
            },
            "minor_singleton_pct": {
                "census": round(100 * minor_singleton_census / minor_people, 2)
                if minor_people
                else None,
                "constructed": round(
                    100 * minor_singleton_constructed / minor_people, 2
                )
                if minor_people
                else None,
            },
            "young_adult_with_parent_pointer_split_pct": {
                "census": round(
                    100
                    * young_parent_pointer_split_census
                    / young_parent_pointer_people,
                    2,
                )
                if young_parent_pointer_people
                else None,
                "constructed": round(
                    100
                    * young_parent_pointer_split_constructed
                    / young_parent_pointer_people,
                    2,
                )
                if young_parent_pointer_people
                else None,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare constructed CPS tax units against Census TAX_ID partitions."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the public-use CPS person CSV or zip archive containing it.",
    )
    parser.add_argument(
        "--csv-name",
        default=None,
        help="CSV name inside the zip archive. Defaults to the first pppub*.csv.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Tax year for construction. Default: %(default)s",
    )
    parser.add_argument(
        "--mode",
        default=POLICYENGINE_MODE,
        choices=sorted(SUPPORTED_TAX_UNIT_CONSTRUCTION_MODES),
        help=(
            "Tax-unit construction mode to benchmark. "
            f"Default: {POLICYENGINE_MODE}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    person = load_person_file(args.input_path, csv_name=args.csv_name)
    result = compute_tax_unit_comparison(person, year=args.year, mode=args.mode)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
