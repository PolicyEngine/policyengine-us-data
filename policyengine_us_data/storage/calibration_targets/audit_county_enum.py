"""
Audit County enum against Census 2020 data.

Identifies bogus entries (counties assigned to wrong states, non-existent
combinations, encoding issues) and generates the INVALID_COUNTY_NAMES set
for use in county_assignment.py.
"""

import re
import requests
import pandas as pd
from io import StringIO
from collections import defaultdict

from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
    County,
)


def audit_county_enum():
    """
    Compare County enum entries against Census 2020 county reference.

    Returns categorized list of invalid entries:
    - wrong_state: county exists but in different state
    - non_existent: county name doesn't exist anywhere
    - encoding_issue: likely character encoding mismatch
    """
    print("Downloading Census 2020 county reference...")
    url = "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"
    response = requests.get(url, timeout=60)
    census_df = pd.read_csv(
        StringIO(response.text),
        delimiter="|",
        dtype=str,
        usecols=["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"],
    )

    # Build Census valid (state, normalized_county_name) pairs
    census_valid = set()
    county_to_states = defaultdict(set)

    for _, row in census_df.iterrows():
        state = row["STATE"]
        county_name = row["COUNTYNAME"].upper()
        # Apply same normalization as make_county_cd_distributions.py
        normalized = re.sub(r"[.'\"]", "", county_name)
        normalized = normalized.replace("-", "_")
        normalized = normalized.replace(" ", "_")

        census_valid.add((state, normalized))
        county_to_states[normalized].add(state)

    print(f"Census has {len(census_valid)} valid (state, county) pairs")

    # Audit each County enum entry
    invalid_entries = {
        "wrong_state": [],
        "non_existent": [],
        "encoding_issue": [],
    }
    valid_count = 0

    for name in County._member_names_:
        if name == "UNKNOWN":
            continue

        # Parse state code (last 2 chars)
        state = name[-2:]
        county_part = name[:-3]  # Remove _XX suffix

        if (state, county_part) in census_valid:
            valid_count += 1
        else:
            # Check if county exists in any state
            if county_part in county_to_states:
                correct_states = county_to_states[county_part]
                invalid_entries["wrong_state"].append(
                    (name, state, list(correct_states))
                )
            elif "Ñ" in name or "Í" in name or "Ó" in name or "Á" in name:
                invalid_entries["encoding_issue"].append((name, state))
            else:
                invalid_entries["non_existent"].append((name, state))

    print(f"\nAudit Results:")
    print(f"  Valid entries: {valid_count}")
    print(
        f"  Wrong state: {len(invalid_entries['wrong_state'])} "
        "(county exists in different state)"
    )
    print(
        f"  Non-existent: {len(invalid_entries['non_existent'])} "
        "(county name doesn't exist)"
    )
    print(
        f"  Encoding issues: {len(invalid_entries['encoding_issue'])} "
        "(special character mismatch)"
    )

    total_invalid = sum(len(v) for v in invalid_entries.values())
    print(f"  TOTAL INVALID: {total_invalid}")

    return invalid_entries, county_to_states


def print_categorized_report(invalid_entries, county_to_states):
    """Print detailed report of invalid entries."""
    print("\n" + "=" * 60)
    print("WRONG STATE ASSIGNMENTS")
    print("=" * 60)
    for name, wrong_state, correct_states in sorted(
        invalid_entries["wrong_state"]
    ):
        print(f"  {name}")
        print(f"    Listed as: {wrong_state}")
        print(f"    Actually exists in: {', '.join(sorted(correct_states))}")

    print("\n" + "=" * 60)
    print("NON-EXISTENT COMBINATIONS")
    print("=" * 60)
    for name, state in sorted(invalid_entries["non_existent"]):
        print(f"  {name}")

    print("\n" + "=" * 60)
    print("ENCODING ISSUES")
    print("=" * 60)
    for name, state in sorted(invalid_entries["encoding_issue"]):
        print(f"  {name}")


def generate_invalid_county_names_set(invalid_entries):
    """Generate Python set literal for INVALID_COUNTY_NAMES."""
    all_invalid = []

    for name, _, _ in invalid_entries["wrong_state"]:
        all_invalid.append(name)
    for name, _ in invalid_entries["non_existent"]:
        all_invalid.append(name)
    for name, _ in invalid_entries["encoding_issue"]:
        all_invalid.append(name)

    all_invalid.sort()

    print("\n" + "=" * 60)
    print("INVALID_COUNTY_NAMES SET (copy to county_assignment.py)")
    print("=" * 60)
    print("INVALID_COUNTY_NAMES = {")
    for name in all_invalid:
        print(f'    "{name}",')
    print("}")

    return set(all_invalid)


if __name__ == "__main__":
    invalid_entries, county_to_states = audit_county_enum()
    print_categorized_report(invalid_entries, county_to_states)
    invalid_set = generate_invalid_county_names_set(invalid_entries)
    print(f"\nTotal entries to exclude: {len(invalid_set)}")
