"""
Generate P(county|CD) distributions from Census block-level data.

Uses 119th Congress block assignments and 2020 Census block populations.
"""

import re
import requests
import pandas as pd
import us
from io import StringIO
from pathlib import Path

from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
    County,
)
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.storage.calibration_targets.make_district_mapping import (
    fetch_block_to_district_map,
    fetch_block_population,
)


def build_county_fips_to_enum_mapping() -> dict:
    """
    Build mapping from 5-digit county FIPS to County enum name.

    Downloads Census county FIPS file and matches to County enum names.
    """
    # Download Census county reference file
    url = "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"
    response = requests.get(url, timeout=60)
    df = pd.read_csv(
        StringIO(response.text),
        delimiter="|",
        dtype=str,
        usecols=["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"],
    )

    # Build set of valid enum names for lookup
    valid_enum_names = set(County._member_names_)

    fips_to_enum = {}
    missing = []

    for _, row in df.iterrows():
        county_fips = row["STATEFP"] + row["COUNTYFP"]
        state_code = row["STATE"]
        county_name = row["COUNTYNAME"]

        # Transform to enum name format:
        # 1. Uppercase
        # 2. Replace special chars (periods, apostrophes) with nothing
        # 3. Replace hyphens and spaces with underscores
        # 4. Append state code
        enum_name = county_name.upper()
        enum_name = re.sub(r"[.'\"]", "", enum_name)
        enum_name = enum_name.replace("-", "_")
        enum_name = enum_name.replace(" ", "_")
        enum_name = f"{enum_name}_{state_code}"

        if enum_name in valid_enum_names:
            fips_to_enum[county_fips] = enum_name
        else:
            missing.append((county_fips, county_name, state_code, enum_name))

    if missing:
        print(f"Warning: {len(missing)} counties not found in enum:")
        for fips, name, state, attempted in missing[:10]:
            print(f"  {fips}: {name} ({state}) -> tried '{attempted}'")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    print(f"Mapped {len(fips_to_enum)} counties to enum names")
    return fips_to_enum


def build_county_cd_distributions():
    """
    Build P(county|CD) distributions from Census block data.

    Algorithm:
    1. Get block → CD mapping (119th Congress)
    2. Get block population (2020 Census)
    3. Extract county FIPS from block GEOID (positions 0-4)
    4. Group by (CD, county_fips) and sum population
    5. Calculate P(county|CD) = pop(county,CD) / pop(CD)
    6. Map county FIPS to County enum names
    7. Save as CSV
    """
    print("Building P(county|CD) distributions from Census block data...")

    # Step 1: Block to CD mapping (119th Congress)
    print("\nFetching 119th Congress block-to-CD mapping...")
    bef = fetch_block_to_district_map(119)
    # Filter out 'ZZ' (unassigned blocks)
    bef = bef[bef["CD119"] != "ZZ"]
    print(f"  {len(bef):,} blocks with CD assignments")

    # Step 2: Block population (all 50 states + DC)
    print("\nFetching block population data (this takes a few minutes)...")
    state_pops = []

    # Get 50 states
    states_to_process = [
        s
        for s in us.states.STATES_AND_TERRITORIES
        if not s.is_territory and s.abbr not in ["ZZ"]
    ]
    # Note: DC excluded - handled as special case below (1 county, 1 CD)

    for i, s in enumerate(states_to_process):
        print(f"  {s.abbr} ({i + 1}/{len(states_to_process)})")
        state_pops.append(fetch_block_population(s.abbr))

    block_pop = pd.concat(state_pops, ignore_index=True)
    print(f"  Total blocks with population: {len(block_pop):,}")

    # Step 3: Merge and extract county FIPS
    print("\nMerging block data...")
    df = bef.merge(block_pop, on="GEOID", how="inner")
    print(f"  Matched blocks: {len(df):,}")

    df["county_fips"] = df["GEOID"].str[:5]
    df["state_fips"] = df["GEOID"].str[:2]

    # Create CD geoid in our format: state_fips * 100 + district
    # Examples: AL-1 = 101, NY-10 = 3610, DC = 1198
    df["cd_geoid"] = df["state_fips"].astype(int) * 100 + df["CD119"].astype(
        int
    )

    # Step 4: Aggregate by (CD, county)
    print("\nAggregating population by CD and county...")
    cd_county_pop = (
        df.groupby(["cd_geoid", "county_fips"])["POP20"].sum().reset_index()
    )
    print(f"  Unique CD-county pairs: {len(cd_county_pop):,}")

    # Step 5: Calculate P(county|CD)
    cd_totals = cd_county_pop.groupby("cd_geoid")["POP20"].transform("sum")
    cd_county_pop["probability"] = cd_county_pop["POP20"] / cd_totals

    # Step 5b: Filter out zero-probability entries (unpopulated county-CD pairs)
    pre_filter_count = len(cd_county_pop)
    cd_county_pop = cd_county_pop[cd_county_pop["probability"] > 0]
    filtered_count = pre_filter_count - len(cd_county_pop)
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} zero-probability entries")

    # Step 6: Map county FIPS to enum names
    print("\nMapping county FIPS to enum names...")
    fips_to_enum = build_county_fips_to_enum_mapping()
    cd_county_pop["county_name"] = cd_county_pop["county_fips"].map(
        fips_to_enum
    )

    # Check for unmapped counties
    unmapped = cd_county_pop[cd_county_pop["county_name"].isna()]
    if len(unmapped) > 0:
        print(f"Warning: {len(unmapped)} rows have unmapped county FIPS")
        # Drop unmapped and renormalize
        cd_county_pop = cd_county_pop.dropna(subset=["county_name"])
        cd_totals = cd_county_pop.groupby("cd_geoid")["POP20"].transform("sum")
        cd_county_pop["probability"] = cd_county_pop["POP20"] / cd_totals

    # Step 7: Add DC (special case: 1 county, 1 CD, probability = 1.0)
    # DC: cd_geoid = 1198 (state 11, district 98 at-large)
    dc_row = pd.DataFrame(
        {
            "cd_geoid": [1198],
            "county_name": ["DISTRICT_OF_COLUMBIA_DC"],
            "probability": [1.0],
        }
    )
    cd_county_pop = pd.concat([cd_county_pop, dc_row], ignore_index=True)

    # Step 8: Save CSV
    output = cd_county_pop[["cd_geoid", "county_name", "probability"]]
    output = output.sort_values(
        ["cd_geoid", "probability"], ascending=[True, False]
    )

    output_path = STORAGE_FOLDER / "county_cd_distributions.csv"
    output.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"  Total rows: {len(output):,}")
    print(f"  Unique CDs: {output['cd_geoid'].nunique()}")

    # Verify probabilities sum to 1 for each CD
    cd_sums = output.groupby("cd_geoid")["probability"].sum()
    bad_sums = cd_sums[~cd_sums.between(0.9999, 1.0001)]
    if len(bad_sums) > 0:
        print(f"Warning: {len(bad_sums)} CDs don't sum to 1.0")
    else:
        print("  All CD probabilities sum to 1.0 ✓")


if __name__ == "__main__":
    build_county_cd_distributions()
