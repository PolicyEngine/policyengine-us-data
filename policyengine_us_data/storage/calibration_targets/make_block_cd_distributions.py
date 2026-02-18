"""
Generate P(block|CD) distributions from Census block-level data.

Uses 119th Congress block assignments and 2020 Census block populations.
Saves to storage/block_cd_distributions.parquet for use by block_assignment.py.
"""

import pandas as pd
import us

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.storage.calibration_targets.make_district_mapping import (
    fetch_block_to_district_map,
    fetch_block_population,
)


def build_block_cd_distributions():
    """
    Build P(block|CD) distributions from Census block data.

    Algorithm:
    1. Get block → CD mapping (119th Congress)
    2. Get block population (2020 Census)
    3. Merge to get populated blocks with CD assignments
    4. Calculate P(block|CD) = pop(block) / pop(CD)
    5. Save as parquet (more efficient for large data)
    """
    print("Building P(block|CD) distributions from Census block data...")

    # Step 1: Block to CD mapping (119th Congress)
    print("\nFetching 119th Congress block-to-CD mapping...")
    bef = fetch_block_to_district_map(119)
    # Filter out 'ZZ' (unassigned blocks)
    bef = bef[bef["CD119"] != "ZZ"]
    print(f"  {len(bef):,} blocks with CD assignments")

    # Step 2: Block population (all 50 states + DC)
    print("\nFetching block population data (this takes a few minutes)...")
    state_pops = []

    # Get 50 states + DC
    states_to_process = [
        s
        for s in us.states.STATES_AND_TERRITORIES
        if not s.is_territory and s.abbr not in ["ZZ"]
    ] + [us.states.DC]

    import time

    for i, s in enumerate(states_to_process):
        print(f"  {s.abbr} ({i + 1}/{len(states_to_process)})")
        for attempt in range(3):
            try:
                state_pops.append(fetch_block_population(s.abbr))
                break
            except Exception as e:
                if attempt < 2:
                    print(f"    Retry {attempt + 1} for {s.abbr}...")
                    time.sleep(2)
                else:
                    print(f"    Warning: Failed to fetch {s.abbr}: {e}")
                    continue

    block_pop = pd.concat(state_pops, ignore_index=True)
    print(f"  Total blocks with population: {len(block_pop):,}")

    # Step 3: Merge block data
    print("\nMerging block data...")
    df = bef.merge(block_pop, on="GEOID", how="inner")
    print(f"  Matched blocks: {len(df):,}")

    # Filter to blocks with non-zero population
    df = df[df["POP20"] > 0]
    print(f"  Populated blocks: {len(df):,}")

    df["state_fips"] = df["GEOID"].str[:2]

    # Create CD geoid in our format: state_fips * 100 + district
    # Examples: AL-1 = 101, NY-10 = 3610, DC = 1198
    df["cd_geoid"] = df["state_fips"].astype(int) * 100 + df["CD119"].astype(
        int
    )

    # Step 4: Calculate P(block|CD)
    print("\nCalculating block probabilities...")
    cd_totals = df.groupby("cd_geoid")["POP20"].transform("sum")
    df["probability"] = df["POP20"] / cd_totals

    # Verify probabilities sum to 1 for each CD
    cd_sums = df.groupby("cd_geoid")["probability"].sum()
    bad_sums = cd_sums[~cd_sums.between(0.9999, 1.0001)]
    if len(bad_sums) > 0:
        print(f"Warning: {len(bad_sums)} CDs don't sum to 1.0")

    # Step 5: Prepare output
    output = df[["cd_geoid", "GEOID", "probability"]].rename(
        columns={"GEOID": "block_geoid"}
    )
    output = output.sort_values(
        ["cd_geoid", "probability"], ascending=[True, False]
    )

    # Step 6: Save as gzipped CSV (parquet requires pyarrow)
    output_path = STORAGE_FOLDER / "block_cd_distributions.csv.gz"
    output.to_csv(output_path, index=False, compression="gzip")
    print(f"\nSaved to {output_path}")
    print(f"  Total rows: {len(output):,}")
    print(f"  Unique CDs: {output['cd_geoid'].nunique()}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Print some stats
    blocks_per_cd = output.groupby("cd_geoid").size()
    print(f"\nBlocks per CD:")
    print(f"  Min: {blocks_per_cd.min():,}")
    print(f"  Median: {blocks_per_cd.median():,.0f}")
    print(f"  Max: {blocks_per_cd.max():,}")


if __name__ == "__main__":
    build_block_cd_distributions()
