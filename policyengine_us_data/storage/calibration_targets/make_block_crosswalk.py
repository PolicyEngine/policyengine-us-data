"""
Build comprehensive block-level geographic crosswalk from Census data.

Downloads Block Assignment Files (BAFs) for all states and creates a single
crosswalk file mapping block GEOID to:
- SLDU (State Legislative District Upper)
- SLDL (State Legislative District Lower)
- Place FIPS (City/CDP)
- PUMA (via tract lookup)

Data sources:
- BAFs: https://www2.census.gov/geo/docs/maps-data/data/baf2020/
- Tract-to-PUMA: https://www2.census.gov/geo/docs/maps-data/data/rel2020/
"""

import io
import requests
import zipfile
from pathlib import Path
import pandas as pd
import us

from policyengine_us_data.storage import STORAGE_FOLDER


BAF_BASE_URL = "https://www2.census.gov/geo/docs/maps-data/data/baf2020/"
TRACT_PUMA_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt"


def download_state_baf(state_fips: str, state_abbr: str) -> dict:
    """
    Download and parse Block Assignment Files for a state.

    Returns dict with DataFrames for SLDU, SLDL, Place.
    """
    url = f"{BAF_BASE_URL}BlockAssign_ST{state_fips}_{state_abbr}.zip"

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    results = {}

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # SLDU - State Legislative District Upper
        sldu_file = f"BlockAssign_ST{state_fips}_{state_abbr}_SLDU.txt"
        if sldu_file in z.namelist():
            df = pd.read_csv(z.open(sldu_file), sep="|", dtype=str)
            results["sldu"] = df.rename(
                columns={"BLOCKID": "block_geoid", "DISTRICT": "sldu"}
            )

        # SLDL - State Legislative District Lower
        sldl_file = f"BlockAssign_ST{state_fips}_{state_abbr}_SLDL.txt"
        if sldl_file in z.namelist():
            df = pd.read_csv(z.open(sldl_file), sep="|", dtype=str)
            results["sldl"] = df.rename(
                columns={"BLOCKID": "block_geoid", "DISTRICT": "sldl"}
            )

        # Place (City/CDP)
        place_file = (
            f"BlockAssign_ST{state_fips}_{state_abbr}_INCPLACE_CDP.txt"
        )
        if place_file in z.namelist():
            df = pd.read_csv(z.open(place_file), sep="|", dtype=str)
            results["place"] = df.rename(
                columns={"BLOCKID": "block_geoid", "PLACEFP": "place_fips"}
            )

        # VTD - Voting Tabulation District
        vtd_file = f"BlockAssign_ST{state_fips}_{state_abbr}_VTD.txt"
        if vtd_file in z.namelist():
            df = pd.read_csv(z.open(vtd_file), sep="|", dtype=str)
            # VTD has COUNTYFP and DISTRICT columns
            df["vtd"] = df["DISTRICT"]
            results["vtd"] = df[["BLOCKID", "vtd"]].rename(
                columns={"BLOCKID": "block_geoid"}
            )

    return results


def download_tract_puma_crosswalk() -> pd.DataFrame:
    """Download tract-to-PUMA crosswalk from Census."""
    df = pd.read_csv(TRACT_PUMA_URL, dtype=str)

    # Build tract GEOID (11 chars: state + county + tract)
    df["tract_geoid"] = df["STATEFP"] + df["COUNTYFP"] + df["TRACTCE"]
    df["puma"] = df["PUMA5CE"]

    return df[["tract_geoid", "puma"]]


def build_block_crosswalk():
    """
    Build comprehensive block-level geographic crosswalk.

    Creates block_crosswalk.csv.gz with columns:
    - block_geoid (15-char)
    - sldu (3-char state legislative upper)
    - sldl (3-char state legislative lower)
    - place_fips (5-char place/city FIPS)
    - puma (5-char PUMA via tract lookup)
    """
    print("Building comprehensive block geographic crosswalk...")

    # Download tract-to-PUMA crosswalk
    print("\nDownloading tract-to-PUMA crosswalk...")
    tract_puma = download_tract_puma_crosswalk()
    print(f"  {len(tract_puma):,} tract-PUMA mappings")

    # Process each state
    print("\nDownloading Block Assignment Files...")
    all_blocks = []

    states_to_process = [
        s
        for s in us.states.STATES_AND_TERRITORIES
        if not s.is_territory and s.abbr not in ["ZZ"]
    ]

    import time

    for i, s in enumerate(states_to_process):
        state_fips = s.fips
        print(f"  {s.abbr} ({i + 1}/{len(states_to_process)})")

        for attempt in range(3):
            try:
                bafs = download_state_baf(state_fips, s.abbr)

                # Start with SLDU as base (has all blocks)
                if "sldu" in bafs:
                    df = bafs["sldu"].copy()

                    # Merge other geographies
                    if "sldl" in bafs:
                        df = df.merge(
                            bafs["sldl"], on="block_geoid", how="left"
                        )
                    else:
                        df["sldl"] = None

                    if "place" in bafs:
                        df = df.merge(
                            bafs["place"], on="block_geoid", how="left"
                        )
                    else:
                        df["place_fips"] = None

                    if "vtd" in bafs:
                        df = df.merge(
                            bafs["vtd"], on="block_geoid", how="left"
                        )
                    else:
                        df["vtd"] = None

                    # Add tract GEOID for PUMA lookup
                    df["tract_geoid"] = df["block_geoid"].str[:11]

                    # Merge PUMA via tract
                    df = df.merge(tract_puma, on="tract_geoid", how="left")

                    # Drop tract_geoid (can be derived from block)
                    df = df.drop(columns=["tract_geoid"])

                    all_blocks.append(df)

                break
            except Exception as e:
                if attempt < 2:
                    print(f"    Retry {attempt + 1}...")
                    time.sleep(2)
                else:
                    print(f"    Warning: Failed to process {s.abbr}: {e}")

    # Combine all states
    print("\nCombining all states...")
    combined = pd.concat(all_blocks, ignore_index=True)
    print(f"  Total blocks: {len(combined):,}")

    # Save
    output_path = STORAGE_FOLDER / "block_crosswalk.csv.gz"
    combined.to_csv(output_path, index=False, compression="gzip")
    print(f"\nSaved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Stats
    print(f"\nCoverage:")
    print(f"  Blocks with SLDU: {combined['sldu'].notna().sum():,}")
    print(f"  Blocks with SLDL: {combined['sldl'].notna().sum():,}")
    print(f"  Blocks with Place: {combined['place_fips'].notna().sum():,}")
    print(f"  Blocks with PUMA: {combined['puma'].notna().sum():,}")


if __name__ == "__main__":
    build_block_crosswalk()
