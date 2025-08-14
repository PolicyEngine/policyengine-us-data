"""
This was built before finding out about the crosswalk provided by the
Missouri Census Data Center (MCDC) at the University of Missouri. This crosswalk can be
accessed at (https://mcdc.missouri.edu/applications/geocorr.html) and would be a logical place
to transition to, though since this is already built and new IRS SOI files may be available soon,
it may not be worth the effort to transition.

To see the definitive "before and after" of congressional redistricting following the 2020 census,
you should compare the block-level data from the 116th Congress to the 119th Congress.

This approach is necessary for states whose initial redistricting maps were altered due to legal
challenges and is aligned with the mapping files provided by the U.S. Census Bureau.

- **116th Congress (The "Before"):** This session (2019-2021) used the congressional maps
based on the 2010 census data. It serves as the stable pre-redistricting baseline, as these
maps were identical to those used by the 117th Congress. The Census Bureau's most recent files
for that decade correspond to the 116th Congress.

- **118th Congress (The "Interim" Stage):** In several states, the initial congressional maps drawn
for the 2022 elections were successfully challenged and invalidated by courts (e.g., for reasons of
partisan or racial gerrymandering). This required the use of temporary, court-ordered, or remedial
maps for the 2022 elections. Consequently, the 118th Congress (2023-2025) in these states represents
an interim stage, not the final outcome of the redistricting cycle.

- **119th Congress (The Definitive "After"):** Following these legal resolutions, new and more permanent
congressional maps were enacted ahead of the 2024 election cycle. The elections in November 2024 were
the first to use these new maps. Therefore, the 119th Congress (2025-2027) is the first to reflect the
final, settled mapping decisions based on the 2020 census data.

By comparing the 116th and 119th Congresses, you bypass the anomalous, non-final maps of the 118th Congress,
providing a clear analysis of the redistricting cycle's ultimate impact.
"""

import requests
import zipfile
import io
from pathlib import Path

import pandas as pd
import numpy as np
import us

from policyengine_us_data.storage import STORAGE_FOLDER, CALIBRATION_FOLDER


def fetch_block_to_district_map(congress: int) -> pd.DataFrame:
    """
    Fetches the Census Block Equivalency File (BEF) for a given Congress.

    This file maps every 2020 census block (GEOID) to its corresponding
    congressional district.

    Args:
        congress: The congressional session number (e.g., 118 or 119).

    Returns:
        A DataFrame with columns ['GEOID', f'CD{congress}'].
    """
    if congress == 116:
        url = "https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files/2019/116-congressional-district-bef/cd116.zip"
        zbytes = requests.get(url, timeout=120).content

        with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
            fname = "National_CD116.txt"
            bef = pd.read_csv(z.open(fname), dtype=str)
            bef.columns = bef.columns.str.strip()
            bef = bef.rename(columns={"BLOCKID": "GEOID"})
            return bef[["GEOID", f"CD{congress}"]]

    elif congress == 118:
        url = "https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files/2023/118-congressional-district-bef/cd118.zip"
        zbytes = requests.get(url, timeout=120).content

        with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
            fname = "National_CD118.txt"
            bef = pd.read_csv(z.open(fname), dtype=str)
            bef.columns = bef.columns.str.strip()
            district_col = [c for c in bef.columns if c != "GEOID"][0]
            bef = bef.rename(columns={district_col: f"CD{congress}"})
            return bef[["GEOID", f"CD{congress}"]]

    elif congress == 119:
        url = "https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files/2025/119-congressional-district-befs/cd119.zip"
        zbytes = requests.get(url, timeout=120).content

        with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
            fname = "NationalCD119.txt"
            bef = pd.read_csv(z.open(fname), sep=",", dtype=str)
            bef.columns = bef.columns.str.strip()
            bef = bef.rename(columns={"CDFP": f"CD{congress}"})
            return bef[["GEOID", f"CD{congress}"]]

    else:
        raise ValueError(
            f"Congress {congress} is not supported by this function."
        )


def fetch_block_population(state) -> pd.DataFrame:
    """
    Download & parse the 2020 PL-94-171 “legacy” files for one state.

    Parameters
    ----------
    state : str
        Two-letter state/territory postal code **or** full state name
        (e.g., "GA", "Georgia", "PR", "Puerto Rico").

    Returns
    -------
    pandas.DataFrame with columns GEOID (15-digit block code) and POP20.
    """
    BASE = (
        "https://www2.census.gov/programs-surveys/decennial/2020/data/"
        "01-Redistricting_File--PL_94-171/{dir}/{abbr}2020.pl.zip"
    )
    st = us.states.lookup(state)
    if st is None:
        raise ValueError(f"Unrecognised state name/abbr: {state}")

    # Build URL components -----------------------------------------------------
    dir_name = st.name.replace(" ", "_")
    abbr = st.abbr.lower()
    url = BASE.format(dir=dir_name, abbr=abbr)

    # Download and open the zip ------------------------------------------------
    zbytes = requests.get(url, timeout=120).content
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        raw = z.read(f"{abbr}geo2020.pl")
        try:
            geo_lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            geo_lines = raw.decode("latin-1").splitlines()

        p1_lines = z.read(f"{abbr}000012020.pl").decode("utf-8").splitlines()

    # ---------------- GEO file: keep blocks (SUMLEV 750) ----------------------
    geo_records = [
        (parts[7], parts[8][-15:])  # LOGRECNO, 15-digit block GEOID
        for ln in geo_lines
        if (parts := ln.split("|"))[2] == "750"  # summary level 750 = blocks
    ]
    geo_df = pd.DataFrame(geo_records, columns=["LOGRECNO", "GEOID"])

    # ---------------- P-file: pull total-population cell ----------------------
    p1_records = [
        (p[4], int(p[5])) for p in map(lambda x: x.split("|"), p1_lines)
    ]
    p1_df = pd.DataFrame(p1_records, columns=["LOGRECNO", "P0010001"])

    # ---------------- Merge & finish -----------------------------------------
    return (
        geo_df.merge(p1_df, on="LOGRECNO", how="left")
        .assign(POP20=lambda d: d["P0010001"].fillna(0).astype(int))
        .loc[:, ["GEOID", "POP20"]]
        .sort_values("GEOID")
        .reset_index(drop=True)
    )


def build_crosswalk_cd116_to_cd119():
    """Builds the crosswalk between 116th and 119th congress"""
    # Pull the census block level population data one state at a time
    state_pops = []
    for s in us.states.STATES_AND_TERRITORIES:
        if not s.is_territory and s.abbr not in ["DC", "ZZ"]:
            print(s.name)
            state_pops.append(fetch_block_population(s.abbr))
    block_pop_df = pd.concat(state_pops)

    # Get census blocks for each district under the 116th and 119th congress
    # Remove 'ZZ': blocks not assigned to any congressional district
    df116 = fetch_block_to_district_map(116)
    df116 = df116.loc[df116["CD116"] != "ZZ"]
    df119 = fetch_block_to_district_map(119)
    df119 = df119.loc[df119["CD119"] != "ZZ"]

    common_blocks = df116.merge(df119, on="GEOID")

    block_stats = block_pop_df.merge(common_blocks, on="GEOID")
    block_stats["state_fips"] = block_stats.GEOID.str[:2]
    shares = (
        block_stats.groupby(["state_fips", "CD116", "CD119"])["POP20"]
        .sum()
        .rename("pop_shared")
        .reset_index()
    )

    def make_cd_code(state, district):
        return f"5001800US{str(state).zfill(2)}{str(district).zfill(2)}"

    shares["code_old"] = shares.apply(
        lambda row: make_cd_code(row.state_fips, row.CD116), axis=1
    )
    shares["code_new"] = shares.apply(
        lambda row: make_cd_code(row.state_fips, row.CD119), axis=1
    )
    shares["proportion"] = shares.groupby("code_old").pop_shared.transform(
        lambda s: s / s.sum()
    )

    ## add DC's district
    dc_row = pd.DataFrame(
        {
            "state_fips": ["11"],  # DC's FIPS
            "CD116": ["98"],  # at-large code in the BEF files
            "CD119": ["98"],
            "pop_shared": [689545],
            "code_old": ["5001800US1198"],
            "code_new": ["5001800US1198"],
            "proportion": [1.0],
        }
    )

    shares = pd.concat([shares, dc_row], ignore_index=True)

    district_mapping = (
        shares[["code_old", "code_new", "proportion"]]
        .sort_values(["code_old", "proportion"], ascending=[True, False])
        .reset_index(drop=True)
    )
    assert len(set(district_mapping.code_old)) == 436
    assert len(set(district_mapping.code_new)) == 436
    mapping_path = Path(STORAGE_FOLDER, "district_mapping.csv")
    district_mapping.to_csv(mapping_path, index=False)


def get_district_mapping():
    """Puts the 436 by 436 - with DC - (old by new) district mapping matrix into memory"""

    mapping_path = Path(STORAGE_FOLDER, "district_mapping.csv")
    mapping_df = pd.read_csv(mapping_path)

    old_codes = sorted(mapping_df.code_old.unique())
    new_codes = sorted(mapping_df.code_new.unique())
    assert len(old_codes) == len(new_codes) == 436

    old_index = {c: i for i, c in enumerate(old_codes)}
    new_index = {c: j for j, c in enumerate(new_codes)}

    mapping_matrix = np.zeros((436, 436), dtype=float)

    for row in mapping_df.itertuples(index=False):
        i = old_index[row.code_old]
        j = new_index[row.code_new]
        mapping_matrix[i, j] = row.proportion

    assert np.allclose(mapping_matrix.sum(axis=1), 1.0)
    return {'mapping_matrix': mapping_matrix, 'old_codes': old_codes, 'new_codes': new_codes}


if __name__ == "__main__":
    build_crosswalk_cd116_to_cd119()
    print(get_district_mapping_matrix())
