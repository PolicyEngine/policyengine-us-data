import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from policyengine_us_data.storage import CALIBRATION_FOLDER

logger = logging.getLogger(__name__)

STATE_NAME_TO_ABBREV = {
    "Alabama": "AL",
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


LABEL_TO_SHORT = {
    "Estimate!!Total!!Total population!!AGE!!Under 5 years": "0-4",
    "Estimate!!Total!!Total population!!AGE!!5 to 9 years": "5-9",
    "Estimate!!Total!!Total population!!AGE!!10 to 14 years": "10-14",
    "Estimate!!Total!!Total population!!AGE!!15 to 19 years": "15-19",
    "Estimate!!Total!!Total population!!AGE!!20 to 24 years": "20-24",
    "Estimate!!Total!!Total population!!AGE!!25 to 29 years": "25-29",
    "Estimate!!Total!!Total population!!AGE!!30 to 34 years": "30-34",
    "Estimate!!Total!!Total population!!AGE!!35 to 39 years": "35-39",
    "Estimate!!Total!!Total population!!AGE!!40 to 44 years": "40-44",
    "Estimate!!Total!!Total population!!AGE!!45 to 49 years": "45-49",
    "Estimate!!Total!!Total population!!AGE!!50 to 54 years": "50-54",
    "Estimate!!Total!!Total population!!AGE!!55 to 59 years": "55-59",
    "Estimate!!Total!!Total population!!AGE!!60 to 64 years": "60-64",
    "Estimate!!Total!!Total population!!AGE!!65 to 69 years": "65-69",
    "Estimate!!Total!!Total population!!AGE!!70 to 74 years": "70-74",
    "Estimate!!Total!!Total population!!AGE!!75 to 79 years": "75-79",
    "Estimate!!Total!!Total population!!AGE!!80 to 84 years": "80-84",
    "Estimate!!Total!!Total population!!AGE!!85 years and over": "85+",
}
AGE_COLS = list(LABEL_TO_SHORT.values())


def _pull_age_data(geo, year=2023):
    base_url = (
        f"https://api.census.gov/data/{year}/acs/acs1/subject?get=group(S0101)"
    )
    docs_url = (
        f"https://api.census.gov/data/{year}/acs/acs1/subject/variables.json"
    )

    if geo == "State":
        url = f"{base_url}&for=state:*"
    elif geo == "District":
        url = f"{base_url}&for=congressional+district:*"
    elif geo == "National":
        url = f"{base_url}&for=us:*"
    else:
        raise ValueError(
            "geo must be either 'National', 'State', or 'District'"
        )

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        docs_response = requests.get(docs_url)
        docs_response.raise_for_status()

        docs = docs_response.json()

        headers = data[0]
        data_rows = data[1:]
        df = pd.DataFrame(data_rows, columns=headers)

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    # map the documentation labels to the actual data set variables
    label_to_variable_mapping = dict(
        [
            (value["label"], key)
            for key, value in docs["variables"].items()
            if value["group"] == "S0101"
            and value["concept"] == "Age and Sex"
            and value["label"] in LABEL_TO_SHORT.keys()
        ]
    )

    # By transitivity, map the data set variable names to short names
    rename_mapping = dict(
        [
            (label_to_variable_mapping[v], LABEL_TO_SHORT[v])
            for v in LABEL_TO_SHORT.keys()
        ]
    )

    df_data = df.rename(columns=rename_mapping)[
        ["GEO_ID", "NAME"] + list(AGE_COLS)
    ]

    # Filter out non-voting districts, e.g., Puerto Rico
    df_geos = df_data[
        ~df_data["GEO_ID"].isin(["5001800US7298", "0400000US72"])
    ].copy()

    omitted_rows = df_data[~df_data["GEO_ID"].isin(df_geos["GEO_ID"])]
    print(f"Ommitted {geo} geographies:\n\n{omitted_rows[['GEO_ID', 'NAME']]}")

    SAVE_DIR = Path(CALIBRATION_FOLDER)
    if geo == "District":
        assert df_geos.shape[0] == 436
        df_geos["GEO_NAME"] = "district_" + df_geos["NAME"].apply(abbrev_name)
    elif geo == "State":
        assert df_geos.shape[0] == 51
        df_geos["GEO_NAME"] = "state_" + df_geos["NAME"].map(
            STATE_NAME_TO_ABBREV
        )
    elif geo == "National":
        assert df_geos.shape[0] == 1
        df_geos["GEO_NAME"] = "national"

    out = df_geos[["GEO_ID", "GEO_NAME"] + AGE_COLS]

    return out


def combine_age_geography_levels() -> None:
    national = _pull_age_data("National")
    logger.info(f"National age data: {national.shape[0]} rows")
    state = _pull_age_data("State")
    logger.info(f"State age data: {state.shape[0]} rows")
    district = _pull_age_data("District")
    logger.info(f"District age data: {district.shape[0]} rows")

    state["STATEFIPS"] = state["GEO_ID"].str[-2:]
    district["STATEFIPS"] = district["GEO_ID"].str[-4:-2]

    for col in AGE_COLS:
        national[col] = pd.to_numeric(national[col], errors="coerce")
        state[col] = pd.to_numeric(state[col], errors="coerce")
        district[col] = pd.to_numeric(district[col], errors="coerce")

    for col in AGE_COLS:
        us_total = national[col].iloc[0]  # scalar
        state_total = state[col].sum()
        if not np.isclose(state_total, us_total):
            logger.warning(
                f"States' sum population does not match national total for age band: {col}. Reescaling state targets."
            )
            state[col] *= us_total / state_total

    for col in AGE_COLS:
        state_totals = state.set_index("STATEFIPS")[col]
        district_totals = district.groupby("STATEFIPS")[col].sum()

        for fips, d_total in district_totals.items():
            s_total = state_totals.get(fips)

            if not np.isclose(d_total, s_total):
                logger.warning(
                    f"Districts' sum population does not match {fips} state total for age band: {col}. Reescaling district targets."
                )
                mask = district["STATEFIPS"] == fips
                district.loc[mask, col] *= s_total / d_total

    combined = pd.concat(
        [
            national,
            state.drop(columns="STATEFIPS"),
            district.drop(columns="STATEFIPS"),
        ],
        ignore_index=True,
    ).sort_values("GEO_ID")

    # Ensure all age columns are numeric before saving
    for col in AGE_COLS:
        combined[col] = combined[col].round().astype(int)

    # Transform from wide to long format
    long_format = pd.melt(
        combined,
        id_vars=["GEO_ID", "GEO_NAME"],
        value_vars=AGE_COLS,
        var_name="AGE_GROUP",
        value_name="VALUE",
    )

    # Parse age bounds from age group labels
    def parse_age_bounds(age_group):
        if age_group == "85+":
            return 85, np.inf  # No upper bound for 85+
        else:
            parts = age_group.split("-")
            return int(parts[0]), int(parts[1])

    # Extract lower and upper bounds
    bounds = long_format["AGE_GROUP"].apply(parse_age_bounds)
    long_format["LOWER_BOUND"] = bounds.apply(lambda x: x[0])
    long_format["UPPER_BOUND"] = bounds.apply(lambda x: x[1])
    long_format["DATA_SOURCE"] = "acs"
    long_format["VARIABLE"] = "age"
    long_format["IS_COUNT"] = True
    long_format["BREAKDOWN_VARIABLE"] = "age"

    # Reorder columns
    final_columns = [
        "DATA_SOURCE",
        "GEO_ID",
        "GEO_NAME",
        "VARIABLE",
        "VALUE",
        "IS_COUNT",
        "BREAKDOWN_VARIABLE",
        "LOWER_BOUND",
        "UPPER_BOUND",
    ]

    final_df = long_format[final_columns]

    # Sort by GEO_ID and age group for better organization
    final_df = final_df.sort_values(["GEO_ID", "BREAKDOWN_VARIABLE"])

    out_path = CALIBRATION_FOLDER / "age.csv"
    final_df.to_csv(out_path, index=False)


def abbrev_name(name):
    # 'Congressional District 1 (118th Congress), Alabama' -> AL-01
    district_number = name.split("District ")[1].split(" ")[0]
    state = STATE_NAME_TO_ABBREV[name.split(", ")[-1].strip()]
    return f"{state}-{district_number.zfill(2)}".replace("(at", "01")


if __name__ == "__main__":
    combine_age_geography_levels()
