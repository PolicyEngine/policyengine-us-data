import requests
import pandas as pd
from pathlib import Path
from policyengine_us_data.storage import STORAGE_FOLDER

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


def pull_age_data(geo, year):
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
    except Exception as e:
        print(f"An error occurred: {e}")

    # Keys: descriptions of the variables we want. Values: short names
    label_to_short_name_mapping = {
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

    # map the documentation labels to the actual data set variables
    label_to_variable_mapping = dict(
        [
            (value["label"], key)
            for key, value in docs["variables"].items()
            if value["group"] == "S0101"
            and value["concept"] == "Age and Sex"
            and value["label"] in label_to_short_name_mapping.keys()
        ]
    )

    # By transitivity, map the data set variable names to short names
    rename_mapping = dict(
        [
            (label_to_variable_mapping[v], label_to_short_name_mapping[v])
            for v in label_to_short_name_mapping.keys()
        ]
    )

    df_data = df.rename(columns=rename_mapping)[
        ["GEO_ID", "NAME"] + list(label_to_short_name_mapping.values())
    ]

    # Filter out non-voting districts, e.g., DC and Puerto Rico
    df_geos = df_data[
        ~df_data["GEO_ID"].isin(
            ["5001800US7298", "5001800US1198", "0400000US72", "0400000US11"]
        )
    ].copy()

    omitted_rows = df_data[~df_data["GEO_ID"].isin(df_geos["GEO_ID"])]
    print(f"Ommitted {geo} geographies:\n\n{omitted_rows[['GEO_ID', 'NAME']]}")

    SAVE_DIR = Path(STORAGE_FOLDER)
    age_cols = list(label_to_short_name_mapping.values())
    if geo == "District":
        assert df_geos.shape[0] == 435
        df_geos["GEO_NAME"] = df_geos["NAME"].apply(abbrev_name)
    elif geo == "State":
        assert df_geos.shape[0] == 50
        df_geos["GEO_NAME"] = df_geos["NAME"].map(STATE_NAME_TO_ABBREV)
    elif geo == "National":
        assert df_geos.shape[0] == 1
        df_geos["GEO_NAME"] = df_geos["NAME"].map({"United States": "US"})

    out = df_geos[["GEO_ID", "GEO_NAME"] + age_cols]
    filename = {
        "District": "age_district.csv",
        "State": "age_state.csv",
        "National": "age_national.csv",
    }[geo]
    out.to_csv(SAVE_DIR / filename, index=False)


def abbrev_name(name):
    # 'Congressional District 1 (118th Congress), Alabama' -> AL-01
    district_number = name.split("District ")[1].split(" ")[0]
    state = STATE_NAME_TO_ABBREV[name.split(", ")[-1].strip()]
    return f"{state}-{district_number.zfill(2)}".replace("(at", "01")


if __name__ == "__main__":
    year = 2023
    pull_age_data("State", year)
