import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from policyengine_us_data.storage import CALIBRATION_FOLDER

import io

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from policyengine_us_data.db.data_models import Base, Stratum, StratumConstraint, Target



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
    "Estimate!!Total!!Total population!!AGE!!85 years and over": "85-inf",
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

    df = df.drop(columns='NAME')
    df = df.rename({'GEO_ID': 'ucgid'}, axis=1)
    df_data = df.rename(columns=rename_mapping)[["ucgid"] + list(AGE_COLS)]

    # Filter out Puerto Rico's district and state records, if needed
    df_geos = df_data[
        ~df_data["ucgid"].isin(["5001800US7298", "0400000US72"])
    ].copy()

    SAVE_DIR = Path(CALIBRATION_FOLDER)
    if geo == "District":
        assert df_geos.shape[0] == 436
    elif geo == "State":
        assert df_geos.shape[0] == 51
    elif geo == "National":
        assert df_geos.shape[0] == 1

    df = df_geos[["ucgid"] + AGE_COLS]

    # So this is really the target table, and the rows are different strata
    # And you can fill that in and then define your strata table
    # I have a clean slate and I can make stratum_ids at will, but how will
    # we avoid collisions in practice?
    df_long = df.melt(
        id_vars='ucgid',
        value_vars=AGE_COLS,
        var_name='age_range',
        value_name='value'
    )
    age_bounds = df_long['age_range'].str.split('-', expand=True)
    df_long['age_greater_than_or_equal_to'] = age_bounds[0].str.replace('+', '').astype(int)
    df_long['age_less_than_or_equal_to'] = pd.to_numeric(age_bounds[1])
    #df_long['target_id'] = range(18)
    df_long['variable'] = 'person_count'
    #df_long['stratum_id'] = range(18)
    df_long['period'] = year
    df_long['reform_id'] = 0
    df_long['source_id'] = 1
    df_long['active'] = True

    DATABASE_URL = "sqlite:///policy_data.db"
    engine = create_engine(DATABASE_URL)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    for _, row in df_long.iterrows():
        # 1. Create a new Stratum for each row. We will make it unique
        # by creating a descriptive note.
        note = f"Age: {row['age_range']}, Geo: {row['ucgid']}"
        new_stratum = Stratum(notes=note)
        session.add(new_stratum)
        session.flush() # Flush to assign stratum_id

        # 2. Create StratumConstraint records based on the DataFrame
        # The 'ucgid' constraint
        ucgid_constraint = StratumConstraint(
            stratum_id=new_stratum.stratum_id,
            constraint_variable='ucgid',
            operation='equals',
            value=row['ucgid']
        )

        # The age constraints
        age_gte_constraint = StratumConstraint(
            stratum_id=new_stratum.stratum_id,
            constraint_variable='age',
            operation='greater_than_or_equal',
            value=str(row['age_greater_than_or_equal_to'])
        )
        
        # Handle the 'inf' case for the upper age bound
        age_lt_value = row['age_less_than_or_equal_to']
        if not np.isinf(age_lt_value):
            age_lt_constraint = StratumConstraint(
                stratum_id=new_stratum.stratum_id,
                constraint_variable='age',
                operation='less_than',
                value=str(age_lt_value + 1) # less_than, so add 1
            )
            session.add(age_lt_constraint)

        session.add(ucgid_constraint)
        session.add(age_gte_constraint)

        # 3. Create the Target record
        new_target = Target(
            stratum_id=new_stratum.stratum_id,
            variable=row['variable'],
            period=row['period'],
            value=row['value'],
            source_id=row['source_id'],
            active=row['active'],
        )
        session.add(new_target)
    
    session.commit()
    print("✅ Data loaded successfully.")

    # HERE

    target_df = df_long[[
        'target_id',
        'variable',
        'period',
        'stratum_id',
        'reform_id',
        'value',
        'source_id',
        'active'
    ]]

    bounds = df_long['age_range'].str.extract(
        r'(?P<low>\d+)-(?P<high>\d+|inf)',  # captures 0‑4, 5‑9 … 85‑inf
        expand=True
    )
    
    lo_rows = (
        df_long[['stratum_id']]
          .join(bounds['low'].rename('value'))
          .assign(operation='greater_than_or_equal')
    )
    
    hi_rows = (
        df_long[['stratum_id']]
          .join(bounds['high'].rename('value'))
          .assign(operation='less_than_or_equal')
    )
    
    out = (
        pd.concat([lo_rows, hi_rows], ignore_index=True)
          .replace({'value': {'inf': 'Inf'}})      # keep ∞ as the string “Inf”
    )
    
    out['value'] = pd.to_numeric(out['value'], errors='ignore')
    out.insert(loc=1, column='breakdown_variable', value='age')

    ucgid_df = df_long[['stratum_id', 'ucgid']].copy()
    ucgid_df['operation'] = 'equals'
    ucgid_df.insert(loc=1, column='contraint_variable', value='ucgid')
    ucgid_df = ucgid_df.rename(columns={'ucgid': 'value'})

    both = pd.concat([ucgid_df, out])
    both = both.sort_values(['stratum_id', 'operation'])


    return out


def combine_age_geography_levels() -> None:
    national = _pull_age_data("National")

    # Rethinking national ----
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
