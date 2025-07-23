import logging
import requests
from pathlib import Path
import io

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from policyengine_us_data.db.create_database_tables import Stratum, StratumConstraint, Target


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


def extract_docs(year=2023):
    docs_url = (
        f"https://api.census.gov/data/{year}/acs/acs1/subject/variables.json"
    )

    try:
        docs_response = requests.get(docs_url)
        docs_response.raise_for_status()

        docs = docs_response.json()
        docs['year'] = year

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    return docs


def extract_age_data(geo, year=2023):
    base_url = (
        f"https://api.census.gov/data/{year}/acs/acs1/subject?get=group(S0101)"
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

        headers = data[0]
        data_rows = data[1:]
        df = pd.DataFrame(data_rows, columns=headers)

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    return df


def transform_age_data(age_data, docs):
    df = age_data.copy()

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

    # TODO: find somewhere else to do these checks
    #if geo == "District":
    #    assert df_geos.shape[0] == 436
    #elif geo == "State":
    #    assert df_geos.shape[0] == 51
    #elif geo == "National":
    #    assert df_geos.shape[0] == 1

    df = df_geos[["ucgid"] + AGE_COLS]

    df_long = df.melt(
        id_vars='ucgid',
        value_vars=AGE_COLS,
        var_name='age_range',
        value_name='value'
    )
    age_bounds = df_long['age_range'].str.split('-', expand=True)
    df_long['age_greater_than_or_equal_to'] = age_bounds[0].str.replace('+', '').astype(int)
    df_long['age_less_than_or_equal_to'] = pd.to_numeric(age_bounds[1])
    df_long['variable'] = 'person_count'
    df_long['period'] = docs['year']
    df_long['reform_id'] = 0
    df_long['source_id'] = 1
    df_long['active'] = True

    return df_long


def load_age_data(df_long):
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


    #target_df = df_long[[
    #    'target_id',
    #    'variable',
    #    'period',
    #    'stratum_id',
    #    'reform_id',
    #    'value',
    #    'source_id',
    #    'active'
    #]]

    #bounds = df_long['age_range'].str.extract(
    #    r'(?P<low>\d+)-(?P<high>\d+|inf)',  # captures 0‑4, 5‑9 … 85‑inf
    #    expand=True
    #)
    #
    #lo_rows = (
    #    df_long[['stratum_id']]
    #      .join(bounds['low'].rename('value'))
    #      .assign(operation='greater_than_or_equal')
    #)
    #
    #hi_rows = (
    #    df_long[['stratum_id']]
    #      .join(bounds['high'].rename('value'))
    #      .assign(operation='less_than_or_equal')
    #)
    #
    #out = (
    #    pd.concat([lo_rows, hi_rows], ignore_index=True)
    #      .replace({'value': {'inf': 'Inf'}})      # keep ∞ as the string “Inf”
    #)
    #
    #out['value'] = pd.to_numeric(out['value'], errors='ignore')
    #out.insert(loc=1, column='breakdown_variable', value='age')

    #ucgid_df = df_long[['stratum_id', 'ucgid']].copy()
    #ucgid_df['operation'] = 'equals'
    #ucgid_df.insert(loc=1, column='contraint_variable', value='ucgid')
    #ucgid_df = ucgid_df.rename(columns={'ucgid': 'value'})

    #both = pd.concat([ucgid_df, out])
    #both = both.sort_values(['stratum_id', 'operation'])
    
    #return out


if __name__ == "__main__":

    # --- ETL is Extract, Transform, Load ----

    # ---- Extract ----------
    docs = extract_docs(2023)
    national_df = extract_age_data("National", 2023)
    state_df = extract_age_data("State", 2023)

    # --- Transform ----------
    long_national_df = transform_age_data(national_df, docs)
    long_state_df = transform_age_data(state_df, docs)

    # --- Load --------
    load_age_data(long_national_df)
    load_age_data(long_state_df)
