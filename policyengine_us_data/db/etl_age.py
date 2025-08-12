import requests
from pathlib import Path
import io

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)


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
        docs["year"] = year

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

    df = df.drop(columns="NAME")
    df = df.rename({"GEO_ID": "ucgid_str"}, axis=1)
    df_data = df.rename(columns=rename_mapping)[["ucgid_str"] + list(AGE_COLS)]

    # Filter out Puerto Rico's district and state records, if needed
    df_geos = df_data[
        ~df_data["ucgid_str"].isin(["5001800US7298", "0400000US72"])
    ].copy()

    df = df_geos[["ucgid_str"] + AGE_COLS]

    df_long = df.melt(
        id_vars="ucgid_str",
        value_vars=AGE_COLS,
        var_name="age_range",
        value_name="value",
    )
    age_bounds = df_long["age_range"].str.split("-", expand=True)
    df_long["age_greater_than_or_equal_to"] = (
        age_bounds[0].str.replace("+", "").astype(int)
    )
    df_long["age_less_than_or_equal_to"] = pd.to_numeric(age_bounds[1])
    df_long["variable"] = "person_count"
    df_long["period"] = docs["year"]
    df_long["reform_id"] = 0
    df_long["source_id"] = 1
    df_long["active"] = True

    return df_long


def get_parent_geo(geo):
    return {"National": None, "State": "National", "District": "State"}[geo]


def load_age_data(df_long, geo, stratum_lookup={}):

    # Quick data quality check before loading ----
    if geo == "National":
        assert len(set(df_long.ucgid_str)) == 1
    elif geo == "State":
        assert len(set(df_long.ucgid_str)) == 51
    elif geo == "District":
        assert len(set(df_long.ucgid_str)) == 436
    else:
        raise ValueError('geo must be one of "National", "State", "District"')

    # Prepare to load data -----------
    DATABASE_URL = "sqlite:///policy_data.db"
    engine = create_engine(DATABASE_URL)

    Session = sessionmaker(bind=engine)
    session = Session()

    if not stratum_lookup:
        if geo != "National":
            raise ValueError("Include stratum_lookup unless National geo")
        stratum_lookup = {"National": {}}
    else:
        stratum_lookup[geo] = {}

    for _, row in df_long.iterrows():

        # Create the parent Stratum object.
        # We will attach children to it before adding it to the session.
        note = f"Age: {row['age_range']}, Geo: {row['ucgid_str']}"
        parent_geo = get_parent_geo(geo)
        parent_stratum_id = (
            stratum_lookup[parent_geo][row["age_range"]]
            if parent_geo
            else None
        )

        new_stratum = Stratum(
            parent_stratum_id=parent_stratum_id, stratum_group_id=0, notes=note
        )

        # Create constraints and link them to the parent's relationship attribute.
        new_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="in",
                value=row["ucgid_str"],
            ),
            StratumConstraint(
                constraint_variable="age",
                operation="greater_than_or_equal",
                value=str(row["age_greater_than_or_equal_to"]),
            ),
        ]

        age_lt_value = row["age_less_than_or_equal_to"]
        if not np.isinf(age_lt_value):
            new_stratum.constraints_rel.append(
                StratumConstraint(
                    constraint_variable="age",
                    operation="less_than",
                    value=str(age_lt_value + 1),
                )
            )

        # Create the Target and link it to the parent.
        new_stratum.targets_rel.append(
            Target(
                variable=row["variable"],
                period=row["period"],
                value=row["value"],
                source_id=row["source_id"],
                active=row["active"],
            )
        )

        # Add ONLY the parent object to the session.
        # The 'cascade' setting will handle the children automatically.
        session.add(new_stratum)

        # Flush to get the id
        session.flush()
        stratum_lookup[geo][row["age_range"]] = new_stratum.stratum_id

    # Commit all the new objects at once.
    session.commit()

    return stratum_lookup


if __name__ == "__main__":

    # --- ETL: Extract, Transform, Load ----

    # ---- Extract ----------
    docs = extract_docs(2023)
    national_df = extract_age_data("National", 2023)
    state_df = extract_age_data("State", 2023)

    # --- Transform ----------
    long_national_df = transform_age_data(national_df, docs)
    long_state_df = transform_age_data(state_df, docs)

    # --- Load --------
    national_strata_lku = load_age_data(long_national_df, "National")
    state_strata_lku = load_age_data(
        long_state_df, "State", national_strata_lku
    )
